# Changelog

All notable changes to Wukong are documented here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows semantic versioning.

## [Unreleased]

### Benchmark honesty — peer-strength audit (2026-08-04)
An adversarial audit of every C/C++/Rust peer kernel in `crates/wukong_xbench` found four systematic
defects, all of which inflated Wukong's published numbers, and corrected them. **No compiler, kernel
or `.wk` source changed; only the benchmark peers and the documents that quote them.**
- **`__restrict__` on every generated C/C++ peer kernel.** `grep -c restrict` over
  `crates/wukong_xbench/src/main.rs` previously returned **0**: not one of the 43 C kernels told gcc
  its output buffer does not overlap its inputs, so gcc could not vectorize or reorder any nested
  kernel, while Wukong's tensor parameters carry non-overlap in the type system. Isolated with the
  kernels in their own translation unit — the harness compiles each peer to a shared library, so a
  one-TU probe over `static` arrays lets gcc's IPA prove non-overlap by itself and makes `restrict`
  look like a no-op — `restrict` alone is worth **8.7× on `matmul_tn` 512³ (176.6 → 20.2 ms)**,
  **5.0× on the direct convolution (1.99 → 0.40 ms)**, **4.2× on the column-outer `colsum`**, 1.33×
  on `saxpy`, and nothing at all on `relu`/`biasadd` (0.96×/1.01×). The conv peer went from **4.9 to
  21.1 GFLOP/s** in the suite, taking the published conv multiple from 5.90× to **1.55×**. Two latent
  aliasing hazards were removed first so the qualifier is true rather than merely fast
  (`fused_linear_relu`'s const-cast-away write, and a dequant bench that passed its own output buffer
  as the unused middle pointer).
- **Column reductions / column argmax rewritten row-outer.** `c_colsum`, `c_colmax`/`min`/`absmax`,
  `c_colstat` (mean/sum-sq/L2/RMS) and `c_colarg` all scanned `for j { for i { … x[i*N+j] } }` — the
  worst loop order for a row-major axis-0 reduction, and the only order measured. Isolated at `-O3
  -march=native`, 4096×1024: colsum **21.33 → 0.58 ms (36.7×)**, colmax **34.82 → 0.86 ms (40.6×)**,
  colmean **33.03 → 1.24 ms (26.7×)**, colargmax **9.87 → 1.57 ms (6.3×)**, output identical.
  **The published ~29–50× column-reduction win becomes a tie at best and a 1.8× LOSS at worst**
  (15 of 16 measured rows are losses; the >L3 shape is a consistent ~1.65× loss across two
  independent A/B rounds), and the column argmax's
  ~2.7–4.1× becomes a 1.5–2.5× loss.
- **`matmul_tn` peer given the natural `kij` order.** It was `ijk` with **both** operands read
  column-strided. Isolated 512³: **176.6 → 9.70 ms (18.2× total: 8.7× `restrict` × 2.1× loop
  order)**. The weight-gradient GEMM's ~128× single / ~445× parallel becomes **2.6–5.3× /
  3.1–13.9×**.
- **Transpose peer cache-blocked 32×32**, matching Wukong's own kernel. Isolated 2048²: **29.51 →
  14.09 ms (2.09×)**. Single-core transpose becomes a **tie**.
- **Ten reduction-bearing benches gained the `C(fast)` [-ffast-math] column** their own fairness rule
  required (gemv, scaled_gemm, linear_bf16, conv, colmax/min/absmax, rowarg, colarg, bf16 dot/sum,
  cumsum, xent_bwd). Four of them turn out to be ties or losses once it exists: the bf16 reductions
  (1.1–1.8× slower), gemv (a tie), conv (1.12× slower), cumsum (1.03–1.42×).
- **The `@parallel` standing lines are direction-aware.** Twenty sites printed a bare ratio, so a
  0.73× — a 27% loss — rendered as "Wukong @parallel is 0.73x idiomatic single-threaded C".
- **Four regression tests** (`every_generated_c_kernel_declares_restrict`,
  `column_family_peers_stay_row_outer`, `matmul_tn_peer_stays_kij`,
  `transpose_peer_stays_cache_blocked`) plus a written peer-strength checklist now pin all of it,
  because a weakened peer breaks nothing observable — the suite still runs, the cross-language check
  still passes, and the only symptom is a bigger multiple.
- `BENCHMARKS.md`, `README.md` and `docs/roadmap.md` republish the corrected figures with the old
  ones struck through, and every "why Wukong wins" explanation that was in fact describing a peer
  defect is rewritten. `BENCHMARKS.md` gains a full before/after board and an explicit list of what
  was **not** re-measured (the end-to-end model peers in `model.rs`, the whole-suite geomean, the GPU
  section).

### Correctness + robustness — code-map-hardening campaign (2026-07-29)
A codebase-wide correctness pass over every crate (read-only audit groups → fix branches over disjoint
write-sets → adversarial re-verification), plus a harness-hardening wave. Almost nothing here is a new
feature: the dominant shape is a construct that used to be silently accepted and miscompiled now being
**rejected with a catalogued diagnostic**, and a kernel recognizer that used to fire on a shape its
kernel cannot represent now **declining to the (correct) scalar path**. Several uncatchable process
aborts and ICEs became ordinary diagnostics.
- **Diagnostics — a closed, two-directionally gated error-code catalogue**: `E0001` ("malformed type
  syntax") is **retired** — no stage ever emitted it, the parser reports malformed types as
  `E0203`/`E0204`, and `--explain E0001` now reports it as an unknown error code (a usage error, exit
  2) instead of printing an explanation. The live catalogue is exactly the
  29 codes `E0101`–`E0104`, `E0200`–`E0209`, `E0300`–`E0305`, `E0401`–`E0403`, `E0405`,
  `E0501`–`E0504`, `C0001` (plus the internal `E9999` fallback). Codes are stable: never renumbered,
  and a retired number is never reused. Two gates keep it honest — every code any crate emits must
  have a catalogue entry *and* every entry must have an emitter
  (`crates/wukong_diag/tests/catalog_coverage.rs`; the scanner skips comment lines, so prose may cite
  codes freely), and every catalogued code must be reached by a `tests/fail` fixture or an inline
  lexer/parser probe (`crates/wukongc/tests/fail.rs`, with a floor on catalogue size so it cannot pass
  vacuously). Adding a diagnostic now means adding a catalogue entry **and** a fixture.
- **Three `--explain` bodies corrected to what the checks actually do**: `E0304` now states that
  mutating through a non-`mut` **aggregate parameter** (`p.f = …`, `p[i] = …`) is rejected — such a
  parameter is passed by reference, so the write would land in the caller's value — while the same
  writes through a `let` binding, and `*p = …` through a pointer parameter, stay legal; `E0402` is
  retitled *recursive struct or enum has infinite size* (a data-carrying enum is reported too); and
  `E0405` drops the value-position framing — a non-exhaustive `match` is rejected in **every**
  position, including statement position and an empty `match`.
- **Diagnostic rendering**: the human renderer expands tabs to a fixed width of 4 on **both** the
  echoed source line and the caret padding, so carets land under the construct in tab-indented source
  (reported line/col — and therefore `--error-format=json` — are unchanged), and the JSON writer
  escapes any control character with no short escape as `\u00xx` so the line stays parseable JSON.
- **Lexer**: an empty char literal `''` is now `E0104` instead of silently decoding to `0` (i.e. being
  indistinguishable from `'\0'`), and a multi-codepoint `'ab'` reports one `E0104` and resyncs past its
  own closing quote instead of re-lexing that quote as an opening one and swallowing the next token.
- **Parser**: the recursion-depth budget is charged for every `.field` / `[i]` / `(args)` / `::name`
  postfix fold, so a long postfix chain reports `E0209` rather than overflowing the compile thread's
  stack with no diagnostic at all; integer literals in **type** positions (tensor dimension, SIMD lane
  count, `.tiled(…)` extent, tuple index) are decoded with full radix / `_`-separator / suffix support,
  and an undecodable or out-of-range one is a catalogued error instead of a silent `0`; attribute
  parsing no longer consumes tokens it does not own (a stray `@` no longer eats the following item and
  silently deletes it, `extern` is the only keyword spelling accepted as an attribute name, and a
  missing attribute value reports `E0207` at the missing value); and struct **functional update**
  `S { x: 1, ..base }` — which has an AST slot but no lowering — is a single clean rejection naming the
  unsupported feature, with `rest` deliberately left `None` so it cannot silently leave fields
  uninitialized (it previously cascaded five errors and fell out of the function body into module
  scope). Struct functional update is **not** a language feature.
- **Sema — new rules and closed holes**: a function name beginning with `wukong_` is **rejected**
  (`E0300`) — the prefix is reserved for the symbols the kernel recognizers emit, and such a function
  used to silently hijack kernel dispatch; a `\u{…}` escape above `10FFFF`, or in the surrogate range
  `D800..=DFFF`, is rejected, so the promise that a `char` holds a Unicode scalar value is now
  enforced; an enum discriminant the constant folder cannot read (typically a reference to a top-level
  `const`) is `E0401` instead of being silently replaced by the auto-increment value — a discriminant
  must be an integer literal or arithmetic on integer literals; integer-literal **patterns**, and both
  bounds of a range pattern, are range-checked against the scrutinee type instead of being truncated
  to its width; an array-repeat initializer's count must equal its annotation's length
  (`let a: [i32; 4] = [7; 2];` is `E0401`); an unresolvable name used as an **array length** is `E0301`
  instead of typing the array as length `0` (parity with tensor dims' `E0504`); struct field, enum
  payload and `extern`-block types are re-lowered in the body pass purely for their diagnostics, so an
  unknown tensor dim reports `E0504`/`E0301` there exactly as in a function signature; and const array
  lengths and turbofish dim values decode radix prefixes, `_` separators and integer suffixes like the
  rest of the language.
- **Call unification tightened**: a `[]T` slice, struct or tuple argument against a `Tensor` parameter
  is rejected instead of falling through a lenient wildcard arm (a slice is not a tensor base); a
  const-dim parameter against a caller's variable dim is rejected in inference mode as it already was
  in rigid mode (`?` stays the escape hatch); and tensor **layout** is part of unification, so a
  `.col_major` tensor can no longer route silently through a contiguous-typed callee.
- **A documented language limit — `MAX_CONST_ARRAY_LEN` = `u32::MAX`**: the const folder shared by
  sema's index bounds check and `mir_build`'s slot sizing declines — folding to the invalid-length
  sentinel `0`, which sema then rejects at the first index — when an **operand** or the result exceeds
  it, and uses checked arithmetic so a wrapping `+`/`-`/`*` declines instead of producing a colossal
  length. No in-range length changes value. The *operand* bound is load-bearing, because sema folds in
  `u64` while `mir_build` narrows to `u32`.
- **Layout invariant restored**: the alignment of a vector type is always a power of two
  (`next_power_of_two(elem_size * lanes)`), re-establishing the mirror with `mir_build`'s bitmask
  round-up so the two aggregate-layout authorities agree. Identity for every power-of-two lane count,
  so no existing program's layout moves.
- **MIR: the two-level story was false, and is now marked so.** `MirLevel::High` is constructed
  nowhere, `Program::new` starts at `Low`, and nothing reads `Program::level` — there is **no**
  High→Low lowering invariant, and `--emit=mir-high` means *pre-optimization* MIR, not a different IR
  level. The enum and field remain in place, documented as unused.
- **MIR verifier, materially stronger**: it now enforces true SSA — a value is defined exactly once,
  inside the value arena, and its definition **dominates** every use (dominator-tree preorder walk) —
  rejects an entry block whose parameters disagree with `Function::params` and any predecessor of the
  entry block, range-checks `Op::VecKernelCall.kernel` against the function's `vec_kernels` table and
  checks the call's result against the recipe kind, and, given a whole `Program`, cross-checks every
  call against its **callee's** declared signature (arity, argument types, result type) — available
  through `verify_program` only, which no compile path calls: `verify_or_ice` and the optimizer's
  debug-only per-pass check both go through `verify_function`, so that one check is test-only today.
  Callees with no MIR body — `print` and the `wukong_*` runtime kernels — are external symbols and stay unchecked;
  blocks unreachable from entry keep their type checks but skip the dominance rule. These are the
  invariants `cse`/`licm`/`mem2reg`/`simplify-phis` argue from. Known gap: operand *lane counts* are
  not yet checked, so a mis-shaped vector `Cast` still reaches the backends (where it is now a
  diagnostic rather than a panic).
- **Optimizer — three `-O0`-vs-`-On` divergences closed**: constant folding no longer folds bf16/f16
  **comparisons** (two distinct f32 literals landing on one narrow-precision grid point folded to "not
  equal" while the runtime compare says equal, flipping an `if`; `fold_bin` already refused narrow
  arithmetic, so the comparison path now matches — bf16/f16 constants are never folded, arithmetic or
  comparison); the self-operand folds `x - x` / `x ^ x` / `x <cmp> x` decline when the instruction's
  result type is a **vector**, where they used to materialize a scalar constant the verifier rejects;
  and inlining now merges the callee's `vec_kernels` recipe table into the caller and rebases every
  copied `VecKernelCall.kernel` index — a vectorized leaf inlined at `-O2` previously ran one of the
  caller's unrelated recipes or indexed an empty table. Contract for anyone writing a MIR pass:
  `VecKernelCall.kernel` is a **function-local index** and must be rebased when instructions move
  between functions.
- **Slices `[]T` are legal operands to the whole recognized-kernel family** (a genuine capability
  expansion): every kernel buffer operand now resolves through one `kernel_base_ptr` helper, and every
  emitter declines to the scalar nest when an operand is unbound. Around thirty emitters previously
  took a slot address directly, handing the kernel the address of the 16-byte **fat pointer** instead
  of its data. Slice-ness comes from a `slice_slots` set recorded from the *sema* type at every binding
  site — parameters, `let`, tuple fields, `@parallel` region captures, the whole-scrutinee `match`
  binding, tuple sub-patterns, data-carrying-variant payloads, and the element binding of a `for`-each
  over an array of slices — never from the MIR slot shape, which cannot distinguish `[]T` from a
  16-byte struct, a tuple or `[u8; 16]`.
- **The recognized-kernel family is f32-only (`i32` labels)**: element-type gates were added to the
  fused norms, LayerNorm backward, the row-loss heads (KL divergence, entropy), RoPE, per-row
  argmax/argmin, the four scan bodies (cumsum, cumprod, gated carry, lrscan), the shared reduction
  index base, and the cross-entropy label gather. `f64`, bf16/f16 data and `i64` labels **decline to
  the scalar nest**. This class was a hard interp/native divergence: the interpreter marshals
  slot-per-scalar and converted, while native reinterpreted the bytes. The explicitly cast spelling
  `(x[k] as f32)` still routes to the low-precision kernels, so the mixed-precision suite above is
  unaffected.
- **Recognizer preconditions that used to be assumed are now checked**, each declining to the correct
  (unaccelerated) scalar path: only half-open unit-step `for` ranges dispatch — a `step` or `..=` range
  used to have its trip count read as `end - start` and dispatch a kernel that walks each index once;
  `pool2d` requires literal dims and the nest's own output extent to equal the full no-padding extent
  the kernel recomputes, so a sub-region pool declines instead of writing the full extent at the wrong
  row stride into a smaller buffer; the fused matmul epilogue declines a peeled alpha, a peeled store
  bias and a transposed A (the alpha folds into the scaled GEMM and the bias into its own epilogue
  call, so declining costs nothing); a **whole-function** matmul whose shape the GEMM emitter declines
  now lowers as the ordinary scalar nest instead of leaving the wrapper as dead constants plus a bare
  `ret`; the batched-norm matcher requires a per-row width independent of the row; the cumulative-max
  seed must be an actual infinity; recognizer coefficients must be loop-invariant **and**
  side-effect-free (a body-local capture silently changed the polynomial, and an arbitrary call ran
  once instead of per element); the activation peelers respect shadowing through one shared predicate,
  so a user `fn gelu` / `fn silu` / `fn fmax` is no longer fused away in favour of the runtime's
  builtin curve; the arg-reduce nest guards its `x[ki]` load under `ki >= 0` (the kernel returns `-1`
  for an empty span or an all-NaN row) and uses the loop's own strict compare for the tie-break, so a
  runtime-zero trip count leaves the seed untouched; and the 256-bit AVX2 recipe path declines when a
  stream base is not a scope binding — a module-level `const` array — reporting the catalogued,
  spanned `C0001` instead of panicking the compiler.
- **The 256-bit recipe contracts a float `x + y*z` into one `Fma`**, matching both the scalar tail and
  the 128-bit CLIF path. The vector part previously rounded twice where the tail rounded once, so one
  loop returned two different answers across its own lane/tail boundary and `WUKONG_P4_NO_256=1`
  changed a program's *numbers* rather than only its instruction selection. That kill-switch is now a
  result-identical A/B knob, gated by a test.
- **Run-time disjointness for the parallel gather/scatter**: `wukong_embedding_f32_parallel` and
  `wukong_scatter_add_f32_parallel` are selected under a run-time byte-range non-overlap test (through
  `PtrToInt`, the one pointer cast both backends implement). An aliased call — two *distinct* array
  parameters bound to the same array at the call site, which the old symbol-equality guard could not
  see — lands on the serial kernel, and a weight with no compile-time extent (a slice) keeps the serial
  kernel outright. Aliased buffers are therefore correct, just serial.
- **`sdpa`'s operand contract is enforced in lowering** (sema has no signature for `sdpa` at all): the
  four buffer arguments must be f32 arrays, slices or tensor views, and when `s` and `d` are
  compile-time constants every buffer whose type pins an extent must hold `s*d` elements. A violation
  declines the lowering so the call reports `C0001` at its own span rather than segfaulting; slice
  arguments now pass their **data** pointer. Pinned by `tests/fail/sdpa_operand_{type,extent}.wk`.
- **Lowering-core fixes**: a binary operator reads each operand's width off the value the builder
  produced and widens an integer operation to hold *both* operands instead of truncating the loop
  counter back to sema's narrower type — closing a whole class of ICEs on `for i in 1..n` with
  `n: i64`; ordered-comparison signedness is derived from both operands by one shared width-aware rule
  faithful to C's usual arithmetic conversions (only an equal-rank mixed-sign pair goes unsigned), and
  the range-`for` counter uses the same rule, so `for i in 0..big` with `big: u32` no longer runs zero
  iterations; the counting-`while` normalization's hoisted bound is restricted to a **pure**
  loop-invariant bound, so a bound with a side effect or one the body mutates re-evaluates per
  iteration; every literal `match` pattern is materialized through one checked helper that declines a
  non-integer scrutinee, a `true`/`false` pattern against a non-bool scrutinee, an out-of-range literal
  (including the i64/u64 widths sema's own check cannot cover) and a non-literal range bound — MIR
  integers are signless, so the accepted range is the union of the signed and unsigned ranges of the
  scrutinee's width; array-initializer length is checked on **every** path that reaches lowering,
  including an assignment RHS and a data-enum tuple payload, with a mismatch reported as `E0401`
  instead of writing outside the buffer or leaving the tail unwritten; the monomorphization key
  separator is `$`, which no user identifier can contain, so a structural-type instantiation can no
  longer share a namespace with a user type name and mint one instance for two ABIs (the remaining
  catch-all that collapses every vector/tensor/fn instantiation onto one key is a documented gap); and
  `parse_float` strips the same suffix list sema accepts, including the bare C-style `f`, so
  `let a: f32 = 5f;` no longer emits `const.f32 0` at every opt level on both backends.
- **`@parallel` declines four shapes and lowers serially**: a labelled loop, a plain `break` out of the
  parallelized loop, a `return` out of it, and a modelled body-local reassigned inside an inner loop. A
  `break`/`continue` inside an **inner** loop remains legal. Each of these used to change what the loop
  means — unreachable code, a chunk-dependent answer, or a race. The failure mode is a silent serial
  fallback (correct), not an error.
- **Interpreter: three uncatchable process aborts became diagnostics.** Unbounded or very deep
  recursion reports `interpreter call-depth limit exceeded (likely unbounded recursion)` through a call
  depth counter; all **five** public entry points now run on the 512 MiB big-stack worker (four
  previously skipped it); and the runtime allocation intrinsic uses `try_reserve` and returns an error
  instead of reaching Rust's allocation-error handler. The interpreter reports resource exhaustion as a
  diagnostic with exit 1, never a process abort. It cannot mirror the runtime's null-return on OOM,
  because interpreter pointers are slot indices and index `0` is a live slot.
- **Interpreter: a non-positive kernel extent means zero iterations on both backends.** All 81
  output-buffer shape reads go through one `dim!` macro that bails out of the arm with the kernel's
  documented no-op when the extent is `<= 0`, mirroring every runtime kernel's own `<= 0` guard, and
  the eight value-returning reduction reads clamp with `.max(0)` so the kernel's empty-input identity
  applies. Such an extent previously wrapped to `~usize::MAX` and aborted with a raw `capacity
  overflow` out of the backend that is supposed to be the semantic oracle. Zero extents were corrected
  the same way.
- **Interpreter: three semantic-oracle repairs.** Integer SIMD lanes are truncated to their declared
  lane width in the vector `Bin`, `Neg` and `Not` arms — a lane used to keep its full-width product and
  was observably scattered to memory, disagreeing with Cranelift; `wukong_embedding_f32` gathers
  row-at-a-time straight from live interpreter memory in ascending order, so an in-place gather
  (`out == weight`) matches the source nest and native, and the real vocabulary argument drives the
  out-of-range zeroing rule instead of a snapshot heuristic; and a mis-shaped vector operand produces a
  diagnostic instead of an index-out-of-bounds panic.
- **Cranelift: the native backend no longer promises 16-byte alignment.** Generated loads and stores
  use `notrap` memory flags rather than `trusted` (`notrap|aligned`). Cranelift's `aligned` is a
  *caller* promise this backend cannot make — the 128-bit CLIF vectorizer emits vector loads at
  addresses it knows are not 16-byte aligned, and on an x86-64 host without AVX the old flag let the
  x64 lowering embed the misaligned operand in a legacy SSE instruction and fault on the first
  vectorized iteration. Strictly weaker: it can only make Cranelift materialize a load it would
  otherwise have folded.
- **Cranelift: the raw 256-bit AVX2 emitter is gated on host ISA *and* ABI.** It requires AVX2 + FMA
  **and** `x86_64` + Windows, because the emitted body hardcodes the Win64 `rcx`/`rdx`/`r8` argument
  mapping and there is no in-kernel fallback; an unsupported host now gets a compile diagnostic rather
  than SIGILL or silent memory corruption. The module's earlier claim that Windows and SysV both pass
  the first arguments in registers it normalizes to was false and is corrected. Its lane/register
  constants alias the vectorizer's so the two bounds cannot drift, and the register free list releases
  a repeated vector operand exactly once (`a*a`, or an `Fma` sharing an operand, used to push the
  register twice and compute the wrong expression).
- **Cranelift: a narrow-float entry point returns through its own ABI.** `f32`/`f16`/`bf16` `main`
  transmutes to `extern "C" fn() -> f32` instead of sharing the `f64` arm — all three are computed in
  f32 registers, so reading 64 bits back reinterpreted a live float as a denormal and
  `fn main() -> f32 { return 42.9; }` disagreed with the interpreter oracle.
- **Cranelift: recognizer/backend drift is a loud compile failure.** A `wukong_*` kernel reaching
  `lower_call` with an arity no arm handles fails the compile naming the symbol and the arity, instead
  of lowering to nothing — which silently deleted a void kernel's call and left its destination buffer
  stale, a native-only wrong answer since the interpreter dispatches on the symbol name with no arity
  check. This is the check that catches a missed arm when adding a runtime kernel. Parallel object
  codegen reports the **first failing function by index**, matching the `WUKONG_PAR_CODEGEN=0` serial
  path, so which diagnostic is reported is deterministic. And the JIT's contract is now stated:
  `JitProgram::run` executes the entry on the 512 MiB worker, while `JitProgram::call` invokes it on
  the caller's stack.
- **Runtime: scalar/AVX2 twin agreement extended to non-finite data and past the f32 mantissa limit.**
  The row-max folds in the norm and log-softmax paths make the **scalar twin** adopt MAXPS semantics
  (`(a > b) ? a : b`) so it mirrors the unblended AVX2 chain, while the cross-entropy path resolves the
  same hazard the other way — its scalar twin keeps `f32::max` (= `maxNum`) and its AVX2 fold blends the
  accumulator back over the NaN lanes; each file is internally twin-exact, and the two deliberately pick
  **different** row maxima on a row containing a NaN; the cummax/cummin vector scan folds NaN-bearing
  blocks with the scalar recurrence; the vector arg-reduce delegates any span leaving a sentinel lane
  to its scalar twin, so an all-identity span argmaxes to index `0` rather than `-1`; the row and
  column arg-reductions dispatch to AVX2 only while the running index — which rides in an f32 lane —
  is exactly representable, handing wider rows to the exact-integer scalar twin; and an unknown vmath
  op code runs the named scalar twin instead of leaving the output untouched. The module headers'
  unconditional bit-identity claims are now true.
- **Runtime: kernel contracts stated rather than hoped for.** Cross-entropy **bounds the label** at the
  one place each kernel uses it (a negative `i32` sign-extends above the row width, so one `<` covers
  both directions): an out-of-range `target[r]` yields `NaN` loss forward and a plain-softmax gradient
  backward, never a read or write past the row, and the unenforceable in-range precondition is dropped
  from the safety blocks. `wukong_embedding_f32[_parallel]` takes its three extents as `i64` — matching
  the compiler's own import declaration — and returns early on a non-positive `t` or `h` (a non-positive
  `v` leaves no valid weight row, so every output row is zeroed instead). The scaled NT GEMM's
  doc states what its writeback implements (alpha applied **after** the beta accumulate, not the BLAS
  ordering), and only the `beta = 0` combination is reachable from `.wk` source, now pinned by a debug
  assertion at the single dispatch choke point. The file-I/O read intrinsics attempt the **open before**
  the zero-length short-circuit, so a missing or unopenable path is `-1` even when `len <= 0`, matching
  the interpreter — ordering is part of that frozen contract.
- **Runtime: thread-pool provisioning is now an invariant.** Every `_parallel` kernel that can be the
  process's first rayon touch calls `ensure_global_pool()` before forking; otherwise rayon builds its
  default small-stack registry and the runtime's later 16 MiB `build_global()` silently loses the race
  for the whole process. As a backstop the pool helper records whether `build_global` actually
  installed the pool and otherwise falls back to a lazily built private 16 MiB pool. Provisioning and
  scheduling only: no chunk boundary or fold order moves, so serial == parallel stays bit-exact. Known
  gap, documented not fixed: at pool width 1 the helper runs the body **inline on the caller**, so an
  outlined `@parallel` region body is not on a 16 MiB stack there — the gate that proves a fork lands
  on a runtime-configured pool returns early at that width for exactly this reason.
- **Runtime: env knobs and reference surface.** Two GEMM instrument knobs validate their input in pure
  helpers instead of trapping inside an `extern "C"` kernel (a `0` divisor is treated as unset; a
  KiB→byte scale saturates), so a sweep script walking either range falls back rather than aborting,
  and the bump `Arena` rejects a non-power-of-two alignment (`align - 1` wrapped and the round-up mask
  became `0`, aliasing every live region).
- **Autodiff: every silent-zero or silently-wrong gradient in reach became a refusal.** The elementwise
  VJP uses a **whitelist** of the two `velem` spellings the affine rule is valid for, so a Hadamard or
  quotient kernel is refused by name instead of differentiated with the linear rule; an `Op::Store`
  into a buffer that already has a live adjoint, whose stored value is not a constant, is refused
  (constant stores — zero-init and the loss sink — stay skippable); `Op::VecKernelCall` is covered by
  the same loud guard as an unrecognized `Op::Call`; and a call to a recognized kernel **name** with a
  non-ABI arity — reachable by declaring the symbol in an `extern "C"` block — is a clean diagnostic
  instead of an index-out-of-bounds panic. Consequence: a mixed scalar+kernel loss that previously
  "succeeded" with a clobbered or zero gradient now fails to compile under `--emit=grad` / `--train`.
  The supported envelope is narrower than the old behaviour suggested, and `--emit=grad` never emits a
  silently-zero gradient — a missing rule is a hard error.
- **Autodiff: two rules added, and the coupling contract reinforced.** A self-dot reduction —
  `loss = Σ x[i]²`, lowered with both reduction operands the same buffer — now differentiates to one
  elementwise scaling (it used to fail with a misleading multiple-contributions error), so a
  sum-of-squares loss / L2 regularizer works. And `wukong_norm_f32_parallel` is mirrored into the tape
  (`Syms` / `is_kernel` / `diff_kernel_call`), so a `@parallel` batched norm — the shipped transformer
  shape — differentiates through the same rule as its serial twin. This is the standing rule: every new
  `_parallel` recognizer arm in `mir_build` **must** get a tape counterpart, or every `@parallel`
  backward through it breaks. Diagnostics now name the offending callee, and the reduction fallback
  message lists its supported set correctly.
- **Autodiff: known limitation, documented not fixed.** When a loss reads a kernel-written buffer with
  a scalar `load` rather than through another recognized kernel, the kernel's output adjoint is never
  seeded and the gradient is silently zero. The scalar-load path does now register its buffer as
  having contributed, so a later kernel-path overwrite cannot clobber it, and a matmul adjoint into a
  scalar-loaded buffer accumulates rather than overwrites.
- **GPU (`--backend=gpu` offload)**: the LayerNorm PTX computes variance in the numerically stable
  two-pass form `mean((x - mean)²)` instead of the one-pass `E[x²] - mean²`, which catastrophically
  cancels in f32 and made whole rows NaN; both siblings — the CPU runtime kernel it replaces and the
  gpu-native lowering — already used the stable form, so this was a CPU-vs-GPU divergence, and
  LayerNorm now agrees across interp / native / gpu / gpu-native. The norm offload also **declines** op
  codes with no PTX entry (log-softmax, L2-norm) and falls back to the CPU kernel instead of aborting
  the process on ordinary user input, joining the existing vmath and reduction gates.
- **GPU (gpu-native MIR→PTX)**: an `i1` in memory is accessed as **one byte** — it fell through to the
  i32 arm, and a 4-byte access at an odd address is `CUDA_ERROR_MISALIGNED_ADDRESS`, reachable from
  `struct F { a: bool, b: bool }` or `[bool; 8]`; a narrow unsigned float→int cast stays sign-extended
  to its MIR width, so it compares equal to the same number materialized as a constant; bf16/f16 values
  round to their grid at a **const** and at an `FpExt` widen, not only at a store, so gpu-native's own
  `-O0` and `-O3` agree (interp and Cranelift already rounded at all three points); and an aggregate or
  void load/store declines with `UNSUPPORTED:` rather than emitting a plausible 4-byte access. Coverage
  grew as well: the seven `_parallel` runtime symbols `mir_build` emits — the f32 activation kernel and
  the six low-precision `{dot,sum,reduce}_{bf16,f16}` symbols — are classified as their serial kinds, so a program whose
  `@parallel` function is recognized as an activation or a low-precision reduction is no longer refused
  outright by gpu-native nor declined by the megakernel, and the remaining `expect` sites became
  `UNSUPPORTED:` declines.
- **GPU: an out-of-contract launch is a recoverable decline, never a panic or a sticky driver fault.**
  The reduction launch derives its argument list from the op (a mismatched pair read one slot past the
  argument vector and latched a process-sticky illegal-address error); the autotune cache honours a
  cached token only when this build has that candidate and it is applicable at the shape, otherwise
  counting it as a miss and re-tuning; the resident-backward, AdamW and paged-KV launchers check every
  buffer length against the geometry the kernel recomputes; the flash planner owns the (entry, config)
  pairing so a shape no kernel covers cannot be built; the tile generators assert their shared-memory
  staging and warp-grid preconditions; and an op with no kernel declines with
  `CUDA_ERROR_NOT_SUPPORTED` instead of panicking inside the compiler process. In the serving stack a
  request's generation length is floored at 1 (a zero-length request underflowed its remaining counter,
  so its slot never retired and never returned its KV blocks) and the host-side decode advance is
  all-or-nothing behind a feasibility pass. The int4 dequant reference discriminates on the field every
  launcher dispatches on (`zeros`) rather than the descriptive `signed` flag, and the host fp8 E5M2
  encoder handles subnormals and signed zero like the hardware instruction — the E4M3 twin has the
  identical defect and is a known open item.
- **CLI: two flag combinations are now rejected instead of silently doing half the work.** `--run`
  cannot be combined with `--emit=<stage>` — the driver dispatches exactly one of them and dropped the
  other while still exiting 0, and *which* one won depended on where the stage sat relative to the run
  block. And `-o` is accepted only with `--emit=obj` or `--emit=exe`, and never with `--run`; `USAGE`
  now states the real semantics: **only `--emit=obj` and `--emit=exe` write a file; every other stage
  prints its artifact on stdout.**
- **Driver: MIR is verified before every backend exit.** `--emit=llvm-ir`, `--emit=obj` and
  `--emit=exe` verify through the same helper `--run` uses. `--emit=mir` / `--emit=mir-high` stay
  ungated at that point on purpose — they verify *after* dumping, so a broken program still prints the
  MIR that shows why. Note the optimizer's per-pass verify-each is `#[cfg(debug_assertions)]`, so
  before this a release compiler verified nothing at all on the AOT path.
- **Driver: the compiler no longer aborts when its output pipe closes.** Artifact writes (tokens, ast,
  llvm-ir, mir) and the diagnostic and verifier-ICE lines go through fallible writes, so
  `wukongc --emit=mir p.wk | head -1` exits with the compile status instead of crashing — the
  workspace's `panic = "abort"` had turned a failed `print!` into a process abort. Exit codes are
  meaningful under truncation.
- **Driver: `--emit=exe` generates its link inputs in a pid-keyed scratch directory** that is removed
  when the link finishes, instead of writing `{stem}_shim.rs` / `{stem}_rt.c` into the process's CWD —
  which overwrote and then deleted a user file of the same name and raced between concurrent same-stem
  links. The object still lands beside the source. The rustc-driven link also passes `-C panic=abort` to
  match the workspace's release panic strategy, without which a release-built compiler's sibling rlib
  rejects the shim.
- **Driver: three `--grad`/`--train` defects fixed.** The default `--grad-wrt` has a *training* flavour
  that excludes the loss-output parameter, so `--train --grad-of=<fn>` works with the documented
  default instead of always failing; a repeated index (`--grad-wrt=0,0`) is rejected rather than
  emitting a backward whose extra gradient buffer is never written; `--train-steps` no longer
  pre-reserves its trajectory vector, so a huge value is not an allocator abort; and `--emit=grad`
  propagates verification failure instead of printing an ICE and exiting 0.
- **Test gates now pin documented contracts.** Object emission must be byte-identical across three
  fresh `--emit=obj -O2` processes per program, and the `WUKONG_PAR_CODEGEN=0` serial path must be
  byte-equal to the parallel one on the 24 largest programs (the ones with enough functions for the
  parallel path to engage); diagnostic emission must be identical across three fresh
  `--error-format=json` processes per compile-fail fixture; all three determinism tests count
  comparisons and require corpus coverage so they cannot pass vacuously. The limit is stated: the
  seeds only perturb std `HashMap`/`HashSet`, and the optimizer's `FxHash` maps have a fixed seed, so
  these gates say nothing about insertion-order dependence there. The AOT-exe gate probes **once**
  which of the driver's two link paths is available, fails on a genuine link failure with the captured
  stderr, refuses to pass having linked zero fixtures, and uses a per-process scratch directory.
- **The examples corpus carries declared classes.** `examples/*.wk` are automatically interp-vs-native
  differential and opt-invariance fixtures — and an example that *fails to compile* satisfied both
  agreement checks, because both backends failed identically. Each example now declares what running it
  must produce: `Runs` (exit 0, non-empty stdout) is the default, so a new example must execute or be
  declared; `NotLowered` (exit 1 + `C0001`) covers `matmul.wk`, `softmax.wk` and `vadd.wk`; `NoEntry`
  covers `gpt2_config.wk`; and `DataGuarded` covers the three GPT-2 programs that take a data-absent
  early return and are excluded from the agreement checks, with a printed reason, when the weight blob
  *is* reachable. A separate gate re-derives the whole GPT-2 offset table from the exporter's own
  tensor order, needing neither torch nor the blob. Five example headers were rewritten to the dispatch
  the emitted MIR actually shows — `--emit=mir -O2` is the only authority, because recognized kernels
  are gate-blind — and `examples/gpt2_forward_bench_small.wk` was added to carry the same dispatch
  surface at tiny dims with no file I/O, so it is fit to be a differential fixture. The GPT-2 export
  tool resolves its output directory from `$GPT2_DATA_DIR`, then `argv[1]`, then `<repo>/data/gpt2`
  instead of hard-coding one machine's path, and the verifier makes the logits artifact's **age** part
  of its verdict so a verify cannot re-assert a result for a run that never happened.
- **Corpus shape and placement rule.** `tests/run` holds 332 `.wk` fixtures and `tests/fail` 110, plus
  eleven inline lexer/parser probes — together covering all 29 catalogued codes. A program that must be
  **rejected** belongs in `tests/fail`, because every `tests/run` program is required to reach
  `--emit={mir-high,mir,llvm-ir}` successfully (two `sdpa`-decline fixtures moved for exactly that
  reason). `tests/run/*.wk` supports a non-default `// RUN:` directive, and eleven element-type and
  aliasing fixtures carry `// RUN: --run --backend=native` so their expected output pins the backend
  that was wrong.
- **GPU test knobs.** `WUKONG_GPU_REQUIRED=1` turns the GPU suite's `[skip]` lines into assertions and
  `WUKONG_PEER_REQUIRED=1` does the same for every peer/toolchain skip — both no-ops by default, so a
  genuinely CPU-only or peer-less box still passes. Skip lines now carry the driver's actual error, so
  an exclusive-mode device or an empty `CUDA_VISIBLE_DEVICES` is not misreported as a machine with no
  GPU, and the paged-attention module's device-free parts (PTX generators, int8 quantizer, f64
  reference) run under a plain `cargo test`.
- **Reverted: the pointer arm of `lower_bool_cond`.** `PtrToInt(p) != 0` is **not** a null test in the
  interpreter, whose addresses are slot indices starting at `0`, so the first pointer taken in a
  function reads as null (`if p` printed 0 under interp and 1 natively) — trading a loud error for a
  silent interp/native divergence, the worse failure. A **pointer** condition (`if p`, `while p`,
  `if p && …`, `assert(p)`) therefore still reaches `cond_br` unchanged and both backends reject it
  with raw verifier text and no span. Do **not** document C-like truthiness for pointers: a correct
  lowering needs a null test the two backends genuinely share — a dedicated `IsNull` op, or the
  interpreter reserving address `0` as never-allocated. The same commit's coercion of integer operands
  to the six activation-**backward** intrinsics was kept and is pinned by a fixture.
- **Refuted, then raised: the interpreter's release call-depth cap.** Each limit is the largest value
  that rejects **no** depth known to complete, so an earlier, lower pair was refuted for rejecting
  depths that run fine — the pre-guard interpreter simply ran until the stack gave out, and a cap
  chosen for "safety margin" is a regression over that band, not a precaution. The debug limit
  additionally has to clear the depth the Cranelift differential gate exercises, or the *interpreter* —
  the semantic oracle — refuses a depth native completes, reintroducing the very divergence that gate
  exists to catch.

### Capability — GPT-2 124M end-to-end inference + typed file-I/O intrinsics (2026-07-13)
- **File-I/O intrinsics — headerless raw little-endian typed blobs**: `read_<T>` / `write_<T>` for `T`
  in `{f32, f64, i32, i64, i8, u8}` read and write a flat file of that element type
  into / out of a `[]T` buffer, with no header. `read_<T>(path, buf)` returns
  `min(buf.len, file_bytes / sizeof T)` on success, `-1` if the file cannot be opened, and `-2` on a
  mid-read I/O error (`0` when the buffer length is `≤ 0` **on an openable path** — the open is
  attempted first, so a missing file is `-1` regardless of the buffer length; that ordering is part of
  the frozen contract); `write_<T>(path, buf)` creates/truncates the
  file and returns the element count written (`-1` if the file cannot be created, `-2` on a write
  error). Both are differentially tested **interp == native** (round-trip, partial-read, and
  missing-file run tests), with compile-fail tests for path / buffer / arity misuse.
- **GPT-2 124M inference end-to-end, matching HuggingFace**: `examples/gpt2_infer.wk` compiles and runs
  the **real pretrained OpenAI GPT-2 124M** (124,439,808 parameters) as an ordinary Wukong program on
  the native Cranelift-JIT backend. It loads the weights from a flat little-endian f32 blob (exported
  from HuggingFace `transformers` by `tools/export_gpt2.py`) via `read_f32`, and the prompt token ids
  via `read_i32`, then runs the full forward — token + learned positional embedding, 12 pre-LayerNorm
  blocks (biased QKV, 12-head causal attention, tanh-GELU MLP, residuals), final LayerNorm, tied LM head
  → logits `[5, 50257]`. **The next-token logits match HuggingFace's reference to a relative max error
  of 1.87×10⁻⁶** (`max|Δ| = 2.44×10⁻⁴`, `max|ref| = 130.28`), and the argmax next token is
  **1757 (" John")** for the prompt *"Hello, my name is"* — matching HuggingFace exactly
  (`tools/verify_gpt2.py`). This is a **numerical-correctness / capability** result — **inference, not
  training; no speed comparison was measured or is claimed**. **Tokenization is external** (the `.wk`
  consumes integer token ids produced by the HuggingFace tokenizer). The full run is **native-only**
  (the interpreter cannot hold 124M parameters); the reduced-config twin `examples/gpt2_infer_small.wk`
  runs the same forward pass **bit-identically interp == native** as the CI examples differential +
  opt-invariance gate.

### Performance + honest measurement — perf-perfect session (2026-07-10/11)
Three-target directive round (A: parallel GEMM ≈ oneMKL + near-linear model scaling; B: beat fused
`torch.compile`; C: compile-time floor). Same-run/adjacent instruments; full ledger
`prompts/results/perf-perfect-session1.md`.
- **Beats fully-optimized `torch.compile`, both batch sizes (Target B MET)**: vs TorchInductor
  max-autotune, fullgraph, warmed — ATEN-pinned because the Windows Inductor CPP FP32 GEMM template
  (`cpp_CppMicroGemmFP32Vec`) is broken, so ATEN/MKL is torch's strongest *working* CPU GEMM
  (disclosed) — all-threads, the 12-layer GPT-2-class stack runs **S=512 1.39–1.81× / S=128
  1.07–1.43× FASTER** across three independent same-day rounds (isolated per-side probes, torch at
  its best; ≤1e-6 cross-check, serial==parallel bit-exact). Wukong-1c also beats compiled-torch-1T
  at both shapes, and beats torch's strongest *eager* config too (S=512 1.07–1.37×). Beating eager
  does not count — this is the compiled bar.
- **Dynamic block-claiming is the parallel-GEMM default (`WUKONG_GEMM_DYN`)**: an atomic block-claim
  queue replaces the static split; root-caused the mid-size variance to OS-preemption straggler
  episodes on the worker pool (per-call min stays fast, p50/p90 widen) and halved it. Standing vs
  MKL-all, same-run: **512³ ~98–99%, 1024³ ~104%, 2048³ 91%, 4096³ 93%, 256³ ~94–110%** (was 70–83%
  @512–1024³); skinny NT shapes **75–112%** (4/6 at/above parity). A `(96,96)` 2D block candidate
  fixed wide-M/narrow-N mis-shaping (512×768·768ᵀ 75→89%).
- **Near-linear-as-physics-allows model scaling (Target A(b) MET)**: pool unification + a **width-1
  serial fast-path** (T=1 now at serial parity, was 0.62–0.80×). S=512 curve
  **1.00/1.70/2.13/3.00/3.68/4.70× (T=1..16)**, default 4.44×, best 5.22× — tracking the ~5.0–5.4×
  same-run hybrid MKL ceiling with no Wukong-attributable serial shortfall.
- **Compile-time floor characterized (Target C)**: central AC run — 296-file corpus **168.1 ms
  front→object** (0.57 ms/file); backend 73.4% (Cranelift codegen 91.7% / object-write 8.3%),
  optimize 16.3%. compile-vs **7.4–11.7×** vs gcc/g++/rustc (both subprocess, obj, -O2); in-process
  vs spawned-toolchain ~306× JIT; `--emit=exe` ~95% rustc-link. Release verifier-off + parallel
  per-function codegen (byte-identical 296/296). See `docs/compile-floor.md`.
- **Documentation reconciled to these records**: a five-way doc audit refreshed every stale perf
  claim (`README.md`, `docs/metrics.md`, `BENCHMARKS.md`, `docs/internals.md`, `docs/compile-floor.md`,
  `next-steps.md`) to the 2026-07-11 numbers, corrected the model-vs-torch bar from *eager* to
  *`torch.compile`* everywhere, and replaced bare-absolute-GFLOP/s headlines with %-of-MKL /
  %-roofline / same-run ratios.

### Performance + honest measurement — perf-sota session 3 (2026-07-09/10)
Ten-target directive round; every baseline re-proven before optimizing, every win from the real
pipeline via same-run/adjacent instruments (full ledger: `prompts/results/perf-sota-session3.md`).
- **`@parallel` head-loop regions — the model now beats all-threads PyTorch at S=512**: an
  independent-iteration `for` loop with body-local scratch inside an `@parallel` fn outlines into
  its own MIR fn + one `wukong_parallel_for` region (the whole-fn outliner's exact contract —
  zero backend changes; 16 MiB worker stacks for the privatized frames). Legality is a
  conservative affine-disjointness proof (one `hh*C` term per written array, outer strides erased
  mod C·N, inner terms bounded below C; declines to serial on anything unproven — captured-scalar
  writes, opaque indices, calls, `print`, slices, cross-iteration deps), each iteration runs the
  SERIAL kernels (identical per-iteration op order ⇒ serial == parallel bit-exact), and autodiff
  declines loudly (no silent zero gradients). The model bench spells its attention head loop that
  way naturally; e2e fixtures pin interp == native == @parallel at every opt level, even + ragged
  head counts, and the decline case. Result (two roofline-validated rounds, all cross-checks
  green): S=512 `@parallel` **440 → ~312 ms** ⇒ **1.10–1.24× FASTER than all-threads eager
  torch** (was 1.01–1.63× behind); S=128 parity (1.19× faster / 1.02× behind at round noise; was
  1.5–1.9× behind); model `@parallel` scaling **3.1–3.6×** (campaign start: ~1.5–2.1×); the
  multicore stack ~61–69× idiomatic single-thread C.
- **Size-keyed 2D parallel GEMM dispatch**: the cooperative shared-pack redesign
  (`sgemm_2d_shared` — panels packed once per K-block) was built, gated bit-exact, and then
  **refuted at mid/large shapes by adjacent ABBA in both orderings** (per-block packing wins
  1024³ ~462 vs ~385 GF/s; its "redundant" packing is each worker warming its own L2, while a
  shared pack hands consumers panels another core packed, plus a per-K-block barrier). Shipped
  as a size-keyed default: per-block ≥2²⁶ MACs, shared-pack + 1 Mi-MAC work-scaled tasks in the
  small band, parallel gate 2²⁶→2²³ (256³ engages at **1.7–2× over serial = 102–123% of
  adjacent MKL-all**; was ~39–52% deliberately-serial). Mid-size standing: **70–83% of MKL-all
  @512–1024³, 88–93% @2048³ vs a healthy peer**. New skinny-M/row-rebalance block-shape policy
  (+16–25% at the model's skinny FFN shapes, MKL-anchored vs the old binary).
- **C-tile prefetch in the GEMM microkernel prologue**: 2048³ single-core
  **86–88% → 96–99% of MKL-1c** (MKL-anchored ABBA, both orderings; `WUKONG_GEMM_PF_C=0`
  reproduces the old tail). Hint-only — bits unchanged everywhere.
- **`@parallel` streaming maps go multicore**: `wukong_velem_f32_parallel` (fixed-chunk,
  serial==parallel bit-for-bit) + recognizer/interp/cranelift/autodiff-tape wiring, so a mixed
  `@parallel` function's residual-add/saxpy loops (the transformer block's hot elementwise ops)
  no longer run serial; e2e fixture pins interp==native at every opt level.
- **Model bench (3 valid rounds)**: `@parallel` scaling **S=128 2.19→2.96×**; vs all-threads
  torch **S=128 1.08–1.20× behind (was 1.5–1.9×)**, S=512 an honest **1.01–1.63× range — the
  width is the peer's power-state swing** (torch-Tn 443→271 ms across rounds; Wukong held
  ~440 ms in all of them — the parallel path does not yet ride clock upside; head-loop
  parallelism is the named open lever). Single-thread: 1.10–1.16× faster than torch-1T @S=512.
- **exp ldexp restructure reverted on measurement**: the bit-identical exponent-field-add tail
  (exhaustively verified over 2³²) measured a consistent ~10–15% throughput **loss in BOTH
  thermal states** (old-vs-new binary interleaved ABBA; tanh, which composes exp8, confirmed
  independently) — port rebalancing cannot beat a clock throttle that slows every port.
  Honest vmath standing vs VML across states: **tanh 2.7–2.9× faster; exp 1.23–1.45× and log
  ~1.25× slower** (the earlier single-session "exp 1.05× faster / log 1.14×" did not reproduce).
- **GPU: warp-specialized flash shipped where it wins**: 2-warp named-barrier anti-phase
  (`flash_d64_ws`) + 3-stage-ring (`flash_d128_ws3_lm`) kernels, tolerance-gated vs the f64
  oracle; the clock-cancelled A/B wins **only 4–6% @S=4096** (tie @2048, 10–52% loss @≤1024),
  so the default route is exactly S≥4096 with pins keeping the losing regimes unroutable
  (`WUKONG_FLASH_WS=0` kill-switch). The long-S cuDNN gap is structural (SFU-bound) and now
  measured shut as a scheduling problem.
- **GPU 4096³ GEMM: v2cs streaming epilogue** (+2.7% round-robin over the swz base →
  **76.8% of cuBLAS-f16 / 80.4% of the honest f32-out peer**) dispatched at A+B ≥ 48 MB; the
  new `f32-out cuBLAS peer column` makes the C-write dtype asymmetry visible (~10pp of the old
  "gap" was peer flattery). 3-stage pipe / raster re-tunes / launch-bounds all measured losses.
- **Serving goodput ceiling doubled**: Bcap parameterized end-to-end with KV-budget guards,
  graph-driven scheduler (one cached `cuGraphLaunch` per step, **bit-identical to eager** over a
  96-request drain), bounded-look-ahead first-fit admission, an honest static-batching peer, and
  opt-in int8-KV (1.88× smaller cache; exact round-trip gate). **Bcap=256: 85.6× goodput vs
  fill=1** (33.1k tok/s; old ceiling 38–39× @Bcap=64, reproduced), scheduler drain **1.14–1.27×
  vs static batching** with the amortization-vs-scheduling decomposition disclosed.

### Performance + correctness — perf-sota session 2 (2026-07-08)
- **vmath exp/log to (near-)VML parity**: the transcendental dispatch loop gained a ×4 ILP unroll +
  an NT-store streaming regime, then both cores were rewritten as 8-bucket in-register-LUT
  (`vpermps`) reductions — exp `2^(j/8)` table + degree-3 residual poly (~1.3 ULP, exhaustively
  swept), log reciprocal/ln tables + degree-5 poly (≤6.9e-7 rel, exhaustive over [0.25,4)) — each
  mirrored value-exactly in the scalar twins and the inlined-MIR emitters. Same-run vs oneMKL VML:
  **tanh 2.4–3× faster, log 1.14×, exp 1.05×-faster(cool)–~1.3×(throttled)**, from the former
  ~1.7–2× loss; log2/log1p now 9.9×/7.7× vs scalar C.
- **2D block-parallel GEMM shipped as the default parallel path**: BLIS/MKL-style per-thread
  L2-resident C-block ownership (`sgemm_2d_blocks` — MR/NR-aligned block grid, per-worker pack
  scratch, one task per block, no barriers) replaces the row-panel/shared-B decomposition —
  adjacent-run ABBA **1.26–1.65×** at 512–1024³ (512³ ~182→~305 GF/s, 1024³ ~365→~460), lifting
  mid-size `@parallel` from ~46–72% to a power-state-stable **66–69% of all-threads oneMKL**
  (87% @2048³ same-run). Bit-exact vs the serial kernel by construction (one owner per C block,
  ascending K-blocks, same kc grouping; pinned by `sgemm_2d_blocks_matches_serial`);
  `WUKONG_GEMM_2D=0` opts back to the previous path as an adjacent-run instrument. Downstream,
  model-bench `@parallel` scaling rose from ~1.5–2.1× to **1.9–3.8×** and the all-threads-torch
  gap at S=512 stabilized at **~1.1×** (1.07/1.10× across two rounds; was 1.05–2.5×
  thermal-dependent).
- **Parallel GEMM (precursor work)**: A+B packing fused into one parallel region per K-block
  (+7.5% @512³ same-run, `WUKONG_PACK_SPLIT_REGIONS=1` kill-switch) and a persistent-broadcast-
  region variant (one pool wake per call). Honest negatives recorded in-code: the persistent
  region A/Bs as a wash, a smaller mid-size pool and hard worker pinning
  (`WUKONG_GEMM_AFFINITY=1`) both measure slower — scheduling was never the mid-size gap; the
  2D decomposition above was.
- **gpu-native (MIR→PTX) correctness, 8 programs repaired**: the `mrt_velem` device kernel ignored
  the Hadamard/Div mode bits (computed `x+y` for `x*y`/`x÷y`); `mrt_norm` sent log-softmax and
  L2-norm to the RMSNorm branch; both sreduce kernels lacked the |x|-sum/|x−y|-sum ops (silent 0);
  float→narrow-int casts truncated mod 2^w instead of saturating; and megakernel mode null-deref'd
  through tid0-guarded pointer slots (`CUDA_ERROR_ILLEGAL_ADDRESS`). Corpus gate: 193/274 matching
  the interp oracle, zero mismatches/faults (81 honest UNSUPPORTED skips); a device fault can no
  longer cascade — the harness records the root fault and reports later programs as NOT RUN.
- **D=128 flash attention productionized**: `wmma_flash_applies/entry` now dispatch the
  `flash_d128_mp_lm` ldmatrix kernel (1.11–1.20× cutlass mem-efficient @S≤1024) — D=128 previously
  fell back to the f32 flash. A single-buffer occupancy probe pins the D=64 long-S plateau as
  SFU/serial-softmax-bound (2× occupancy = 1.03× wash).
- **End-to-end model bench hardened**: two full-peer rounds (C, `-ffast-math` C, PyTorch eager
  1-thread/all-threads), all cross-checks <2e-6, interp gate + serial==@parallel bit-exact:
  ~19–21× C(gcc), parity vs torch-1T (up to 1.49× faster @S=512), behind all-threads torch
  multicore — post-2D-GEMM a stable ~1.1× @S=512 and 1.5–1.9× @S=128 (see the 2D bullet above).

### Performance — close-the-NVIDIA-gap campaign (vs the vendor libraries)
- **CPU GEMM to oneMKL parity** (`perf/cpu-library-grade`): size-adaptive `select_kc(k)` + a private
  physical-core rayon pool put single-core GEMM at 102–104% of MKL-1-thread (256/512³) and 84–95%
  (1024–4096³); `@parallel` reaches 86–129% of MKL-all-threads at ≥2048³. MKL cblas/VML peers added to
  `wukong_xbench`. (vmath's then-open ~1.7–2× VML loss was closed in the 2026-07-08 session above.)
- **fp16/bf16 large-GEMM cliff vs cuBLAS** (`perf/gpu-gemm-cliff-2`): the no-pad ldmatrix+XOR-swizzle
  workhorse takes 2048³ to ~87–90% and 4096³ to ~83% of cuBLAS (past the prior 77% `mma.sync` ceiling).
- **int8/fp8 GEMM vs cuBLAS IMMA / cuBLASLt** (`perf/gpu-quant-2`): int8 beats IMMA at 2048³
  (~96–105%); the fused int8 GEMM+dequant beats the cuBLAS GEMM+dequant chain 1.1–2.2×; fp8 is 82–151%
  of cuBLASLt. w64 warp-tile + autotuner candidates.
- **Fused attention vs a genuinely-fused FA2-class peer** (`perf/gpu-attention-2`): vs PyTorch SDPA's
  cuDNN / cutlass mem-efficient backends, the fused flash wins fused-RoPE 1.8–5.7× and causal D=64
  1.03–1.16× (beats both) for S≤512; D=128 ldmatrix beats cutlass for S≤1024.
- **Conv vs cuDNN** (`perf/gpu-conv-2`): a real cuDNN-9 peer plus implicit-GEMM + Winograd
  F(2×2,3×3)/F(4×4,3×3) + a fused epilogue + `conv2d_best` per-shape dispatch reach cuDNN
  parity-to-win (1×1 3.5–5.6×, deep-channel 0.93–1.21×).
- **GPU serving stack** (`perf/gpu-serving`): vLLM-style paged KV-cache + Orca continuous batching +
  whole-model decode CUDA graph + int8 KV (3.88× footprint) + a TP partition sim — 39.1× batching
  goodput at fill=64.

All measured same-run (clock-invariant) on a mobile RTX 4050; the documented gaps and measured negative
results are recorded in `prompts/results/`, and every kernel stays gated against the interpreter oracle.

### Added
- **Front-end**: lexer (with `@attributes` and error recovery), recursive-descent + Pratt parser,
  AST with a pretty-printer, and `--emit=tokens|ast`.
- **Types & semantics**: the shared type vocabulary (`wukong_types`), name resolution, type
  checking, and **compile-time shape checking** for tensors (rank/dimension unification, symbolic
  dims), with errors `E0501`/`E0502`.
- **Middle-end**: block-parameter SSA MIR, a builder, a pretty-printer, and an SSA/dominance
  verifier; AST → MIR lowering (alloca-per-local). There is **no** `MirLevel` invariant —
  `MirLevel::High` is constructed nowhere, `Program::new` starts at `Low`, nothing reads
  `Program::level`, and `--emit=mir-high` means *pre-optimization* MIR, not a different IR level. See
  the 2026-07-29 entry for the verifier's actual rule list.
- **Optimizer**: a fixpoint pass manager backed by CFG and dominator analyses (Cooper–Harvey–Kennedy
  immediate dominators + dominance frontiers), with whole-program leaf-function `inlining`, `mem2reg`
  (promote scalar slots to block-parameter SSA), `simplify` (constant folding + algebraic identities
  + self-comparison folding), `simplify-cfg` (constant-branch folding + straight-line block merging +
  unreachable-block pruning), `simplify-phis` (dead/trivial block-parameter elimination), `dce`,
  `cse` (dominator-tree value numbering with load forwarding), `dse` (dead-store elimination), and
  `licm` (loop-invariant code motion), wired across `-O0..-O3`. In debug builds the pass manager
  verifies the MIR after every pass. Across the run suite and kernels, `-O3` removes ~42% of IR ops
  (~48–54% on the heavy transformer/GEMM kernels) and runs ~1.5–2.5x faster than `-O0` under the interpreter.
- **Back-ends**: a zero-dependency MIR interpreter (`--run`), a from-scratch **native Cranelift
  backend** (JIT + `--emit=obj` host object + `--emit=exe`, the latter linked via a rustc-driven link
  that falls back to the system `cc`/`$CC` — **no LLVM toolchain**), and a textual LLVM-IR emitter
  (`--emit=llvm-ir`, text only — emitting it needs no LLVM installed). The native backend is
  differentially tested against the interpreter bit-for-bit.
- **Matmul → tuned GEMM dispatch**: the compiler recognizes a matmul loop nest — the `ikj` accumulate
  and `ijk` dot-product forms, including the `nn.Linear` `C = A·Bᵀ` spelling — and lowers the whole
  nest to a register-blocked (6×16), cache-tiled, packed **AVX2/FMA** GEMM microkernel in the runtime
  (`wukong_sgemm` / `_nt` / `_parallel`). On a Meteor Lake laptop this beats `gcc -O3 -march=native`
  on the naive nest by **~2.4–3.5× single-thread and ~18× parallel** on `C = A·B` (and **~20–75× on
  `nn.Linear`**, where naive C stays latency-bound), the lead growing with matrix size. The serial
  kernel holds ~90–105 GFLOP/s (~80% of one P-core's AVX2-FMA peak; ~105 pinned at 512³); the parallel
  one packs panels across cores — reusing one pack-scratch allocation across all cache blocks instead
  of re-allocating per K-block (≈+45% at 1024³, ~395–434 GFLOP/s) — and skips threading below a work
  threshold. The 6×16 microkernel stores full tiles straight to C and unrolls the K loop ×4. The interpreter calls the identical
  kernel (marshalling its memory) so the oracle stays exact.
- **Auto-vectorization**: straight-line elementwise loops (incl. branchy ones via if-conversion) and
  float **reductions** (reassociated to vector-lane accumulators) lower to SIMD automatically;
  `x + y*z` contracts to a hardware FMA; adjacent same-range loops fuse. Reductions (`dot`, L2 loss)
  run ~2.6–2.8× faster than serial C. The general vectorizer emits 128-bit CLIF by default (Cranelift's
  x64 vector ISA still caps there — a 256-bit `f32x8` SSA value is rejected, per the
  `cranelift_still_rejects_f32x8` tripwire) and dispatches to a **raw 256-bit AVX2 machine-code path**
  (VEX-encoded via `iced-x86`) for large trip counts (trip-gated; kill-switch `WUKONG_P4_NO_256`;
  **x86-64 Windows hosts with AVX2 + FMA only** — the emitted body hardcodes the Win64 `rcx`/`rdx`/`r8`
  argument mapping, and any other host gets a compile diagnostic rather than a fallback).
- **Transcendental → 256-bit AVX2 dispatch**: a pure `out[i] = f(x[i])` loop for **35** functions —
  `exp`/`log`/`expm1`/`log1p`/`tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`mish`/`selu`/`tanhshrink`/
  `hardsigmoid`/`hardswish` plus **`softsign`** (bounded poly activation) and **`logsigmoid`** (the stable
  log-sigmoid behind binary-cross-entropy-with-logits / contrastive losses), **`sin`/`cos`/`tan`/`atan`/`asin`/`acos`**
  (RoPE rotary embeddings and the geometry/3D-vision/graphics-ML angle ops), **`erf`** (exact
  BERT/GPT-2 GELU), **`exp2`/`log2`/`exp10`/`log10`** (FlashAttention base-2 softmax, quantization, and
  base-10 decibel/log-scale features), **`cbrt`** (the all-real cube root — LAB color, variance-stabilizing
  transforms — completing the `sqrt`/`rsqrt`/`cbrt` root family), and the full
  **hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh`** (`atanh` = the Fisher z-transform; the
  inverse trio powers hyperbolic/Poincaré embeddings and normalizing flows) — lowers to a tuned **256-bit AVX2/FMA runtime
  kernel** (`wukong_vmath_f32`) — the width Cranelift's general vectorizer can't emit (it caps at
  128-bit SSE). `silu` (Llama/SwiGLU) and `gelu` (BERT/GPT-2/ViT) are first-class intrinsics; the
  kernel's per-element op sequence mirrors the inlined Cephes/A&S polynomial, and the interpreter marshals
  through the identical kernel, so the differential oracle stays exact and dispatched/composed forms
  agree. A multi-statement (fusion-merged) body dispatches one kernel call per activation, and an
  `@parallel` activation dispatches each thread's chunk — so it runs multicore × 256-bit. Versus C's
  scalar `libm` (which can't vectorize a loop with a call), the family runs **~4–11.5×
  faster** single-thread (`asinh`/`acosh` and `sin`/`cos` win most — `libm`'s `asinhf`/`sinf`/`cosf`
  are heavier than `expf`), ~28× `@parallel`. The in-process `vmath_throughput` probe (kernel vs the
  scalar `libm`-call loop gcc/rustc are forced to emit) confirms it locally: exp ~5.9×, gelu ~5.5×,
  tan ~3.6×, asin ~5.4×, exp10 ~13×, logsigmoid ~4.6×.
- **Two-arg transcendentals → 256-bit AVX2 dispatch (`pow`/`atan2`/`hypot`)**: `pow(x,y) = exp(y·log(x))`,
  `atan2(y,x)` (the full-circle angle — geometry, robotics, complex argument), and `hypot(a,b)` (the
  overflow-safe 2-norm) get a two-input kernel (`wukong_vmath2_f32`): a `for j { out[j] = f(x[j], y[j]) }`
  loop lowers to the **256-bit** AVX2 kernel (the same width jump that ~doubled the single-arg
  activations), and the three kernels mirror the inlined `emit_pow`/`emit_atan2`/`emit_hypot` op-for-op
  (via the shared exp8/log8/atan8/√), so dispatched == composed and the interpreter marshals through the
  identical kernel — interp == native, -O0 == -O3. A composed/scalar use (or a body that fuses with an
  adjacent store loop) still lowers to the inlined ≈1-ULP poly (128-bit auto-vec). Either way a win over
  C's scalar `powf`/`atan2f`/`hypotf` (a loop with the call won't vectorize).
- **Streaming elementwise → 256-bit AVX2 dispatch**: a recognized streaming map
  (`out[i] = act(a·x[i] (+ b·y[i]) + c)`, incl. ReLU/ReLU6) lowers to `wukong_velem_f32` and a Horner
  polynomial to `wukong_vhorner_f32` — both true 256-bit AVX2/FMA, unrolled, emitting **non-temporal
  stores** once the working set spills L3 (the read-for-ownership-skipping store gcc/rustc won't emit).
  This turns the former memory-bound *ties* into wins: saxpy ~1.3×, poly ~1.2× at `N=2²⁰`, widening to
  ~1.3–1.6× at realistic >L3 tensor sizes. A velem **identity-affine fast path** (skip the wasted
  `fma(1·x+0)` for a bare `relu`/copy) plus gating the software prefetch on the DRAM/non-temporal
  regime removed a ~1.2× `relu` regression at L3-resident sizes (now a clean tie there, ~1.4× at >L3);
  `vhorner` runs **six** independent Horner chains (was four — a 5-deep dependent-FMA chain needs ~8 in
  flight to fill both FMA ports), turning the degree-4 poly tie into a consistent win over gcc's own
  256-bit autovec. The interpreter marshals through the identical kernel, so the oracle stays exact.
- **`@parallel` reduction → multicore reduction kernel**: a reduction loop in a `@parallel` function
  (`s += x[k]*y[k]`, `(x[k]-y[k])²`, or `x[k]`) lowers to a **deterministic multicore reduction
  kernel** (`wukong_sreduce_f32_parallel`: dot/ssd/sum/sumsq) instead of a sequential per-thread
  accumulation. The parallel result is bit-identical to the serial one regardless of core count
  (fixed-size chunks, ascending partial combine), and the interpreter calls the serial form, so the
  differential oracle stays exact. Spreads the stream across cores to aggregate memory bandwidth:
  **dot ~7.9×, ssd ~8.6× faster** than single-threaded C (which stays serial & latency-bound).
- **`@parallel`**: loops execute across CPU cores via a rayon runtime, each per-core chunk itself
  vectorized — ~2.2–8× faster than idiomatic single-threaded C on the (memory-bound) elementwise and
  reduction kernels.
- **Mixed-precision (bf16 *and* f16) → SIMD dispatch**: both `bf16` and `f16` are real 2-byte storage
  (round-to-nearest-even on store/cast, f32 compute; bf16 via inline bit-math, f16 via shared
  `half`-crate shims since Cranelift x64 lacks f16 convert lowering). A full, symmetric op suite over
  `[bf16]`/`[f16]` arrays read through an `as f32` widening cast (lossless: `<<16` for bf16, F16C
  `vcvtph2ps` for f16) with f32 accumulate/compute dispatches to half-precision runtime kernels:
  **`dot`/`sum`** (`wukong_{dot,sum}_{bf16,f16}`, ~3–8× vs C), **`max`/`min`/`absmax`**
  (`wukong_reduce_{bf16,f16}` — the per-tensor absmax is the symmetric int8-quant scale), **streaming
  `axpby`** (`wukong_axpby_{bf16,f16}`, half-in/f32-out), and the **36-op activation set**
  (`wukong_vmath_{bf16,f16}`). Precision-generic recognizers; the interpreter marshals through the
  identical kernel, so native == interp bit-for-bit. C/Rust can vectorize neither a `libm` call nor the
  half→f32 widen, so the gap is structural.
- **Arrays**: fixed-size `[T; N]` run end to end — literal/repeat initializers, indexed load/store
  with a runtime index, and array parameters passed by base pointer (out-params work). Real kernels
  (dot product, SAXPY, a flat GEMM) run on the interpreter.
- **Tuples, structs, pointers, `loop`, and constant-shape tensors execute**: tuples (`(a, b)`, field
  access/assign `t.0`, heterogeneous padded fields), structs (`struct S { … }`, literals with fields
  in any order, field access/assign), **nested structs** (struct-in-struct to any depth, an aggregate
  field deep-copied from a variable, arrays of structs, tuple-of-struct), pointers/references
  (`&mut x`, `*p` load/store, a pointer threaded through a call — address-taken locals stay in memory,
  so `-O0` == `-O3`), and `loop { … }` with `break`/`continue` all run end-to-end on both the
  interpreter and the native Cranelift backend. Aggregates lower to a flat padded byte buffer with no
  dedicated aggregate MIR type (the local's value *is* its base pointer, like an array; nested fields
  recurse, a non-literal aggregate field is a leaf-precise deep copy). **Constant-shape tensors** also
  run: a `Tensor[f32, R, C]` parameter passes by base pointer and a multi-dimensional index `a[i, j]`
  flattens to a row-major GEP — the shape-typed surface executing, not just shape-checking. A matmul
  written in that tensor notation (`c[i,j] = Σ a[i,k]·b[k,j]`, both the dot-product and accumulate
  spellings, plus the `b[j,k]` `nn.Linear` `A·Bᵀ` form) **dispatches to the same tuned `wukong_sgemm`
  microkernel** as the flat `a[i*K+k]` spelling — a 2-index operand access supplies its row stride
  from the tensor's inner dimension (gated by `tensor_matmul_is_correct`/`tensor_matmul_accumulate_form`).
  Fixtures `tests/run/{tuple,struct,struct_nested,pointer,loop,tensor_add,tensor_matmul}.wk`; the
  aggregate path is differentially gated by `differential_{tuple,struct,nested_struct}` and pointers by
  `differential_pointer` (native vs interpreter, bit-for-bit). By-value aggregate parameters/returns
  (an sret ABI) now lower too (see the language-surface additions below); **symbolic-generic tensor
  dimensions now execute too** (`fn f<M, N>(t: Tensor[f32, M, N])`, via hidden dim params — see below).
- **Intrinsics**: `print`/`println` (captured stdout) and `assert` (traps on false).
- **Runtime**: a bump `Arena` allocator and a deterministic `parallel_for` — both *reference* surface
  with no caller in the workspace (kernels use thread-local `Vec` scratch, and the native `@parallel`
  lowering target is `wukong_parallel_for`). The runtime's real surface is the dispatched kernel family
  (GEMM/GEMV, vmath, reductions, norms, …). `Arena::alloc` rejects a non-power-of-two alignment rather
  than aliasing a live region.
- **Diagnostics**: rustc-style renderer, a stable error-code catalog with `--explain <CODE>`, and
  `--error-format=json` (JSON Lines).
- **Tooling & tests**: end-to-end run-suite with `// EXPECT-*` directives, an opt-level differential
  test (`-O0` vs `-O1/-O2/-O3`), per-stage `--emit` smoke tests, the `wukong_bench` harness (IR-op
  reduction + `-O0`-vs-`-O3` interpreter speedup, doubling as an optimizer-equivalence gate over
  heavy kernels in `bench/kernels`), a GitHub Actions CI (fmt + clippy + test on Linux & Windows),
  and the language guide and internals docs.
- **Performance**: the interpreter pools per-call register files and passes block-parameter arguments
  through a reused buffer, roughly halving its wall-clock; large `[v; n]` array initializers lower to
  a fill loop instead of unrolled stores.
- **Embedding-lookup dispatch**: the LLM token-id row gather `out[t,:] = weight[ids[t],:]` (over an
  `i32` index array — the first recognized dispatch with an integer *index input*) is recognized and
  lowered to `wukong_embedding_f32[_parallel]` (a 256-bit row copy; bit-exact data movement, mapped
  across the independent output rows under `@parallel` — since 2026-07-29 the parallel kernel is
  selected under a **run-time** byte-range non-overlap test on the two buffers, so an aliased call runs
  the serial kernel, and a slice weight with no compile-time extent keeps it outright).
- **2D pooling dispatch**: the idiomatic 5-deep max/avg-pool nest lowers to
  `wukong_{max,avg}pool2d_f32[_parallel]` (the CNN spatial downsampler; `@parallel` across channels;
  bit-exact — max is idempotent, the avg `(dy,dx)` sum order is fixed). Honest sharp edge: gcc
  auto-vectorizes regular-stride (e.g. 2×2/s2) pooling, so single-core is a tie there, not a win.
  **Superseded (2026-07-29):** dispatch additionally requires all eight dims to be literal and the
  nest's own `OH`/`OW` bounds to equal the full no-padding extent the kernel recomputes; a
  partial-window / sub-region pool declines to the (correct, unaccelerated) scalar nest.
- **xbench**: a broadcast bias-add (`out[r,c] = x[r,c] + bias[c]`) cross-language row, plus a
  **fused FFN** row (`C = silu(A·Bᵀ)` — the real Dense/SwiGLU layer, matmul + activation folded into
  one `wukong_sgemm_nt_epi` C-write; **~24–26× single-core, ~48–95× `@parallel`** vs C, where C pays
  an un-tiled serial-reduction GEMM + a separate scalar-`expf` silu pass) and an **`argmax@parallel`**
  row (**~17× vs single-threaded C** — the multicore global argmax).
- **`match` expressions** (`tests/run/{match_expr,match_patterns,match_tuple}.wk`): integer/bool
  literal, identifier-binding, and wildcard `_` patterns, **or-patterns** `1 | 2 | 3`, half-open
  `0..10` / inclusive `0..=10` **range** patterns, **enum-variant** patterns `Color::Red` (matched by
  discriminant), and **tuple** patterns `(0, _)` (per-field tests + bindings, nesting, and composition
  like `(0 | 1, y)`), each with an optional `if` guard. The scrutinee is evaluated once and the whole
  `match` lowers to an if-else chain; it works in value and statement position. Gated by
  `differential_match`/`differential_match_patterns`/`differential_tuple_match` (native==interp, -O0..3).
- **C-style enums** (`tests/run/enum_cstyle.wk`): `enum Code { Ok = 10, Err = 20 }`, auto-incrementing
  `enum Color { Red, Green, Blue }` (0,1,2), and continue-after-explicit `enum Step { A = 5, B, C }`
  (5,6,7). A variant *is* its integer discriminant — usable in `let`, `==`, `as i32`, and as a `match`
  pattern. **Data-carrying (tagged-union) variants with tuple/struct payloads and payload `match`**
  (with bindings, `if` guards, and literal sub-patterns) **now run too**
  (`tests/run/enum_payload_{tuple,struct}.wk`). Gated by `differential_enum`.
- **Top-level `const` usable as a value** (`tests/run/top_level_const.wk`): sema type-checks each
  initializer against its annotation (an unsuffixed literal adapts) and records it; mir_build inlines
  it at every use site — a bare value, in arithmetic, as an array index, as a loop bound, and when one
  `const` references another (recursive inlining). Gated by `differential_top_level_const`.
- **Tuple destructuring in `let`** (`tests/run/let_destructure.wk`): `let (a, b) = …`, nested
  `let ((m, n), o) = …`, a wildcard `let (keep, _) = …`, and destructuring a tuple-returning call
  result — each sub-pattern binds a view into the initialized tuple buffer (value semantics). Gated by
  `differential_let_destructure`.
- **Nested tuple-field access** (`tests/run/nested_tuple_field.wk`): `t.0.1`, `t.0.0.0`, as reads and
  as assignment targets — the parser splits a lexer-glued `N.M` float in field position into two
  consecutive tuple-field accesses. Gated by `differential_nested_tuple_field`.
- **By-value aggregate parameters and returns** (`tests/run/{struct_fn,struct_return}.wk`): a
  tuple/struct passed by value and a `fn … -> Struct` return are modeled with a MIR-level **sret** ABI
  (a hidden leading pointer, a deep-copy into it, a void return; the call site allocates the
  destination and passes it as the hidden first argument), so no aggregate ever rides in a register and
  both backends agree. A struct param resolves to its registry-aware byte-buffer type (passed by base
  pointer), and `p.x` through a `&`/`*mut` reference auto-derefs. Whole-aggregate **assignment**
  (`s = other;`, `*p = Struct{..}`) now deep-copies leaf by leaf (`tests/run/struct_assign.wk`). Gated
  by `differential_struct_across_fns`/`differential_struct_return`/`differential_struct_assign`.
- **Radix integer literals** (`tests/run/radix_literals.wk`): hex `0xFF`, octal `0o17`, binary
  `0b1010` — with `_` digit separators and an optional type suffix — parse to their real value (they
  previously all evaluated to `0`, since `parse_int` kept only the leading run of decimal digits). Gated
  by `differential_radix_literals`.
- **Char literals** (`tests/run/char_literals.wk`): `'A'` lowers to its `u32` Unicode scalar value,
  with the one-character escapes (`\n` `\t` `\\` `\'` `\0`), `\xHH` hex, and `\u{…}` Unicode escapes
  (the lexer's escape scanner now consumes the multi-byte forms). Gated by `differential_char_literals`.
- **String literals** (`tests/run/string_literal.wk`): `"hello"` materializes its UTF-8 bytes plus a
  NUL terminator into a stack byte buffer (the same by-pointer convention as arrays), typed `*u8`; the
  escapes `\n` `\r` `\t` `\\` `\"` `\'` `\0` `\xHH` `\u{…}` decode, each code point re-encoded as UTF-8.
  `print`/`println` of a `*u8` (a literal, a `let`-bound string, or one returned from a function) is
  routed (type-directed) to a `print_str`/`println_str` path that renders the bytes — the interpreter
  walks its slot memory, native reads the buffer via `rt_print_str` — while numeric `print` still
  prints numbers. No string *type* beyond `*u8` yet (no concatenation/indexing/length, no general
  static-data section). Gated by `differential_string_literals`.
- **Labeled loops** (`tests/run/labeled_loop.wk`): `'label: loop/while/for { … break 'label;
  continue 'label; }` — a labeled `break`/`continue` targets the named enclosing loop, not just the
  innermost. The lexer has a `Label` token disambiguated from a char literal exactly as Rust (`'a'`
  closed by a `'` is a char; `'outer:` is a label); sema tracks an enclosing-label stack, so a
  `break`/`continue` outside any loop **or** one naming an undeclared label is `E0303`
  (`tests/fail/break_unknown_label.wk`); mir_build's loop stack carries each loop's label and resolves
  the branch target. Loop-as-expression / break-with-value (`let x = loop { break 5; };`) **now works
  too**: `loop` is a value-producing expression and `break <v>` carries a value, merged on a typed
  exit-block param like `if`/`match` (`tests/run/loop_break_value.wk`). Gated by
  `differential_labeled_loops`.
- **Slices `[]T`** (`tests/run/slice_basics.wk`): a `[]T` fat pointer `{ data: *T @ 0, len: i64 @ 8 }`
  viewing existing array storage — `s.len()`, indexed read/write `s[i]`, iteration `for x in s`, passing
  to a function (including an array **unsized** to a slice at the call site), and write-through aliasing
  of the backing array. A 16-byte by-pointer aggregate addressed identically by both backends
  (interpreter slot memory == native byte offsets), so interp == native bit-for-bit and `-O0` == `-O3`.
- **Symbolic-generic tensor shapes** (`tests/run/generic_shape*.wk`, `matmul_dynamic.wk`): a
  `fn f<M, N>(t: Tensor[f32, M, N])` runs via hidden per-dimension `i64` params bound under their symbol
  names (stride / matmul-dim / dim-as-value lookups resolve through them), so the shape-typed surface
  executes on **runtime** dimensions with zero backend change; an undeclared dimension is `E0504`.
- **Autodiff reachable from the CLI**: `wukong_autodiff` (a reverse-mode VJP MIR→MIR transform plus a
  fused AdamW kernel, finite-difference-gated) is now wired into `wukongc` — `--emit=grad` prints a
  loss function's backward MIR (`--grad-of=<fn>`, `--grad-wrt=<i,..>`), and `--train` runs a
  forward→backward→optimizer loop (`--train-steps`, `--train-lr`, `--train-opt=sgd|adamw`,
  `--train-seed`) printing the loss trajectory (driver `build_grad` / `run_train`).
- **Correctness fixes** (interpreter↔native divergences and ICEs removed; each gated bit-for-bit):
  - **`break`/`continue` outside any loop** is now a clean **`E0303`** instead of a backend divergence
    — the lowerer left a fallback `unreachable` the interpreter trapped (exit 1) but the native backend
    turned into a SIGILL (exit 132). Registered in the diagnostic catalog (`--explain E0303`);
    `tests/fail/break_outside_loop.wk`.
  - **`&&` / `||` now short-circuit** — they lowered to a bitwise `And`/`Or` of both eagerly-evaluated
    operands (so a side-effecting or guarded RHS always ran, e.g. `n != 0 && 100/n > 0` divided by
    zero); now lowered to control flow. Gated by `differential_short_circuit`.
  - **`continue` in a range `for`** now runs the loop step (it had branched straight back to the header
    without advancing the index — an infinite loop); a dedicated latch block holds the step, on both the
    sequential and `@parallel` per-thread paths. `tests/run/for_continue.wk` (`differential_for_continue`).
  - **float → narrow-int casts** (`1e30 as i8`) no longer ICE the native backend and now **saturate**:
    Cranelift's `fcvt_to_*_sat` can't target a sub-32-bit result, so the backend converts to `i32`
    saturating then clamps/reduces to the narrow range — Rust `as` semantics, matching the oracle.
    `tests/run/float_cast_narrow.wk`.
  - **Deep recursion** in the interpreter no longer aborts the process: `run_with_output` runs on a
    scoped 512 MiB-stack worker thread, so a deeply recursive program returns instead of overflowing the
    ~8 MiB main stack (which had taken the whole differential gate down). Gated by
    `deep_recursion_does_not_overflow_oracle`. **Superseded (2026-07-29):** all *five* public entry
    points run on that worker (four previously skipped it), and *unbounded* recursion is now a
    diagnostic — `interpreter call-depth limit exceeded (likely unbounded recursion)`, exit 1 — rather
    than an eventual overflow of the 512 MiB reservation.
  - **int → f32 casts above 2⁵³** round in one step in the interpreter oracle — it had double-rounded
    `int → f64 → f32`, disagreeing with native's single `fcvt_from_*` by a full ULP. Gated by
    `differential_int_to_f32_rounding`.
  - **Math intrinsics on integer operands** no longer emit float-op-on-int MIR: `abs`/`round`/`floor`/
    `ceil`/`trunc` are now type-preserving (integer `abs` = `select(x<0, −x, x)`, rounding an integer is
    the identity) and `sqrt`/the transcendentals promote an integer operand to `f32` — native had
    rejected the old MIR while the interpreter ran it lossily. `tests/run/int_math.wk`
    (`differential_int_math_intrinsics`).

### Changed
- **Cast precedence fixed**: `*p as T` now parses as `(*p) as T`, not `*(p as T)` (which had
  mis-typed the deref as a `ptrtoint` then a load). `as` binds looser than `*`/unary, tighter than the
  binary operators.
- A construct lowering cannot handle (e.g. explicit SIMD `f32x8` load/store intrinsics) is a hard
  `error[C0001]` instead of a warning, and the driver refuses to optimize, run, or codegen a module
  whose lowering failed — so the compiler never emits or executes invalid MIR.
- **Global argmax/argmin** (`wukong_argreduce_f32`) gained a 256-bit AVX2 path (4 accumulators × 8
  `f32` value + `i32` index lanes, strict-compare + `blendv`, collapsed through the scalar tie-break).
  It was the one memory-bound reduction lacking one, so it had *lost* to gcc's branch-predicted scalar
  loop by 1.30× — now **~5–9× faster**, bit-identical to the scalar lowest-index result.
- **Column argmax/argmin** (`wukong_colarg{max,min}_i32`) rewritten to a single row-major pass with
  the column range's running best kept L1-resident, instead of re-reading the matrix `cols/8` times
  per 8-column band — a former 0.9–1.6× tie/loss became **2.7–5.3× faster** (bit-exactness unchanged).
- **Fused elementwise chains**: the vectorizer now forwards a just-stored intermediate's register value
  to its consumer in the same fused body, dropping the per-element store+reload round-trip (one fewer
  memory stream per chained op; the intermediate need not touch memory between producer and consumer).
- **`@parallel` global argmax/argmin** now dispatches to the multicore `wukong_argreduce_f32_parallel`.
  It previously fell through to the *serial* kernel (the parallel reduction recognizer handles only
  plain `+`/`fmax`/`fmin` reductions, not the argmax `(value, index)` bookkeeping), so global argmax
  never scaled across cores. The parallel kernel folds the same fixed `RCHUNK` decomposition in
  ascending order, so the index is bit-identical to the serial kernel the interpreter calls — **~17×
  vs single-threaded C** (the bandwidth-limited ~2× scaling over the 8.9× single-core fold).
- **Optimizer compile time ~31% faster** (in-process, 400-function `-O2`: 18.0 ms → 12.5 ms), from two
  output-preserving changes — the MIR is bit-identical to before, verified by `optimization_preserves_
  results`, the native-vs-interpreter differential gate, and the `-O0`-vs-`-O{1,2,3}` invariance gate:
  - **CSE value-numbering key** is now a packed, allocation-free `enum` instead of a `format!` string
    built per pure instruction on every pass (CSE is the costliest pass; the key induces the identical
    equality relation, so value numbering is unchanged).
  - **Fixpoint loop** skips passes already at fixpoint: a pass that ran with no change is not re-run
    until another pass mutates the function, which drops the optimizer's final all-passes no-op
    *confirmation* sweep without changing the sequence of mutations.

### Notes
- **Constant-shape** *and* **symbolic-generic** tensors, **C-style `enum`s** (including **data-carrying
  tagged-union** variants), **by-value aggregate parameters/returns**, **slices `[]T`**, and
  **loop-as-value** (`break <v>`) now all execute end-to-end (above). What still only parses and
  type/shape-checks without lowering is **SIMD vector *values*** (explicit `f32x8` load/store
  intrinsics). Native code generation is **Cranelift** (`--emit=obj|exe`, no LLVM); the LLVM path is
  the textual `--emit=llvm-ir` emitter only (see `docs/llvm-setup.md`).
