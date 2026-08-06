# Wukong Roadmap & Known Limitations

Wukong is built openly and incrementally. This page is an honest snapshot of what works, what is
checked-but-not-executed, and what is planned — so expectations match reality.

## Works end to end (interpreter `--run`, **and native code** `--backend=native`)

Two CPU execution backends now run the full language and agree bit-for-bit (a differential gate proves
it across opt levels): the zero-dependency tree-walking interpreter (the reference oracle) and a
from-scratch **Cranelift native backend** (JIT for `--run --backend=native`, object/exe via
`--emit=obj|exe`) — **no LLVM toolchain required**. See `BENCHMARKS.md` for cross-language numbers.

- **Multi-file modules**: `import a.b` loads `a/b.wk` (resolved against the *root* source file's
  directory) and splices its items into one merged flat namespace, so cross-file fns/consts/structs
  just work on both backends (`tests/run/import_multi.wk`). Files load once each — an import cycle
  or diamond dedups by canonical path, never errors. An unresolvable import is `E0305`; a cross-file
  duplicate name is the ordinary `E0300`, pointing at both definitions with each file's own
  path/line. (`import … as …` / `import x.{a,b}` parse but don't rename/restrict yet;
  `--emit=tokens|ast` stay root-file-only by design.)
- Functions (including recursion and mutual recursion — the interpreter oracle runs on a
  512 MiB worker thread, so ordinary recursion does not overflow the host's small default stack), and
  direct calls. *Caveat:* the tree-walking interpreter's call frames are ~an order of magnitude larger
  than the native backend's machine frames, so it recurses less deeply than native. Its ceiling is an
  explicit, measured cap rather than whatever the stack happens to allow: past `MAX_CALL_DEPTH` —
  **300,000** nested Wukong calls in a release build — the largest value that rejects no depth measured
  to complete on the 512 MiB worker — and **40,000** in a debug build, deliberately below that build's
  measured ceiling but above the depth the `differential_deep_recursion` gate exercises, so the oracle
  never refuses a depth that gate needs — the interpreter reports
  `error: interpreter call-depth limit exceeded (likely unbounded recursion)` and exits 1. That is a
  **diagnostic**, never an uncatchable process abort, so an unbounded recursion can no longer take the
  differential oracle down with it. Recursion past the cap is still a **resource limit outside the
  bit-for-bit differential contract** — the oracle *refuses* where the native backend still completes,
  the same "outside the defined contract" status as an out-of-bounds access. Raising the cap past its
  measured ceiling would reintroduce the abort; lowering it would reject working programs.
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
  construction. `f64` is full. (The GPU adds tensor-core fp16/bf16/fp8 GEMM on top.)
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
  (per-field tests + bindings, nesting and composition like `(0 | 1, y)`), each with an optional `if`
  guard. The scrutinee is evaluated once and the whole `match` lowers to an if-else chain
  (`tests/run/{match_expr,match_patterns,match_tuple}.wk`).
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
  (E0402); a non-exhaustive `match` over the variants is E0405.
- **Slices `[]T`** — a fat-pointer view `{ data: *T @ 0, len: i64 @ 8 }` over existing array storage:
  length `s.len()`, indexed read/write `s[i]`, iteration `for x in s`, and passing to a function —
  either an already-materialized slice or a fixed-size array *unsized* to a slice parameter at the
  call site (`let s: []T = arr` / `f(arr)`). A write through the slice aliases the backing array. The
  16-byte fat pointer is a by-pointer aggregate the two backends address identically, so interp ==
  native bit-for-bit and -O0 == -O3 (`tests/run/slice_basics.wk`). The element types must match on
  the unsizing coercion, and a scalar passed where a slice is expected is E0401; slice indexing is a
  runtime-length view (unchecked, like a pointer — a slice has no static length to bounds-check).
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
  (non-integer count / freeing a non-slice E0401, arity E0503 — `tests/fail/heap_*.wk`);
  double-free, freeing a non-alloc slice, or use-after-free is **undefined on native** and
  mark-and-forget (harmless, never a crash) in the interpreter. interp == native bit-for-bit and
  -O0 == -O3 (`tests/run/heap_{alloc,alloc_fn,zero_init,len}.wk`, `differential_heap_alloc`, and
  `heap_alloc.wk` in the linked-exe AOT gate — the rustc link resolves the two runtime symbols
  from the `wukong_runtime` rlib). Loops over alloc'd slices reach the **recognized-kernel** path:
  every kernel buffer operand resolves through `kernel_base_ptr`, which loads a `[]T`'s data pointer
  out of its 16-byte fat pointer, so `for i in 0..n { y[i] = exp(x[i]); }` over two `alloc_f32` slices
  emits `wukong_vmath_f32`, and velem / GEMM / reduction nests over slices dispatch too. What still
  declines on a slice base is the *general* auto-vectorizer — the same loop written over slices emits
  no vector ops where the fixed-array form does (a future perf lever, correctness unaffected).
- **Typed file I/O** — `read_<T>(path, buf) -> i64` and `write_<T>(path, buf) -> i64` over the dtypes
  {`f32`, `f64`, `i32`, `i64`, `i8`, `u8`}, where `path` is a `*u8` string literal and `buf` a `[]T`
  slice (typically from `alloc_<T>`). The on-disk format is frozen and **headerless: raw contiguous
  little-endian elements**, no magic and no length prefix, encoded with `to_le_bytes`/`from_le_bytes`
  regardless of host endianness, so a blob written by one backend is byte-identical to the other's.
  Contract: the open is attempted **first**, so a missing or unopenable path is `-1` even for
  `len <= 0`; a read then transfers `n = min(len, file_size / sizeof)` elements (a trailing partial
  element is ignored, `buf[n..]` left as allocated) and returns `n`, or `-2` on a mid-read I/O error;
  a write creates/truncates and returns `len` (`-1` create failed, `-2` write error). Lowered to
  opaque `wukong_rt_{read,write}_<T>` runtime calls — **no new MIR op** — and the interpreter
  implements the identical contract, so interp == native
  (`tests/run/io_roundtrip_{f32,f64,i32,i64,i8,u8}.wk`, `io_missing_file.wk`, `io_partial_read.wk`).
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
  (`tests/run/string_literal.wk`). No string *type* beyond `*u8` yet — no concatenation/indexing/
  length (🟡).
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
- **SIMD auto-vectorization**: straight-line elementwise loops (incl. branchy ones via
  if-conversion) lower to 128-bit vector ops, 4×-unrolled, with a scalar remainder — automatically,
  on the native backend (only a **half-open, unit-step** range vectorizes: an inclusive `a..=b` or a
  `step s` loop lowers as the plain scalar loop — correct, just unaccelerated — and the
  recognized-kernel cascade declines those two forms for the same reason). Where the body fits the
  raw-AVX2 recipe (f32 lanes, register pressure within the 16 YMM registers) an **elementwise** loop
  is instead emitted as **true 256-bit AVX2** by a raw machine-code path
  (`crates/wukong_codegen_cranelift/src/avx2.rs`, VEX-encoded via `iced-x86`) that sidesteps
  Cranelift's 128-bit CLIF cap (`WUKONG_P4_NO_256` disables it) — with **no** trip-count requirement,
  so a 32-element loop and a runtime-`n` loop both take it. Only a float **reduction** additionally
  requires a *compile-time-known* trip ≥ 2048 (`VEC256_REDUCTION_MIN_TRIP`), below which the inlined
  128-bit reduction wins. The raw-AVX2 emitter is **host-gated**: `host_supports_kernels()` requires
  **x86-64 Windows with AVX2 + FMA** (the emitted body hardcodes the Win64 rcx/rdx/r8 argument
  registers, and there is no in-kernel fallback), and `assemble_kernel` refuses otherwise. The refusal
  is *not* a fallback: `mir_build` attaches the kernel recipe with no host check, so on any other host
  native codegen fails the compile with that error rather than silently reverting to the 128-bit CLIF
  form (`WUKONG_P4_NO_256=1` is the knob that actually keeps the 128-bit form).
  saxpy/poly/relu/relu6 vectorize.
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
  scale applied in the GEMM's own writeback. Whole-function and statement forms both fire, and the
  integer accumulate is exact mod 2^32, so interp == native bit-for-bit and `-O0` == `-O3`
  (`tests/run/{i8_linear,i8_linear_dequant,i8_linear_parallel,dequant_perchan}.wk`). The
  per-tensor/per-channel `absmax` that produces the symmetric quantization scale is the
  column-reduction family above.
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
- Intrinsics `print`/`println`/`assert`.
- The optimizer (`-O0..-O3`), backed by CFG and dominator analyses: whole-program **inlining**
  bottom-up over the call graph (non-recursive callees before their callers, scored by a cost model),
  **mem2reg** (alloca → SSA), constant folding, algebraic simplification, CFG cleanup
  with block merging, dead/trivial block-parameter elimination, DCE, dominator-tree CSE with load
  forwarding, DSE, **loop-invariant code motion**, and **partial loop unrolling** (4×, with a
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

## GPU backend (NVIDIA RTX 4050, behind `--features gpu`)

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
(the `tensor_1d_kernels@O3` `CUDA_ERROR_ILLEGAL_ADDRESS`). Corpus standing is printed by the gates
themselves, over every fixture in `tests/run` (333 today): `lower::tests::run_corpus_matches_interp_oracle`
sweeps each program at `-O0` and `-O3`, requires zero mismatches and zero device faults and non-zero
coverage, and reports the rest as honest `UNSUPPORTED:` skips; `megakernel::tests::mega_corpus_matches_oracle`
does the same over the megakernel-eligible subset, counting (program, opt-level) configs and treating a
launch-time `Ok(None)` decline as neither coverage nor a miscompile. Re-run them for the current
numbers — they are a function of the corpus, not a fixed figure.
Both gates now also **isolate device faults**: a genuine `ILLEGAL_ADDRESS` poisons the CUDA state
**process-fatally** — measured on this driver (RTX 4050, Windows/WDDM), `cuDevicePrimaryCtxReset`
returns Ok but re-retaining the primary context still returns error 700, and cudarc exposes no
non-primary `cuCtxCreate`, so in-process recovery is impossible. `crate::gpu::reset_gpu` therefore
degrades to marking the device **lost**; the gates record the root fault on a loud ledger and
report every later program as NOT RUN (never as passed, never as spuriously failed) — one faulting
program can no longer cascade into ~100 false failures across both gates.

## Automatic differentiation (`wukong_autodiff`)

Reverse-mode autodiff runs as a **MIR→MIR transform**: given a forward function computing a scalar
loss, it emits a new function that also accumulates the gradient w.r.t. designated input buffers (the
vector-Jacobian product). Matmul adjoints ride the same tuned GEMM kernels, and a fused AdamW step is
emitted as one kernel. Every VJP rule is **finite-difference-gated** (forward + backward run in f64)
against a closed-form reference. It is reachable from `wukongc`: `--emit=grad` dumps the backward MIR
of a loss function, and `--train` runs a fwd→bwd→optimizer loop (`--grad-of`/`--grad-wrt` select the
loss and parameters; `--train-opt=sgd|adamw` the optimizer). The envelope is deliberately narrow and
every hole outside it is a **loud refusal, never a zeroed gradient**. The forward function must be a
single basic block in SSA terminated by `ret <float scalar>`. Five kernel families have rules —
`wukong_sreduce_f32` (SUM / DOT / SSD, including the self-dot `Σ x²`), `wukong_sgemm_nt`,
`wukong_vmath_f32` (relu/sigmoid/tanh/exp differentiated as a synthesized loop; silu/gelu/elu/softplus
as one fused `wukong_vmath2_f32` backward call), `wukong_velem_f32` (**identity-affine only** — the
Hadamard `x⊙y` and division compute modes are refused by name), and `wukong_norm_f32` (softmax /
LayerNorm / RMSNorm) — each with its `@parallel` twin. A norm with learned γ/β
(`wukong_norm_affine_f32`), a mid-function `@parallel` region, an autovectorized `VecKernelCall`, and
any buffer that would need accumulation from two non-matmul contributions all decline with a message
naming the construct. One known gap is documented rather than fixed: when a loss reads a kernel-written
buffer with a scalar load instead of through another recognized kernel, that kernel's output adjoint is
never seeded. Both `--emit=grad` and `--train` force `-O1` or higher, since the transform needs
single-block SSA (mem2reg + simplify-cfg).

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
  no trip-count requirement for an elementwise body; only a float *reduction* needs a
  compile-time-known trip ≥ 2048; the recognized kernels get 256-bit AVX2 via the
  runtime microkernels.)
- **Attributes** other than `@parallel`: **parsed but not validated and not consumed.**
  `@simd`/`@tile`/`@align`/`@extern`/`@export` are shaped by the parser and then ignored — there is no
  name check, so an unknown or misspelled attribute (`@bogus(nonsense = 3)`, `@align(3)`) compiles
  clean with no diagnostic, and no crate reads them (`wukong_sema` never inspects `.attrs`). The single
  consumer in the tree is `mir_build`'s `has_parallel_attr`, a name test for `"parallel"`; even
  `@parallel`'s own arguments (e.g. `grain = …`) are parsed and discarded. `E0207` fires only for a
  malformed argument such as a missing value after `=`. (`@parallel` itself executes — see above.)
- **Non-contiguous tensor layouts** `.col_major` / `.strided` / `.tiled(N, M)`: they parse (an unknown
  layout name is `E0204`), they are part of the tensor type, and they are enforced across a call
  boundary — passing a `.col_major` tensor to a `Contiguous`-typed parameter is `E0502`. But **indexing
  one cannot be lowered**: `a[i, j]` on any non-contiguous or symbolically-strided tensor is a hard
  `C0001`. Only the default row-major `.contiguous` layout executes.

## Planned

- Fusing arbitrary elementwise chains *under* `@parallel`. (The **parallel fused-epilogue GEMM** that
  was the remaining item here has since shipped — a `@parallel` `act(A·Bᵀ [+ bias])` nest dispatches
  to the multicore `wukong_sgemm_nt_epi_parallel`; see "Works end to end" above.)
- Execution of explicit `f32x8`-typed *values* (they parse and type-check today — see "Checked but
  not yet executed" above). The *general loop* vectorizer already reaches true 256-bit AVX2 for large
  trips via a raw machine-code emitter (see "Works end to end" and "Known limitations"); it is
  *source-level* `f32x8` values that still do not execute.
- A minimal stdlib of reusable functions/collections. (The language surface once grouped here —
  structs, enums including data-carrying tagged unions, slices, and multi-dimensional indexing
  `a[i, j]` — now runs end to end; see "Works end to end" above.)
- AMDGPU/ROCm device codegen (the NVIDIA PTX path already ships behind `--features gpu`, and
  reverse-mode autodiff already ships as the `wukong_autodiff` crate — both above).

## Known limitations / sharp edges

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
  `cranelift_still_rejects_f32x8`/`p4_probe_vec256_ops` tripwire tests). An elementwise f32 loop is
  instead dispatched to a **raw-AVX2 256-bit machine-code emitter** at any trip count; a float
  **reduction** takes it only at a **large, compile-time-known trip (~≥2048 elems)**
  (`crates/wukong_codegen_cranelift/src/avx2.rs`, VEX-encoded via `iced-x86`;
  `WUKONG_P4_NO_256` disables it) — the same way the GEMM/vmath runtime microkernels reach 256-bit,
  and precisely *why* that raw emitter exists (Cranelift can't legalize the wider lane). That path is
  **platform-gated**: `avx2::host_supports_kernels()` requires x86-64 **Windows** with runtime
  AVX2 + FMA3, because the Win64 argument registers (rcx/rdx/r8) are literal in the emitted bytes and
  there is no in-kernel dispatch; on any other host `assemble_kernel` refuses and the compile reports a
  diagnostic rather than emitting non-executable code. `WUKONG_P4_NO_256=1` forces the 128-bit CLIF
  path and is **result-identical** — the 256-bit recipe contracts `x + y*z` into one FMA exactly like
  its scalar tail, so the knob changes instruction selection and never the answer (pinned by
  `p4_kill_switch_is_result_identical`). Below that
  threshold an out-of-line 256-bit reduction call would lose to the inlined 128-bit path, so small or
  runtime-unknown *reduction* trips stay 128-bit + 4× unrolling; there compute-bound kernels use 2×
  the FMA ports they could, but the vectorized **transcendentals still beat scalar `libm` ~2.5–3×**.
  The AST loop vectorizer assumes distinct array parameters do not alias. **That assumption is
  informal and unchecked, and it is not the language's rule** — nothing rejects `f(a, a)`, and there
  is no `restrict`/`&mut` annotation to carry the promise. The MIR alias analysis
  (`wukong_opt::alias`, see `docs/internals.md`) deliberately does *not* adopt it: `may_alias` answers
  "may alias" for two distinct pointer parameters, and `tests/run/alias_slice_params.wk` pins a
  program whose answer depends on it. Making the promise real is a **language** decision — an opt-in
  parameter annotation, or a rule that a `mut` aggregate parameter may not alias another parameter —
  not something an analysis can derive. Until then the analysis exploits only what *is* guaranteed:
  distinct stack slots, a local slot versus a parameter, non-escaping slots versus everything, and
  disjoint constant offsets.
  An attempt to turn the assumption into an actual wrong answer did **not** succeed, and the negative
  result is recorded so the next person does not repeat it: `fn f(src: []f32, mut dst: []f32)` doing
  `dst[i] = src[i-1] + 1.0` — a cascade when the caller passes one buffer twice, and a different
  answer if lanes are widened — returns the correct scalar values at `-O0`/`-O2` on both backends,
  because the AST vectorizer **declines slice-parameter loops outright** (`--emit=mir -O2` shows no
  `<N x f32>` for either that loop or the safe same-index `dst[i] = src[i] * 2.0`, with a constant
  4096 trip count). So the assumption is currently unreachable through `[]T` parameters rather than
  proven harmless — it is *not* evidence that adopting a no-alias rule would be safe, and a fixed-size
  array parameter or a future MIR vectorizer may well reach it.
- Array *length* in a type may be an integer literal or a top-level `const` (resolved through
  const-to-const chains; `tests/run/const_array_length.wk`). It may also be **arithmetic over those** —
  `[i32; 2 + 2]`, or a `const N: i32 = 2 + 2` used as a length — because sema's `eval_usize` and
  mir_build's `const_usize_expr` fold through the one shared `wukong_ast::BinOp::fold_const_len`, so
  the allocated slot and the compile-time bounds check cannot disagree (`a[9]` on such a length-4 array
  is still `E0501`). Radix prefixes, `_` separators and integer suffixes all decode (`[i32; 0x10]` is
  16). The ceiling is `MAX_CONST_ARRAY_LEN = u32::MAX`: a longer literal length is `E0401`, and the
  shared folder declines (yielding the invalid-length sentinel 0) whenever an operand or the result
  exceeds it or the arithmetic would wrap — deliberately, because sema folds in `u64` while mir_build
  narrows the slot to `u32`. A length naming something that is neither a `const` nor a declared generic
  is `E0301` rather than a silent length 0. A **symbolic** length (a generic `N`) still has no
  compile-time value: an array *parameter* `[i32; N]` is just a base pointer and runs, but a *local*
  `let a: [i32; N]` types as `[T; 0]` and is rejected `E0401`. matmul *dimensions* may be runtime
  values — a runtime-dimension matmul still dispatches to the GEMM kernel.
- Ordered comparison (`< <= > >=`) is not defined on `bool`, so a **chained comparison** `a < b < c`
  (which parses left-associatively as `(a < b) < c`) is a compile error (`E0401`) rather than a silent
  wrong result — write `a < b && b < c` (`tests/fail/chained_comparison.wk`). Equality `==`/`!=`
  between a `bool` and a non-bool scalar is likewise rejected, so the `==`/`!=` chain `5 == 3 == 0` is
  caught too (`tests/fail/chained_equality.wk`).
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
- No **runtime** bounds checking on array indexing (manual memory is a decided constraint). A
  **compile-time-constant** index past the end is still caught at compile time (`E0501`) — for both a
  fixed-size array (`a[5]` on a `[T; 4]`) and a static tensor dimension (`a[5, 0]` on a
  `Tensor[f32, 2, 2]`); runtime/computed indices remain unchecked. On such an out-of-bounds access the
  backends differ: the interpreter (the oracle) *traps* on an index it can detect as out of range (a
  debugging aid — its memory is a bounded slot vector), while the native backend reads/writes past the
  buffer, classic UB like C (and an optimization level can even change the garbage observed). An
  out-of-bounds program is therefore outside the defined contract — the differential gate's bit-for-bit
  `interp == native` and `-O0 == -O3` invariants hold only for well-defined programs.
- `mem2reg` promotes scalar integer, float **and pointer** slots. A pointer slot qualifies only when
  it is provably written before it is read: the first access inside its own `alloca`'s block must be
  a store, and that block must be the entry block or have an empty dominance frontier. That covers
  every pointer-typed *parameter* (`*T`, `&T`, and every `Tensor[…]`, all of which lower to one MIR
  `ptr`) — the case that matters, since the front end otherwise re-loads the base pointer from its
  stack slot at every element access — including after `-O2` inlining has spliced a callee's entry
  block into the middle of a caller block. A pointer local whose `alloca` and initializing store are
  separated by a branch keeps its slot: promoting it would need a typed "undefined" pointer for the
  read-before-write path, and there is no sound one (`inttoptr 0` is a genuine null natively but a
  *valid, addressable* slot in the interpreter). Arrays, vectors, address-taken locals of any type,
  and any slot accessed at a width other than its own stay in memory (the interpreter and `cse`/`dse`
  handle those directly).
- A **`[]T` slice parameter still re-loads its data pointer at every element access.** A slice is a
  16-byte `{ data, len }` fat pointer passed *by address*, so the base comes from
  `load ptr (gep <param>, 0)` — a load out of caller-owned memory, not out of a local slot — and
  `mem2reg` has nothing to promote. Removing it needs loop-invariant load motion (or a `noalias`
  fact about the fat pointer), not slot promotion; `licm` does not hoist loads today.
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
  mismatch is a hard metadata error that rejects the link outright. `--emit=obj`
  writes the object unchanged. The differential oracle remains `--run`.
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
