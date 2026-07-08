# Mercury Roadmap & Known Limitations

Mercury is built openly and incrementally. This page is an honest snapshot of what works, what is
checked-but-not-executed, and what is planned — so expectations match reality.

## Works end to end (interpreter `--run`, **and native code** `--backend=native`)

Two CPU execution backends now run the full language and agree bit-for-bit (a differential gate proves
it across opt levels): the zero-dependency tree-walking interpreter (the reference oracle) and a
from-scratch **Cranelift native backend** (JIT for `--run --backend=native`, object/exe via
`--emit=obj|exe`) — **no LLVM toolchain required**. See `BENCHMARKS.md` for cross-language numbers.

- **Multi-file modules**: `import a.b` loads `a/b.mer` (resolved against the *root* source file's
  directory) and splices its items into one merged flat namespace, so cross-file fns/consts/structs
  just work on both backends (`tests/run/import_multi.mer`). Files load once each — an import cycle
  or diamond dedups by canonical path, never errors. An unresolvable import is `E0305`; a cross-file
  duplicate name is the ordinary `E0300`, pointing at both definitions with each file's own
  path/line. (`import … as …` / `import x.{a,b}` parse but don't rename/restrict yet;
  `--emit=tokens|ast` stay root-file-only by design.)
- Functions (including recursion and mutual recursion — the interpreter oracle runs on a
  512 MiB worker thread, so ordinary recursion does not overflow the host's small default stack), and
  direct calls. *Caveat:* the tree-walking interpreter's call frames are ~an order of magnitude larger
  than the native backend's machine frames, so the interpreter overflows at a far shallower recursion
  depth (~tens of thousands of nested calls) than native (which handles millions). Recursion past the
  interpreter's stack bound is a **resource limit outside the bit-for-bit differential contract** — the
  oracle aborts where the native backend may still complete, the same "outside the defined contract"
  status as an out-of-bounds access. Matching native's depth would need ~10× the interpreter stack
  (impractical); a real program rarely recurses that deep on a stackful backend.
- `let`/`let mut`/`const`, shadowing, block-as-expression values, **`let` tuple destructuring**
  (`let (a, b) = …`, nested patterns, `_`; `tests/run/let_destructure.mer`), and a **top-level
  `const` used as a value** (its initializer inlined at every use site — arithmetic, array index,
  loop bound, **array length in a type** (`let a: [i32; N]`; `tests/run/const_array_length.mer`),
  const-referencing-const; `tests/run/top_level_const.mer`).
- Integers (`i8..i64`, `u8..u64`, `usize`/`isize`), `bool`, `char` (a 32-bit Unicode scalar,
  interconvertible with the integers via `as`; `tests/run/char_type.mer`), and floats. `f32` is computed at **`f32`
  precision** (interpreter and native agree exactly); **both `bf16` and `f16` are real 2-byte storage**
  rounded to that grid (round-to-nearest-even) on store and on the cast, with `f32` compute. bf16 rounds
  with cheap inline bit-math (it's the top 16 bits of an `f32`); f16's IEEE-half layout has no such
  shortcut and Cranelift x64 has no f16 convert lowering, so f16 cast/load/store call shared `half`-crate
  shims (`mercury_f32_to_f16_bits`/`mercury_f16_bits_to_f32`) the interpreter uses too — bit-exact by
  construction. `f64` is full. (The GPU adds tensor-core fp16/bf16/fp8 GEMM on top.)
- **Mixed-precision (bf16 *and* f16) CPU suite → SIMD dispatch**: low-precision `[bf16]`/`[f16]` arrays
  read through an `as f32` widening cast (lossless: `<<16` for bf16, F16C `vcvtph2ps` for f16) with
  **f32 accumulate/compute** — the standard ML contract — are recognized and lowered to half-precision
  runtime kernels. Symmetric across both precisions:
  - **reductions** `dot`/`sum` (`mercury_{dot,sum}_{bf16,f16}[_parallel]`) — ~3.0–3.5× vs C for dot,
    ~6–8× for sum, growing as the working set spills L3 (`tests/run/reduce_{bf16,f16}.mer`);
  - **max/min/absmax** (`mercury_reduce_{bf16,f16}[_parallel](x,n,op)`) — the per-tensor absmax is the
    symmetric-quantization scale; exact (max/min round nothing) (`reduce_bf16_minmax.mer`). Inside a
    `@parallel` function each of dot/sum/max/min/absmax dispatches to its multicore `_parallel` twin —
    a deterministic fixed-`RCHUNK` chunk fold (the per-chunk serial kernel + an ascending partial
    combine), bit-identical regardless of thread count and the bf16/f16 analog of the f32
    `mercury_sreduce_f32_parallel` (`tests/run/parallel_reduce_lowp.mer`);
  - **streaming axpby** `out = a·x + b·y` (`mercury_axpby_{bf16,f16}`), half-in/f32-out, ~1.3× ≫ L3
    (requires two additive terms; a 1-term scale would force a `0*inf` the source lacks);
  - **activations** — the full 36-op transcendental set over a half-precision input
    (`mercury_vmath_{bf16,f16}`, `out[i] = f((x[i] as f32))`).
  The recognizers are precision-generic (`match_lowp_reduction`/`match_lowp_axpby`/`match_vmath_stmt`),
  and the interpreter marshals through the identical kernel, so native == interp bit-for-bit. C/Rust
  can vectorize neither a `libm` call nor the half→f32 widen, so the gap is structural. A half-precision
  **output** now ships too: a narrowing store (`f32_to_{bf16,f16}_bits`, the reverse of the widen)
  feeds all-half twins — the streaming **axpby** `mercury_axpby_{bf16,f16}_out` (bf16/f16 in *and* out,
  6 bytes/elem vs an all-f32 axpby's 12, so ~2× on the memory-write-bound stream) and the half-output
  **activations** `mercury_vmath_{bf16,f16}_out` (half in *and* out — storage-halving, but the
  transcendental itself is call-bound, so a footprint win not a throughput one). The narrowing round
  goes through the same shim the interpreter uses, so native == interp bit-for-bit and `-O0` == `-O3`
  (`tests/run/{axpby_half_out,vmath_half_out}.mer`).
- All arithmetic/comparison/bitwise/boolean operators (`&&`/`||` **short-circuit**), compound
  assignment, casts — including a float → narrow-int cast (`1e30 as i8`) that **saturates** identically
  on both backends (`tests/run/float_cast_narrow.mer`).
- `if`/`else` (statement and value position), `while`, `for … in a..b [step s]`, `loop { … }` with
  `break`/`continue`, including **labeled loops** `'outer: for … { … break 'outer; continue 'outer; }`
  — a labeled `break`/`continue` targets the named enclosing loop, not just the innermost (the lexer
  tells a label `'outer` from a char literal `'a'` the way Rust does; `tests/run/labeled_loop.mer`).
  `continue` in a range `for` runs the loop step (`tests/run/for_continue.mer`); a `break`/`continue`
  outside any loop, or one naming an undeclared label, is rejected with `E0303`
  (`tests/fail/{break_outside_loop,break_unknown_label}.mer`). **Loop-as-expression / break-with-value**
  (`let x = loop { break 5; };`) works: `loop` is a value-producing expression whose type is inferred by
  unifying every `break <value>` (composing with `if`/`match` value merges), so a value `loop` is a call
  argument, array element, tail/return value, or aggregate field; a labeled `break 'outer v` carries a
  value out of an outer loop. The value merges on a typed block param on the loop's exit block — the same
  machinery `if`/`match` merges use, so interp == native, `-O0` == `-O3`. A `break <value>` targeting a
  statement-position loop (value discarded, like `while`/`for`) is rejected `E0401`, and breaks with
  mismatched shapes `E0502` (`tests/run/loop_break_{value,labeled,compose}.mer`,
  `tests/fail/break_value_{in_stmt_loop,shape_mismatch}.mer`).
- **`match`** in value and statement position: integer/bool literal, identifier-binding, and wildcard
  `_` patterns, **or-patterns** `1 | 2 | 3`, half-open `0..10` / inclusive `0..=10` **range** patterns,
  **enum-variant** patterns `Color::Red` (matched by discriminant), and **tuple** patterns `(0, _)`
  (per-field tests + bindings, nesting and composition like `(0 | 1, y)`), each with an optional `if`
  guard. The scrutinee is evaluated once and the whole `match` lowers to an if-else chain
  (`tests/run/{match_expr,match_patterns,match_tuple}.mer`).
- **C-style enums** `enum Code { Ok = 10, Err }` — explicit or auto-incrementing discriminants; a
  variant *is* its integer discriminant, usable in `let`, `==`, `as i32`, and as a `match` pattern
  (`tests/run/enum_cstyle.mer`).
- **Data-carrying (tagged-union) enums** `enum Expr { Num(i32), Add(i32, i32), Nil }` — tuple *and*
  struct payloads: construction `E::V(a, b)` / `E::V { x, y }`, and payload `match` with field
  bindings, literal sub-patterns, `if` guards, nesting in an array of enums, and by-value passing to a
  function. A value is a 4-byte i32 discriminant + a padded payload union, addressed by base pointer
  exactly like a struct — no backend change; interp == native bit-for-bit and -O0 == -O3
  (`tests/run/enum_payload_{tuple,struct}.mer`). A payload-carrying variant is *not* castable to an
  integer, and an enum that holds itself by value (`Cons(i32, List)`) is rejected as infinitely sized
  (E0402); a non-exhaustive `match` over the variants is E0405.
- **Slices `[]T`** — a fat-pointer view `{ data: *T @ 0, len: i64 @ 8 }` over existing array storage:
  length `s.len()`, indexed read/write `s[i]`, iteration `for x in s`, and passing to a function —
  either an already-materialized slice or a fixed-size array *unsized* to a slice parameter at the
  call site (`let s: []T = arr` / `f(arr)`). A write through the slice aliases the backing array. The
  16-byte fat pointer is a by-pointer aggregate the two backends address identically, so interp ==
  native bit-for-bit and -O0 == -O3 (`tests/run/slice_basics.mer`). The element types must match on
  the unsizing coercion, and a scalar passed where a slice is expected is E0401; slice indexing is a
  runtime-length view (unchecked, like a pointer — a slice has no static length to bounds-check).
- **Radix & char literals**: hex `0xFF` / octal `0o17` / binary `0b1010` integer literals with `_`
  digit separators and type suffixes (`tests/run/radix_literals.mer`), and char literals `'A'` (the
  one-character / `\xHH` / `\u{…}` escapes) typed `char` — a 32-bit Unicode scalar value,
  interconvertible with the integers via `as` (`tests/run/char_literals.mer`, `tests/run/char_type.mer`).
- **String literals**: `"hello"` lives once in a read-only static-data section (`.rodata`), typed
  `*u8`; the escapes `\n` `\r` `\t` `\\` `\"` `\'` `\0` `\xHH` `\u{…}` decode. Each *unique* literal is
  emitted once (deduped by content) and referenced by address via `Op::GlobalAddr`, so a **returned or
  threaded `*u8` no longer dangles** — the pointer stays valid after the callee's frame is gone, and
  the interpreter and native backend agree (`tests/run/string_return.mer`). `print`/`println` of a
  `*u8` (a literal or a `let`-bound string) renders the bytes, not the pointer value
  (`tests/run/string_literal.mer`). No string *type* beyond `*u8` yet — no concatenation/indexing/
  length (🟡).
- **Pointers & references**: `&x`/`&mut x` take an address, `*p` loads/stores through it, and a
  pointer parameter threads through calls — address-taken locals correctly stay in memory under the
  optimizer (`tests/run/pointer.mer`). `as` casts bind looser than `*`/unary, tighter than binary
  (`*p as T` is `(*p) as T`).
- **Fixed-size arrays** `[T; N]`: literal/repeat init, indexed load/store, array parameters passed
  by base pointer (out-params). Real kernels run: dot, SAXPY, GEMM, matmul, ReLU, clamp, transpose.
- **Tuples & structs**: `(a, b)` / `Name { f: v, … }` literals, field access `t.0` / `s.f` (read and
  assign), heterogeneous fields with correct padded layout, **and nested aggregates** (struct-in-struct
  to any depth, arrays of structs, an aggregate field deep-copied from a variable). Lowered as a flat
  byte buffer with byte-offset field GEPs (the local's value is its base pointer, like an array; nested
  fields recurse), so the interpreter and native backend agree bit-for-bit with no backend-specific
  aggregate handling (`tests/run/{tuple,struct,struct_nested}.mer`), and nested tuple-field access
  `t.0.1` / `t.0.0.0` plus whole-aggregate assignment `s = other;` both run
  (`tests/run/{nested_tuple_field,struct_assign}.mer`). A tuple/struct also crosses function
  boundaries — passed **in by reference** (base pointer, zero-copy; mutating it needs `mut`, else
  E0304) and **returned by value** via a hidden-pointer (sret) ABI modeled in mir_build, so no
  aggregate ever rides in a register and both backends agree
  (`tests/run/{struct_fn,struct_return}.mer`).
- **Constant-shape tensors** `Tensor[f32, R, C]`: multi-dimensional indexing `a[i, j]` lowers to a
  row-major GEP (the shape-typed surface), so elementwise tensor kernels and tensor matmuls execute
  on both backends (`tests/run/tensor_*.mer`) — and a matmul written in tensor notation dispatches to
  the tuned GEMM kernel (see below).
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
  native, `-O0` == `-O{1,2,3}` (`tests/run/generic_shape.mer`). A **symbolic-dim matmul still dispatches
  to the tuned GEMM kernel** (the recognizer already compares strides by dimension identity; once the
  dims are runtime values `emit_sgemm` materializes them from the hidden params), so `matmul<M, N, K>`
  runs on the AVX2/FMA microkernel at any size (`tests/run/generic_shape_matmul.mer`). Ranks 1–3,
  elementwise kernels, and a generic caller forwarding its own symbolic tensors all run — and so does
  a `@parallel` symbolic-shape function (a matmul reaches the multicore `mercury_sgemm_parallel`; an
  elementwise nest runs correctly through the ordinary path, exactly as a constant-shape `@parallel`
  tensor function does).
- **Matmul → GEMM dispatch**: the compiler recognizes a matmul loop nest (the `ikj` accumulate and
  `ijk` dot-product forms, including the `nn.Linear` `C = A·Bᵀ` spelling) and lowers the whole nest
  to a tuned register-blocked (6×16), cache-tiled, packed **AVX2/FMA** microkernel in the runtime —
  the way XLA/TVM/oneDNN lower a matmul op. Serial and `@parallel`. Beats gcc/rustc's naive nest
  ~3–3.6× single-thread (~110–120 GFLOP/s ≈ 90% of one P-core's roofline) and up to ~18× parallel on `C = A·B` (~19–26× single-core / up to ~104× parallel on `nn.Linear`), the lead
  growing with size. Dimensions may be compile-time literals **or runtime values** (function
  params/locals): the recognizer checks strides symbolically, so a general matmul function dispatches
  to the kernel, not just fixed-size benchmark kernels. The two factors may even be the **same array**
  (a Gram matrix `A·Aᵀ`, or self-attention `Q·Kᵀ` sharing a buffer) — both sides are read-only. The
  interpreter calls the identical kernel (marshalling its memory), so the two stay bit-exact. The
  **transposed-A weight-gradient** form `C = Aᵀ·B` (`dW = dYᵀ·X`, A stored `[k,m]` with the
  contraction axis outermost) is recognized too and dispatched to `mercury_sgemm_tn`, which transposes
  A once then reuses the same NN microkernel — so the training backward pass leaves the scalar nest
  (`tests/run/matmul_tn.mer`). The **bf16/f16 mixed-precision** `nn.Linear` (`[bf16]`/`[f16]` inputs
  widened `as f32`, f32 accumulate) likewise dispatches to `mercury_sgemm_{bf16,f16}_nt` — a lossless
  widen prepass then the same tuned kernel — ~25× the idiomatic bf16 C (`tests/run/linear_{bf16,f16}.mer`),
  and its **fused FFN epilogue** (`act(A·Bᵀ + bias)`) folds to `mercury_sgemm_{bf16,f16}_nt_epi` — bias +
  activation in the GEMM writeback for free (`tests/run/linear_{bf16,f16}_ffn.mer`; see epilogue fusion below).
  The mixed-precision **weight-gradient** `C = Aᵀ·B` (`dW = dYᵀ·X`, the training backward) is recognized
  too → `mercury_sgemm_{bf16,f16}_tn` (widen prepass → the f32 `mercury_sgemm_tn`), closing the half
  training-backward gap where the naive nest loses to both the un-vectorizable widen and the column-strided
  A reads (`tests/run/matmul_{bf16,f16}_tn.mer`).
- **Batched matmul → per-head GEMM dispatch**: a matmul nest wrapped in a batch loop, with each index
  carrying a per-batch base offset (`x[h*S*D + i*K + k]` — the shape of **multi-head attention**, one
  matmul per head), also dispatches. The recognizer peels the offset off each flattened index (it
  must be invariant in the matmul's own `i,j,k`) and the kernel call GEPs each base pointer by it, so
  every head runs the tuned microkernel instead of a scalar nest. Both the `Q·Kᵀ` and `P·V` matmuls of
  an MHA forward dispatch (see `tests/run/{batched_matmul,multi_head_attention}.mer`).
- **Matrix transpose → cache-blocked kernel**: the nest `for i { for j { dst[j*R+i] = src[i*C+j] } }`
  dispatches to a `B=32` cache-blocked `mercury_transpose_f32[_parallel]`. The naive transpose writes
  `dst` with stride `R` (a cache miss per element for large `R`) and gcc/rustc do not loop-tile it at
  `-O3`, so the blocked kernel wins ~1.5× single-core / ~9–14× `@parallel` on this memory-bound layout
  op (attention score / weight-layout transposes). A permutation, so bit-exact (`tests/run/transpose_f32.mer`).
  **bf16/f16** transposes dispatch to the same blocked kernel at 16-bit width (`mercury_transpose_u16`, one
  kernel for both — a transpose moves the raw bits) for the half-precision KV/attention layouts (`transpose_bf16.mer`).
- **Column reduction → SIMD colsum kernel**: the column-outer nest `for j { for i { s += x[i*N+j] }; out[j]=s }`
  (the bias gradient `db = Σ_batch dY`, batch sum, reduce-along-axis-0) dispatches to
  `mercury_colsum_f32[_parallel]`, which streams `x` row-major and accumulates eight columns at a time into
  a cache-resident `out[]`. The naive form strides `x` down the rows *and* — verified on the emitted assembly —
  gcc/rustc leave it fully scalar (no `vaddps`), so the kernel wins ~29–47× single-core / ~52–55× `@parallel`.
  Each column sums in `i`-ascending order, so it is bit-exact (`tests/run/colsum.mer`). The **max**/**min**/
  **abs-max** down the same axis (`out[j] = max/min_i x[i,j]`, `max_i |x[i,j]|` — per-channel quant stats,
  axis-0 max/min pooling, and the symmetric int8-quant scale `amax_j`) dispatch to
  `mercury_col{max,min,maxabs}_f32[_parallel]` (first-row seed + `_mm256_max_ps`/`_mm256_min_ps` fold, abs
  via sign-mask `andnot`); the same strided gap (gcc/rustc stay scalar — `fmax`/`fmin` are non-associative)
  gives ~34–50× single-core / ~37–107× `@parallel` (`tests/run/colmax.mer`, `colmin.mer`, `colmaxabs.mer`).
- **Softmax backward → fused dot+apply kernel**: the batched nest `for r { let s=0; for j { s += y·dy };
  for i { dx = y·(dy−s) } }` (the gradient through a row softmax — attention + classifier training) folds to
  `mercury_softmax_bwd_f32[_parallel]`, which delegates the per-row dot to the bit-exact `sreduce` (8 lane
  accumulators) then applies `y·(dy−s)` 8-wide. gcc/rustc keep the dot's *accumulation* scalar (a serial
  `vaddss` chain), so Mercury wins ~1.0–2.0× single-core (the apply is already vectorized in both) and
  ~5–6× `@parallel` (rows across cores). The dot reassociates (the reduction exception), so the differential
  gate is bit-exact while the cross-language check is a tolerance (`tests/run/softmax_bwd.mer`).
- **Activation backward → 256-bit transcendental gradient**: a `for i { dx[i] = act_backward(x[i],
  dy[i]) }` loop (`act` ∈ {`silu`,`gelu`,`sigmoid`,`tanh`,`elu`,`softplus`} — the gradient through every
  FFN/attention/gate nonlinearity in training) folds to one `mercury_vmath2_f32` call with a new two-input
  op code, fusing the upstream `dy·` multiply into `dy·act'(x)`. The derivative is itself a transcendental
  (`silu'`/`sigmoid'`/`softplus'` fold a sigmoid, `gelu'`/`tanh'` a tanh, `elu'` an exp on the negative
  branch — an `expf` C/Rust keep scalar), so the 256-bit kernel wins **~3–12.5× single-core, ~5.5–25×
  `@parallel`** vs scalar C (`tanh_backward` the largest, `elu_backward` the most modest — only x≤0 folds
  `exp`). Pure elementwise (no reduction) → the kernel is bit-identical lane-for-lane, so the differential
  gate is trivial (no reassociation exception); scalar twin / AVX2 / inlined-MIR fallback share one op
  sequence (`tests/run/{silu,gelu,gate,elu_softplus}_backward.mer`). relu/leaky backward are not added (a
  select on x>0 — gcc vectorizes those, so they'd only tie).
- **Transformer building blocks compose**: a transformer FFN (`gelu(x·W1ᵀ)·W2ᵀ`), scaled
  dot-product attention (`softmax(Q·Kᵀ)·V`), **multi-head** attention (the batched per-head form) and
  its **causal** (decoder/autoregressive) variant, 2D convolution (im2col + matmul), and a full
  pre-norm (Llama-style) transformer block all lower with their matmuls dispatched to the GEMM kernel
  and their softmax/GELU/RMSNorm vectorized — and run bit-identically on both backends (see
  `tests/run/{ffn_block,attention,multi_head_attention,causal_attention,conv_im2col,transformer_block,rmsnorm,log_softmax}.mer`).
- **SIMD auto-vectorization**: straight-line elementwise loops (incl. branchy ones via
  if-conversion) lower to 128-bit vector ops, 4×-unrolled, with a scalar remainder — automatically,
  on the native backend; and for **large, compile-time-known trips (~≥2048 elems)** the loop is
  instead emitted as **true 256-bit AVX2** by a raw machine-code path
  (`crates/mercury_codegen_cranelift/src/avx2.rs`, VEX-encoded via `iced-x86`) that sidesteps
  Cranelift's 128-bit CLIF cap (`MERCURY_P4_NO_256` disables it). saxpy/poly/relu/relu6 vectorize.
- **FMA contraction**: a float `x + y*z` becomes one fused multiply-add (`Op::Fma`, a hardware
  `vfmadd`), on both the scalar and vector paths; the interpreter mirrors it with `mul_add`, so the
  two backends stay bit-identical.
- **Reduction vectorization**: a float reduction `s = s + x[k]*y[k]` / `s += ..` lowers to
  vector-lane accumulators (independent FMA chains) + a horizontal reduce + scalar remainder, turning
  the latency-bound serial sum into a throughput-bound one. `dot` runs ~2.7× faster than serial C.
  `fmax`/`fmin` reductions (`m = fmax(m, x[i])`, softmax's row-max) vectorize the same way.
- **`@parallel` reduction → multicore reduction kernel**: a reduction loop in a `@parallel` function
  (dot `x[k]*y[k]`, ssd `(x[k]-y[k])²`, or the unary sum `x[k]`) is **dispatched to a deterministic
  multicore reduction kernel** (`mercury_sreduce_f32_parallel`), spreading the stream across cores to
  aggregate memory bandwidth — `dot@parallel` ~7.9×, `ssd@parallel` ~8.6× faster than single-threaded
  C. The parallel sum is bit-identical to the serial one (fixed-size chunks independent of core count
  + ascending partial combine), so the differential oracle holds.
- **Transcendental intrinsics**: `sqrt`/`rsqrt`/`cbrt` (the root family; `cbrt` for LAB color /
  variance-stabilizing transforms), `exp`/`log`/`exp2`/`log2`/`exp10`/`log10` (≈1-ULP `f32`
  minimax polynomials; base-10 for decibel/log-scale features), `expm1`/`log1p` (Kahan-stable `eˣ−1` / `ln(1+x)`, ≈1-ULP near 0),
  `pow` (= `exp(y·log(x))`), `atan2`/`hypot` (the two-arg geometry pair — full-circle angle, overflow-safe
  2-norm; a `for j { out[j]=f(x[j],y[j]) }` loop dispatches to the 256-bit two-input `mercury_vmath2_f32`),
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
  to a tuned 256-bit AVX2/FMA kernel** (`mercury_vmath_f32`) — the width Cranelift's general (128-bit)
  vectorizer can't reach; a *composed* use auto-vectorizes the inlined poly at 128-bit. So softmax, layernorm,
  GELU (tanh and exact erf), SiLU/swish, ELU, softplus, mish, tanh, RoPE, and **log-softmax /
  cross-entropy** run on SIMD instead of scalar `libm` — **~2–13× faster** than gcc/rustc's scalar
  `libm` call (which can't vectorize a loop containing it; ~28× across cores under `@parallel`). See `tests/run/{transcendental,softmax,
  layernorm,gelu,elu,leaky_relu,softplus,mish,activations,log,erf,trig,ihyp,atan,log_softmax,ffn_block}.mer`.
- **Math intrinsics on integer operands**: `abs`/`round`/`floor`/`ceil`/`trunc` are type-preserving on
  an integer (integer `abs` = `select(x<0, −x, x)`; rounding an integer is the identity), and `sqrt` /
  the transcendentals promote an integer operand to `f32` — so they no longer emit the float-op-on-int
  MIR that the native backend rejected and the interpreter ran lossily (`tests/run/int_math.mer`).
- **Convolution via im2col + GEMM**: a conv written as an im2col gather followed by a matmul has its
  matmul recognized and dispatched to the tuned GEMM microkernel (the XLA/cuDNN lowering), so Mercury
  runs a 3×3 conv **~6–7× faster** than idiomatic hand-written direct convolution in C. See
  `tests/run/conv_im2col.mer`.
- **Operator fusion**: adjacent same-range elementwise loops (e.g. a linear map then ReLU) fuse into
  one loop when the combined body is dependence-safe; CSE then forwards the intermediate through
  registers rather than memory.
- **GEMM epilogue fusion**: a recognized `nn.Linear` matmul (`C = A·Bᵀ`) immediately followed by a
  bias-add / activation loop over `C` (`C[i,j] = act(C[i,j] [+ bias[j]])`) fuses into one
  `mercury_sgemm_nt_epi` call that folds the bias + activation into the microkernel's C-tile
  writeback — so `C` is written once instead of paying a separate read-modify-write pass over it. The
  saving is a fraction of the C-pass traffic, so it grows as K shrinks: ~1.0× at 512³ (compute-bound,
  no harm), ~1.34× at K=64/N=2048, ~1.65× at K=32/N=4096 — exactly the small-K/large-N projections
  (attention-output, down-projection). Serial **and `@parallel`** — the multicore
  `mercury_sgemm_nt_epi_parallel` is bit-identical to the serial kernel (a fixed-chunk parallel
  reduction), dispatched from a `@parallel` fused-epilogue nest (`tests/run/linear_bias_relu_parallel.mer`).
  The activation set is identity (bias-only), **ReLU,
  GELU, and SiLU** — the transformer FFNs — with **bias optional**, so the bias-free `silu(x·Wᵀ)`
  **SwiGLU** projection (LLaMA/Mistral) fuses too. Both backends call the identical kernel, so it
  stays bit-exact. See `tests/run/{linear_bias_relu,linear_bias_gelu,linear_silu}.mer`. The
  **bf16/f16 mixed-precision** FFN fuses the same way — a half matmul + its bias/activation loop folds
  to `mercury_sgemm_{bf16,f16}_nt_epi` (the widen prepass feeds the identical f32 epilogue), so the
  mixed-precision transformer FFN gets bias + activation for free (`tests/run/linear_{bf16,f16}_ffn.mer`).
  The **residual projection** `x = x + act(x·Wᵀ + bias)` (the transformer skip connection) also folds
  to `mercury_sgemm_nt_epi`, with **beta = 1** so the kernel accumulates `act(x_residual + A·Bᵀ + bias)`
  in its writeback. The accumulate store would otherwise block the matmul recognizer and drop the whole
  nest to a scalar loop, so this recovers the full GEMM dispatch + the fused residual/bias/activation
  with no backend change (`match_matmul_residual`, `tests/run/linear_residual{,_relu}.mer`).
- **`@parallel`** functions execute across CPU cores (rayon runtime); the per-core chunk is itself
  vectorized. The interpreter runs the same range sequentially, so results stay differential-equal.
- Intrinsics `print`/`println`/`assert`.
- The optimizer (`-O0..-O3`), backed by CFG and dominator analyses: whole-program **inlining** of
  leaf functions, **mem2reg** (alloca → SSA), constant folding, algebraic simplification, CFG cleanup
  with block merging, dead/trivial block-parameter elimination, DCE, dominator-tree CSE with load
  forwarding, DSE, and **loop-invariant code motion**. Guarded by an `-O0`-vs-`-O{1,2,3}` differential
  test and post-pass MIR verification; across the run suite and kernels it removes ~42% of IR ops
  (~48–54% on the heavy transformer/GEMM kernels) and runs ~1.5–2.5x faster than `-O0`.

## GPU backend (NVIDIA RTX 4050, behind `--features gpu`)

A GPU backend, `mercury_codegen_gpu`: being a compiler, it **emits PTX text** and **driver-JIT-loads
it via `cudarc`** (`cuModuleLoadData` — the driver's built-in PTX→SASS JIT, so **no `nvcc`/`ptxas`/CUDA
toolkit** is needed to build or run, only the driver). Every transformer op category is a device
kernel, each gated against a CPU reference by a **tolerance** differential (`c·√K·ε`, deterministic
grids) — the CPU↔GPU analogue of the bit-exact CPU gate. Measured honestly on a power-capped 6 GB
mobile 4050 (see `BENCHMARKS.md`):

- **Tensor-core GEMM** (fp16/bf16/fp8 inputs, f32 accumulate): WMMA `m16n16k16` for fp16/bf16
  (~9–13 TFLOP/s, ~5–6× the f32 path); **fp8 (E4M3)** via hand-laid `mma.sync.m16n8k32` (no WMMA fp8 on
  `sm_89`), validated bit-exact. Its fragment-reuse multi-tile kernel (`fp8_gemm_mt_ptx`, 2×4 block of
  16×8 tiles per warp) is now the **fastest** tensor-core path — ~2.1–2.4× the naive single-tile fp8 and
  ~1.3–2.3× fp16/bf16 in the same run (single-tile retained as the fallback for non-divisible shapes).
- **Fused flash-attention** (online softmax, never materializes the `S×S` scores — the kernel that
  *loses* on CPU): warp-per-query-row + `cp.async` double-buffering — **beats the genuinely-fused
  cuDNN + cutlass fMHA in the causal-S≤512 and fused-RoPE regimes** (and is 3.6–5.0× the unfused
  cuBLAS chain, 205–738× naive CUDA-C), trailing cuDNN only at long context (S≥2048).
- **Fused row norms** (softmax/LayerNorm/RMSNorm, one warp per row), **activations** (SFU), **reductions**
  (deterministic; max bit-exact), **conv2d**, and elementwise.
- **A whole pre-norm transformer layer runs end-to-end GPU-resident** — RMSNorm → QKV → flash-attn →
  output proj → residual → RMSNorm → FFN(SiLU) → residual, all on device buffers with no host round-trip
  between ops, matching a CPU f64 reference to max_rel 2.7e-4 and **deterministic** run-to-run.

**End-to-end `--backend=gpu`.** Built with `--features gpu` (a *cargo build* flag, not a `mercuryc`
runtime flag), `mercuryc --backend=gpu --run foo.mer` executes the
program through an **offloading interpreter**: the whole program is tree-walked on the CPU (identical
control flow, buffer layout, and every non-kernel op to the oracle), but recognized GEMM / activation
/ reduction / fused-norm calls run on the device via an `Accelerator` seam (`mercury_interp`) the
driver implements with `mercury_codegen_gpu` (`GpuAccel`). With no accelerator — every other caller,
the differential oracle — the path is byte-for-byte unchanged, so the toolchain-free core is
untouched. The CPU↔GPU boundary is tolerance-gated, so a device error surfaces as an error rather than
a silent CPU fallback. Gated by `gpu_backend_*` tests (driver, `--features gpu`): each family runs on
the interp oracle and the GPU over identical buffers and matches within tolerance (GEMM bit-exact;
silu ~5e-7, dot ~7e-7, softmax ~3e-8 abs), asserting the offload actually fired.

Run the kernel suite with `cargo test -p mercury_codegen_gpu --features gpu` (skips cleanly with no
GPU).

**General MIR→PTX — `--backend=gpu-native`.** Beyond the recognizer-offload path above, a fourth
backend (`GpuLower`) lowers the *whole* program's MIR to PTX, so arbitrary non-recognized kernels run
GPU-side too; an eligible program is fused into a single-block cooperative **megakernel** (one launch,
no host round-trips). It is tolerance-gated against the interpreter oracle and optimization-invariant
(`-O0` ≡ `-O3`), the same contract as the offload path. Coverage is **partial** (UNSUPPORTED ops
skip). The two documented device-kernel miscompiles are now **fixed**: the `velem` Hadamard/Div
binary modes and the `norm` log-softmax/L2 ops are implemented in the static PTX device kernels
(`lower.rs`), which both the single-thread and megakernel paths share — so `hadamard`,
`log_softmax_fused`, `l2norm`, `norm_divide`, and `norm_out_of_place` now match the interp oracle.
A handful of *unrelated* general-lowering gaps remain in the corpus (a float→int narrowing cast, a
parallel reduction, and a `tensor_1d_kernels` megakernel illegal-address) — separate from the
recognized-kernel device helpers.

## Automatic differentiation (`mercury_autodiff`)

Reverse-mode autodiff runs as a **MIR→MIR transform**: given a forward function computing a scalar
loss, it emits a new function that also accumulates the gradient w.r.t. designated input buffers (the
vector-Jacobian product). Matmul adjoints ride the same tuned GEMM kernels, and a fused AdamW step is
emitted as one kernel. Every VJP rule is **finite-difference-gated** (forward + backward run in f64)
against a closed-form reference. It is reachable from `mercuryc`: `--emit=grad` dumps the backward MIR
of a loss function, and `--train` runs a fwd→bwd→optimizer loop (`--grad-of`/`--grad-wrt` select the
loss and parameters; `--train-opt=sgd|adamw` the optimizer).

## Checked but not yet executed

- **Explicit SIMD vector types** `f32x8` etc. in *source*: parse and type-check; user-written vector
  *values* are not yet executed, and the native ISA path (Cranelift) caps vector SSA at 128-bit
  (`f32x4`), so a wider explicit `f32x8` cannot lower even once execution lands — it must split into
  128-bit halves. (Loop auto-vectorization above is separate and *does* run — 128-bit + unrolling by
  default, and now **true 256-bit AVX2 for large, compile-time-known trips (~≥2048 elems)** via a raw
  machine-code emitter that sidesteps this CLIF cap; the recognized kernels get 256-bit AVX2 via the
  runtime microkernels.)
- **Attributes** `@simd`/`@tile`/`@align`/`@extern`/`@export`: parse and validate; consumers in
  progress. (`@parallel` now executes — see above.)

## Planned

- Fusing arbitrary elementwise chains *under* `@parallel`. (The **parallel fused-epilogue GEMM** that
  was the remaining item here has since shipped — a `@parallel` `act(A·Bᵀ [+ bias])` nest dispatches
  to the multicore `mercury_sgemm_nt_epi_parallel`; see "Works end to end" above.)
- Execution of explicit `f32x8`-typed *values* (they parse and type-check today — see "Checked but
  not yet executed" above). The *general loop* vectorizer already reaches true 256-bit AVX2 for large
  trips via a raw machine-code emitter (see "Works end to end" and "Known limitations"); it is
  *source-level* `f32x8` values that still do not execute.
- A minimal stdlib of reusable functions/collections. (The language surface once grouped here —
  structs, enums including data-carrying tagged unions, slices, and multi-dimensional indexing
  `a[i, j]` — now runs end to end; see "Works end to end" above.)
- AMDGPU/ROCm device codegen (the NVIDIA PTX path already ships behind `--features gpu`, and
  reverse-mode autodiff already ships as the `mercury_autodiff` crate — both above).

## Known limitations / sharp edges

- `bf16` **and** `f16` are both real 2-byte storage (round-to-nearest-even), f32 compute, with a full
  symmetric mixed-precision op suite (reductions, max-family, axpby, activations — see above) **and a
  mixed-precision GEMM**: a bf16/f16 `C = A·Bᵀ` `nn.Linear` nest dispatches to `mercury_sgemm_{bf16,
  f16}_nt[_parallel]` — a lossless widen prepass (`<<16` / F16C, ~1/n of the GEMM) feeding the tuned
  AVX2 f32 microkernel (`tests/run/linear_{bf16,f16}.mer`). On this AVX2+F16C box (no AVX-512-BF16) the
  GEMM itself runs in f32, so it is a *footprint* feature on the FLOPs — but it still beats the
  idiomatic bf16 C **~25× single-core** (that C can vectorize neither the inline `bf16→f32` widen nor
  the serial reduction), ~10× vs a hand-optimized widen-then-tile bf16 C. A half-precision *output* on
  the streaming ops **now ships** — the narrowing store feeds all-half `mercury_axpby_{bf16,f16}_out` /
  `mercury_vmath_{bf16,f16}_out` twins (~2× on the memory-write-bound axpby, its write traffic halved;
  a footprint win on the call-bound activations); see "Works end to end" above.
- The *general* vectorizer emits **128-bit CLIF by default** — Cranelift's vector ISA still rejects a
  256-bit `f32x8` SSA value (verified empirically on Cranelift 0.124, pinned as the
  `cranelift_still_rejects_f32x8`/`p4_probe_vec256_ops` tripwire tests). For **large,
  compile-time-known trip counts (~≥2048 elems)** it now dispatches the loop to a **raw-AVX2 256-bit
  machine-code emitter** (`crates/mercury_codegen_cranelift/src/avx2.rs`, VEX-encoded via `iced-x86`;
  `MERCURY_P4_NO_256` disables it) — the same way the GEMM/vmath runtime microkernels reach 256-bit,
  and precisely *why* that raw emitter exists (Cranelift can't legalize the wider lane). Below the
  threshold an out-of-line 256-bit call would lose to the inlined 128-bit path, so small or
  runtime-unknown trips stay 128-bit + 4× unrolling; there compute-bound *elementwise* kernels use 2×
  the FMA ports they could, but the vectorized **transcendentals still beat scalar `libm` ~2.5–3×**.
  The loop vectorizer assumes distinct array parameters do not alias.
- Array *length* in a type may be an integer literal or a top-level `const` (resolved through
  const-to-const chains; `tests/run/const_array_length.mer`); a **symbolic** length (a generic `N`) or
  a **computed** one (a const whose initializer is an expression, e.g. `const N = 2 + 2`) still falls
  back to an opaque pointer. matmul *dimensions* may be runtime values — a runtime-dimension matmul
  still dispatches to the GEMM kernel.
- Ordered comparison (`< <= > >=`) is not defined on `bool`, so a **chained comparison** `a < b < c`
  (which parses left-associatively as `(a < b) < c`) is a compile error (`E0401`) rather than a silent
  wrong result — write `a < b && b < c` (`tests/fail/chained_comparison.mer`). Equality `==`/`!=`
  between a `bool` and a non-bool scalar is likewise rejected, so the `==`/`!=` chain `5 == 3 == 0` is
  caught too (`tests/fail/chained_equality.mer`).
- Tensor shape checking is **sound for both concrete and generic shapes**. A function's own generic
  dimension variables are **rigid** inside its body: when two shapes that both carry the function's
  generics are checked — its declared-vs-returned shape, an elementwise operator's operands, an
  assignment, or two `if`/`match` value arms — the dims must match by *identity* (`N` matches only
  `N`); they are never bound to each other or to a constant. So a generic function can no longer **lie
  about its output shape**: `fn f<M, N>(a: Tensor[f32, M, N]) -> Tensor[f32, N, 5]` is rejected
  (`E0502`, `tests/fail/generic_return_shape_lie.mer`). Previously the body-check bound `N := M` and
  silently accepted the constant `5` against the generic `N`, and a turbofish (`f::<2, 2>`) then made
  the lie concrete and over-strided — turning a type-valid index into an out-of-bounds read the
  interpreter trapped on but native did not. Call-site unification is a *different* context and still
  **infers** a callee's dims from the argument shapes (there the callee's generics are inference
  variables, not rigid — `matmul::<…>(a, b, c)` binds `M, N, K` from the arguments as before). A
  related hole is also closed: a rank-1 tensor parameter binds its symbolic dim from a decaying
  array's length, so `f<N>(a: Tensor[f32, N], b: Tensor[f32, N])` rejects arrays of different lengths
  (`tests/fail/generic_tensor_arg_length_mismatch.mer`). The former undeclared-dim **lenience** is now
  a diagnostic: an **undeclared** dim name in a tensor type (not a declared generic, integer, `?`, or
  `const`) is rejected with **E0504** and a did-you-mean hint, so a typo like `Tensor[f32, KK]` for `K`
  no longer silently introduces a fresh implicit dim and drops the shared constraint
  (`tests/fail/generic_shape_unknown_dim.mer`).
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
- `mem2reg` promotes only scalar integer/float slots; arrays, pointers, and address-taken locals
  stay in memory (the interpreter and `cse`/`dse` handle those directly).
- **String literals live in a read-only static-data section** (`.rodata`), referenced by address via
  `Op::GlobalAddr` and deduped by content (one blob per unique literal). So **returning or threading a
  `*u8`** that points at a literal created inside a callee is valid — the pointer outlives the frame,
  and the interpreter and native backend print it identically (`tests/run/string_return.mer`). (This
  closed a real interp-vs-native divergence: a literal used to live in the caller's frame, so a
  returned `*u8` dangled on native — freed stack, an empty line — while the interpreter's persistent
  memory masked it.)
- **`--emit=exe` links the `mercury_runtime` kernels**, so a program that dispatches a recognized
  kernel (`for i in 0..N { y[i] = exp(x[i]); }`, a `matmul`, a reduction) compiles to a standalone
  executable whose output matches `--run` (`crates/mercuryc/tests/exe.rs`). The Cranelift object is
  linked by a **rustc-driven link** (rustc invokes the object's native platform linker and links
  `mercury_runtime` as a dependency; a generated shim supplies the `mercury_rt_*` runtime), which also
  links the string `.rodata` relocations — neither of which the MinGW `cc` path on this host can do.
  It falls back to the C-runtime `cc` link (scalar, no-data, no-kernel programs) when rustc or the
  runtime rlib is unavailable, and the exe gate skips cleanly when no toolchain can link. `--emit=obj`
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
