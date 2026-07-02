# The Mercury Language Guide

This guide describes the Mercury language as it exists today. Mercury is built incrementally; where
a feature parses and type-checks but does not yet execute end-to-end, that is called out explicitly.

> **Maturity legend**
> - ✅ **runs** — lexes, parses, type/shape-checks, lowers to MIR, and executes — both via the
>   interpreter *and* the native Cranelift backend (JIT/object), which are differentially tested to
>   agree bit-for-bit.
> - 🟡 **checked** — parses and type/shape-checks; not yet lowered/executed.
> - 🔵 **planned** — designed for, syntax may be accepted, semantics not implemented.

## A first program

```mercury
module hello

fn main() -> i32 {
    print(42);
    print(7 * 6);
    return 0;
}
```

```sh
mercuryc --run hello.mer       # prints 42 then 42; exits with main's return value
```

Every file begins with a `module` declaration. Execution starts at `fn main() -> i32`, and the
integer it returns becomes the process exit code.

## Modules

```mercury
module examples.matmul
import std.mem
```

A module path is dotted. `import` brings another module's items into scope. 🟡

## Functions ✅

```mercury
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

Functions take typed parameters and declare a return type after `->`. A function with no `->`
returns nothing. Recursion is fully supported. Calls use ordinary `f(x, y)` syntax; generic
functions may be called with a turbofish `f::<512, 512, 513>(a, b, c)`.

## Bindings ✅

```mercury
let x: i32 = 10;        // immutable
let mut acc: i32 = 0;   // mutable
acc = acc + x;          // reassignment requires `mut`
const TILE: usize = 64; // compile-time constant
```

An unsuffixed numeric literal adapts to its annotation, so `let i: usize = 0;` is fine. A typed
value must match its annotation exactly (see error `E0401`).

## Types

| Category    | Examples                                            | Status |
|-------------|-----------------------------------------------------|--------|
| Integers    | `i8 i16 i32 i64`, `u8 u16 u32 u64`, `usize isize`    | ✅     |
| Floats      | `f16 bf16 f32 f64`                                   | ✅ scalar |
| Boolean     | `bool`                                               | ✅     |
| Pointers    | `*T`, `*mut T`, references `&T`                       | 🟡     |
| Arrays      | fixed-size `[T; N]` (literal/repeat init, indexed load/store) | ✅ |
| Aggregates  | slices `[]T`, tuples, `struct`, `enum`               | 🟡  |
| SIMD vectors| `f32x8`, `i32x4`, generic `vec[T, N]`                | 🟡     |
| Tensors     | `Tensor[f32, M, N]` with optional layout suffix      | 🟡 (shape-checked) |

## Operators ✅

Arithmetic `+ - * / %`, comparison `== != < <= > >=`, bitwise `& | ^`, boolean `&& ||` (short-
circuit), unary `-` and `!`. Precedence is the usual C/Rust ordering, resolved by a Pratt parser.
Compound assignment (`+=`, `*=`, …) is supported.

## Control flow ✅

```mercury
if cond { ... } else { ... }
while cond { ... }
for i in 0..n { ... }
for i in 0..n step 2 { ... }   // strided range; empty if lo >= hi
loop { ... }                    // 🟡 infinite loop
return expr;
```

Blocks are expressions: the trailing expression of a block (no semicolon) is its value.

## Tensors and compile-time shape checking 🟡 (the headline feature)

```mercury
fn matmul<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N], c: Tensor[f32, M, N]) { ... }
```

Tensor dimensions are part of the type. The semantic analyzer unifies dimensions across a call:
the shared `K` above must agree on both operands. Mismatches are **compile errors**, not runtime
faults:

- `E0501` — rank mismatch (different number of dimensions).
- `E0502` — dimension mismatch (e.g. calling `matmul::<512, 512, 513>` with `Tensor[f32, 512, 512]`),
  including conflicting bindings of a symbolic dimension.

Run `mercuryc --explain E0502` for a worked example. Dimensions may be integer literals, symbolic
generic names, or `?` for a runtime dimension. Tensor element types must be scalars (`E0302`).

## Attributes 🟡

```mercury
@inline
@simd
@tile(64, 64)
@parallel(grain = 1)
@align(32)
@extern("C")
@export("mercury_saxpy")
```

Attributes attach to functions, loops, and declarations, and parse/validate today. Several now have
real consumers on the native backend: **`@parallel`** functions execute across CPU cores (✅), and
loop **auto-vectorization, FMA contraction, and elementwise fusion run automatically** (✅) — a
plain `for i in 0..n { out[i] = a*x[i] + y[i] }` is vectorized, fused with an adjacent loop, and
FMA-contracted with no annotation. A **reduction** in a `@parallel` function — `let mut s = 0.0; for
k in 0..n { s += x[k]*y[k] }` (also `(x[k]-y[k])²`, the plain sum, a running `fmax`/`fmin` max/min,
and `fmax(m, abs(x[k]))` absmax — the per-tensor max/range softmax stability and dynamic int8
quantization need) — dispatches to a
deterministic multicore reduction kernel that reaches aggregate memory bandwidth (~8× single-threaded
C), with a result independent of core count (✅). `@tile` (cache tiling) and explicit `@simd`-typed
vector values are still under construction (🔵).

## Built-in intrinsics ✅

- `print(x)` — print an integer/float followed by a newline.
- `println(x)` — alias of `print`.
- `assert(cond)` — trap with a nonzero exit code if `cond` is false (zero); a no-op otherwise.

These are recognized by the MIR builder and implemented directly by the interpreter (and, with the
LLVM backend, by the runtime).

### Math intrinsics ✅

`f32` (scalar or in a loop), each a ≈1-ULP minimax polynomial built from primitive ops:

- `sqrt(x)` / `rsqrt(x)` — hardware square root (and its reciprocal); `cbrt(x)` — the all-real cube root.
- `abs(x)` — `|x|` (as `max(x, −x)`); vectorizes, and a `fmax(m, abs(x[k]))` loop is the per-tensor
  absmax dynamic symmetric int8 quantization uses for its scale.
- `round(x)` / `floor(x)` / `ceil(x)` / `trunc(x)` — round to an integral value (one hardware
  `roundss`/`roundps` each; vectorizes). `round` is round-to-nearest-ties-to-**even** (so `2.5 → 2`,
  `3.5 → 4`); `round(x / scale)` is the quantization step that pairs with `absmax`.
- `exp(x)` / `log(x)` / `pow(x, y)` — `pow` is `exp(y·log(x))`; defined for `x > 0`. Also `exp2`/`log2`
  (base-2, FlashAttention/quantization) and `exp10`/`log10` (base-10, decibel/log-scale features), plus
  the Kahan-stable `expm1`/`log1p`.
- `atan2(y, x)` / `hypot(a, b)` — the full-circle angle of a point, and the overflow-safe 2-norm.
- `erf(x)` — for the exact (erf-based) GELU of BERT/GPT-2.
- `sin(x)` / `cos(x)` / `tan(x)` / `atan(x)` / `asin(x)` / `acos(x)` — RoPE rotary position embeddings
  (`sin`/`cos`) and the angle/geometry/3D-vision/graphics-ML ops (the inverse trig).
- `tanh(x)` / `sigmoid(x)` / `silu(x)` / `gelu(x)` / `elu(x)` / `leaky_relu(x)` / `softplus(x)` /
  `softsign(x)` / `logsigmoid(x)` / `mish(x)` / `selu(x)` / `tanhshrink(x)` / `hardsigmoid(x)` / `hardswish(x)` — the transformer/vision activation family (`silu(x) = x·sigmoid(x)`; `gelu` is the tanh
  approximation; `elu(x) = x>0 ? x : eˣ−1`; `leaky_relu` has slope 0.01; `softplus(x) = ln(1+eˣ)`;
  `softsign(x) = x/(1+|x|)`; `logsigmoid(x) = ln σ(x)`, the stable BCE-with-logits primitive;
  `mish(x) = x·tanh(softplus(x))`). Plus the hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh`.
- `fmax(a, b)` / `fmin(a, b)`.

When written as a pure `for i { out[i] = f(x[i]) }` loop over `f32` arrays, **any of the 35**
transcendentals (`exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu`/the inverse trig/the hyperbolic family/…)
are **dispatched to a tuned 256-bit
AVX2/FMA kernel** (`mercury_vmath_f32`) — the same domain-aware lowering as matmul→GEMM — so the
activation family runs ~5–7.5× faster than C's scalar `libm`, and ~28× across cores under
`@parallel`. Composed/scalar uses (and `erf`/`sin`/`cos`) auto-vectorize the inlined poly at 128-bit.
Every form is bit-identical across the interpreter and native backends.

## Memory and parallelism

No garbage collector and no hidden allocations: every heap byte comes from an allocator you name
(`System`, `Arena`, `Scratch`, `Pool` — 🔵). Cleanup is via `defer` (🔵). **`@parallel` functions
execute today** (✅): the native backend outlines the loop body and dispatches it across CPU cores
via the `mercury_runtime` rayon-backed `parallel_for`, and each per-core chunk is itself
auto-vectorized (parallelism × SIMD). The interpreter runs the same range sequentially, so results
stay differentially equal. Allocator selection and `defer` are still being wired to the surface.

## Command-line interface

```
mercuryc [OPTIONS] <input.mer>

--run                 compile and execute (interpreter by default; see --backend)
--backend=<b>         interp | native | gpu | gpu-native   (default: interp)
                      native = Cranelift JIT; gpu / gpu-native require --features gpu + a CUDA device
--emit=<stage>        tokens | ast | mir-high | mir (alias mir-low) | llvm-ir | obj | exe
-O0|-O1|-O2|-O3       optimization level
-o <path>             output path
--error-format=<f>    human | json
--explain <CODE>      print an extended explanation for an error code
--color=<when>        auto | always | never
```

Use `--emit` to inspect any stage of the pipeline, e.g. `mercuryc --emit=mir -O2 kernel.mer` to see
the optimized IR, or `mercuryc --emit=ast kernel.mer` to see the parse tree.
