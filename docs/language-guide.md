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

Parameters are **immutable by default**, the same rule `let` follows — a function may read one but
not reassign or mutate it. Prefix a parameter with `mut` to opt into mutation:

```mercury
fn scale(mut w: [f32; 256], k: f32) {      // `w` is mutated in place
    for i in 0..256 { w[i] = w[i] * k; }
}
```

A scalar `mut` parameter is a private, mutable copy — changes stay local, exactly like a `mut` `let`.
An **aggregate** `mut` parameter (struct / tuple / array / tensor) is passed **by reference** (its
base pointer, zero-copy — the tensor-kernel default), so mutating it in place is **visible to the
caller**; this is the idiomatic way a kernel writes an output buffer (`out`, `c`, …). Mutating a
non-`mut` parameter — whether a direct rebind (`p = …`) or a projection of an aggregate (`p.f = …`,
`p[i] = …`) — is a compile error (E0304), so an accidental caller-visible write can't slip through. A
**pointer** parameter is exempt for writes through its pointee (`*p = …` needs no `mut`, since that
mutates the pointee, not the binding).

## Bindings ✅

```mercury
let x: i32 = 10;        // immutable
let mut acc: i32 = 0;   // mutable
acc = acc + x;          // reassignment requires `mut`
const TILE: usize = 64; // compile-time constant
let (a, b) = (3, 4);    // tuple destructuring (nested patterns and `_` work too)
```

An unsuffixed numeric literal adapts to its annotation, so `let i: usize = 0;` is fine — and that
threading reaches every position that pins a type (a `let`/`const` annotation, a function argument, a
`return`, a struct-field initializer, and a plain assignment) and **descends into aggregate
literals**, so a typed buffer can be built straight from literals: `let a: [i8; 2] = [127, 0]`,
`let t: (u8, u8) = (200, 1)` (`tests/run/aggregate_literal_adapt.mer`). It also descends into a
**constant binary expression** of literals, so `let v: i64 = 0 - 16` adapts like the unary `-16`
already did (`tests/run/binary_const_adapt.mer`); the folded value is still range-checked, so
`let v: i8 = 100 + 100` is rejected. A typed value must match its
annotation exactly (see error `E0401`). An unsuffixed literal that does not fit the type it adapts to
is rejected (`E0401`, e.g. `let x: i8 = 200;`, or `s.x = 9000000000;` for an `i32` field), not
silently wrapped. With **no** annotation an unsuffixed integer literal defaults to `i32`, but one that
overflows `i32` **widens to `i64`** so its value is never silently truncated
(`tests/run/int_literal_widen.mer`); use a suffix (`9000000000i64`, `3000000000u32`) to pick a
specific type.

A `let` binding may **destructure a tuple** — `let (a, b) = …`, nested `let ((m, n), o) = …`, or a
wildcard `let (keep, _) = …`, including the result of a tuple-returning call
(`tests/run/let_destructure.mer`). A top-level **`const` is usable as a value**: its initializer is
inlined at every use site — in arithmetic, as an array index, as a loop bound, as an **array length**
in a type (`let a: [i32; N]`, including a const-references-const chain;
`tests/run/const_array_length.mer`), and when one `const` references another
(`tests/run/top_level_const.mer`).

## Literals ✅

Integer literals may be **decimal, hex `0xFF`, octal `0o17`, or binary `0b1010`**, with `_` digit
separators (`1_000_000`) and an optional type suffix (`250u8`) (`tests/run/radix_literals.mer`). A
malformed literal — a mistyped radix like `0z123`, an empty `0x`, a bad digit `0b2`, a garbled float
`1.5z`, or a value past `u64` — is a compile error (`E0401`), never silently zeroed
(`tests/fail/malformed_int_literal.mer`). A
**char literal** `'A'` has type **`char`** (a 32-bit Unicode scalar value) — covering the
one-character escapes (`\n` `\t` `\\` `\'` `\0`), `\xHH` hex, and `\u{…}` Unicode escapes. `char` is a
usable annotated type (`let c: char = 'A'`) and is interconvertible with the integer types via `as`
in both directions, so it can be cast, compared, and used in arithmetic (`tests/run/char_literals.mer`,
`tests/run/char_type.mer`).

A **string literal** `"hello"` is typed `*u8` — the same by-pointer convention as an array. Each
unique literal is interned once into a read-only **`.rodata`** static blob (deduplicated by content,
plus a trailing NUL) and its value is that blob's address (`Op::GlobalAddr`). The escapes `\n` `\r`
`\t` `\\` `\"` `\'` `\0` `\xHH` `\u{…}` decode (each code point re-encoded as UTF-8). `print`/`println`
of a `*u8` — a literal or a `let s = "hi";` binding — renders the bytes, while numeric `print` still
prints numbers (`tests/run/string_literal.mer`). Because the blob lives in static data (not the
stack frame), a `*u8` can be **returned from a function and threaded across calls** without dangling
(`tests/run/string_return.mer`). There is still **no string type beyond `*u8`**: no
concatenation/indexing/length operators — a string is a NUL-terminated `*u8` into `.rodata` (🟡).

## Types

| Category    | Examples                                            | Status |
|-------------|-----------------------------------------------------|--------|
| Integers    | `i8 i16 i32 i64`, `u8 u16 u32 u64`, `usize isize`    | ✅     |
| Floats      | `f16 bf16 f32 f64`                                   | ✅ scalar |
| Boolean     | `bool`                                               | ✅     |
| Pointers    | `*T`, `*mut T`, references `&T`/`&mut T`, deref `*p`  | ✅     |
| Arrays      | fixed-size `[T; N]` (literal/repeat init, indexed load/store) | ✅ |
| Tuples      | `(A, B, …)`, field access `t.0`, nested `t.0.1`       | ✅     |
| Structs     | `struct S { … }`, literal `S { f: v }`, field `s.f`  | ✅     |
| Enums       | C-style `enum E { A = 10, B }` (variant = its `i32` discriminant) **and data-carrying** `V(i32)` / `V { f: T }` variants | ✅ |
| Slices      | `[]T` — fat pointer `{data, len}`; `.len()`, indexed load/store, iteration, array→slice unsizing | ✅ |
| SIMD vectors| `f32x4`/`i32x4` (128-bit), generic `vec[T, N]`; wider `f32x8` parses/checks but caps at the 128-bit native ISA | 🟡 |
| Tensors     | `Tensor[f32, M, N]` (+layout) — indexing & ops run for **const *and* symbolic-generic** dims | ✅ |

## Operators ✅

Arithmetic `+ - * / %`, comparison `== != < <= > >=`, bitwise `& | ^`, shifts `<< >>`, boolean
`&& ||` (short-circuit), unary `-`, `!`, and `~`. Like Rust, `!`/`~` are the same operator —
bitwise complement on an integer, logical negation on a `bool`; `~` is the conventional integer
spelling. Precedence is the usual C/Rust ordering, resolved by a Pratt parser. Compound assignment
(`+=`, `*=`, `<<=`, `>>=`, …) is supported.

## Control flow ✅

```mercury
if cond { ... } else { ... }
while cond { ... }
for i in 0..n { ... }
for i in 0..n step 2 { ... }   // strided range; empty if lo >= hi
loop { ... }                    // ✅ infinite loop; exit with `break`
break; continue;                // ✅ innermost loop
'outer: for i in 0..n {         // ✅ a loop label
    for j in 0..n { break 'outer; continue 'outer; }  // target an outer loop by name
}
return expr;
```

Blocks are expressions: the trailing expression of a block (no semicolon) is its value. A
`break`/`continue` with no enclosing loop is a compile error (`E0303`).

A **loop label** `'name:` on a `loop`/`while`/`for` lets a nested `break 'name` / `continue 'name`
target that named outer loop instead of the innermost one — the lexer tells a label `'outer` from a
char literal `'a'` exactly as Rust does (`tests/run/labeled_loop.mer`). A labeled `break`/`continue`
naming an **undeclared** label is rejected with `E0303` (`tests/fail/break_unknown_label.mer`).

`loop` is a **value-producing expression** and `break` carries a value (`let x = loop { break 5; };` ✅).
The loop's type is inferred by unifying every `break <value>` (composing with `if`/`match` value merges),
so a value `loop` can be a `let` initializer, call argument, array element, block tail / `return` value,
or aggregate field; a labeled `break 'outer v` carries a value out of an outer loop. A `break <value>`
targeting a *statement-position* loop (its value discarded, like `while`/`for`) is rejected `E0401`, and
breaks whose shapes disagree are `E0502` — the same merge rules as `if`/`match` arms.

## Pattern matching ✅

```mercury
fn classify(n: i32) -> i32 {
    return match n {
        0 => 10,             // literal pattern
        1 | 2 | 3 => 20,     // or-pattern
        4..10 => 30,         // half-open range (`4..=9` is the inclusive form)
        x if x > 100 => 99,  // identifier binding + an `if` guard
        _ => 0,              // wildcard catch-all
    };
}
```

`match` evaluates its scrutinee once and lowers to an if-else chain over the arms. Patterns are
integer/bool **literals**, **or-patterns** `A | B | C`, half-open `lo..hi` / inclusive `lo..=hi`
**ranges**, **enum-variant** patterns `Color::Red` (matched by discriminant), **tuple** patterns
`(0, _) => …` (each field tested and bound, nesting allowed — and these compose, e.g. `(0 | 1, y)`),
an **identifier** binding (binds the scrutinee or field), and the wildcard `_`. Any arm may carry an
optional `if` guard, and `match` works in both value and statement position. See
`tests/run/{match_expr,match_patterns,match_tuple}.mer`.

A `match` used in **value position must be exhaustive**, like Rust: an `enum` needs every variant, a
`bool` needs both cases, and any other scalar (an unbounded domain) needs a `_` catch-all. A
provably-incomplete value match is rejected at compile time (`E0405`); a guard (`if …`) does not count
toward coverage. This closes a silent-wrong-answer hole — a non-exhaustive value match used to fall
through to a zero default (`tests/fail/match_nonexhaustive.mer`).

## Tuples and structs ✅

```mercury
struct Point { x: f32, y: f32 }

fn main() -> i32 {
    let t = (3, 4);                       // tuple; mixed types allowed: (1.5, 2)
    let p = Point { x: 1.0, y: 2.0 };     // struct literal (fields may be out of order)
    let mut m = p;                        // (aggregates are by-pointer locals)
    print(t.0 + t.1);                     // tuple field access -> 7
    print((p.x + p.y) as i32);            // struct field access -> 3
    return 0;
}
```

Tuples and structs lower to a flat, padded byte buffer (the local's value *is* its base pointer, the
same convention arrays follow); field access is a typed load/store at the field's byte offset, and a
field that is itself a tuple is reached by chaining — `t.0.1`, `t.0.0.0` (`tests/run/nested_tuple_field.mer`).
**Nested aggregates** work too: a struct/tuple field that is itself a struct (any depth), and arrays
of structs, lay out recursively, and an aggregate field initialized from a non-literal value is
deep-copied leaf by leaf (`tests/run/struct_nested.mer`). Whole-aggregate **assignment** (`s = other;`)
deep-copies leaf by leaf as well (`tests/run/struct_assign.mer`). Both run identically on the
interpreter and the native backend. An aggregate passes **into a function by reference** (its base
pointer, zero-copy) and is **returned by value** through a hidden-pointer (sret) ABI in mir_build, so
no aggregate ever rides in a register and the two backends agree (`tests/run/{struct_fn,struct_return}.mer`).
Because a by-reference parameter aliases the caller's storage, mutating an aggregate parameter
requires `mut` on it (see *Functions* above) — a non-`mut` aggregate parameter is effectively
read-only, and a `mut` one is the in-place output buffer a kernel writes.

## Enums ✅

```mercury
enum Code { Ok = 10, Err = 20 }
enum Color { Red, Green, Blue }   // 0, 1, 2 (auto-increment from 0)
enum Step { A = 5, B, C }         // 5, 6, 7 (continue after the last explicit value)
enum Expr { Num(i32), Add(i32, i32), Nil }                 // tuple-payload (tagged-union) variants
enum Shape { Circle { r: i32 }, Rect { w: i32, h: i32 } }  // struct-payload variants
```

A **C-style enum** gives each variant an integer discriminant — explicit (`= 10`) or
auto-incrementing from the previous. A variant `E::Name` *is* its discriminant, so it can be bound to
a `let`, compared (`==`), cast (`Code::Ok as i32`), and used as a `match` pattern
(`tests/run/enum_cstyle.mer`). **Data-carrying (tagged-union) variants** also run: a variant may
carry a tuple payload (`Num(i32)`, `Add(i32, i32)`) or named struct fields (`Circle { r: i32 }`), is
constructed as `Expr::Add(3, 4)` / `Shape::Circle { r: 5 }`, and is taken apart by a **payload
`match`** that binds each field — with literal sub-patterns (`Add(0, y)`), `if` guards, nesting in an
array of enums, and embedding in a struct field (`tests/run/enum_payload_tuple.mer`,
`enum_payload_struct.mer`). A value is a 4-byte `i32` discriminant plus a padded payload union
addressed by base pointer, so the interpreter and the native backend address it identically
(interp == native, `-O0` == `-O3`).

## Tensors and compile-time shape checking ✅ shape-check + const- and symbolic-shape exec (the headline feature)

```mercury
fn matmul<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N], mut c: Tensor[f32, M, N]) { ... }
```

Tensor dimensions are part of the type. The semantic analyzer unifies dimensions across a call:
the shared `K` above must agree on both operands. Mismatches are **compile errors**, not runtime
faults:

- `E0501` — rank mismatch (different number of dimensions).
- `E0502` — dimension mismatch (e.g. calling `matmul::<512, 512, 513>` with `Tensor[f32, 512, 512]`),
  including conflicting bindings of a symbolic dimension.

Run `mercuryc --explain E0502` for a worked example. Dimensions may be integer literals, symbolic
generic names, or `?` for a runtime dimension. Tensor element types must be scalars (`E0302`).

Shape checking is not limited to call arguments: an elementwise binary op `a + b` whose operands
have different shapes, and a function whose returned value's shape disagrees with its declared
`-> Tensor[…]`, are both `E0502` (see `tests/fail/shape_binop_mismatch.mer`,
`shape_return_mismatch.mer`). Inside a **generic** function these body checks treat the function's
own dimension variables as **rigid** — `N` matches only `N`, never another generic or a constant — so
a generic function cannot lie about its output shape either: `fn f<M, N>(a: Tensor[f32, M, N]) ->
Tensor[f32, N, 5]` is `E0502` (`tests/fail/generic_return_shape_lie.mer`). Call-site unification is a
different context and still **infers** a callee's dims from its arguments (`matmul::<…>(a, b, c)`
binds `M, N, K` from the operands). A constant index past a static tensor dimension, like a fixed-size
array, is `E0501`.

**What runs today.** A tensor with **compile-time-constant shape** executes end-to-end on both
backends: multi-dimensional indexing `a[i, j]` flattens to a row-major GEP off the base pointer (a
tensor is passed by base pointer, like an array out-param), so elementwise tensor kernels and tensor
matmuls run — `tests/run/tensor_*.mer`. A matmul written in tensor notation
(`c[i,j] = Σ a[i,k]·b[k,j]`, both the dot-product `s += a[i,k]*b[k,j]` and accumulate
`c[i,j] += a[i,k]*b[k,j]` spellings, including the `b[j,k]` `nn.Linear` `A·Bᵀ` form) dispatches to the
same tuned `mercury_sgemm` microkernel as the flat `a[i*K+k]` spelling — a 2-index access supplies its
row stride from the tensor's inner dimension. A **symbolic-generic** shape now executes too (✅):
`fn add<M, N>(a: Tensor[f32, M, N], …)` runs at any per-call size — the dims are threaded in as
hidden runtime `i64` parameters, so `a[i, j]`'s row stride (`i*N + j`, with `N` a runtime value) and
the loop bounds (`0..M`) resolve at run time, and a turbofish supplies them (`add::<2, 3>(…)`)
(`tests/run/generic_shape.mer`). Because the symbolic address arithmetic matches the constant-shape
form, it is byte-identical to the same kernels written with literal dims — and even a matmul with
runtime `m, n, k` dispatches to the tuned GEMM kernel (`tests/run/matmul_dynamic.mer`).

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

These are recognized by the MIR builder and implemented directly by the interpreter (and, on the
native backend, by the runtime).

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

These intrinsics also accept **integer** operands: `abs`/`round`/`floor`/`ceil`/`trunc` are
type-preserving on an integer (integer `abs` is `select(x < 0, −x, x)`; rounding an integer is the
identity), while `sqrt` and the transcendentals promote an integer operand to `f32`
(`tests/run/int_math.mer`).

When written as a pure `for i { out[i] = f(x[i]) }` loop over `f32` arrays, **any of the 35**
transcendentals (`exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu`/the inverse trig/the hyperbolic family/…)
are **dispatched to a tuned 256-bit
AVX2/FMA kernel** (`mercury_vmath_f32`) — the same domain-aware lowering as matmul→GEMM — so the
activation family runs ~2–13× faster than C's scalar `libm`, and ~28× across cores under
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
--emit=<stage>        tokens | ast | mir-high | mir (alias mir-low) | grad | llvm-ir | obj | exe
                      grad = reverse-mode backward MIR of a loss fn (see --grad-of/--grad-wrt)
--grad-of=<fn>        function to differentiate for --emit=grad / --train   (default: loss)
--grad-wrt=<i,..>     parameter indices to differentiate w.r.t.             (default: all buffer params)
--train               run a fwd→bwd→optimizer loop and print the loss trajectory
--train-steps=<n>     training steps (default 100); pair with --train-lr, --train-opt=sgd|adamw, --train-seed
-O0|-O1|-O2|-O3       optimization level (-O3 currently runs the -O2 pipeline)
-o <path>             output path
--error-format=<f>    human | json
--explain <CODE>      print an extended explanation for an error code
--color=<when>        auto | always | never
```

Use `--emit` to inspect any stage of the pipeline, e.g. `mercuryc --emit=mir -O2 kernel.mer` to see
the optimized IR, or `mercuryc --emit=ast kernel.mer` to see the parse tree. Reverse-mode autodiff is
CLI-driven too: `mercuryc --emit=grad --grad-of=loss model.mer` prints the backward MIR of a loss
function, and `mercuryc --train --train-opt=adamw model.mer` runs its fwd→bwd→optimizer training loop.
