# The Wukong Language Guide

Wukong is a language for writing ML/DL tensor kernels. You express kernels as explicit loops over
`Tensor[...]` values with shape-generic parameters and `@parallel` schedules; `wukongc` checks shapes
at compile time (`wukong_sema`) and lowers your program to MIR for execution on CPU or (optionally)
GPU backends. The goal is to be easier and more productive than hand-written CUDA/CUTLASS while
keeping costs visible — not to hide kernels behind a framework runtime. A win is a `.wk` program
through the compiler, not a Rust-launched kernel zoo.

The living thesis is [LANGUAGE_DIRECTION.md](LANGUAGE_DIRECTION.md); the sequenced mission (including AI-editor / vibe-coding) is [WUKONG_MASTER_PLAN.md](WUKONG_MASTER_PLAN.md). Neither file claims those surfaces run today.

This guide describes the Wukong language as it exists today. Wukong is built incrementally; where
a feature parses and type-checks but does not yet execute end-to-end, that is called out explicitly.

> **Maturity legend**
> - ✅ **runs** — lexes, parses, type/shape-checks, lowers to MIR, and executes — both via the
>   interpreter *and* the native Cranelift backend (JIT/object), which are differentially tested to
>   agree bit-for-bit.
> - 🟡 **checked** — parses and type/shape-checks; not yet lowered/executed.
> - 🔵 **planned** — designed for, syntax may be accepted, semantics not implemented.

## A first program

```wukong
module hello

fn main() -> i32 {
    print(42);
    print(7 * 6);
    return 0;
}
```

```sh
wukongc --run hello.wk       # prints 42 then 42; exits with main's return value
```

A file conventionally begins with a `module` declaration; the header is **optional** and purely
informational (see *Modules & imports*), so a single-file program may omit it. Execution starts at
`fn main() -> i32`, and the integer it returns becomes the process exit code. `main` takes no
parameters and returns an integer, a float or nothing; the exit code is the **low 8 bits** of the
value (a float converts toward zero first, saturating; no return type exits 0), so `return 300`
exits 44 under `--run` and from a linked executable on every OS. The program runs on a 512 MiB
stack under `--run --backend=native` and in an `--emit=exe` executable; the interpreter's calls
never use the host stack (its call-depth ceiling is 300,000 frames in every build).

## Source text ✅

Source files are UTF-8; a leading byte-order mark is ignored. Outside string literals, char
literals and comments the text is **ASCII**: identifiers are letters, digits and `_` (a Greek
letter must be spelled out, `alpha` for `α`; `E0101` names the code point and suggests the
spelling). A Unicode character that merely *looks* like ASCII — `−` (U+2212 MINUS SIGN) or a dash
for `-`, `×` for `*`, `≤` for `<=`, fullwidth punctuation, curly quotes, a no-break or zero-width
space — is `E0105`, which names the code point and the ASCII character it imitates. The compiler
then reads it as that ASCII token (a lookalike space is skipped), so one character pasted from a
paper costs one error (`tests/fail/confusable_minus.wk`); a lookalike quote cannot open a literal
and is an error token.

## Modules & imports ✅

```wukong
module examples.matmul
import lib.mathlib
```

A module path is dotted, and `import` makes a program **multi-file**: `import a.b` — in any file of
the program — loads `a/b.wk`, resolved relative to the directory of the **root** source file passed
to `wukongc`. Each file is loaded exactly once, by canonical path: an import **cycle** (`a` imports
`b` imports `a`) or diamond is not an error. An import that resolves to no file is **E0305**
(`wukongc --explain E0305`); resolution is **case-exact on every host**, so an import whose
spelling matches the file only up to letter case (`import Util` for `util.wk`) is **E0330**
even where the file system would have opened it. A diagnostic inside an imported file renders with that file's own
path and line.

**Every file is a module with its own namespace** ✅. Its top-level items (`fn`, `struct`, `enum`,
`const`, and the functions of an `extern` block) are **private** unless declared `pub`, and a file
sees only its own items plus what *its own* imports bring in — imports are not transitive. Three
import forms:

```wukong
import lib.mathlib;          // every `pub` item of lib/mathlib.wk, unqualified — and `mathlib::item`
import lib.geom as g;        // only the qualifier: g::norm1(v), g::Vec2 { .. }, g::Axis::Y, g::SCALE
import lib.stats.{bump};     // exactly `bump`, unqualified
```

A qualified name works in every position: calls and values (`g::norm1(v)`), struct literals
(`g::Vec2 { x: 1, y: 2 }`), enum variants and patterns (`g::Axis::Y`), and types (`fn f(v: g::Vec2)`,
or the item-level spelling `g.Vec2`). Two modules may each define a private `helper` — or both export
`norm1` — without colliding (`tests/run/import_namespaces.wk`). The rules:

- naming a **private** item of another module — by `import m.{x}`, by `m::x`, or by expecting a plain
  `import m;` to bring it in — is **E0312**;
- a definition in the file itself, or an explicit `import m.{x}`, **shadows** a plain import; two
  plain imports that both export a name make an unqualified **use** of it ambiguous (**E0313**), never
  a silent choice — qualify it, select it explicitly, or rename an import with `as`;
- an explicit `import m.{x}` of a name the file also defines, or one `as` name bound to two modules,
  is the ordinary duplicate-name error (**E0300**);
- local variables, parameters and generic parameters shadow imported items exactly as they shadow
  local ones.

Field-level `pub` (`pub x: i32` inside a struct) is accepted but not yet enforced 🟡: a struct's
fields are readable wherever the struct is. An imported item keeps its source name in every
diagnostic, prefixed by its module path (`lib.geom.Vec2`). The `module` header itself remains
informational — it does not participate in resolution and need not match the path a file was
imported under. Note that `--emit=tokens|ast` dump the **root file only**; every later stage — and
`--run` on either backend — sees the whole resolved program.

## Functions ✅

```wukong
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
```

Functions take typed parameters and declare a return type after `->`. A function with no `->`
returns nothing. Recursion is fully supported. Calls use ordinary `f(x, y)` syntax; generic
functions may be called with a turbofish `f::<512, 512, 513>(a, b, c)`.

Parameters are **immutable by default**, the same rule `let` follows — a function may read one but
not reassign or mutate it. Prefix a parameter with `mut` to opt into mutation:

```wukong
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

**`wukong_` is a reserved function-name prefix.** Declaring `fn wukong_anything(...)` is a compile
error (`E0300`, "the name … is reserved for a compiler runtime kernel"): that namespace belongs to
the runtime kernels the compiler's own recognizers emit, and a user function of the same name used to
hijack kernel dispatch silently. The restriction applies to function names only — a `let`, `const`,
`struct` or `enum` may still be named `wukong_*`.

### Function-typed parameters ✅

A parameter may have a **function type**, `fn(A, B) -> R` (`fn(A)` for `()`), and every call binds it
to a function **named at compile time** — a non-generic top-level function, or the caller's own
function-typed parameter passed along:

```wukong
fn square(x: f32) -> f32 { return x * x; }
fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); }
fn twice(f: fn(f32) -> f32, x: f32) -> f32 { return apply(f, apply(f, x)); }

fn main() -> i32 {
    print(twice(square, 3.0));             // 81
    return 0;
}
```

There are no function pointers at run time: the callee is **specialized per bound function** exactly
like a type-generic function is per type (`apply$square`), the function parameter carries no value,
and a call through it is a direct call — inlinable, and visible to every kernel recognizer. A
type-generic higher-order function binds its type parameters from the function argument too
(`fn fold3<T>(op: fn(T, T) -> T, a: T, b: T, c: T) -> T`). The rules (`E0411` otherwise): a function
type is only a parameter's type, or the annotation of a `let` that binds a closure
(`let f: fn(f32) -> f32 = |x| x * k;`, see [Closures](#closures-)) — not another local's, a field's,
an element's, a return type or an `extern` parameter's; a named function cannot be bound to a `let`
(a closure can) or passed to a non-function parameter; the argument must name a function, not
compute one; and a function-typed parameter is neither `mut` nor named after a builtin (`gelu`,
`print`, …). A wrong signature is the ordinary `E0401` type mismatch. A parameter whose type
names a dimension generic of the callee (`fn mapn<N>(f: fn(Tensor[f32, N]) -> f32, t: Tensor[f32, N])`)
accepts a function of the extent the call binds: from the function itself (`fn head(t: Tensor[f32, 4])`
binds `N = 4`) or from another argument, the first binding winning and every other argument checked
against it (`E0401` at the function, `E0501` at a buffer). `tests/run/fn_params.wk`,
`tests/run/closure_infer.wk`.

### Closures ✅

A **closure literal** — `|x| x * k`, `|| 0`, `|a, b: i32| a + b`, or the block form
`|x: f32| -> f32 { … }` — may be passed wherever a function-typed parameter expects a function, or
bound to a name with `let`:

```wukong
fn apply(f: fn(f32) -> f32, x: f32) -> f32 { return f(x); }

fn main() -> i32 {
    let scale = 2.5;
    print(apply(|x| x * scale, 4.0));                                  // 10
    print(apply(|x| -> f32 { if x < 0.0 { return -x; } x }, -4.0));    // 4
    let double = |x: f32| x * 2.0;
    print(double(3.0) + apply(double, 1.0));                           // 8
    return 0;
}
```

- **Where.** As an argument of a function-typed parameter — of a call to a function, a method
  (`v.map(|a| a * k)`, `V::make(|i| ..)`), a function-typed parameter or a closure-bound local — or
  as the initializer of a `let`. A **closure-bound local** (`let f = |x: f32| x * k;`) can be called
  (`f(2.0)`) and passed to a function-typed parameter (`apply(f, 3.0)`), and nothing else: copying,
  returning, storing, printing or computing with it is `E0421`, because it has no run-time value —
  every use is resolved to the closure at compile time. The binding is one immutable name
  (`let mut`, a pattern or a reassignment `f = ..` is `E0422`); an annotation, if any, is its
  function type (`let f: fn(f32) -> f32 = |x| x * k;`). Anywhere else — returned, stored in a field
  or element, an operand, a `const` initializer, a `where` predicate or a `static_assert` — a
  closure is `E0411`: there are no run-time function values.
- **Parameters and return type.** Each parameter is a name with an optional `: Type` (no patterns,
  no `mut`). An unannotated parameter takes the type the function-typed parameter declares for it.
  When that names a generic parameter of the callee (`fn fold3<T>(f: fn(T, T) -> T, a: T, ..)`), even
  as a tensor's element type or dimension (`fn(Tensor[T, N]) -> f32`), the call's other arguments
  bind it first — the non-literal ones, then a literal at its default type — so
  `fold3(|a, b| a * 10 + b, x, y, 3)` types `a` and `b` as `x`'s type. A generic none of them binds
  needs an annotation (`E0419`), and so does every parameter of a `let`-bound closure without a
  function-type annotation; one bound to the enclosing function's own generic is `E0418` (the
  closure is not generic). The return type is the annotation, else the declared one, else the
  body's type, and a body that uses `return` then needs the annotation (`E0419`). With `->` the body
  must be a block. The closure's type must equal the parameter's once the call binds the callee's
  generics (`E0401`).
- **Captures.** Every enclosing local the body reads is captured. One it only **reads** is copied
  where the closure is created — when the call evaluates that argument (left to right with the
  other arguments), or at the `let` — so a later write to the variable is not seen: a scalar
  (including `bool` and `char`), an enum, a raw pointer or reference, a fixed-size array, a tuple or
  a struct, the last four copied whole. One it **writes** — `c = ..`, `c += ..`, `&mut c`, an element
  or field (`a[i] = ..`, `p.x = ..`), `atomic_add(a[i], ..)`, a `mut self` method, an aggregate passed
  to a `mut` parameter (`relu6(x, out)`, which writes `out` in place), or a builtin's output operand
  (`relu(x, out)`, `argsort_f32(x, idx)`, `sprintf(buf, ..)`), directly or through a tensor view
  of it (`relu(x, out as Tensor[f32, 2, 3])`; a view bound by `let` counts as a write, since it can
  be written through) — is captured **by reference**: the closure and the enclosing function share
  the variable, so the write is seen after the call (and by every later call of a `let`-bound
  closure), and the variable's own rules apply (`E0304` for an immutable binding, exactly as outside
  the closure). Writing through a captured
  pointer (`*pt = ..`) writes the pointee, and the pointer stays a copy. A local named as a
  dimension in a turbofish (`sumn::<n>(..)`, `sum::<n>(..)`) is read too. A slice or `str` (a view
  of storage its copy would not carry), a tensor or vector, the enclosing function's function-typed
  parameter, and anything naming one of its generic parameters cannot be captured (`E0418`):
  capture a pointer instead (`let ps = &s;`, then `(*ps)[i]`), and bind a generic dimension to a
  local first (`let n = N;`). A `for` counter is captured at the width it is stored at, the wider
  of its bounds' integer types (`for i in 0..n` with `n: i64` stores `i` in 8 bytes, and the
  closure reads and writes all of them), so a bound of a generic type, whose width each instance
  decides, makes the counter `E0418` too: copy it to a local first (`let k = i;`). A closure-bound
  local is captured through its closure: the inner closure receives what the outer one captured.
- **Control flow.** `return` returns from the closure; `break`/`continue` cannot cross it (`E0303`).
  Closures nest, and a name an inner closure captures is captured by every closure around it.
- **Attributes.** A closure body is code of the function it is written in, so that function's
  `@strict_fp` (no contraction: `|x| x * a + b` is a rounded multiply and a rounded add), `@generic`
  and `@unchecked` apply to it as well (`tests/run/closure_strict_fp.wk`).

It costs nothing at run time: the closure is lifted to a function whose leading parameters are its
captures (a pointer for one it writes), and the callee is specialized per closure exactly as per named
function (`apply$main$closure$0`), so a call through `f`, or through a closure-bound local, is a
direct call — no function pointer, no heap environment, and inlined like any call at `-O2`.
`tests/run/{closures,closure_let,closure_method,closure_captures,closure_capture_out_params,closure_wide_counter,closure_infer}.wk`.

### Generic functions and dtype-generic tensors ✅

A type parameter (`fn add<T>(a: T, b: T) -> T`) is bound at each call from the arguments, and the
function is compiled once per distinct binding (`add$f32`, `add$i64`). The body is type-checked with
`T` opaque, and **every instantiation is checked again at its concrete types**: `add(p, q)` with
`p, q: Point` is `E0433` at `a + b`, with a secondary label "required by this instantiation of `add`
with T = Point" on the call (`tests/fail/generic_instantiation_bad_op.wk`). Instantiations reached
through another generic are checked too (`tests/fail/generic_instantiation_transitive.wk`). An
unsuffixed literal argument takes the type the call's other arguments bind its parameter to:
`add(x_i64, 1)` and `add(1, x_i64)` are both `add$i64` (`tests/run/generic_literal_arg.wk`).

A tensor's **element** may be a type parameter, so one kernel serves every dtype:

```wukong
fn axpy<T, N>(a: T, x: Tensor[T, N], mut y: Tensor[T, N]) {
    for i in 0..N { y[i] = a * x[i] + y[i]; }
}
```

`T` is bound from a tensor argument's element (or a decaying array's scalar element), indexing a
`Tensor[T, ..]` yields a `T`, and each instance (`axpy$f32`, `axpy$bf16`) is lowered with the
concrete element in every tensor type (`tests/run/dtype_generic_tensor.wk`). An array of structs fed to a
`Tensor[T, ..]` parameter is `E0502` (a tensor element is a scalar). A named kernel builtin
(`matmul`, `softmax`, `sdpa`, …) takes `f32` buffers and a typed file-I/O builtin (`read_bf16`,
`write_f32`, `st_read_f32`, `topk_f32`, …) the one element type it names, so a `Tensor[T, ..]` operand of one is checked **per
instantiation**: the generic body is accepted, an instance that binds `T` to the builtin's element
reaches it exactly like the hand-written function (`fn mm<T>(a: Tensor[T, 4, 4], ..) { matmul(a, ..) }`
instantiated at `f32` dispatches `wukong_sgemm`, `tests/run/dtype_generic_builtin.wk`), and any other
binding is **`E0415`** (kernel builtin, `tests/fail/dtype_generic_builtin_elem.wk`) or **`E0416`**
(typed I/O builtin, `tests/fail/dtype_generic_io_elem.wk`) at the operand, labelled with the call
that required the instance. A `[]T` slice or `[T; N]` array operand of a builtin is still the plain
element mismatch inside the generic body, and the decode builtins (`sdpa_kv`, `kv_append`,
`gemv_bf16`, …) take no `Tensor[T, ..]` operand at all (`E0437`). `alloc::<T>(n)` is the generic heap
allocator — a zeroed `[]T`, the same buffer `alloc_<T>(n)` returns, with `T` a scalar or a type
parameter (anything else is `E0325`).

### Dimension constraints: `where` and `static_assert` ✅

A `where` clause may constrain a function's dimensions, alongside its trait bounds:

```wukong
const TILE: usize = 4;

fn tile_sum<K>(a: Tensor[f32, K]) -> f32 where K % TILE == 0, K <= 4096 {
    static_assert(K >= TILE, "need at least one tile");
    ...
}
```

A predicate compares expressions over integer literals, the function's generic dimensions and
`const`s (`+ - * / %` in 128-bit integers, truncating), joined with `&&` / `||`; anything else —
a local, a call such as `a.len()`, a cast — is `E0524`. Each predicate is evaluated at every call
whose dimensions it names are compile-time constants, and a false one is `E0521` at the call:
"constraint `K % 16 == 0` of `tile_sum` is not satisfied (K = 24)". `static_assert(cond)` /
`static_assert(cond, "message")` is a statement that generates no code: over `const`s it is decided
where it is written, and over the function's dimensions it is checked at each call like a `where`
predicate (and must then be a top-level statement of the body); a false one is `E0522`. A call
from a generic body whose dimensions are still symbolic is not checked there — the constraint is
checked where the dimensions become known. A `const` generic parameter is a dimension and must be
declared `usize` (`E0523`). `tests/run/where_constraints.wk`, `tests/fail/where_predicate_violated.wk`.

## Name resolution ✅

Every name a program uses resolves, or the program is rejected with a resolution code — a typo is
never accepted and left for code generation to trip over. When a close name exists the error carries
a "did you mean" suggestion, which `--error-format=json` emits as a structured replacement an editor
can apply.

| code | what does not resolve |
|---|---|
| `E0301` | a value (`does_not_exist + 1`), or an enum variant (`Color::Gren` in a value, a call or a pattern) |
| `E0320` | a called function — no `fn`, function-typed parameter or builtin has the name (`sqare(x)`: did you mean `square`?) |
| `E0321` | a field: `p.z` on a struct without `z` (the struct's fields are listed), any field of an enum value or of a builtin type (`x.re` on an `f32`), a numbered field of a non-tuple (`x.0`) |
| `E0322` | a method of a builtin type: an array, slice or tensor has `len()`, an array or tensor `dim(k)`, and nothing else; math functions are free functions — `sqrt(x)`, not `x.sqrt()` |
| `E0323` | a call of something that is not a function: a local, a constant, a struct or enum name, a type (`f32(x)` — write `x as f32`), a struct field. The usual cause is a missing `;`: a `(` that begins a line continues the expression above it |
| `E0324` | a type: not a scalar, `Tensor[..]`/`vec[..]`, a declared struct or enum, or a generic parameter in scope (`flot`; `F32` and `float` suggest `f32`); also a struct literal of an undeclared name and a trait written where a type belongs |
| `E0315` | a method or associated function of a struct or enum (see *Methods*) |

Three things stay accepted without being modelled: the explicit-SIMD surface (`f32x8::load(p)`,
`p.offset(i)`, `t.row_ptr(r)`, a vector's `store`/`fma`), which type-checks and is the codegen limit
`C0001` until it lowers; a type used as a namespace (`f32::MAX`, `T::m()` on an unbounded generic);
and a method of a generic receiver (`x.len()` on `x: T`), resolved per instantiation when the
function is lowered. In each instance, `x.len()` and `x.dim(k)` are the `i64` extent of the array,
slice or tensor that `T` is bound to, as an initializer, a `return` value, an argument or a loop
bound (`tests/run/generic_receiver_extent.wk`). A binding initialized from one has no type until
you annotate it, so to do arithmetic on it write `let k: i64 = x.len();`. Arithmetic on an
unannotated one is the codegen limit `C0001`.

## Bindings ✅

```wukong
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
`let t: (u8, u8) = (200, 1)` (`tests/run/aggregate_literal_adapt.wk`). It also descends into a
**constant binary expression** of literals, so `let v: i64 = 0 - 16` adapts like the unary `-16`
already did (`tests/run/binary_const_adapt.wk`); the folded value is still range-checked, so
`let v: i8 = 100 + 100` is rejected. A typed value must match its
annotation exactly (see error `E0401`) — and for an aggregate that means the shape too: an array
literal and an array **repeat** initializer must both supply exactly the annotated length, so
`let a: [i32; 4] = [7; 2];` is `E0401`, not a slot silently filled out to 4. A repeat `[v; n]` evaluates `v` **once** and copies it into every slot, for an aggregate element too: `[make(); 3]` calls `make` once (`tests/run/array_repeat_aggregate_once.wk`). An unsuffixed literal
that does not fit the type it adapts to
is rejected (`E0431`, e.g. `let x: i8 = 200;`, or `s.x = 9000000000;` for an `i32` field), not
silently wrapped. With **no** annotation an unsuffixed integer literal defaults to `i32`, but one that
does not fit widens to the narrowest type that holds it — `i64`, and then `u64` for a magnitude in
`(i64::MAX, u64::MAX]` — so its value is never silently truncated
(`tests/run/int_literal_widen.wk`); a value past `u64` is malformed (`E0431`). Use a suffix
(`9000000000i64`, `3000000000u32`) to pick a specific type.

**Literal inference** (✅): a `let` with no annotation whose initializer is only unsuffixed literals
(`let mut s = 0.0;`, `let k = 2;`) is an `{float}`/`{integer}` binding, like Rust's — its type is
decided by its **first constraining use**, and only that one: an assignment `s = v`, a `return s`,
a call argument `f(s)` — also a method-call argument `p.m(s)` / `Type::m(s)` (never the receiver)
and a call through a function-typed parameter — a struct-literal field `S { f: s }` (or a
struct-payload variant's field), or a `let x: T = s` makes it that type (same kind, any width), so
`let x = 0.1; S { w: x }` with `w: f64` stores the f64 0.1 (`tests/run/literal_infer_struct_field.wk`,
`tests/run/literal_infer_method_arg.wk`; these positions were `E0401` before); a compound
assignment (`s += v`), an arithmetic operator or a comparison with a typed operand of the same kind
makes it that type when it is at least as wide, and otherwise keeps the default. So
`let mut s = 0.0; for i in 0..n { s += x[i]; }` over `f64` data accumulates in `f64`
(`tests/run/literal_infer_f64_accum.wk`), and `let k = 3; x * k` with `x: u32` is a `u32` product.
An integer literal never becomes a float, a narrower operand never narrows the binding (`let k = 2;
k + x_u8` stays an `i32` sum), a later use never re-decides what an earlier one decided (in
`let k = 2; g_i32(k); x_i64 * k` the call keeps `k` an `i32`), and a binding nothing constrains keeps
the literal's default.

A `let` with **neither an annotation nor an initializer** (`let mut x; x = 2.75;`) is not inferred
from its first assignment yet: the program type-checks, and lowering refuses it (`C0001`,
`tests/fail/lower_unclassified_type.wk`) — annotate it, `let mut x: f32;`. It used to get a silent
`i32` slot, so that program printed `2`-based results. More generally, lowering sizes a value's
storage only from a type it can classify; one it cannot (a value the type checker left undetermined,
or a type parameter no call binds — see Traits) is refused as `C0001`, never given a guessed `i32`
slot. An integer written where a type belongs (`let x: 512 = 3;`) is `E0324`, and a struct, enum or
type parameter given type arguments (`let p: P<2>`) is `E0327`; both used to check clean.

A `let` binding may **destructure a tuple** — `let (a, b) = …`, nested `let ((m, n), o) = …`, or a
wildcard `let (keep, _) = …`, including the result of a tuple-returning call
(`tests/run/let_destructure.wk`). A top-level **`const` is usable as a value**: its initializer is
inlined at every use site — in arithmetic, as an array index, as a loop bound, as an **array length**
in a type (`let a: [i32; N]`, including a const-references-const chain;
`tests/run/const_array_length.wk`), and when one `const` references another
(`tests/run/top_level_const.wk`). An array length is an integer literal (any radix, with `_`
separators and an optional suffix — `[i32; 0x10]` is 16 elements) or a top-level `const`; simple
arithmetic on those folds too (`[i32; 2 + 2]`). A **generic** length is the shape of a
**parameter**: `fn sum<N>(a: [f32; N])` (or `[f32; 2*N]`) takes a shape-generic buffer — it is the
`Tensor[f32, N]` it denotes, so each call binds `N` from its argument's length and the body indexes
it, iterates it, and reads `a.len()` (`tests/run/generic_array_param.wk`). A **local** array or a
**return** type of generic length would need storage whose size is only known at run time, and is
`E0406` (`tests/fail/generic_array_length.wk`); so is a generic length whose element is not a scalar.
Any other name is `E0301`, not a silent length of 0, and a fixed-size array may hold at most
`u32::MAX` elements (`E0401` past that).

## Literals ✅

Integer literals may be **decimal, hex `0xFF`, octal `0o17`, or binary `0b1010`**, with `_` digit
separators (`1_000_000`) and an optional type suffix (`250u8`) (`tests/run/radix_literals.wk`). A
malformed literal — a mistyped radix like `0z123`, an empty `0x`, a bad digit `0b2`, a garbled float
`1.5z`, or a value past `u64` — is a compile error (`E0431`), never silently zeroed
(`tests/fail/malformed_int_literal.wk`). A **float literal** takes `_` separators and an optional
exponent, and its suffix may be `f32`, `f64`, `f16`, `bf16`, or the bare C-style `f` (= `f32`);
unsuffixed it defaults to **`f32`** (`5f`, `1.5`, `1_000.5e-3`, `0.5bf16`). A malformed one — `1.5z`,
an incomplete exponent `1.5e`, a doubled suffix `5ff` — is `E0431`, never silently `0.0`. A
**char literal** `'A'` has type **`char`** (a 32-bit Unicode scalar value) — covering the
one-character escapes (`\n` `\r` `\t` `\\` `\'` `\"` `\0`), `\xHH` hex, and `\u{…}` Unicode escapes. A
char literal holds **exactly one** Unicode scalar value, and the compiler enforces it: an empty `''`
and a multi-codepoint `'ab'` are `E0104`, and a `\u{…}` above `\u{10FFFF}` or inside the surrogate
range `\u{D800}..=\u{DFFF}` is `E0432` (in a string literal too) — none of these is silently decoded
to `0`. `char` is a usable annotated type (`let c: char = 'A'`) and is interconvertible with the
integer types via `as` in both directions, so it can be cast, compared, and used in arithmetic
(`tests/run/char_literals.wk`, `tests/run/char_type.wk`).

A **string literal** `"hello"` is typed `*u8` — the same by-pointer convention as an array. Each
unique literal is interned once into a read-only **`.rodata`** static blob (deduplicated by content,
plus a trailing NUL) and its value is that blob's address (`Op::GlobalAddr`). The escapes `\n` `\r`
`\t` `\\` `\"` `\'` `\0` `\xHH` `\u{…}` decode (each code point re-encoded as UTF-8). `print`/`println`
of a `*u8` — a literal or a `let s = "hi";` binding — renders the bytes, while numeric `print` still
prints numbers (`tests/run/string_literal.wk`). Because the blob lives in static data (not the
stack frame), a `*u8` can be **returned from a function and threaded across calls** without dangling
(`tests/run/string_return.wk`). Unannotated, a literal is that NUL-terminated `*u8` — the type
`printf`, file paths and `extern "C"` take. Where a **`str`** is expected, the same literal is a
`str` instead: a length-carrying, bounds-checked, immutable view of the same bytes (see
[Strings: `str`](#strings-str-)).

## Types

| Category    | Examples                                            | Status |
|-------------|-----------------------------------------------------|--------|
| Integers    | `i8 i16 i32 i64`, `u8 u16 u32 u64`, `usize isize` — no 128-bit type (a `u128`/`i128` literal suffix is `E0431`) | ✅     |
| Floats      | `f16 bf16 f32 f64`                                   | ✅ scalar |
| FP8         | E4M3FN / E5M2 codes stored as `u8`, converted with the [scalar builtins](#scalar-builtins-) (`f8e4m3_to_f32`, `f32_to_f8e4m3`, …); no first-class `f8` type | 🟡 storage |
| Boolean     | `bool`                                               | ✅     |
| Pointers    | `*T`, `*mut T`, deref `*p`, address-of `&x`/`&mut x`; `&T`/`&mut T` accepted as a parameter type | ✅ |
| Arrays      | fixed-size `[T; N]` (literal/repeat init, indexed load/store) | ✅ |
| Tuples      | `(A, B, …)`, field access `t.0`, nested `t.0.1`       | ✅     |
| Structs     | `struct S { … }`, literal `S { f: v }`, field `s.f`  | ✅     |
| Enums       | C-style `enum E { A = 10, B }` (variant = its `i32` discriminant) **and data-carrying** `V(i32)` / `V { f: T }` variants | ✅ |
| Slices      | `[]T` — fat pointer `{data, len}`; `.len()`, **bounds-checked** indexed load/store, iteration, array→slice unsizing, range indexing `s[a..b]` | ✅ |
| Strings     | `str` — an immutable UTF-8 byte string, a `[]u8`-shaped fat pointer; literals, `.len()`, bounds-checked byte indexing, `s[a..b]`, `==`/`!=`, printing ([below](#strings-str-)) | ✅ |
| SIMD vectors| `f32x4`/`i32x4`, generic `vec[T, N]` — the type **parses and type-checks** (a non-scalar element is `E0302`, a lane/element mismatch at a call is `E0401`), but no vector value can be constructed in source (`f32x4::load(p)` is `C0001`) and float-vector arithmetic is an internal error, so nothing uses it. SIMD ships through the **automatic** 128-bit/256-bit loop vectorizer, which needs no vector type in source. | 🔵 |
| Tensors     | `Tensor[f32, M, N]` — shape checking, and **indexing** (`a[i, j]`) for both const and symbolic-generic dims, on the default `contiguous` layout. A constant-shape tensor costs **nothing** over the hand-flattened `[f32; M*N]` + `a[i*N + j]` spelling: the two compile to byte-identical MIR, same kernel dispatch and same vectorization. `.col_major` indexing executes for constant-shape tensors (`a[i, j]` is byte-identical to `a[i + j*M]`). `.strided`/`.tiled(…)` type-check but cannot be indexed (`C0001`). Whole-tensor arithmetic (`a + b`) is `E0433` — operate elementwise. | ✅ |

`&x` and `&mut x` have type `*T` and `*mut T`. A `&T`/`&mut T` **parameter** accepts them
(`fn deref(p: &i32) -> i32 { return *p; }` with `deref(&x)`), but the two spellings are distinct types
elsewhere, so a local must be annotated with the pointer form — `let p: *i32 = &x;`, not
`let p: &i32 = &x;` (that is `E0401`).

`0 as *T` is the null pointer, and a pointer used as a condition (`if p`, `while p`, `p && q`) is a
null test — `p != null` — with the same answer on every backend and at every `-O` level (✅). The
interpreter reserves its address 0 as a never-allocated null so it agrees with native here, and a
load or store through null is the interpreter diagnostic `null pointer dereference` (native faults).
A condition must be a boolean, a number (nonzero is true) or a pointer: an aggregate, slice, tensor
view, SIMD vector or function value is `E0436`.

### Strings: `str` ✅

`str` is an **immutable UTF-8 byte string**: a 16-byte fat pointer `(data, byte length)` — the
layout of a `[]u8` slice — over a string literal's read-only `.rodata` bytes, or over part of
another `str`. It passes and returns like a `[]T`, and can be a struct field, a tuple field or an
array element (`tests/run/str_basics.wk`).

```wukong
const GREETING: str = "hello";
struct User { name: str }

fn count_byte(s: str, target: u8) -> i64 {
    let mut n: i64 = 0;
    for b in s {                    // each byte, as a u8
        if b == target { n = n + 1; }
    }
    return n;
}

let s: str = "wukong";
println(s);                         // wukong
println(s.len());                   // 6: bytes, not characters
println(s[1]);                      // 117: a u8, bounds-checked
println(s[1..4]);                   // uko: a str viewing the same bytes
println(s == "wukong");             // true: length, then bytes
printf("[{}|{:?}]\n", s, s[..2]);   // [wukong|"wu"]
```

- **Literals.** A string literal is a `str` wherever the context expects one: exactly where a
  numeric literal adapts to its annotation (a `let`/`const` annotation, a call or method argument,
  a `return` of either spelling, a struct field, an assignment, an array or tuple element), as an
  `==`/`!=` operand against a `str`, and as the base of a range index. In an array literal it
  becomes a `str` beside one on either side (`[s, "y"]` and `["x", s]` are both `[str; 2]`,
  `tests/run/str_array_elems.wk`): every element of an array literal has the one element type, so
  a `*u8` value or a `[]u8` beside a `str` is `E0401`. Unannotated, a literal is still the
  NUL-terminated `*u8` (`let p = "x";`). As with numbers, an `if`/`match` whose arms are all
  literals takes its type from those arms (`*u8`), and annotating the `let` does not change that
  (`let s: str = if c { "a" } else { "b" };` is `E0401`). Make the FIRST arm a `str` and the later
  literal arms adapt to it (`let yes: str = "yes";` then `if c { yes } else { "no" }`), or `return`
  each literal from a function that returns `str`.
- **Bytes.** `s.len()` is the byte length, an `i64`. `s[i]` is a `u8`, bounds-checked exactly like
  a slice element (the same `index out of bounds` failure, exit 1). `for b in s` yields each byte.
  The bytes are UTF-8 (`"h\u{e9}llo"` is 6 bytes long), and nothing decodes characters yet.
- **Range indexing.** `s[a..b]`, `s[a..]`, `s[..b]` and `s[..]` are `str` views of the half-open
  byte range — no copy. The same range index works on any slice `[]T` and gives a `[]T`
  (`tests/run/slice_range_index.wk`). The bounds are integers, checked at run time as
  `a <= b <= len` (`tests/run/str_slice_bounds.wk`) unless the function is `@unchecked`. `..=`, and
  a range mixed with other indices (`t[0, 1..2]`), are `E0202`; a range index of an array is
  `E0437` (bind it to a `[]T` first); a bound that is not an integer is `E0446`.
- **Equality.** `==` and `!=` between two `str`s compare the lengths, then the bytes. `<`,
  arithmetic, and a `str` compared with anything but a `str` are `E0433`.
- **Printing.** `print`, `println`, `print_raw`, `eprint`, `eprintln` and the `printf` / `eprintf` /
  `sprintf` placeholders print the text, and `{:?}` quotes it, as for a `*u8`. A `str` inside a
  printed aggregate (a struct, tuple or array holding one) is `E0448` for now: print the field.
- **Immutable.** A `str`'s bytes are never a place: `s[i] = b`, `s[i] += b`,
  `atomic_add(s[i], v)` and `&mut s[i]` are `E0304`. A `mut` binding can still be re-pointed (`s = s[1..];`).
- **No conversions.** A `str` converts to and from nothing: not `[]u8`, not `*u8`, not by `as`
  (`E0401`). There is no concatenation yet.
- **Not generic.** A `str` cannot bind a type parameter (`id(s)` with `fn id<T>(x: T) -> T` is
  `E0420`, `tests/fail/str_generic_param.wk`); write the function for `str` itself. A struct, enum,
  trait, type alias or generic parameter named `str` is `E0300`, in an imported module too.

### Type aliases ✅

`type Name = T;` gives a type a name; a generic alias takes type, dimension or `const`
parameters. An alias is **transparent**: every use is replaced by the type it stands for before
type checking, so a function over `Mat<M, N>` is exactly the function over `Tensor[f32, M, N]` —
same shape checking, same MIR, same kernel dispatch.

```wukong
type Real = f32;
type Mat<M, N> = Tensor[Real, M, N];
type Buf<T, N> = [T; N];

fn matvec<M, N>(a: Mat<M, N>, x: Tensor[Real, N], mut y: Tensor[Real, M]) { … }
fn interleave<N>(a: Tensor[Real, N], b: Tensor[Real, N], mut out: Tensor[Real, N * 2]) { … }

let q: Buf<Buf<Real, 2>, 3> = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.5]];
let h = alloc::<Real>(n);
let r = x as Real;
```

- An alias may be used wherever a type is written: parameters, returns, `let` annotations, struct
  and variant fields, casts, `alloc::<T>` and other turbofish arguments, and nested inside other
  types (including other aliases). It does **not** name a struct literal, a pattern or an `impl`
  block — write the struct's own name there.
- Arguments go in `<..>`: a type, an integer, `?`, or dimension arithmetic (`Row<N * 2>`); a bare
  name serves as either a type or a dimension. `>>` closes two lists. After `as`, a bare `<` is the
  comparison (`x as i32 < n`), so a generic alias as a cast target is spelled `x as Mat::<2, 2>`.
- `type` is a contextual keyword — only `type Name` at item position starts an alias, and `type`
  stays an ordinary identifier elsewhere.
- Aliases follow the module rules of every other item: `pub type` is importable (`import
  lib.shapes;`, `import lib.shapes.{Mat}`, `shapes.Mat`), a private alias is `E0312` to an
  importer, and a generic parameter of the enclosing function shadows an alias of the same name.
- Errors: an alias that reaches itself (`type A = [B; 2]; type B = A;`) is `E0326`; a wrong number
  or kind of arguments (`Mat<4>`, `f32<2>`, a type where the alias uses a dimension) is `E0327`; an
  alias body naming no type is `E0328`, reported even if the alias is unused; an alias named like a
  built-in scalar, `str`, or an existing struct/enum is `E0300`.

## Operators ✅

Arithmetic `+ - * / %`, comparison `== != < <= > >=`, bitwise `& | ^`, shifts `<< >>`, boolean
`&& ||` (short-circuit), unary `-`, `!`, and `~`. Like Rust, `!`/`~` are the same operator —
bitwise complement on an integer, logical negation on a `bool`; `~` is the conventional integer
spelling. Precedence follows **Rust's** table, resolved by a Pratt parser: `||` < `&&` < comparisons
< `|` < `^` < `&` < `<<`/`>>` < `+`/`-` < `*`/`/`/`%`, all left-associative. Note this differs from C
for the bitwise operators — `1 | 2 == 2` is `(1 | 2) == 2` here, not `1 | (2 == 2)`. A **chained
comparison** is rejected rather than silently mis-parsed: `a < b < c` would parse as `(a < b) < c`, so
it is an `E0433` that names the fix (`a < b && b < c`); the equality form `a == b == c` is caught the
same way whenever the third operand is not itself a `bool` (an all-`bool` chain still parses as
`(a == b) == c`). Compound assignment (`+=`, `*=`, `<<=`, `>>=`, …) is supported, under the plain
assignment's rule: `x op= v` stores `x op v`, which must be of `x`'s own type, so `v` must be of that
type, promote into it losslessly (`i64 += i32`, `f64 += f32`), or be an unsuffixed literal (which
must fit an integer place); `f32 += f64`, `i32 += i64` and `i32 += f32` are `E0441` — they used to
convert the operand down to the place and accumulate in the narrower type
(`tests/fail/compound_assign_narrowing.wk`). A shift amount (`<<=`, `>>=`) may be any integer type.

**Mixed-type arithmetic** (✅) promotes only where nothing is lost and the operand order cannot
matter: two integers of one signedness compute in the wider (`i32 + i64` is `i64`); an unsigned
integer and a *strictly wider* signed one compute in the signed type (`u32 + i64` is `i64`, so
`u32 / i64` is a signed divide: `7 / -2 == -3`); `f16`/`bf16` widen to `f32` and `f32` to `f64`;
an integer meets a float only when the float holds every value of it exactly (8- and 16-bit
integers with `f32`, up to 32-bit integers with `f64`). A signed value known never to be negative —
a `for` counter that starts at a non-negative literal and is never assigned, or a non-negative
`const` — also converts to an unsigned type at least as wide, so `for i in 0..m { … i * n … }` with
`n: usize` needs no cast. Every other mix — `i32` with `u32`, `i64` with `f32`, `f16` with `bf16`,
`char` with an integer — is `E0440`; write the `as` that says which type you mean. An unsuffixed
literal operand adapts to the other operand and never needs a cast (`x * 2`, `2.0 * x`), and with
an equally wide typed operand it takes that operand's type on either side (`1 + x` is `u32` for
`x: u32`). `/` and `%` divide with the signedness of that common type. A shift amount is a count
and may be any integer type (`tests/run/mixed_width_div_signedness.wk`,
`tests/fail/mixed_sign_arith.wk`, `tests/fail/int_float_arith.wk`).

Integer arithmetic wraps at the declared width. Division and remainder by zero return zero;
signed minimum divided by `-1` wraps to signed minimum, with remainder zero. Shift counts are
masked to the operand width: `1i32 << 32` is `1`, and `1i32 << 33` is `2`. Right shift fills
with the sign bit for signed integers and with zero for unsigned integers. These are the
interpreter and native backend contracts, including optimized execution.

Float-to-integer `as` casts truncate toward zero and saturate at the destination range; NaN
becomes zero, and negative values cast to unsigned integers become zero. `f64` is double
precision throughout: arithmetic, and every transcendental on an `f64` operand — `exp`, `log`,
`exp2`/`log2`/`exp10`/`log10`, `expm1`/`log1p`, `sin`/`cos`/`tan`, `asin`/`acos`/`atan`/`atan2`,
`sinh`/`cosh`/`tanh`, `asinh`/`acosh`/`atanh`, `erf`, `cbrt`, `pow`, `hypot`, and the activations
built on them — is one call into the runtime's pure-Rust double-precision library
(`wukong_runtime::math64`: the fdlibm/musl algorithms, no platform libm, no FMA), so the
interpreter, the native JIT and the AOT executable return the same bits on every host. The core
(`exp`, `log`, the base-2/base-10 family, `expm1`, `log1p`, `pow`, `hypot`, `sin`/`cos`/`tan` on
**every** finite argument via Payne–Hanek reduction, the inverse trig, `erf`, `cbrt`) is within
**1 ULP** of the correctly rounded value; the hyperbolics `sinh`/`cosh`/`tanh`/`asinh`/`acosh`/
`atanh` within 2 ULP; `exp10` within 3. The C23 special values hold (`exp(−∞) = +0`,
`log(±0) = −∞`, `sin(±∞) = NaN`, `atan2`'s full quadrant table, `pow(x, ±0) = pow(1, y) = 1` even
for NaN, odd functions return `−0` for `−0`, and every function is exactly `x` below its linear
threshold, sign included). `tests/run/transcendental_f64.wk` prints twelve significant digits of
each.

## Control flow ✅

```wukong
if cond { ... } else { ... }
while cond { ... }
for i in 0..n { ... }
for i in 0..n step 2 { ... }   // strided range; empty if lo >= hi
for i in 0..=n { ... }          // inclusive range
for x in arr { ... }            // iterate an array, slice, or alloc_* buffer by element
for (i, x) in pairs { ... }     // ✅ destructure each element of an array or slice
loop { ... }                    // ✅ infinite loop; exit with `break`
break; continue;                // ✅ innermost loop
'outer: for i in 0..n {         // ✅ a loop label
    for j in 0..n { break 'outer; continue 'outer; }  // target an outer loop by name
}
return expr;
```

Blocks are expressions: the trailing expression of a block (no semicolon) is its value. A
`break`/`continue` with no enclosing loop is a compile error (`E0303`). A `for` iterates either a
range (`..` or `..=`, with an optional `step`) or any iterable value — a fixed-size array, a slice, or
a heap buffer — binding each element.

**Destructuring `for` heads** (✅): over an array or a slice, the loop pattern may destructure each
element — `for (i, x) in pairs`, `for Point { x, y } in pts`, `for [a, .., z] in rows`,
`for ((a, b), c) in nest`, `for w @ (a, _) in ts`. Each element binds exactly as
`let <pat> = a[i];` would: a scalar part is a fresh copy (assigning to it never writes back), an
aggregate part is bound in place. The pattern is never tested, so it must be irrefutable by its
form — names, `_`, tuples, structs (with or without `..`), array patterns, `name @ sub` and the
variant of a one-variant enum. A literal, range, path, or-pattern or multi-variant enum variant
anywhere inside is `E0442` (`for (0, x) in pairs`; match inside the body instead), and a range
`for` or any iterand other than an array or slice cannot destructure (`E0401`)
(`tests/run/for_destructure.wk`, `tests/fail/for_head_refutable.wk`,
`tests/fail/for_range_destructure.wk`).

A **loop label** `'name:` on a `loop`/`while`/`for` lets a nested `break 'name` / `continue 'name`
target that named outer loop instead of the innermost one — the lexer tells a label `'outer` from a
char literal `'a'` exactly as Rust does (`tests/run/labeled_loop.wk`). A labeled `break`/`continue`
naming an **undeclared** label is rejected with `E0303` (`tests/fail/break_unknown_label.wk`).

`loop` is a **value-producing expression** and `break` carries a value (`let x = loop { break 5; };` ✅).
The loop's type is inferred by unifying every `break <value>` (composing with `if`/`match` value merges),
so a value `loop` can be a `let` initializer, call argument, array element, block tail / `return` value,
or aggregate field; a labeled `break 'outer v` carries a value out of an outer loop. A `break <value>`
targeting a *statement-position* loop (its value discarded, like `while`/`for`) is rejected `E0401`, and
breaks whose shapes disagree are `E0502` — the same merge rules as `if`/`match` arms.

## Pattern matching ✅

```wukong
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

`match` evaluates its scrutinee once and lowers to an if-else chain over the arms (at `-O2`, a run
of four or more equality tests on one integer scrutinee becomes a single `switch`, which the native
backend emits as a jump table or a search tree; the semantics are the chain's). A guard runs only
after its arm's pattern matched, so it may call functions with effects and read the arm's bindings
(`tests/run/match_guard_order.wk`). A match the compiler **proves exhaustive** takes its last
unguarded arm without testing it: a value that failed every arm before it can only match that one.
An exhaustive `bool` match is one branch and an N-variant enum dispatch N − 1 tests, identically at
every `-O` level and on every backend (`tests/run/match_exhaustive_fallthrough.wk`). Patterns are
integer/bool **literals**, **or-patterns** `A | B | C`, half-open `lo..hi` / inclusive `lo..=hi`
**ranges** whose bounds are integer or char **literals** (a `const` bound such as `0..N` parses but
does not lower — it reports `C0001` at the pattern; inline the literal), **enum-variant** patterns
`Color::Red` (matched by discriminant), **tuple** patterns
`(0, _) => …` (each field tested and bound, nesting allowed — and these compose, e.g. `(0 | 1, y)`),
an **identifier** binding (binds the scrutinee or field), and the wildcard `_`. Any arm may carry an
optional `if` guard, and `match` works in both value and statement position. See
`tests/run/{match_expr,match_patterns,match_tuple}.wk`.

```wukong
struct Point { x: i32, y: i32 }

fn describe(p: Point, n: i32, a: [i32; 4]) -> i32 {
    let q = match p {
        Point { x: 0, y: 0 } => 0,   // struct pattern: listed fields are tested…
        Point { x: 0, .. } => 1,     // …unlisted ones are free (`..` says so)
        Point { x, y } => x + y,     // shorthand binds `x` and `y`
    };
    let r = match n {
        d @ 0..=9 => d,              // `name @ sub`: bind the whole value `sub` tested
        _ => 10,
    };
    let s = match a {
        [first, .., last] => first + last,   // array pattern on a `[T; N]`
    };
    return q + r + s;
}
```

Three more pattern forms (✅): a **struct** pattern `Point { x: 0, y, .. }` on a value of that
struct (an unlisted field is unconstrained), a **binding** `name @ sub` that binds the whole value
`sub` matches (also nested: `(a @ 1..=5, b)`), and an **array** pattern `[a, b, .., z]` on a
fixed-size array — without `..` it lists every element. All three also destructure in a `let`
(`let Point { x, y } = p;`, `let [first, .., last] = a;`). An array pattern on a `[]T` slice is
`E0401`: its length is a run-time value, so test `s.len()` and index instead. A struct pattern names
its struct and a data-carrying variant pattern (`Shape::Circle(r)`) its enum, and the value matched
must be of that type — `E0401` otherwise, in a `match`, a `let` and a `for` head alike
(`tests/fail/for_head_variant_mismatch.wk`).

**`if let`, `while let` and `let … else`** (✅) are what they are in Rust, and the parser rewrites
each into a `match`:

```wukong
if let Shape::Circle(r) = s { use(r); } else { other(); }   // match s { P => …, _ => … }
while let 1..=100 = k { k = k - 7; }                         // loop { match k { P => …, _ => break } }
let Shape::Circle(r) = s else { return 0; };                 // `r` is bound after this line
if let Shape::Circle(r) = s && r > 2 && r < 9 { use(r); }    // match s { P if r > 2 && r < 9 => …, _ => … }
while let 1..=9 = k && k != 5 { k = k + 1; }                 // the guard joins the loop's arm
```

An **`&&` chain** after an `if let` / `while let` scrutinee (✅) becomes the guard of the pattern's
arm: the conditions run only after the pattern matched, left to right with short-circuit, and see
its bindings — Rust's let chains with one `let`. The scrutinee therefore stops before a top-level
`&&` (`if let true = a == b && c` tests `a == b`, then `c`); a second `let` in the chain, or a `||`
after the condition, is `E0200` (group it: `&& (a || b)`; `tests/run/if_let_chain.wk`,
`tests/fail/if_let_chain_or.wk`). To match a whole `&&` or `||` expression, parenthesize it:
`if let false = (a && b)` tests `a && b`, while `if let false = a && b` tests `a` and then
requires `b` — the same result only for a `true` pattern.

The `else` of a `let … else` must **diverge** — `return`, `break`, `continue`, or a `loop` with no
exit — because the names its pattern binds do not exist on that path (`E0401` otherwise). A plain
`let` pattern, in turn, must be **irrefutable**: `let Shape::Circle(r) = s;` would leave `r` unbound
for every other shape, so it is `E0442`, with a witness naming a value it misses
(`tests/run/patterns_struct_binding_slice.wk`, `tests/fail/let_refutable_pattern.wk`,
`tests/fail/let_else_not_diverging.wk`). Not yet: a `rest @ ..` sub-array binding, a binding
inside an or-pattern (`C0001` at the pattern), an array pattern on a `[]T` slice, and more than one
`let` in an `if let` chain.

A literal pattern must be **in range for the scrutinee's type**: `match n { 200 => … }` against an
`i8` is an `E0431` ("literal `200` is out of range for `i8` (-128..=127)"), not a value truncated to
the scrutinee's width. The same check applies to both bounds of a range pattern.

A `match` must be **exhaustive**, like Rust — and in **every** position, value *and* statement (an
empty `match x {}` included). Exhaustiveness is decided by **pattern usefulness** (Maranget's
algorithm, as in rustc), which looks *inside* patterns: an enum variant's payload sub-patterns, the
fields of a tuple, and integer and `char` ranges split over the type's whole domain. So
`Expr::Add(0, y)` covers only the `Add` values whose first field is 0, `(true, _)` plus
`(false, true)` leaves `(false, false)` uncovered, and `0..=99 => …, 100..=255 => …` covers a `u8`
with no `_` (`tests/run/match_u8_full_range.wk`). An `enum` may also be covered by integer-literal
and range patterns that match its declared discriminants. A provably-incomplete match is rejected at
compile time (`E0405`) with a **witness** — a pattern naming values no arm covers:

```text
error[E0405]: non-exhaustive `match`: pattern `Expr::Add(i32::MIN..=-1 | 1..=i32::MAX, _)` not covered
```

A guard (`if …`) never counts toward coverage, because it may be false. A `char` can hold any 32-bit
value (an integer `as char` is unchecked), so a `char` match needs a `_` arm; struct and array
patterns are checked field by field, and floats, slices and pointers are covered only by a `_` or a
binding. This closes a silent-wrong-answer hole — a
non-exhaustive value match used to fall through to a zero default, and a statement/empty one to an
`Unreachable` the two backends handled differently (`tests/fail/match_nonexhaustive.wk`,
`match_payload_subpattern_nonexhaustive.wk`, `match_tuple_nonexhaustive.wk`). The same analysis
reports an arm no value can reach — every value it matches is matched by an earlier unguarded arm —
as the warning `W0020` (`unreachable_patterns`).

The proof that lets the last arm go untested covers only values the analysis models. One value
escapes it: a `let e: E;` with no initializer is zero-filled, so an `enum E { A = 1, B = 2 }` local
holds discriminant 0, which no variant has. A match over a type whose all-zero value is not one the
analysis models (such an enum, or a struct or tuple holding one) is therefore not *proven*: it keeps
every test, and the forged value takes no arm (a value `match` traps there, as before).

## Tuples and structs ✅

```wukong
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

A struct literal must list **every** field (in any order). Rust's functional-update shorthand
`S { x: 1, ..base }` is **not supported** and is rejected with a single `E0201` that says so — it is
never partially accepted, so no field is left uninitialized.

Tuples and structs lower to a flat, padded byte buffer (the local's value *is* its base pointer, the
same convention arrays follow); field access is a typed load/store at the field's byte offset, and a
field that is itself a tuple is reached by chaining — `t.0.1`, `t.0.0.0` (`tests/run/nested_tuple_field.wk`).
**Nested aggregates** work too: a struct/tuple field that is itself a struct (any depth), and arrays
of structs, lay out recursively, and an aggregate field initialized from a non-literal value is
deep-copied leaf by leaf (`tests/run/struct_nested.wk`). Whole-aggregate **assignment** (`s = other;`)
deep-copies leaf by leaf as well (`tests/run/struct_assign.wk`). Both run identically on the
interpreter and the native backend. An aggregate passes **into a function by reference** (its base
pointer, zero-copy) and is **returned by value** through a hidden-pointer (sret) ABI in mir_build, so
no aggregate ever rides in a register and the two backends agree (`tests/run/{struct_fn,struct_return}.wk`).
Because a by-reference parameter aliases the caller's storage, mutating an aggregate parameter
requires `mut` on it (see *Functions* above) — a non-`mut` aggregate parameter is effectively
read-only, and a `mut` one is the in-place output buffer a kernel writes. Passing one buffer to two
parameters (`mm(x, y, x)`) is legal and means exactly what the function's loops say when they run
over the shared storage; a loop the compiler replaces with a runtime kernel keeps that meaning — the
kernel runs only when the operands do not overlap (or overlap exactly, where the kernel supports
in-place operation), and the loop as written runs otherwise (`tests/run/alias_versioned_kernels.wk`).

## Methods and `impl` blocks ✅

```wukong
struct Vec2 { x: f32, y: f32 }

impl Vec2 {
    fn new(x: f32, y: f32) -> Self { return Self { x: x, y: y }; }   // associated function
    fn dot(self, o: Vec2) -> f32 { return self.x * o.x + self.y * o.y; }
    fn scale(&mut self, k: f32) { self.x = self.x * k; self.y = self.y * k; }
}

fn main() -> i32 {
    let mut v = Vec2::new(3.0, 4.0);
    v.scale(2.0);                          // mutates `v` in place
    print(v.dot(v));                       // 100
    return 0;
}
```

An `impl` block adds functions to a struct or an enum. A function whose first parameter is the
receiver is a **method**, called `x.m(..)`; one without is an **associated function**, called
`Type::m(..)`. Inside the block `Self` names the type. Receivers follow the parameter rules above —
an aggregate is passed by reference, so `self` and `&self` read the caller's value and `mut self` /
`&mut self` may write it in place — and a receiver reached through a reference or pointer
(`p.dot(q)` with `p: &Vec2`) is dereferenced automatically. Dispatch is **static**: each method is
the ordinary function `Type.method`, and `x.m(a)` is exactly the call `Type.m(x, a)`, so methods
inline, specialize and alias-check like any call. A misspelled method, an associated function called
with method syntax, or an `impl` of something that is not a struct or enum is `E0315`.
`tests/run/methods_impl.wk`.

## Traits ✅

```wukong
trait Shape {
    fn area(self) -> f32;
    fn unit() -> Self;
}

impl Shape for Rect {
    fn area(self) -> f32 { return self.w * self.h; }
    fn unit() -> Self { return Rect { w: 1.0, h: 1.0 }; }
}

fn total<T: Shape>(a: T, b: T) -> f32 { return a.area() + b.area(); }   // or `where T: Shape`
```

A trait is a set of method signatures (`Self` is the implementing type). `impl Trait for Type` must
define exactly those methods with exactly those signatures — a missing, extra or mismatched method is
`E0410`, and so is a default method body, which traits do not have. A type parameter bounded by
traits (`<T: Shape + Named>`, or a `where` clause) may call their methods on its values (`a.area()`)
and their associated functions on itself (`T::unit()`, when `T` is the type of a parameter). Dispatch
is **static, by monomorphization**: each call instantiates `total` at a concrete type and the
instance calls that type's `impl` directly — no vtable, no indirect call — so a call at a type with
no `impl Shape` is rejected at the call (`E0410`). A bounded generic body is checked against its
bounds whether or not anything calls it: a method its bounds do not provide is `E0315`, and passing
its parameter on to a generic that needs a bound it lacks is `E0410`. Operations on an **unbounded**
`T` (`x + x`) are checked per instantiation, as a C++ template's are: an uncalled body is not
rejected for them, and a call that instantiates one at a type without the operation is an error in
the body labelled "required by this instantiation". A type parameter that appears in no parameter
or return type cannot be bound by a call (a turbofish does not bind it either), so a value of it in
the body is refused when lowered (`C0001`). Not supported yet: generic traits or trait
methods, default methods, two traits on one type that define the same method name (both are the
function `Type.method`), and generic structs/enums (`Buf<T>`). `tests/run/traits_static_dispatch.wk`.

## Enums ✅

```wukong
enum Code { Ok = 10, Err = 20 }
enum Color { Red, Green, Blue }   // 0, 1, 2 (auto-increment from 0)
enum Step { A = 5, B, C }         // 5, 6, 7 (continue after the last explicit value)
enum Expr { Num(i32), Add(i32, i32), Nil }                 // tuple-payload (tagged-union) variants
enum Shape { Circle { r: i32 }, Rect { w: i32, h: i32 } }  // struct-payload variants
```

A **C-style enum** gives each variant an integer discriminant — explicit (`= 10`) or
auto-incrementing from the previous. An explicit discriminant must be a compile-time integer
**literal, or arithmetic on literals** (`= 0x10`, `= -1`, `= 4 * 8`); a reference to a top-level
`const` is **not** accepted yet and is an `E0431` rather than being silently renumbered. Because an
enum value lowers to a 4-byte discriminant, a value outside the `i32` range is also `E0431`.
A variant `E::Name` *is* its discriminant, so it can be bound to
a `let`, compared (`==`), cast (`Code::Ok as i32`), and used as a `match` pattern
(`tests/run/enum_cstyle.wk`). **Data-carrying (tagged-union) variants** also run: a variant may
carry a tuple payload (`Num(i32)`, `Add(i32, i32)`) or named struct fields (`Circle { r: i32 }`), is
constructed as `Expr::Add(3, 4)` / `Shape::Circle { r: 5 }`, and is taken apart by a **payload
`match`** that binds each field — with literal sub-patterns (`Add(0, y)`), `if` guards, nesting in an
array of enums, and embedding in a struct field (`tests/run/enum_payload_tuple.wk`,
`enum_payload_struct.wk`). A value is a 4-byte `i32` discriminant plus a padded payload union
addressed by base pointer, so the interpreter and the native backend address it identically
(interp == native, `-O0` == `-O3`).

## Tensors and compile-time shape checking ✅ shape-check + const- and symbolic-shape exec (the headline feature)

```wukong
fn matmul<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N], mut c: Tensor[f32, M, N]) { ... }
```

Tensor dimensions are part of the type. The semantic analyzer unifies dimensions across a call:
the shared `K` above must agree on both operands. Mismatches are **compile errors**, not runtime
faults:

- `E0501` — **rank / element-count** errors: a different number of dimensions, indexing a tensor with
  more indices than its rank (fewer select a view), a compile-time-constant index past a static
  dimension, a fixed-size
  array whose length cannot satisfy the tensor's element count (exactly the element count when every
  dim is a literal; a multiple of the product of the literal dims when a symbolic/`?` dim is still
  free), and a kind clash — a scalar, SIMD vector, `[]T` slice, tuple or struct value passed where a
  tensor is expected (or the reverse).
- `E0502` — **value / element-type / layout** errors: two dimensions that disagree (e.g. calling
  `matmul::<512, 512, 513>` with `Tensor[f32, 512, 512]`), conflicting bindings of a symbolic
  dimension, a tensor element-type mismatch, and a tensor **layout** mismatch — the declared layout is
  part of the type, so a `.col_major` value cannot be routed through a `contiguous`-typed callee (and a
  row-major fixed-size array can only decay to a `contiguous` tensor).
- `E0503` — the wrong number of value arguments, **or** a turbofish whose generic-argument count does
  not match the declared generic list.
- `E0504` — a tensor type named a dimension that is not a declared generic parameter, an integer
  literal, `?`, or a `const` (reported with a did-you-mean hint against the nearest declared generic).
- `E0505` — a dimension expression that is not an extent: it folds to a negative constant, divides by
  zero, overflows, or uses `?` as an operand — at the declaration, or for the dimensions a call binds.
- `E0506` — a reshape (`x as Tensor[…]`) whose element count is not provably unchanged.
- `E0507` — a misused view or extent query: a partial index of a non-contiguous tensor, an
  assignment to a view (`a[i] = …` on a rank-2 `a`), or a `.dim(k)` whose axis is not a constant.
- `E0520` — a dimension of a call that nothing fixes: no turbofish, no argument shape, no solvable
  element count and no `?` argument extent determines it — typically a `Tensor[f32, M, N]` parameter
  fed a flat 12-element array. The help spells the turbofish (`total::<M, N>(..)`). A `const` named in
  a turbofish (`f::<W>(t)`) is its value, exactly as in a type.

**A shape error names what disagrees.** The primary message carries the dimension and both values,
so the fix is readable without opening the callee:

```
error[E0502]: dimension mismatch on `K` (axis 0): expected 512, found 513
error[E0501]: tensor rank mismatch: expected rank 2 [M, N], found rank 3 [4, 4, 4]
error[E0501]: index 5 is out of bounds for axis 1 of length 2
```

The name is the one the *declared* signature wrote (`K`), and the axis is 0-based; a dimension with
no declared name is reported as `axis 1`. Where there is no single dimension to name — a rank
mismatch, an index-count mismatch, an array-to-tensor decay, an element-type or layout mismatch —
the message names the ranks, the shapes, or the layout instead. The same text reaches tooling
verbatim as the `message` field of `--error-format=json`, alongside the unchanged `code`.

**A call's shape error also says where the dimension came from.** The error points at the
*argument* that disagrees, and its labels name what fixed the dimension and where the callee
declares it:

```
error[E0502]: dimension mismatch on `K` (axis 0): expected 512, found 513
  matmul(a, b);
         -  ^ `K` (axis 0) is 513 in argument 2
         |
         `K` = 512 was fixed here, by axis 1 of argument 1
  fn matmul<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N]) {}
                                              ----------------- `K` is declared in parameter `b` of `matmul`
```

A dimension fixed by a turbofish is labelled `fixed here, by the turbofish`, and one fixed by
another axis of the same argument is a note. A wrong argument count (`E0503`) names the callee —
"function `add` takes 2 arguments but 3 were supplied", a method as `Type::method`, counting the
arguments after the receiver — labels each extra argument and the callee's definition, and lists the
parameters a short call left out.

Run `wukongc --explain E0502` for a worked example. Dimensions may be integer literals, a declared
symbolic generic name, the name of a top-level `const`, `?` for a run-time dimension, or **arithmetic
on dimensions** (below). A `const` dim
resolves to its integer **value**, so it is a fixed dimension like a literal — `const D: usize = 3;
Tensor[f32, 2, D]` really is 3 wide, and `a[1, 3]` on it is `E0501`. A declared generic shadows a
same-named `const`. Any other name is `E0504` rather than a silently-introduced new symbolic dimension.
Tensor element types must be scalars (`E0302`).

**Dimension arithmetic ✅.** A dimension may be computed from others with `+ - * / %` and
parentheses — `Tensor[f32, S, 3*D]` (a fused QKV projection), `Tensor[f32, S, H*HD]`, `Tensor[f32,
(L - K)/S + 1]`. `/` is floor division and `%` the matching non-negative remainder (truncation, for
the non-negative operands an extent has). The checker keeps every such extent in a **canonical
normal form** (an integer polynomial over the dimension variables, with irreducible `/`/`%` terms),
so two spellings of one extent are one type — `2*N` and `N + N`, `H*HD` and `HD*H`, `(2*N + 3)/2`
and `N + 1` — and a body may return its `Tensor[f32, 2*N]` argument as `Tensor[f32, N + N]`. Equality
is *sound, not complete*: a relation the normal form cannot prove (`D` against `H*(D/H)`, which loses
the remainder) is rejected, never assumed. At a call, a dimension that appears only inside an
expression is **solved** from the argument: `Tensor[f32, 2*N]` fed 8 elements binds `N = 4`, and fed
7 elements is `E0502` (no whole-number `N`); a caller's own `M` against `2*N` is `E0502` unless the
caller can prove `M` even by spelling it `2*K`. A flat array fed to a rank-2+ tensor solves the one
dimension its element count leaves free: `Tensor[f32, M, N]` beside a `Tensor[f32, N]` of 4 elements,
fed 12, binds `M = 3`. A dimension expression must be an extent (`E0505`):
`Tensor[f32, 2 - 5]`, a division by zero, or a `?` operand are rejected, as is a call that makes a
result extent `N - 4` negative. In lowering, every variable of a derived extent is a hidden dimension
parameter and the extent is recomputed from them in the same normal form the checker compared
(`tests/run/tensor_dim_arith.wk`).

**Views: partial indexing ✅.** Indexing a contiguous tensor with **fewer indices than its rank** is
the sub-tensor of the remaining axes — a *view* of the same buffer, not a copy: `m[i]` on a
`Tensor[f32, M, N]` is the row `Tensor[f32, N]`, `c[b, h]` on a rank-4 tensor a `[S, D]` panel.
A view can be bound (`let r = m[i];`), indexed further (`m[i][j]` is exactly `m[i, j]`), passed to a
shape-generic function, returned from one, written through, and iterated: `for row in m` walks the
rows of a rank-2+ tensor (and the elements of a rank-1 one). The leading indices are bounds-checked
like any constant index. A view is never a scalar — the commonest way to get one is a forgotten
index, so using it as a number (`s + a[i]`, `print(a[i])`, `exp(a[i])`, a condition, a comparison) is
rejected naming the view's type (`E0433` for an operator, `E0436` for a condition, `E0437` for
a builtin's argument, `E0401` against a scalar place); and it is not an assignable place (`a[i] = b[i]` would have to copy a
row: `E0507`). Only a contiguous (row-major) tensor has row views (`E0507` for `.col_major` and the
other layouts). Writing through a view of a non-`mut` parameter is `E0304`, exactly like writing the
parameter (`tests/run/tensor_views.wk`).

**Reshape ✅.** `x as Tensor[T, d0, d1, …]` views a contiguous buffer — a tensor, or a fixed-size
array of scalars — under a new shape without copying. The element type must be the same (it moves no
data: `E0502`), both sides contiguous, and the element count **provably** unchanged (`E0506`):
`x as Tensor[f32, S, H, HD]` from a `Tensor[f32, S, H*HD]` is accepted (`S*(H*HD)` = `S*H*HD`), from
`Tensor[f32, S, D]` it is not. A `?` extent has no count to prove and is rejected
(`tests/run/tensor_reshape_heads.wk`).

**Extents ✅.** `t.dim(k)` is the extent of axis `k` and `t.len()` the extent of axis 0, as `i64`, on
a tensor or a fixed-size array (`.len()` also on a slice). The axis must be a compile-time constant in
range (`E0507`, `E0501`). A fixed axis folds to its constant and a symbolic or derived one is computed
from the hidden dimension parameters. A **`?` axis of a tensor parameter carries its run-time
extent**: the caller passes it as one more hidden `i64` argument — read off a tensor argument's own
shape, or a decaying array's length divided by the parameter's other extents — so the callee can read
it (`a.dim(0)`), loop to it, iterate the tensor, use it as the row stride of an interior `?` axis,
forward it to another `?` parameter, and pass the tensor where a callee has a generic dimension (which
the run-time extent then binds); an immutable view of the parameter keeps the extents of the axes
it keeps. A `?` extent that is not known at a call (a `?` tensor *returned* by a call, say) is a clean
`C0001` there, never a guessed value (`tests/run/tensor_extent.wk`).

In the other direction a `?` argument binds a **fixed** extent unchecked. Passing a `Tensor[f32, ?]`
(or an array that decayed to one) where the callee declares `[f32; 37]` or `Tensor[f32, 37]`
compiles with no diagnostic and no run-time length test; only another *constant* length is `E0501`.
The caller promises that the buffer holds at least the declared extent, and a shorter one is
outside the defined contract, like any wrong run-time size. The optimizer may read anywhere inside
the declared extent: over such a parameter the vectorizer runs a search loop's test a group ahead of
the scalar loop, so `-O2` can read past a short buffer that `-O0` happened to stop inside.

**Layout.** A tensor type may carry a physical-layout suffix, which must come last inside the
brackets: `Tensor[f32, M, N, .contiguous]` (the default, row-major), `.col_major`, `.strided`, or
`.tiled(64, 64)`. An unrecognized layout name is `E0204`. The layout is part of the type — sema unifies
it at a call, so a `.col_major` value cannot be routed through a `contiguous`-typed callee (`E0502`).
Today `contiguous` and `.col_major` **execute** for multi-dimensional indexing (`.col_major` is
byte-identical to the hand-flattened `a[i + j*M]` spelling). `.strided`/`.tiled(…)` still decline at
index (`C0001`). A fixed-size array decays only to a `contiguous` tensor.

Shape checking is not limited to call arguments. A function whose returned value's shape disagrees
with its declared `-> Tensor[…]` is rejected — `E0501` when the rank differs, `E0502` when a dimension
value, the element type or the layout differs (`tests/fail/shape_return_mismatch.wk`). The same rule is
applied to the operands of an elementwise binary operator (`tests/fail/shape_binop_mismatch.wk`), but
note that **whole-tensor arithmetic is not a supported operation**: `a + b` on two `Tensor` values is
`E0433` ("operate on elements in a loop") whatever the shapes, so the shape diagnostic is an additional
report on a program that is rejected anyway. Elementwise tensor work is written with indexed scalars —
`c[i, j] = a[i, j] + b[i, j]`. Inside a **generic** function these body checks treat the function's
own dimension variables as **rigid** — `N` matches only `N`, never another generic or a constant — so
a generic function cannot lie about its output shape either: `fn f<M, N>(a: Tensor[f32, M, N]) ->
Tensor[f32, N, 5]` is `E0502` (`tests/fail/generic_return_shape_lie.wk`). Call-site unification is a
different context and still **infers** a callee's dims from its arguments (`matmul::<…>(a, b, c)`
binds `M, N, K` from the operands) — with two limits. A callee parameter with a **fixed** dimension
cannot be satisfied by the caller's own generic dim (`fn fwd<N>(a: Tensor[f32, N]) { takes64(a) }` is
`E0502`: a universally-quantified dimension cannot be *assumed* equal to 64 — declare the callee's
parameter `?` when the size is a runtime value). And a `?` argument never *binds* a callee's symbolic
dim for checking, so a later concrete sibling still binds it and every subsequent operand is checked
against that binding; when nothing else determines the dim, lowering passes the `?` argument's run-time
extent (`tests/run/tensor_extent.wk`). A constant index past a static tensor dimension, like a fixed-size array, is `E0501`.

**What runs today.** A tensor with **compile-time-constant shape** executes end-to-end on both
backends: multi-dimensional indexing `a[i, j]` flattens to a row-major GEP off the base pointer (a
tensor is passed by base pointer, like an array out-param), so elementwise tensor kernels and tensor
matmuls run — `tests/run/tensor_*.wk`. A matmul written in tensor notation
(`c[i,j] = Σ a[i,k]·b[k,j]`, both the dot-product `s += a[i,k]*b[k,j]` and accumulate
`c[i,j] += a[i,k]*b[k,j]` spellings, including the `b[j,k]` `nn.Linear` `A·Bᵀ` form) dispatches to the
same tuned `wukong_sgemm` microkernel as the flat `a[i*K+k]` spelling — a 2-index access supplies its
row stride from the tensor's inner dimension. A **symbolic-generic** shape now executes too (✅):
`fn add<M, N>(a: Tensor[f32, M, N], …)` runs at any per-call size — the dims are threaded in as
hidden runtime `i64` parameters, so `a[i, j]`'s row stride (`i*N + j`, with `N` a runtime value) and
the loop bounds (`0..M`) resolve at run time, and a turbofish supplies them (`add::<2, 3>(…)`)
(`tests/run/generic_shape.wk`). Because the symbolic address arithmetic matches the constant-shape
form, it is byte-identical to the same kernels written with literal dims — and even a matmul with
runtime `m, n, k` dispatches to the tuned GEMM kernel (`tests/run/matmul_dynamic.wk`).

The turbofish also accepts a **runtime integer value**, not just a literal: `rowsum::<m, n>(a, out)`
with `n` computed at run time threads the live value through the same hidden dim parameters, so a
shape-typed kernel serves sizes nobody knew at compile time (the serving-code story — runtime
`seq_len`/`batch`), and a symbolic matmul called this way still dispatches to the tuned GEMM kernel.
For such dims the compile-time shape checks degrade gracefully to the runtime-`?` level — a wrong
runtime size is outside the defined contract, like any runtime index
(`tests/run/generic_shape_runtime.wk`).

## Attributes 🟡

```wukong
@generic
@strict_fp
@inline
@simd
@tile(64, 64)
@parallel(grain = 1)
@extern("C")
@export("wukong_saxpy")
fn k(@noalias mut out: []f32, x: []f32) { … }
```

Attribute *syntax* is parsed on functions, parameters and statements (a missing argument value is
`E0207`), and the attribute **name is validated against a closed set** (✅): `@parallel`, `@generic`,
`@tile`, `@extern`, `@noalias`, `@unchecked`, `@strict_fp`, `@inline`, `@simd`, `@align`, `@export`.
Anything else — `@bogus(1, 2)`, and far more usefully a typo of a real one such as `@parellel` — is
**`E0306`**, where it used to be accepted and discarded in silence (`tests/fail/unknown_attr.wk`,
`unknown_attr_typo.wk`). Five attributes have a consumer on a **function**: **`@parallel`** (✅),
which makes its loop execute across CPU cores, **`@generic`** (✅), which opts the function out of
the kernel recognizers (below), **`@strict_fp`** (✅), which evaluates the function's float
arithmetic in source order (below), **`@tile`** (✅), which strip-mines loops (below), and
**`@unchecked`** (✅), which removes the function's slice bounds checks (see *Memory and
parallelism*; on anything but a `fn` item it is `E0306`). One has a consumer on a **parameter**:
**`@noalias`** (✅), a promise about the storage the parameter points into (below). One has a
consumer on a **`let`**: **`@align(N)`** (✅), which aligns that binding's stack slot (below).
`@parallel(grain = G)` (✅) is consumed too: each worker claims runs of at least `G` consecutive
iterations of the function's parallel loop (whole-function or mid-function region), the knob an
OpenMP `schedule(dynamic, G)` or a TBB grain size gives — useful when one iteration is too cheap to be
worth a claim of its own. `@parallel` accepts nothing else: an unknown key (`@parallel(chunk = 4)`), a
positional argument, a repeated `grain`, or a grain outside `1..=2^31` is **`E0306`**
(`tests/fail/parallel_bad_grain.wk`), where the whole argument list used to be silently discarded. The
grain changes scheduling only, never results (`tests/run/parallel_grain.wk` prints every lane at grains
that do not divide the trip count). Without a `grain`, the CPU backends pick one (✅): a loop whose
body is straight-line code gets `G = ceil(2^14 / body size in MIR instructions)`, so a short loop is a
single chunk that runs without a fork and a long one is claimed in chunks of at least that much work;
a body with an inner loop or a call keeps one iteration per claim, and `--backend=gpu-native` always
does (one GPU thread per iteration). Write `grain = 1` to opt a loop back into per-iteration claims.
Everything else in the list above — `@inline`, `@simd`, `@extern` and `@export` — is
parsed and discarded (🔵). `@tile` is consumed as loop
strip-mining (see below), and when it cannot be applied it is **`E0309`**, not a silent drop. A
*discarded* attribute on a statement or loop is dropped there too:
`@parallel` must sit on the function, and the mid-function parallel region is derived from the enclosing function's
attribute, not from one written on the loop. `@generic` on a `for` or `while` loop is honored (next
section); on any other statement it — and `@strict_fp`, read only from the `fn` — is an error,
not a silent drop.

### `@generic` — “I meant this loop” ✅

```wukong
@generic
fn saxpy(a: f32, x: [f32; 4], y: [f32; 4], mut out: [f32; 4]) {
    for i in 0..4 { out[i] = a * x[i] + y[i]; }   // stays a loop; no wukong_velem_f32
}
```

`@generic` on a **function** suppresses the kernel recognizers for that function alone: none of the
`match_*` recognizers, whole-function interceptions, statement-window matchers or elementwise kernel
probes fire inside it, so its loops are compiled as written (auto-vectorized, but not replaced by a
tuned `wukong_*` kernel). Everything that is *not* a recognizer still applies — auto-vectorization,
loop fusion, FMA contraction, `@parallel` outlining, and the whole optimizer — and the builtins you
call by name (`alloc_f32`, `read_f32`, `sdpa`, …) still lower to their runtime entry points. To also
turn off FMA contraction and reduction reassociation, use `@strict_fp` (which implies `@generic`).

Two reasons to reach for it. First, honesty: a recognized kernel is invisible to every behavioral
gate, so `@generic` is how you write the loop you want measured (`tests/run/lang_generic_saxpy.wk`,
`lang_generic_naive_gemm.wk` are the same body with and without the attribute — the MIR then holds
exactly one kernel call). Second, control: it is the escape hatch when a nest merely *looks* like a
GEMM and you want your own arithmetic order — the two orders are spelled out under **What a matmul
computes** below, and `tests/run/lang_generic_gemm_fma_order.wk` pins the size at which they
provably coincide.

It may also sit on a **loop**: `@generic` on a `for` or `while` statement suppresses recognition for
that loop and every loop nested in it, while its siblings in the same function still dispatch:

```wukong
fn two(x: [f32; 64], mut o: [f32; 64], mut p: [f32; 64]) {
    for i in 0..64 { o[i] = exp(x[i]); }   // wukong_vmath_f32
    @generic
    for i in 0..64 { p[i] = exp(x[i]); }   // stays a loop
}
```

The rule is that no recognizer replaces code containing a `@generic` loop: a statement window that
would include it (a GEMM + epilogue pair, a multi-pass norm) is not formed across it, a loop that
encloses it is not matched as a whole nest (an outer `for i { for j { … @generic for k … } }`
stays loops, while the enclosing loop's other inner loops still dispatch), and a function holding
one is never claimed whole by a function-level interception (`tests/run/generic_loop_optout.wk`).
Anywhere else — a `let`, a
parameter, a non-`fn` item — `@generic` is **`E0306`**, because nothing reads it there
(`tests/fail/generic_on_stmt.wk`). For a whole-program opt-out during measurement there is the
`WUKONG_NO_RECOGNIZE` environment instrument, documented in `docs/internals.md`; it is not a CLI flag
and not a user feature.

To see what was dispatched, compile with **`--remarks=dispatch`**: every statement whose lowering
became a recognized runtime kernel call gets one `note` on stderr naming the kernel symbol(s) — at the
innermost such statement, never again at its enclosing loops — and a function replaced wholesale by a
function-level interception is reported at its name. The remark is pure observation: the MIR, every
artifact and the program's output are identical with and without it.

`@parallel` **declines and lowers serially, with no diagnostic**, when the loop cannot be split into
independent chunks: a labelled loop, a plain `break` out of the parallelized loop, a `return` out of
it, or a mid-function region whose modelled body-local is reassigned inside an inner loop. (A
`break`/`continue` targeting an *inner* loop is that loop's own control flow and stays legal.) The
result is always correct — the fallback is the ordinary scalar/vectorized loop — but the parallel
speedup is silently absent, so check with `wukongc --emit=mir -O2 f.wk | grep wukong_parallel_for`.
Fixtures: `tests/run/parallel_control_flow.wk`, `parallel_region_reassigned_local.wk`.

"Correct" here means *a* correct evaluation of the program under the fast floating-point contract,
not *the same bits*. An `f32` **reduction** has three possible summation orders depending on how it
is compiled — the source's left-to-right fold (`@strict_fp`, or a loop no path vectorizes), the
sequential vectorizer's lane-partitioned tree, and the `@parallel` reduction kernel's fixed
8192-element chunks (see [Floating-point semantics](#floating-point-semantics-)) — and adding
`@parallel`, removing it, or making the reduction loop decline (a `step`, an `..=` range, an `f64`
accumulator) moves the loop between the last two, which changes low-order bits. Each order is
deterministic on its own: independent of thread count, `-O` level, backend and host. Write
`@strict_fp` when the bits must not depend on the spelling.

A sequential `for` **reduction** — `s += e`, `s = s + e`, any `+` chain holding `s` once
(`s = s + x + y` is `s += x + y`), or a running `fmax`/`fmin`, over an `f32`, `f64`, `i32` or `i64`
accumulator — is vectorized with a **canonical summation tree** that is part of the language, not of
the build: four accumulators of one 256-bit register's lanes (8 for `f32`/`i32`, 4 for `f64`/`i64`),
element `i + u·L + l` of each block into accumulator `u`, lane `l`; leftover full groups into
accumulator 0; accumulators combined in order 1, 2, 3; lanes folded left to right; the original seed
combined once (left untouched when no full group ran); then the remaining elements one at a time. The
host ISA and OS, the 2,048-element width threshold, a literal versus a runtime trip,
`WUKONG_P4_NO_256`, the `-O` level and the addend's register pressure choose only the instruction
form, never this order — the interpreter, the 128-bit strips and the 256-bit kernel all compute it
(`tests/run/reduction_canonical_tree.wk`; the full statement is in `docs/internals.md`). It is a
reassociated order, not strict source-order summation, and it is a compiler-release contract: it is
not promised stable across releases or equal to a different recognized runtime kernel's order. A loop
that is not vectorized (a `while`, a declined body, or a slice access out of bounds at run time) keeps
the strict left fold it spells.


### What a matmul computes ✅

A matmul nest that the compiler recognizes (`c[i,j] = Σ a[i,k]·b[k,j]` and its `nn.Linear`, `Aᵀ·B`,
fused-epilogue and α-scaled spellings) runs the `wukong_sgemm*` kernel family, and that kernel has a
fixed arithmetic order:

- **Each `c[i,j]` is its own reduction.** K is split into `nb = ceil(K / kc)` blocks with
  `kc = select_kc(K)` — the fewest equal-ish blocks of at most 384 steps, rounded to a multiple of 4
  (`K = 805` → 272 + 272 + 261). Each block accumulates **ascending k from +0.0 with one rounding per
  step** (a fused multiply-add), and the blocks are folded into `c` in order with one rounding each.
  `beta = 1` (accumulate into the existing `c`) adds one more rounding up front.
- **A fused epilogue is exact.** `act(alpha·(A·Bᵀ) + bias)` is applied once, after the last block,
  and is bit-identical to computing the plain matmul and then applying the activation in a separate
  loop.
- **Nothing else changes the bits.** Thread count and `@parallel`, the scheduler, the cache-block
  sizes other than the K block, packing, tile shapes, the `WUKONG_GEMM_*` tuning knobs, AVX2 versus
  the portable fallback (a host without AVX2+FMA, or any non-x86 target, runs a scalar twin that
  uses the same order with a correctly-rounded `mul_add`), and the interpreter versus the native
  backend all produce the same bits.
- **What does change them:** the 384-step K-block cap is part of the result. It is a compiler-build
  contract, like the reduction partition above: stable across hosts, thread counts and backends,
  **not** promised across releases — a retune moves the low bits of every recognized matmul.
- **Accuracy.** Every runtime test checks `|c − exact| ≤ γ_r · Σ_k |a_ik|·|b_kj|` against an f64
  reference, with `r = kc + nb − 1` (plus one for `beta = 1`) and `γ_r = r·u / (1 − r·u)`,
  `u = 2⁻²⁴` — the netlib Level-3 BLAS test criterion with the rounding count this order performs.
- **Half and int8 inputs.** bf16/f16 operands are widened exactly and then follow the f32 order; the
  int8 `u8×i8→i32` kernel is exact (wrapping `i32`, any order gives the same bits).

A matmul that is **not** recognized — declined by the matcher, written under `@generic`, or built
with `WUKONG_NO_RECOGNIZE` — computes the loop as written: `s + a*b` contracts to one FMA, and the
inner reduction follows the vectorized-reduction partition described above. The two orders coincide
when the contraction is shorter than one vector group (K ≤ 3), and the corpus pins exactly that with
non-dyadic data whose fused and unfused results differ in every lane
(`tests/run/lang_generic_gemm_fma_order.wk`). Past that, a recognized and an unrecognized matmul may
differ in the last bits: both are documented reassociations of the same sum, and neither is "the"
exact answer. Integer-valued data whose partial sums stay below 2²⁴ is exact under
any order, which is how `tests/run/gemm_blocked_every_lane.wk` checks all three kernel forms across
three K-blocks and full and edge tiles lane by lane.

Separately, loop **auto-vectorization, FMA contraction, and elementwise fusion run automatically**
(✅) — a plain `for i in 0..n { out[i] = a*x[i] + y[i] }` is vectorized, fused with an adjacent loop,
and FMA-contracted with no annotation. A **reduction** in a `@parallel` function — `let mut s = 0.0;
for k in 0..n { s += x[k]*y[k] }` (also `(x[k]-y[k])²`, the plain sum, a running `fmax`/`fmin`
max/min, and `fmax(m, abs(x[k]))` absmax — the per-tensor max/range softmax stability and dynamic int8
quantization need) — dispatches to a
deterministic multicore reduction kernel that reaches aggregate memory bandwidth (~8× single-threaded
C), with a result independent of core count (✅) — provided the data is **`f32`** and the loop is a
**half-open, unit-step** range. A `step k` or `..=` range declines to the ordinary (correct, but
single-threaded) scalar loop with no diagnostic (`tests/run/for_step_gemm.wk`), and that
half-open/unit-step precondition holds for every recognized kernel. What does *not* matter is how the
index is spelled: a row base hoisted into a local (`let ib = i*K;` then `a[ib + p]`) and a dimension
written as a bare module `const` are normalized back to the canonical `a[i*K + p]` form before
recognition (`wukong_mir_build::canon`), so ordinary refactoring does not silently cost you the
kernel. The same pass moves a `let mut acc = 0.0;` down to the loop that accumulates into it, so
declaring an accumulator a statement or two early does not break the multi-statement windows that
softmax / log-softmax / cross-entropy match. Four more spellings that used to cost the kernel are
recognized for what they compute: a value named before it is used (`let v = x[i]; o[i] = gelu(v);` —
the elementwise kernels see the body with the binding inlined, when nothing between the `let`s and
the one store can write memory, nothing after the store reads the binding, and the binding is not
a float product that inlining would contract into an FMA); the cache-friendly
**row-outer** column reduction (`out[j] = 0.0` or a `[0.0; N]` declaration, then `for i { for j {
out[j] += x[i*N + j] } }`, likewise `+= x*x` and `fmax`/`fmin`/absmax seeded from row 0 — the same
`wukong_col*` kernels as the column-outer nest, performing the same operations per column); an
**out-of-place** softmax or log-softmax that keeps `x` and writes `o` (one `wukong_norm_f32(x, o, …)`);
and a **hand-tiled** matmul, which is the untiled nest it enumerates and dispatches the GEMM kernel.
The tile loops may be `for i0 in 0..M step T`, the `let mut i0 = 0; while i0 < M { …; i0 = i0 + T; }`
pair, or the counted form `examples/matmul.wk` uses (`for it in 0..(M + T - 1) / T { let i0 = it *
T; let i1 = min(i0 + T, M); … for i in i0..i1 { … } }`); a point loop spans a full tile (`i0..i0 + T`)
or one clamped to the extent (`i0..min(i0 + T, M)`, so the tiles need not divide `M`); the extents may
be runtime values (a symbolic `Tensor` dim, a parameter) and the tile width a literal or a `const`
(a function-local `const` declared in the same block as the tiled loop counts too). Where the cover
cannot be proved at compile time — full tiles over a runtime extent need `M % T == 0`, counted
clamped tiles need `ceil(M/T)*T >= M` — it is tested at run time on entry, with the loops as written
as the fallback. The contraction may be tiled as well, around `c[i, j] += a*b` or around a scalar
accumulation seeded from `C` (`let mut s = c[i, j]; for k in k0..k1 { s += a[i, k] * b[k, j]; } c[i,
j] = s;`), which on its own is `C += A·B`. A loop over a `usize` counter — a
`usize` parameter, or the dimension `N` of a `Tensor[f32, N]` — dispatches whatever kernel the same
loop over a signed counter would (an elementwise map, argmax/argmin, a low-precision dot, sum or
map), or vectorizes, behind a run-time `end >= 0` test, since every kernel counts in signed `i64`; a
plain float fold with no kernel (`s = s + x[i]`) over such a counter keeps its source-order fold.
Nor does folding the follow-up work **into** a matmul
store rather than writing it as its own loop: `c[i*N+j] = silu(bias[j] + s)` fuses to one
`wukong_sgemm_nt_epi`, `c[i*N+j] = x[i*N+j] + s` (a residual from another array) and a second store
`d[i*N+j] = d[i*N+j] * s` alongside `c[i*N+j] = s` (the SwiGLU up-projection) each lower to the same
GEMM + `wukong_velem_f32` pair the two-loop spelling does. Element type is handled case by
case rather than being an f32-only gate: a `bf16`/`f16` array read through the explicit widening cast —
`s = s + (x[k] as f32)` — dispatches to the multicore low-precision reduction kernels
(`tests/run/parallel_reduce_lowp.wk`), while the *uncast* spelling `s = s + x[k]` stays a scalar loop
(`tests/run/parallel_reduce_elem_bf16.wk`); an `f64` reduction has no kernel and always declines.
Explicit `@simd`-typed vector values are still under construction (🔵); `@tile` ships (next section).

### `@strict_fp` — source-order floating point ✅

```wukong
@strict_fp
fn residual(x: f32, w: f32, b: f32) -> f32 {
    return x + w * b;            // a rounded multiply, then a rounded add: no FMA
}
```

`@strict_fp` on a **function** evaluates that function's floating-point arithmetic exactly as
written: left to right, one IEEE-754 rounding per `+ - * /`, no fused multiply-add, no reassociated
reduction. Concretely, inside the function

- **no FMA contraction**: `x + y*z` is `round(x + round(y*z))`, in scalar code, in a vectorized
  loop body and in its scalar remainder alike;
- **float reductions keep their sequential fold**: `s = s + x[i]` adds the elements in index order
  (it may still run SIMD loads and multiplies, but the additions stay one chain). Integer reductions
  are exact in any order and still vectorize, and so do elementwise loops, whose lane `k` performs
  exactly the operations of scalar iteration `i + k`;
- **no kernel recognizers** — `@strict_fp` implies `@generic`, because a recognized GEMM,
  reduction or norm kernel brings its own blocking, accumulator tree and FMA use. `@parallel` still
  outlines independent iterations; a reduction in a strict `@parallel` function runs serially.

What it does not change: the definition of a math intrinsic (`exp`, `fmax`, …) or of a named kernel
builtin (`sdpa`, …) — those are functions with their own documented algorithms, which may use FMA
internally. A strict function always gets an intrinsic's *inlined* form; an elementwise loop that a
fast function would have dispatched to `wukong_vmath_f32` therefore uses the inlined polynomial,
which matches the kernel except where [Math intrinsics](#math-intrinsics-) says otherwise (the
standalone `tanh` kernel is a different approximation). Strictness is
a property of the function *body*: a fast function called from a strict one keeps its contractions,
even after the optimizer inlines it, and vice versa (`tests/run/strict_fp.wk`). The optimizer itself
never reassociates, contracts, or applies algebraic float identities in either mode, so the choice is
fixed at lowering and is identical at every `-O` level and on both backends.

`--fp=strict` applies the attribute to every function of the program (`tests/run/strict_fp_cli.wk`);
it is refused with `--emit=grad`/`--train`/`--emit=obj --grad-of`, whose VJP rules need the recognized kernels. Like
`@generic`, `@strict_fp` must sit on the `fn` — on a statement, loop or parameter it is **`E0306`**.
Strict code gives up FMA throughput, reduction ILP and the tuned kernels; reach for it where the
result must match a reference evaluated in source order (a C/NumPy port, a checksum, a numerical
test), not as a default.

### `@tile` — loop strip-mining ✅

```wukong
@tile(2)           // on the fn: queue consumed by nested `for` loops in order
@tile(64, 64)      // on the fn: first loop gets 64, second gets 64
fn saxpy_tiled(...) {
    for i in 0..n { out[i] = a * x[i] + y[i]; }   // or `@tile(B)` on one loop overrides
}
```

`@tile(B)` on a **half-open, unit-step** `for i in start..end` strip-mines that loop into full tiles
(outer `io`, inner `ii` over `0..B`) plus a **remainder loop** for the tail — not implicit padding.
`@tile(BM, BN, …)` on a function queues widths consumed by nested `for` loops in source order; a
statement-level `@tile(B)` overrides for one loop. Strip-mining runs **before** the kernel recognizer
cascade — it is not a back-door into `wukong_*` kernels. Works with `@generic`, and with a label
(`'outer: for i in 0..n` is strip-mined and keeps its label). Fixture:
`tests/run/lang_tile_saxpy.wk` (`@tile(2)`, `n=5`, remainder lane prints 3).

**A `@tile` that cannot be applied is `E0309`, not silence.** Four conditions reach it:

- the **loop form** — anything other than one identifier over an exclusive, step-less range with a
  known end. `for i in 0..=n`, a stepped range, an open range and `for x in xs` cannot be strip-mined
  (`tests/fail/tile_bad_form.wk`). A `for _ in 0..n` is refused for the *binder*, and says so;
- the **argument** — a width must be a positive integer literal, so a bare `@tile`, `@tile(0)`,
  `@tile(rows)` and a width that does not fit in 64 bits are all rejected
  (`tests/fail/tile_zero_width.wk`, `tests/fail/tile_non_integer_width.wk`). Several `@tile`s on one
  statement mean their widths in source order, exactly like one `@tile` with several arguments, and
  every one of them is validated (`tests/fail/tile_second_on_stmt.wk`);
- a **width nothing applied** — the function-item queue is consumed by loops in source order, and a
  width still in it when the body ends tiled nothing: `@tile(64, 32)` over a function with one
  strip-minable loop, or a `@tile` on a function with no `for` at all
  (`tests/fail/tile_never_applied.wk`);
- a **lowering that cannot carry it** — a whole-function recognizer interception replaces the nest
  with a kernel that brings its own blocking (`tests/fail/tile_on_recognized_kernel.wk`), and the
  whole-function `@parallel` outliner rewrites the loop into runtime chunks
  (`tests/fail/tile_on_parallel_fn.wk`). Both build their lowerer with no tile queue, so `@generic`
  — which keeps the loops — is what makes `@tile` apply. Composing `@parallel` with `@tile` means
  writing the nest yourself, as `tests/run/lang_tiled_transpose.wk` does.

A refused width is **not** consumed, so the function-item queue does not shift: it stays at the head
for the next loop that can take it. Before `E0309`, an unsupported first loop discarded its width
without a word *and* moved every later width one loop earlier, so the second loop was silently
strip-mined at the wrong tile. A tiled loop's body is lowered twice — the full-tile nest and the
remainder — and the queue is rewound between the two, so a nested `for` written once takes one
width rather than two. `wukongc --explain E0309` prints the rule.

`@tile` is only accepted where something reads it: on a `fn` item, or on the `for` statement it
schedules. On a `struct`, a parameter, a `let` or a `while` it is **`E0306`** — the same
attribute-placement code `@generic` off a `fn` item carries, and for the same reason. `@tile` on a
`while` is the one worth calling out: a `while` has no induction variable and no range, so there is
nothing to strip-mine, and the attribute used to sit there reading exactly as if it applied
(`tests/fail/tile_on_while.wk`).

### `@align(N)` — an aligned stack local ✅

```wukong
@align(64) let mut tile: [f32; 64] = [0.0; 64];   // base address is a multiple of 64
```

`@align(N)` on a `let` makes that binding's stack slot start at a multiple of `N` bytes, for a
local scratch array that 256-bit or non-temporal accesses should not split across cache lines.
`N` is one power of two from 1 to 4096; it only ever raises the alignment the slot would have had
(a slot of 32 bytes or more is already 32-aligned, a smaller one aligned to its size up to 16) and
never changes a value. Every lane of an aligned local reads and writes exactly as it would without
the attribute, on every backend and at every `-O` level (`tests/run/align_local.wk`), and the
native base address is a multiple of `N` in every frame, however deep the recursion
(`wukong_codegen_cranelift/tests/alloca_align.rs`). `--emit=mir` prints the slot as
`alloca [64 x f32] align 64`.

Where it is legal (all other cases are **`E0306`**):

- on a `let` only — on a `fn`, a parameter, a loop or any other statement nothing would honour it
  (`tests/fail/align_misplaced.wk`);
- with exactly one power-of-two argument up to 4096 — `@align(48)`, `@align(0)` and a bare `@align`
  are rejected (`tests/fail/align_bad_arg.wk`);
- on a binding of one name whose type **owns** its storage: an array, a tuple or a struct. A scalar
  lives in a register, a `[]T` slice, pointer or reference names storage it does not own, and a
  tensor local is a view of someone else's buffer. Heap and scratch buffers (`alloc_<T>`,
  `scratch_<T>`) are already 64-byte aligned.

The textual-LLVM path emits `alloca …, align N`; `--backend=gpu-native` lays frame slots out at 8
bytes and declines an `@align` above 8 as `UNSUPPORTED` rather than break the promise.

### `@noalias` — a parameter's storage promise ✅

```wukong
fn axpy(@noalias mut y: []f32, x: []f32, a: f32) {
    for i in 0..y.len() { y[i] = y[i] + a * x[i]; }
}
```

`@noalias` on a parameter promises that, **for the duration of a call, the storage it points into is
reached only through that parameter** — nothing else the call can name overlaps it. It is C's
`restrict` and LLVM's `noalias` argument attribute, and it is the one fact an optimizer cannot derive
for itself: without it two buffer parameters may be the same caller buffer (`axpy(v, v, 2.0)` is
legal), so a store to `y[i]` has to be assumed to change `x[i + 1]`. With it:

- the alias analysis (`wukong_opt::alias`) treats the promised parameter as disjoint from every other
  parameter, and — when nothing derived from it escapes the function — from every pointer it lost
  track of and from every callee, so CSE forwards loads across stores to it, LICM hoists a slice's
  data-pointer re-read out of the loop, and DSE sees through it;
- the loop dependence test and the vectorizer's runtime alias guard need no range check for a pair it
  separates — and a loop whose guard could not have been expressed vectorizes anyway;
- the Cranelift backend files its accesses in alias regions of their own (a store to `y` no longer
  clobbers a load of `x`), and marks the slice's fat-pointer loads pure so they are hoisted; the
  textual LLVM IR carries `ptr noalias`;
- a recognized kernel that needs disjoint operands (a GEMM writing `C` while reading `A`/`B`) may
  dispatch on the promise even where some call site elsewhere aliases other parameters — for
  fixed-size array and statically-shaped tensor parameters, whose names cannot be re-pointed;
- a `@parallel` region may thread a loop whose promised destination is read through a gather index
  from another parameter (`tests/run/noalias_params.wk` prints every lane of each case).

One promise is enough for a pair: `y` above is disjoint from `x` even though `x` promised nothing,
which is why marking only the **written** buffers is the usual spelling. The attribute is legal only on
a parameter, takes no arguments, and only on a parameter passed by pointer (an array, slice, tensor,
pointer, reference, tuple or struct); anywhere else — on a `fn`, a statement, a scalar parameter — it
is **`E0306`** (`tests/fail/noalias_on_scalar.wk`).

**What the compiler checks (`E0316`).** A call that passes the promised parameter's argument together
with another buffer argument rooted in the same storage is rejected: `axpy(v, v, 2.0)`
(`tests/fail/noalias_aliased_call.wk`), a view of the same array (`let s: []f32 = v; axpy(s, v, 2.0)`,
`tests/fail/noalias_aliased_view.wk`), a field together with its struct (`k(p.w, p)`), or two elements
of one array (`k(m[0], m[1])` — element indices are not compared, the same conservative line Rust's
borrow checker draws). Two different fields of one struct, `k(p.w, p.b)`, are fine. A `for` element
or a `match` scrutinee of aggregate type is bound by pointer into the storage it came from, and so is
each aggregate part a destructuring pattern binds out of it: inside `for (row, _) in grid`,
`k(row, grid)` is rejected exactly as `k(grid[0].0, grid)` is (`tests/fail/noalias_for_destructure.wk`;
`match grid[0] { (row, _) => k(row, grid) }` likewise). And a promised
parameter of view type (`[]T`, `*T`, `&T`, a tensor) cannot be re-pointed as a whole (`y = other`,
`tests/fail/noalias_reassigned_slice.wk`); writing *through* it is the point, and a fixed array
parameter is assigned by copy, so reassigning one is fine.

**What it does not check.** A call whose arguments the compiler cannot prove to overlap — two views of
unknown origin, a pointer read out of memory, the caller's own unpromised parameters passed on — is
accepted. If they do overlap after all, the program has broken its promise: the interpreter still
computes the sequential meaning, but optimized native code may not, exactly as with `restrict` in C.
That is why the attribute is opt-in; nothing changes for a program that does not write it (its MIR is
byte-identical).

### `shared [T; N]` and `sync_block()` ✅

```wukong
@parallel
fn stage(mut out: [f32; 4], inp: [f32; 4]) {
    let mut tile: shared [f32; 4];
    tile[0] = inp[0];
    // ...
    sync_block();
    out[0] = tile[0];
}
```

`shared [T; N]` declares a statically-sized array in **shared address space**. On CPU backends
(interpreter, Cranelift) shared lowers to an ordinary `alloca` — thread-local stack — and
`sync_block()` is a no-op (interp) or a compiler fence (native). On `--backend=gpu-native`, shared
lowers to PTX `.shared` SMEM and `sync_block()` to `bar.sync 0` (`wukong_codegen_gpu::lower.rs` only).
`sync_block()` is only legal inside a `@parallel` function (`E0308` otherwise;
`tests/fail/sync_outside_parallel.wk`). A static shared size above 48 KiB is `E0307`
(`tests/fail/shared_too_large.wk`). Fixtures: `tests/run/lang_shared_transpose.wk`,
`tests/run/lang_tiled_transpose.wk` (composed `@tile(2,2)` + shared + `sync_block()` tiled transpose;
`@generic` keeps `wukong_transpose_f32` out of the MIR).

## Built-in intrinsics ✅

- `print(x)` — print one value followed by a newline, **rendered at its own type** (Rust's
  `Display` for that type), identically on every backend and in a linked executable:
  - a signed integer prints as itself; an **unsigned** integer prints its unsigned magnitude (a
    high-bit-set `u32`/`u64` is not shown as a negative);
  - a float prints the **shortest** decimal that parses back to the same value *at its own
    precision*, positionally (`inf`, `-inf`, `NaN`, `-0` spelled out): `print(0.1)` of an `f32`
    prints `0.1`, not the `0.10000000149011612` of the `f64` it widens to. `f16`/`bf16` values live
    in `f32` registers and print as that `f32` (as the `half` crate's `Display` does); write
    `x as f64` to see the widened value;
  - a `bool` prints `true` / `false`; a `char` prints its glyph, UTF-8 encoded (a value that is not
    a Unicode scalar value — a surrogate, or above U+10FFFF via `as char` — prints U+FFFD); `c as
    i32` prints the code point;
  - a `*u8` (a string literal) prints its NUL-terminated bytes rather than the pointer.

  - an **aggregate** prints **structurally**, like Rust's `{:?}`: a struct as `Point { x: 1, y:
    2.5 }`, a tuple as `(7, 'z', "hi")`, an array or a slice of scalars as `[1, -2, 3]` — each
    element in `Debug` style (a float always with a point, a `char` or string quoted). An array or
    slice longer than 1000 elements prints summarized, NumPy-style: `[0, 0, 0, ..., 0, 0, 9]`;
  - a reference prints what it refers to.

  A value with no printed form is **`E0448`**: a raw pointer other than `*u8` (an address — the
  interpreter and a native run would print different numbers; dereference it), a data-carrying enum
  (match on it), a slice of non-scalars, a reference inside an aggregate, a SIMD vector, a function
  value. `print()` with no argument prints a bare newline. More than one argument is `E0503`
  (arguments are not concatenated — use `printf`), and a `()` value or a whole tensor is `E0437`.
  `tests/run/print_scalar_types.wk`, `tests/run/print_float_display.wk`,
  `tests/run/print_aggregate.wk`.
- `println(x)` — alias of `print` (both already append the newline).
- `assert(cond)` — stop the program at once if `cond` is false (zero); a no-op otherwise. Exactly
  one argument (`E0503` otherwise). A float condition means `cond != 0.0`, so `assert(0.5)` passes.
  A failed assertion means the same thing on every execution path: no statement after it runs (in
  the failing function or any caller), everything printed before it stays on stdout, `assertion
  failed` goes to stderr and the exit status is `1` — under `wukongc --run` on the interpreter and
  on `--backend=native`, and in an `--emit=exe` executable. Inside a `@parallel` region the run
  reports the failure of the earliest iteration that fails, as the in-order interpreter does. A
  slice index out of bounds stops the program the same way.
- `now_ns()` — a zero-argument monotonic nanosecond clock returning `i64`, for in-process timing
  (`let t1 = now_ns(); … let dt = now_ns() - t1;`). Any argument is `E0503`. Only *differences* between
  reads are meaningful, so a program's stdout stays backend-identical.
- `sdpa(q, k, v, out, s, d, scale, causal)` — fused scaled-dot-product attention; one of the
  shape-checked kernel builtins below.

### Formatted output ✅

| builtin | writes | result |
|---|---|---|
| `printf(fmt, args…)` / `eprintf(fmt, args…)` | stdout / stderr, **no** trailing newline | — |
| `sprintf(buf, fmt, args…)` | the `[]u8` / `[u8; N]` buffer `buf`, NUL-terminated | `i64` length |
| `print_raw(x)` | one value to stdout, no newline | — |
| `eprint(x)` / `eprintln(x)` | one value to stderr, then a newline (both, like `print`/`println`) | — |
| `write_stdout(buf, n)` / `write_stderr(buf, n)` | the first `n` bytes of a `u8` buffer, verbatim | `i64` count |
| `write_text(path, buf, n)` | the first `n` bytes of a `u8` buffer to a file (created or truncated) | `i64` count, `< 0` on error |

The format is a **string literal, parsed when the program is compiled** — Rust's `format!`
grammar: `{}` / `{:spec}` placeholders take the arguments in order, `{{` / `}}` print a brace, and
a spec is `[[fill]align][+][#][0][width][.precision][type]` (ASCII fill; `<` `^` `>`; width and
precision at most 255; `type` one of `?` debug, `x` `X` `o` `b` radix, `e` `E` scientific). Each
placeholder renders exactly as `format!` renders the same spec for the same type:

```wukong
printf("step {} loss {:.4}\n", step, loss);       // step 42 loss 3.1416
printf("[{:>6}] [{:<6}] [{:^6}]\n", 7, 7, 7);    // [     7] [7     ] [  7   ]
printf("{:#x} {:08b} {:+.2e}\n", 255, 5, 1234.5); // 0xff 00000101 +1.23e3
let n = sprintf(path, "layer_{:03}.bin", i);      // "layer_007.bin", n = 13
```

A radix style prints an integer's bits at its own width (`{:x}` of `-1i8` is `ff`), and `{}` of an
aggregate prints it structurally, its elements formatted with the same spec. `sprintf` has
`snprintf` semantics: text past `len - 1` bytes is dropped, the buffer is always NUL-terminated,
and the result is the **full** formatted length (`>= buf.len()` means it was truncated) — so
`fmt_i64`/`fmt_f32` are spelled `sprintf(buf, "{}", x)` / `sprintf(buf, "{:.3}", x)`. A malformed
format string (an unclosed `{`, a named argument `{x}`, an unknown type) is **`E0449`**; a
placeholder count that differs from the argument count, or a spec that does not apply to its
argument (`{:x}` of a float, `{:.2}` of an integer, `#` without a radix style) is **`E0450`**.

Every piece of output is one call into `wukong_runtime::fmt`, which the interpreter, the JIT and a
linked executable all render through, so the bytes cannot depend on the backend.
`--backend=gpu-native` declines a program that formats (the usual UNSUPPORTED skip), and the `cc`
fallback link of `--emit=exe` (no `rustc`) has no formatting runtime: such a program fails to link
there, loudly. `tests/run/printf_format.wk`, `tests/fail/printf_*.wk`.

### Program arguments, environment, standard input, exit ✅

| builtin | result |
|---|---|
| `arg_count()` | `i64`: the number of arguments, the program itself included |
| `arg(i)` | `*u8`: argument `i` (`arg(0)` is the program); the empty string when out of range |
| `arg_i64(i, default)` / `arg_f32(i, default)` | argument `i` parsed as a number, or `default` when absent or unparsable |
| `env(name)` | `*u8`: the environment variable `name`, the empty string when unset |
| `read_stdin_u8(buf)` | `i64`: standard input read into a `[]u8` / `[u8; N]` until it is full or input ends |
| `exit(code)` | ends the program with status `code`, as returning `code` from `main` does |

`wukongc --run prog.wk -- data/weights.bin 64` passes everything after `--` to the program
(`arg(1)` is `data/weights.bin`, `arg_i64(2, 16)` is 64); a linked executable reads its own
`argv`. Everything printed before `exit` is written out first, and on the native JIT `exit` from a
nested call returns through every caller at once, exactly as a failed `assert` does. A string
returned by `arg` or `env` is valid for the rest of the run and usable directly as a `read_*` /
`write_*` path. `tests/run/process_exit.wk`, `crates/wukongc/tests/process_io.rs`.

These are recognized by the MIR builder and implemented directly by the interpreter (and, on the
native backend, by the runtime). A user-defined function of the same name shadows any of these
builtins.

### Kernel builtins ✅

The tuned runtime kernels are callable **by name**, not only through a loop nest a recognizer happens
to match. Each builtin has a real shape signature that sema checks with the same codes a generic
function's call gets, and each lowers to exactly one runtime kernel on every backend (the
interpreter calls the same symbol), so it is the explicit way to say "I mean this kernel" — and it
still means it under `@generic` or with recognition switched off. All operands are `f32` buffers
(a `Tensor[f32, ..]`, an `[f32; N]` array, or a `[]f32` slice), row-major:

| builtin | shape signature | kernel |
|---|---|---|
| `matmul::<M, K, N>(a, b, out)` | `a: [M, K]`, `b: [K, N]`, `out: [M, N]` | `wukong_sgemm` |
| `matmul_nt::<M, K, N>(a, b, out)` | `a: [M, K]`, `b: [N, K]`, `out = a·bᵀ` | `wukong_sgemm_nt` |
| `matmul_tn::<M, K, N>(a, b, out)` | `a: [K, M]`, `b: [K, N]`, `out = aᵀ·b` | `wukong_sgemm_tn` |
| `linear::<M, K, N>(x, w, b, out)` | `x: [M, K]`, `w: [N, K]`, `b: [N]`, `out = x·wᵀ + b` | `wukong_sgemm_nt_epi` |
| `transpose::<R, C>(x, out)` | `x: [R, C]`, `out: [C, R]` | `wukong_transpose_f32` |
| `softmax`, `log_softmax ::<R, C>(x, out)` | `x, out: [R, C]`, over the last axis | `wukong_norm_f32` |
| `layernorm::<R, C>(x, gamma, beta, out, eps)` | `x, out: [R, C]`, `gamma, beta: [C]` | `wukong_norm_affine_f32` |
| `rmsnorm::<R, C>(x, gamma, out, eps)` | `x, out: [R, C]`, `gamma: [C]` | `wukong_norm_affine_f32` |
| `exp`, `log`, `tanh`, `sigmoid`, `silu`, `gelu`, `relu (x, out)` | `x, out: [N]` | `wukong_vmath_f32` |
| `sum`, `amax`, `amin (x) -> f32`; `dot(x, y) -> f32` | `x, y: [N]` | `wukong_sreduce_f32` |
| `sdpa(q, k, v, out, s, d, scale, causal)` | each buffer holds at least `s·d` | `wukong_attention_f32` |

The one-argument `gelu(x)`/`exp(x)`/… stay the scalar math intrinsics; only the two-argument buffer
form is the kernel. A user function — or a local — of the same name shadows a builtin.

**Where the dimensions come from**, first binding wins: an explicit turbofish (an integer literal, a
`const`, a generic dimension, or an in-scope integer variable, which is then a *runtime* extent);
the shape of each `Tensor` operand, unified axis by axis (a conflict is `E0502`, a rank clash
`E0501`); the length of a fixed array, which fixes the one dimension of its shape still free (so
`linear(x, w, b, out)` over arrays reads N from `b`, then M from `out`, then K from `x`); a
row-wise operation's row count defaults to 1; and last a slice's runtime length. A dimension nothing
fixes is **`E0409`** — `matmul(a, b, c)` over three flat arrays could be any factorization, so it
needs `matmul::<2, 3, 4>(a, b, c)`. A static array whose length disagrees with its shape is `E0501`,
a non-`f32` operand `E0502`, and writing the output through a non-`mut` aggregate parameter
`E0304`. `sdpa`'s `s`/`d` are values: when both are constants, every operand whose type pins an
extent must hold `s·d` elements (`E0501`) and a rank-2 tensor operand must be `[s, d]` (`E0502`).

What the lowering guarantees on top of the kernel call:

- **Runtime extents are checked.** A runtime dimension must lie in `0..=i32::MAX`, and an operand
  whose length is only known at run time (a slice, or an array against a runtime dimension) must hold
  exactly its shape's elements (`sdpa`: at least `s·d`). A violation traps like a failed `assert`
  (exit 1, `assertion failed`) instead of reading or writing past a buffer
  (`tests/run/builtin_extent_trap.wk`). A constant extent the kernels' signed 64-bit ABI cannot
  carry is the codegen limit `C0001` (`tests/fail/sdpa_dimension_abi.wk`).
- **Value semantics.** A builtin reads its inputs as they were before the call, even when the output
  overlaps one: the call compares address ranges at run time and, on overlap, computes into scratch
  and copies the result out (`matmul::<2, 2, 2>(p, q, p)` is `p·q`). An elementwise or row-wise
  builtin spelled in place with one name (`softmax(x, x)`) runs in place with no test.
- **`sdpa`'s static form.** An input whose storage is provably apart from `out` — two different
  fixed-array locals, a local and a fixed-array parameter, or two different fixed-array parameters
  that no call site binds to one array or that a `@noalias` promise separates — needs no overlap
  test. When all three inputs are proved apart there is no branch: the call is one bare
  `wukong_attention_f32`, preceded only by the extent checks that are not compile-time facts, as
  straight-line `assert`s (with constant `s`/`d` and statically sized operands there are none). A
  slice, a view, a may-alias pair or one array under two names (`match m { x => sdpa(x, x, x, m,
  ..) }`, `tests/run/sdpa_match_alias.wk`) keeps the guard.
- **Empty contractions.** `matmul` with `K = 0` is the zero matrix and `linear` the broadcast bias,
  although the GEMM kernels themselves write nothing for `K = 0`.
- Inside a `@parallel` function each call uses the kernel's multicore twin, which is bit-identical.
- Reductions follow the kernel's documented lane order and identities: an empty `sum` is `0`, an
  empty `amax`/`amin` is `-inf`/`+inf`.

`sum` and `dot` lower to the same single `wukong_sreduce_f32` call the reduction recognizer emits,
which the autodiff tape already differentiates; `amax`/`amin` have no VJP, and the buffer-writing
builtins add the overlap branch, which the single-block autodiff tape does not take yet — except a
static `sdpa` (above), which is branch-free and differentiates through the fused attention VJP
(`tests/run/sdpa_loss_train.wk`, `crates/wukongc/tests/autodiff_attention.rs`). Fixtures: `tests/run/builtin_matmul.wk`, `builtin_norms.wk`, `builtin_activations.wk`,
`builtin_reductions.wk`, `sdpa_blocks.wk` (two query blocks, D = 9, both mask modes), each printing
every lane against hand-derived values; `tests/fail/builtin_dim_uninferable.wk`,
`sdpa_operand_extent.wk`, `sdpa_operand_type.wk`.

### Decode builtins ✅

The kernels an autoregressive decode step is made of — attention over a KV cache (one sequence,
or a padded batch of them), appending to the cache, and matrix-vector products over half-precision
or block-quantized weights — are builtins too. Most of their shapes are *values* (lengths, head
counts, strides), so each call is checked at run time against its operands' lengths before anything
is read or written: a geometry that does not fit traps like a failed `assert` (exit 1, `assertion
failed`) on every backend (`tests/run/decode_shape_trap.wk`). The batched attention `sdpa_kv_batch`
has a real shape signature instead, checked at compile time (below). Operand kinds are checked at
compile time with the usual codes: a wrong arity is `E0503`, and so is a turbofish — except
`sdpa_kv_batch`'s, which is its list of six dimensions, where only a wrong count is `E0503`; a wrong
operand kind or element type (an `i64` `kv_lens`, say) is `E0437`; and writing the output through a
non-`mut` aggregate parameter is `E0304`. A buffer is a `[]T` slice, a `[T; N]` array or a
contiguous `Tensor[T, …]`. Each returns `()`.

| builtin | operands | runtime kernel |
|---|---|---|
| `sdpa_kv(q, k, v, out, sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride, scale, causal)` | `f32` buffers; integers; a float `scale`; an integer or `bool` `causal` | `wukong_attention_kv_f32` |
| `sdpa_kv_batch::<B, SQ, SKV, H, HKV, D>(q, k, v, out, kv_lens, q_pos, scale, causal)` | `f32` `q, out: [B, SQ, H, D]`, `k, v: [B, SKV, HKV, D]`; `i32` `kv_lens, q_pos: [B]`; a float `scale`; an integer or `bool` `causal` | `wukong_attention_kv_batch_f32` |
| `kv_append(cache, row, pos, stride)` | `f32` buffers; integers | `wukong_kv_append_f32` |
| `gemv_bf16(w, x, y)`, `gemv_f16(w, x, y)` | `bf16`/`f16` weight `[N·K]`; `f32` `x[K]`, `y[N]` | `wukong_sgemv_{bf16,f16}` |
| `quantize_q8_0(x, q, k)`, `quantize_q4_0(x, q, k)` | `f32` rows of `k` values; `u8` blocks | `wukong_quantize_{q8_0,q4_0}_f32` |
| `gemv_q8_0(w, x, y)`, `gemv_q4_0(w, x, y)` | `u8` blocks `W[N, K]`; `f32` `x[K]`, `y[N]` | `wukong_gemv_{q8_0,q4_0}_f32` |

- **`sdpa_kv`** is attention of `sq` query rows against a cache of `skv` key/value rows: head `h` of
  query row `i` is the `d` elements at `q[i·q_stride + h·d ..]` (and is written to the same place in
  `out`); KV head `g` of key `j` is at `k[j·kv_stride + g·d ..]` (and `v`). Query head `h` reads KV
  head `h / (n_heads / n_kv_heads)` — grouped-query attention, with `n_kv_heads == n_heads` the
  ordinary multi-head form and `1` multi-query — so `n_kv_heads` must divide `n_heads`, and each
  stride must hold its row of heads. Query row `i` sits at position `q_pos0 + i`; with `causal` it
  sees keys `j <= q_pos0 + i`. A decode step is `sq = 1, q_pos0 = skv - 1`; a chunked prefill
  against a cache is `q_pos0 = skv - sq`. Elements of `out` between heads (a stride wider than
  `n_heads·d`) are left untouched. The softmax runs online in 64-key blocks on the runtime's own
  `exp`, the same blocks for every query position, so decoding a sequence one token at a time
  produces bit-for-bit the rows a causal prefill of the whole sequence produces
  (`tests/run/decode_kv_attention.wk`).
- **`sdpa_kv_batch`** runs `B` sequences in one call — a server's batched decode step, or a batch of
  chunked prefills — over packed tensors: `q` and `out` are `[B, SQ, H, D]` (sequence, query row,
  head, lane), the caches `k` and `v` are `[B, SKV, HKV, D]` with `SKV` the capacity every sequence
  is padded to. It is **shape-checked like `matmul`**: the six dimensions come from a turbofish
  (`sdpa_kv_batch::<2, 1, 512, 32, 8, 128>(…)`; a literal, a `const`, a generic dimension or an
  integer variable), from the shapes of `Tensor` operands, and from the length of a fixed array
  (which fixes the one dimension of its shape still free); disagreeing operands are `E0502` (a wrong
  rank `E0501`), an array that does not hold its shape exactly `E0501`, constant head counts that do
  not group (`H` not a multiple of `HKV`) `E0502`, and a dimension nothing fixes `E0409` (a slice's
  length fixes none). As for `matmul`, an operand whose length is only known at run time (a slice,
  or an array against a runtime dimension) must hold *exactly* its shape's elements, or the call
  traps like a failed `assert`: a cache allocated for a larger capacity than the `SKV` passed is
  refused, not read at the wrong per-sequence offsets (`tests/run/decode_kv_batch_trap.wk`). The
  `i32` vector `kv_lens[b]` (`0 ..= SKV`, checked at run time) is how many leading cache rows of
  sequence `b` hold real keys, and `q_pos[b]` is the position of its first query row: row `i` sees
  keys `j < kv_lens[b]` and, with `causal`, `j <= q_pos[b] + i`. A decode step is `q_pos[b] =
  kv_lens[b] - 1`; FlashAttention's bottom-right mask is `kv_lens[b] - SQ`; a right-padded prompt is
  `0`; a negative position is allowed (those rows see nothing yet).
  **Masked keys contribute exactly zero probability** — they are never read, so the padding may hold
  anything, NaN included — and each row is bit-for-bit the row `sdpa_kv` computes over that
  sequence alone. **A fully masked row** (an empty sequence, or a causal row before position 0) is
  defined to be exactly `0` in every lane, FlashAttention's convention rather than the NaN of a
  softmax over nothing (`tests/run/decode_kv_batch.wk`, `tests/run/decode_kv_batch_gqa.wk`).
- **GQA-group blocking.** Both attention builtins process the query heads that share a KV head
  together (up to eight per pass), so each cached key and value row is loaded once for the whole
  group instead of once per head; the result is bit-for-bit the per-head computation.
- **`kv_append(cache, row, pos, stride)`** copies `row` into `cache[pos·stride ..]` (`stride` at least
  `len(row)`); appending token `t`'s keys is `kv_append(k_cache, k_row, t, kv_stride)`.
- **`gemv_bf16` / `gemv_f16`**: `y[j] = Σ_p x[p]·w[j·K + p]` with `N = len(y)`, `K = len(x)`, and `w`
  holding exactly `N·K` weights (the `nn.Linear` `[out, in]` layout). The weights are widened in
  registers, never copied to f32, and each output is computed exactly as the f32 `matmul_nt` row
  would compute it on the widened weights (`tests/run/decode_gemv_half.wk`). The loop nest
  `for j in 0..N { let s = 0.0; for p in 0..K { s = s + x[p] * (w[j*K + p] as f32) }; y[j] = s }`
  over a `bf16`/`f16` weight is recognized into the same kernel (`decode_gemv_half_loop.wk`), and a
  recognized bf16/f16 `nn.Linear` with one to four rows takes the same path.
- **Block quantization** uses llama.cpp's `Q8_0` and `Q4_0` layouts byte for byte: a row of `K`
  values is `ceil(K/32)` blocks of 32 (the last one zero-padded), each an `f16` scale `d` (little
  endian) followed by 32 signed bytes (`Q8_0`, 34 bytes per block, value `d·q`, `d = max|x| / 127`)
  or 16 bytes of nibble pairs (`Q4_0`, 18 bytes, element `j` in the low nibble of byte `j` and element
  `j + 16` in the high one, value `d·(nibble − 8)`, `d = x_max / −8` for the element of largest
  magnitude). `quantize_q8_0(x, q, k)` packs the rows of `x` (`len(x)` a multiple of `k`) into `q`,
  which must hold exactly their blocks. `gemv_q8_0` / `gemv_q4_0` compute `y = W·x` the way
  llama.cpp's `ggml_vec_dot_q4_0_q8_0` does: `x` is quantized to `Q8_0` per call (its block scales
  kept as exact `f32`), each block's integer dot is exact, and the row accumulates
  `(d_w·d_x)·Σ w_q·x_q` in block order (`tests/run/decode_gemv_quant.wk`). The result is therefore a
  quantized approximation of `W·x` — the activation's rounding is part of the definition.
- Together they make a decode loop: `tests/run/decode_tiny_lm.wk` generates six tokens greedily with
  bf16 projections, `kv_append` into growing caches, GQA `sdpa_kv` and a `Q8_0` LM head.
- Every result is bit-identical on the interpreter and natively, and between each kernel's AVX2 path
  and its portable twin. Inside a `@parallel` function the multicore twins run (bit-identical).
  An output that overlaps an input is computed from the inputs as they were before the call.
  None of these is differentiable: autodiff refuses a loss that calls one, naming the kernel.

### Math intrinsics ✅

`f32` (scalar or in a loop), using hardware operations and polynomial approximations.
Accuracy depends on the operation and input range; this suite has no uniform 1-ULP guarantee:

- `sqrt(x)` / `rsqrt(x)` — hardware square root (and its reciprocal); `cbrt(x)` — the all-real cube root.
- `abs(x)` — `|x|` (clearing the float sign bit); vectorizes, and a `fmax(m, abs(x[k]))` loop is the per-tensor
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
- `silu_backward(x, dy)` / `gelu_backward(x, dy)` / `sigmoid_backward(x, dy)` /
  `tanh_backward(x, dy)` / `elu_backward(x, dy)` / `softplus_backward(x, dy)` — the activation
  **backward** family, each computing `dy · act'(x)` (two arguments; any other arity is not lowered as
  an intrinsic). These are the training-gradient primitives; written as an elementwise `for` loop they
  dispatch to the two-input 256-bit `wukong_vmath2_f32` kernel, and the inlined form is bit-identical
  to it.
- `fmax(a, b)` / `fmin(a, b)` — on floats. An integer *variable* operand is **`E0445`**: these used
  to convert both operands to `f32`, rounding integers above 2^24; write `max`/`min` for integers.
- `min(a, b)` / `max(a, b)` — integer **or** float minimum/maximum (`a < b ? a : b`, `a > b ? a : b`).
  On integers the compare uses the result type's signedness, so a `usize` compares unsigned (the
  largest `u64` stays the largest, it is not `-1`); on floats they are exactly `fmin`/`fmax`. The result is
  the operands' common type, as for `a + b`: the wider integer, or the float when either is one. This is
  what a ragged loop tile is written with: `for i in i0..min(i0 + T, M)`
  (`tests/run/min_max_builtin.wk`). They are not auto-vectorizer intrinsics yet, so a loop body that
  calls them stays scalar — in a hot float loop write `fmin`/`fmax`.

These intrinsics also accept **integer** operands: `abs`/`round`/`floor`/`ceil`/`trunc` are
type-preserving on an integer (integer `abs` is `select(x < 0, −x, x)`; rounding an integer is the
identity), while `sqrt` and the transcendentals promote an integer operand to `f32`
(`tests/run/int_math.wk`). The **two-argument** intrinsics follow the promoting rule, not the
preserving one: `fmax`/`fmin`, `pow`, `atan2`, `hypot` and the six `*_backward` intrinsics coerce
*both* operands to the float result type, so `fmax(3, 5)` is the `f32` value `5.0`, not an integer max
(`tests/run/activation_backward_int_args.wk`). That conversion applies to integer *literals* only for
`fmax`/`fmin`: an integer variable there is `E0445` (`tests/fail/fmax_integer_operand.wk`).

In the f32 math core, `exp` and `log` preserve NaNs. `exp(+∞) = +∞` and `exp(−∞) = +0`; finite inputs
above the polynomial's upper bound still saturate to its historical finite value (about
`2.406e38`), and sufficiently negative finite inputs flush to zero. `log(±0) = −∞`,
`log(+∞) = +∞`, and negative arguments produce NaN. Positive subnormals are normalized
before logarithm evaluation. The composed `expm1` and `log1p` handle their infinite
endpoints explicitly. These rules apply to runtime kernels and inlined scalar/SIMD formulas;
they do not establish full libm accuracy or full f64 precision. A softmax row containing NaN,
positive infinity, or only negative infinities propagates NaNs through its ordinary arithmetic.

`atan2` handles all signed-zero axes and infinite diagonals, including `atan2(-0, -0) = -π`.
`hypot` returns positive infinity if either input is infinite, even if the other is NaN;
otherwise NaNs propagate and two zero inputs produce positive zero. `pow(x, ±0) = 1` and
`pow(1, y) = 1` take precedence over NaNs. General negative-base powers remain outside the
positive-base approximation contract, and finite overflow retains the exp core's saturation.
These endpoint rules are shared by the runtime and inlined formulas; they do not imply correctly
rounded results for other inputs.

When written as a pure `for i { out[i] = f(x[i]) }` loop over `f32` arrays, **any of the 35**
transcendentals (`exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu`/the inverse trig/the hyperbolic family/…)
are **dispatched to a tuned 256-bit
AVX2/FMA kernel** (`wukong_vmath_f32`) through domain-aware lowering like matmul→GEMM.
Historical measurements reported ~2–13× over C's scalar `libm`, and ~28× across cores under
`@parallel`. Those measurements predate the exceptional-value repairs recorded in the
[September closeout report](sota-campaign-2026-09/CLOSEOUT-2026-09-23.md); current throughput
requires fresh matched-peer measurements. `erf`, `sin` and `cos` are among those 35 and dispatch too (this is what a RoPE
`sin`/`cos` loop and an exact erf-GELU loop compile to). What does *not* dispatch is a **composed or
scalar** use — an intrinsic inside a larger expression, or a single call outside an elementwise `for` —
which instead auto-vectorizes the inlined polynomial at 128-bit; `sqrt`/`rsqrt` also stay inline, being
hardware instructions. `tanh` is **one function however it is spelled**: a dispatched `tanh` loop,
a scalar `tanh(x)`, a composed `tanh(x) * s` and a `@generic` loop all compute the runtime's ≤1-ULP
piecewise kernel bit-for-bit — the inlined form mirrors it op-for-op (the same FMA chain, the same
NaN-to-clamp `min`, the same table lookup), including `−0 → −0` and `tanh(x) == x` for every
|x| < 2⁻¹³ (`tests/run/tanh_spellings.wk`; `dispatched_tanh_equals_composed_tanh` pins every lane).
The composed activations `gelu`, `mish`, `tanhshrink`, `tanh_backward` and `gelu_backward` keep the
exp-composed `1 − 2/(exp(2x) + 1)` as their *internal* building block, because their dispatched
kernels are built on it and each activation stays bit-identical to its kernel; that block is not
the language's `tanh`. On `f64` every spelling is the double-precision runtime `tanh`.
The tanh-approximation GELU written out by hand — `0.5 * v * (1.0 + tanh(0.7978845608 * (v +
0.044715 * (v * v * v))))`, in any association — is therefore **not** `gelu()`: its `tanh` is the
language's piecewise one, `gelu`'s is the exp-composed block, and the two differ in the last bits on
ordinary inputs. It lowers as written (vectorized, with the inline `tanh`) and is never substituted by
the GELU kernel (`tests/run/spelling_value_bindings.wk`); call `gelu()` for the kernel. The erf form
is a different function again.

### Scalar builtins ✅

The operations a systems language gives its number types. All are pure (usable inside `@parallel`
regions), all lower to MIR every backend runs identically, and a user function of the same name
shadows any of them.

| builtin | meaning | result |
|---|---|---|
| `clamp(x, lo, hi)` | `x < lo ? lo : (x > hi ? hi : x)`, integer or float; a NaN `x` passes through | common type, like `min`/`max`; literals adapt to the other operands |
| `fma(a, b, c)` | `a·b + c` rounded **once**, regardless of `@strict_fp`/`--fp` (a half type computes in `f32`, then rounds) | the operands' float type |
| `popcount(x)`, `clz(x)`, `ctz(x)` | one bits; leading / trailing zero bits — the width for zero | `x`'s integer type |
| `bswap(x)` | the bytes reversed (a byte is itself) | `x`'s type |
| `rotl(x, n)`, `rotr(x, n)` | rotate by `n` modulo the width (so `-1` rotates a byte by 7) | `x`'s type |
| `sat_add(a, b)`, `sat_sub(a, b)` | `a ± b` clamped to the type's range | `T` |
| `sat_cast::<T>(x)` | an integer clamped into the integer type `T`; a float converts as `as` does | `T` |
| `mulhi(a, b)` | the high half of the double-width product (`floor(a·b / 2^w)`) | `T` |
| `widening_mul(a, b)` | the exact product one width up (`i8`→`i16` … `u32`→`u64`; 64-bit operands are `E0446`) | `2T` |
| `overflowing_add/_sub/_mul(a, b)` | the wrapped result and whether it overflowed | `(T, bool)` |
| `bits(x)` | a float's IEEE bit pattern (`f32`→`u32`, `f64`→`u64`, `f16`/`bf16`→`u16`) | unsigned int |
| `f32_from_bits(u)`, `f64_from_bits`, `f16_from_bits`, `bf16_from_bits` | the float with those bits (an integer of exactly that width) | the float |
| `isnan(x)`, `isinf(x)`, `isfinite(x)`, `signbit(x)` | classification; `signbit(-0.0)` is true | `bool` |
| `copysign(x, y)` | `x`'s magnitude with `y`'s sign bit (NaNs included) | float |
| `ldexp(x, n)` | `x·2ⁿ` rounded once to nearest even, subnormals and overflow included | `x`'s type |
| `frexp(x)` | `(m, e)` with `x = m·2ᵉ`, `0.5 ≤ \|m\| < 1`; `(x, 0)` for `±0`, `±∞`, NaN | `(F, i32)` |
| `erfc(x)` | `1 − erf(x)` with full relative accuracy in the tail (≤ 3 ULP) | float |
| `erfinv(x)` | the inverse of `erf` on `(−1, 1)` (≤ 3 ULP through `1 − 2⁻⁵³`); `±1 → ±∞` | float |
| `lgamma(x)` | `ln\|Γ(x)\|` (≤ 2 ULP for `x > 0`); `+∞` at the poles | float |
| `logaddexp(a, b)` | `ln(eᵃ + eᵇ)` without overflow; `a == b` gives `a + ln 2` | float |
| `f8e4m3_to_f32(b)`, `f8e5m2_to_f32(b)` | decode an OCP FP8 code held in a `u8` (exact) | `f32` |
| `f32_to_f8e4m3(x)`, `f32_to_f8e5m2(x)` | encode, round to nearest even; E4M3FN **saturates** at ±448 (NaN → `0x7F`), E5M2 overflows to ±∞ (NaN → `0x7E`) | `u8` |

`erfc`, `erfinv` and `lgamma` evaluate in double precision (the runtime's `special.rs`): an `f32`
argument widens, computes and rounds once. `logaddexp` uses the shared `exp`/`log1p` of its type.
The **bit builtins never produce or report a signaling NaN**: `*_from_bits` of a signaling pattern
returns it with the quiet bit set, other payload bits kept, and `bits` reports any NaN quiet (the
interpreter holds an `f32` as an `f64`, which cannot carry a signaling `f32` NaN). FP8 values are
stored as raw `u8` (`alloc_u8`, `read_u8`, …) — there is no first-class `f8` scalar type yet.

Misuse has its own codes: an integer variable passed to a floating-point builtin (including
`fmax`/`fmin`) is `E0445`; a non-integer to an integer builtin, or a bit pattern of the wrong width
(`f32_from_bits(x_u16)`), is `E0446`; and two different integer types where the type is the answer
(`sat_add(x_i8, y_i32)` — 127 or 200?) is `E0447`. Fixtures: `tests/run/scalar_int_bits.wk`,
`scalar_int_arith.wk`, `scalar_float_bits.wk`, `scalar_special_functions.wk`, `fp8_convert.wk`.

## Floating-point semantics ✅

Wukong's float model is IEEE-754 binary32/binary64, round-to-nearest-ties-to-even, in one of two
modes. **fast** (the default, `--fp=fast`) licenses exactly two value-changing transformations —
FMA contraction and reassociation of float reductions — plus the substitution of a recognized
kernel, which brings its own documented order. **strict** (`@strict_fp` on a function,
`--fp=strict` for the program; see [`@strict_fp`](#strict_fp--source-order-floating-point-)) licenses
none of them. Everything else holds in both modes:

- `+ - * /` and `sqrt` on `f32`/`f64` are correctly rounded IEEE operations. Subnormal inputs and
  results are honoured (no flush-to-zero or denormals-are-zero mode is ever set), and NaN, ±∞ and
  signed zero follow IEEE: nothing assumes finite math or ignores the sign of zero. The optimizer
  applies **no algebraic float rewrite** (`x + 0.0`, `x * 0.0`, `x / c → x * (1/c)`, `x - x` are
  left alone) and constant-folds each operation at its own precision.
- `bf16`/`f16` arithmetic is carried out in `f32` and rounded to the half grid at every value.
- Comparisons are IEEE-ordered: `==`, `<`, `<=`, `>`, `>=` are false when either side is NaN and
  `!=` is true. `fmax(a, b)` is `a > b ? a : b` and `fmin(a, b)` is `a < b ? a : b` — the x86
  `maxps`/`minps` rule, *not* IEEE `maxNum`: with a NaN in either operand, or a `+0`/`-0` tie, they
  return `b`.
- A float→int `as` cast saturates, and NaN converts to 0.
- For a given mode, every backend (interpreter, native JIT, AOT objects) and every `-O` level
  computes the same bits. The mode is decided once, during lowering; the optimizer never
  contracts or reassociates, Cranelift lowers `fadd`/`fmul` as written, and the LLVM-IR emitter
  writes plain `fadd`/`fmul` with no fast-math flags and an `Op::Fma` as the `llvm.fma` intrinsic.

**FMA contraction (fast mode).** A binary `+` with a `*` operand becomes one fused multiply-add,
rounded once (`tests/run/fp_contract_rules.wk` pins every clause):

- `x + y*z` and `y*z + x` both become `fma(y, z, x)`. Parentheses are not significant, so
  `x + (y*z)` contracts too; `x + a*b*c` fuses its top-level product, `fma(round(a*b), c, x)`.
- When **both** operands are products the **left** one is fused: `a*b + c*d` is
  `fma(a, b, round(c*d))`. Which product wins changes the result, so swapping the operands of such a
  `+` is a numerical edit, not a refactor.
- Never contracted: `x - y*z`; a product reached through a binding (`let p = y*z; x + p`); and a
  compound `s += y*z` outside a vectorized float reduction, which rounds the product first (the
  spelling `s = s + y*z` contracts).
- A vectorized float reduction accumulates `s += x[i]*y[i]` (either spelling) with one FMA per
  lane — part of the reduction tree below.
- A vectorized loop body, its 256-bit recipe and its scalar remainder apply the identical rule, so
  every lane of one loop agrees.

This is the contract of Clang's `-ffp-contract=on` (fusion within one expression, never across
statements). GCC's GNU-dialect default, `-ffp-contract=fast`, also fuses across statements; Wukong
never does. `--fp=strict` / `@strict_fp` is `-ffp-contract=off`.

**Reduction order (fast mode).** A float sum loop `s = s + e(i)` (or `s += e(i)`) is reassociated,
into one of two fixed trees depending on how it is compiled:

- **Sequential vectorized**: the vectorizer's lane-partitioned tree — `U` independent eight-lane
  accumulators (`U` from the recipe's register pressure), combined in accumulator order, then lanes
  0–7 folded in order, then the loop's initial value, then the scalar tail. The partition depends on
  the expression only, never on the host ISA, the 2,048-element 256-bit threshold or
  `WUKONG_P4_NO_256` (the `@parallel` paragraph under [Attributes](#attributes-) and
  `docs/internals.md` give the full rule). Other float element types keep the portable 128-bit
  partition.
- **`@parallel` reduction kernel**: in a `@parallel` function a recognized `f32` reduction (dot
  product, sum, squared distance, max/min/absmax — the shapes listed under
  [Attributes](#attributes-)) calls
  `wukong_sreduce_f32_parallel`. The array is cut into **fixed 8,192-element chunks**; within a
  chunk, 64-element blocks feed eight independent eight-lane accumulators (block element `8·t + j`
  folds into accumulator `t`, lane `j`; one FMA per element for the product forms), the `< 64`
  remainder folds in 8-element steps into accumulator 0, the accumulators combine by the fixed
  halving tree `acc[t] = acc[t + h] ⊕ acc[t]` for `h = 4, 2, 1`, the `< 8` tail folds into the low
  lanes, the lanes combine as `((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7))`, and the chunk partials fold in
  ascending chunk order. A chunk shorter than 64 elements therefore runs on one accumulator.
  `bf16`/`f16` reductions widen to `f32` and cut on the same boundary. The result is the same on
  one core or sixteen, natively or in the interpreter.

The two trees differ from each other and from the source's left-to-right fold, so adding or removing
`@parallel` — or a reduction declining the kernel (a `step`, an `..=` range, an `f64` accumulator) —
changes low-order bits; see the `@parallel` note under [Attributes](#attributes-). Max/min reductions
use the same trees; their finite non-zero result is order-independent, but because `fmax`/`fmin`
return `b` on a NaN or a `±0` tie, *which* NaN or which zero survives can depend on the order.

*Why 8,192.* The chunk length is part of the numerical contract, not a cache-tuning knob. It is
fixed — the same on every host — because the serial kernel, the parallel kernel and the interpreter
must cut the array at the same boundaries to agree. A streaming reduction reads each element once,
so a chunk's cache residency buys nothing; the length only sets how many independent tasks exist
(128 at 2^20 elements) and how long the partial array is, and load balancing can be tuned by how
many chunks one task claims without moving a bit. Changing 8,192 would change every recognized
reduction's result, so like the vectorizer's partition it changes only with a compiler release.

**Recognized kernels (fast mode).** A loop nest the compiler recognizes is replaced by a tuned
runtime kernel that evaluates the same mathematical expression in its own documented order — a GEMM
blocks and packs the `K` sum and accumulates with FMA, a fused softmax/norm kernel reduces each row
with its own lane tree. The result is deterministic and identical across backends, thread counts and
`-O` levels, but it is not the source's order; `@generic` keeps the loop (with contraction and the
vectorizer's reduction tree), `@strict_fp` keeps the source order.

**NaN in the math intrinsics.** Every math intrinsic returns a NaN for a NaN argument. *Which* NaN
(sign and payload) is pinned only where the runtime documents it — its elementwise kernels return
the input NaN, quieted, for `exp`, `log`, `tanh`, `atan`, `asinh`, `cbrt`, `silu`, `mish`, `sinh`,
`cosh` and `hardsigmoid` (the per-op table heads `wukong_runtime/src/vmath.rs`). The exceptions are by
definition: `fmax`/`fmin` (the `maxps` rule above), `pow(x, ±0) = pow(1, y) = 1`, `hypot(±∞, NaN) = +∞`,
and a ReLU *written* as `if x > 0.0 { x } else { 0.0 }`, which maps NaN to `0` because the source
comparison does — the recognized ReLU kernel computes exactly that.

## Memory and parallelism

**Heap allocation runs today** (✅): `alloc_<T>(n)` returns a **zero-initialized** slice `[]T` of
runtime length `n`, and `free(s)` releases it — the first runtime-sized buffers in the language
(everything else is a fixed-size stack array). The v1 surface is the typed per-scalar family —
`alloc_f32` / `alloc_f64` / `alloc_i32` / `alloc_i64` / `alloc_i8` / `alloc_u8` / `alloc_f16` /
`alloc_bf16` / `alloc_i16` / `alloc_u16` / `alloc_u32` / `alloc_u64` (every element type the typed file
I/O below moves) — chosen over a generic `alloc<T>(n)` for a small, predictable surface typed in one
place (a user-defined function of the same name shadows the builtin). The count may be any integer
type; a **negative count yields an empty slice** (`len() == 0`). Contents are deterministically
zero on both backends — calloc'd bytes on native, typed zero values in the interpreter — so a
read-before-write is well-defined. Natively every allocation starts on a **64-byte (cache-line)
boundary**, in a linked executable too, so aligned 256-bit accesses and non-temporal store streams
need no peeling prologue and two threads splitting a buffer at a line boundary never false-share.
The result is an ordinary slice: `s[i]`, `s.len()`,
`for x in s`, and fn-boundary passing/mutation all compose (`tests/run/heap_*.wk`) — and a slice is a
first-class **kernel operand**: an `alloc_*` buffer handed to a matmul/GEMV nest, a norm, an activation
loop, a reduction, a scan or `sdpa` dispatches to the same tuned runtime kernel a fixed-size `[T; N]`
array does, so a runtime-sized weight blob is not a slow path. A loop no kernel matches is
auto-vectorized over a slice (or a symbolic-shape tensor) exactly as over an array, 256-bit included,
behind a one-time run-time check that every slice access is in bounds and that no written slice
overlaps another buffer the loop touches; if either check fails the loop runs as written, scalar
(`tests/run/vec256_runtime_streams.wk`).

```wukong
let mut acts: []f32 = alloc_f32(tokens * hidden);   // zero-initialized, runtime-sized
for i in 0..acts.len() {
    acts[i] = 1.0;
}
free(acts);
```

**Slice accesses are bounds-checked** (✅): `s[i]` compares `i` against the length in `s`'s own fat
pointer (one unsigned compare, so a negative index fails too) and, out of range, stops the program
with `error: index out of bounds: the len is N but the index is I at file.wk:line:col` and exit
status 1 — identically on the interpreter, the native JIT and a linked executable. At `-O2` a
bounds-check elimination pass versions each counted loop on one guard computed before it, so a loop
whose accesses are provably in range runs a copy with no check at all (and stays vectorizable),
while an out-of-range loop runs the original and stops at exactly the same iteration. `@unchecked` on
a function removes its checks — the explicit, visible opt-out, like Rust's `get_unchecked`; an
out-of-range access there is undefined behaviour on the native backend. Fixed-size arrays and
constant-shape tensors are checked at compile time for constant indices (`E0501`); a *runtime* index
into a fixed-size array is not yet checked. A recognized kernel (a loop the compiler replaces with a
runtime kernel call) runs over the extent the loop names and is not checked against slice lengths.

Freeing anything not returned by `alloc_*`, double-freeing, or touching a slice after `free` is
**undefined behavior** on the native backend; the interpreter ignores a free of anything that
is not a live `alloc_*` block and never crashes on it, but such programs are outside the
differential contract. The interpreter reuses a freed block for the next allocation that fits
it, re-zeroed, and releases a function's local arrays when it returns, so a long decode or
training loop runs in bounded memory on both backends. Allocation failure is the one place the two backends differ by construction. On
the native backend a failed (or zero-byte) allocation yields a slice whose data pointer is **null**,
which must not be dereferenced. The interpreter does not mirror that: a null slice whose length still
says `n` would only move the failure to the first element access, so it reports `interpreter heap
exhausted allocating N element(s)` as a diagnostic and exits 1 instead of returning a null slice. A program must therefore not branch on
allocation success and expect backend-identical behaviour.

**Scratch allocation runs today** (✅): `scratch_<T>(n)` (the same eight element types as
`alloc_<T>`) returns a zero-initialized, 64-byte-aligned `[]T` from a per-thread **scratch arena**
for temporaries, `scratch_mark()` returns the number of live scratch allocations as an `i64`, and
`scratch_release(m)` releases every scratch allocation made after mark `m` was taken — LIFO, like
Odin's temp allocator or Zig's arena reset. Released memory is handed out again (zeroed again), so a
kernel that needs a buffer per call stops paying a system allocation, a free and fresh-page faults
every time: after its first call, a balanced `mark; scratch_…; release` allocates nothing.

```wukong
fn layer(x: []f32, mut out: []f32) {
    let m: i64 = scratch_mark();
    let mut tmp: []f32 = scratch_f32(x.len());   // zeroed, 64-byte aligned
    // … fill and consume tmp …
    scratch_release(m);                          // tmp is dead from here on
}
```

The mark is a *count*, so it is the same value on every backend. A negative mark releases
everything; a mark at or past the current count releases nothing. Using a slice after it has been
released is undefined behaviour on the native backend (its bytes are reused), outside the
differential contract exactly like a use after `free`. Zero-length requests and allocation failure
behave as for `alloc_<T>` (null data natively, a diagnostic in the interpreter) and still count
toward the mark (`tests/run/alloc_scratch_arena.wk`).

No garbage collector and no hidden allocations beyond what you `alloc_*` or `scratch_*`; cleanup pairs
with `defer` (✅, below), and *named* allocator selection (`System`, `Pool` — 🔵) is still being
wired to the surface.

### Atomic integer add: `atomic_add` ✅

`atomic_add(place, v)` adds `v` to the integer at `place` as one indivisible read-modify-write —
`place = place + v`, wrapping at the place's width — and is a statement of type `()`. It is how
iterations of a `@parallel` loop combine into one location (a histogram, a counter, a scatter
count) without a data race:

```wukong
@parallel
fn hist(x: [i32; 4096], mut h: [i64; 16]) {
    for i in 0..4096 {
        atomic_add(h[x[i] & 15], 1);
    }
}
```

- `place` is a place in memory other than a bare name: an element `b[i]` or `t[i, j]` (array, slice
  or tensor), a field `s.f` or `t.0`, or a pointee `*p`. A bare local is private to the thread
  running it, so `atomic_add(n, 1)` is **`E0437`** (write `n += 1`). The write obeys `=`'s rules: a
  non-`mut` aggregate parameter is `E0304`, a slice index is bounds-checked.
- The place's type is an integer of any width and signedness — `i8`..`i64`, `u8`..`u64`, `isize`,
  `usize` — never `bool`, `char` or a float (**`E0446`**). Integer-only on purpose: integer addition
  mod 2^n commutes, so every thread schedule leaves the same total, while a float sum depends on the
  order of its adds.
- `v` has the place's type. An unsuffixed literal adapts to it (out of range is `E0431`); a value of
  another integer type is **`E0447`** (convert with `as`). Wrong arity is `E0503`.
- Under `@parallel`, every iteration's add lands exactly once, so the native run prints the
  interpreter's serial answer bit for bit. A loop whose only shared writes are atomic adds to a
  captured fixed-size array is still outlined — whole-function or as a mid-function region — as
  long as nothing else in the loop reads or writes that array, or any array parameter of the same
  element type that the caller could pass as the same buffer (a `@noalias` promise rules that
  out). Otherwise the loop runs serially, as for any write the proof cannot place.
- Reading a location that other iterations are still adding to is a race, as for a plain store.
  There is no fetch-add (the old value is not returned), no float atomic, and no other memory
  order: every `atomic_add` is sequentially consistent.

It is legal anywhere, not only under `@parallel`. On `--backend=gpu-native` it is not lowered
(the program is skipped as `UNSUPPORTED`). A user function named `atomic_add` shadows the builtin
(`tests/run/atomic_add_histogram.wk`; `tests/fail/atomic_add_float.wk`,
`tests/fail/atomic_add_not_place.wk`).

### Scope-exit cleanup: `defer` ✅

`defer <expr>;` runs `<expr>` when control leaves the enclosing block — however it leaves:

```wukong
fn sum_scratch(n: i64) -> i32 {
    let mut buf: []f32 = alloc_f32(n);
    defer free(buf);                 // runs on every exit below
    if n == 0i64 {
        return 0;                    // runs free(buf), then returns
    }
    // ... fill and sum buf ...
    return s as i32;                 // `s as i32` is computed FIRST, then free(buf) runs
}
```

- **Every exit**: falling off the end of the block, `return` (all pending defers of the function),
  and `break`/`continue` (the defers of every block inside the targeted loop — so a `defer` in a
  loop body runs once per iteration).
- **LIFO**: several defers run in reverse order of registration, innermost block first.
- **Values first**: a returned value, a `break` value, and a block's tail value are computed *before*
  the defers run, so `defer n += 1; return n;` returns the old `n`.
- `<expr>` may be any expression, including a block, and `defer x += 1;` defers an assignment.
- **E0314**: a deferred expression may not leave itself (`return`, or `break`/`continue` to a loop
  outside it), and a name it uses may not be rebound (`let`, `for`, a match binding) while it is
  pending — its copies at the exits would silently read the new binding.
- A failed `assert` aborts without running defers.

`defer` runs on every backend (`tests/run/defer_scope_exit.wk`): it is lowered by copying the
deferred expression onto each exit before any other lowering, so the interpreter, the native JIT,
`--emit=exe`, the LLVM emitter, the kernel recognizers and autodiff all see ordinary statements.

**`@parallel` functions
execute today** (✅): the native backend outlines the loop body into its own function and dispatches it
across CPU cores through the `wukong_runtime` rayon-backed C-ABI entry point
`wukong_parallel_for(n, body, env)`, which hands each worker a contiguous chunk of `[0, n)` and returns
only once every chunk has completed; each per-core chunk is itself auto-vectorized (parallelism ×
SIMD). The interpreter runs the same range sequentially, so results stay differentially equal.

The whole-function vehicle fires on a narrow shape: **every** parameter must be a fixed-size array
(`[T; N]`) — a `[]T` slice, a `Tensor[..]`, a pointer or a scalar parameter declines (the region
vehicle below then gets the loop) — the body must be
exactly one unlabelled `for i in 0..hi { … }` over a half-open, unit-step range starting at literal `0`,
and the body must not `return` or `break`/`continue` out of that loop (a `break`/`continue`
inside an *inner* loop is fine; a `defer` is lowered to ordinary statements first, so a body holding
one is no longer a lone `for` and declines). Anything the recognizer cannot prove declines **silently to the
ordinary serial lowering**: the program is still correct, it just does not use more than one core. A
`@parallel` function whose body is instead a recognized kernel nest (a matmul, GEMV, batched norm,
transpose, elementwise activation, …) is intercepted *before* the outliner and dispatches the multicore
`wukong_*_parallel` kernel directly. A reduction takes the other route: it is not a single `for`
statement, so the outliner declines it and the ordinary lowering that follows dispatches the multicore
reduction kernel from inside the loop.

`@parallel` has a second vehicle for functions the whole-function outliner declines. Inside a
`@parallel` function, an inner `for hh in 0..N { … }` with a literal `0` start and provably
per-iteration-disjoint writes is outlined on its own into a region body
(`wukong$par$<n>(start, end, env)`) plus one `wukong_parallel_for` call — the same runtime contract, so
the interpreter still runs `[0, N)` sequentially. The trip count is a constant `N >= 2` (a literal or
a `const`), or (✅) one known only at run time — `0..rows`, `0..s.len()`, any pure arithmetic of
names and literals over a **signed** counter — which the call site evaluates once. The disjointness
proof requires every escaping write to hit a *captured fixed-size array* or (✅) a *captured `[]T`
slice of scalars* at an index carrying the loop variable as a mixed-radix digit (`hh*C + …`, or the
tiled `t/C` / `t%C` pair), with everything else in the index bounded inside one band of the digit's
stride: `out[hh*C + j]` and `out[(hh + 1)*C + j]` both pass, a buffer written one band up from where
it is read (`out[hh + 1] = out[hh]`) does not. With a run-time trip count no term can vanish modulo
`C·N`, so an index offset by a captured name (`out[b*32 + hh*8 + j]`) needs a constant `N`.
Per-iteration scratch must be a `let` local inside the body. It declines — and the loop lowers
serially — for a loop-carried accumulator (a write to a captured scalar), a non-affine or offset
index, a local bound to a slice/pointer/reference (it might alias a capture), a slice passed whole to
a user function, a call other than a pure math intrinsic, a kernel builtin (below), a counter-based
RNG scalar (`rand_u32`, `rand_f32`, `randn_f32` — stateless, so chunking cannot change a lane;
`print` would reorder output) or an `atomic_add` to a captured array or slice element that nothing
else in the loop touches (above), a `match`/`loop` expression in the body, a `return`/`defer`, and a
labelled or loop-level `break`/`continue`. Two array **parameters** over the same element type may be
one caller buffer whatever their declared lengths, so the proof covers the union of their accesses —
unless a `@noalias` promise separates the pair, which lets a gather from one parameter into another
thread. There is no cap on the number of regions per program (the compiler used to keep
every region past the 32nd serial, silently).

**Scratch and the worker stack.** Both vehicles move the loop body into its own function, so the
body's `let` locals — per-iteration scratch arrays included — live in *that* function's frame, and it
runs on the runtime's pool workers, whose stacks are 16 MiB, rather than on the calling thread
(256 MiB under `wukongc --run`). A loop whose outlined body would declare more than **8 MiB** of
locals (half a worker stack; `let mut t: [f32; 5000000]` is 20 MB) is therefore not outlined: it
runs serially, with its scratch in the enclosing function's frame, exactly as without `@parallel`.
The rest of the worker stack is headroom for what the count cannot see — spill slots and the runtime
kernels and math intrinsics the body calls (neither vehicle outlines a body that calls a user
function).

**Slices in a region** (✅). A `[]T` capture (parameter or local) is indexed through its fat pointer,
which nothing in a region can re-point, but its *name* says nothing about where it points: two slices
of one element type — or a slice and an array — may view one buffer at any offset
(`f(x[0..64], x[8..72])`). So such a pair is settled at **run time**: the threaded call runs only when
the two element ranges are disjoint, or start at the same address when the union of their accesses
passes the proof (`f(x, x)`, in place); otherwise the same outlined body runs once over every chunk,
in order, on the calling thread — the serial loop. `--emit=mir -O2` shows the test as `ptrtoint`s
ahead of the `wukong_parallel_for`. Slices of different element types never share storage and need no
test. Fixture: `tests/run/parallel_region_slices.wk`, whose overlapping-views call prints a wrong,
run-to-run varying answer natively if the test is removed.

**Kernel builtins in a region** (✅). `softmax`, `log_softmax`, `layernorm`, `rmsnorm`, the
two-argument activations, `matmul`/`matmul_nt`/`matmul_tn`/`linear`, `transpose`, `sum`/`amax`/
`amin`/`dot` and `sdpa` are admitted as exactly the reads and writes of their buffer operands — each
kernel touches every element of each operand (the lowering's run-time extent checks pin a slice to its
shape) and writes only its output. An operand may be a range view `s[lo..hi]` of a captured slice with
a constant length `hi - lo` (the elements `lo..hi`), a captured buffer named whole (fine to read,
never to write — every iteration would write all of it), or a body-local array (private). So a row
kernel on row `r`, `softmax(x[r*C..r*C + C], out[r*C..r*C + C])`, threads the rows, while a view of
another row, a run-time length or `out[lo..]` keeps the loop serial. Inside the region each call is
the serial kernel; the region supplies the threading. Fixture: `tests/run/parallel_region_kernels.wk`.

**How many threads, and when they are used.** The workers are one pool per process, sized to the
CPUs the process may use (affinity mask and container quota respected) — or to `--threads=<n>` for a
`wukongc --run`/`--train`, or `WUKONG_NUM_THREADS=<n>` for any process, including a built executable.
On a machine whose cores carry SMT siblings the pool has one worker per *physical* core. A region whose
trip count fits one grain -- the user's, or the implicit one above -- runs once without a fork, and
the runtime's multicore kernels
(`wukong_*_parallel`) split only when the work — rows × columns × a per-element cost — can repay the
fork (`WUKONG_PAR_MIN_WORK` moves that threshold); below it they run their serial twin. Every one of
those choices is scheduling only: the output is bit-identical at every thread count.

## Calling C: `extern "C"` ✅

```wukong
extern "C" {
    fn ldexp(x: f64, e: i32) -> f64;
    fn llabs(x: i64) -> i64;
    pub fn fmodf(x: f32, y: f32) -> f32;   // `pub` exports the declaration to importing modules
}

fn main() -> i32 {
    print(ldexp(3.0, 4) as i64);   // 48
    return 0;
}
```

An `extern "C"` block declares C functions by their C symbol names; calls type-check against the
declared signature like any other call (`tests/run/extern_c_libm.wk`). A signature may use integers
(signedness is honored for narrow ones), `bool`, `f32`, `f64`, and raw pointers / references; anything
without a C mapping — a slice, array, tuple, struct, enum, tensor, SIMD vector, `f16`/`bf16`, or a
generic signature — is **C0001** at the declaration (pass `*T` plus a length instead). Variadic C
functions (`printf`) cannot be declared.

- **Native** (`--backend=native`, `--emit=obj|exe`): the symbol is imported with the C ABI. The JIT
  binds it from the running process — the C runtime and libm, plus anything already loaded (on
  Windows the Universal CRT is loaded on demand, so all of C99's libm is reachable) — and an
  unresolvable symbol is a clean error; an object file leaves it to the linker.
- **Interpreter**: calls the **same host symbol** through a scalar marshaller, so the two backends run
  identical machine code and stay bit-exact even for a libm function whose last bit differs between
  C runtimes. It supports up to 4 parameters, each an integer, `bool`, `f32` or `f64`, and a `void`,
  integer, `f32` or `f64` result. A pointer signature (the interpreter's memory is slots, not bytes)
  or an unresolvable symbol is refused **before the run** with C0001 — use `--backend=native`.
- **LLVM text** (`--emit=llvm-ir`): a `declare` with `signext`/`zeroext` on narrow integers.
- `--backend=gpu-native` declines a program that calls C (the usual UNSUPPORTED skip).

The same C function declared identically in several modules is one symbol (the duplicates merge);
an `extern` named after a builtin's lowering symbol (`print`, `print_u`, …) is E0300, and the
`wukong_` prefix stays reserved.

## File I/O ✅

**Reading and writing typed blobs runs today** (✅): a per-scalar family loads and stores raw binary
buffers, so a program can pull weights or a dataset **off disk** instead of synthesizing them.

```wukong
let mut w: []f32 = alloc_f32(768 * 768);       // zero-initialized, runtime-sized
let n: i64 = read_f32("weights.bin", w);        // n = elements actually read
if n < 0 { return 1; }                           // -1 = couldn't open, -2 = io error
// … compute in place …
let written: i64 = write_f32("out.bin", w);      // flush w as raw little-endian f32
```

The surface is `read_<T>(path: *u8, buf) -> i64`, `write_<T>(path: *u8, buf) -> i64` and
`read_<T>_at(path: *u8, byte_off, buf) -> i64` for **every fixed-width numeric `T`** — `f32`, `f64`,
`f16`, `bf16`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, the same typed, declared-in-one-place
convention as the `alloc_<T>` family — plus `file_size(path: *u8) -> i64` (the byte length, `-1` if
the file can't be opened). The on-disk format is **headerless raw little-endian**: element `i` of
`buf` is the `sizeof(T)` bytes at byte offset `i · sizeof(T)`, and nothing else — no shape, dtype tag,
or length prefix (for files that carry those, see *Checkpoints* below).

`buf` is a **`[]T` slice, a `[T; N]` array, or a contiguous `Tensor[T, …]`** whose every dimension is
known at the call (a constant or a bound dimension variable; a `?` dimension is `E0437`), and its
element type must be exactly `T` (a `[]i32` handed to `read_f32` is `E0437`, not a silent wrong
stride). A builtin that writes `buf` through a non-`mut` slice, array or tensor **parameter** is
`E0304`, exactly as `p[i] = …` through that parameter is.

- **`read_<T>(path, buf)`** fills `buf` from the file and returns the **element count read** =
  `min(len(buf), file_bytes / sizeof(T))`, so a short file reads only what it has and a long one
  fills `buf` and stops. It returns **`-1`** if the file can't be opened and **`-2`** on an io
  error. After `-1`, and for the tail `buf[n..]` left over from a short read, the destination slots
  are **untouched**, so a zero-initialized `alloc_<T>` buffer stays defined; after a `-2` (a mid-read
  device error) the first `n` slots are unspecified, because the native read lands straight in the
  destination — there is no second, staging copy of the data. **Ordering is part of the contract**:
  the open is attempted *before* the count is computed, so an unopenable path returns `-1` even for
  an empty `buf` — `read_f32("missing.bin", empty)` is `-1`, not `0`. On an openable path an empty
  `buf` reads nothing and returns `0`. Both backends implement this order.
- **`read_<T>_at(path, byte_off, buf)`** is the same read starting `byte_off` bytes into the file,
  so one tensor can be read out of a shard: `n = min(len(buf), (file_bytes - byte_off) / sizeof(T))`,
  `0` at or past the end, **`-2`** for a negative offset. `byte_off` may be any integer type.
- **`write_<T>(path, buf)`** **creates or truncates** `path`, writes all `len(buf)` elements as
  little-endian bytes, and returns the **count written** (**`-1`** if the file can't be created,
  **`-2`** on an io error).

`path` is any NUL-terminated `*u8` — usually a string literal (the `.rodata` pointer described under
*Literals*), but a `*u8` binding works too. Anything else is `E0437`, and a wrong arity is `E0503`.
Both directions are implemented by the `wukong_runtime` layer on the native backend and mirrored by
the interpreter on the same planning code — the return codes, the little-endian layout, the narrowing
store, and the canonical value of every element read (a `u8`/`u16`/`u32` read from a file compares
exactly like the same value stored by the program) — so a read/write round-trip is **differentially
tested to agree bit-for-bit** across the interpreter and the native Cranelift backend.

### Checkpoints: safetensors and `.npy` ✅

`st_read_<T>(path: *u8, name: *u8, buf) -> i64` loads the tensor called `name` from a
**safetensors** file, and `npy_read_<T>(path: *u8, buf) -> i64` the array of a NumPy **`.npy`** file
(format versions 1–3), for the same twelve `T` and the same buffer forms. Each parses the file's
header — a small, strict JSON or Python-literal parser in the runtime, no dependency — and checks the
stored tensor against the destination **at load time**, the load-time half of shape safety:

```wukong
fn load_wte(path: *u8, mut wte: Tensor[f32, 50257, 768]) -> i64 {
    return st_read_f32(path, "wte.weight", wte);   // -4 unless the file says [50257, 768]
}
```

- the stored **dtype** must equal `T` — except that a `BF16`/`F16` (or `'<f2'`) payload **widens
  exactly** into an `f32` destination, so an f32 program loads a half-precision checkpoint directly;
- a shape-typed **`Tensor`** destination must match the stored **rank and every dimension**, not just
  the element count (`[768, 50257]` is not `[50257, 768]`); an array or slice destination must hold
  **exactly** the stored element count (a checkpoint load is an equality, not the raw reads' `min`);
- a `fortran_order: True` `.npy` of rank ≥ 2 is refused: a row-major destination would receive its
  transpose.

The result is the element count, or a negative code: **`-1`** can't open, **`-2`** io error, **`-3`**
dtype mismatch, **`-4`** shape or count mismatch, **`-5`** no tensor of that name (safetensors),
**`-6`** malformed header (bad magic or JSON, a payload outside the file, a byte length that disagrees
with dtype × shape, a duplicated name). On every negative code but `-2` the destination is untouched.
Supported stored dtypes are `F32 F64 F16 BF16 I8 I16 I32 I64 U8 U16 U32 U64` (safetensors) and the
little-endian `f2 f4 f8 i1 i2 i4 i8 u1 u2 u4 u8` descrs (`.npy`); `BOOL`, the fp8 formats, a
big-endian array and a structured dtype parse and then report `-3`. A memory-mapped, zero-copy load
(`map_<T>`) is not implemented yet: every load copies once, straight into the destination.

## Random numbers and sampling ✅

Wukong's random numbers are **counter-based** — Philox4x32-10, the generator of Random123, cuRAND and
Triton's `tl.rand`: each value is a pure function of a `seed` and a `counter`, so there is no
generator state to create, share or advance, and a stream is reproducible under any split — across
`@parallel` chunks, across the interpreter and native code, across runs.

```wukong
for i in 0..n {
    w[i] = randn_f32(seed, i) * 0.02;                  // element i is always the same number
}
let keep: bool = rand_f32(seed, step * n + i) >= p;    // a dropout mask is a counter, not a state
```

- **`rand_u32(seed, ctr) -> u32`** — word 0 of the Philox4x32-10 block with key `(seed lo, seed hi)`
  and counter `(ctr lo, ctr hi, 0, 0)`; `rand_u32(0, 0)` is `0x6627e8d5`, Random123's published
  known answer. `seed` and `ctr` may be any integer type (their 64 bits, as `x as u64` gives them).
- **`rand_f32(seed, ctr) -> f32`** — the top 24 bits of that word times 2⁻²⁴: uniform on `[0, 1)`,
  every value an exact multiple of 2⁻²⁴.
- **`randn_f32(seed, ctr) -> f32`** — a standard normal sample by Box–Muller from words 0 and 1,
  `sqrt(-2 ln u1) · cos(2π u2)` with `u1 ∈ (0, 1]`, computed in `f32` on the compiler's own `log` and
  `cos`: bit-identical across backends, and close to (not bit-equal with) an `f64` reference.
- **`rand_fill_f32(buf, seed, ctr0)`** — `buf[i] = rand_f32(seed, ctr0 + i)` for every element of a
  `[]f32`, `[f32; N]` or contiguous `f32` tensor, as one runtime kernel.

The scalar builtins lower to plain integer MIR (a 32×32→64 multiply, shifts, xors), so they need no
runtime call and are accepted inside a `@parallel` region. To autodiff they are constants:
`rand_fill_f32` contributes no gradient, like a constant store.

**Sampling kernels** (✅), each defined by one total order so the result is unique whatever the
algorithm or backend:

- **`topk_f32(x, vals, idx) -> i64`** — the `k = min(len(x), len(vals), len(idx))` largest elements
  of the `f32` buffer `x`, in descending order, into `vals` (values) and the `i64` buffer `idx`
  (indices); returns `k`. **NaN ranks largest** and ties (including `-0.0` against `0.0`) go to the
  **lower index** — `torch.topk`'s order. An output may alias `x`. It has no gradient rule yet, so
  autodiff refuses it rather than drop the gradient.
- **`argsort_f32(x, idx) -> i64`** / **`argsort_desc_f32(x, idx) -> i64`** — a **stable** argsort (ties
  keep index order; NaN last ascending, first descending) of `x` into `idx`; a shorter `idx` receives
  the prefix of the order. Returns `min(len(x), len(idx))`.
- **`sample_categorical(probs, u) -> i64`** — one draw from the unnormalized weights `probs` by inverse
  CDF at the float `u`: the first index whose running mass exceeds `u · total`, where only positive,
  finite weights have mass; `u` is clamped to `[0, 1]`; `-1` when no weight has mass. With
  `u = rand_f32(seed, step)` the draw is a pure function of `(probs, seed, step)`, so temperature,
  top-k and top-p sampling are a few lines over these three kernels.

## Warnings (lints) ✅

A program that type- and shape-checks clean is linted: the compiler warns about code that compiles
but is probably not what was meant. A warning never changes what a program computes and never
stops a build unless its level says so. Each lint has a stable code (`wukongc --explain W0001`)
and a name:

| Code | Name | Reports |
|---|---|---|
| `W0001` | `unused_variables` | a `let`, parameter, `for` variable or `match` binding never read — including an accumulator that is only ever assigned (`acc += x` with `acc` never read) |
| `W0002` | `unused_mut` | a scalar or pointer binding declared `mut` that is never reassigned or borrowed (`*p = v` writes the pointee, not `p`) |
| `W0003` | `unreachable_code` | code after a `return`/`break`/`continue` or after a construct that diverges on every path, judged exactly as the definite-return check judges it |
| `W0004` | `unused_result` | an expression statement whose value is thrown away with no effect — above all `out[i] == a[i] + b[i];` written for `=` |
| `W0005` | `dead_code` | a function of the compiled file no entry point reaches (`main`, the `--grad-of` function under `--emit=grad`/`--train`/`--emit=obj --grad-of`, `pub`, `@export`/`@extern`); nothing under `--emit=obj`/`llvm-ir` or in a file with no `main` |
| `W0006` | `unwritten_output` | a `mut` array/slice/tensor/tuple/struct parameter — an output buffer — that is never stored to, assigned, passed on or borrowed |
| `W0007` | `unknown_lints` | an `@allow`/`@warn`/`@deny` argument that names no lint |
| `W0008` | `unused_imports` | an `import` no name of its file resolves through: a glob or `as` import none of whose names or qualifier is used, a second import of an already imported module, or the unused names of a `{x, y}` selection (see below) |
| `W0020` | `unreachable_patterns` | a `match` arm no value can reach: every value its pattern matches is matched by an earlier unguarded arm (`5 => …` after `0..=9 => …`, anything after `_`) |
| `W0030` | `missing_semicolon` | a statement without its `;` (see below) |
| `W0031` | `ambiguous_continuation` | a `(` or `[`, or a `-` touching its operand, at the start of a line, continuing the expression above it |

A name prefixed with `_` (`_scratch`) is never reported as unused. Levels: **allow** (silent),
**warn** (the default for every lint) and **deny** (an error; the build fails). Set them on the
command line — `-A <lint>`, `-W <lint>`, `-D <lint>`, where `<lint>` is a name, a code or
`warnings` (every lint), so `-D warnings` fails the build on any warning — or in source, on a
function, a parameter or a statement:

```wukong
@allow(dead_code)
fn kept_for_later() -> i32 { return 1; }

fn main() -> i32 {
    @allow(unused_variables)
    let scratch: i32 = 7;
    return 0;
}
```

The innermost attribute wins over an outer one and over the command line; `-D warnings` applies to
every lint still at warning level, so it does not resurrect an allowed one. Warnings are printed
after any error. The lints that need types run only on a program that checks clean; the parser's
two (`W0030`, `W0031`) are reported for every file of the program — an imported module's carry its
own path — even when a later stage fails, since a line that silently continued the one above is
often why it failed. The fixtures in `tests/warn/` pin each lint's exact output, and every program
in `tests/run/` and `examples/` compiles warning-free (`corpus_is_warning_free`).

**Unused imports.** `W0008` judges each file's imports with the same module-scoped resolution that
decides what every name means: a local that shadows an imported name does not use the import, and
of two imports of one module the second is the unused one. An `impl` is the one thing found
without a name: a method is found through its receiver's type, and a trait bound is satisfied by an
`impl` anywhere in the program. So an import is also kept when the module it loads (directly or
through its own imports) holds an `impl` that defines a method called in any file, or implements a
trait that any generic parameter is bounded by; deleting every reported import at once never
changes what the program means. It is reported in imported files too, and
`@allow(unused_imports)` may be written on the `import` itself; `--emit=obj` and `--emit=llvm-ir`
report none, since a library may import a module only for the functions it exports.

**Fixes as data.** Where the fix is mechanical a diagnostic carries it as a suggestion, rendered as
a `help:` line and emitted in `--error-format=json`'s `suggestions` with an exact byte range and an
applicability: `W0030` inserts the `;`, `W0008` deletes the import (or rewrites a selection to the
names that are used), an assignment to an immutable binding or parameter (`E0304`) inserts `mut `
at its declaration (machine-applicable; maybe-incorrect when it would make an aggregate parameter's
store reach the caller; before the whole pattern of a `let … else`, `let mut E::A(v) = e else …`;
none for a local `const`), and the resolution errors carry their did-you-mean and the edits their
help text describes (`Type::f` for an associated function called bare, `()` for a method read like
a field, the deleted argument list of a field called like a method). `E0304`'s other forms — a
builtin or a `mut self` method writing through a non-`mut` parameter, or a store through a tensor
view of one — carry no fix yet. `Type::f` is offered only for a type declared in the root file: an
imported module's type is named in the help as its module declares it (`Pt`), with no edit, since
how the calling file may spell it depends on that file's imports.

**Statements end with `;`.** It may be left out only after a block-like statement — `if`,
`match`, `loop`, `while`, `for`, a `{ … }` block — and, for any statement but a `let`, right
before the `}` that closes the block (Rust's rule). Leaving it out elsewhere still parses the same
(`W0030` is a warning, not yet an error; a `defer` statement needs its `;` too, unless it defers a
block), but it is what makes the next hazard possible: a `(`
or `[` that starts a line is a call or an index of the expression on the line above, and so is a
`-` that starts a line and touches its operand (`-1`). `acc = a[i]` followed by `(b + c)` on the
next line is `acc = a[i](b + c)`; `W0031` reports it. A `- y` with a space is the ordinary way to
continue a long subtraction and is not reported.

## Command-line interface

```
wukongc [OPTIONS] <input.wk> [-- <program args>...]
wukongc [OPTIONS] -              read the program from standard input (named <stdin>)
wukongc [OPTIONS] -e <source>    the program is the <source> argument itself

--                    everything after it is passed to the program --run executes: arg(1),
                      arg(2), ... (arg(0) is the program); without --run it is a usage error
--run                 compile and execute (interpreter by default; see --backend).
                      The program's stdout streams as it runs (per line on a terminal, in 64 KiB
                      blocks into a pipe or file) on interp and native, so a long run shows its
                      progress and a failing one has shown everything it printed before the error.
                      Mutually exclusive with --emit=<stage>: the compiler honours exactly one of
                      them, so the combination is rejected rather than silently discarding one.
--backend=<b>         interp | native | gpu | gpu-native   (default: interp)
                      native = Cranelift JIT; gpu / gpu-native require --features gpu + a CUDA device
                      aliases: interpreter = interp, cranelift = native, cuda = gpu,
                      gpu-lower = gpu-native; any other value is a usage error (exit 2)
--emit=<stage>        tokens | ast | check | mir-high | mir (alias mir-low) | mir-text | grad |
                      llvm-ir | obj | exe   (default: exe)
                      mir-text = the optimized MIR with every type, float constant bits, static,
                      extern signedness and parameter promise spelled out. An input ending in
                      .mir is such a dump: `wukongc --emit=mir|mir-text [-O<n>] [--passes=..]
                      f.mir` parses, verifies, optimizes and prints it (no other stage, no --run)
--target-cpu=<c>      native | x86-64 | x86-64-v2 | x86-64-v3 | x86-64-v4   (default: native)
                      instruction-set level for native code (--emit=obj/exe, --backend=native).
                      native = every feature of the compiling host; a named level is exact and
                      host-independent. Below x86-64-v3 the raw-AVX2 loop kernels are not used
                      (the loop keeps its result-identical 128-bit form), so the object runs on
                      any CPU of that level. The JIT refuses a level its host lacks. Runtime
                      kernels select AVX2/FMA at run time, so they are portable at every level.
                      grad = reverse-mode backward MIR of a loss fn (see --grad-of/--grad-wrt)
                      check = run lex/parse/imports/sema only (no lowering); the fast typo loop
--grad-of=<fn>        function to differentiate for --emit=grad / --train   (default: loss);
                      with --emit=obj, also exports <fn>_grad in the object
--grad-wrt=<i,..>     buffer-parameter indices to differentiate w.r.t. (default: every buffer
                      parameter of the loss fn; with --train, every one EXCEPT the last, which is
                      the scalar loss output). An index that is out of range, not a buffer
                      (pointer) parameter, or repeated is an error.
--train               run a fwd→bwd→optimizer loop and print the loss trajectory
                      interp or native JIT; concrete f32 buffers. With --emit=exe|obj (or -o),
                      build the loop into an executable/object that prints exactly that
                      trajectory instead (not with --train-init/--train-data/--train-save)
--train-steps=<n>     training steps                                       (default: 100)
--train-lr=<f>        learning rate                                        (default: 0.01)
--train-opt=<o>       sgd | adamw                                          (default: sgd)
--train-seed=<n>      seed for the deterministic buffer initialization      (default: 0x5EED1234)
--train-weight-decay=<f>  AdamW decoupled decay; finite, non-negative       (default: 0)
--train-init=<i>=<path>   load buffer <i> from <path> (raw little-endian f32, exactly the
                      buffer's elements) instead of the seeded init; repeatable
--train-data=<i>=<path>   stream fixed-input buffer <i> from <path>, a file of whole batches:
                      update s trains on batch s mod batches (the final report on the next one)
--train-save=<i>=<path>   write buffer <i> to <path> as raw little-endian f32 after training
--train-clip=<f>      clip the global gradient L2 norm to <f> before each update (finite, > 0)
--train-warmup=<n>    ramp the learning rate linearly over the first n updates   (default: 0)
--train-schedule=<s>  constant | cosine: the learning rate after the warmup (default: constant)
-O0|-O1|-O2|-O3       optimization level (default: -O0; -O3 currently runs the -O2 pipeline)
--passes=<list>       run exactly this comma-separated optimizer pass list instead of the -O
                      level's (-O still sets the native code generator's level); `--passes=`
                      runs none. Function passes (run together to a fixpoint): sroa, mem2reg,
                      simplify, jump-thread, simplify-cfg, simplify-phis, dce, tre, cse, dse,
                      loop-canon, loop-delete, fuse, licm, bce, unswitch, promote, vectorize;
                      program steps (run once where named): inline, unroll, slp, form-switches.
                      An unknown name is a usage error (exit 2) that lists them all
--print-after=<p>     print the MIR to stderr after each run of pass <p> on a function (a run
                      that changed nothing is one `unchanged` line), or after program step <p>;
                      `all` matches every one. Stdout is unchanged. Both flags are rejected with
                      --emit=tokens/ast/check/mir-high and on --emit=grad, --train and
                      --emit=obj --grad-of
--fp=<mode>           fast | strict   (default: fast). strict = @strict_fp on every function: no
                      FMA contraction, source-order float reductions, no kernel recognizers (see
                      "Floating-point semantics"). Rejected with --emit=grad and --train (exit 1).
-o <path>             write the artifact to <path> — only with --emit=obj or --emit=exe (default:
                      <stem>.o for objects; executables get <stem>.exe on Windows and plain <stem>
                      on unix). Rejected with --run and with every other stage, which print their
                      artifact on stdout — redirect instead.
--verify-each         verify MIR after every optimizer pass (also in release builds)
--time-passes         print per-stage compile times to stderr when the compile ends, with the
                      optimizer's per-pass and Cranelift's codegen/object-write splits and a MIR
                      size summary (heaviest functions); one JSON line under --error-format=json
--time-trace=<f>      write the same timings to <f> as Chrome trace-event JSON
                      (chrome://tracing, Perfetto, speedscope)
--depfile=<f>         write a Make-syntax dependency file (root + every import, plus an empty rule
                      per import as gcc -MP does); target = the -o artifact for --emit=obj/exe,
                      otherwise <f> itself. Rejected with --emit=tokens/ast (they stop before imports)
--error-format=<f>    human | json   (default: human)
                      json spans carry byte/line end positions, plus suggestions, children,
                      a rendered field and "version":1 (the v0 severity/code/message prefix is
                      unchanged)
-A|-W|-D <lint>       allow / warn on / deny (report as an error) a lint; <lint> is a name
                      (unused_variables), its code (W0001) or `warnings` (every lint):
                      `-D warnings` fails the build on any warning. Repeatable; `-Dwarnings`
                      works too; an unknown lint is a usage error (exit 2). See "Warnings (lints)"
--error-limit=<n>     print at most n errors before suppressing the rest (0 = unlimited; default)
--explain [CODE]      print an extended explanation for an error code; no code lists every code
--color=<when>        auto | always | never   (default: auto)
                      auto honours NO_COLOR, CLICOLOR_FORCE and TERM=dumb, and enables Windows VT
--remarks=dispatch    print a note (stderr) at every statement — or function — the compiler replaced
                      with a recognized runtime kernel call, naming the kernel; output is unchanged
-h, --help            print the usage text and exit 0
-V, --version         print the version and exit 0
```

Use `--emit` to inspect any stage of the pipeline, e.g. `wukongc --emit=mir -O2 kernel.wk` to see
the optimized IR, or `wukongc --emit=ast kernel.wk` to see the parse tree. `wukongc --emit=check
kernel.wk` runs the front end (lex, parse, imports, sema, then the lints) and stops before
lowering — the fast loop for an editor or `--error-format=json` consumer, with no `C0001` lowering
limits in the way. `--run` and `--emit=<stage>` cannot be combined; pick one. A usage error prints `error: <msg>` plus the
usage text on stderr and exits **2**.

Diagnostics render rustc-style: a label in another file gets its own `::: file:line:col` header, and
a diagnostic may carry a `help:` suggestion (a concrete replacement). `--error-format=json` emits
one object per diagnostic whose spans include byte and line/column end positions, plus `suggestions`,
`children`, a `rendered` plain-text copy and `"version":1`; the `severity`/`code`/`message` prefix is
stable. `--color=auto` honours `NO_COLOR`, `CLICOLOR_FORCE` and `TERM=dumb` and turns on Windows VT
processing; `--error-limit=<n>` caps how many errors print before the rest are suppressed.

Reverse-mode autodiff is CLI-driven too: `wukongc --emit=grad --grad-of=loss model.wk` prints the
backward MIR of a loss function, and `wukongc --train --train-opt=adamw model.wk` runs its
fwd→bwd→optimizer training loop. Both flags force at least `-O1` no matter what level you pass — the
autodiff transform consumes single-block SSA (mem2reg + simplify-cfg) — so `--emit=grad -O0` still
prints optimized MIR. Generated backward MIR is also optimized with that minimum and verified
before execution. A loss may call helper functions at any `-O` level: straight-line helpers are
spliced into the loss before it is differentiated, and a helper that cannot be (multi-block or
recursive) is named in the refusal. Every recognized tensor kernel either has a VJP — the whole
f32 GEMM family including the fused bias/activation epilogue, all activations, the norms,
cross-entropy, RoPE, the embedding gather, the fused attention kernel — or declines by name with
the reason (`docs/roadmap.md`, *Automatic differentiation*). A named kernel builtin that writes an
output buffer other than its own input (`matmul(a, b, c)`, `softmax(x, y)`) lowers behind a
run-time output-overlap dispatch — a branch — so a loss that calls one is not single-block and is
refused; write that layer as the recognized loops to train it. `sdpa` is the exception when its
static form applies — fixed-array operands with every input proved apart from `out` (for example
`@noalias` parameters): it is one bare kernel call, and a loss calling it differentiates and trains
through the fused attention VJP (`tests/run/sdpa_loss_train.wk`). `s` and `d` need not be
constants: their checks are straight-line `assert`s, which the tape passes through.

**✅ AOT gradient:** `wukongc --emit=obj --grad-of=loss -o model.o model.wk` exports `loss_grad`
next to `loss` (C ABI: the loss's parameters, then one caller-zeroed gradient buffer per
`--grad-wrt` entry), for a host training loop to link. Without `--grad-of`, `--emit=obj` is unchanged.

**✅ CPU training:** `--train` honors `--backend=interp` (the default) and `--backend=native`.
Every loss parameter must be a concrete f32 array (including nested arrays) or a dense f32 tensor
(contiguous or column-major); slices, raw pointers, scalar parameters, other element types and unrepresentable byte extents are
rejected. The last buffer must contain exactly one f32 for the scalar loss output. SGD and AdamW use deterministic initialization
(every buffer drawn uniformly from `[-0.4, 0.4)` by an LCG seeded with `--train-seed`);
AdamW accepts finite nonnegative `--train-weight-decay`, while SGD rejects nonzero decay.

**✅ Training on data:** the buffers are files in the layout `read_f32`/`write_f32` use — raw
little-endian f32 with no header, which `numpy.ndarray.astype('<f4').tofile` also writes.
`--train-init=<i>=<path>` replaces buffer `i`'s seeded values with a file of exactly its size (a
checkpoint, a pretrained weight). `--train-data=<i>=<path>` streams a fixed input from a file of
whole batches, one batch per update — update `s` reads batch `s mod batches` from disk into the
live buffer, so a dataset larger than memory trains in O(batch) space; a streamed buffer cannot also
be a `--grad-wrt` weight. `--train-save=<i>=<path>` writes buffer `i` after training (the trained
weights; printed as `wrote <path>` on stderr). `--train-clip=<f>` rescales every gradient by
`f / (norm + 1e-6)` when the global L2 norm exceeds `f` (`torch.nn.utils.clip_grad_norm_`);
`--train-warmup=<n>` ramps the rate linearly over the first `n` updates and
`--train-schedule=cosine` then decays it as `CosineAnnealingLR` does over the rest. The run header
names each of these when it differs from the default. Buffers are updated in place: the gradient
runs on the live buffers, and only a buffer the forward itself writes is snapshotted and restored
around it, so each step still sees its data and weights exactly as before the forward ran.

**✅ AOT training loop:** `wukongc --train <flags> --emit=exe -o train.exe model.wk` builds the
whole loop into an executable — the forward, its generated backward, the fused AdamW kernels and
the loop that drives them — whose stdout is byte-identical to `wukongc --train <flags> model.wk` on
the interpreter and on `--backend=native` (`--emit=obj` writes the object instead; `-o` alone means
the default `exe` stage). The equality holds by construction: the executable's `main` calls the
runtime's `wukong_rt_train_main` with the run's settings baked in, which runs the same loop and
prints the same text the in-process run does (`crates/wukongc/tests/train_exe.rs` checks it on
five models, at `-O0`/`-O2`, under SGD and AdamW). Linking it needs the rustc + `wukong_runtime`
rlib path; the C fallback runtime has no training loop and is refused by name (exit 2). An
`--emit=obj` object links the same way: besides the rlib's training loop it imports
`wukong_rt_run_main` and the print/assert entries of wukongc's generated link shim, so it does not
link against `wukong_runtime` alone. The executable reads and writes no files, so `--train-init`, `--train-data` and `--train-save` with an
artifact are refused by name (exit 2) rather than dropped. Whatever the loss prints goes to stdout
in both — once per update (the backward replays the forward) and once for the final forward, before
the report. A loss that fails at run time (a failed `assert`, a bounds failure) stops both with exit
1, and what it printed before the failing statement reaches stdout in both, ahead of the error.

`--train --run` trains the selected loss without calling `main`. Combining `--train` with any
other explicit `--emit=<stage>` is a usage error (exit 2). GPU training selectors are unsupported
by this CPU trainer, and no training speedup is established by the correctness checks.
