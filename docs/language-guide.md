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
`fn main() -> i32`, and the integer it returns becomes the process exit code.

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
(`wukongc --explain E0305`), and a diagnostic inside an imported file renders with that file's own
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
type is only a parameter's type (not a local, field, element, return type or `extern` parameter); a
function cannot be bound to a `let` or passed to a non-function parameter; the argument must name a
function, not compute one; and a function-typed parameter is neither `mut` nor named after a builtin
(`gelu`, `print`, …). A wrong signature is the ordinary `E0401` type mismatch. `tests/run/fn_params.wk`.

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
is rejected (`E0401`, e.g. `let x: i8 = 200;`, or `s.x = 9000000000;` for an `i32` field), not
silently wrapped. With **no** annotation an unsuffixed integer literal defaults to `i32`, but one that
does not fit widens to the narrowest type that holds it — `i64`, and then `u64` for a magnitude in
`(i64::MAX, u64::MAX]` — so its value is never silently truncated
(`tests/run/int_literal_widen.wk`); a value past `u64` is malformed (`E0401`). Use a suffix
(`9000000000i64`, `3000000000u32`) to pick a specific type.

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
`1.5z`, or a value past `u64` — is a compile error (`E0401`), never silently zeroed
(`tests/fail/malformed_int_literal.wk`). A **float literal** takes `_` separators and an optional
exponent, and its suffix may be `f32`, `f64`, `f16`, `bf16`, or the bare C-style `f` (= `f32`);
unsuffixed it defaults to **`f32`** (`5f`, `1.5`, `1_000.5e-3`, `0.5bf16`). A malformed one — `1.5z`,
an incomplete exponent `1.5e`, a doubled suffix `5ff` — is `E0401`, never silently `0.0`. A
**char literal** `'A'` has type **`char`** (a 32-bit Unicode scalar value) — covering the
one-character escapes (`\n` `\r` `\t` `\\` `\'` `\"` `\0`), `\xHH` hex, and `\u{…}` Unicode escapes. A
char literal holds **exactly one** Unicode scalar value, and the compiler enforces it: an empty `''`
and a multi-codepoint `'ab'` are `E0104`, and a `\u{…}` above `\u{10FFFF}` or inside the surrogate
range `\u{D800}..=\u{DFFF}` is `E0401` (in a string literal too) — none of these is silently decoded
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
(`tests/run/string_return.wk`). There is still **no string type beyond `*u8`**: no
concatenation/indexing/length operators — a string is a NUL-terminated `*u8` into `.rodata` (🟡).

## Types

| Category    | Examples                                            | Status |
|-------------|-----------------------------------------------------|--------|
| Integers    | `i8 i16 i32 i64`, `u8 u16 u32 u64`, `usize isize`    | ✅     |
| Floats      | `f16 bf16 f32 f64`                                   | ✅ scalar |
| Boolean     | `bool`                                               | ✅     |
| Pointers    | `*T`, `*mut T`, deref `*p`, address-of `&x`/`&mut x`; `&T`/`&mut T` accepted as a parameter type | ✅ |
| Arrays      | fixed-size `[T; N]` (literal/repeat init, indexed load/store) | ✅ |
| Tuples      | `(A, B, …)`, field access `t.0`, nested `t.0.1`       | ✅     |
| Structs     | `struct S { … }`, literal `S { f: v }`, field `s.f`  | ✅     |
| Enums       | C-style `enum E { A = 10, B }` (variant = its `i32` discriminant) **and data-carrying** `V(i32)` / `V { f: T }` variants | ✅ |
| Slices      | `[]T` — fat pointer `{data, len}`; `.len()`, **bounds-checked** indexed load/store, iteration, array→slice unsizing | ✅ |
| SIMD vectors| `f32x4`/`i32x4`, generic `vec[T, N]` — the type **parses and type-checks** (a non-scalar element is `E0302`, a lane/element mismatch at a call is `E0401`), but no vector value can be constructed in source (`f32x4::load(p)` is `C0001`) and float-vector arithmetic is an internal error, so nothing uses it. SIMD ships through the **automatic** 128-bit/256-bit loop vectorizer, which needs no vector type in source. | 🔵 |
| Tensors     | `Tensor[f32, M, N]` — shape checking, and **indexing** (`a[i, j]`) for both const and symbolic-generic dims, on the default `contiguous` layout. A constant-shape tensor costs **nothing** over the hand-flattened `[f32; M*N]` + `a[i*N + j]` spelling: the two compile to byte-identical MIR, same kernel dispatch and same vectorization. `.col_major` indexing executes for constant-shape tensors (`a[i, j]` is byte-identical to `a[i + j*M]`). `.strided`/`.tiled(…)` type-check but cannot be indexed (`C0001`). Whole-tensor arithmetic (`a + b`) is `E0401` — operate elementwise. | ✅ |

`&x` and `&mut x` have type `*T` and `*mut T`. A `&T`/`&mut T` **parameter** accepts them
(`fn deref(p: &i32) -> i32 { return *p; }` with `deref(&x)`), but the two spellings are distinct types
elsewhere, so a local must be annotated with the pointer form — `let p: *i32 = &x;`, not
`let p: &i32 = &x;` (that is `E0401`).

`0 as *T` is the null pointer, and a pointer used as a condition (`if p`, `while p`, `p && q`) is a
null test — `p != null` — with the same answer on every backend and at every `-O` level (✅). The
interpreter reserves its address 0 as a never-allocated null so it agrees with native here, and a
load or store through null is the interpreter diagnostic `null pointer dereference` (native faults).
A condition must be a boolean, a number (nonzero is true) or a pointer: an aggregate, slice, tensor
view, SIMD vector or function value is `E0401`.

## Operators ✅

Arithmetic `+ - * / %`, comparison `== != < <= > >=`, bitwise `& | ^`, shifts `<< >>`, boolean
`&& ||` (short-circuit), unary `-`, `!`, and `~`. Like Rust, `!`/`~` are the same operator —
bitwise complement on an integer, logical negation on a `bool`; `~` is the conventional integer
spelling. Precedence follows **Rust's** table, resolved by a Pratt parser: `||` < `&&` < comparisons
< `|` < `^` < `&` < `<<`/`>>` < `+`/`-` < `*`/`/`/`%`, all left-associative. Note this differs from C
for the bitwise operators — `1 | 2 == 2` is `(1 | 2) == 2` here, not `1 | (2 == 2)`. A **chained
comparison** is rejected rather than silently mis-parsed: `a < b < c` would parse as `(a < b) < c`, so
it is an `E0401` that names the fix (`a < b && b < c`); the equality form `a == b == c` is caught the
same way whenever the third operand is not itself a `bool` (an all-`bool` chain still parses as
`(a == b) == c`). Compound assignment (`+=`, `*=`, `<<=`, `>>=`, …) is supported.

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
backend emits as a jump table or a search tree; the semantics are the chain's). Patterns are
integer/bool **literals**, **or-patterns** `A | B | C`, half-open `lo..hi` / inclusive `lo..=hi`
**ranges** whose bounds are integer or char **literals** (a `const` bound such as `0..N` parses but
does not lower — it reports `C0001` at the pattern; inline the literal), **enum-variant** patterns
`Color::Red` (matched by discriminant), **tuple** patterns
`(0, _) => …` (each field tested and bound, nesting allowed — and these compose, e.g. `(0 | 1, y)`),
an **identifier** binding (binds the scrutinee or field), and the wildcard `_`. Any arm may carry an
optional `if` guard, and `match` works in both value and statement position. See
`tests/run/{match_expr,match_patterns,match_tuple}.wk`.

A literal pattern must be **in range for the scrutinee's type**: `match n { 200 => … }` against an
`i8` is an `E0401` ("literal `200` is out of range for `i8` (-128..=127)"), not a value truncated to
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
value (an integer `as char` is unchecked), so a `char` match needs a `_` arm; floats, structs, arrays
and pointers are covered only by a `_` or a binding. This closes a silent-wrong-answer hole — a
non-exhaustive value match used to fall through to a zero default, and a statement/empty one to an
`Unreachable` the two backends handled differently (`tests/fail/match_nonexhaustive.wk`,
`match_payload_subpattern_nonexhaustive.wk`, `match_tuple_nonexhaustive.wk`). The same analysis
reports an arm no value can reach — every value it matches is matched by an earlier unguarded arm —
as the warning `W0020` (`unreachable_patterns`).

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
no `impl Shape` is rejected at the call (`E0410`). Not supported yet: generic traits or trait
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
`const` is **not** accepted yet and is an `E0401` rather than being silently renumbered. Because an
enum value lowers to a 4-byte discriminant, a value outside the `i32` range is also `E0401`.
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
`E0401` naming the view's type; and it is not an assignable place (`a[i] = b[i]` would have to copy a
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
`E0401` ("operate on elements in a loop") whatever the shapes, so the shape diagnostic is an additional
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
@align(32)
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
**`@noalias`** (✅), a promise about the storage the parameter points into (below).
`@parallel(grain = G)` (✅) is consumed too: each worker claims runs of at least `G` consecutive
iterations of the function's parallel loop (whole-function or mid-function region), the knob an
OpenMP `schedule(dynamic, G)` or a TBB grain size gives — useful when one iteration is too cheap to be
worth a claim of its own. `@parallel` accepts nothing else: an unknown key (`@parallel(chunk = 4)`), a
positional argument, a repeated `grain`, or a grain outside `1..=2^31` is **`E0306`**
(`tests/fail/parallel_bad_grain.wk`), where the whole argument list used to be silently discarded. The
grain changes scheduling only, never results (`tests/run/parallel_grain.wk` prints every lane at grains
that do not divide the trip count).
Everything else in the list above — `@inline`, `@simd`, `@align`, `@extern` and `@export` — is
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
borrow checker draws). Two different fields of one struct, `k(p.w, p.b)`, are fine. And a promised
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

- `print(x)` — print one value followed by a newline. A signed integer or float prints as itself; an
  **unsigned** integer prints its unsigned magnitude (a high-bit-set `u32`/`u64` is not shown as a
  negative); a `*u8` (a string literal) prints its NUL-terminated bytes rather than the pointer.
  `print()` with no argument prints a bare newline. More than one argument is `E0503` (arguments are
  not concatenated), and a `()` value — from an `else`-less `if`, or a `match` with statement arms —
  is `E0401`.
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
- **Empty contractions.** `matmul` with `K = 0` is the zero matrix and `linear` the broadcast bias,
  although the GEMM kernels themselves write nothing for `K = 0`.
- Inside a `@parallel` function each call uses the kernel's multicore twin, which is bit-identical.
- Reductions follow the kernel's documented lane order and identities: an empty `sum` is `0`, an
  empty `amax`/`amin` is `-inf`/`+inf`.

`sum` and `dot` lower to the same single `wukong_sreduce_f32` call the reduction recognizer emits,
which the autodiff tape already differentiates; `amax`/`amin` have no VJP, and the buffer-writing
builtins add the overlap branch, which the single-block autodiff tape does not take yet. Fixtures: `tests/run/builtin_matmul.wk`, `builtin_norms.wk`, `builtin_activations.wk`,
`builtin_reductions.wk`, `sdpa_blocks.wk` (two query blocks, D = 9, both mask modes), each printing
every lane against hand-derived values; `tests/fail/builtin_dim_uninferable.wk`,
`sdpa_operand_extent.wk`, `sdpa_operand_type.wk`.

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
- `fmax(a, b)` / `fmin(a, b)`.
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
(`tests/run/activation_backward_int_args.wk`).

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
**undefined behavior** on the native backend; the interpreter's mark-and-forget model (its
run-scoped memory is never reclaimed) never crashes on it, but such programs are outside the
differential contract. Allocation failure is the one place the two backends differ by construction. On
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
(`[T; N]`) — a `[]T` slice, a `Tensor[..]`, a pointer or a scalar parameter declines — the body must be
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
`@parallel` function, an inner `for hh in 0..N { … }` with a literal `0` start, a literal trip count
`N >= 2`, and provably per-iteration-disjoint writes is outlined on its own into a region body
(`wukong$par$<n>(start, end, env)`) plus one `wukong_parallel_for` call — the same runtime contract, so
the interpreter still runs `[0, N)` sequentially. The disjointness proof requires every escaping write
to hit a *captured fixed-size array* at an index carrying the loop variable as a mixed-radix digit
(`hh*C + …`, or the tiled `t/C` / `t%C` pair); per-iteration scratch must be a `let` local inside the
body. It declines — and the loop lowers serially — for a loop-carried accumulator (a write to a
captured scalar), a non-affine or offset index, a local bound to a slice/pointer/reference (it might
alias a capture), a call other than a pure math intrinsic or a counter-based RNG scalar (`rand_u32`,
`rand_f32`, `randn_f32` — stateless, so chunking cannot change a lane; `print` would reorder output), a
`match`/`loop` expression in the body, a `return`/`defer`, and a labelled or loop-level
`break`/`continue`. Two array **parameters** over the same element type may be one caller buffer
whatever their declared lengths, so the proof covers the union of their accesses — unless a
`@noalias` promise separates the pair, which lets a gather from one parameter into another thread.
There is no cap on the number of regions per program (the compiler used to keep
every region past the 32nd serial, silently).

**How many threads, and when they are used.** The workers are one pool per process, sized to the
CPUs the process may use (affinity mask and container quota respected) — or to `--threads=<n>` for a
`wukongc --run`/`--train`, or `WUKONG_NUM_THREADS=<n>` for any process, including a built executable.
On a machine whose cores carry SMT siblings the pool has one worker per *physical* core. A region whose
trip count fits one grain runs once without a fork, and the runtime's multicore kernels
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
known at the call (a constant or a bound dimension variable; a `?` dimension is `E0401`), and its
element type must be exactly `T` (a `[]i32` handed to `read_f32` is `E0401`, not a silent wrong
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
*Literals*), but a `*u8` binding works too. Anything else is `E0401`, and a wrong arity is `E0503`.
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
two (`W0030`, `W0031`) are reported for the compiled file even when a later stage fails, since a
line that silently continued the one above is often why it failed. The fixtures in `tests/warn/`
pin each lint's exact output.

**Statements end with `;`.** It may be left out only after a block-like statement — `if`,
`match`, `loop`, `while`, `for`, a `{ … }` block — and, for any statement but a `let`, right
before the `}` that closes the block (Rust's rule). Leaving it out elsewhere still parses the same
(`W0030` is a warning, not yet an error), but it is what makes the next hazard possible: a `(`
or `[` that starts a line is a call or an index of the expression on the line above, and so is a
`-` that starts a line and touches its operand (`-1`). `acc = a[i]` followed by `(b + c)` on the
next line is `acc = a[i](b + c)`; `W0031` reports it. A `- y` with a space is the ordinary way to
continue a long subtraction and is not reported.

## Command-line interface

```
wukongc [OPTIONS] <input.wk>

--run                 compile and execute (interpreter by default; see --backend).
                      Mutually exclusive with --emit=<stage>: the compiler honours exactly one of
                      them, so the combination is rejected rather than silently discarding one.
--backend=<b>         interp | native | gpu | gpu-native   (default: interp)
                      native = Cranelift JIT; gpu / gpu-native require --features gpu + a CUDA device
                      aliases: interpreter = interp, cranelift = native, cuda = gpu,
                      gpu-lower = gpu-native; any other value is a usage error (exit 2)
--emit=<stage>        tokens | ast | check | mir-high | mir (alias mir-low) | grad | llvm-ir | obj | exe
                      (default: exe)
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
                      interp or native JIT; concrete f32 buffers; incompatible with --emit/-o
--train-steps=<n>     training steps                                       (default: 100)
--train-lr=<f>        learning rate                                        (default: 0.01)
--train-opt=<o>       sgd | adamw                                          (default: sgd)
--train-seed=<n>      seed for the deterministic buffer initialization      (default: 0x5EED1234)
--train-weight-decay=<f>  AdamW decoupled decay; finite, non-negative       (default: 0)
-O0|-O1|-O2|-O3       optimization level (default: -O0; -O3 currently runs the -O2 pipeline)
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
cross-entropy, RoPE, the embedding gather — or declines by name with the reason
(`docs/roadmap.md`, *Automatic differentiation*).

**✅ AOT gradient:** `wukongc --emit=obj --grad-of=loss -o model.o model.wk` exports `loss_grad`
next to `loss` (C ABI: the loss's parameters, then one caller-zeroed gradient buffer per
`--grad-wrt` entry), for a host training loop to link. Without `--grad-of`, `--emit=obj` is unchanged.

**✅ CPU training:** `--train` honors `--backend=interp` (the default) and `--backend=native`.
Every loss parameter must be a concrete f32 array (including nested arrays) or a dense f32 tensor
(contiguous or column-major); slices, raw pointers, scalar parameters, other element types and unrepresentable byte extents are
rejected. The last buffer must contain exactly one f32 for the scalar loss output. SGD and AdamW use deterministic initialization;
AdamW accepts finite nonnegative `--train-weight-decay`, while SGD rejects nonzero decay.

`--train --run` trains the selected loss without calling `main`. Combining `--train` with any
explicit `--emit=<stage>` or `-o` is a usage error (exit 2). GPU training selectors are unsupported
by this CPU trainer. An AOT training loop remains unavailable (the AOT *gradient* is above), and
no training speedup is established by the correctness checks.
