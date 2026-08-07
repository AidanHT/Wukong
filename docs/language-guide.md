# The Wukong Language Guide

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

## Modules & imports ✅

```wukong
module examples.matmul
import lib.mathlib
```

A module path is dotted, and `import` makes a program **multi-file**: `import a.b` — in any file of
the program — loads `a/b.wk`, resolved relative to the directory of the **root** source file passed
to `wukongc`, and splices its items into one merged **flat namespace**. Cross-file calls, consts,
structs, and enums then just work, with no qualification (`tests/run/import_multi.wk`). Each file is
loaded exactly once, by canonical path: an import **cycle** (`a` imports `b` imports `a`) or diamond
is not an error, and every item still lands exactly once. A top-level name defined in two files is
the ordinary duplicate-name error (E0300), pointing at **both** definitions; an import that resolves
to no file is **E0305** (`wukongc --explain E0305`). A diagnostic inside an imported file renders
with that file's own path and line.

The `module` header itself remains informational — it does not participate in resolution and need
not match the path a file was imported under. `import x as y` aliases and `import x.{a, b}` item
lists parse but neither rename nor restrict anything yet (v1 is a flat namespace) 🟡. Note that
`--emit=tokens|ast` dump the **root file only**; every later stage — and `--run` on either backend —
sees the merged multi-file program.

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
`let a: [i32; 4] = [7; 2];` is `E0401`, not a slot silently filled out to 4. An unsuffixed literal
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
arithmetic on those folds too (`[i32; 2 + 2]`). A **generic** parameter name is not a usable length: it
is exempt from the name check but evaluates to **0**, so `[i32; N]` inside `fn f<N>` is a zero-length
array and any constant index into it is `E0501` — use a `const`. Any other name is `E0301`, not a
silent length of 0, and a fixed-size array may hold at most `u32::MAX` elements (`E0401` past that).

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
| Slices      | `[]T` — fat pointer `{data, len}`; `.len()`, indexed load/store, iteration, array→slice unsizing | ✅ |
| SIMD vectors| `f32x4`/`i32x4`, generic `vec[T, N]` — the type **parses and type-checks** (a non-scalar element is `E0302`, a lane/element mismatch at a call is `E0401`), but no vector value can be constructed in source (`f32x4::load(p)` is `C0001`) and float-vector arithmetic is an internal error, so nothing uses it. SIMD ships through the **automatic** 128-bit/256-bit loop vectorizer, which needs no vector type in source. | 🔵 |
| Tensors     | `Tensor[f32, M, N]` — shape checking, and **indexing** (`a[i, j]`) for both const and symbolic-generic dims, on the default `contiguous` layout. A constant-shape tensor costs **nothing** over the hand-flattened `[f32; M*N]` + `a[i*N + j]` spelling: the two compile to byte-identical MIR, same kernel dispatch and same vectorization. Whole-tensor arithmetic (`a + b`) is `E0401` — operate elementwise. A `.col_major`/`.strided`/`.tiled(…)` layout type-checks but cannot be indexed (`C0001`). | ✅ |

`&x` and `&mut x` have type `*T` and `*mut T`. A `&T`/`&mut T` **parameter** accepts them
(`fn deref(p: &i32) -> i32 { return *p; }` with `deref(&x)`), but the two spellings are distinct types
elsewhere, so a local must be annotated with the pointer form — `let p: *i32 = &x;`, not
`let p: &i32 = &x;` (that is `E0401`).

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

`match` evaluates its scrutinee once and lowers to an if-else chain over the arms. Patterns are
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
empty `match x {}` included): an `enum` needs every variant, a `bool` needs both cases, and any other
scalar (an unbounded domain) needs a catch-all. A catch-all is a guard-less `_` **or** a guard-less
identifier arm; an `enum` may also be covered by integer-literal/range patterns that match its
declared discriminants. A provably-incomplete match is rejected at compile time (`E0405`); a guard
(`if …`) never counts toward coverage. This closes a silent-wrong-answer hole — a non-exhaustive value
match used to fall through to a zero default, and a statement/empty one to an `Unreachable` the two
backends handled differently (`tests/fail/match_nonexhaustive.wk`).

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
read-only, and a `mut` one is the in-place output buffer a kernel writes.

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
  the wrong number of indices, a compile-time-constant index past a static dimension, a fixed-size
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

Run `wukongc --explain E0502` for a worked example. Dimensions may be integer literals, a declared
symbolic generic name, the name of a top-level `const`, or `?` for a runtime dimension. A `const` dim
resolves to its integer **value**, so it is a fixed dimension like a literal — `const D: usize = 3;
Tensor[f32, 2, D]` really is 3 wide, and `a[1, 3]` on it is `E0501`. A declared generic shadows a
same-named `const`. Any other name is `E0504` rather than a silently-introduced new symbolic dimension.
Tensor element types must be scalars (`E0302`).

**Layout.** A tensor type may carry a physical-layout suffix, which must come last inside the
brackets: `Tensor[f32, M, N, .contiguous]` (the default, row-major), `.col_major`, `.strided`, or
`.tiled(64, 64)`. An unrecognized layout name is `E0204`. The layout is part of the type — sema unifies
it at a call, so a `.col_major` value cannot be routed through a `contiguous`-typed callee (`E0502`).
Today only `contiguous` **executes**: multi-dimensional indexing of a `.col_major`/`.strided`/
`.tiled(…)` tensor is `C0001`, and a fixed-size array decays only to a `contiguous` tensor, so the
other three layouts parse and type-check but cannot yet be indexed (🔵).

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
dim, so a later concrete sibling still binds it and every subsequent operand is checked against that
binding. A constant index past a static tensor dimension, like a fixed-size array, is `E0501`.

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
@inline
@simd
@tile(64, 64)
@parallel(grain = 1)
@align(32)
@extern("C")
@export("wukong_saxpy")
```

Attribute *syntax* is parsed on functions, parameters and statements (a missing argument value is
`E0207`), but the attribute **name is not validated** — an unknown attribute such as `@bogus(1, 2)` is
silently accepted. Exactly one attribute has a consumer today: **`@parallel` on a function** (✅),
which makes its loop execute across CPU cores. Everything else in the list above — `@inline`, `@simd`,
`@tile`, `@align`, `@extern`, `@export`, and `@parallel`'s own `grain = …` argument — is parsed and
discarded (🔵). An attribute on a *statement or loop* is likewise dropped: `@parallel` must sit on the
function, and the mid-function parallel region is derived from the enclosing function's attribute, not
from one written on the loop.

`@parallel` **declines and lowers serially, with no diagnostic**, when the loop cannot be split into
independent chunks: a labelled loop, a plain `break` out of the parallelized loop, a `return` out of
it, or a mid-function region whose modelled body-local is reassigned inside an inner loop. (A
`break`/`continue` targeting an *inner* loop is that loop's own control flow and stays legal.) The
result is always correct — the fallback is the ordinary scalar/vectorized loop — but the parallel
speedup is silently absent, so check with `wukongc --emit=mir -O2 f.wk | grep wukong_parallel_for`.
Fixtures: `tests/run/parallel_control_flow.wk`, `parallel_region_reassigned_local.wk`.

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
softmax / log-softmax / cross-entropy match. Nor does folding the follow-up work **into** a matmul
store rather than writing it as its own loop: `c[i*N+j] = silu(bias[j] + s)` fuses to one
`wukong_sgemm_nt_epi`, `c[i*N+j] = x[i*N+j] + s` (a residual from another array) and a second store
`d[i*N+j] = d[i*N+j] * s` alongside `c[i*N+j] = s` (the SwiGLU up-projection) each lower to the same
GEMM + `wukong_velem_f32` pair the two-loop spelling does. Element type is handled case by
case rather than being an f32-only gate: a `bf16`/`f16` array read through the explicit widening cast —
`s = s + (x[k] as f32)` — dispatches to the multicore low-precision reduction kernels
(`tests/run/parallel_reduce_lowp.wk`), while the *uncast* spelling `s = s + x[k]` stays a scalar loop
(`tests/run/parallel_reduce_elem_bf16.wk`); an `f64` reduction has no kernel and always declines.
`@tile` (cache tiling) and
explicit `@simd`-typed vector values are still under construction (🔵).

## Built-in intrinsics ✅

- `print(x)` — print one value followed by a newline. A signed integer or float prints as itself; an
  **unsigned** integer prints its unsigned magnitude (a high-bit-set `u32`/`u64` is not shown as a
  negative); a `*u8` (a string literal) prints its NUL-terminated bytes rather than the pointer.
  `print()` with no argument prints a bare newline. More than one argument is `E0503` (arguments are
  not concatenated), and a `()` value — from an `else`-less `if`, or a `match` with statement arms —
  is `E0401`.
- `println(x)` — alias of `print` (both already append the newline).
- `assert(cond)` — trap with a nonzero exit code if `cond` is false (zero); a no-op otherwise. Exactly
  one argument (`E0503` otherwise). A float condition means `cond != 0.0`, so `assert(0.5)` passes.
- `now_ns()` — a zero-argument monotonic nanosecond clock returning `i64`, for in-process timing
  (`let t1 = now_ns(); … let dt = now_ns() - t1;`). Any argument is `E0503`. Only *differences* between
  reads are meaningful, so a program's stdout stays backend-identical.
- `sdpa(q, k, v, out, s, d, scale, causal)` — fused scaled-dot-product attention for one `[s, d]`
  head: `out = softmax(scale·Q·Kᵀ [+ causal mask])·V`, computed by one runtime kernel with no S×S score
  buffer. `q`/`k`/`v`/`out` must be `f32` arrays, `[]f32` slices, or tensor views; `s`, `d` and
  `causal` are integers and `scale` is a float. When `s` and `d` are compile-time constants every
  buffer whose type pins an extent must hold `s*d` elements. `sdpa` has no signature in sema, so its
  whole contract is enforced at lowering: a malformed call is a catalogued `C0001` at the call site,
  never a crash (`tests/fail/sdpa_operand_type.wk`, `sdpa_operand_extent.wk`).

These are recognized by the MIR builder and implemented directly by the interpreter (and, on the
native backend, by the runtime). A user-defined function of the same name shadows any of these
builtins.

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
- `silu_backward(x, dy)` / `gelu_backward(x, dy)` / `sigmoid_backward(x, dy)` /
  `tanh_backward(x, dy)` / `elu_backward(x, dy)` / `softplus_backward(x, dy)` — the activation
  **backward** family, each computing `dy · act'(x)` (two arguments; any other arity is not lowered as
  an intrinsic). These are the training-gradient primitives; written as an elementwise `for` loop they
  dispatch to the two-input 256-bit `wukong_vmath2_f32` kernel, and the inlined form is bit-identical
  to it.
- `fmax(a, b)` / `fmin(a, b)`.

These intrinsics also accept **integer** operands: `abs`/`round`/`floor`/`ceil`/`trunc` are
type-preserving on an integer (integer `abs` is `select(x < 0, −x, x)`; rounding an integer is the
identity), while `sqrt` and the transcendentals promote an integer operand to `f32`
(`tests/run/int_math.wk`). The **two-argument** intrinsics follow the promoting rule, not the
preserving one: `fmax`/`fmin`, `pow`, `atan2`, `hypot` and the six `*_backward` intrinsics coerce
*both* operands to the float result type, so `fmax(3, 5)` is the `f32` value `5.0`, not an integer max
(`tests/run/activation_backward_int_args.wk`).

When written as a pure `for i { out[i] = f(x[i]) }` loop over `f32` arrays, **any of the 35**
transcendentals (`exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu`/the inverse trig/the hyperbolic family/…)
are **dispatched to a tuned 256-bit
AVX2/FMA kernel** (`wukong_vmath_f32`) — the same domain-aware lowering as matmul→GEMM — so the
activation family runs ~2–13× faster than C's scalar `libm`, and ~28× across cores under
`@parallel`. `erf`, `sin` and `cos` are among those 35 and dispatch too (this is what a RoPE
`sin`/`cos` loop and an exact erf-GELU loop compile to). What does *not* dispatch is a **composed or
scalar** use — an intrinsic inside a larger expression, or a single call outside an elementwise `for` —
which instead auto-vectorizes the inlined polynomial at 128-bit; `sqrt`/`rsqrt` also stay inline, being
hardware instructions. Either way the inlined form is built from the same primitives as the kernel, so
a dispatched loop and a composed expression agree bit-for-bit, and every form is bit-identical across
the interpreter and native backends.

## Memory and parallelism

**Heap allocation runs today** (✅): `alloc_<T>(n)` returns a **zero-initialized** slice `[]T` of
runtime length `n`, and `free(s)` releases it — the first runtime-sized buffers in the language
(everything else is a fixed-size stack array). The v1 surface is the typed per-scalar family —
`alloc_f32` / `alloc_f64` / `alloc_i32` / `alloc_i64` / `alloc_i8` / `alloc_u8` / `alloc_f16` /
`alloc_bf16` — chosen over a generic `alloc<T>(n)` for a small, predictable surface typed in one
place (a user-defined function of the same name shadows the builtin). The count may be any integer
type; a **negative count yields an empty slice** (`len() == 0`). Contents are deterministically
zero on both backends — calloc'd bytes on native, typed zero values in the interpreter — so a
read-before-write is well-defined. The result is an ordinary slice: `s[i]`, `s.len()`,
`for x in s`, and fn-boundary passing/mutation all compose (`tests/run/heap_*.wk`) — and a slice is a
first-class **kernel operand**: an `alloc_*` buffer handed to a matmul/GEMV nest, a norm, an activation
loop, a reduction, a scan or `sdpa` dispatches to the same tuned runtime kernel a fixed-size `[T; N]`
array does, so a runtime-sized weight blob is not a slow path.

```wukong
let mut acts: []f32 = alloc_f32(tokens * hidden);   // zero-initialized, runtime-sized
for i in 0..acts.len() {
    acts[i] = 1.0;
}
free(acts);
```

Freeing anything not returned by `alloc_*`, double-freeing, or touching a slice after `free` is
**undefined behavior** on the native backend; the interpreter's mark-and-forget model (its
run-scoped memory is never reclaimed) never crashes on it, but such programs are outside the
differential contract. Allocation failure is the one place the two backends differ by construction. On
the native backend a failed (or zero-byte) allocation yields a slice whose data pointer is **null**,
which must not be dereferenced. The interpreter cannot mirror that — its pointers are slot indices and
index 0 is an ordinary live slot — so it reports `interpreter heap exhausted allocating N element(s)` as
a diagnostic and exits 1 instead of returning a null slice. A program must therefore not branch on
allocation success and expect backend-identical behaviour.

No garbage collector and no hidden allocations beyond what you `alloc_*`: *named* allocator
selection (`System`, `Arena`, `Scratch`, `Pool` — 🔵) and cleanup via `defer` (🔵) are still being
wired to the surface. **`@parallel` functions
execute today** (✅): the native backend outlines the loop body into its own function and dispatches it
across CPU cores through the `wukong_runtime` rayon-backed C-ABI entry point
`wukong_parallel_for(n, body, env)`, which hands each worker a contiguous chunk of `[0, n)` and returns
only once every chunk has completed; each per-core chunk is itself auto-vectorized (parallelism ×
SIMD). The interpreter runs the same range sequentially, so results stay differentially equal.

The whole-function vehicle fires on a narrow shape: **every** parameter must be a fixed-size array
(`[T; N]`) — a `[]T` slice, a `Tensor[..]`, a pointer or a scalar parameter declines — the body must be
exactly one unlabelled `for i in 0..hi { … }` over a half-open, unit-step range starting at literal `0`,
and the body must not `return`, `defer`, or `break`/`continue` out of that loop (a `break`/`continue`
inside an *inner* loop is fine). Anything the recognizer cannot prove declines **silently to the
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
alias a capture), a call other than a pure math intrinsic (`print` would reorder output), a
`match`/`loop` expression in the body, a `return`/`defer`, and a labelled or loop-level
`break`/`continue`. At most 32 such regions are outlined per program; further ones stay serial.

## File I/O ✅

**Reading and writing typed blobs runs today** (✅): a small per-scalar family loads and stores raw
binary buffers, so a program can pull weights or a dataset **off disk** instead of synthesizing them.

```wukong
let mut w: []f32 = alloc_f32(768 * 768);       // zero-initialized, runtime-sized
let n: i64 = read_f32("weights.bin", w);        // n = elements actually read
if n < 0 { return 1; }                           // -1 = couldn't open, -2 = io error
// … compute in place …
let written: i64 = write_f32("out.bin", w);      // flush w as raw little-endian f32
```

The surface is `read_<T>(path: *u8, buf: []T) -> i64` and `write_<T>(path: *u8, buf: []T) -> i64`
for **`T` in {`f32`, `f64`, `i32`, `i64`, `i8`, `u8`}** — the same typed, declared-in-one-place convention as the
`alloc_<T>` family. The on-disk format is **headerless raw little-endian**: element `i` of `buf` is
the `sizeof(T)` bytes at byte offset `i · sizeof(T)`, and nothing else — no shape, dtype tag, or
length prefix (you carry those out of band, the way a `.bin` weight shard does).

- **`read_<T>(path, buf)`** fills `buf` from the file and returns the **element count read** =
  `min(buf.len(), file_bytes / sizeof(T))`, so a short file reads only what it has and a long one
  fills `buf` and stops. It returns **`-1`** if the file can't be opened and **`-2`** on an io
  error; on either failure — and for the tail `buf[n..]` left over from a short read — the
  destination slots are **untouched**, so a zero-initialized `alloc_<T>` buffer stays defined.
  **Ordering is part of the contract**: the open is attempted *before* the count is computed, so an
  unopenable path returns `-1` even for an empty `buf` — `read_f32("missing.bin", empty)` is `-1`, not
  `0`. On an openable path an empty `buf` reads nothing and returns `0`. Both backends implement this
  order.
- **`write_<T>(path, buf)`** **creates or truncates** `path`, writes all `buf.len()` elements as
  little-endian bytes, and returns the **count written** (**`-1`** if the file can't be created,
  **`-2`** on an io error).

`path` is any NUL-terminated `*u8` — usually a string literal (the `.rodata` pointer described under
*Literals*), but a `*u8` binding works too. Anything else is `E0401`, and a wrong arity is `E0503`;
`buf` must be a `[]T` slice whose element type matches the name's dtype (a `[]i32` handed to `read_f32`
is `E0401`, not a silent wrong stride). Both directions are implemented by the `wukong_runtime` layer
on the native backend and mirrored **byte-for-byte** by the interpreter — the `-1`/`-2`/count return
codes, the little-endian layout, and the `u8`-narrowing store are identical — so a read/write
round-trip is **differentially
tested to agree bit-for-bit** across the interpreter and the native Cranelift backend, exactly as the
✅ maturity legend requires.

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
--emit=<stage>        tokens | ast | mir-high | mir (alias mir-low) | grad | llvm-ir | obj | exe
                      (default: exe)
                      grad = reverse-mode backward MIR of a loss fn (see --grad-of/--grad-wrt)
--grad-of=<fn>        function to differentiate for --emit=grad / --train   (default: loss)
--grad-wrt=<i,..>     buffer-parameter indices to differentiate w.r.t. (default: every buffer
                      parameter of the loss fn; with --train, every one EXCEPT the last, which is
                      the scalar loss output). An index that is out of range, not a buffer
                      (pointer) parameter, or repeated is an error.
--train               run a fwd→bwd→optimizer loop and print the loss trajectory
--train-steps=<n>     training steps                                       (default: 100)
--train-lr=<f>        learning rate                                        (default: 0.01)
--train-opt=<o>       sgd | adamw                                          (default: sgd)
--train-seed=<n>      seed for the deterministic buffer initialization      (default: 0x5EED1234)
-O0|-O1|-O2|-O3       optimization level (default: -O0; -O3 currently runs the -O2 pipeline)
-o <path>             write the artifact to <path> — only with --emit=obj or --emit=exe (default:
                      <stem>.o for objects; executables get <stem>.exe on Windows and plain <stem>
                      on unix). Rejected with --run and with every other stage, which print their
                      artifact on stdout — redirect instead.
--error-format=<f>    human | json   (default: human)
--explain <CODE>      print an extended explanation for an error code
--color=<when>        auto | always | never   (default: auto — colour when stderr is a terminal)
-h, --help            print the usage text and exit 0
-V, --version         print the version and exit 0
```

Use `--emit` to inspect any stage of the pipeline, e.g. `wukongc --emit=mir -O2 kernel.wk` to see
the optimized IR, or `wukongc --emit=ast kernel.wk` to see the parse tree. `--run` and
`--emit=<stage>` cannot be combined; pick one. A usage error prints `error: <msg>` plus the usage text
on stderr and exits **2**.

Reverse-mode autodiff is CLI-driven too: `wukongc --emit=grad --grad-of=loss model.wk` prints the
backward MIR of a loss function, and `wukongc --train --train-opt=adamw model.wk` runs its
fwd→bwd→optimizer training loop. Both flags force at least `-O1` no matter what level you pass — the
autodiff transform consumes single-block SSA (mem2reg + simplify-cfg) — so `--emit=grad -O0` still
prints optimized MIR. And `--train` wins over the other two dispatches: it is handled before both
`--run` and `--emit=grad`, so `--run --train` trains and never runs the program, and
`--train --emit=grad` trains and never prints the backward MIR. Unlike `--run` with `--emit=<stage>`,
these combinations are accepted rather than rejected — pass `--train` on its own.
