//! The stable error-code catalog.
//!
//! Every diagnostic code Wukong can emit has a stable identifier (`E0501`, `C0001`, …) and an
//! extended explanation here. `wukongc --explain <CODE>` prints the explanation; this keeps the
//! inline diagnostic terse while giving users a place to learn the rule and how to fix it.
//!
//! Code ranges:
//!   * `E00xx` — generic / internal-surface errors
//!   * `E01xx` — lexer
//!   * `E02xx` — parser
//!   * `E03xx` — name resolution
//!   * `E04xx` — type checking
//!   * `E05xx` — shape checking (Wukong's headline analysis)
//!   * `W00xx` — warnings (lints; `W0001`–`W0019` from `wukong_sema::lints`, `W003x` from the
//!     parser), each also named for `-A/-W/-D` and `@allow/@warn/@deny`
//!   * `C0xxx` — codegen / lowering limitations
//!
//! Codes are stable: never renumber one, and never reuse a retired number for a new rule. Retired:
//! `E0001` (was "malformed type syntax"; no stage ever emitted it — the parser reports malformed
//! types as `E0203`/`E0204`).

/// One catalog entry: a stable code, a one-line title, a longer explanation (Markdown-ish), and an
/// optional worked example pair.
pub struct Explanation {
    pub code: &'static str,
    pub title: &'static str,
    pub body: &'static str,
    /// A complete, erroneous `.wk` program whose *first* diagnostic is exactly `code`. Compiled by
    /// `crates/wukongc/tests/explain.rs` (via `--emit=check`), so a catalogue example can never
    /// silently drift away from the rule it documents. `None` where no example has been written yet.
    pub bad: Option<&'static str>,
    /// The corrected counterpart of [`bad`](Self::bad): the same program, fixed, which must compile
    /// clean under `--emit=check`.
    pub good: Option<&'static str>,
}

/// Look up the extended explanation for an error code (case-insensitive).
pub fn explain(code: &str) -> Option<&'static Explanation> {
    let code = code.to_ascii_uppercase();
    CATALOG.iter().find(|e| e.code == code)
}

/// Every code Wukong knows about, for `--explain` and for listing in docs.
pub fn all() -> &'static [Explanation] {
    CATALOG
}

macro_rules! entry {
    ($code:literal, $title:literal, $body:literal) => {
        Explanation {
            code: $code,
            title: $title,
            body: $body,
            bad: None,
            good: None,
        }
    };
    ($code:literal, $title:literal, $body:literal, bad: $bad:literal, good: $good:literal) => {
        Explanation {
            code: $code,
            title: $title,
            body: $body,
            bad: Some($bad),
            good: Some($good),
        }
    };
}

static CATALOG: &[Explanation] = &[
    entry!(
        "E0101",
        "unexpected character",
        "The lexer found a character that cannot begin any Wukong token. Remove it or place it \
         inside a string/char literal. The diagnostic names a non-ASCII character by its code \
         point (`α` (U+03B1)). Identifiers are ASCII-only — letters, digits and `_` — so a Greek \
         letter must be spelled out (`alpha`). A character that merely looks like ASCII \
         punctuation is reported as E0105 instead."
    ),
    entry!(
        "E0102",
        "unterminated string literal",
        "A string literal opened with `\"` was never closed before end of file. Add the closing \
         quote."
    ),
    entry!(
        "E0103",
        "unterminated block comment",
        "A `/* ... */` block comment was never closed. Block comments nest, so make sure every \
         `/*` has a matching `*/`."
    ),
    entry!(
        "E0104",
        "unterminated character literal",
        "A character literal opened with `'` was never closed. Write a single character as `'a'`."
    ),
    entry!(
        "E0105",
        "confusable Unicode character",
        "A character that looks like ASCII punctuation, an operator or whitespace is a different \
         Unicode code point — usually pasted from a paper, a word processor or a chat client: `−` \
         (U+2212 MINUS SIGN) or a dash for `-`, `×` for `*`, `≤` for `<=`, curly quotes for `\"`, \
         a no-break or zero-width space for a space. Wukong source is ASCII outside string \
         literals, character literals and comments, so the character cannot begin a token. The \
         diagnostic names the code point and the ASCII character it imitates: type that \
         instead. The compiler keeps reading as if the ASCII character had been written (a \
         lookalike space is skipped), so one pasted character costs one error and nothing \
         downstream of it is misreported; a lookalike quote cannot open a literal and is an \
         error token."
    ),
    entry!(
        "E0200",
        "expected a specific token",
        "The parser needed a particular token (for example a closing `)` or `}`) but found \
         something else. Check for a missing delimiter or separator. The message lists every \
         token the parser would have accepted at that point (`expected one of `,` or `)``). When \
         the parser had already moved on to the next line, the error points just after the last \
         token of the previous line, where the missing token belongs; a missing closer also \
         labels the opening delimiter that was never closed."
    ),
    entry!(
        "E0201",
        "expected an identifier",
        "A name was required here — for instance after `fn`, `let`, or `struct`."
    ),
    entry!(
        "E0202",
        "expected an expression",
        "An expression was required (for example after `=` or `return`) but the parser found a \
         token that cannot start one."
    ),
    entry!(
        "E0203",
        "expected a type or dimension",
        "A type was required here, or — inside a tensor/vector type — a dimension. Dimensions are \
         integer literals, generic names, or `?` for a runtime dimension."
    ),
    entry!(
        "E0204",
        "unknown tensor layout",
        "A tensor layout suffix was not recognized. Valid layouts are `contiguous`, `col_major`, \
         `strided`, and `tiled(...)`."
    ),
    entry!(
        "E0205",
        "malformed turbofish",
        "A turbofish `::<...>` generic-argument list must be followed by a call `(...)`."
    ),
    entry!(
        "E0206",
        "expected a pattern",
        "A binding pattern was required here, e.g. in `let`."
    ),
    entry!(
        "E0207",
        "expected an attribute argument",
        "An attribute like `@align(...)` or `@tile(...)` expected an argument but none was given."
    ),
    entry!(
        "E0208",
        "expected an item",
        "At module scope Wukong expects an item: `fn`, `struct`, `enum`, `const`, `import`, or \
         an `extern` block."
    ),
    entry!(
        "E0209",
        "nesting too deep",
        "An expression or type nests more deeply than the parser allows (for example thousands of \
         nested parentheses or array types, or an enormously long `a + b + c + …` operator chain). \
         The limit exists so that pathological or machine-generated input fails with this stable \
         diagnostic instead of crashing the compiler with a stack overflow. No realistic program \
         comes close; if you hit this, restructure the deeply nested expression or type — for \
         instance by introducing intermediate `let` bindings."
    ),
    entry!(
        "E0300",
        "duplicate definition",
        "A name was defined more than once in the same scope. Rename one of the definitions."
    ),
    entry!(
        "E0301",
        "name not found",
        "A referenced name does not resolve to any definition in scope. Check for a typo or a \
         missing `import`."
    ),
    entry!(
        "E0302",
        "invalid element type",
        "The element type of a SIMD vector or tensor must be a scalar (e.g. `f32`, `i32`), not a \
         compound type."
    ),
    entry!(
        "E0303",
        "`break`/`continue` outside of a loop",
        "A `break` or `continue` statement appeared outside of any enclosing `while`, `for`, or \
         `loop`. These statements only have meaning inside a loop body. Remove it, or wrap the code \
         in a loop.",
        bad: "module ex.e0303\nfn main() -> i32 {\n    break;\n    return 0;\n}\n",
        good: "module ex.e0303\nfn main() -> i32 {\n    let i: i32 = 0;\n    while i < 1 {\n        break;\n    }\n    return 0;\n}\n"
    ),
    entry!(
        "E0304",
        "assignment to immutable binding",
        "A binding introduced with `let` (without `mut`) cannot be reassigned. Declare it `let mut` \
         to allow reassignment, or introduce a new binding with another `let`. Mutating *through* a \
         `let` binding — an array element `a[i] = …` or a struct field `s.f = …` — is still \
         allowed; only rebinding the name itself is rejected. Mutating through a non-`mut` \
         *aggregate parameter* is NOT: an aggregate parameter is passed by reference, so \
         `fn f(p: S) { p.a = 9; }` would write into the caller's value — add `mut` to the parameter \
         (`fn f(mut p: S)`). Writing through a pointer parameter (`*p = …`) needs no `mut`, because \
         the pointer itself is not being reassigned.",
        bad: "module ex.e0304\nfn main() -> i32 {\n    let x: i32 = 5;\n    x = 10;\n    return x;\n}\n",
        good: "module ex.e0304\nfn main() -> i32 {\n    let mut x: i32 = 5;\n    x = 10;\n    return x;\n}\n"
    ),
    entry!(
        "E0305",
        "unresolved import",
        "An `import` does not resolve to a source file. `import a.b.c` resolves to \
         `<root-dir>/a/b/c.wk`, where `<root-dir>` is the directory containing the root source \
         file passed to the compiler and dots are path separators (a single segment `import util` \
         is `<root-dir>/util.wk`). Check the spelling and the on-disk layout. Note that the \
         imported file's own `module` header is informational — it does not participate in \
         resolution and need not match the import path."
    ),
    entry!(
        "E0306",
        "unknown or misplaced attribute",
        "An `@attribute` is not one Wukong knows, or is not legal where it was written. The known \
         set is closed: `@parallel`, `@generic`, `@tile`, `@extern`, `@noalias`, `@unchecked`, \
         `@strict_fp`, `@inline`, `@simd`, `@align`, `@export`. Anything else — most importantly a \
         typo of a real one, `@parellel` — used to be accepted and discarded, so the program \
         compiled with none of the requested behavior and no diagnostic. Placement: `@generic` (opt \
         out of kernel recognition) is read from a `fn` item (the whole function) or a `for`/`while` \
         loop (that loop and everything nested in it), so it is rejected on any other statement, a \
         parameter, or a non-`fn` item rather than silently promising an opt-out nothing reads. \
         `@strict_fp` (source-order float evaluation) and `@unchecked` (drop this function's slice \
         bounds checks) are read from the `fn` item only, so each is rejected on a statement, a \
         parameter, or a non-`fn` item rather than silently promising a per-loop opt-out. \
         `@noalias` describes one parameter's storage, so it is legal only on a parameter, takes no \
         arguments, and only on a parameter passed by pointer (an array, slice, tensor, pointer, \
         reference, tuple or struct) — on a scalar there is no storage to promise anything about. \
         Arguments: `@parallel` takes nothing or exactly `grain = G` with `1 <= G <= 2^31` (each \
         worker then claims runs of at least G iterations); an unknown key, a positional argument \
         or an out-of-range grain is rejected instead of discarded. The lint-level attributes \
         `@allow`, `@warn` and `@deny` are known too and legal anywhere; their arguments are lint \
         names, and an unknown one is the warning `W0007`, not this error. The parsed-but-discarded \
         attributes (`@inline`, `@simd`, `@align`, `@export`) stay legal wherever they parse."
    ),
    entry!(
        "E0307",
        "shared array too large",
        "A `shared [T; N]` array's static size exceeds the per-CTA shared memory budget (48 KiB in \
         v1). Use a smaller tile or split the staging across multiple barriers."
    ),
    entry!(
        "E0308",
        "sync_block outside @parallel",
        "`sync_block()` is a cooperative thread-block barrier and is only meaningful inside a \
         `@parallel` function. On CPU backends it is a no-op, but the language requires the SPMD \
         context so misuse is rejected at compile time."
    ),
    entry!(
        "E0309",
        "`@tile` cannot be applied",
        "A `@tile` attribute names a tiling the compiler cannot perform, so it would have no \
         effect. Four conditions reach this code. (1) The loop form: `@tile` strip-mines \
         `for <i> in <start>..<end>` — one identifier, an exclusive range with a known end and no \
         `step` — into a full-tile nest plus a remainder loop. `for i in 0..=n`, a stepped or open \
         range, and iteration over an array cannot be rewritten that way. (2) The argument: a tile \
         width must be a positive integer literal, so a bare `@tile`, `@tile(0)`, `@tile(rows)` and \
         a width that does not fit in 64 bits are all rejected. (3) The width was never applied: a \
         function-item `@tile(BM, BN, …)` is a queue consumed by loops in source order, and a width \
         still in it when the body is done tiled nothing. (4) The lowering cannot carry it: a \
         whole-function recognizer interception replaces the nest with a kernel that brings its own \
         blocking, and the whole-function `@parallel` outliner rewrites the loop into runtime \
         chunks — both build their lowerer with no tile queue at all, so `@generic` is the way to \
         keep the loops tileable. Every one of these used to be silent — the width was discarded \
         where it stood, and a function-item queue additionally SHIFTED, so the next loop was \
         strip-mined at the wrong width. Either drop the attribute or write the loop in the form it \
         can rewrite."
    ),
    entry!(
        "E0312",
        "private item",
        "An item of another module was named, but that module does not export it. Every file is a \
         module with its own namespace, and its top-level items (`fn`, `struct`, `enum`, `const`, \
         and the functions of an `extern` block) are private to it unless declared `pub`. A private \
         item is reachable from its own file only: `import m.{x}` and the qualified `m::x` reject \
         it, and a plain `import m;` simply does not bring it into scope. Declare the item `pub` in \
         its module, or keep it private and call the public function that wraps it."
    ),
    entry!(
        "E0313",
        "ambiguous name",
        "An unqualified name is brought into scope by two different plain imports (`import a;` and \
         `import b;` both export an item with that name), or a module qualifier is bound by two \
         imports whose last path segment is the same (`import x.util;` and `import y.util;` both \
         bind `util`). Importing both is not an error; only the ambiguous USE is, so the compiler \
         never silently picks one module's item. Qualify the use (`a::name`), import the intended \
         item explicitly (`import a.{name};`, which takes precedence over plain imports), or give \
         one import a distinct name with `as`. A definition in the file itself always shadows a \
         plain import."
    ),
    entry!(
        "E0314",
        "invalid `defer`",
        "`defer e;` runs `e` at every exit of the enclosing block — falling off its end, `return`, \
         and `break`/`continue` out of it — in reverse order of registration, innermost block \
         first; a returned or broken value is computed before the defers run. Two things would \
         make that ill-defined and are rejected. (1) Control flow leaving the deferred \
         expression: a `return`, or a `break`/`continue` aimed at a loop outside it, inside `e` \
         (a loop entirely inside `e` may break out of itself). (2) Rebinding a name `e` uses while \
         the defer is pending: `e` is resolved where the `defer` is written, but its copies are \
         placed at the exits, where a newer `let x` (or a `for x` / match binding) would silently \
         retarget it. Rename the new binding, or end the block holding the `defer` before \
         rebinding."
    ),
    entry!(
        "E0315",
        "unresolved trait, impl target or method",
        "A name used by a method, trait or bound does not resolve. Four situations reach this code: \
         a method call names no method of the receiver's type (`v.nrom()` on a struct with no \
         `nrom`, or `Type::f(..)` naming no associated function); an associated function (one with \
         no `self`) is called with method syntax, or a trait method is called the other way round; \
         an `impl` names a type that is not a struct or enum of this program, or a trait that does \
         not exist (likewise a bound `T: Tr` or a `where` clause naming no type parameter); or a \
         method is called on a type parameter that none of its trait bounds provides — add the \
         bound (`fn f<T: Shape>(x: T)`). Fix the spelling, declare the method in an `impl` block, \
         or bound the type parameter by a trait that has it."
    ),
    entry!(
        "E0316",
        "`@noalias` promise broken",
        "A `@noalias` parameter promises that, for the duration of a call, the storage it points \
         into is reached only through that parameter — nothing else the call can name overlaps it \
         (C's `restrict`, LLVM's `noalias`). The optimizer relies on it: a store through the \
         promised parameter is assumed not to change any other argument, so loads move across it, \
         loops vectorize without a runtime alias check, and recognized kernels dispatch. Two \
         spellings break the promise provably and are rejected. (1) A call passes the promised \
         parameter's argument together with another buffer argument rooted in the same storage: \
         `k(a, a)`, `k(s, a)` after `let s: []f32 = a`, `k(p.w, p)` (a field and its struct), or \
         `k(m[0], m[1])` (element indices are not compared, so two elements of one array count as \
         one buffer). Two different fields of one struct, `k(p.w, p.b)`, are fine. Copy one side \
         into a fresh local first, or drop the attribute if the callee must accept aliased \
         arguments. (2) A `@noalias` parameter of view type (`[]T`, `*T`, `&T`, a tensor) is \
         reassigned as a whole (`s = t`), which would point the promised name at storage the \
         promise never covered; write through it (`s[i] = …`) or bind a local view instead. A call \
         whose arguments the compiler cannot prove to overlap is accepted: if they do overlap, the \
         program has broken its promise, and optimized native code may differ from the \
         interpreter, exactly as with `restrict` in C."
    ),
    entry!(
        "E0320",
        "cannot find function",
        "A call names a function that does not exist: no `fn` of this program, no function-typed \
         parameter in scope, and no builtin (`sqrt`, `matmul`, `alloc_f32`, `read_f32`, `print`, \
         …) has that name. The usual cause is a typo, and the diagnostic suggests the closest \
         existing name. A function of another module is only visible after an `import` of that \
         module (and only if it is `pub`); a method is called on its value, `v.norm()`, or as \
         `Type::norm(v)`. This used to compile as an unmodelled call and fail in lowering with the \
         misleading codegen limit `C0001`.",
        bad: "module ex.e0320\nfn square(x: f32) -> f32 {\n    return x * x;\n}\nfn main() -> i32 {\n    return sqare(3.0) as i32;\n}\n",
        good: "module ex.e0320\nfn square(x: f32) -> f32 {\n    return x * x;\n}\nfn main() -> i32 {\n    return square(3.0) as i32;\n}\n"
    ),
    entry!(
        "E0321",
        "no such field",
        "A field access names a field the value's type does not have: `p.z` on a struct with only \
         `x` and `y`, a field of an enum value, or any field of a builtin type (`x.re` on an \
         `f32`, `t.0` on a tensor). The diagnostic lists the struct's fields and suggests the \
         closest one. Only a struct has named fields and only a tuple has numbered ones (`t.0`); \
         an array, slice, tensor or pointer element is read by indexing (`a[i]`), and `len`/`dim` \
         are methods (`a.len()`).",
        bad: "module ex.e0321\nstruct Point { x: f32, y: f32 }\nfn main() -> i32 {\n    let p = Point { x: 1.0, y: 2.0 };\n    return p.z as i32;\n}\n",
        good: "module ex.e0321\nstruct Point { x: f32, y: f32 }\nfn main() -> i32 {\n    let p = Point { x: 1.0, y: 2.0 };\n    return p.y as i32;\n}\n"
    ),
    entry!(
        "E0322",
        "no such method on a builtin type",
        "A method call on a value of a builtin type names a method that type does not have. The \
         builtin method surface is small: `len()` on an array, slice or tensor and `dim(k)` on an \
         array or tensor, plus the explicit-SIMD spellings (`row_ptr`, `offset`, `store`, `fma`) \
         that type-check but are not lowered yet. Math functions are free functions, not \
         methods: write `sqrt(x)`, not `x.sqrt()`. A struct or enum's methods are the ones its \
         `impl` blocks define (a missing one is `E0315`).",
        bad: "module ex.e0322\nfn main() -> i32 {\n    let x: f32 = 4.0;\n    return x.sqrt() as i32;\n}\n",
        good: "module ex.e0322\nfn main() -> i32 {\n    let x: f32 = 4.0;\n    return sqrt(x) as i32;\n}\n"
    ),
    entry!(
        "E0323",
        "expected function, found a non-function",
        "Something that is not a function is called: a local variable or parameter (`scale(3.0)` \
         with `scale: f32`), a constant, a struct or enum name, a type (`f32(x)` — convert with \
         `x as f32`), a struct field (`p.w(1)`), or an expression whose value is not a function. \
         A common cause is a missing `;`: a `(` that begins a line continues the expression on \
         the line above, so `acc = a[i]` followed by `(b + c) * 2;` is the call `a[i](b + c)`. \
         Only functions and function-typed parameters can be called.",
        bad: "module ex.e0323\nfn main() -> i32 {\n    let scale: f32 = 2.0;\n    let y: f32 = scale(3.0);\n    return y as i32;\n}\n",
        good: "module ex.e0323\nfn main() -> i32 {\n    let scale: f32 = 2.0;\n    let y: f32 = scale * 3.0;\n    return y as i32;\n}\n"
    ),
    entry!(
        "E0324",
        "cannot find type",
        "A type names nothing: it is not a scalar (`f32`, `i64`, `bool`, …), not `Tensor[..]` or \
         `vec[..]`, not a struct or enum declared in this program, and not a generic parameter of \
         the enclosing item. The diagnostic suggests the closest type, and the Wukong spelling of \
         a C or Python name (`float` is `f32`, `int` is `i32`, `double` is `f64`). A struct \
         literal of an undeclared name, and a trait used where a type belongs, are reported with \
         this code too (bound a generic parameter by the trait instead: `fn f<T: Shape>(x: T)`). \
         An unknown type name used to be accepted and silently lowered as a 32-bit integer slot.",
        bad: "module ex.e0324\nfn main() -> i32 {\n    let x: flot = 2.5;\n    return x as i32;\n}\n",
        good: "module ex.e0324\nfn main() -> i32 {\n    let x: f32 = 2.5;\n    return x as i32;\n}\n"
    ),
    entry!(
        "E0401",
        "type mismatch",
        "A value's type does not match the type its context requires: a `let` annotation, an \
         assignment's place, the function's declared return type, a call's parameter, a struct \
         field, or the other arm of an `if`/`match`. Unsuffixed numeric literals adapt to the \
         expected type, but typed values must match exactly — convert with `as`. A few other type \
         errors carry this code too: an `as` cast between types that have no conversion, `break` \
         with a value out of a loop whose value is discarded, a type too large to lay out, and an \
         array longer than `u32::MAX` elements. The more specific families have codes of their \
         own: struct-literal fields (`E0430`), literals (`E0431`), escapes (`E0432`), operators \
         (`E0433`), a missing return value (`E0434`), enum payloads (`E0435`), conditions \
         (`E0436`) and the arguments of builtins (`E0437`).",
        bad: "module ex.e0401\nfn main() -> i32 {\n    let a: i64 = 5;\n    let b: i32 = a;\n    return b;\n}\n",
        good: "module ex.e0401\nfn main() -> i32 {\n    let a: i64 = 5;\n    let b: i32 = a as i32;\n    return b;\n}\n"
    ),
    entry!(
        "E0402",
        "recursive struct or enum has infinite size",
        "A struct, or a data-carrying enum, contains itself by value — directly \
         (`struct S { x: S }`, `enum List { Cons(i32, List), Nil }`) or through a chain of such \
         types — so its size would be infinite and the compiler cannot lay it out. Store the \
         recursive field behind a pointer (e.g. `*S`), which has a fixed size and breaks the cycle, \
         as in C or Rust."
    ),
    entry!(
        "E0403",
        "recursive const initializer",
        "A `const`'s initializer depends on its own value — directly (`const A: i32 = A + 1;`) or \
         through a chain of consts (`A` uses `B`, `B` uses `A`). A const must be evaluable at \
         compile time without referring back to itself; inlining such a cycle would never \
         terminate. Break the cycle so each const's value is defined in terms of already-known values."
    ),
    entry!(
        "E0405",
        "non-exhaustive match",
        "A `match` does not cover every possible value of the scrutinee, and no arm is an \
         unconditional catch-all. This is rejected in every position, not just value position: a \
         value match would produce an injected zero default when no arm matches — a silent wrong \
         answer (or, for a tuple/struct result, invalid code) — while a statement or empty match \
         falls through to `Unreachable`, which the interpreter traps but the native backend hits as \
         an illegal instruction. Add a `_ => …` arm (or, for an `enum`, an arm for every remaining \
         variant; for a `bool`, both `true` and `false`) so the match is total, as in Rust. Guarded \
         arms (`if …`) do not count toward coverage because their guard may be false.",
        bad: "module ex.e0405\nenum Color { Red, Green, Blue }\nfn name(c: Color) -> i32 {\n    return match c {\n        Color::Red => 1,\n        Color::Green => 2,\n    };\n}\nfn main() -> i32 {\n    return name(Color::Blue);\n}\n",
        good: "module ex.e0405\nenum Color { Red, Green, Blue }\nfn name(c: Color) -> i32 {\n    return match c {\n        Color::Red => 1,\n        Color::Green => 2,\n        Color::Blue => 3,\n    };\n}\nfn main() -> i32 {\n    return name(Color::Blue);\n}\n"
    ),
    entry!(
        "E0406",
        "array length is not a compile-time constant",
        "A fixed-size array's length is part of its type, so it must be readable at compile time: \
         an integer literal, a `const` (possibly through a chain of consts), a non-negative C-style \
         enum variant, or arithmetic on those. Anything else — a cast (`[i32; 4 as usize]`), a \
         `sizeof`/`alignof`, a call, a unary operator, a multi-segment path — cannot be folded, and \
         was previously accepted in silence: the type checker recorded length 0 while lowering \
         emitted no array slot at all, just an uninitialized pointer that the program then stored \
         through. A GENERIC parameter is rejected for the same reason and needs the same fix: \
         `Ty::Array` carries a plain integer length, with no representation for a symbolic extent, \
         so `[T; N]` inside `fn f<N>(..)` folded to length 0. Where the extent really is a \
         per-call-site parameter, spell the value `Tensor[T, N]`, whose dimensions are shape-checked \
         symbolically and bound at each call; where it is a fixed size, declare a `const`."
    ),
    entry!(
        "E0409",
        "kernel builtin dimension cannot be inferred",
        "A named kernel builtin (`matmul`, `linear`, `transpose`, `softmax`, `layernorm`, `gelu`, \
         `sum`, …) is shape-checked like a generic function: each of its dimensions must be fixed \
         by an explicit turbofish (`matmul::<M, K, N>(a, b, out)`), by the shape of a \
         `Tensor[f32, ..]` operand, by the length of a fixed `[f32; N]` array operand (which fixes \
         the one dimension of its shape that is still free), or by the runtime length of a `[]f32` \
         slice operand. A row-wise operation's row count defaults to 1. This error means none of \
         those fixed the named dimension — for example `matmul(a, b, out)` over three flat arrays, \
         where a 6-element `a` could be 2x3 or 3x2. Supply the dimensions explicitly with a \
         turbofish, or pass tensors whose types carry the shape."
    ),
    entry!(
        "E0410",
        "trait implementation or bound not satisfied",
        "A trait contract is broken. Either an `impl Trait for Type` does not match the trait — a \
         trait method is missing, an extra method is not a member of the trait, a method's \
         receiver, parameters or return type differ from the trait's signature (with `Self` read as \
         `Type`), or a trait method has a default body or generic parameters, which traits do not \
         support — or a generic function bounded by a trait is called at a type that has no \
         `impl Trait for Type`. Dispatch is static (each call instantiates the function at a \
         concrete type), so the instance would have no method to call. Implement the trait for the \
         type, or call the function at a type that implements it."
    ),
    entry!(
        "E0411",
        "function value outside a function-typed parameter",
        "Functions are passed by name to parameters of function type (`fn apply(f: fn(f32) -> f32, \
         x: f32)`), and the callee is specialized for that function at compile time — there are no \
         function pointers at run time. So a function type may only be the type of a parameter of a \
         function with a body (not a local, field, element, return type or `extern` parameter); the \
         argument must name a non-generic top-level function or forward the caller's own \
         function-typed parameter; a function may not be bound to a `let` or passed to a \
         non-function parameter; and a function-typed parameter may be neither `mut` nor named \
         after a builtin (`gelu`, `print`, …), whose name the compiler reads as the builtin. Pass \
         the function directly at each call instead."
    ),
    entry!(
        "E0430",
        "struct literal field error",
        "A struct literal (and a struct-payload enum variant, `E::V { .. }`) initializes every field \
         of its type exactly once and nothing else. A field the type does not have, a field written \
         twice, and a field left out are each rejected — a missing field would otherwise be an \
         uninitialized read. Rust's functional-update shorthand `..base` is not supported. A field \
         VALUE of the wrong type is the plain mismatch `E0401`.",
        bad: "module ex.e0430\nstruct Point { x: i32, y: i32 }\nfn main() -> i32 {\n    let p = Point { x: 7 };\n    return p.x;\n}\n",
        good: "module ex.e0430\nstruct Point { x: i32, y: i32 }\nfn main() -> i32 {\n    let p = Point { x: 7, y: 0 };\n    return p.x;\n}\n"
    ),
    entry!(
        "E0431",
        "malformed or out-of-range literal",
        "A numeric literal that does not parse (`0z12`, an empty radix `0x`, a bad digit `0b2`, \
         `1.5e`), or an integer literal whose value does not fit the type it is used at — the type \
         an unsuffixed literal adapts to from an annotation, a parameter, a field, an assignment, a \
         return type or a `match` scrutinee (`let x: i8 = 200`) — is rejected instead of wrapping \
         silently. A constant expression of literals is checked by its folded value \
         (`let v: i8 = 100 + 100`). An enum discriminant must be such a constant and fit the \
         32-bit enum representation. Use a wider type, or convert with `as`, which wraps by \
         definition.",
        bad: "module ex.e0431\nfn main() -> i32 {\n    let x: i8 = 200;\n    return x as i32;\n}\n",
        good: "module ex.e0431\nfn main() -> i32 {\n    let x: i16 = 200;\n    return x as i32;\n}\n"
    ),
    entry!(
        "E0432",
        "invalid escape in a literal",
        "A `\\` escape in a string or character literal must be one the language decodes: `\\n`, \
         `\\r`, `\\t`, `\\\\`, `\\'`, `\\\"`, `\\0`, `\\xHH`, or `\\u{..}` — braces required and \
         closed — naming a Unicode scalar value (at most `\\u{10FFFF}`, and not a surrogate in \
         `\\u{D800}`..`\\u{DFFF}`). Any other escape used to decode to a string the source never \
         wrote (`\"A\\dB\"` printed `AdB`). Write `\\\\` for a literal backslash.",
        bad: "module ex.e0432\nfn main() -> i32 {\n    println(\"A\\dB\");\n    return 0;\n}\n",
        good: "module ex.e0432\nfn main() -> i32 {\n    println(\"A\\\\dB\");\n    return 0;\n}\n"
    ),
    entry!(
        "E0433",
        "operator not defined for the operand type",
        "An operator is applied to a value of a type it is not defined for: arithmetic, bitwise or \
         shift operators on a struct, tuple, array, `()`, pointer (no pointer arithmetic — take \
         `&a[i]`) or whole tensor (operate elementwise); arithmetic or ordered comparison on \
         `bool`; bitwise, shift or `!`/`~` on a float; `==`/`!=` on an aggregate (compare its \
         fields); a comparison of `bool` with a number or of signed with unsigned integers (a \
         chained comparison `a < b < c` parses as `(a < b) < c`); a compound assignment such as \
         `a += b` on an aggregate; `*x` on a non-pointer; and indexing `x[i]` on a scalar, struct \
         or tuple, or with a `bool` index.",
        bad: "module ex.e0433\nstruct P { x: i32 }\nfn main() -> i32 {\n    let a = P { x: 1 };\n    let b = P { x: 1 };\n    if a == b {\n        return 1;\n    }\n    return 0;\n}\n",
        good: "module ex.e0433\nstruct P { x: i32 }\nfn main() -> i32 {\n    let a = P { x: 1 };\n    let b = P { x: 1 };\n    if a.x == b.x {\n        return 1;\n    }\n    return 0;\n}\n"
    ),
    entry!(
        "E0434",
        "missing return value",
        "A function that declares a return type must produce a value on every path: its body can \
         fall off its end (an `if` with no `else`, a loop that can exit), or a bare `return;` \
         returns nothing. Either would hand the caller an uninitialized value. Return a value on \
         every path, as in Rust.",
        bad: "module ex.e0434\nfn pick(n: i32) -> i32 {\n    if n > 0 {\n        return 1;\n    }\n}\nfn main() -> i32 {\n    return pick(-5);\n}\n",
        good: "module ex.e0434\nfn pick(n: i32) -> i32 {\n    if n > 0 {\n        return 1;\n    }\n    return 0;\n}\nfn main() -> i32 {\n    return pick(-5);\n}\n"
    ),
    entry!(
        "E0435",
        "enum variant payload misused",
        "A variant is written or matched with a payload shape it does not have: a tuple variant \
         constructed or destructured with the wrong number of fields (`Op::Add(1)` for \
         `Add(i32, i32)`), a payload variant used bare as a value (`Op::Add`), a unit variant given \
         a payload, or a tuple variant written with braces / a struct variant with parentheses. \
         Write the variant the way it is declared.",
        bad: "module ex.e0435\nenum Op { Add(i32, i32), Nop }\nfn main() -> i32 {\n    let a = Op::Add(1);\n    return 0;\n}\n",
        good: "module ex.e0435\nenum Op { Add(i32, i32), Nop }\nfn main() -> i32 {\n    let a = Op::Add(1, 2);\n    return match a {\n        Op::Add(x, y) => x + y,\n        Op::Nop => 0,\n    };\n}\n"
    ),
    entry!(
        "E0436",
        "condition is not a boolean, number or pointer",
        "The condition of an `if` or `while` must be a `bool`, a number (nonzero is true) or a \
         pointer (non-null is true). A `()` value — a call of a function that returns nothing, an \
         `if` with no `else` — or an aggregate, slice, tensor view, SIMD vector or function value \
         has no truth value. Compare explicitly (`if s.len() > 0`).",
        bad: "module ex.e0436\nfn nothing() {}\nfn main() -> i32 {\n    if nothing() {\n        return 1;\n    }\n    return 0;\n}\n",
        good: "module ex.e0436\nfn ready() -> bool {\n    return true;\n}\nfn main() -> i32 {\n    if ready() {\n        return 1;\n    }\n    return 0;\n}\n"
    ),
    entry!(
        "E0437",
        "builtin argument of the wrong kind",
        "A builtin was given an argument of a kind it cannot take: a math function (`sqrt`, `exp`, \
         …) a struct, pointer or whole tensor; `alloc_<T>`/`scratch_<T>` a non-integer count; \
         `free` something that is not a slice; `print` a `()` value or a whole tensor; a file-I/O \
         builtin a path that is not a `*u8` string, an offset that is not an integer, or a buffer \
         of the wrong element type or with a `?` dimension; an RNG builtin a float seed; a kernel \
         builtin (`matmul`, `sdpa`, …) an operand that is not an `f32` buffer or a non-integer \
         dimension. The message names the argument and what it must be.",
        bad: "module ex.e0437\nfn main() -> i32 {\n    let s = alloc_f32(2.5);\n    free(s);\n    return 0;\n}\n",
        good: "module ex.e0437\nfn main() -> i32 {\n    let s = alloc_f32(4);\n    free(s);\n    return 0;\n}\n"
    ),
    entry!(
        "E0501",
        "rank, extent or arity mismatch",
        "Two tensors were required to have the same number of dimensions (rank) but did not. For \
         example, passing a rank-3 tensor where a rank-2 tensor is expected. This is checked at \
         compile time — it can never become a runtime shape bug.\n\nThis code is also reported \
         when a compile-time-constant index is out of bounds — for a fixed-size array (e.g. \
         `a[5]` on a `[T; 4]`) or for a static tensor dimension (e.g. `a[5, 0]` on a \
         `Tensor[f32, 2, 2]`): the valid indices are `0..len` on each axis.\n\nThe same code \
         covers the non-tensor extents, which are checked by the same unifier and mean the same \
         thing: an ARRAY LENGTH mismatch at a call site (a `[i32; 2]` supplied where `[i32; 8]` is \
         declared), a TUPLE ARITY mismatch at a call site (a 2-element tuple supplied where a \
         3-element tuple is declared), and a wrong NUMBER OF INDICES on an array, a slice or a \
         pointer (`a[i, j]` on a `[i32; 4]`, which is rank 1 — only a tensor takes one index per \
         axis)."
    ),
    entry!(
        "E0502",
        "tensor dimension mismatch",
        "Tensor dimensions did not agree. In `matmul(a: Tensor[f32, M, K], b: Tensor[f32, K, N])` \
         the inner dimension `K` must match on both sides; supplying `Tensor[f32, 512, 512]` and \
         `Tensor[f32, 513, 512]` is a compile error because `512 != 513`. Symbolic dimensions \
         unify across a call, so a single conflicting binding is reported here too."
    ),
    entry!(
        "E0503",
        "wrong number of arguments",
        "A call supplied the wrong number of value or generic arguments for the callee's \
         signature."
    ),
    entry!(
        "E0504",
        "unknown tensor dimension",
        "A tensor type named a dimension that is not a declared generic parameter (nor an integer \
         literal, `?`, or a `const`). For example `fn f<K>(a: Tensor[f32, KK])` — `KK` is a typo for \
         the declared `K`. Previously such a name was silently treated as a brand-new symbolic \
         dimension, quietly dropping the constraint the programmer intended (e.g. that two operands \
         share `K`); it is now an error. Declare the dimension in the generic list (`<KK>`), fix the \
         typo, or use an integer / `?`."
    ),
    entry!(
        "E0505",
        "dimension expression has no extent",
        "A tensor dimension may be arithmetic on dimensions — `Tensor[f32, 3*D]`, \
         `Tensor[f32, S, H*HD]`, `Tensor[f32, (L - K)/S + 1]` — using `+ - * / %` (`/` is floor \
         division, `%` the matching non-negative remainder) on integer literals, generic \
         dimensions and `const`s. Such an expression must denote an extent: it may not fold to a \
         negative constant (`2 - 5`), divide by zero or by a negative constant, or overflow the \
         64-bit extent range; and a `?` dimension cannot be an operand, since it has no value to \
         compute with (name the dimension with a generic parameter instead). The same rule applies \
         at a call: `fn shrink<N>(a: Tensor[f32, N]) -> Tensor[f32, N - 4]` called with a length-2 \
         tensor has no result shape."
    ),
    entry!(
        "E0506",
        "reshape changes the element count",
        "`x as Tensor[T, d0, d1, ...]` views a contiguous buffer (a tensor, or a fixed-size array \
         of scalars) under a new shape without copying, so the new shape must hold exactly the \
         same number of elements — PROVABLY, for every value of the dimensions. `x as \
         Tensor[f32, S, H, HD]` from `Tensor[f32, S, H*HD]` is accepted (`S*H*HD` both ways); from \
         `Tensor[f32, S, D]` it is rejected, because `D` is not provably `H*HD` — write the \
         source's shape with the factor spelled out. A `?` dimension has no compile-time extent, so \
         a reshape involving one cannot be proved and is rejected too."
    ),
    entry!(
        "E0507",
        "invalid tensor view or extent query",
        "Indexing a tensor with fewer indices than its rank selects a VIEW of the remaining axes \
         (`a[i]` on `Tensor[f32, M, N]` is the row, a `Tensor[f32, N]` naming part of `a`'s \
         buffer). Only a contiguous (row-major) tensor has such views — the rows of a \
         `.col_major`, `.strided` or `.tiled` tensor are not contiguous. A view is not a place a \
         value can be assigned to (`a[i] = ...` would have to copy a whole row): assign its \
         elements with one index per axis. The axis of `t.dim(k)` must be a compile-time \
         constant, because each axis of a shape-typed value has its own extent."
    ),
    entry!(
        "E0520",
        "cannot infer a dimension of a call",
        "Every dimension a function's parameter shapes use must be known at each call, because the \
         compiled function receives it as a hidden argument. A dimension is known when an explicit \
         turbofish gives it (`f::<3, 4>(a)`), when an argument's tensor shape or a fixed array's \
         length fixes it, when it can be solved from a derived extent or an element count (a \
         `Tensor[f32, N, 4]` fed a `[f32; 12]` has `N = 3`), or, at run time, from a `?` axis of a \
         tensor argument. This error means none of those applies — typically a multi-axis tensor \
         parameter fed a flat array whose length could be split over its free axes in more than \
         one way (`Tensor[f32, M, N]` fed 12 elements). Supply the dimensions with a turbofish, or \
         pass a tensor whose type spells its shape. It used to surface only in lowering, as the \
         codegen limit `C0001`.",
        bad: "module ex.e0520\nfn total<M, N>(a: Tensor[f32, M, N]) -> f32 {\n    let mut s: f32 = 0.0;\n    for i in 0..M {\n        for j in 0..N {\n            s = s + a[i, j];\n        }\n    }\n    return s;\n}\nfn main() -> i32 {\n    let a: [f32; 12] = [1.0; 12];\n    return total(a) as i32;\n}\n",
        good: "module ex.e0520\nfn total<M, N>(a: Tensor[f32, M, N]) -> f32 {\n    let mut s: f32 = 0.0;\n    for i in 0..M {\n        for j in 0..N {\n            s = s + a[i, j];\n        }\n    }\n    return s;\n}\nfn main() -> i32 {\n    let a: [f32; 12] = [1.0; 12];\n    return total::<3, 4>(a) as i32;\n}\n"
    ),
    // ---- Warnings (lints). Each has a name for `-A/-W/-D <name>` and `@allow/@warn/@deny(name)`;
    // `-D warnings` makes every warning an error. Lints run only on a program that checks clean.
    entry!(
        "W0001",
        "unused variable (`unused_variables`)",
        "A `let` binding, a parameter, a `for` variable or a `match` binding is never read. A \
         variable that is only assigned to (`acc += x` with `acc` never read afterwards) counts as \
         unused too: the values computed into it go nowhere, which in a kernel usually means the \
         result is written to the wrong place. Use the value, delete the binding, or prefix the \
         name with an underscore (`_x`) to say the binding is intentionally unused."
    ),
    entry!(
        "W0002",
        "variable does not need to be mutable (`unused_mut`)",
        "A scalar or pointer binding is declared `mut` but is never reassigned, compound-assigned or \
         borrowed. Writing through a pointer (`*p = v`) changes the pointee, not the pointer, so it \
         does not need `mut` on the pointer binding. Remove the `mut`. (An aggregate `let` is not \
         judged: element stores need no `mut` there.)"
    ),
    entry!(
        "W0003",
        "unreachable code (`unreachable_code`)",
        "A statement or expression follows a `return`, `break` or `continue`, or a construct that \
         diverges on every path — an `if`/`else` whose both branches diverge, a `match` with a \
         catch-all arm whose every arm diverges, or a `loop` with no `break` — so it never runs. \
         Divergence is judged exactly as sema's definite-return check judges it, so deleting the \
         reported code never makes a `return` necessary. Delete the code, or move it before the \
         diverging statement if it was meant to run."
    ),
    entry!(
        "W0004",
        "unused result (`unused_result`)",
        "An expression statement computes a value with no effect and throws it away: a comparison, \
         an arithmetic or logical operation, a cast, a literal or a plain name. The classic case is \
         `out[i] == a[i] + b[i];` — a comparison written where the assignment `out[i] = a[i] + b[i];` \
         was meant, which leaves `out` unwritten and compiles silently otherwise. A call, an index \
         (its bounds check is observable), a dereference and any block, `if`, `match` or `loop` are \
         never reported."
    ),
    entry!(
        "W0005",
        "function is never used (`dead_code`)",
        "A function of the compiled file cannot be reached from any entry point. The entry points \
         are `main`, the `--grad-of` function under `--emit=grad`/`--train`, every `pub` function, \
         every `@export`/`@extern` function, and any function a `const` initializer names; a \
         function only called by itself is still unreachable. A file with no `main`, and \
         `--emit=obj`/`--emit=llvm-ir` (whose artifact a host may call into anywhere), report \
         nothing. Functions of imported files are not reported. Call the function, make it `pub`, \
         or delete it."
    ),
    entry!(
        "W0006",
        "output parameter is never written (`unwritten_output`)",
        "A `mut` aggregate parameter — an array, slice, tensor, tuple or struct, passed by reference \
         so that the caller sees the stores — is read but never stored to, assigned, passed on or \
         borrowed. `mut` declares the parameter an output buffer, so a kernel that never writes it \
         computes nothing the caller can see: usually a typo (`out[i] == …`, or the store went to \
         another buffer). Store the result through it, or drop `mut` if the function only reads it."
    ),
    entry!(
        "W0007",
        "unknown lint (`unknown_lints`)",
        "An `@allow(..)`, `@warn(..)` or `@deny(..)` attribute names something that is not a lint, \
         or has no argument at all, so it silences or promotes nothing. The arguments are lint names \
         (`unused_variables`), lint codes (`W0001`) or `warnings` (every lint). `wukongc --explain \
         W0001` and its neighbours describe each lint."
    ),
    entry!(
        "W0030",
        "missing `;` (`missing_semicolon`)",
        "A statement is not terminated by `;`. A statement ends with `;`; it may be left out only \
         after a block-like statement (`if`, `match`, `loop`, `while`, `for`, a `{ … }` block) and, \
         for any statement but a `let`, right before the `}` that closes the block — Rust's rule. \
         The parse is the same either way, but an optional terminator is what lets a `(`, `[` or \
         `-` on the next line silently continue the statement (see W0031), so write the `;`. Not \
         reported for a statement that already has a parse error, or that a rejected character cut \
         short. Reserved: E0210, should a missing `;` ever become an error."
    ),
    entry!(
        "W0031",
        "line-leading continuation (`ambiguous_continuation`)",
        "A `(` or `[` at the start of a line, or a `-` that starts a line and touches its operand \
         (`-1`), continues the expression on the line above: `acc = a[i]` followed by `(b + c)` on \
         the next line is the call `a[i](b + c)`, and `x` followed by `-1` is `x - 1` — although each \
         pair reads as two statements. Reported only where a new statement could begin (inside \
         `( )` or `[ ]` nothing is ambiguous), and not for `- y` with a space, the ordinary \
         continued-subtraction style. If a new statement starts there, end the previous one with \
         `;`; if the continuation is meant, move the token to the end of the line above."
    ),
    entry!(
        "C0001",
        "unsupported in codegen",
        "A construct parsed and type-checked but is not yet supported by MIR lowering or the \
         selected backend. The interpreter and LLVM backends grow coverage over time; check the \
         language guide for the currently supported subset."
    ),
    entry!(
        "C0002",
        "type too large for the native layout",
        "The native (Cranelift) backend sizes stack slots and scales element offsets with 32-bit \
         byte counts, so one array, vector or aggregate whose byte size exceeds 2^32 (4 GiB) cannot \
         be laid out by `--backend=native`, `--emit=obj` or `--emit=exe`. The diagnostic names the \
         type and the function that declares it. The interpreter has no byte layout and runs the \
         program. Allocate the buffer on the heap (`alloc_f32(n)` and its siblings) or split it \
         into arrays below 4 GiB."
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explain_is_case_insensitive() {
        assert!(explain("e0502").is_some());
        assert_eq!(explain("E0502").unwrap().code, "E0502");
        assert!(explain("E9999").is_none());
    }

    #[test]
    fn every_entry_has_a_code_and_body() {
        for e in all() {
            assert!(
                e.code.starts_with('E') || e.code.starts_with('C') || e.code.starts_with('W'),
                "{} is in no code range",
                e.code
            );
            assert!(
                e.code.len() == 5 && e.code[1..].bytes().all(|b| b.is_ascii_digit()),
                "{} is not a letter and four digits",
                e.code
            );
            assert!(!e.title.is_empty());
            assert!(e.body.len() > 20, "{} has a thin explanation", e.code);
        }
    }

    #[test]
    fn examples_come_in_pairs_and_some_exist() {
        let mut with_examples = 0;
        for e in all() {
            assert_eq!(
                e.bad.is_some(),
                e.good.is_some(),
                "{} has only one of bad/good; an example needs both",
                e.code
            );
            if e.bad.is_some() {
                with_examples += 1;
            }
        }
        assert!(
            with_examples >= 3,
            "expected at least a few worked examples; found {with_examples}"
        );
    }
}
