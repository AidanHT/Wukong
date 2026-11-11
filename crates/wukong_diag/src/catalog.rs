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
//!   * `W00xx` — warnings (lints; `W0001`–`W0020` from `wukong_sema::lints` — `W0008` found by the
//!     driver's import loader — and `W003x` from the parser), each also named for `-A/-W/-D` and
//!     `@allow/@warn/@deny`
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
    /// A complete, erroneous `.wk` program whose *first* diagnostic is exactly `code` — an error
    /// that fails the build, or for a `W` code a warning on a build that succeeds. Compiled by
    /// `crates/wukongc/tests/explain.rs` (via `--emit=check`), so a catalogue example can never
    /// silently drift away from the rule it documents. A line `// file: <path>` starts another
    /// file of the program at that path relative to the first one (an import example needs two).
    /// `None` where no example has been written yet.
    pub bad: Option<&'static str>,
    /// The corrected counterpart of [`bad`](Self::bad): the same program, fixed, which must compile
    /// under `--emit=check` — for a `W` code with no diagnostic at all.
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
        "E0210",
        "doc comment documents nothing",
        "A doc comment is attached to what follows it, and this one has nothing to attach to. A \
         `///` comment documents the item, struct field, enum variant, trait or `impl` method, or \
         `extern` declaration written after it (attributes and `pub` may come between); a `//!` \
         comment documents the file and must come before its first item (after the `module` \
         header, if there is one). Here it is followed by a `}` or the end of the file, or sits \
         before a statement or inside an expression, or is a `//!` after an item. Doc comments \
         are what `--emit=doc` publishes and `--doctest` runs, so one that documents nothing is \
         an error rather than silently lost (rustc's E0585). Write `//` for a plain comment, or \
         move the comment onto the item it describes. One error is reported per run of \
         consecutive doc lines, and none when the file already has a syntax error."
    ),
    entry!(
        "E0300",
        "duplicate definition",
        "A name was defined more than once in the same scope. Rename one of the definitions. The \
         same code rejects a declaration that takes a built-in type's name: a struct, enum, trait \
         or generic parameter named `str`, or a type alias named `str` or after a scalar (`f32`)."
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
         compound type. A SIMD vector's lanes are at most 64 bits wide: no vector instruction set \
         has `i128`/`u128` lanes. An 8-bit float (`f8e4m3`, `f8e5m2`) is a scalar type with no \
         vector form either: use `f32` lanes and convert with `as`."
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
         the pointer itself is not being reassigned. The bytes of a `str` are never a place at \
         all: `s[i] = …`, `s[i] += …`, `atomic_add(s[i], …)` and `&mut s[i]` are rejected, \
         because a `str` views a string literal's read-only storage (a `mut` binding may still be \
         re-pointed, `s = t`).",
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
         `@align(N)` aligns one `let`'s stack slot, so it is legal only on a `let` that binds one \
         name of an array, tuple or struct type (a scalar has no slot; a slice, pointer or tensor \
         local names storage it does not own) and rejected on a `fn`, a parameter, a loop or any \
         other statement. Arguments: `@parallel` takes nothing or exactly `grain = G` with \
         `1 <= G <= 2^31` (each worker then claims runs of at least G iterations); an unknown key, \
         a positional argument or an out-of-range grain is rejected instead of discarded. `@align` \
         takes exactly one power of two from 1 to 4096 (`@align(48)` is rejected). The lint-level \
         attributes `@allow`, `@warn` and `@deny` are known too and legal anywhere; their arguments \
         are lint names, and an unknown one is the warning `W0007`, not this error. The \
         parsed-but-discarded attributes (`@inline`, `@simd`, `@export`) stay legal wherever they \
         parse."
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
        "A `@tile` attribute names a tiling the compiler cannot perform. Five conditions reach \
         this code. (1) The loop form: `@tile` strip-mines \
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
         keep the loops tileable. The first four used to be silent — the width was discarded where \
         it stood, and a function-item queue additionally SHIFTED, so the next loop was \
         strip-mined at the wrong width. Either drop the attribute or write the loop in the form it \
         can rewrite. (5) The loop is a `@parallel` `f32` reduction: its value is the parallel \
         reduction tree, and a strip-mined body folds one statement at a time, in source order, so \
         the width used to be applied and change the RESULT rather than the schedule. Drop the \
         `@tile` (the tree already works in fixed 8,192-element chunks), or mark the function \
         `@strict_fp`, whose reductions fold in source order and can be strip-mined."
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
        "E0325",
        "invalid element type for `alloc::<T>`",
        "The generic heap allocator `alloc::<T>(n)` returns a zeroed `[]T` of `n` elements, and \
         takes exactly one element type: a scalar (`alloc::<f32>(n)`, the same buffer as \
         `alloc_f32(n)`) or a type parameter of the enclosing function (`alloc::<T>(n)` inside \
         `fn f<T, N>(x: Tensor[T, N])`, which each instantiation makes concrete). A missing \
         element type, more than one, or a struct, tuple or array element is this error: a heap \
         buffer holds scalars. Allocate one buffer per scalar field, or an array of the scalar \
         and index it with a stride.",
        bad: "module ex.e0325\nstruct Point { x: f32, y: f32 }\nfn main() -> i32 {\n    let b = alloc::<Point>(4);\n    free(b);\n    return 0;\n}\n",
        good: "module ex.e0325\nfn main() -> i32 {\n    let b = alloc::<i32>(4);\n    let n = b.len();\n    free(b);\n    return n as i32;\n}\n"
    ),
    entry!(
        "E0326",
        "type alias refers to itself",
        "A `type` alias is replaced by its definition wherever it is used, so an alias that names          itself, directly (`type L = *L;`) or through other aliases (`type A = [B; 2]; type B =          A;`), never finishes expanding. The diagnostic lists the cycle. A recursive data          structure needs a nominal type: declare a struct or enum and refer to it through a          pointer or slice.",
        bad: "module ex.e0326
type A = [B; 2];
type B = A;
fn main() -> i32 {
    return 0;
}
",
        good: "module ex.e0326
type A = [B; 2];
type B = f32;
fn main() -> i32 {
    let a: A = [1.0, 2.0];
    return a[1] as i32;
}
"
    ),
    entry!(
        "E0327",
        "wrong number or kind of type alias arguments",
        "A generic alias such as `type Mat<M, N> = Tensor[f32, M, N];` must be given exactly one          argument per parameter (`Mat<4, 8>`), and a type that is not a generic alias takes none          (`f32<2>` or `Point<f32>` is this error). Each argument must also fit where the alias          uses its parameter: a parameter used as a dimension or array length needs an integer, a          name or dimension arithmetic, not a type like `f32`, and a parameter used as a type          cannot be given `?` or `N*2`.",
        bad: "module ex.e0327
type Mat<M, N> = Tensor[f32, M, N];
fn first(a: Mat<4>) -> f32 {
    return a[0, 0];
}
fn main() -> i32 {
    return 0;
}
",
        good: "module ex.e0327
type Mat<M, N> = Tensor[f32, M, N];
fn first(a: Mat<4, 4>) -> f32 {
    return a[0, 0];
}
fn main() -> i32 {
    return 0;
}
"
    ),
    entry!(
        "E0328",
        "type alias names an unknown type",
        "The definition of a `type` alias names something that is not a type: not a scalar          (`f32`, `i64`, `bool`, ...), not a struct or enum of the program, not another alias, and          not one of the alias's own parameters. It is reported at the definition even when the          alias is never used, since aliases are expanded before type checking. Check the spelling,          declare the missing parameter (`type Pair<T> = (T, T);`), or import the module that          defines the type.",
        bad: "module ex.e0328
type Pair = (T, T);
fn main() -> i32 {
    return 0;
}
",
        good: "module ex.e0328
type Pair<T> = (T, T);
fn main() -> i32 {
    let p: Pair<i32> = (1, 2);
    return p.0 + p.1;
}
"
    ),
    entry!(
        "E0330",
        "import path differs from the file only in letter case",
        "An `import` names a file that exists, but only under a different capitalization: \
         `import Util` when the file is `util.wk`, or `import Lib.ops` for a directory `lib`. \
         Imports resolve case-sensitively on every host. Opening the path would succeed on a \
         case-insensitive file system (Windows, and macOS by default) and fail with E0305 on a \
         case-sensitive one (Linux), so the same program used to compile on one machine and not on \
         another. Each path component is compared byte for byte with the directory listing; the \
         diagnostic names the on-disk spelling. Spell the import exactly as the file and its \
         directories are named."
    ),
    entry!(
        "E0331",
        "invalid `main` signature",
        "The program entry `fn main` is called by the runtime with no arguments, and its result \
         becomes the process exit code, so it must take no parameters, have no generic parameters, \
         and return a scalar (an integer, a float, `bool` or `char`) or nothing. The exit code is \
         the low 8 bits of the value: a float is converted toward zero first, and a `main` with no \
         return type exits 0. Read command-line arguments with `arg_count()` and `arg(i)`, and \
         print or write an aggregate result rather than returning it.",
        bad: "module ex.e0331\nfn main(n: i32) -> i32 {\n    return n;\n}\n",
        good: "module ex.e0331\nfn main() -> i32 {\n    let n: i32 = 0;\n    return n;\n}\n"
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
        "A `match` does not cover every possible value of the scrutinee. The diagnostic names \
         values no arm covers — a *witness* pattern such as `Expr::Add(i32::MIN..=-1 | \
         1..=i32::MAX, _)` or `(false, false)` — found by pattern usefulness, the analysis Rust \
         uses: it sees through enum payload sub-patterns, tuples and integer and `char` ranges, so \
         `Expr::Add(0, y)` covers only the `Add` values whose first field is 0, `(true, _)` and \
         `(false, true)` leave `(false, false)` uncovered, and `0..=99 | 100..=255` on a `u8` is \
         complete. This is rejected in every position, not just value position: a value match would \
         produce an injected zero default when no arm matches — a silent wrong answer (or, for a \
         tuple/struct result, invalid code) — while a statement or empty match falls through to \
         `Unreachable`, which the interpreter traps but the native backend hits as an illegal \
         instruction. Add an arm for the missing values or a `_ => …` arm, as in Rust. Guarded arms \
         (`if …`) do not count toward coverage because their guard may be false. A `char` can hold \
         any 32-bit value (an integer `as char` is unchecked), so a `char` match needs a `_` arm.",
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
        "function type or closure where it is not allowed",
        "A function type `fn(T) -> R` is an ordinary value type: a local (`let mut f`), a field, an \
         element, a return value and a parameter can each hold a function value, made from a named \
         non-generic function or a closure that captures nothing, and calling one calls the \
         function it holds. A few places cannot have one. An `extern` declaration, or a function \
         declared without a body, cannot spell a function type: the C side has no Wukong function \
         values. A function-typed parameter cannot be `mut` (it names the function each call \
         binds; copy it to a `let mut` local to reassign it). A function-typed parameter or local \
         cannot be named after a builtin (`gelu`, `print`, ...): kernel recognizers and the \
         vectorizer read a call `gelu(x)` as the builtin unless a top-level function shadows it. \
         And a closure literal, like a function used as a value, is an expression of a function \
         body: a `const` initializer, a `where` predicate or a `static_assert` cannot hold one (a \
         `const` is evaluated again at each use, so it cannot hold a function value; bind the \
         function with `let` in the function that uses it)."
    ),
    entry!(
        "E0415",
        "kernel builtin requires `f32` elements",
        "The named kernel builtins (`matmul`, `linear`, `softmax`, `layernorm`, the two-argument \
         activations, `sum`/`dot`, `sdpa`, …) call `f32` runtime kernels, so every buffer operand \
         must hold `f32`. A dtype-generic `Tensor[T, ..]` operand is accepted in the generic body \
         and checked per instantiation: an instance that binds `T = f32` dispatches the kernel \
         exactly like the hand-written `f32` function, and an instance that binds any other \
         element type (`f64`, `bf16`, an integer) is this error, reported at the operand with the \
         call that required the instantiation. Instantiate the function at `f32` (convert the \
         data first), or write the operation as a loop over `T`, which works for every element \
         type.",
        bad: "module ex.e0415\nfn mm<T>(a: Tensor[T, 2, 2], b: Tensor[T, 2, 2], mut c: Tensor[T, 2, 2]) {\n    matmul(a, b, c);\n}\nfn main() -> i32 {\n    let a: [f64; 4] = [1.0; 4];\n    let b: [f64; 4] = [1.0; 4];\n    let mut c: [f64; 4] = [0.0; 4];\n    mm(a, b, c);\n    return c[0] as i32;\n}\n",
        good: "module ex.e0415\nfn mm<T>(a: Tensor[T, 2, 2], b: Tensor[T, 2, 2], mut c: Tensor[T, 2, 2]) {\n    matmul(a, b, c);\n}\nfn main() -> i32 {\n    let a: [f32; 4] = [1.0; 4];\n    let b: [f32; 4] = [1.0; 4];\n    let mut c: [f32; 4] = [0.0; 4];\n    mm(a, b, c);\n    return c[0] as i32;\n}\n"
    ),
    entry!(
        "E0416",
        "typed I/O builtin requires the element type it names",
        "A typed builtin of the ML I/O family moves exactly the element type it is named for — \
         `read_bf16` and `write_bf16` move `bf16`, `st_read_f32` and `topk_f32` move `f32` — \
         because the bytes on disk (or the kernel's stride) are that type's. A dtype-generic \
         `Tensor[T, ..]` operand is accepted in the generic body and checked per instantiation: an \
         instance that binds `T` to the builtin's element reaches it exactly like the concrete \
         function, and an instance that binds any other element type is this error, reported at \
         the operand with the call that required the instantiation. Call the builtin spelled for \
         the instantiated element (`write_f64` for `T = f64`), or instantiate the function at the \
         builtin's element.",
        bad: "module ex.e0416\nfn save<T>(x: Tensor[T, 4]) -> i64 {\n    return write_f32(\"e0416.bin\", x);\n}\nfn main() -> i32 {\n    let x: [f64; 4] = [1.0; 4];\n    return save(x) as i32;\n}\n",
        good: "module ex.e0416\nfn save<T>(x: Tensor[T, 4]) -> i64 {\n    return write_f32(\"e0416.bin\", x);\n}\nfn main() -> i32 {\n    let x: [f32; 4] = [1.0; 4];\n    return save(x) as i32;\n}\n"
    ),
    entry!(
        "E0418",
        "closure captures a value it cannot copy",
        "A closure captures every local of the enclosing code that its body reads. One it only \
         reads is copied where the closure is created (when the call evaluates a closure \
         argument, or at the `let` that binds it), and the closure is lifted to a function that \
         receives each copy as a parameter; one it writes is shared by reference. What can be \
         captured is a number, `bool` or `char`, an enum, a raw pointer or reference, a \
         fixed-size array, a tuple, a struct, a slice or a `str` (whose copy is its fat pointer: \
         another view of the same elements), or a function value. A tensor or a SIMD vector \
         cannot, and neither can the enclosing function's function-typed parameter (it has no \
         run-time value) or anything that names one of the enclosing function's generic \
         parameters (the lifted function is not generic). To read a tensor, capture a pointer \
         to it (`let pt = &t;` before the closure, then `(*pt)[i]` in it) or pass it to the \
         callee as an ordinary parameter; to use a generic dimension, bind it to a local first \
         (`let n = N;`).",
        bad: "module ex.e0418\nfn apply(f: fn(f32) -> f32, x: f32) -> f32 {\n    return f(x);\n}\nfn second(t: Tensor[f32, 2], x: f32) -> f32 {\n    return apply(|y| y * t[1], x);\n}\nfn main() -> i32 {\n    let a: [f32; 2] = [1.0, 2.0];\n    return second(a, 2.0) as i32;\n}\n",
        good: "module ex.e0418\nfn apply(f: fn(f32) -> f32, x: f32) -> f32 {\n    return f(x);\n}\nfn second(t: Tensor[f32, 2], x: f32) -> f32 {\n    let w = t[1];\n    return apply(|y| y * w, x);\n}\nfn main() -> i32 {\n    let a: [f32; 2] = [1.0, 2.0];\n    return second(a, 2.0) as i32;\n}\n"
    ),
    entry!(
        "E0419",
        "closure parameter type cannot be inferred",
        "A closure parameter without an annotation takes its type from the function type of the \
         parameter the closure is passed to. When that type names a generic parameter of the \
         callee, as the type itself (`fn fold3<T>(f: fn(T, T) -> T, ..)`) or as a tensor's \
         element type or dimension (`fn(Tensor[T, N]) -> f32`), the call's other arguments bind \
         it first (`fold3(|a, b| a + b, x, y, z)` types `a` and `b` as `x`'s type). When none of \
         them does, or the closure is bound by `let` and has no function type to take it from, \
         the parameter needs an annotation: `|a: i32, b: i32| a + b`. The same holds for the \
         return type of a closure whose body uses `return` while no return type is known: \
         annotate it (`|..| -> i32 { .. }`).",
        bad: "module ex.e0419\nfn twice<T>(_f: fn(T) -> T, n: i32) -> i32 {\n    return n * 2;\n}\nfn main() -> i32 {\n    return twice(|x| x, 1);\n}\n",
        good: "module ex.e0419\nfn twice<T>(_f: fn(T) -> T, n: i32) -> i32 {\n    return n * 2;\n}\nfn main() -> i32 {\n    return twice(|x: i32| x, 1);\n}\n"
    ),
    entry!(
        "E0420",
        "`str` cannot instantiate a type parameter",
        "A call binds a generic function's type parameter to `str`, or to a type containing one \
         (`(str, i32)`, `[str; 2]`, `*str`). Sema checks every `str` rule — immutability, no \
         conversion to or from `[]u8` and `*u8`, printing as text — and then erases each `str` to \
         the `[]u8` slice it is laid out as, before generic functions are instantiated; an \
         instance at `str` could not tell its text from a byte slice. Write the function for \
         `str` itself: a non-generic `fn f(s: str)`.",
        bad: "module ex.e0420\nfn first<T>(x: T) -> T {\n    return x;\n}\nfn main() -> i32 {\n    let s: str = \"hi\";\n    let t = first(s);\n    return t.len() as i32;\n}\n",
        good: "module ex.e0420\nfn first(x: str) -> str {\n    return x;\n}\nfn main() -> i32 {\n    let s: str = \"hi\";\n    let t = first(s);\n    return t.len() as i32;\n}\n"
    ),
    entry!(
        "E0421",
        "a closure-bound local that captures, used as a value",
        "A local bound to a closure that captures (`let f = |x: f32| x * k;`) is resolved at \
         compile time: each call `f(..)` is a direct call of the closure's code with the values it \
         captured, and passing `f` to a function-typed parameter specializes the callee for it. A \
         run-time function value is only a code address, with no room for the captured values, so \
         the name can only be called or passed to a parameter of function type. Copying it to \
         another local, returning it, storing it in a field, element or tuple, taking its address, \
         printing it or computing with it is this error. (A closure that captures nothing is an \
         ordinary function value and can be used anywhere one can.) Call it, or pass it on.",
        bad: "module ex.e0421\nfn main() -> i32 {\n    let k: i32 = 1;\n    let f = |x: i32| x + k;\n    let g = f;\n    return g(1);\n}\n",
        good: "module ex.e0421\nfn apply(f: fn(i32) -> i32, x: i32) -> i32 {\n    return f(x);\n}\nfn main() -> i32 {\n    let k: i32 = 1;\n    let f = |x: i32| x + k;\n    return apply(f, 1);\n}\n"
    ),
    entry!(
        "E0422",
        "a closure bound by a `mut` binding or a pattern, or reassigned",
        "A closure-bound local names the one closure its `let` initializes it with, because every \
         call through it is resolved at compile time. So the binding must be a plain, immutable \
         `let name = |..| ..;`: a destructuring pattern, an assignment `f = ..` to a \
         closure-bound `f`, and a `let mut` of a closure that captures are rejected. (A `let mut` \
         of a closure that captures nothing is an ordinary function variable, which can be \
         reassigned.) Bind each closure to a name of its own.",
        bad: "module ex.e0422\nfn main() -> i32 {\n    let k: i32 = 1;\n    let mut f = |x: i32| x + k;\n    return f(1);\n}\n",
        good: "module ex.e0422\nfn main() -> i32 {\n    let k: i32 = 1;\n    let f = |x: i32| x + k;\n    return f(1);\n}\n"
    ),
    entry!(
        "E0423",
        "or-pattern alternatives bind different names",
        "An or-pattern `A | B` matches when any alternative does, and its arm then reads the names \
         the pattern binds from whichever alternative matched. So every alternative must bind \
         exactly the same names: `Shape::Circle(r) | Shape::Square(r)` is fine, but \
         `Shape::Circle(r) | Shape::Empty` leaves `r` unbound when the value is `Shape::Empty`. \
         Bind the name in every alternative, or split the alternatives into arms of their own.",
        bad: "module ex.e0423\nenum Shape { Circle(i32), Square(i32), Empty }\nfn size(s: Shape) -> i32 {\n    return match s {\n        Shape::Circle(r) | Shape::Empty => r,\n        Shape::Square(w) => w,\n    };\n}\nfn main() -> i32 {\n    return size(Shape::Square(3));\n}\n",
        good: "module ex.e0423\nenum Shape { Circle(i32), Square(i32), Empty }\nfn size(s: Shape) -> i32 {\n    return match s {\n        Shape::Circle(r) | Shape::Square(r) => r,\n        Shape::Empty => 0,\n    };\n}\nfn main() -> i32 {\n    return size(Shape::Square(3));\n}\n"
    ),
    entry!(
        "E0424",
        "or-pattern binds a name at different types",
        "Every alternative of an or-pattern binds the same names, and each name must have one \
         type across them: the arm reads a single binding, whichever alternative matched. \
         `Num::Int(v) | Num::Real(v)` with `Int(i32)` and `Real(f32)` binds `v` as an `i32` in \
         one alternative and an `f32` in the other. Match the alternatives in separate arms and \
         convert there (`Num::Int(v) => v as f32`).",
        bad: "module ex.e0424\nenum Num { Int(i32), Real(f32) }\nfn val(n: Num) -> f32 {\n    return match n {\n        Num::Int(v) | Num::Real(v) => v,\n    };\n}\nfn main() -> i32 {\n    return val(Num::Int(2)) as i32;\n}\n",
        good: "module ex.e0424\nenum Num { Int(i32), Real(f32) }\nfn val(n: Num) -> f32 {\n    return match n {\n        Num::Int(v) => v as f32,\n        Num::Real(v) => v,\n    };\n}\nfn main() -> i32 {\n    return val(Num::Int(2)) as i32;\n}\n"
    ),
    entry!(
        "E0425",
        "`.chars()` outside a `for` head",
        "`s.chars()` decodes a `str` one character at a time, and it is legal only as the \
         iterable of a `for` loop: `for c in s.chars() { .. }`, where each `c` is a `char`. The \
         language has no iterator values, so the loop decodes the text in place and there is \
         nothing for `s.chars()` to be anywhere else — bound to a name, passed, or called a method \
         on. Iterate it directly; `s.char_count()` counts the characters, `s.len()` the bytes, and \
         `s[i]` reads a byte.",
        bad: "module ex.e0425\nfn main() -> i32 {\n    let s: str = \"hi\";\n    let it = s.chars();\n    return 0;\n}\n",
        good: "module ex.e0425\nfn main() -> i32 {\n    let s: str = \"hi\";\n    let mut n: i32 = 0;\n    for c in s.chars() {\n        n = n + (c as i32);\n    }\n    return n;\n}\n"
    ),
    entry!(
        "E0426",
        "string-literal pattern against a value that is not a `str`",
        "A string-literal pattern (`\"add\" =>`) matches a `str` whose bytes equal the literal's, \
         and a `str` is matched by string literals (and by names, `_` and or-patterns of them). A \
         string literal against any other type, or an integer, `char`, `bool` or range pattern \
         against a `str`, could never match. A `*u8` — an unannotated literal, `arg(i)`, \
         `env(name)` — is a C string whose literal patterns would compare addresses, not text: \
         bind the text as a `str` first (`let s: str = ..`, or `str_from_utf8` of its bytes).",
        bad: "module ex.e0426\nfn main() -> i32 {\n    let n: i32 = 3;\n    return match n {\n        \"three\" => 1,\n        _ => 0,\n    };\n}\n",
        good: "module ex.e0426\nfn main() -> i32 {\n    let s: str = \"three\";\n    return match s {\n        \"three\" => 1,\n        _ => 0,\n    };\n}\n"
    ),
    entry!(
        "E0427",
        "a closure that captures used as a run-time function value",
        "A run-time function value — a `fn(T) -> R` stored in a local, a field or an element, \
         returned, or kept by the function it is passed to — is the address of code and nothing \
         else. A named function or a closure that captures nothing can be one. A closure that \
         reads or writes a local of the enclosing code needs that local (or a pointer to it) every \
         time it runs, and the value has no room to carry it: there is no environment to allocate, \
         keep alive or free. Such a closure can still be called through the `let` that binds it, or \
         passed directly to a function-typed parameter the callee only calls or passes on (the \
         callee is then specialized for it and receives the captured values). A call through a \
         function value specializes nothing, so a closure given to one — through a `let` of \
         function type, or through a function-typed parameter in the instance that receives it at \
         run time — is a value too. To store it, take the captured values as ordinary \
         parameters, or capture nothing.",
        bad: "module ex.e0427\nfn main() -> i32 {\n    let k: i32 = 3;\n    let mut f: fn(i32) -> i32 = |x| x;\n    f = |x| x + k;\n    return f(1);\n}\n",
        good: "module ex.e0427\nfn main() -> i32 {\n    let mut f: fn(i32) -> i32 = |x| x;\n    f = |x| x + 3;\n    return f(1);\n}\n"
    ),
    entry!(
        "E0428",
        "function values cannot be compared",
        "A function value is the address of compiled code, and which functions share an address \
         is the compiler's choice, not the program's: the optimizer may merge, clone or inline a \
         function, and the interpreter numbers functions where native code has addresses. So \
         `f == g` (or `!=`, `<`, ...) on two function values would not mean the same thing on \
         every backend and at every `-O` level, and is rejected. Keep a tag beside the function \
         (an enum or an integer) and compare that.",
        bad: "module ex.e0428\nfn one() -> i32 {\n    return 1;\n}\nfn main() -> i32 {\n    let f: fn() -> i32 = one;\n    let g: fn() -> i32 = one;\n    if f == g {\n        return 1;\n    }\n    return 0;\n}\n",
        good: "module ex.e0428\nfn one() -> i32 {\n    return 1;\n}\nfn main() -> i32 {\n    let f: fn() -> i32 = one;\n    let g: fn() -> i32 = one;\n    if f() == g() {\n        return 1;\n    }\n    return 0;\n}\n"
    ),
    entry!(
        "E0429",
        "this function cannot be a run-time function value",
        "A run-time function value is the address of one compiled function. A generic function \
         (with type or dimension parameters) has no single address: it is compiled once per \
         instantiation, chosen at each call from the arguments. An `extern` function, or one \
         declared without a body, belongs to C, which the interpreter cannot call through a \
         pointer. A function with a tensor parameter of run-time extent (`Tensor[f32, ?]`) \
         receives each such extent as a hidden argument that a direct call reads off the tensor \
         it passes, which a call through a pointer cannot pass. None of these can be stored, \
         returned or passed as a value (a generic function is also refused as the argument of a \
         function-typed parameter). Wrap it in a function of a concrete, fixed-extent signature \
         (`fn id_i32(x: i32) -> i32 { return id(x); }`) and use that.",
        bad: "module ex.e0429\nfn id<T>(x: T) -> T {\n    return x;\n}\nfn main() -> i32 {\n    let f: fn(i32) -> i32 = id;\n    return f(1);\n}\n",
        good: "module ex.e0429\nfn id<T>(x: T) -> T {\n    return x;\n}\nfn id_i32(x: i32) -> i32 {\n    return id(x);\n}\nfn main() -> i32 {\n    let f: fn(i32) -> i32 = id_i32;\n    return f(1);\n}\n"
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
         `1.5e`) or names a type Wukong does not have (a `u128`/`i128` suffix: the widest \
         integers are `i64` and `u64`), or an integer literal whose value does not fit the type it \
         is used at — the type an unsuffixed literal adapts to from an annotation, a parameter, \
         a field, an assignment, a return type or a `match` scrutinee (`let x: i8 = 200`) — is \
         rejected instead of wrapping silently. A constant expression of literals is checked by \
         its folded value (`let v: i8 = 100 + 100`). An enum discriminant must be such a constant \
         and fit the 32-bit enum representation. Use a wider type, or convert with `as`, which \
         wraps by definition.",
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
         fields; two `str`s are the one exception, compared by value, and a `str` compares with \
         nothing else); a comparison of `bool` with a number or of signed with unsigned integers (a \
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
         builtin a path or tensor name that is neither a `*u8` string nor a NUL-terminated `[]u8` \
         / `[u8; N]` buffer (what `sprintf` writes), an offset that is not an integer, or a buffer \
         of the wrong element type or with a `?` dimension; an RNG builtin a float seed; a kernel \
         builtin (`matmul`, `sdpa`, …) an operand that is not an `f32` buffer or a non-integer \
         dimension; `atomic_add` a first argument that is not a place in memory other threads can \
         share (an element `b[i]`, a field `s.f` or `t.0`, or a pointee `*p` -- a bare local name \
         is private to its thread, so use `+=`); a range index `x[a..b]` a base that is neither a \
         `str` nor a `[]T` slice (bind an array to a `[]T` first). The message names the argument \
         and what it must be.",
        bad: "module ex.e0437\nfn main() -> i32 {\n    let s = alloc_f32(2.5);\n    free(s);\n    return 0;\n}\n",
        good: "module ex.e0437\nfn main() -> i32 {\n    let s = alloc_f32(4);\n    free(s);\n    return 0;\n}\n"
    ),
    entry!(
        "E0438",
        "8-bit float literal out of range",
        "A literal of an 8-bit float type — suffixed (`1.5f8e4m3`, `0.25f8e5m2`), or unsuffixed \
         where it takes an `f8e4m3`/`f8e5m2` type from an annotation, a parameter, a field, a \
         return type or a compound-assignment place, and an unsuffixed integer literal that \
         converts to one beside an 8-bit float value (`x * 1000`, `x < 1000`, `e += 1000`, \
         `max(x, 1000)`, an `if` arm; an all-integer expression by its value) — is worth its \
         rounding onto the format's grid \
         (nearest, ties to even). Two roundings would silently turn it into a different number, \
         so both are rejected: a magnitude that rounds past the largest finite value (448 for \
         `f8e4m3`, which has no infinity and saturates there; 57344 for `f8e5m2`, which would \
         become infinity), and a nonzero literal that rounds to zero (below half the least value, \
         2^-9 for `f8e4m3` and 2^-16 for `f8e5m2`). Losing digits is the format's precision and \
         fine (`0.1f8e4m3` is 0.1015625). Write a value in range, `0.0` for zero, or keep the \
         value in a wider float and convert it with `as`, which saturates or overflows by \
         definition.",
        bad: "module ex.e0438\nfn main() -> i32 {\n    let w: f8e4m3 = 500.0;\n    return w as i32;\n}\n",
        good: "module ex.e0438\nfn main() -> i32 {\n    let w: f8e4m3 = 448.0;\n    return w as i32;\n}\n"
    ),
    entry!(
        "E0439",
        "4-bit value out of range",
        "`pack_i4(lo, hi)` and `pack_u4(lo, hi)` put two 4-bit integers in one byte, so a literal \
         operand (suffixed like `16u8` or not, or an expression of literals like `8 + 8`) must be \
         a value a nibble holds: `-8..=7` for `pack_i4` (two's complement) and \
         `0..=15` for `pack_u4`. A run-time operand keeps its low four bits, exactly as an `as` \
         conversion keeps the low bits, but a literal outside the range can only be a mistake — \
         `pack_u4(16, 0)` would pack `0`, and `pack_i4(8, 0)` the nibble `-8` — so it is rejected \
         instead of wrapping silently. Use a value in range, or the other signedness \
         (`pack_u4(8, 0)` holds 8).",
        bad: "module ex.e0439\nfn main() -> i32 {\n    let b = pack_u4(16, 3);\n    return b as i32;\n}\n",
        good: "module ex.e0439\nfn main() -> i32 {\n    let b = pack_u4(15, 3);\n    return b as i32;\n}\n"
    ),
    entry!(
        "E0440",
        "mixed-type arithmetic with no lossless common type",
        "An arithmetic or bitwise operator combines two values of different scalar types, and no \
         type holds every value of both: `i32` with `u32` (or any signed with an unsigned type at \
         least as wide), an integer with a float that cannot represent all of it exactly (`i32` \
         with `f32`, `i64` with `f64`), `f16` with `bf16`, `f8e4m3` with `f8e5m2` (or an 8-bit \
         float with any integer), or `char` with any other type. Such an operation used to pick a \
         type by width and then by operand ORDER, so `a / b` and `b / a` \
         could compute in different types and `u32 / i64` divided UNSIGNED (`7 / -2 == 0`). \
         Implicit promotion now only widens, and in either order: integers of one signedness to the \
         wider one, an unsigned integer to a strictly wider signed one (`u32 + i64` is `i64`), \
         `f8e4m3`/`f8e5m2` to `f16`/`bf16` to `f32` to `f64`, and an integer to a float that \
         holds it exactly (8- and 16-bit integers to `f32`, up to 32-bit integers to `f64`). A \
         signed value known never to \
         be negative — a `for` counter that starts at a non-negative literal and is never \
         assigned, or a non-negative `const` — also converts to an unsigned type at least as wide, \
         so `i * n` with `n: usize` stays legal. An unsuffixed literal adapts to the other operand \
         and never needs a cast; `<<`/`>>` take any integer amount. Convert an operand with `as` \
         to the type the result should have.",
        bad: "module ex.e0440\nfn ratio(a: u32, b: i32) -> i32 {\n    return a / b;\n}\nfn main() -> i32 {\n    return ratio(7, -2);\n}\n",
        good: "module ex.e0440\nfn ratio(a: u32, b: i32) -> i64 {\n    return (a as i64) / (b as i64);\n}\nfn main() -> i32 {\n    return ratio(7, -2) as i32;\n}\n"
    ),
    entry!(
        "E0441",
        "compound assignment would not fit back into its place",
        "A compound assignment `x op= v` means `x = x op v`, so the value it stores is the result of \
         `x op v`, and that must have `x`'s own type — the rule a plain assignment already follows. \
         So `v` may be of `x`'s type, of a type that promotes into it losslessly (`i64 += i32`, \
         `f64 += f32`, `i32 += u8`; see E0440), or an unsuffixed literal (which must fit an \
         integer place). An operand that is wider (`f32 += f64`, `i32 += i64`), of the other kind \
         (`i32 += f32`) or of an incomparable type (`i32 += u32`) is rejected: the compound form used \
         to convert the operand down to the place first, so `s += x` with `s: f32` and `x: f64` \
         silently accumulated in `f32`, and `acc += -1.0` on an integer `acc` changed nothing. Convert \
         the operand with `as`, or give the place the wider type. A `let` initialized from a bare \
         literal and not annotated — `let mut s = 0.0;` — takes its type from its first such use, \
         so it becomes an `f64` accumulator here with no annotation. A shift (`<<=`, `>>=`) takes an \
         amount of any integer type.",
        bad: "module ex.e0441\nfn total(x: f64) -> f32 {\n    let mut s: f32 = 0.5;\n    s += x;\n    return s;\n}\nfn main() -> i32 {\n    return total(1.5) as i32;\n}\n",
        good: "module ex.e0441\nfn total(x: f64) -> f64 {\n    let mut s = 0.5;\n    s += x;\n    return s;\n}\nfn main() -> i32 {\n    return total(1.5) as i32;\n}\n"
    ),
    entry!(
        "E0442",
        "refutable pattern in `let`",
        "The pattern of a plain `let` must match every value of its type, because the names it binds \
         have to exist afterwards. `let Shape::Circle { r } = s;` leaves `r` unbound when `s` is \
         any other shape, and `let (0, x) = t;` when `t.0` is not 0, so both are rejected; the \
         diagnostic names a value the pattern misses, found by the same analysis as E0405. Where \
         that analysis cannot see inside the type (an array of more than 32 elements, a struct of \
         more than 32 fields), the pattern is checked by form instead, as a `for` head is (below): \
         `let [[a, ..]] = rows;` over a `[][f32; 64]` is rejected because `[x]` on a slice tests \
         its length. Use \
         `let … else` to leave the enclosing function or loop when the value does not match — \
         `let Shape::Circle { r } = s else { return 0.0; };` — or a `match`. Destructuring patterns \
         that cannot fail are fine: a tuple, struct or array of names (`let (a, b) = t;`, \
         `let Point { x, y } = p;`, `let [first, .., last] = a;`), and the single variant of a \
         one-variant enum. The same rule holds for the pattern of a destructuring `for` head \
         (`for (i, x) in pairs`), which binds every element of an array or slice without testing \
         it: there it is checked by form, so a literal, range, path, or-pattern or variant of a \
         multi-variant enum anywhere inside (`for (0, x) in pairs`) is rejected even when it \
         happens to cover every value, and so is an array pattern that lists an element of a \
         `[]T` (a run-time length test). Match inside the loop body instead.",
        bad: "module ex.e0442\nenum Shape { Circle(i32), Square(i32) }\nfn radius(s: Shape) -> i32 {\n    let Shape::Circle(r) = s;\n    return r;\n}\nfn main() -> i32 {\n    return radius(Shape::Circle(3));\n}\n",
        good: "module ex.e0442\nenum Shape { Circle(i32), Square(i32) }\nfn radius(s: Shape) -> i32 {\n    let Shape::Circle(r) = s else {\n        return 0;\n    };\n    return r;\n}\nfn main() -> i32 {\n    return radius(Shape::Circle(3));\n}\n"
    ),
    entry!(
        "E0445",
        "integer operand of a floating-point builtin",
        "A builtin that is defined on floating-point values was given an integer value (an \
         integer *literal* is fine: it converts, as `sqrt(2)` does). The common case is \
         `fmax`/`fmin`: on integers they used to convert both operands to `f32` without a word, \
         so `fmax(a, b)` over two `i64` above 2^24 returned a rounded number. Use `max`/`min`, \
         which keep the operands' integer type, or convert explicitly (`fmax(a as f64, b as \
         f64)`) if a float result is really meant. The same applies to `fma`, `copysign`, \
         `logaddexp`, `bits`, `isnan`, `isinf`, `isfinite`, `signbit`, `ldexp`'s value, `frexp`, \
         `erfc`, `erfinv`, `lgamma` and `f32_to_f8e4m3`/`f32_to_f8e5m2`.",
        bad: "module ex.e0445\nfn larger(a: i64, b: i64) -> i64 {\n    return fmax(a, b) as i64;\n}\nfn main() -> i32 {\n    return larger(16777217, 3) as i32;\n}\n",
        good: "module ex.e0445\nfn larger(a: i64, b: i64) -> i64 {\n    return max(a, b);\n}\nfn main() -> i32 {\n    return larger(16777217, 3) as i32;\n}\n"
    ),
    entry!(
        "E0446",
        "operand is not an integer this builtin accepts",
        "An integer builtin was given something it is not defined on. `popcount`, `clz`, `ctz`, \
         `bswap`, `rotl`, `rotr`, `sat_add`, `sat_sub`, `mulhi`, `widening_mul` and the \
         `overflowing_*` family operate on integers (any width and signedness, not `bool` or \
         `char`) — take a float's bits first with `bits(x)`. A bit-pattern builtin needs exactly \
         its width: `f32_from_bits` a 32-bit integer, `f64_from_bits` 64, `f16_from_bits` and \
         `bf16_from_bits` 16, `f8e4m3_to_f32`/`f8e5m2_to_f32` 8 (an unsuffixed literal adapts). \
         `sat_cast::<T>` converts to an integer type `T` only, and `ldexp`'s exponent must be an \
         integer. `widening_mul` of two 128-bit operands has no result type — there are no 256-bit \
         integers — so use `mulhi` for the high half and `*` for the low half. `atomic_add` adds \
         to an integer place of at most 64 bits only, with an integer value: integer addition \
         commutes, so the total is the same in every thread schedule, while a float sum depends \
         on the order of the adds (there is no atomic float add, and no single-instruction \
         128-bit one). The bounds of a range index `s[a..b]` must be integers too.",
        bad: "module ex.e0446\nfn ones(x: f32) -> i32 {\n    return popcount(x);\n}\nfn main() -> i32 {\n    return ones(1.5);\n}\n",
        good: "module ex.e0446\nfn ones(x: f32) -> i32 {\n    return popcount(bits(x)) as i32;\n}\nfn main() -> i32 {\n    return ones(1.5);\n}\n"
    ),
    entry!(
        "E0447",
        "integer builtin operands of different types",
        "`sat_add`, `sat_sub`, `mulhi`, `widening_mul` and `overflowing_add`/`_sub`/`_mul` compute \
         at one integer type, and that type is the answer: `sat_add` of 100 and 100 is 127 at \
         `i8` but 200 at `i32`. So two operands of different integer types are not silently \
         widened, as `+` would widen them; convert one with `as` to the type you mean. An \
         unsuffixed literal adapts to the other operand's type (`sat_add(x, 1)`), and is \
         range-checked there. `atomic_add(place, v)` likewise adds a value of the place's own \
         type, wrapping at the place's width.",
        bad: "module ex.e0447\nfn brighten(p: u8, delta: i32) -> u8 {\n    return sat_add(p, delta);\n}\nfn main() -> i32 {\n    return brighten(200, 100) as i32;\n}\n",
        good: "module ex.e0447\nfn brighten(p: u8, delta: u8) -> u8 {\n    return sat_add(p, delta);\n}\nfn main() -> i32 {\n    return brighten(200, 100) as i32;\n}\n"
    ),
    entry!(
        "E0448",
        "value has no printed form",
        "`print`, `println`, `print_raw`, `eprint`, `eprintln` and every `printf` / `eprintf` / \
         `sprintf` argument print a scalar (an integer, a float, a `bool`, a `char`), a `*u8` \
         string or a `str` (its text), a C-style enum (its discriminant) or — structurally, like Rust's `{:?}` — an \
         array, a slice of scalars, a tuple or a struct whose every element prints; a reference \
         prints what it refers to. A value with no printed form is rejected instead of printing \
         something backend-dependent: a raw pointer other than `*u8` is an address (the \
         interpreter's slot index and a native run's machine address disagree), so dereference it; \
         a data-carrying enum prints by matching on it; a slice of non-scalars, a reference or a \
         `str` inside an aggregate, a SIMD vector or a function value prints element by element, \
         field by field, lane by lane or not at all. (A whole tensor and `()` are `E0437` under `print`.)",
        bad: "module ex.e0448\nfn main() -> i32 {\n    let mut x: i32 = 5;\n    let p: *mut i32 = &mut x;\n    print(p);\n    return 0;\n}\n",
        good: "module ex.e0448\nfn main() -> i32 {\n    let mut x: i32 = 5;\n    let p: *mut i32 = &mut x;\n    print(*p);\n    return 0;\n}\n"
    ),
    entry!(
        "E0449",
        "malformed format string",
        "The format string of `printf`, `eprintf` or `sprintf` is parsed when the program is \
         compiled, so it must be a string literal and well-formed: each `{}` or `{:spec}` \
         placeholder takes the next argument in order, `{{` and `}}` print a literal brace, and a \
         spec is `[[fill]align][+][#][0][width][.precision][type]` with an ASCII fill, `<`, `^` \
         or `>` alignment, a width and precision of at most 255, and `type` one of `?` (debug: a \
         string or char quoted), `x`, `X`, `o`, `b` (an integer in hex, octal or binary) or `e`, \
         `E` (scientific). An unclosed `{`, a lone `}`, a named or positional argument (`{0}`, \
         `{x}`), a `.` with no precision or an unknown type is this error.",
        bad: "module ex.e0449\nfn main() -> i32 {\n    printf(\"total {\", 3);\n    return 0;\n}\n",
        good: "module ex.e0449\nfn main() -> i32 {\n    printf(\"total {}\", 3);\n    return 0;\n}\n"
    ),
    entry!(
        "E0450",
        "format arguments do not match the format string",
        "A `printf` / `eprintf` / `sprintf` call must supply exactly one argument per placeholder, \
         and each placeholder's spec must apply to its argument's type: `x`, `X`, `o` and `b` \
         format an integer; `e` and `E` a number; `#` only adds the prefix of those radix styles; \
         a precision applies to a float (its digits after the point), a string (its maximum \
         length) or the elements of an aggregate, never to an integer, `bool` or `char`; and the \
         `+` and `0` flags format a number. Supply the missing argument, drop the extra one, or \
         fix the spec — or cast the value to the type the spec formats (`x as i64`).",
        bad: "module ex.e0450\nfn main() -> i32 {\n    printf(\"{} {}\", 1);\n    return 0;\n}\n",
        good: "module ex.e0450\nfn main() -> i32 {\n    printf(\"{} {}\", 1, 2);\n    return 0;\n}\n"
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
    entry!(
        "E0521",
        "`where` constraint not satisfied",
        "A function's `where` clause may constrain its dimensions: `where K % 16 == 0, M <= 4096` \
         compares expressions over integer literals, the function's generic dimensions and \
         `const`s (`+ - * / %`, joined with `&&` / `||`). Each predicate is evaluated at every call \
         whose dimensions it names are compile-time constants, and this error means one is false \
         for this call; the message shows the dimensions the call fixed (`K = 24`). Pass a buffer \
         that satisfies the constraint, or pad it. A predicate that names no dimension and is \
         false can never be satisfied and is reported at the declaration. A call from a generic \
         body whose dimensions are still symbolic is not checked there: the constraint is checked \
         where the dimensions become known.",
        bad: "module ex.e0521\nfn first<K>(a: Tensor[f32, K]) -> f32 where K % 4 == 0 {\n    return a[0];\n}\nfn main() -> i32 {\n    let a: [f32; 6] = [1.0; 6];\n    return first(a) as i32;\n}\n",
        good: "module ex.e0521\nfn first<K>(a: Tensor[f32, K]) -> f32 where K % 4 == 0 {\n    return a[0];\n}\nfn main() -> i32 {\n    let a: [f32; 8] = [1.0; 8];\n    return first(a) as i32;\n}\n"
    ),
    entry!(
        "E0522",
        "static assertion failed",
        "`static_assert(cond)` (or `static_assert(cond, \"message\")`) is checked at compile time \
         and generates no code. Its condition is a comparison over integer literals, `const`s \
         and, inside a generic function, the function's dimensions. A condition over literals and \
         `const`s is decided where it is written; one over the function's dimensions is decided at \
         each call, exactly like a `where` predicate, and this error then names the dimensions the \
         call fixed. The message, if any, is shown as a note."
    ),
    entry!(
        "E0523",
        "a `const` generic parameter must be `usize`",
        "A `const` generic parameter (`fn f<const N: usize>`) is a compile-time dimension: it is \
         bound by a turbofish or an argument's shape and passed as a hidden `usize`, and used as a \
         value it is a `usize`. Any other declared type (`const FLAG: bool`, `const ACT: i32`) \
         used to be accepted and silently treated as a `usize`; declare the parameter `usize`, or \
         pass a flag or an activation code as an ordinary parameter."
    ),
    entry!(
        "E0524",
        "not a compile-time dimension expression",
        "A `where` predicate, and a `static_assert` condition, must be evaluable at compile time: \
         integer literals, the function's generic dimensions and `const`s, combined with \
         `+ - * / %`, compared with `== != < <= > >=`, and joined with `&&` / `||`. A local \
         variable, a parameter's value, a call (`a.len()`) or a cast is not. A `static_assert` \
         that names the function's dimensions is checked at each call, so it must be a top-level \
         statement of the function body, not nested in a block or a loop."
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
        "W0008",
        "unused import (`unused_imports`)",
        "An `import` brings nothing the file uses into scope: no name written in the file resolves \
         through it — not an item of a glob import (`import a.b;`), not its qualifier (`b::f`), \
         not an `as` alias (`import a.b as c;` then `c::f`), and not a name of a `{x, y}` \
         selection, whose unused names are reported on their own when others are used. Imports \
         are judged per file, by the same module-scoped resolution that decides what every name \
         means: a local that shadows an imported name does not use the import, and of two imports \
         of one module the second is the unused one. An `impl` is found without a name — a \
         method through its receiver's type, a trait bound through any `impl` in the program — \
         so an import is also kept when the module it loads holds an `impl` that defines a method \
         called in any file, or implements a trait any generic parameter is bounded by: deleting \
         every reported import never changes what the program means. The warning carries a \
         machine-applicable fix that deletes the import, or rewrites the selection to the names \
         that are used. An object or IR artifact (`--emit=obj`, \
         `--emit=llvm-ir`) reports nothing: a library may import a module only for the functions \
         it exports. In the examples, `// file: <path>` starts another file of the program.",
        bad: "module ex.w0008\nimport geom;\n\nfn main() -> i32 {\n    return 0;\n}\n// file: geom.wk\nmodule geom\n\npub fn area(w: f32, h: f32) -> f32 {\n    return w * h;\n}\n",
        good: "module ex.w0008\nimport geom;\n\nfn main() -> i32 {\n    return area(2.0, 3.0) as i32;\n}\n// file: geom.wk\nmodule geom\n\npub fn area(w: f32, h: f32) -> f32 {\n    return w * h;\n}\n"
    ),
    entry!(
        "W0020",
        "unreachable pattern (`unreachable_patterns`)",
        "A `match` arm can never be selected: every value its pattern matches is already matched by \
         an earlier arm without a guard — `5 => …` after `0..=9 => …`, `(0, 0) => …` after \
         `(_, _) => …`, or any arm after a `_` arm. Arms are tried in order, so the arm is dead, \
         which usually means the arms are in the wrong order or an earlier pattern is broader than \
         intended. A guarded arm covers nothing (its guard may be false), so it never makes a later \
         arm unreachable. The verdict comes from the same pattern-usefulness analysis as E0405. \
         Delete the arm, or move it before the arm that covers it."
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
         short."
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
         type and the function that declares it, and points at the declaration. The interpreter \
         has no byte layout and runs the \
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
