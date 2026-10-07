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

/// One catalog entry: a stable code, a one-line title, and a longer explanation (Markdown-ish).
pub struct Explanation {
    pub code: &'static str,
    pub title: &'static str,
    pub body: &'static str,
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
         in a loop."
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
         the pointer itself is not being reassigned."
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
         set is closed: `@parallel`, `@generic`, `@tile`, `@extern`, `@inline`, `@simd`, `@align`, \
         `@export`. Anything else — most importantly a typo of a real one, `@parellel` — used to be \
         accepted and discarded, so the program compiled with none of the requested behavior and no \
         diagnostic. Placement: `@generic` (opt this function out of kernel recognition) is read \
         from the `fn` item, so it is rejected on a statement, a parameter, or a non-`fn` item \
         rather than silently promising a per-loop opt-out. The parsed-but-discarded attributes \
         (`@inline`, `@simd`, `@align`, `@export`) stay legal wherever they parse."
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
        "E0401",
        "type mismatch",
        "A value's type does not match the type required by its context — for example a `let` with \
         an explicit annotation whose initializer has a different type. Unsuffixed numeric \
         literals adapt to an annotation, but typed values must match exactly."
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
         arms (`if …`) do not count toward coverage because their guard may be false."
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
}
