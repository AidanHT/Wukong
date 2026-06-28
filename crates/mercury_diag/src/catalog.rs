//! The stable error-code catalog.
//!
//! Every diagnostic code Mercury can emit has a stable identifier (`E0501`, `C0001`, …) and an
//! extended explanation here. `mercuryc --explain <CODE>` prints the explanation; this keeps the
//! inline diagnostic terse while giving users a place to learn the rule and how to fix it.
//!
//! Code ranges:
//!   * `E00xx` — generic / internal-surface errors
//!   * `E01xx` — lexer
//!   * `E02xx` — parser
//!   * `E03xx` — name resolution
//!   * `E04xx` — type checking
//!   * `E05xx` — shape checking (Mercury's headline analysis)
//!   * `C0xxx` — codegen / lowering limitations

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

/// Every code Mercury knows about, for `--explain` and for listing in docs.
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
        "E0001",
        "malformed type syntax",
        "A type expression was not well-formed. Tensor and SIMD-vector element types must be \
         scalars, dimensions must be integers, names, or `?`, and layouts must be one of the \
         known kinds (`contiguous`, `col_major`, `strided`, `tiled`)."
    ),
    entry!(
        "E0101",
        "unexpected character",
        "The lexer found a character that cannot begin any Mercury token. Remove it or place it \
         inside a string/char literal."
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
        "E0200",
        "expected a specific token",
        "The parser needed a particular token (for example a closing `)` or `}`) but found \
         something else. Check for a missing delimiter or separator."
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
        "At module scope Mercury expects an item: `fn`, `struct`, `enum`, `const`, `import`, or \
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
         to allow reassignment, or introduce a new binding with another `let`. Mutating *through* \
         the binding — an array element `a[i] = …`, a struct field `s.f = …`, or a pointee \
         `*p = …` — is still allowed; only rebinding the name itself is rejected."
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
        "recursive struct has infinite size",
        "A struct contains itself by value — directly (`struct S { x: S }`) or through a chain of \
         structs — so its size would be infinite and the compiler cannot lay it out. Store the \
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
        "E0501",
        "tensor rank mismatch",
        "Two tensors were required to have the same number of dimensions (rank) but did not. For \
         example, passing a rank-3 tensor where a rank-2 tensor is expected. This is checked at \
         compile time — it can never become a runtime shape bug.\n\nThis code is also reported \
         when a compile-time-constant index is out of bounds — for a fixed-size array (e.g. \
         `a[5]` on a `[T; 4]`) or for a static tensor dimension (e.g. `a[5, 0]` on a \
         `Tensor[f32, 2, 2]`): the valid indices are `0..len` on each axis."
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
            assert!(e.code.starts_with('E') || e.code.starts_with('C'));
            assert!(!e.title.is_empty());
            assert!(e.body.len() > 20, "{} has a thin explanation", e.code);
        }
    }
}
