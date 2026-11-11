//! `--emit=doc` pages, and doc comments through the formatter. Every expected text here was written
//! by hand from the rules in `doc.rs` and the crate docs, never pasted from the output.

use crate::doc::document;
use crate::{format_source, FmtError, FmtOptions};
use wukong_span::SourceId;

fn page(src: &str) -> String {
    match document(src, SourceId(0), "file.wk", &FmtOptions::default()) {
        Ok(s) => s,
        Err(e) => panic!("refused: {e:?}"),
    }
}

const GEO: &str = r#"//! Geometry helpers.
//!
//! Second paragraph.
module geo

/// Adds two numbers.
///
/// ```wk
/// assert(add(1, 2) == 3);
/// ```
@inline
pub fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn undocumented() {}

/// A point.
struct Point {
    /// Horizontal.
    pub x: f32,
    y: f32,
}

/// Shapes.
enum Shape {
    /// A circle of radius r.
    Circle(f32),
    Rect { /// Width.
        w: f32, h: f32 },
    Dot,
}

/// The answer.
const ANSWER: i32 = 42;

/// Real numbers.
type Real = f32;

/// Things with an area.
trait Area {
    /// The area.
    fn area(self) -> f32;
}

impl Area for Point {
    /// Always zero.
    fn area(self) -> f32 { return 0.0; }
}
"#;

const GEO_MD: &str = r#"# Module `geo`

Geometry helpers.

Second paragraph.

## `fn add`

```wk
@inline
pub fn add(a: i32, b: i32) -> i32
```

Adds two numbers.

```wk
assert(add(1, 2) == 3);
```

## `fn undocumented`

```wk
fn undocumented()
```

## `struct Point`

```wk
struct Point {
    pub x: f32,
    y: f32,
}
```

A point.

### Fields

- `pub x: f32`

  Horizontal.

- `y: f32`

## `enum Shape`

```wk
enum Shape {
    Circle(f32),
    Rect { w: f32, h: f32 },
    Dot,
}
```

Shapes.

### Variants

- `Circle(f32)`

  A circle of radius r.

- `Rect { w: f32, h: f32 }`

  - `w: f32`

    Width.

- `Dot`

## `const ANSWER`

```wk
const ANSWER: i32 = 42;
```

The answer.

## `type Real`

```wk
type Real = f32;
```

Real numbers.

## `trait Area`

```wk
trait Area {
    fn area(self) -> f32;
}
```

Things with an area.

### Methods

- `fn area(self) -> f32`

  The area.

## `impl Area for Point`

```wk
impl Area for Point
```

### Methods

- `fn area(self) -> f32`

  Always zero.
"#;

/// The whole page for a file with every documentable construct.
#[test]
fn a_page_documents_every_construct_in_source_order() {
    assert_eq!(page(GEO), GEO_MD, "\n--- got ---\n{}", page(GEO));
}

/// A file with no `module` header is titled by the name it is given; a signature too long for
/// the line wraps exactly as the formatter wraps it; an `extern` block lists its declarations; an
/// undocumented list stays tight.
#[test]
fn title_wrapping_and_extern_declarations() {
    let src = "fn a_rather_long_function_name(first_argument: f32, second_argument: f32, third_argument: f32) -> f32 { return 0.0; }\n\
               extern \"C\" {\n    /// Absolute value.\n    fn abs(x: i32) -> i32;\n    fn labs(x: i64) -> i64;\n}\n\
               struct S { a: i32, b: i32 }\n";
    let want = "# `file.wk`\n\
                \n\
                ## `fn a_rather_long_function_name`\n\
                \n\
                ```wk\n\
                fn a_rather_long_function_name(\n    first_argument: f32,\n    second_argument: f32,\n    third_argument: f32,\n) -> f32\n\
                ```\n\
                \n\
                ## `extern \"C\"`\n\
                \n\
                ```wk\n\
                extern \"C\" {\n    fn abs(x: i32) -> i32;\n    fn labs(x: i64) -> i64;\n}\n\
                ```\n\
                \n\
                ### Declarations\n\
                \n\
                - `fn abs(x: i32) -> i32`\n\
                \n\
                \x20 Absolute value.\n\
                \n\
                - `fn labs(x: i64) -> i64`\n\
                \n\
                ## `struct S`\n\
                \n\
                ```wk\n\
                struct S {\n    a: i32,\n    b: i32,\n}\n\
                ```\n\
                \n\
                ### Fields\n\
                \n\
                - `a: i32`\n\
                - `b: i32`\n";
    assert_eq!(page(src), want, "\n--- got ---\n{}", page(src));
}

/// The page is a function of the source alone, and a file that does not parse — a misplaced doc
/// comment included — has none.
#[test]
fn pages_are_deterministic_and_need_a_clean_parse() {
    assert_eq!(page(GEO), page(GEO));
    match document(
        "fn f() {}\n/// dangling\n",
        SourceId(0),
        "x.wk",
        &FmtOptions::default(),
    ) {
        Err(FmtError::Syntax(d)) => assert_eq!(d[0].code, Some("E0210")),
        other => panic!("expected E0210, got {other:?}"),
    }
}

// ---- Doc comments through the formatter ----

fn fmt(src: &str) -> Result<String, FmtError> {
    format_source(src, SourceId(0), &FmtOptions::default())
}

/// Doc comments survive formatting on every documentable construct, and on the construct they
/// documented: the self-check's AST law compares the docs, so a moved doc comment is refused.
#[test]
fn doc_comments_survive_formatting_and_stay_attached() {
    let input = "//! File.
module m
/// Adds.
@inline
/// More.
fn add(a:i32,b:i32)->i32{return a+b;}
struct P{ /// x
x:f32, /// y
y:f32}
enum E{/// a
A,/// b
B}
trait T{/// m
fn m(self)->i32;}
impl T for P{/// impl m
fn m(self)->i32{return 1;}}
extern \"C\"{/// abs
fn abs(x:i32)->i32;}
";
    let want = "//! File.
module m
/// Adds.
@inline
/// More.
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
struct P {
    /// x
    x: f32, /// y
    y: f32,
}
enum E {
    /// a
    A, /// b
    B,
}
trait T {
    /// m
    fn m(self) -> i32;
}
impl T for P {
    /// impl m
    fn m(self) -> i32 {
        return 1;
    }
}
extern \"C\" {
    /// abs
    fn abs(x: i32) -> i32;
}
";
    let got = fmt(input).unwrap_or_else(|e| panic!("refused: {e:?}"));
    assert_eq!(got, want, "\n--- got ---\n{got}");
    assert_eq!(fmt(&got).ok().as_deref(), Some(want), "not idempotent");
    // The docs the reformatted file parses to are the input's, item for item.
    let docs = |src: &str| {
        let mut i = wukong_span::Interner::new();
        let (m, d) = wukong_parser::parse_module(src, SourceId(0), &mut i);
        assert!(d.is_empty(), "{d:?}");
        format!("{:?}", m)
            .split("Docs { lines: [")
            .skip(1)
            .map(|s| {
                s.split("text: \"")
                    .skip(1)
                    .map(|t| t.split('"').next().unwrap_or_default().to_string())
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(docs(input), docs(want));
}

/// A file whose doc comment documents nothing is a syntax error to the formatter, as to the
/// compiler: it is left untouched.
#[test]
fn a_misplaced_doc_comment_is_refused() {
    match fmt("fn f() {\n    /// on a statement\n    let x = 1;\n}\n") {
        Err(FmtError::Syntax(d)) => assert_eq!(d[0].code, Some("E0210")),
        other => panic!("expected a syntax refusal, got {other:?}"),
    }
}

/// Law (b) sees a doc comment that moved to another item, even though the comments, in order, and
/// the tokens are unchanged.
#[test]
fn the_ast_law_sees_a_doc_comment_change_items() {
    use crate::check::{self_check, Original};
    let src = "fn f() {}\n/// g's\nfn g() {}\n";
    let (tokens, comments, _) = wukong_lexer::tokenize_with_comments(src, SourceId(0));
    let mut interner = wukong_span::Interner::new();
    let parsed = wukong_parser::parse_module_tokens_with_lints(&tokens, src, &mut interner, 0);
    let orig = Original {
        src,
        tokens: &tokens,
        comments: &comments,
        module: &parsed.module,
        source: SourceId(0),
    };
    assert_eq!(self_check(&orig, &mut interner, src), Ok(()));
    let e = self_check(&orig, &mut interner, "/// g's\nfn f() {}\nfn g() {}\n").unwrap_err();
    assert!(e.contains("self-check (b)"), "{e}");
}
