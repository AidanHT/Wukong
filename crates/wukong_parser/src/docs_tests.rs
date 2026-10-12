//! Doc-comment attachment and `E0210` (see `docs.rs`).

use crate::parse_module;
use wukong_ast::*;
use wukong_diag::Diagnostic;
use wukong_span::{Interner, SourceId};

fn parse(src: &str) -> (Module, Vec<Diagnostic>, Interner) {
    let mut i = Interner::new();
    let (m, d) = parse_module(src, SourceId(0), &mut i);
    (m, d, i)
}

fn clean(src: &str) -> (Module, Interner) {
    let (m, d, i) = parse(src);
    assert!(d.is_empty(), "unexpected diagnostics: {d:?}");
    (m, i)
}

fn codes(src: &str) -> Vec<&'static str> {
    parse(src).1.iter().filter_map(|d| d.code).collect()
}

fn fn_docs<'m>(m: &'m Module, i: &Interner, name: &str) -> &'m Docs {
    m.items
        .iter()
        .find(|it| matches!(&it.kind, ItemKind::Fn(f) if i.resolve(f.name.sym) == name))
        .map(|it| &it.docs)
        .unwrap_or_else(|| panic!("no fn {name}"))
}

fn item(m: &Module, pick: impl Fn(&ItemKind) -> bool) -> &Item {
    m.items.iter().find(|it| pick(&it.kind)).expect("item")
}

const EVERY_CONSTRUCT: &str = r#"//! The file.
//! Second line.
module demo

/// Adds.
/// Twice.
@inline
/// After the attribute.
pub fn add(a: i32, b: i32) -> i32 { return a + b; }

fn plain() {}

/// A point.
struct P {
    /// x coordinate
    pub x: f32,
    y: f32,
    /// last
    @align(4) z: f32,
}

/// Shapes.
enum Shape {
    /// round
    Circle(f32),
    Square { /// side
        side: f32 },
    Dot,
}

/// Answer.
const N: i32 = 42;

/// Alias.
type Real = f32;

/// A trait.
trait Area {
    /// area
    fn area(self) -> f32;
    fn sides(self) -> i32;
}

/// The impl.
impl Area for P {
    /// impl area
    fn area(self) -> f32 { return self.x; }
    /// impl sides
    pub fn sides(self) -> i32 { return 4; }
}

/// C.
extern "C" {
    /// abs
    fn abs(x: i32) -> i32;
    pub fn labs(x: i64) -> i64;
}
"#;

/// Every documentable construct takes the `///` lines before it — through its attributes and
/// `pub` — and nothing else's; the `//!` lines before the first item are the file's.
#[test]
fn docs_attach_to_the_construct_that_follows() {
    let (m, i) = clean(EVERY_CONSTRUCT);
    assert_eq!(m.docs.text(), "The file.\nSecond line.");
    assert_eq!(
        fn_docs(&m, &i, "add").text(),
        "Adds.\nTwice.\nAfter the attribute."
    );
    assert!(fn_docs(&m, &i, "plain").is_empty());

    let st = item(&m, |k| matches!(k, ItemKind::Struct(_)));
    assert_eq!(st.docs.text(), "A point.");
    let ItemKind::Struct(s) = &st.kind else {
        unreachable!()
    };
    let fd: Vec<String> = s.fields.iter().map(|f| f.docs.text()).collect();
    assert_eq!(fd, ["x coordinate", "", "last"]);

    let en = item(&m, |k| matches!(k, ItemKind::Enum(_)));
    assert_eq!(en.docs.text(), "Shapes.");
    let ItemKind::Enum(e) = &en.kind else {
        unreachable!()
    };
    let vd: Vec<String> = e.variants.iter().map(|v| v.docs.text()).collect();
    assert_eq!(vd, ["round", "", ""]);
    let VariantData::Struct(sf) = &e.variants[1].data else {
        unreachable!()
    };
    assert_eq!(sf[0].docs.text(), "side");

    assert_eq!(
        item(&m, |k| matches!(k, ItemKind::Const(_))).docs.text(),
        "Answer."
    );
    assert_eq!(
        item(&m, |k| matches!(k, ItemKind::TypeAlias(_)))
            .docs
            .text(),
        "Alias."
    );

    let tr = item(&m, |k| matches!(k, ItemKind::Trait(_)));
    assert_eq!(tr.docs.text(), "A trait.");
    let ItemKind::Trait(t) = &tr.kind else {
        unreachable!()
    };
    let md: Vec<String> = t.method_docs.iter().map(|d| d.text()).collect();
    assert_eq!(md, ["area", ""]);

    assert_eq!(
        item(&m, |k| matches!(k, ItemKind::Impl(_))).docs.text(),
        "The impl."
    );
    assert_eq!(fn_docs(&m, &i, "P.area").text(), "impl area");
    assert_eq!(fn_docs(&m, &i, "P.sides").text(), "impl sides");

    let ex = item(&m, |k| matches!(k, ItemKind::Extern(_)));
    assert_eq!(ex.docs.text(), "C.");
    let ItemKind::Extern(x) = &ex.kind else {
        unreachable!()
    };
    let xd: Vec<String> = x.item_docs.iter().map(|d| d.text()).collect();
    assert_eq!(xd, ["abs", ""]);
}

/// `--emit=ast` shows the docs where they attach.
#[test]
fn the_ast_printer_shows_docs() {
    let (m, i) = clean("//! File.\n/// F.\nfn f() {}\nstruct S {\n    /// x\n    x: i32,\n}\n");
    let out = print::print_module(&m, &i);
    assert!(
        out.starts_with("doc \"File.\"\ndoc \"F.\"\nfn f\n"),
        "{out}"
    );
    assert!(out.contains("field x: i32\n    doc \"x\"\n"), "{out}");
}

/// The text keeps everything after the marker; the common indent goes, an extra one stays; a
/// CRLF line's CR is not text; `////` and block comments are not doc comments.
#[test]
fn doc_text_is_unindented_markdown() {
    let (m, i) = clean(
        "///  Two spaces.\r\n///\r\n///      indented code\r\n//// not a doc\r\n/** nor this */\r\nfn f() {}\r\n",
    );
    let d = fn_docs(&m, &i, "f");
    assert_eq!(d.lines.len(), 3);
    assert_eq!(d.lines[0].text, "  Two spaces.");
    assert_eq!(d.text(), "Two spaces.\n\n    indented code");
    // Each unindented line knows where its text starts in the source.
    let src = "/// a\n///   b\nfn g() {}\n";
    let (m, i) = clean(src);
    let lines = fn_docs(&m, &i, "g").unindented();
    assert_eq!(lines[0], ("a", 4));
    let at = lines[1].1 as usize;
    assert_eq!(&src[at..at + 3], "  b");
}

/// A doc comment that documents nothing is E0210, once per run of lines.
#[test]
fn a_doc_comment_that_documents_nothing_is_e0210() {
    let cases = [
        // Before the `}` that closes a struct, an enum, a trait, a block.
        "struct S { x: i32, /// dangling\n}\n",
        "enum E { A, /// dangling\n/// two lines, one error\n}\n",
        "trait T { fn m(self); /// dangling\n}\n",
        "fn f() { let x = 1; /// dangling\n}\n",
        // At the end of the file.
        "fn f() {}\n/// nothing after\n",
        // Before a statement or inside an expression.
        "fn f() -> i32 {\n    /// on a statement\n    let x = 1;\n    return x;\n}\n",
        "fn f() -> i32 { return 1 + /// inside\n 2; }\n",
        // `///` on the `module` header; `//!` after the first item.
        "/// header\nmodule m\nfn f() {}\n",
        "fn f() {}\n//! too late\nfn g() {}\n",
    ];
    for src in cases {
        assert_eq!(codes(src), ["E0210"], "{src:?}");
    }
    // The diagnostic covers the run, and a one-line run carries the `//` fix.
    let (_, d, _) = parse("fn f() {}\n/// a\n/// b\n");
    assert_eq!(d.len(), 1);
    let sp = d[0].primary_span().unwrap();
    assert_eq!((sp.lo, sp.hi), (10, 21));
    assert!(d[0].suggestions.is_empty());
    let (_, d, _) = parse("fn f() {}\n/// a\n");
    assert_eq!(d[0].suggestions.len(), 1);
    assert_eq!(d[0].suggestions[0].replacement, "//");
    // Two runs in different places: two errors.
    assert_eq!(
        codes("fn f() { /// one\n let x = 1; /// two\n}\n"),
        ["E0210", "E0210"]
    );
}

/// A doc comment inside a stretch error recovery skipped is not reported on top of the real
/// error.
#[test]
fn stray_docs_add_nothing_to_a_parse_error() {
    assert_eq!(
        codes("fn f( { /// x\n}\n/// y\n"),
        codes("fn f( { // x\n}\n// y\n")
    );
}

/// A statement-list parse (a REPL input) rejects a doc comment too: statements take none.
#[test]
fn a_statement_list_rejects_doc_comments() {
    let src = "/// doc\nlet x = 1;\n";
    let (tokens, _) = wukong_lexer::tokenize(src, SourceId(0));
    let out = crate::parse_stmts_tokens_with_lints(&tokens, src, &mut Interner::new(), 0);
    let codes: Vec<_> = out.diags.iter().filter_map(|d| d.code).collect();
    assert_eq!(codes, ["E0210"]);
}

/// A doc example parses as its items, then its statements: an attribute before a statement keeps
/// it a statement, and a tail expression stays the block's tail.
#[test]
fn a_doc_example_is_items_then_statements() {
    let src = "struct P { x: i32 }\n@inline\nfn get(p: P) -> i32 { return p.x; }\n\
               let p = P { x: 3 };\n@parallel\nfor i in 0..4 { }\nget(p)\n";
    let (tokens, lex) = wukong_lexer::tokenize(src, SourceId(0));
    assert!(lex.is_empty(), "{lex:?}");
    let mut i = Interner::new();
    let out = crate::parse_doctest_tokens(&tokens, src, &mut i, 0);
    assert!(out.diags.is_empty(), "{:?}", out.diags);
    assert_eq!(out.items.len(), 2);
    assert_eq!(out.block.stmts.len(), 2);
    assert!(out.block.tail.is_some());
    // Statements only, and items only.
    for (src, items, stmts) in [
        ("let x = 1;\nx + 1;\n", 0, 2),
        ("fn f() {}\ntype T = i32;\n", 2, 0),
    ] {
        let (tokens, _) = wukong_lexer::tokenize(src, SourceId(0));
        let out = crate::parse_doctest_tokens(&tokens, src, &mut Interner::new(), 0);
        assert!(out.diags.is_empty(), "{src:?}: {:?}", out.diags);
        assert_eq!(
            (out.items.len(), out.block.stmts.len()),
            (items, stmts),
            "{src:?}"
        );
    }
}
