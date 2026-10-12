//! Golden input/output pairs for every printing rule, each also checked for idempotence, plus the
//! refusals and the self-check's own teeth.
//!
//! Every expected output here was written by hand from the rules in the crate docs and
//! `docs/internals.md` (Formatter), never pasted from the formatter's output.

use crate::{format_source, FmtError, FmtOptions};
use wukong_span::SourceId;

fn fmt_w(src: &str, max_width: usize) -> Result<String, FmtError> {
    format_source(src, SourceId(0), &FmtOptions { max_width })
}

fn fmt(src: &str) -> Result<String, FmtError> {
    fmt_w(src, 100)
}

/// `input` formats to `want`, and `want` is a fixed point.
#[track_caller]
fn golden_w(input: &str, want: &str, max_width: usize) {
    let got = match fmt_w(input, max_width) {
        Ok(s) => s,
        Err(e) => panic!("refused: {e:?}\n--- input ---\n{input}"),
    };
    assert_eq!(got, want, "\n--- got ---\n{got}\n--- want ---\n{want}");
    match fmt_w(&got, max_width) {
        Ok(again) => assert_eq!(again, got, "not idempotent:\n{again}"),
        Err(e) => panic!("its own output is refused: {e:?}"),
    }
}

#[track_caller]
fn golden(input: &str, want: &str) {
    golden_w(input, want, 100);
}

/// A golden whose input is already canonical.
#[track_caller]
fn fixed(src: &str) {
    golden(src, src);
}

// ---- Expressions ----

#[test]
fn minimal_parentheses_follow_the_parser_precedences() {
    golden(
        "fn f() {
    let a = ((x + y)) * z - (p - q) - r / (s * t);
    let b = (x - y) - z;
    let c = x - (y - z);
    let d = (x || y) && z || (w && v);
    let e = (x << 2) + 1 < (y | z) ^ k;
    let g = x == (y < z);
}
",
        "fn f() {
    let a = (x + y) * z - (p - q) - r / (s * t);
    let b = x - y - z;
    let c = x - (y - z);
    let d = (x || y) && z || w && v;
    let e = (x << 2) + 1 < (y | z) ^ k;
    let g = x == (y < z);
}
",
    );
}

#[test]
fn cast_binds_tighter_than_binary_and_looser_than_prefix() {
    golden(
        "fn f() {
    let a = (-x) as u8;
    let b = -(x as i32);
    let c = (x as i32) + 1;
    let d = (x + 1) as i64;
    let e = (*p) as f32;
    let g = (y as Vec3).x;
    let h = (x as i32) as i64;
    let i = x as i32 < n;
}
",
        "fn f() {
    let a = -x as u8;
    let b = -(x as i32);
    let c = x as i32 + 1;
    let d = (x + 1) as i64;
    let e = *p as f32;
    let g = (y as Vec3).x;
    let h = x as i32 as i64;
    let i = x as i32 < n;
}
",
    );
}

#[test]
fn nested_references_never_print_as_and_and() {
    golden(
        "fn f(r: & &i32) {
    let a = & &x;
    let b = &(&mut y);
    let c = !(!z);
    let d = -(-w);
}
",
        "fn f(r: & &i32) {
    let a = & &x;
    let b = & &mut y;
    let c = !!z;
    let d = --w;
}
",
    );
}

#[test]
fn a_struct_literal_in_a_condition_head_keeps_its_parentheses() {
    golden(
        "fn f() {
    if (P { x: 1 }).x > 0 { g(); }
    if a[(P { x: 2 }).x] > 0 { g(); }
    if h((P { x: 3 })) { g(); }
    while (Q { y: 1 }).ok() { g(); }
    match (P { x: 4 }) { _ => {} }
    let p = (P { x: 5 });
}
",
        "fn f() {
    if (P { x: 1 }).x > 0 {
        g();
    }
    if a[(P { x: 2 }).x] > 0 {
        g();
    }
    if h(P { x: 3 }) {
        g();
    }
    while (Q { y: 1 }).ok() {
        g();
    }
    match (P { x: 4 }) {
        _ => {}
    }
    let p = P { x: 5 };
}
",
    );
}

#[test]
fn tuples_fields_paths_and_not() {
    golden(
        "fn f() {
    let t = (a,);
    let u = ((b));
    let v = t.0.0;
    let w = t.1 . 2;
    let c = Color :: Red;
    let d = p.x;
    let e = ~m;
    let g = !b;
    let h = (5).0;
    let i = ();
}
",
        "fn f() {
    let t = (a,);
    let u = b;
    let v = t.0.0;
    let w = (t.1).2;
    let c = Color::Red;
    let d = p.x;
    let e = ~m;
    let g = !b;
    let h = (5).0;
    let i = ();
}
",
    );
}

#[test]
fn postfix_bases_that_bind_looser_are_parenthesized() {
    golden(
        "fn f() {
    let a = (-x).abs();
    let b = (x + y)[0];
    let c = ({ v }).len();
    let d = (if c { p } else { q }).x;
    let e = f(x)(y)[1].z;
}
",
        "fn f() {
    let a = (-x).abs();
    let b = (x + y)[0];
    let c = ({
        v
    }).len();
    let d = (if c {
        p
    } else {
        q
    }).x;
    let e = f(x)(y)[1].z;
}
",
    );
}

// ---- Desugarings ----

#[test]
fn impl_blocks_print_their_receivers_and_self() {
    golden(
        "struct V { x: f32 }
impl V {
    fn a(self) -> f32 { return self.x; }
    fn b(mut   self) { self.x = 0.0; }
    fn c(&self) -> f32 { return self.x; }
    fn d(& mut self, k: f32) { self.x = self.x + k; }
    pub fn make() -> Self { return Self { x: 1.0 }; }
}
impl Shape for V { fn area(self) -> f32 { return 0.0; } }
",
        "struct V {
    x: f32,
}
impl V {
    fn a(self) -> f32 {
        return self.x;
    }
    fn b(mut self) {
        self.x = 0.0;
    }
    fn c(&self) -> f32 {
        return self.x;
    }
    fn d(&mut self, k: f32) {
        self.x = self.x + k;
    }
    pub fn make() -> Self {
        return Self { x: 1.0 };
    }
}
impl Shape for V {
    fn area(self) -> f32 {
        return 0.0;
    }
}
",
    );
}

#[test]
fn traits_keep_their_signatures() {
    golden(
        "pub trait Shape { fn area(self) -> f32; fn grow(mut self, k: f32); fn unit() -> Self; }\n",
        "pub trait Shape {
    fn area(self) -> f32;
    fn grow(mut self, k: f32);
    fn unit() -> Self;
}
",
    );
}

#[test]
fn if_let_with_no_else_an_else_and_an_else_if() {
    golden(
        "fn f(o: Opt) -> i32 {
    if let Opt::Some(v) = o { print(v); }
    let a = if let Opt::Some(v) = o { v } else { 0 };
    let b = if let Opt::Some(v) = o { v } else if let Opt::None = o { 1 } else if c { 2 } else { 3 };
    return a + b;
}
",
        "fn f(o: Opt) -> i32 {
    if let Opt::Some(v) = o {
        print(v);
    }
    let a = if let Opt::Some(v) = o {
        v
    } else {
        0
    };
    let b = if let Opt::Some(v) = o {
        v
    } else if let Opt::None = o {
        1
    } else if c {
        2
    } else {
        3
    };
    return a + b;
}
",
    );
}

#[test]
fn a_labeled_while_let() {
    golden(
        "fn f() {
    'outer: while let Opt::Some(v) = next(i) { i += v; if i > 9 { break 'outer; } }
    while let Opt::Some(w) = next(i) { i += w; }
}
",
        "fn f() {
    'outer: while let Opt::Some(v) = next(i) {
        i += v;
        if i > 9 {
            break 'outer;
        }
    }
    while let Opt::Some(w) = next(i) {
        i += w;
    }
}
",
    );
}

#[test]
fn let_else_with_zero_one_and_two_names_and_mut() {
    golden(
        "fn f(o: Opt, p: Pair) -> i32 {
    let Opt::None = o else { return 0; };
    let Opt::Some(v) = o else { return 1; };
    let Pair::P(a, b) = p else { return 2; };
    let mut Opt::Some(w) = o else { return 3; };
    return v + a + b + w;
}
",
        "fn f(o: Opt, p: Pair) -> i32 {
    let Opt::None = o else {
        return 0;
    };
    let Opt::Some(v) = o else {
        return 1;
    };
    let Pair::P(a, b) = p else {
        return 2;
    };
    let mut Opt::Some(w) = o else {
        return 3;
    };
    return v + a + b + w;
}
",
    );
}

#[test]
fn let_else_on_an_else_less_if_keeps_its_parentheses() {
    golden(
        "fn f() { let Opt::Some(v) = (if c { o }) else { return; }; }\n",
        "fn f() {
    let Opt::Some(v) = (if c {
        o
    }) else {
        return;
    };
}
",
    );
}

#[test]
fn local_const_and_defer_forms() {
    golden(
        "fn f() {
    const K: i32 = 4;
    let mut x = K;
    defer x += 1;
    defer { x = 0; }
    defer print(x);
}
",
        "fn f() {
    const K: i32 = 4;
    let mut x = K;
    defer x += 1;
    defer {
        x = 0;
    }
    defer print(x);
}
",
    );
}

#[test]
fn an_expression_bodied_fn_keeps_its_form() {
    golden(
        "fn sq(x: i32) -> i32 = x * x;\nfn id(x: i32) -> i32 = (x)\n",
        "fn sq(x: i32) -> i32 = x * x;\nfn id(x: i32) -> i32 = x;\n",
    );
}

#[test]
fn where_bounds_and_predicates_stay_in_the_where_clause() {
    golden(
        "fn f<T: A + Z, U, const N: usize>(x: T, y: U) -> i32 where U: B + C, N % 4 == 0, T: D { return 0; }\n",
        "fn f<T: A + Z, U, const N: usize>(x: T, y: U) -> i32 where U: B + C, N % 4 == 0, T: D {
    return 0;
}
",
    );
}

#[test]
fn extern_blocks_with_and_without_an_abi() {
    golden(
        "extern \"C\" { fn sinf(x: f32) -> f32; pub fn cosf(x: f32) -> f32; }
extern { fn tanf(x: f32) -> f32; }
pub extern { fn expf(x: f32) -> f32; }
extern fn logf(x: f32) -> f32;
pub extern \"C\" fn g();
",
        "extern \"C\" {
    fn sinf(x: f32) -> f32;
    pub fn cosf(x: f32) -> f32;
}
extern {
    fn tanf(x: f32) -> f32;
}
pub extern {
    fn expf(x: f32) -> f32;
}
extern fn logf(x: f32) -> f32;
pub extern \"C\" fn g();
",
    );
}

#[test]
fn items_attributes_imports_aliases_and_the_module_header() {
    golden(
        "module demo.app;
import std.math;
import lib.geom as g;
import lib.stats.{mean, var};
@inline
@parallel(grain = 4, restrict)
pub fn k() {}
pub const N: usize = 8
type Mat<M, N> = Tensor[f32, M, N];
enum E { A = 1, B(i32, f32), C { r: f32 }, D {} }
struct Unit {}
",
        "module demo.app
import std.math;
import lib.geom as g;
import lib.stats.{mean, var};
@inline @parallel(grain = 4, restrict)
pub fn k() {}
pub const N: usize = 8;
type Mat<M, N> = Tensor[f32, M, N];
enum E {
    A = 1,
    B(i32, f32),
    C { r: f32 },
    D {},
}
struct Unit {}
",
    );
}

#[test]
fn statement_attributes_go_on_one_line_above() {
    golden(
        "fn f(n: i32) {
    @parallel
    @simd for i in 0..n { g(i); }
}
",
        "fn f(n: i32) {
    @parallel @simd
    for i in 0..n {
        g(i);
    }
}
",
    );
}

// ---- Types and calls ----

#[test]
fn tensor_types_dimension_arithmetic_and_layouts() {
    golden(
        "fn k<N, M>(a: Tensor[f32, N*2, ((M-1)/2)+1, ?], b: Tensor[f32, 4, 0x10, .tiled(2, 0x2)]) {}
fn c(x: Tensor[f32, N, (N - 1) * 2, .col_major]) {}
",
        "fn k<N, M>(a: Tensor[f32, N * 2, (M - 1) / 2 + 1, ?], b: Tensor[f32, 4, 0x10, .tiled(2, 0x2)]) {}
fn c(x: Tensor[f32, N, (N - 1) * 2, .col_major]) {}
",
    );
}

#[test]
fn vector_spellings_generic_calls_and_cast_targets() {
    golden(
        "type Buf<T, N> = [T; N];
fn v(a: f32x8, b: vec[f32, 4], c: (i32), d: (i32,), e: *mut [i32; 4], f: fn(f32) -> f32) -> f32 {
    let p = alloc::<Buf<f32, 4>>(2);
    let q = sizeof[f32x8] + alignof[Buf<f32, 2>];
    return x as A::<f32>;
}
",
        "type Buf<T, N> = [T; N];
fn v(a: f32x8, b: vec[f32, 4], c: (i32), d: (i32,), e: *mut [i32; 4], f: fn(f32) -> f32) -> f32 {
    let p = alloc::<Buf<f32, 4>>(2);
    let q = sizeof[f32x8] + alignof[Buf<f32, 2>];
    return x as A::<f32>;
}
",
    );
}

#[test]
fn range_indexes_print_back_as_written() {
    golden(
        "fn f(s: str) {
    let a = s[1..4];
    let b = s[2..];
    let c = s[..3];
    let d = s[..];
    let e = (s[1..])[0];
    print(s[i + 1..n - 1]);
}
",
        "fn f(s: str) {
    let a = s[1..4];
    let b = s[2..];
    let c = s[..3];
    let d = s[..];
    let e = s[1..][0];
    print(s[i + 1..n - 1]);
}
",
    );
}

#[test]
fn closures_and_a_call_that_holds_a_block_closure() {
    // A call whose argument is a block-bodied closure (or any block-like expression) never
    // wraps: its arguments stay inline and the block opens over the following lines.
    golden(
        "fn f() {
    let a = apply(|x| x * k, 2.0);
    let b = run(|| 7);
    let c = map(|x: f32, y| -> f32 { x * y }, 1.0);
    let d = fold(|a: i32, b: i32| a + b + off, 1, 2, 3);
}
",
        "fn f() {
    let a = apply(|x| x * k, 2.0);
    let b = run(|| 7);
    let c = map(|x: f32, y| -> f32 {
        x * y
    }, 1.0);
    let d = fold(|a: i32, b: i32| a + b + off, 1, 2, 3);
}
",
    );
}

// ---- Loops and patterns ----

#[test]
fn labels_and_every_range_form() {
    golden(
        "fn f(n: i32) {
    'l: loop { break 'l; }
    'w: while n > 0 { continue 'w; }
    'f: for i in 0..4 { }
    for i in 0..n step 2 { }
    for i in 0.. { }
    for i in 0..=n { }
    for i in 0.. step 3 { }
    for x in xs { }
    let v = 'v: loop { break 'v 5; };
}
",
        "fn f(n: i32) {
    'l: loop {
        break 'l;
    }
    'w: while n > 0 {
        continue 'w;
    }
    'f: for i in 0..4 {}
    for i in 0..n step 2 {}
    for i in 0.. {}
    for i in 0..=n {}
    for i in 0.. step 3 {}
    for x in xs {}
    let v = 'v: loop {
        break 'v 5;
    };
}
",
    );
}

#[test]
fn every_pattern_kind() {
    golden(
        "fn f(v: T) -> i32 {
    match v {
        _ => 0,
        x => 1,
        (a, b) => 2,
        (c,) => 3,
        () => 4,
        7 => 5,
        -7 => 6,
        'q' => 7,
        true => 8,
        A | B | C => 9,
        E::Red => 10,
        0..10 => 11,
        'a'..='z' => 12,
        Op::Add(l, r) => 13,
        Shape::Circle { r, c: q, .. } => 14,
        Point { x, y: 0, .. } => 15,
        Point { x } => 16,
        n @ 1..=9 => 17,
        [h, .., t] => 18,
        [p, q] => 19,
    }
}
",
        "fn f(v: T) -> i32 {
    match v {
        _ => 0,
        x => 1,
        (a, b) => 2,
        (c,) => 3,
        () => 4,
        7 => 5,
        -7 => 6,
        'q' => 7,
        true => 8,
        A | B | C => 9,
        E::Red => 10,
        0..10 => 11,
        'a'..='z' => 12,
        Op::Add(l, r) => 13,
        Shape::Circle { r, c: q, .. } => 14,
        Point { x, y: 0, .. } => 15,
        Point { x } => 16,
        n @ 1..=9 => 17,
        [h, .., t] => 18,
        [p, q] => 19,
    }
}
",
    );
}

#[test]
fn match_guards_and_block_bodied_arms() {
    golden(
        "fn f(t: (i32, i32)) {
    match t {
        (0, y) if y > 0 => { print(y); },
        (x, _) => { print(x); }
        _ => print(0)
    }
    match k { 1 => { a(); }, -1 => { b(); }, _ => {} }
}
",
        "fn f(t: (i32, i32)) {
    match t {
        (0, y) if y > 0 => {
            print(y);
        },
        (x, _) => {
            print(x);
        }
        _ => print(0),
    }
    match k {
        1 => {
            a();
        },
        -1 => {
            b();
        }
        _ => {}
    }
}
",
    );
}

#[test]
fn if_let_chains_and_their_parentheses() {
    golden(
        "fn f(s: Shape) {
    if let Shape::Circle(r) = s && r > 2 { g(r); }
    if let Shape::Sq(w) = s && w > 1 && w < 10 && (w != 5 || k) { g(w); }
    if let true = (a && b) { g(0); }
    if let true = (a || b) && c { g(1); }
    while let 1..=9 = k && k != 5 { k += 1; }
}
",
        "fn f(s: Shape) {
    if let Shape::Circle(r) = s && r > 2 {
        g(r);
    }
    if let Shape::Sq(w) = s && w > 1 && w < 10 && (w != 5 || k) {
        g(w);
    }
    if let true = (a && b) {
        g(0);
    }
    if let true = (a || b) && c {
        g(1);
    }
    while let 1..=9 = k && k != 5 {
        k += 1;
    }
}
",
    );
}

#[test]
fn destructuring_for_heads() {
    fixed(
        "fn f(pairs: [(i32, f32); 4], pts: [Point; 2]) {
    for (i, x) in pairs {}
    for Point { x, y } in pts {}
    for [a, .., z] in rows {}
    for w @ (a, b) in pairs {}
}
",
    );
}

// ---- Statements ----

#[test]
fn a_block_like_statement_gets_its_semicolon_only_when_needed() {
    golden(
        "fn f() -> i32 {
    if c { a(); }
    if d { b(); };
    loop { break; }
    match x { _ => {} };
    *p = 1;
    if e { g(); };
    -x;
    { h(); };
    (y, z);
    if q { 1 } else { 2 }
}
fn g() { if c { a(); }; }
",
        "fn f() -> i32 {
    if c {
        a();
    }
    if d {
        b();
    }
    loop {
        break;
    }
    match x {
        _ => {}
    };
    *p = 1;
    if e {
        g();
    };
    -x;
    {
        h();
    };
    (y, z);
    if q {
        1
    } else {
        2
    }
}
fn g() {
    if c {
        a();
    };
}
",
    );
}

#[test]
fn a_missing_semicolon_is_added() {
    golden(
        "fn f() -> i32 {
    let a = 3
    print(a)
    return a
}
",
        "fn f() -> i32 {
    let a = 3;
    print(a);
    return a;
}
",
    );
}

// ---- Comments ----

#[test]
fn leading_and_trailing_comments_and_blank_lines() {
    golden(
        "// header
module m


/// doc for f
fn f() { // after the brace
    let a = 1; // trailing a


    // about b
    let b = 2;
    /* before the close */
}
// between items

fn g() {
    // only a comment
}
fn h() { /* inline */ }
// end of file
",
        "// header
module m

/// doc for f
fn f() {
    // after the brace
    let a = 1; // trailing a

    // about b
    let b = 2;
    /* before the close */
}
// between items

fn g() {
    // only a comment
}
fn h() {
    /* inline */
}
// end of file
",
    );
}

#[test]
fn two_trailing_line_comments_and_a_comment_inside_an_expression() {
    golden(
        "fn f() {
    let x = a + // first
        b; // second
    let y = c * /* scale */ d;
    z = 1; /* one */ /* two */ // three
}
",
        "fn f() {
    let x = a + b; // first
    // second
    let y = c * d; /* scale */
    z = 1; /* one */ /* two */ // three
}
",
    );
}

#[test]
fn per_parameter_comments_make_the_parameters_vertical() {
    golden(
        "fn f(a: i32, // the first
      b: i32 /* the second */, c: i32) -> i32 { return a; }
",
        "fn f(
    a: i32, // the first
    b: i32, /* the second */
    c: i32,
) -> i32 {
    return a;
}
",
    );
}

#[test]
fn comments_in_calls_and_struct_literals_make_them_vertical() {
    golden(
        "fn f() {
    g(a, /* why b */ b, c);
    let p = P { x: 1, // x
        y: 2 };
}
",
        "fn f() {
    g(
        a, /* why b */
        b,
        c,
    );
    let p = P {
        x: 1, // x
        y: 2,
    };
}
",
    );
}

// ---- Layout ----

#[test]
fn crlf_and_a_byte_order_mark_are_kept() {
    golden(
        "\u{FEFF}// c\r\nfn f() { let a = 1; }\r\n",
        "\u{FEFF}// c\r\nfn f() {\r\n    let a = 1;\r\n}\r\n",
    );
    golden(
        "/* a\r\n   b */\r\nfn g() {}\r\n",
        "/* a\r\n   b */\r\nfn g() {}\r\n",
    );
    // A file whose first line break is `\n` is written with `\n`.
    golden("fn h() {}\n", "fn h() {}\n");
    golden("fn h() {}", "fn h() {}\n");
    golden("", "");
}

#[test]
fn signatures_calls_struct_literals_and_arrays_wrap_at_max_width() {
    golden_w(
        "fn long_name(alpha: f32, beta: f32, gamma: f32) -> f32 {
    let r = compute(alpha, beta, gamma, alpha);
    let p = Point { x: alpha, y: beta, z: gamma };
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let b = [f(1), f(2), f(3), f(4), f(5), f(6)];
    let c = [1, 2];
    return r;
}
",
        "fn long_name(
    alpha: f32,
    beta: f32,
    gamma: f32,
) -> f32 {
    let r = compute(
        alpha,
        beta,
        gamma,
        alpha,
    );
    let p = Point {
        x: alpha,
        y: beta,
        z: gamma,
    };
    let a = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12,
    ];
    let b = [
        f(1),
        f(2),
        f(3),
        f(4),
        f(5),
        f(6),
    ];
    let c = [1, 2];
    return r;
}
",
        40,
    );
}

// ---- Refusals ----

#[test]
fn text_the_ast_drops_is_refused_not_lost() {
    for (src, why) in [
        ("struct S {\n    @align(16) x: f32,\n}\n", "self-check (d)"),
        // The repr type also takes a node id, so the enum is caught by (b) first.
        ("enum E: u8 {\n    A,\n    B,\n}\n", "self-check (b)"),
        ("trait T {\n    @inline fn m(self);\n}\n", "self-check (d)"),
        // The printer walks an extern body the way the parser does and refuses first.
        (
            "extern {\n    junk fn f();\n}\n",
            "`junk` inside an `extern` block",
        ),
    ] {
        match fmt(src) {
            Err(FmtError::Unsupported(m)) => assert!(m.contains(why), "{src}: {m}"),
            other => panic!("{src}: expected Unsupported, got {other:?}"),
        }
    }
}

/// Text the AST drops that holds no identifier or literal. A bare `@extern` is an `@` and a
/// keyword. The parser eats a `pub` before `import` or `impl` and discards it. `parse_extern`
/// skips keyword or punctuation junk without a diagnostic. None of these takes a node id, so only
/// a keyword-aware law (d), or the printer, can see them. Each must be refused, never deleted.
#[test]
fn keyword_and_punctuation_text_the_ast_drops_is_refused_not_lost() {
    for (src, why) in [
        ("struct S {\n    @extern x: f32,\n}\n", "self-check (d)"),
        ("trait T {\n    @extern fn m(self);\n}\n", "self-check (d)"),
        ("pub import a.b;\n", "`pub import`"),
        ("@inline pub import a.b;\n", "`pub import`"),
        ("pub impl V {}\n", "`pub impl`"),
        (
            "extern {\n    ; fn f();\n}\n",
            "`;` inside an `extern` block",
        ),
        (
            "extern {\n    fn f();;\n}\n",
            "`;` inside an `extern` block",
        ),
        (
            "extern {\n    fn f() {} ;\n}\n",
            "`;` inside an `extern` block",
        ),
        (
            "extern {\n    fn f(); struct\n}\n",
            "`struct` inside an `extern` block",
        ),
        (
            "extern {\n    @extern fn f();\n}\n",
            "`@` inside an `extern` block",
        ),
        (
            "pub extern {\n    pub pub fn f();\n}\n",
            "`pub` inside an `extern` block",
        ),
    ] {
        match fmt(src) {
            Err(FmtError::Unsupported(m)) => assert!(m.contains(why), "{src}: {m}"),
            other => panic!("{src}: expected Unsupported, got {other:?}"),
        }
    }
}

/// What the refusals above must not catch: every declaration shape `parse_extern` accepts,
/// including a `where` clause, a trailing `,` after it, an `= e` body and a block body.
#[test]
fn extern_declarations_of_every_shape_are_covered() {
    golden(
        "extern {
    fn a<T: A>(x: T) -> T where T: B, N > 0,;
    fn b() where;
    fn c() -> i32 = (1);
    fn d() = 2
    fn e() {}
    fn g(x: (i32),) -> (f32)
}
",
        "extern {
    fn a<T: A>(x: T) -> T where T: B, N > 0;
    fn b() where;
    fn c() -> i32 = 1;
    fn d() = 2;
    fn e() {}
    fn g(x: (i32)) -> (f32);
}
",
    );
}

/// A `pub` written on a declaration inside a `pub extern { .. }` is kept as written: the block's
/// own `pub` already exports every declaration, so the AST cannot tell the two apart, and dropping
/// one would delete source text.
#[test]
fn a_pub_written_inside_a_pub_extern_block_is_kept() {
    fixed("pub extern {\n    pub fn f();\n    fn g();\n}\n");
}

#[test]
fn a_parse_error_is_a_syntax_error() {
    match fmt("fn f( {\n") {
        Err(FmtError::Syntax(d)) => assert!(d.iter().all(|x| x.is_error()) && !d.is_empty()),
        other => panic!("expected Syntax, got {other:?}"),
    }
    match fmt("fn f() { let s = \"open; }\n") {
        Err(FmtError::Syntax(d)) => assert_eq!(d[0].code, Some("E0102")),
        other => panic!("expected a lexer Syntax error, got {other:?}"),
    }
}

// ---- The self-check has teeth ----

/// Feed the self-check outputs that are wrong in each of the four ways and confirm each law
/// fires. Without this, a check that compared nothing would pass every golden above.
#[test]
fn each_self_check_law_rejects_a_wrong_output() {
    use crate::check::{self_check, Original};
    let src = "// c\nfn f(a: (i32), b: f32x8) -> i32 {\n    return a;\n}\n";
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
    let mut check = |out: &str| self_check(&orig, &mut interner, out);
    assert_eq!(check(src), Ok(()));
    // (a) a statement whose `;` is missing is a W0030 lint.
    let e =
        check("// c\nfn f(a: (i32), b: f32x8) -> i32 {\n    return a\n    1;\n}\n").unwrap_err();
    assert!(e.contains("self-check (a)"), "{e}");
    // (b) dropping the type's parentheses renumbers every later node.
    let e = check("// c\nfn f(a: i32, b: f32x8) -> i32 {\n    return a;\n}\n").unwrap_err();
    assert!(e.contains("self-check (b)"), "{e}");
    // (c) a lost comment.
    let e = check("fn f(a: (i32), b: f32x8) -> i32 {\n    return a;\n}\n").unwrap_err();
    assert!(e.contains("self-check (c)"), "{e}");
    // (d) `vec[f32, 8]` is the same AST as `f32x8`, but not the same source tokens.
    let e = check("// c\nfn f(a: (i32), b: vec[f32, 8]) -> i32 {\n    return a;\n}\n").unwrap_err();
    assert!(e.contains("self-check (d)"), "{e}");
}

/// Law (d) counts keywords and `@`, not only identifiers and literals: each output below re-parses
/// to the same AST as its source, so only (d) can see the text it lost, and it must name it.
#[test]
fn law_d_sees_a_lost_keyword_or_at_sign() {
    use crate::check::{self_check, Original};
    for (src, out, lost) in [
        // Both tokens are lost; (d) names the first in sorted order.
        (
            "struct S {\n    @extern x: f32,\n}\n",
            "struct S {\n    x: f32,\n}\n",
            "extern",
        ),
        ("pub import a.b;\n", "import a.b;\n", "pub"),
        ("pub impl V {}\n", "impl V {}\n", "pub"),
        ("fn f() where {}\n", "fn f() {}\n", "where"),
        (
            "pub extern {\n    pub fn f();\n}\n",
            "pub extern {\n    fn f();\n}\n",
            "pub",
        ),
    ] {
        let (tokens, comments, _) = wukong_lexer::tokenize_with_comments(src, SourceId(0));
        let mut interner = wukong_span::Interner::new();
        let parsed = wukong_parser::parse_module_tokens_with_lints(&tokens, src, &mut interner, 0);
        assert!(parsed.diags.is_empty(), "{src}: {:?}", parsed.diags);
        let orig = Original {
            src,
            tokens: &tokens,
            comments: &comments,
            module: &parsed.module,
            source: SourceId(0),
        };
        assert_eq!(self_check(&orig, &mut interner, src), Ok(()), "{src}");
        let e = self_check(&orig, &mut interner, out).unwrap_err();
        assert!(e.contains("self-check (d)"), "{src}: {e}");
        assert!(e.contains(&format!("`{lost}`")), "{src}: {e}");
    }
}
