//! `str`: an immutable UTF-8 byte string.
//!
//! A `str` is a 16-byte fat pointer `(data, byte length)` — exactly the layout of a `[]u8` slice
//! (`wukong_types::Ty::size_of`). It is a distinct type only inside this crate, which is the single
//! authority that its bytes are never written:
//!
//! * no conversion exists between `str` and `[]u8` or `*u8`, in either direction (`E0401`; an `as`
//!   cast is rejected by `cast_is_valid`);
//! * an element store `s[i] = b`, a compound one, and `&mut s[i]` are `E0304`
//!   ([`Sema::check_str_write`]);
//! * a `str` comes only from a string literal (a `.rodata` blob) or from slicing a `str`, so there
//!   is no writable storage behind one to begin with.
//!
//! That is Zig's `[]const u8`, with the `const` enforced at the type level. A string literal stays
//! `*u8` (NUL-terminated, the C/printf/file-path type) unless the context asks for a `str`: it
//! adapts to `str` exactly where a numeric literal adapts to its annotation (`literal_adapts`), as
//! an `==`/`!=` operand against a `str`, and as the base of a range index.
//!
//! **Erasure at the boundary.** [`erase_published`] maps every type sema publishes through
//! [`Ty::erase_str`], so `wukong_mir_build`, the optimizer and every backend see only the existing,
//! gated `[]u8` machinery: indexing with its bounds check, `.len()`, `for b in s`, passing and
//! returning a fat pointer. The one thing a slice cannot carry is that its bytes are TEXT, which
//! matters only to printing: [`SemaResult::str_print_args`](crate::SemaResult) names every print
//! or format argument that is a `str`, and `mir_build` prints those as text instead of as a list of
//! bytes. Because `str` is gone before monomorphization, a type parameter cannot be bound to it
//! (`E0420`).
//!
//! **Range indexing.** `s[a..b]` is parsed as the call `$slice(s, a, b)` (`$slice_from(s, a)` with
//! no upper bound; a missing lower bound is `0`) — a name no source identifier can spell. It is
//! typed here ([`Sema::type_range_slice`]): a `str` base gives a `str`, a `[]T` base a `[]T`.

use wukong_ast::{BinOp, Expr, ExprKind, Path};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashMap as HashMap, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{DefKind, Sema, SemaResult, VariantPayload};

/// The callee a bounded range index `s[a..b]` desugars to: `$slice(s, a, b)`.
pub const RANGE_SLICE: &str = "$slice";
/// The callee an open-ended range index `s[a..]` desugars to: `$slice_from(s, a)`.
pub const RANGE_SLICE_FROM: &str = "$slice_from";

/// Whether `name` is one of the two range-index callees.
pub fn is_range_slice(name: &str) -> bool {
    name == RANGE_SLICE || name == RANGE_SLICE_FROM
}

/// Map every type in a published [`SemaResult`] through [`Ty::erase_str`]: the expression types
/// and every definition's signature, constant type, struct fields and enum payloads.
pub(crate) fn erase_published(r: &mut SemaResult) {
    fn erase(t: &mut Ty) {
        if t.mentions_str() {
            *t = t.erase_str();
        }
    }
    for t in r.types.values_mut() {
        erase(t);
    }
    for def in &mut r.defs.defs {
        match &mut def.kind {
            DefKind::Fn(sig) => {
                sig.params.iter_mut().for_each(erase);
                erase(&mut sig.ret);
            }
            DefKind::Const(t) => erase(t),
            DefKind::Struct(fields) => fields.iter_mut().for_each(|(_, t)| erase(t)),
            DefKind::Enum(variants) => {
                for v in variants {
                    match &mut v.payload {
                        VariantPayload::Unit => {}
                        VariantPayload::Tuple(ts) => ts.iter_mut().for_each(erase),
                        VariantPayload::Struct(fs) => fs.iter_mut().for_each(|(_, t)| erase(t)),
                    }
                }
            }
        }
    }
}

impl Sema<'_> {
    /// Whether the type path `p` names the built-in `str` (a single segment; sema reserves the
    /// name, so nothing the program declares can shadow it).
    pub(crate) fn names_str(&self, p: &Path) -> bool {
        p.is_single() && self.sym_str(p.first().sym) == "str"
    }

    /// `E0300`: a struct, enum, trait or generic parameter may not be named `str`, the built-in
    /// type — the name is how every type position, including `mir_build`'s annotation readers,
    /// recognizes it.
    pub(crate) fn reject_str_decl(&mut self, name: Symbol, span: Span, what: &str) {
        if self.sym_str(name) != "str" {
            return;
        }
        self.diags.push(
            Diagnostic::error(format!("a {what} cannot be named `str`"))
                .with_code("E0300")
                .primary(span, "`str` is a built-in type")
                .help("choose another name"),
        );
    }

    /// `E0304` when the place `target` (of an assignment, a compound assignment, or `&mut`) is a
    /// byte of a `str`: a `str` views a string literal's read-only storage, or a slice of one.
    pub(crate) fn check_str_write(&mut self, target: &Expr) {
        let ExprKind::Index { base, .. } = &target.kind else {
            return;
        };
        if !matches!(self.types.get(&base.id), Some(Ty::Str)) {
            return;
        }
        self.diags.push(
            Diagnostic::error("a `str` is immutable; its bytes cannot be assigned")
                .with_code("E0304")
                .primary(target.span, "this would write into a `str`")
                .help(
                    "build the text in a `[u8; N]` buffer (or a `[]u8` from `alloc_u8`) and write \
                     into that instead",
                ),
        );
    }

    /// An `==`/`!=` operand that is a string literal, compared against a `str`, is that `str`
    /// (`s == "abc"`). Returns the two operand types after the adaptation.
    pub(crate) fn adapt_str_eq_literals(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        l: Ty,
        r: Ty,
    ) -> (Ty, Ty) {
        if !matches!(op, BinOp::Eq | BinOp::Ne) {
            return (l, r);
        }
        let (mut l, mut r) = (l, r);
        if l == Ty::Str && r != Ty::Str && self.literal_adapts(&Ty::Str, rhs) {
            self.retype_adapted_literal(rhs, &Ty::Str);
            r = Ty::Str;
        }
        if r == Ty::Str && l != Ty::Str && self.literal_adapts(&Ty::Str, lhs) {
            self.retype_adapted_literal(lhs, &Ty::Str);
            l = Ty::Str;
        }
        (l, r)
    }

    /// The `==`/`!=` rule for `str` operands: two `str`s compare by length and then bytes; a `str`
    /// against anything else is `E0433`. Returns whether either operand is a `str` (the comparison
    /// is then decided here, and the aggregate rule that rejects every other `==` on a fat pointer
    /// does not apply).
    pub(crate) fn check_str_eq(&mut self, l: &Ty, r: &Ty, span: Span) -> bool {
        match (l, r) {
            (Ty::Str, Ty::Str) => true,
            (Ty::Str, other) | (other, Ty::Str) => {
                if !other.is_unknown() && !other.is_error() {
                    let shown = other.display(self.interner);
                    self.diags.push(
                        Diagnostic::error(format!(
                            "`==`/`!=` is not defined between a `str` and a `{shown}`"
                        ))
                        .with_code("E0433")
                        .primary(span, "a `str` compares only with another `str`")
                        .help("compare two `str`s; a string literal operand adapts to `str`"),
                    );
                }
                true
            }
            _ => false,
        }
    }

    /// Whether the concrete types `place` and `value` disagree on being a `str` — the type half of
    /// the rule that `str` converts to and from nothing. `None` when neither is a `str` (the caller's
    /// own rules apply). A generic type variable, `Unknown` and `Error` stay lenient.
    pub(crate) fn str_type_clash(&self, place: &Ty, value: &Ty) -> Option<bool> {
        let (p, v) = (matches!(place, Ty::Str), matches!(value, Ty::Str));
        if !p && !v {
            return None;
        }
        if p && v {
            return Some(false);
        }
        let other = if p { value } else { place };
        Some(!self.is_generic_ty(other) && !other.is_unknown() && !other.is_error())
    }

    /// `E0401` for a struct field or enum payload value that is not a `str` where one is declared,
    /// or a `str` where one is not (the field checks compare only scalar types otherwise). An
    /// adapting string literal is accepted.
    pub(crate) fn check_str_field_value(&mut self, fty: &Ty, value: &Expr) {
        let vty = self.types.get(&value.id).cloned().unwrap_or(Ty::Unknown);
        if self.str_type_clash(fty, &vty) != Some(true) || self.literal_adapts(fty, value) {
            return;
        }
        let (want, found) = (fty.display(self.interner), vty.display(self.interner));
        self.error(
            value.span,
            "E0401",
            format!(
                "type mismatch: this field has type `{want}`, but a value of type `{found}` is \
                 given (`str` converts to and from nothing; a string literal adapts to `str`)"
            ),
        );
    }

    /// The element type of the array literal `items`, whose elements typed as `tys`, with every
    /// element checked against it.
    ///
    /// `mir_build` stores each element at its OWN type (`lower_array_init` -> `init_field`) into a
    /// slot sized by the array's, so an element whose type is not the element type is a layout lie
    /// rather than a conversion. The element type used to be the first element's, with the rest
    /// never compared, and a `str` made that observable: `[s, "y"]` was a `[str; 2]` whose `"y"`
    /// stayed `*u8`, stored as one 8-byte pointer into a 16-byte slot whose length nobody wrote
    /// (stack garbage on native, an uninitialized slot on the interpreter); `["x", s]` stored the
    /// ADDRESS of `s`'s fat pointer as a C string; and `[b, s]`, `b: []u8`, handed a writable view
    /// of a string literal's deduplicated `.rodata` bytes to `v[1][0] = 65`.
    ///
    /// The element type is the first element type every element holds — as is, or as an adapting
    /// literal (`literal_adapts`), so a string literal becomes a `str` on either side of one, and
    /// `[3, 4]` beside an `[i8; 2]` is an `[i8; 2]`. An adapted element is re-stamped (and
    /// range-checked) at that type; any other element that clashes with it is `E0401`. "Clashes"
    /// is `aggregate_type_clash`, the place-versus-value relation `return` and `=` use with the
    /// array's element type as the place: it covers `str` against anything else and two concrete
    /// aggregates of different layouts, and leaves scalars, tensors, generics and `Unknown` to the
    /// rules that already apply to them.
    pub(crate) fn unify_array_lit_elems(&mut self, items: &[Expr], tys: &[Ty]) -> Ty {
        let holds = |this: &Self, elem: &Ty| {
            items
                .iter()
                .zip(tys)
                .all(|(it, t)| !this.aggregate_type_clash(elem, t) || this.literal_adapts(elem, it))
        };
        let first = tys
            .iter()
            .find(|t| !t.is_unknown())
            .cloned()
            .unwrap_or(Ty::Unknown);
        let elem = if holds(self, &first) {
            first
        } else {
            tys.iter()
                .find(|t| !t.is_unknown() && holds(self, t))
                .cloned()
                .unwrap_or(first)
        };
        for (it, t) in items.iter().zip(tys) {
            if !self.aggregate_type_clash(&elem, t) {
                continue;
            }
            if self.literal_adapts(&elem, it) {
                self.retype_adapted_literal(it, &elem);
                self.range_check_int_literal(it, &elem);
                continue;
            }
            let (want, found) = (elem.display(self.interner), t.display(self.interner));
            self.error(
                it.span,
                "E0401",
                format!(
                    "type mismatch: this array element is a `{found}`, but the array's elements are \
                     `{want}` (every element of an array literal has the one element type; `str` \
                     converts to and from nothing, and a string literal adapts to `str`)"
                ),
            );
        }
        elem
    }

    /// `E0420`: a type parameter bound to a type that mentions `str`. `str` is erased to its
    /// `[]u8` representation before monomorphization, so an instance could not tell its text from
    /// a byte slice. The binding is replaced by `Ty::Error` so nothing cascades from the call.
    pub(crate) fn reject_str_type_args(
        &mut self,
        generics: &[Symbol],
        tys: &mut HashMap<Symbol, Ty>,
        span: Span,
    ) {
        let mut reported = false;
        for g in generics {
            let Some(t) = tys.get(g) else { continue };
            if !t.mentions_str() {
                continue;
            }
            if !reported {
                let (gname, shown) = (self.sym_str(*g).to_string(), t.display(self.interner));
                self.diags.push(
                    Diagnostic::error(format!(
                        "`str` cannot instantiate a type parameter: `{gname}` would be `{shown}`"
                    ))
                    .with_code("E0420")
                    .primary(span, "this call binds a type parameter to a `str` type")
                    .help(
                        "`str` is checked only outside generic code; write the function for \
                         `str` itself (a non-generic `fn f(s: str)`)",
                    ),
                );
                reported = true;
            }
            tys.insert(*g, Ty::Error);
        }
    }

    /// Type a range index — `$slice(base, lo, hi)` / `$slice_from(base, lo)` (see the module docs) —
    /// or return `None` when `name` is neither. A `str` base (a string literal adapts) gives a
    /// `str`, a `[]T` base a `[]T`; any other base is `E0437`. The bounds are integers (an
    /// unsuffixed literal adapts to `i64`), else `E0446`. The bounds `lo <= hi <= len` are checked
    /// at run time by the lowering.
    pub(crate) fn type_range_slice(
        &mut self,
        name: Symbol,
        callee: &Expr,
        args: &[Expr],
        arg_tys: &mut [Ty],
    ) -> Option<Ty> {
        if !is_range_slice(self.sym_str(name)) || args.len() < 2 {
            return None;
        }
        self.types.insert(callee.id, Ty::Unknown);
        let base = &args[0];
        if self.literal_adapts(&Ty::Str, base) {
            self.retype_adapted_literal(base, &Ty::Str);
            arg_tys[0] = Ty::Str;
        }
        let i64_ty = Ty::Scalar(Scalar::I64);
        for (bound, ty) in args[1..].iter().zip(arg_tys[1..].iter_mut()) {
            if self.literal_adapts(&i64_ty, bound) {
                self.retype_adapted_literal(bound, &i64_ty);
                self.range_check_int_literal(bound, &i64_ty);
                *ty = i64_ty.clone();
            }
            match &*ty {
                Ty::Scalar(s) if s.is_int() && *s != Scalar::Bool && *s != Scalar::Char => {}
                Ty::Unknown | Ty::Error => {}
                other => {
                    let shown = other.display(self.interner);
                    self.error(
                        bound.span,
                        "E0446",
                        format!("a range index bound must be an integer, found `{shown}`"),
                    );
                }
            }
        }
        Some(match &arg_tys[0] {
            Ty::Str => Ty::Str,
            t @ Ty::Slice(_) => t.clone(),
            Ty::Unknown | Ty::Error => Ty::Unknown,
            other => {
                let shown = other.display(self.interner);
                self.error(
                    base.span,
                    "E0437",
                    format!(
                        "only a `str` or a slice `[]T` can be sliced with `[a..b]`, not a value of \
                         type `{shown}`; bind an array to a `[]T` first"
                    ),
                );
                Ty::Error
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use wukong_span::{Interner, SourceId};
    use wukong_types::Ty;

    use crate::{DefKind, SemaResult, VariantPayload};

    fn check(src: &str) -> (SemaResult, Vec<&'static str>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (res, diags) = crate::check(&module, &interner);
        let codes = diags.iter().filter_map(|d| d.code).collect();
        (res, codes, interner)
    }

    #[track_caller]
    fn accepts(src: &str) {
        let (_, codes, _) = check(src);
        assert!(codes.is_empty(), "unexpected {codes:?} for {src}");
    }

    #[track_caller]
    fn rejects(src: &str, code: &str) {
        let (_, codes, _) = check(src);
        assert!(
            codes.contains(&code),
            "expected {code}, got {codes:?} for {src}"
        );
    }

    const STR_HEAVY: &str = "
const GREETING: str = \"hello\";
struct User { name: str, id: i32 }
enum Tag { Named(str), Code { text: str }, None }
fn first(s: str) -> u8 { return s[0]; }
fn pick(b: bool) -> str { if b { return \"yes\"; } \"no\" }
fn apply(f: fn(str) -> u8, s: str) -> u8 { return f(s); }
fn main() -> i32 {
    let s: str = \"wukong\";
    let mid = s[1..4];
    let all = s[..];
    let names: [str; 2] = [\"a\", \"bb\"];
    let pair: (str, i32) = (\"x\", 1);
    let u = User { name: \"ada\", id: 1 };
    let t = Tag::Named(\"t\");
    let c = Tag::Code { text: \"c\" };
    let p: *str = &s;
    let mut n: i64 = s.len() + mid.len() + all.len() + names[1].len() + pair.0.len();
    for b in s { n = n + (b as i64); }
    if s == \"wukong\" && mid != \"uk\" && GREETING == pick(true) { n = n + 1; }
    println(s);
    printf(\"{} {:?}\\n\", u.name, s[0..2]);
    let _ = (t, c, p, first(s), apply(first, s));
    return n as i32;
}
";

    #[test]
    fn str_programs_check_and_erase_completely() {
        let (res, codes, interner) = check(STR_HEAVY);
        assert!(codes.is_empty(), "unexpected {codes:?}");
        // No `str` survives publication: every expression type and every definition is erased.
        for (id, t) in &res.types {
            assert!(!t.mentions_str(), "node {id:?} still typed {t:?}");
        }
        for def in &res.defs.defs {
            let tys: Vec<Ty> = match &def.kind {
                DefKind::Fn(sig) => sig.params.iter().chain([&sig.ret]).cloned().collect(),
                DefKind::Const(t) => vec![t.clone()],
                DefKind::Struct(fs) => fs.iter().map(|(_, t)| t.clone()).collect(),
                DefKind::Enum(vs) => vs.iter().flat_map(|v| v.payload.field_tys()).collect(),
            };
            for t in tys {
                assert!(
                    !t.mentions_str(),
                    "`{}` still mentions str: {t:?}",
                    interner.resolve(def.name)
                );
            }
        }
        // The `str` field erased to the slice it is laid out as.
        let user = res.defs.lookup(interner.get("User").unwrap()).unwrap();
        let DefKind::Struct(fields) = &user.kind else {
            panic!("User is a struct")
        };
        assert_eq!(
            fields[0].1,
            Ty::Slice(Box::new(Ty::Scalar(wukong_types::Scalar::U8)))
        );
        let tag = res.defs.lookup(interner.get("Tag").unwrap()).unwrap();
        let DefKind::Enum(vs) = &tag.kind else {
            panic!("Tag is an enum")
        };
        assert!(matches!(&vs[0].payload, VariantPayload::Tuple(ts) if !ts[0].mentions_str()));
        // Both printed `str`s, and both format arguments, are recorded as text.
        assert_eq!(res.str_print_args.len(), 3, "{:?}", res.str_print_args);
    }

    #[test]
    fn str_literals_adapt_where_numeric_literals_do() {
        accepts(
            "fn f(s: str) -> i64 { return s.len(); } fn main() -> i32 { return f(\"ab\") as i32; }",
        );
        accepts("struct S { t: str } fn main() -> i32 { let mut s = S { t: \"a\" }; s.t = \"b\"; return s.t.len() as i32; }");
        accepts("fn g() -> str { \"tail\" } fn main() -> i32 { let mut x: str = g(); x = \"y\"; return x.len() as i32; }");
        accepts("fn main() -> i32 { let a: [str; 2] = [\"a\", \"b\"]; let t: (str, str) = (\"a\", \"b\"); return (a[0] == t.1) as i32; }");
        accepts("fn main() -> i32 { let s: str = \"abc\"; return (\"abc\" == s) as i32; }");
        accepts("fn main() -> i32 { let n = \"abcdef\"[1..3].len(); return n as i32; }");
        // Unannotated, a literal is still the NUL-terminated `*u8`.
        accepts("fn main() -> i32 { let p = \"x\"; let q: *u8 = p; print(q); return 0; }");
    }

    #[test]
    fn str_converts_to_and_from_nothing() {
        rejects(
            "fn main() -> i32 { let p = \"x\"; let s: str = p; return 0; }",
            "E0401",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"x\"; let p: *u8 = s; return 0; }",
            "E0401",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"x\"; let b: []u8 = s; return 0; }",
            "E0401",
        );
        rejects(
            "fn f(s: str) {} fn main() -> i32 { let p = \"x\"; f(p); return 0; }",
            "E0401",
        );
        rejects(
            "fn f(b: []u8) {} fn main() -> i32 { let s: str = \"x\"; f(s); return 0; }",
            "E0401",
        );
        rejects(
            "fn f(p: *u8) -> str { return p; } fn main() -> i32 { return 0; }",
            "E0401",
        );
        rejects(
            "fn main() -> i32 { let p = \"x\"; let mut s: str = \"y\"; s = p; return 0; }",
            "E0401",
        );
        rejects(
            "struct S { t: str } fn main() -> i32 { let p = \"x\"; let s = S { t: p }; return 0; }",
            "E0401",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"x\"; let n = s as i64; return 0; }",
            "E0401",
        );
    }

    #[test]
    fn str_bytes_are_immutable() {
        rejects(
            "fn main() -> i32 { let mut s: str = \"abc\"; s[0] = 65; return 0; }",
            "E0304",
        );
        rejects(
            "fn main() -> i32 { let mut s: str = \"abc\"; s[0] += 1; return 0; }",
            "E0304",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"abc\"; let p = &mut s[1]; return 0; }",
            "E0304",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"abc\"; s[0..2][1] = 7; return 0; }",
            "E0304",
        );
        rejects(
            "fn f(mut s: str) { s[0] = 1; } fn main() -> i32 { return 0; }",
            "E0304",
        );
        // Re-pointing a `mut` binding is not a write into the bytes.
        accepts(
            "fn main() -> i32 { let mut s: str = \"abc\"; s = s[1..]; return s.len() as i32; }",
        );
    }

    #[test]
    fn str_operators_and_indexing() {
        accepts("fn main() -> i32 { let s: str = \"ab\"; let b: u8 = s[1]; return b as i32; }");
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; let t: str = \"cd\"; return (s < t) as i32; }",
            "E0433",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; let t = -s; return 0; }",
            "E0433",
        );
        rejects("fn main() -> i32 { let s: str = \"ab\"; let t: str = \"cd\"; let u = s + t; return 0; }", "E0433");
        rejects(
            "fn main() -> i32 { let mut s: str = \"ab\"; let t: str = \"cd\"; s += t; return 0; }",
            "E0433",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; let p = \"ab\"; return (s == p) as i32; }",
            "E0433",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; return s[0, 1] as i32; }",
            "E0501",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; return s.size() as i32; }",
            "E0322",
        );
    }

    #[test]
    fn range_indexes() {
        accepts("fn main() -> i32 { let a: [i32; 4] = [1, 2, 3, 4]; let s: []i32 = a; let t: []i32 = s[1..3]; return t[0]; }");
        accepts("fn main() -> i32 { let s: str = \"abcd\"; let n: u32 = 1; let t: str = s[n..]; return t.len() as i32; }");
        rejects(
            "fn main() -> i32 { let a: [i32; 4] = [1, 2, 3, 4]; let t = a[1..3]; return 0; }",
            "E0437",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"abcd\"; let t = s[1.5..2]; return 0; }",
            "E0446",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"abcd\"; let t = s[true..]; return 0; }",
            "E0446",
        );
    }

    #[test]
    fn str_is_not_generic_and_its_name_is_reserved() {
        rejects("fn id<T>(x: T) -> T { return x; } fn main() -> i32 { let s: str = \"a\"; let t = id(s); return 0; }", "E0420");
        rejects("fn id<T>(x: T) -> T { return x; } fn main() -> i32 { let s: str = \"a\"; let t = id::<str>(s); return 0; }", "E0420");
        rejects("fn fst<T>(x: (T, i32)) -> i32 { return x.1; } fn main() -> i32 { let p: (str, i32) = (\"a\", 1); return fst(p); }", "E0420");
        rejects(
            "struct str { x: i32 } fn main() -> i32 { return 0; }",
            "E0300",
        );
        rejects("enum str { A } fn main() -> i32 { return 0; }", "E0300");
        rejects(
            "trait str { fn f(self) -> i32; } fn main() -> i32 { return 0; }",
            "E0300",
        );
        rejects(
            "fn f<str>(x: i32) -> i32 { return x; } fn main() -> i32 { return 0; }",
            "E0300",
        );
    }

    /// Whether any published type of `src` (which must check clean) still mentions `*u8`. In these
    /// programs every `*u8` would be a string literal that never became a `str`.
    #[track_caller]
    fn literal_left_as_ptr(src: &str) -> bool {
        let (res, codes, interner) = check(src);
        assert!(codes.is_empty(), "unexpected {codes:?} for {src}");
        res.types
            .values()
            .any(|t| t.display(&interner).contains("*u8"))
    }

    /// An array literal holds every element at ONE element type, and `mir_build` stores each
    /// element at its own (`lower_array_init` -> `init_field`). A string literal beside a `str`
    /// must therefore become a `str` wherever it sits in the literal, and anything else that is
    /// not the element type is `E0401`, never a 16-byte slot holding an 8-byte pointer.
    #[test]
    fn array_literal_elements_share_one_type() {
        let s = "let s: str = \"x\";";
        for body in [
            "let a = [s, \"y\"]; return a[1].len() as i32;",
            "let a = [\"y\", s]; return a[0].len() as i32;",
            "let a: [str; 2] = [s, \"y\"]; return a[1].len() as i32;",
            "let a = [[s], [\"y\"]]; return a[1][0].len() as i32;",
        ] {
            let src = format!("fn main() -> i32 {{ {s} {body} }}");
            assert!(
                !literal_left_as_ptr(&src),
                "a literal stayed `*u8` in {src}"
            );
        }
        assert!(!literal_left_as_ptr(&format!(
            "fn f(w: [str; 2]) -> i64 {{ return w[1].len(); }} \
             fn main() -> i32 {{ {s} return f([s, \"y\"]) as i32; }}"
        )));
        for (body, why) in [
            ("let p = \"y\"; let a = [s, p];", "a `*u8` value"),
            ("let p = \"y\"; let a = [p, s];", "a `*u8` value first"),
            ("let a = [s, 1];", "an integer"),
            (
                "let mut buf: [u8; 3] = [1, 2, 3]; let b: []u8 = buf; let a = [b, s];",
                "a `[]u8` view, which would make the literal's bytes writable",
            ),
            (
                "let mut buf: [u8; 3] = [1, 2, 3]; let b: []u8 = buf; let a = [s, b];",
                "a `[]u8` view second",
            ),
        ] {
            let src = format!("fn main() -> i32 {{ {s} {body} return 0; }}");
            let (_, codes, _) = check(&src);
            assert_eq!(codes, vec!["E0401"], "{why}: {src}");
        }
        // The rule is the aggregate one, not only `str`'s: two different structs cannot share a
        // slot, and an adapting numeric literal takes an established element width.
        rejects(
            "struct A { x: i32 } struct B { y: f32 } \
             fn main() -> i32 { let a = A { x: 1 }; let b = B { y: 2.0 }; let v = [a, b]; return 0; }",
            "E0401",
        );
        let (res, codes, interner) = check(
            "fn main() -> i32 { let n: [i8; 2] = [1, 2]; let v = [n, [3, 4]]; return v[1][0] as i32; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert!(
            res.types
                .values()
                .all(|t| !t.display(&interner).contains("i32; 2")),
            "the `[3, 4]` element must take the `[i8; 2]` element type"
        );
    }

    #[test]
    fn str_prints_at_top_level_only() {
        accepts("fn main() -> i32 { let s: str = \"a\"; print(s); println(s); print_raw(s); eprintln(s); printf(\"{}{:?}\", s, s); return 0; }");
        rejects("struct S { t: str } fn main() -> i32 { let s = S { t: \"a\" }; println(s); return 0; }", "E0448");
        rejects(
            "fn main() -> i32 { let a: [str; 1] = [\"a\"]; println(a); return 0; }",
            "E0448",
        );
    }
}
