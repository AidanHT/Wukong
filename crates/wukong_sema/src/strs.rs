//! `str`: an immutable UTF-8 byte string.
//!
//! A `str` is a 16-byte fat pointer `(data, byte length)` — exactly the layout of a `[]u8` slice
//! (`wukong_types::Ty::size_of`) — that VIEWS bytes it does not own. It is a distinct type only
//! inside this crate, which is the single authority that nothing writes through one:
//!
//! * nothing converts implicitly between `str` and `[]u8` or `*u8`, in either direction (`E0401`;
//!   an `as` cast is rejected by `cast_is_valid`); the explicit conversions are named calls —
//!   `s.as_bytes()`, `s.as_ptr()` and the run-time-checked `str_from_utf8(b)` ([`StrOp`]);
//! * an element store `s[i] = b`, a compound one, `atomic_add(s[i], v)` and `&mut s[i]` are
//!   `E0304` ([`Sema::check_str_write`], called from `check_projection_write` and `&mut`).
//!
//! That is Zig's `[]const u8`, with the `const` enforced at the type level. A `str` comes from a
//! string literal (a `.rodata` blob), from slicing a `str`, from `str_from_utf8` (a view of the
//! caller's bytes) or from `str_concat` / `str_format` (a fresh heap allocation). A string literal
//! stays `*u8` (NUL-terminated, the C/printf/file-path type) unless the context asks for a `str`:
//! it adapts to `str` exactly where a numeric literal adapts to its annotation (`literal_adapts`,
//! which also takes an `if`/`match` of literal arms), as a comparison operand against a `str`, as
//! the base of a range index, as a `str` method's receiver, and as a scrutinee tested against
//! string-literal patterns.
//!
//! **The operations** — methods (`type_str_method`), builtins (`type_str_builtin`), ordering
//! (`check_str_eq`), string-literal patterns (`check_str_literal_pattern`; usefulness gives `str`
//! one constructor per distinct literal) and `str_format` (`text.rs`) — are typed here and recorded
//! in [`StrInfo`], which `mir_build` lowers them from.
//!
//! **Erasure at the boundary.** [`erase_published`] maps every type sema publishes through
//! [`Ty::erase_str`], so `wukong_mir_build`, the optimizer and every backend see only the existing,
//! gated `[]u8` machinery: indexing with its bounds check, `.len()`, `for b in s`, passing and
//! returning a fat pointer. The one thing a slice cannot carry is that its bytes are TEXT, which
//! matters only to printing: [`SemaResult::str_print_args`](crate::SemaResult) names every print
//! or format argument that is a `str`, [`StrInfo::print_types`] / [`StrInfo::struct_fields`] keep
//! the unerased types of the aggregates that hold one, and `mir_build` prints those as text instead
//! of as a list of bytes. Because `str` is gone before monomorphization, a type parameter cannot be
//! bound to it (`E0420`).
//!
//! **Range indexing.** `s[a..b]` is parsed as the call `$slice(s, a, b)` (`$slice_from(s, a)` with
//! no upper bound; a missing lower bound is `0`) — a name no source identifier can spell. It is
//! typed here ([`Sema::type_range_slice`]): a `str` base gives a `str`, a `[]T` base a `[]T`.

use wukong_ast::{
    BinOp, Block, Expr, ExprKind, ForIter, Ident, MatchArm, NodeId, PatKind, Path, Pattern,
};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{DefKind, Sema, SemaResult, VariantPayload};

/// A `str` operation sema typed, published in [`StrInfo::ops`] keyed by the NodeId of the call's
/// CALLEE (the `s.as_bytes` field for a method, the `str_from_utf8` path for a builtin). `str` is
/// erased to `[]u8` before `wukong_mir_build` sees it, so this tag is how lowering tells
/// `s.as_bytes()` from a call it does not know.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrOp {
    /// `s.as_bytes() -> []u8`: the same fat pointer, viewed as bytes — no copy.
    AsBytes,
    /// `s.as_ptr() -> *u8`: the data pointer (NUL-terminated only when the `str` is a whole
    /// literal or a `str_concat` / `str_format` result).
    AsPtr,
    /// `s.chars()`, legal only as the iterable of a `for` head: each character, as a `char`.
    Chars,
    /// `s.char_count() -> i64`: how many characters `s.chars()` yields.
    CharCount,
    /// `s.cmp(t) -> i32`: `-1`, `0` or `1`, comparing the bytes lexicographically.
    Cmp,
    /// `str_from_utf8(b) -> str`: a view of the bytes `b`, after a run-time UTF-8 check that
    /// stops the program when they are not UTF-8.
    FromUtf8,
    /// `utf8_valid_up_to(b) -> i64`: the length of the longest valid UTF-8 prefix of `b`.
    ValidUpTo,
    /// `str_concat(a, b, …) -> str`: the operands' bytes, in order, in a fresh heap allocation.
    Concat,
}

/// The method names a `str` receiver accepts, sorted (`len` is an extent query, typed with the
/// slices').
pub const STR_METHODS: &[&str] = &["as_bytes", "as_ptr", "char_count", "chars", "cmp", "len"];

/// What sema publishes about `str` beyond the erased types (`SemaResult::str_info`).
#[derive(Clone, Debug, Default)]
pub struct StrInfo {
    /// Every `str` method call and `str` builtin, keyed by its callee's NodeId ([`StrOp`]).
    pub ops: HashMap<NodeId, StrOp>,
    /// The UNERASED type of every print or format argument that holds a `str` inside an
    /// aggregate or behind a reference (`(str, i32)`, `[str; 2]`, `&str`, a struct with a `str`
    /// field). `mir_build` walks it to print those `str`s as text; it must never lower storage
    /// from it, and it does not have to: a `str` is laid out exactly as the `[]u8` it erases to.
    pub print_types: HashMap<NodeId, Ty>,
    /// The UNERASED field types of every struct that has a `str` among them, in declaration
    /// order — what the structural printer reads for a struct field ([`Self::print_types`]).
    pub struct_fields: HashMap<Symbol, Vec<Ty>>,
}

/// The per-run `str` bookkeeping of the checker.
#[derive(Default)]
pub(crate) struct StrState {
    /// Published as [`SemaResult::str_info`](crate::SemaResult).
    pub(crate) info: StrInfo,
    /// The callees of the `s.chars()` calls the `for` heads being typed iterate, so the method
    /// typing can tell them from a misplaced `s.chars()` (`E0425`). A set, because a head's
    /// receiver can itself hold a `for` (`for c in { for d in t.chars() {} s }.chars()`).
    pub(crate) chars_heads: HashSet<NodeId>,
}

/// The callee a bounded range index `s[a..b]` desugars to: `$slice(s, a, b)`.
pub const RANGE_SLICE: &str = "$slice";
/// The callee an open-ended range index `s[a..]` desugars to: `$slice_from(s, a)`.
pub const RANGE_SLICE_FROM: &str = "$slice_from";

/// Whether `name` is one of the two range-index callees.
pub fn is_range_slice(name: &str) -> bool {
    name == RANGE_SLICE || name == RANGE_SLICE_FROM
}

impl SemaResult {
    /// Whether `callee` is the callee of a range index (`$slice` / `$slice_from`), for a pass that
    /// holds no interner. A range index is a VIEW of its base's storage, not a call handing back
    /// storage of unknown origin, and an analysis that roots buffers must say so
    /// (`wukong_mir_build`'s `alias.rs`, like `noalias.rs`'s `view_target` here).
    pub fn is_range_slice_call(&self, callee: &Expr) -> bool {
        matches!(&callee.kind, ExprKind::Path(p)
            if p.is_single() && self.range_slice_callees.contains(&Some(p.first().sym)))
    }

    /// The operand whose bytes a `str` call hands back a VIEW of — the receiver of `s.as_bytes()`,
    /// `s.as_ptr()` and `s.chars()`, the argument of `str_from_utf8(b)` — or `None` for any other
    /// call. Like a range index ([`Self::is_range_slice_call`]), such a result is a place in its
    /// operand's storage, not storage of its own, and an analysis that roots buffers must say so.
    pub fn str_view_operand<'e>(&self, callee: &'e Expr, args: &'e [Expr]) -> Option<&'e Expr> {
        str_view_operand_in(&self.str_info.ops, callee, args)
    }
}

/// [`SemaResult::str_view_operand`] over the operation table `ops`, for the checks that run inside
/// sema (`noalias.rs`).
pub(crate) fn str_view_operand_in<'e>(
    ops: &HashMap<NodeId, StrOp>,
    callee: &'e Expr,
    args: &'e [Expr],
) -> Option<&'e Expr> {
    match ops.get(&callee.id)? {
        StrOp::AsBytes | StrOp::AsPtr | StrOp::Chars => match &callee.kind {
            ExprKind::Field { base, .. } => Some(base),
            _ => None,
        },
        StrOp::FromUtf8 => args.first(),
        StrOp::CharCount | StrOp::Cmp | StrOp::ValidUpTo | StrOp::Concat => None,
    }
}

/// Map every type in a published [`SemaResult`] through [`Ty::erase_str`]: the expression types
/// and every definition's signature, constant type, struct fields and enum payloads. The struct
/// fields are kept unerased in [`StrInfo::struct_fields`] first, for the printer; nothing else in
/// [`StrInfo`] is a storage type.
pub(crate) fn erase_published(r: &mut SemaResult) {
    fn erase(t: &mut Ty) {
        if t.mentions_str() {
            *t = t.erase_str();
        }
    }
    for def in &r.defs.defs {
        if let DefKind::Struct(fields) = &def.kind {
            if fields.iter().any(|(_, t)| t.mentions_str()) {
                let tys = fields.iter().map(|(_, t)| t.clone()).collect();
                r.str_info.struct_fields.insert(def.name, tys);
            }
        }
    }
    for t in r.types.values_mut() {
        erase(t);
    }
    // A closure's recorded types are what `wukong_mir_build` gives its lifted function's signature
    // (`closures.rs`): a `str` parameter or return erases like a named function's.
    for info in r.closures.values_mut() {
        info.params.iter_mut().for_each(erase);
        erase(&mut info.ret);
        info.captures.iter_mut().for_each(|(_, t)| erase(t));
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
    ///
    /// The comparison is on the name the program WROTE: the loader renames an imported module's
    /// items to `<import.path>.<name>` before sema runs (`wukong_driver::loader::resolve`, a
    /// spelling no source identifier can produce), so a `struct str` in `lib/m.wk` arrives as
    /// `lib.m.str`, and compared whole it slipped through and made `str` mean the struct in that
    /// module and through a bare `import lib.m;`.
    pub(crate) fn reject_str_decl(&mut self, name: Symbol, span: Span, what: &str) {
        if self.sym_str(name).rsplit('.').next() != Some("str") {
            return;
        }
        self.diags.push(
            Diagnostic::error(format!("a {what} cannot be named `str`"))
                .with_code("E0300")
                .primary(span, "`str` is a built-in type")
                .help("choose another name"),
        );
    }

    /// `E0304` when the place `target` (of an assignment, a compound assignment, `atomic_add` or
    /// `&mut`) is a byte of a `str`: a `str` views a string literal's read-only storage, or a
    /// slice of one.
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

    /// A comparison operand (`==`, `!=`, `<`, `<=`, `>`, `>=`) that is a string literal, compared
    /// against a `str`, is that `str` (`s == "abc"`, `"m" <= s`). Returns the two operand types
    /// after the adaptation.
    pub(crate) fn adapt_str_eq_literals(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        l: Ty,
        r: Ty,
    ) -> (Ty, Ty) {
        if !matches!(
            op,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        ) {
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

    /// The comparison rule for `str` operands: two `str`s compare by value — `==`/`!=` by length
    /// and then bytes, `<`/`<=`/`>`/`>=` lexicographically by bytes (which is code-point order,
    /// UTF-8 being order-preserving); a `str` against anything else is `E0433`. Returns whether
    /// either operand is a `str` (the comparison is then decided here, and the aggregate rule that
    /// rejects every other comparison of a fat pointer does not apply).
    pub(crate) fn check_str_eq(&mut self, op: BinOp, l: &Ty, r: &Ty, span: Span) -> bool {
        match (l, r) {
            (Ty::Str, Ty::Str) => true,
            (Ty::Str, other) | (other, Ty::Str) => {
                if !other.is_unknown() && !other.is_error() {
                    let shown = other.display(self.interner);
                    self.diags.push(
                        Diagnostic::error(format!(
                            "`{}` is not defined between a `str` and a `{shown}`",
                            op.glyph()
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

    /// Type `s.<method>(args)` on a `str` receiver `base` (typed `bty`), or return `None` when it
    /// is not a `str` or `method` is not one of its methods ([`STR_METHODS`]; `len` of a `str`
    /// value is typed with the slice extent queries). A string-literal receiver adapts to `str`
    /// here, as it does as the base of a range index: `"abc".len()`, `"ab".cmp(s)`. Records the
    /// [`StrOp`] for `mir_build`, keyed by `callee`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn type_str_method(
        &mut self,
        bty: &Ty,
        base: &Expr,
        callee: &Expr,
        method: Ident,
        args: &[Expr],
        arg_tys: &mut [Ty],
        span: Span,
    ) -> Option<Ty> {
        let name = self.sym_str(method.sym).to_string();
        if !matches!(bty, Ty::Str) {
            if !STR_METHODS.contains(&name.as_str()) || !self.literal_adapts(&Ty::Str, base) {
                return None;
            }
            self.retype_adapted_literal(base, &Ty::Str);
            if name == "len" {
                let t = self.type_extent_method(&Ty::Str, false, args, span);
                self.types.insert(callee.id, Ty::Unknown);
                return t;
            }
        }
        let (op, ret, arity) = match name.as_str() {
            "as_bytes" => (
                StrOp::AsBytes,
                Ty::Slice(Box::new(Ty::Scalar(Scalar::U8))),
                0,
            ),
            "as_ptr" => (
                StrOp::AsPtr,
                Ty::Ptr {
                    mutable: false,
                    pointee: Box::new(Ty::Scalar(Scalar::U8)),
                },
                0,
            ),
            // The loop desugaring reads the receiver itself; typed as that `str`, a stray lowering
            // of the call as a value would still produce a well-formed fat pointer.
            "chars" => (StrOp::Chars, Ty::Str, 0),
            "char_count" => (StrOp::CharCount, Ty::Scalar(Scalar::I64), 0),
            "cmp" => (StrOp::Cmp, Ty::Scalar(Scalar::I32), 1),
            _ => return None,
        };
        self.types.insert(callee.id, Ty::Unknown);
        if args.len() != arity {
            let want = if arity == 0 {
                "no arguments".to_string()
            } else {
                "exactly 1 argument (another `str`)".to_string()
            };
            self.error(
                span,
                "E0503",
                format!(
                    "`.{name}()` on a `str` takes {want}, but {} were supplied",
                    args.len()
                ),
            );
            return Some(ret);
        }
        if op == StrOp::Cmp {
            if self.literal_adapts(&Ty::Str, &args[0]) {
                self.retype_adapted_literal(&args[0], &Ty::Str);
                arg_tys[0] = Ty::Str;
            }
            if !matches!(arg_tys[0], Ty::Str | Ty::Unknown | Ty::Error) {
                let shown = arg_tys[0].display(self.interner);
                self.error(
                    args[0].span,
                    "E0433",
                    format!(
                        "`.cmp()` compares a `str` with another `str`, not with a `{shown}` (a \
                         string literal adapts)"
                    ),
                );
            }
        }
        if op == StrOp::Chars && !self.str_state.chars_heads.contains(&callee.id) {
            self.diags.push(
                Diagnostic::error(
                    "`.chars()` can only be iterated by a `for` loop: there is no iterator value",
                )
                .with_code("E0425")
                .primary(span, "this `.chars()` is not the iterable of a `for` head")
                .help(
                    "write `for c in s.chars() { … }`; `s.char_count()` counts the characters, and \
                     `s[i]` reads a byte",
                ),
            );
        }
        self.str_state.info.ops.insert(callee.id, op);
        Some(ret)
    }

    /// Mark the `s.chars()` call that is the iterable of the `for` head `iter` (when it is one) so
    /// that typing it does not report `E0425`. Syntactic: the receiver is checked to be a `str` by
    /// [`Self::type_str_method`], and the loop takes `char` elements only when that recorded
    /// [`StrOp::Chars`] ([`Self::for_chars_elem`], which unmarks it).
    pub(crate) fn mark_chars_head(&mut self, iter: &ForIter) {
        let ForIter::Expr(e) = iter else {
            return;
        };
        if let ExprKind::Call { callee, args, .. } = &e.kind {
            if let ExprKind::Field { name, .. } = &callee.kind {
                if args.is_empty() && self.sym_str(name.sym) == "chars" {
                    self.str_state.chars_heads.insert(callee.id);
                }
            }
        }
    }

    /// The element type of a `for` over `s.chars()`, or `None` when `e` is not one (after
    /// [`Self::mark_chars_head`] and typing `e`).
    pub(crate) fn for_chars_elem(&mut self, e: &Expr) -> Option<Ty> {
        let ExprKind::Call { callee, .. } = &e.kind else {
            return None;
        };
        (self.str_state.chars_heads.remove(&callee.id)
            && self.str_state.info.ops.get(&callee.id) == Some(&StrOp::Chars))
        .then_some(Ty::Scalar(Scalar::Char))
    }

    /// Whether `e` is a `for` iterable sema typed as `s.chars()`.
    pub(crate) fn is_chars_iter(&self, e: &Expr) -> bool {
        matches!(&e.kind, ExprKind::Call { callee, .. }
            if self.str_state.info.ops.get(&callee.id) == Some(&StrOp::Chars))
    }

    /// Type one of the `str` builtins — `str_from_utf8(b)`, `utf8_valid_up_to(b)`,
    /// `str_concat(a, b, …)` — or return `None` for any other name. Records the [`StrOp`] for
    /// `mir_build`. A user function of the same name shadows the builtin (the caller resolves it
    /// first).
    pub(crate) fn type_str_builtin(
        &mut self,
        name: &str,
        callee: &Expr,
        args: &[Expr],
        arg_tys: &mut [Ty],
        span: Span,
    ) -> Option<Ty> {
        let (op, ret) = match name {
            "str_from_utf8" => (StrOp::FromUtf8, Ty::Str),
            "utf8_valid_up_to" => (StrOp::ValidUpTo, Ty::Scalar(Scalar::I64)),
            "str_concat" => (StrOp::Concat, Ty::Str),
            _ => return None,
        };
        self.types.insert(callee.id, Ty::Unknown);
        self.str_state.info.ops.insert(callee.id, op);
        if op == StrOp::Concat {
            if args.is_empty() {
                self.error(
                    span,
                    "E0503",
                    "`str_concat` takes one or more `str`s to join, but none were supplied",
                );
            }
            for (a, t) in args.iter().zip(arg_tys.iter_mut()) {
                if self.literal_adapts(&Ty::Str, a) {
                    self.retype_adapted_literal(a, &Ty::Str);
                    *t = Ty::Str;
                }
                if !matches!(t, Ty::Str | Ty::Unknown | Ty::Error) {
                    let shown = t.display(self.interner);
                    self.error(
                        a.span,
                        "E0437",
                        format!(
                            "`str_concat` joins `str`s, but this argument is a `{shown}` (a string \
                             literal adapts; `str_format` formats other values into a `str`)"
                        ),
                    );
                }
            }
            return Some(ret);
        }
        if args.len() != 1 {
            self.error(
                span,
                "E0503",
                format!(
                    "`{name}` takes exactly 1 argument (a `[]u8` or `[u8; N]` buffer), but {} \
                     were supplied",
                    args.len()
                ),
            );
            return Some(ret);
        }
        if matches!(arg_tys[0], Ty::Str) {
            self.error(
                args[0].span,
                "E0437",
                format!("`{name}` takes bytes, and this is already a `str`; use it as it is"),
            );
        } else {
            self.check_u8_buffer(name, &args[0], &arg_tys[0], 0, false);
        }
        Some(ret)
    }

    /// The typing of a string-literal pattern `"…"` (and of the other literal patterns against a
    /// `str`): a string-literal pattern matches only a `str`, and a `str` is matched only by string
    /// literals (plus names, `_` and or-patterns of them). Anything else is `E0426`.
    pub(crate) fn check_str_literal_pattern(&mut self, pat: &Pattern, ty: &Ty) {
        let is_str_pat = matches!(pat.kind, PatKind::Str(_));
        if let PatKind::Str(sym) = &pat.kind {
            self.check_unicode_escapes(*sym, pat.span);
        }
        let mismatch = match ty {
            Ty::Unknown | Ty::Error => false,
            Ty::Str => !is_str_pat,
            _ => is_str_pat && !self.is_generic_ty(ty),
        };
        if !mismatch {
            return;
        }
        let shown = ty.display(self.interner);
        let (msg, label) = if is_str_pat {
            (
                format!("a string-literal pattern matches only a `str`, not a `{shown}`"),
                "a string literal",
            )
        } else {
            (
                "a `str` is matched by string-literal patterns, not by this literal".to_string(),
                "not a string literal",
            )
        };
        let c_string = matches!(ty, Ty::Ptr { pointee, .. }
            if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)));
        self.diags.push(
            Diagnostic::error(msg)
                .with_code("E0426")
                .primary(pat.span, label)
                .help(if !is_str_pat {
                    "write the text as a string literal (`\"abc\" =>`), or bind the `str` with a \
                     name and test it in a guard"
                } else if c_string {
                    "a `*u8` (an unannotated string literal, `arg`, `env`) is a C string; match a \
                     `str` instead (`let s: str = \"…\";`, or `str_from_utf8` of its bytes)"
                } else {
                    "match the value against patterns of its own type"
                }),
        );
    }

    /// A `match` scrutinee that is a string literal (a `*u8` on its own) is a `str` when an arm
    /// tests it against a string literal: `match "add" { "add" => .., _ => .. }` compares text.
    pub(crate) fn adapt_str_scrutinee(&mut self, scrutinee: &Expr, ty: &mut Ty, arms: &[MatchArm]) {
        fn tests_text(p: &Pattern) -> bool {
            match &p.kind {
                PatKind::Str(_) => true,
                PatKind::Or(alts) => alts.iter().any(tests_text),
                PatKind::Binding { sub, .. } => tests_text(sub),
                _ => false,
            }
        }
        if !matches!(ty, Ty::Str)
            && arms.iter().any(|a| tests_text(&a.pat))
            && self.literal_adapts(&Ty::Str, scrutinee)
        {
            self.retype_adapted_literal(scrutinee, &Ty::Str);
            *ty = Ty::Str;
        }
    }

    /// Whether every value `e` can produce is a string literal, through `if`/`else`, `match` arms
    /// and block tails — an expression that therefore adapts to `str` exactly as one literal does
    /// (`let s: str = if c { "a" } else { "b" };`). An arm that is already a `str` counts too.
    pub(crate) fn str_literal_arms(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Str(_) => true,
            ExprKind::Block(b) => self.str_block_arms(b),
            ExprKind::If {
                then_branch,
                else_branch: Some(els),
                ..
            } => self.str_block_arms(then_branch) && self.str_literal_arms(els),
            ExprKind::Match { arms, .. } => {
                !arms.is_empty() && arms.iter().all(|a| self.str_literal_arms(&a.body))
            }
            _ => matches!(self.types.get(&e.id), Some(Ty::Str)),
        }
    }

    fn str_block_arms(&self, b: &Block) -> bool {
        b.tail.as_ref().is_some_and(|t| self.str_literal_arms(t))
    }

    /// Re-stamp an expression [`Self::str_literal_arms`] accepted as `str`: itself, and every
    /// arm and tail on the way down to its literals.
    pub(crate) fn retype_str_arms(&mut self, e: &Expr) {
        self.types.insert(e.id, Ty::Str);
        match &e.kind {
            ExprKind::Block(b) => self.retype_str_block(b),
            ExprKind::If {
                then_branch,
                else_branch: Some(els),
                ..
            } => {
                self.retype_str_block(then_branch);
                self.retype_str_arms(els);
            }
            ExprKind::Match { arms, .. } => {
                for a in arms {
                    self.retype_str_arms(&a.body);
                }
            }
            _ => {}
        }
    }

    fn retype_str_block(&mut self, b: &Block) {
        if let Some(t) = &b.tail {
            self.retype_str_arms(t);
        }
    }

    /// Before an `if`/`match` merge: when some arm is a `str` (`any_str`), an arm whose value is a
    /// string literal (or an `if`/`match` of them) becomes a `str` too, wherever it sits — the
    /// first arm included, which the merge rule alone (a literal adapts to the type ESTABLISHED by
    /// the arms before it) could not reach: `if c { "a" } else { s }`.
    pub(crate) fn adapt_str_arm(&mut self, arm: &Expr, ty: &mut Ty, any_str: bool) {
        if any_str && !matches!(ty, Ty::Str) && self.str_literal_arms(arm) {
            self.retype_str_arms(arm);
            *ty = Ty::Str;
        }
    }

    /// [`Self::adapt_str_arm`] for an `if`'s then-block.
    pub(crate) fn adapt_str_block_arm(&mut self, b: &Block, ty: &mut Ty, any_str: bool) {
        if any_str && !matches!(ty, Ty::Str) && self.str_block_arms(b) {
            self.retype_str_block(b);
            *ty = Ty::Str;
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
    use wukong_ast::ItemKind;
    use wukong_span::{Interner, SourceId};
    use wukong_types::Ty;

    use crate::{DefKind, SemaResult, StrOp, VariantPayload};

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
    fn a_closures_recorded_types_are_erased_too() {
        // `wukong_mir_build` builds a lifted closure's signature from `SemaResult::closures`, so a
        // `str` parameter, return or pointer capture left there reached MIR lowering (an ICE).
        let (res, codes, _) = check(
            "fn apply_s(f: fn(str) -> i64, s: str) -> i64 { return f(s); }
             fn name_of(f: fn(i64) -> str, i: i64) -> str { return f(i); }
             fn main() -> i32 {
                 let s: str = \"hi\";
                 let ps = &s;
                 let n = apply_s(|t| t.len() + (*ps).len(), s);
                 let m = name_of(|i| -> str { if i > 0 { return \"a\"; } return \"b\"; }, n);
                 return m.len() as i32;
             }",
        );
        assert!(codes.is_empty(), "unexpected {codes:?}");
        assert_eq!(res.closures.len(), 2);
        for info in res.closures.values() {
            let tys = info.params.iter().chain([&info.ret]);
            for t in tys.chain(info.captures.iter().map(|(_, t)| t)) {
                assert!(!t.mentions_str(), "a closure still records {t:?}");
            }
        }
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
        // `atomic_add` writes its place like `=` does: on a `str` it would write `.rodata`.
        rejects(
            "fn main() -> i32 { let s: str = \"abc\"; atomic_add(s[0], 1); return 0; }",
            "E0304",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"abc\"; atomic_add(s[1..][0], 1); return 0; }",
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
        // Two `str`s are ordered by their bytes; a literal operand adapts on either side.
        accepts(
            "fn main() -> i32 { let s: str = \"ab\"; let t: str = \"cd\"; \
             return ((s < t) && (s <= t) && (t > s) && (t >= s) && (\"a\" < s) && (s >= \"a\")) as i32; }",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; let p = \"ab\"; return (s < p) as i32; }",
            "E0433",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; return (s < 3) as i32; }",
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

    /// An imported module's items reach sema renamed `<import.path>.<name>` by the loader
    /// (`wukong_driver::loader::resolve`), so the reservation must compare the name the program
    /// WROTE: a `struct str` in `lib/m.wk` arrives as `lib.m.str`, and inside that module (or
    /// through a bare import) a type written `str` then silently meant the struct.
    #[test]
    fn str_is_reserved_in_an_imported_module_too() {
        for (src, mangled, reserved) in [
            ("struct str { x: i32 }", "lib.m.str", true),
            ("enum str { A }", "lib.m.str", true),
            ("trait str { fn f(self) -> i32; }", "lib.m.str", true),
            ("struct S { x: i32 }", "lib.m.strs", false),
            ("struct S { x: i32 }", "lib.str.S", false),
        ] {
            let mut interner = Interner::new();
            let src = format!("{src} fn main() -> i32 {{ return 0; }}");
            let (mut module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
            assert!(pd.is_empty(), "parse errors: {pd:?}");
            let sym = interner.intern(mangled);
            for item in &mut module.items {
                match &mut item.kind {
                    ItemKind::Struct(d) => d.name.sym = sym,
                    ItemKind::Enum(d) => d.name.sym = sym,
                    ItemKind::Trait(d) => d.name.sym = sym,
                    _ => {}
                }
            }
            let (_, diags) = crate::check(&module, &interner);
            let codes: Vec<_> = diags.iter().filter_map(|d| d.code).collect();
            assert_eq!(
                codes.contains(&"E0300"),
                reserved,
                "{mangled} from `{src}`: {codes:?}"
            );
        }
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
    fn str_prints_in_aggregates_too() {
        accepts("fn main() -> i32 { let s: str = \"a\"; print(s); println(s); print_raw(s); eprintln(s); printf(\"{}{:?}\", s, s); return 0; }");
        // A `str` inside an aggregate or behind a reference prints its text; the printer walks the
        // unerased type sema keeps for it.
        let (res, codes, interner) = check(
            "struct S { t: str, n: i32 } fn show(r: &str) { println(r); } \
             fn main() -> i32 { let s = S { t: \"a\", n: 1 }; let a: [str; 1] = [\"b\"]; \
             let p: (str, i32) = (\"c\", 2); let x: str = \"d\"; show(&x); \
             println(s); println(a); printf(\"{} {:?}\\n\", p, x); return 0; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        let mut shown: Vec<String> = res
            .str_info
            .print_types
            .values()
            .map(|t| t.display(&interner))
            .collect();
        // The array, the reference and the tuple; the struct is found through its declaration
        // (`struct_fields`), whatever its type is spelled as.
        shown.sort();
        assert_eq!(shown, ["&str", "(str, i32)", "[str; 1]"], "{shown:?}");
        let s_sym = interner.get("S").unwrap();
        assert_eq!(
            res.str_info.struct_fields.get(&s_sym),
            Some(&vec![Ty::Str, Ty::Scalar(wukong_types::Scalar::I32)])
        );
        // A slice of `str`s is a slice of non-scalars, which prints element by element.
        rejects(
            "fn main() -> i32 { let a: [str; 1] = [\"a\"]; let v: []str = a; println(v); return 0; }",
            "E0448",
        );
    }

    #[test]
    fn str_methods_are_typed_and_recorded() {
        let (res, codes, _) = check(
            "fn main() -> i32 { let s: str = \"h\\u{e9}\"; let b: []u8 = s.as_bytes(); \
             let p: *u8 = s.as_ptr(); let n: i64 = s.char_count(); let c: i32 = s.cmp(\"x\"); \
             let m: i64 = \"lit\".len() + \"lit\".char_count(); let o: i32 = \"a\".cmp(s); \
             for ch in s.chars() { let k: char = ch; } \
             for ch in \"xy\".chars() { let k: char = ch; } \
             let _ = (b, p, n, c, m, o); return 0; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        let mut ops: Vec<StrOp> = res.str_info.ops.values().copied().collect();
        ops.sort_by_key(|o| format!("{o:?}"));
        assert_eq!(
            ops,
            vec![
                StrOp::AsBytes,
                StrOp::AsPtr,
                StrOp::CharCount,
                StrOp::CharCount,
                StrOp::Chars,
                StrOp::Chars,
                StrOp::Cmp,
                StrOp::Cmp,
            ]
        );
        rejects(
            "fn main() -> i32 { let s: str = \"a\"; let n = s.as_bytes(1); return 0; }",
            "E0503",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"a\"; let n = s.cmp(); return 0; }",
            "E0503",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"a\"; let n = s.cmp(1); return 0; }",
            "E0433",
        );
        rejects(
            "fn main() -> i32 { let s: str = \"a\"; let p = \"b\"; let n = s.cmp(p); return 0; }",
            "E0433",
        );
        // A `[]u8` has no text methods.
        rejects(
            "fn main() -> i32 { let a: [u8; 1] = [1]; let b: []u8 = a; let n = b.char_count(); return 0; }",
            "E0322",
        );
    }

    #[test]
    fn chars_is_only_a_for_iterable() {
        for body in [
            "let it = s.chars();",
            "let n = s.chars().len();",
            "for c in s.chars().chars() { }",
            "for b in s.as_bytes() { let x = s.chars(); }",
        ] {
            let src = format!("fn main() -> i32 {{ let s: str = \"ab\"; {body} return 0; }}");
            rejects(&src, "E0425");
        }
        // A head whose receiver holds another `for` over `.chars()` is still a head.
        accepts(
            "fn main() -> i32 { let s: str = \"ab\"; let mut n: i32 = 0; \
             for c in { for d in s.chars() { n = n + (d as i32); } s }.chars() { \
             n = n + (c as i32); } return n; }",
        );
        // A `char` cannot be destructured.
        rejects(
            "fn main() -> i32 { let s: str = \"ab\"; for (a, b) in s.chars() { } return 0; }",
            "E0401",
        );
    }

    #[test]
    fn str_builtins_are_typed() {
        let (res, codes, _) = check(
            "fn main() -> i32 { let a: [u8; 2] = [104, 105]; let b: []u8 = a; \
             let s: str = str_from_utf8(a); let t: str = str_from_utf8(b); \
             let n: i64 = utf8_valid_up_to(b) + utf8_valid_up_to(a); \
             let j: str = str_concat(s, \"-\", t); let k: str = str_concat(\"only\"); \
             let f: str = str_format(\"{}{}\", n, s); free(j); free(k); free(f); return 0; }",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert_eq!(
            res.str_info
                .ops
                .values()
                .filter(|o| **o == StrOp::Concat)
                .count(),
            2
        );
        for (body, code) in [
            ("let s: str = \"a\"; let t = str_from_utf8(s);", "E0437"),
            ("let t = str_from_utf8(3);", "E0437"),
            ("let a: [i32; 1] = [1]; let t = str_from_utf8(a);", "E0437"),
            (
                "let a: [u8; 1] = [1]; let t = str_from_utf8(a, a);",
                "E0503",
            ),
            ("let t = str_concat();", "E0503"),
            ("let t = str_concat(\"a\", 1);", "E0437"),
            ("let p = \"a\"; let t = str_concat(p);", "E0437"),
            ("let t = str_format();", "E0503"),
            ("let t = str_format(\"{}\");", "E0450"),
            ("let f = \"{}\"; let t = str_format(f, 1);", "E0449"),
            (
                "let a: [u8; 1] = [1]; let t: str = utf8_valid_up_to(a);",
                "E0401",
            ),
        ] {
            let src = format!("fn main() -> i32 {{ {body} return 0; }}");
            rejects(&src, code);
        }
        // A user function of the same name shadows the builtin.
        accepts(
            "fn str_concat(a: i32) -> i32 { return a; } fn main() -> i32 { return str_concat(1); }",
        );
    }

    #[test]
    fn string_literal_patterns() {
        accepts(
            "fn k(s: str) -> i32 { match s { \"a\" | \"b\" => 1, x @ \"c\" => x.len() as i32, _ => 0 } } \
             fn main() -> i32 { let t: (str, i32) = (\"a\", 1); \
             let n = match t { (\"a\", 1) => 1, (_, _) => 0 }; \
             let m = match \"lit\" { \"lit\" => 1, _ => 0 }; \
             if let \"a\" = t.0 { return n + m + k(\"a\"); } return 0; }",
        );
        // A `str` match needs a catch-all: no set of literals covers every `str`.
        rejects(
            "fn main() -> i32 { let s: str = \"a\"; return match s { \"a\" => 1, \"b\" => 2 }; }",
            "E0405",
        );
        rejects(
            "fn main() -> i32 { let x: i32 = 1; return match x { \"a\" => 1, _ => 2 }; }",
            "E0426",
        );
        rejects(
            "fn main() -> i32 { let p = arg(0); return match p { \"a\" => 1, _ => 2 }; }",
            "E0426",
        );
        for pat in ["1", "'a'", "true", "1..=3"] {
            let src = format!(
                "fn main() -> i32 {{ let s: str = \"a\"; return match s {{ {pat} => 1, _ => 2 }}; }}"
            );
            rejects(&src, "E0426");
        }
        rejects(
            "fn main() -> i32 { let s: str = \"a\"; return match s { \"\\u{D800}\" => 1, _ => 2 }; }",
            "E0432",
        );
    }

    #[test]
    fn all_literal_arms_adapt_to_str() {
        for src in [
            "fn main() -> i32 { let c = true; let s: str = if c { \"a\" } else { \"bc\" }; return s.len() as i32; }",
            "fn main() -> i32 { let k = 2; let s: str = match k { 1 => \"a\", 2 => { \"bb\" } _ => if k > 9 { \"c\" } else { \"d\" } }; return s.len() as i32; }",
            "fn f(c: bool) -> str { if c { \"a\" } else { \"b\" } } fn main() -> i32 { return f(true).len() as i32; }",
            "fn g(s: str) -> i64 { return s.len(); } fn main() -> i32 { let c = false; return g(if c { \"a\" } else { \"bb\" }) as i32; }",
            // A literal arm beside a `str` arm adapts wherever it sits, the first arm included.
            "fn main() -> i32 { let c = true; let t: str = \"t\"; let s = if c { \"a\" } else { t }; return s.len() as i32; }",
            "fn main() -> i32 { let k = 1; let t: str = \"t\"; let s = match k { 0 => \"a\", 1 => t, _ => \"c\" }; return s.len() as i32; }",
        ] {
            assert!(!literal_left_as_ptr(src), "a literal stayed `*u8` in {src}");
        }
        // Unannotated and with no `str` arm, the arms stay `*u8` (as numeric literals keep their
        // own type).
        accepts("fn main() -> i32 { let c = true; let p = if c { \"a\" } else { \"b\" }; print(p); return 0; }");
    }
}
