//! Match exhaustiveness and arm reachability, decided by pattern **usefulness** — Maranget,
//! "Warnings for pattern matching" (JFP 2007), the algorithm behind rustc's
//! `rustc_pattern_analysis` and the OCaml and Scala 3 match checkers.
//!
//! A pattern vector `q` is *useful* with respect to a matrix of pattern rows `P` when some value
//! matches `q` and no row of `P`. Both match diagnostics reduce to that one question:
//!
//! * **Exhaustiveness (`E0405`).** A `match` is exhaustive iff the wildcard `_` is *not* useful
//!   with respect to the rows of its unguarded arms. When it is useful, the same recursion builds a
//!   *witness* — a pattern describing values no arm covers — which the diagnostic prints:
//!   `Expr::Add(i32::MIN..=-1 | 1..=i32::MAX, _)`.
//! * **Reachability (`W0020`).** An arm is unreachable iff its pattern is not useful with respect
//!   to the unguarded arms before it. A guarded arm never covers anything (its guard may be false),
//!   but it can itself be unreachable.
//!
//! The constructor space of a column's type:
//!
//! | type | constructors |
//! |---|---|
//! | `bool` | `false`, `true` |
//! | integers and `char` | ranges over the type's whole domain, split at every bound a row of the column names (a `char` holds any 32-bit value, since an integer `as char` is unchecked) |
//! | `enum` | one per discriminant; a payload variant's fields are sub-columns ([`VariantPayload::field_tys`]) |
//! | tuple, `()`, struct, array `[T; N]` (up to 32 fields) | the single constructor whose fields are the elements |
//! | slice `[]T` | one per length — split as below — whose fields are the elements |
//! | `str` | one per distinct string literal a row names (by its decoded bytes), and the infinitely many other strings, which only `_` or a binding covers |
//! | anything else — floats, pointers, tensors, generics | none the analysis can name: only `_` or a binding covers it |
//!
//! A slice column has a constructor per length, infinitely many, which the split makes finite the
//! way rustc does: with `L` past every length a row pins down — longer than any `[a, b]` without
//! `..` lists, and at least the longest `[a, .., z]` prefix plus the longest suffix — the lengths
//! `0..L` are one constructor each and every length `>= L` is one variable-length constructor whose
//! fields are its first `p` and last `s` elements (`p + s = L`). No row can tell two lengths `>= L`
//! apart, so that constructor stands for all of them. A slice match is exhaustive only when every
//! length is covered: `[] | [_, ..]` is, `[] | [_]` is not (witness `[_, _, ..]`).
//!
//! The analysis is **total or silent**. A pattern it cannot place in its column's space — a pattern
//! of the wrong type (sema reports that on its own), a non-literal range bound, an integer pattern
//! on a data-carrying enum, a tagged union two of whose variants share a discriminant, a variant
//! named through another enum — makes [`analyze`] return `None` and the caller reports nothing.
//! `E0405` and `W0020` therefore fire only on a *certain* miss, which is the contract the
//! variant-name check this replaces kept too; that check, though, ignored payload sub-patterns,
//! tuples and integer ranges, so a refutable `Expr::Add(0, y)` counted as covering all of `Add`,
//! `match (a, b)` was never checked, and a full `0..=255` on a `u8` was rejected.
//!
//! A positive verdict is only as good as the values it models, and one way to build a value the
//! analysis does not model survives sema: `let x: T;` without an initializer is zero-filled by
//! mir_build (there is no definite-assignment check), so `enum E { A = 1, B = 2 }; let e: E;`
//! holds discriminant 0, which no variant has. [`zero_value_modeled`] answers whether the all-zero
//! bit pattern of a type is one of the values modeled here; a `match` counts as *proven* exhaustive
//! (`SemaResult::exhaustive_matches`) only when it is. (`int as Enum` is E0401, and a forged
//! pointer's pointee is accepted undefined behaviour, so zero-initialization is the one remaining
//! way to forge such a value.)
//!
//! **Mirror pair.** A proven verdict now *removes a test*: mir_build lowers the last unguarded arm
//! of a proven match with no compare at all. The semantics of [`DPat`] here — which values each
//! pattern matches, how a literal is read at the scrutinee's width, what a `char` literal decodes
//! to — must therefore equal mir_build's `pattern_cond`. Before, a disagreement could only miss or
//! invent a diagnostic; now a value the two disagree on silently takes the last arm instead of
//! trapping. `wukong_opt`'s `exhaustive_int_match_agrees_with_rust_on_every_value` pins the integer
//! side on every `u8` and `i8` value; the two `char`-literal decoders (`char_value` here,
//! `decode_char_literal` there) are the remaining unpinned half.

use std::cell::RefCell;

use wukong_ast::{MatchArm, PatKind, Path, Pattern, VariantPat};
use wukong_span::{Interner, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{DefKind, DefMap, EnumVariant, VariantPayload};

/// Recursion budget for one `match`. Usefulness is NP-complete in general; a match that exhausts
/// the budget gets no verdict rather than a slow compile.
const BUDGET: u32 = 100_000;

/// How many alternatives of one incomplete column a witness spells out before it summarizes the
/// rest as a count (`… (and 5 more)`).
const MAX_LISTED: usize = 3;

/// The verdict on one `match`.
pub(crate) struct MatchReport {
    /// Values no arm covers, or `None` when the match is exhaustive.
    pub(crate) missing: Option<Witness>,
    /// The arms, by index, that can never be selected: every value their pattern matches is
    /// matched by an earlier unguarded arm.
    pub(crate) unreachable: Vec<usize>,
}

/// A rendered witness of non-exhaustiveness.
pub(crate) struct Witness {
    /// The uncovered values as a pattern, `Expr::Add(i32::MIN..=-1 | 1..=i32::MAX, _)`.
    pub(crate) pattern: String,
    /// Further uncovered alternatives the pattern does not list (beyond [`MAX_LISTED`] per column).
    pub(crate) more: usize,
}

/// Judge `arms` against a scrutinee of type `scrut`: which values no arm covers, and which arms are
/// unreachable. `None` when the analysis cannot reason about the match (see the module docs) — the
/// caller must then report nothing.
pub(crate) fn analyze(
    defs: &DefMap,
    interner: &Interner,
    scrut: &Ty,
    arms: &[MatchArm],
) -> Option<MatchReport> {
    if matches!(scrut, Ty::Unknown | Ty::Error) {
        return None;
    }
    let mut cx = Cx {
        defs,
        interner,
        budget: BUDGET,
        strs: RefCell::default(),
    };
    let mut pats = Vec::with_capacity(arms.len());
    for a in arms {
        pats.push(cx.lower(&a.pat, scrut)?);
    }
    let tys = [scrut.clone()];
    let mut rows: Vec<Vec<DPat>> = Vec::new();
    let mut unreachable = Vec::new();
    for (i, (arm, p)) in arms.iter().zip(&pats).enumerate() {
        if cx
            .useful(&rows, std::slice::from_ref(p), &tys)
            .ok()?
            .is_none()
        {
            unreachable.push(i);
        }
        if arm.guard.is_none() {
            rows.push(vec![p.clone()]);
        }
    }
    let witness = cx.useful(&rows, &[DPat::Wild], &tys).ok()?;
    let missing = witness.map(|w| {
        let mut more = 0;
        let pattern = match w.first() {
            Some(head) => cx.render(head, scrut, &mut more),
            None => "_".to_string(),
        };
        Witness { pattern, more }
    });
    Some(MatchReport {
        missing,
        unreachable,
    })
}

/// A value the single pattern `pat` does not match, for a value of type `ty` — the witness a `let`
/// with a refutable pattern reports (E0442). `None` when `pat` matches every value, or when the
/// analysis cannot reason about it.
pub(crate) fn refutable(
    defs: &DefMap,
    interner: &Interner,
    ty: &Ty,
    pat: &Pattern,
) -> Option<Witness> {
    if matches!(ty, Ty::Unknown | Ty::Error) {
        return None;
    }
    let mut cx = Cx {
        defs,
        interner,
        budget: BUDGET,
        strs: RefCell::default(),
    };
    let p = cx.lower(pat, ty)?;
    let tys = [ty.clone()];
    let w = cx.useful(&[vec![p]], &[DPat::Wild], &tys).ok()??;
    let mut more = 0;
    let pattern = match w.first() {
        Some(head) => cx.render(head, ty, &mut more),
        None => "_".to_string(),
    };
    Some(Witness { pattern, more })
}

/// Whether the analysis can place `pat` against a value of type `ty` at all. When it cannot,
/// [`refutable`] answers `None` without having proved the pattern matches everything, and a caller
/// that must not let a refutable pattern through decides by some other rule.
pub(crate) fn placeable(defs: &DefMap, interner: &Interner, ty: &Ty, pat: &Pattern) -> bool {
    let cx = Cx {
        defs,
        interner,
        budget: BUDGET,
        strs: RefCell::default(),
    };
    cx.lower(pat, ty).is_some()
}

/// How deep [`zero_value_modeled`] follows nested types before it gives up (answering `false`).
const ZERO_DEPTH: u32 = 64;

/// Whether the all-zero bit pattern of `ty` — what mir_build gives an uninitialized `let x: T;` —
/// is a value this analysis models (see the module docs). A column patterns cannot constrain
/// (a float, pointer, slice, `str`, tensor, vector, function) holds any value harmlessly; a
/// product is modeled when every field is; an enum when some variant has discriminant 0 and that
/// variant's payload is modeled. An unknown or generic type is not.
pub(crate) fn zero_value_modeled(defs: &DefMap, ty: &Ty) -> bool {
    zero_modeled_at(defs, ty, 0)
}

fn zero_modeled_at(defs: &DefMap, ty: &Ty, depth: u32) -> bool {
    if depth >= ZERO_DEPTH {
        return false;
    }
    let next = depth + 1;
    match ty {
        Ty::Scalar(_)
        | Ty::Unit
        | Ty::Ptr { .. }
        | Ty::Ref { .. }
        | Ty::Slice(_)
        | Ty::Str
        | Ty::Tensor { .. }
        | Ty::Vector { .. }
        | Ty::Fn { .. } => true,
        Ty::Tuple(ts) => ts.iter().all(|t| zero_modeled_at(defs, t, next)),
        Ty::Array { elem, .. } => zero_modeled_at(defs, elem, next),
        Ty::Named(n) => match defs.lookup(*n).map(|d| &d.kind) {
            Some(DefKind::Struct(fs)) => fs.iter().all(|(_, t)| zero_modeled_at(defs, t, next)),
            Some(DefKind::Enum(vs)) => vs.iter().any(|v| {
                v.disc == 0
                    && v.payload
                        .field_tys()
                        .iter()
                        .all(|t| zero_modeled_at(defs, t, next))
            }),
            Some(DefKind::Fn(_) | DefKind::Const(_)) | None => false,
        },
        Ty::Unknown | Ty::Error => false,
    }
}

/// A pattern deconstructed against its column's constructor space.
#[derive(Clone, Debug)]
enum DPat {
    /// `_` or a binding: matches everything.
    Wild,
    Bool(bool),
    /// An integer (or `char`) range `lo..=hi`, already normalized to the column's type. Empty when
    /// `lo > hi`: such a pattern matches nothing.
    Range(i128, i128),
    /// Enum constructor `k` (an index into [`EnumSpace::classes`]) with its payload fields.
    Variant(usize, Vec<DPat>),
    /// The single constructor of a tuple or `()`, with its fields.
    Product(Vec<DPat>),
    /// A string literal on a `str` column: an index into [`Cx::strs`], one per distinct byte
    /// string, so `"a"` and `"\x61"` are one constructor.
    Str(u32),
    /// An array pattern on a `[]T`: without `..` (`var: false`, `suffix` empty) it matches the
    /// slices of exactly `prefix.len()` elements; with one, those of at least
    /// `prefix.len() + suffix.len()`.
    Slice {
        prefix: Vec<DPat>,
        suffix: Vec<DPat>,
        var: bool,
    },
    Or(Vec<DPat>),
}

/// A constructor of a column, as split for one step of the algorithm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctor {
    Bool(bool),
    /// One segment `lo..=hi` of a split integer column: every row range either contains it or is
    /// disjoint from it.
    Seg(i128, i128),
    /// An enum constructor (a discriminant class).
    Class(usize),
    /// The one constructor of a tuple or `()`.
    Single,
    /// The slices of exactly this many elements.
    SliceFixed(usize),
    /// The slices of at least `p + s` elements, whose fields are the first `p` and the last `s`.
    SliceVar(usize, usize),
    /// The one `str` a string literal names (an index into [`Cx::strs`]).
    Str(u32),
}

/// A witness under construction: one per column.
#[derive(Clone, Debug)]
enum Wit {
    Wild,
    Ctor(Ctor, Vec<Wit>),
    /// An incomplete column: the constructors no row names, each with wildcard fields. Empty when
    /// no row names a constructor at all, which renders as `_`.
    Missing(Vec<Ctor>),
}

/// The constructors of a column, split against the rows' heads.
enum Split {
    /// Every constructor of the type appears among the heads: recurse into each.
    Complete(Vec<Ctor>),
    /// These constructors appear in no head: the rows with a wildcard head decide.
    Incomplete(Vec<Ctor>),
}

/// The constructor space of a type.
enum Space {
    Bool,
    Int(Scalar),
    Enum(EnumSpace),
    Product(Product),
    /// A `[]T` slice of this element type, one constructor per length (see the module docs).
    Slice(Ty),
    /// A `str`: one constructor per string literal, and infinitely many no pattern names.
    Str,
    /// No constructors the analysis can name.
    Opaque,
}

struct EnumSpace {
    name: Symbol,
    variants: Vec<EnumVariant>,
    /// The constructors: indices into `variants` sharing one discriminant. One variant each,
    /// except in a C-style enum that gives two variants the same value — at run time they are the
    /// same value, so a pattern naming either covers both.
    classes: Vec<Vec<usize>>,
}

impl EnumSpace {
    /// The constructor a `Enum::Variant` path names, with its payload arity. `None` for a path
    /// through another enum or an unknown variant (sema reports the latter).
    fn class_of(&self, path: &Path) -> Option<(usize, usize)> {
        let [ename, vname] = path.segments.as_slice() else {
            return None;
        };
        if ename.sym != self.name {
            return None;
        }
        let vi = self.variants.iter().position(|v| v.name == vname.sym)?;
        let k = self.classes.iter().position(|c| c.contains(&vi))?;
        Some((k, self.variants[vi].payload.field_tys().len()))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProductKind {
    Unit,
    Tuple,
    /// A declared struct; [`Product::names`] holds its field names.
    Struct(Symbol),
    /// A fixed-size array, one field per element.
    Array,
}

struct Product {
    kind: ProductKind,
    fields: Vec<Ty>,
    /// The field names of a struct, in declaration order; empty otherwise.
    names: Vec<Symbol>,
}

/// The most fields a struct or array column is split into; a wider one is opaque (only `_` or a
/// binding covers it), which keeps the matrix narrow and the recursion shallow.
const MAX_PRODUCT_FIELDS: usize = 32;

/// The analysis was cut off by [`BUDGET`].
struct Exhausted;

struct Cx<'a> {
    defs: &'a DefMap,
    interner: &'a Interner,
    budget: u32,
    /// The decoded bytes of every string literal lowered so far; [`DPat::Str`] and [`Ctor::Str`]
    /// index it. A `RefCell` because lowering takes `&self`.
    strs: RefCell<Vec<Vec<u8>>>,
}

impl Cx<'_> {
    fn space(&self, ty: &Ty) -> Space {
        match ty {
            Ty::Scalar(Scalar::Bool) => Space::Bool,
            Ty::Scalar(s) if s.is_int() => Space::Int(*s),
            Ty::Unit => Space::Product(Product {
                kind: ProductKind::Unit,
                fields: Vec::new(),
                names: Vec::new(),
            }),
            Ty::Tuple(ts) => Space::Product(Product {
                kind: ProductKind::Tuple,
                fields: ts.clone(),
                names: Vec::new(),
            }),
            Ty::Array { elem, len, .. } if *len as usize <= MAX_PRODUCT_FIELDS => {
                Space::Product(Product {
                    kind: ProductKind::Array,
                    fields: vec![(**elem).clone(); *len as usize],
                    names: Vec::new(),
                })
            }
            Ty::Named(n) => match self.defs.lookup(*n).map(|d| &d.kind) {
                Some(DefKind::Enum(vs)) => enum_space(*n, vs),
                Some(DefKind::Struct(fs)) if fs.len() <= MAX_PRODUCT_FIELDS => {
                    Space::Product(Product {
                        kind: ProductKind::Struct(*n),
                        fields: fs.iter().map(|(_, t)| t.clone()).collect(),
                        names: fs.iter().map(|(f, _)| *f).collect(),
                    })
                }
                _ => Space::Opaque,
            },
            Ty::Slice(elem) => Space::Slice((**elem).clone()),
            Ty::Str => Space::Str,
            _ => Space::Opaque,
        }
    }

    /// The constructor index of the string literal `sym` (its raw source text): equal decoded
    /// bytes share one index — `mir_build` compares bytes, so must this.
    fn str_id(&self, sym: Symbol) -> u32 {
        let bytes = crate::text::decode_string_literal(self.interner.resolve(sym));
        let mut strs = self.strs.borrow_mut();
        match strs.iter().position(|b| *b == bytes) {
            Some(i) => i as u32,
            None => {
                strs.push(bytes);
                (strs.len() - 1) as u32
            }
        }
    }

    // ---- Lowering AST patterns ----

    /// Deconstruct `p` against a column of type `ty`, or `None` when it cannot be placed there.
    fn lower(&self, p: &Pattern, ty: &Ty) -> Option<DPat> {
        Some(match &p.kind {
            PatKind::Wildcard | PatKind::Ident(_) => DPat::Wild,
            PatKind::Unit => match ty {
                Ty::Unit => DPat::Wild,
                _ => return None,
            },
            PatKind::Or(alts) => DPat::Or(
                alts.iter()
                    .map(|a| self.lower(a, ty))
                    .collect::<Option<Vec<_>>>()?,
            ),
            PatKind::Bool(b) => match ty {
                Ty::Scalar(Scalar::Bool) => DPat::Bool(*b),
                _ => return None,
            },
            // On a `str` — or on the `[]u8` it is published as, which is how the lints, which run
            // on the published types, see one: sema admits a string-literal pattern only against a
            // `str` (E0426), so no real `[]u8` column ever holds one.
            PatKind::Str(sym) => match ty {
                Ty::Str => DPat::Str(self.str_id(*sym)),
                Ty::Slice(e) if matches!(e.as_ref(), Ty::Scalar(Scalar::U8)) => {
                    DPat::Str(self.str_id(*sym))
                }
                _ => return None,
            },
            PatKind::Int { .. } | PatKind::Char(_) => {
                let v = self.lit_value(p)?;
                self.int_pattern(ty, v, v, true)?
            }
            PatKind::Range { lo, hi, inclusive } => {
                let l = self.lit_value(lo)?;
                let h = self.lit_value(hi)?;
                self.int_pattern(ty, l, h, *inclusive)?
            }
            PatKind::Path(path) => {
                let Space::Enum(es) = self.space(ty) else {
                    return None;
                };
                let (k, arity) = es.class_of(path)?;
                DPat::Variant(k, vec![DPat::Wild; arity])
            }
            PatKind::Variant { path, fields } => {
                let Space::Enum(es) = self.space(ty) else {
                    return None;
                };
                let (k, _) = es.class_of(path)?;
                let payload = &es.variants[es.classes[k][0]].payload;
                let subs = match (fields, payload) {
                    (VariantPat::Tuple(subs), VariantPayload::Tuple(tys))
                        if subs.len() == tys.len() =>
                    {
                        subs.iter()
                            .zip(tys)
                            .map(|(s, t)| self.lower(s, t))
                            .collect::<Option<Vec<_>>>()?
                    }
                    (VariantPat::Struct(fps), VariantPayload::Struct(named)) => {
                        if fps
                            .iter()
                            .any(|fp| !named.iter().any(|(n, _)| *n == fp.name))
                        {
                            return None;
                        }
                        // A field the pattern does not name is unconstrained.
                        named
                            .iter()
                            .map(|(n, t)| match fps.iter().find(|fp| fp.name == *n) {
                                Some(fp) => self.lower(&fp.pat, t),
                                None => Some(DPat::Wild),
                            })
                            .collect::<Option<Vec<_>>>()?
                    }
                    _ => return None,
                };
                DPat::Variant(k, subs)
            }
            PatKind::Tuple(subs) => match ty {
                Ty::Tuple(tys) if tys.len() == subs.len() => DPat::Product(
                    subs.iter()
                        .zip(tys)
                        .map(|(s, t)| self.lower(s, t))
                        .collect::<Option<Vec<_>>>()?,
                ),
                _ => return None,
            },
            PatKind::Binding { sub, .. } => self.lower(sub, ty)?,
            PatKind::Struct { path, fields, .. } => {
                let Space::Product(Product {
                    kind: ProductKind::Struct(name),
                    fields: tys,
                    names,
                }) = self.space(ty)
                else {
                    return None;
                };
                if path.segments.last()?.sym != name
                    || fields.iter().any(|fp| !names.contains(&fp.name))
                {
                    return None;
                }
                // A field the pattern does not name is unconstrained.
                DPat::Product(
                    names
                        .iter()
                        .zip(&tys)
                        .map(|(n, t)| match fields.iter().find(|fp| fp.name == *n) {
                            Some(fp) => self.lower(&fp.pat, t),
                            None => Some(DPat::Wild),
                        })
                        .collect::<Option<Vec<_>>>()?,
                )
            }
            PatKind::Slice {
                prefix,
                rest,
                suffix,
                ..
            } => {
                if let Ty::Slice(elem) = ty {
                    let lower_all = |ps: &[Pattern]| {
                        ps.iter()
                            .map(|p| self.lower(p, elem))
                            .collect::<Option<Vec<_>>>()
                    };
                    return Some(DPat::Slice {
                        prefix: lower_all(prefix)?,
                        suffix: lower_all(suffix)?,
                        var: *rest,
                    });
                }
                let Space::Product(Product {
                    kind: ProductKind::Array,
                    fields: tys,
                    ..
                }) = self.space(ty)
                else {
                    return None;
                };
                let listed = prefix.len() + suffix.len();
                if listed > tys.len() || (!*rest && listed != tys.len()) {
                    return None;
                }
                let elem = tys.first()?;
                let mut subs = Vec::with_capacity(tys.len());
                for p in prefix {
                    subs.push(self.lower(p, elem)?);
                }
                subs.resize(tys.len() - suffix.len(), DPat::Wild);
                for p in suffix {
                    subs.push(self.lower(p, elem)?);
                }
                DPat::Product(subs)
            }
        })
    }

    /// The integer value of an integer- or char-literal pattern (a range bound included). A
    /// literal past `i128::MAX` (which only a `u128` can hold) is its 128-bit pattern, read back
    /// as unsigned by [`wrap`].
    fn lit_value(&self, p: &Pattern) -> Option<i128> {
        match &p.kind {
            PatKind::Int { sym, neg } => {
                let m = crate::parse_u128_text(self.interner.resolve(*sym))?;
                match i128::try_from(m) {
                    Ok(v) => Some(if *neg { -v } else { v }),
                    Err(_) if *neg => (m == 1u128 << 127).then_some(i128::MIN),
                    Err(_) => Some(m as i128),
                }
            }
            PatKind::Char(sym) => char_value(self.interner.resolve(*sym)),
            _ => None,
        }
    }

    /// A literal or range pattern `lo..hi` / `lo..=hi` against a column of type `ty`: an integer
    /// range, or — on a C-style enum, which matches by discriminant — the constructors whose
    /// discriminant it contains.
    fn int_pattern(&self, ty: &Ty, lo: i128, hi: i128, inclusive: bool) -> Option<DPat> {
        // Each bound is read as the compared type first (mir_build compares bit patterns); only
        // then does an exclusive upper bound stop one short, so `0..0` matches nothing.
        let bounds = |s: Scalar| -> Option<(i128, i128)> {
            let (l, h) = (wrap(lo, s)?, wrap(hi, s)?);
            Some(match (inclusive, h.checked_sub(1)) {
                (true, _) => (l, h),
                (false, Some(below)) => (l, below),
                // `..MIN` of a 128-bit type: no value lies below its least one.
                (false, None) => (1, 0),
            })
        };
        match self.space(ty) {
            Space::Int(s) => {
                let (l, h) = bounds(s)?;
                Some(DPat::Range(l, h))
            }
            Space::Enum(es) if es.variants.iter().all(|v| v.payload.is_unit()) => {
                // A C-style enum value is its `i32` discriminant, compared bit for bit.
                let (lo, hi) = bounds(Scalar::I32)?;
                Some(DPat::Or(
                    es.classes
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| {
                            let d = i128::from(es.variants[c[0]].disc);
                            (lo..=hi).contains(&d)
                        })
                        .map(|(k, _)| DPat::Variant(k, Vec::new()))
                        .collect(),
                ))
            }
            _ => None,
        }
    }

    // ---- Usefulness ----

    /// Whether the vector `v` is useful with respect to `rows` (columns typed by `tys`); when it is,
    /// a witness: one pattern per column describing values that match `v` and no row.
    fn useful(
        &mut self,
        rows: &[Vec<DPat>],
        v: &[DPat],
        tys: &[Ty],
    ) -> Result<Option<Vec<Wit>>, Exhausted> {
        self.budget = self.budget.checked_sub(1).ok_or(Exhausted)?;
        let Some((head, rest)) = v.split_first() else {
            return Ok(rows.is_empty().then(Vec::new));
        };
        // An or-pattern is useful when one of its alternatives is.
        if let DPat::Or(alts) = head {
            for a in alts {
                let mut q = Vec::with_capacity(v.len());
                q.push(a.clone());
                q.extend_from_slice(rest);
                if let Some(w) = self.useful(rows, &q, tys)? {
                    return Ok(Some(w));
                }
            }
            return Ok(None);
        }
        let rows = expand_or_heads(rows);
        let space = self.space(&tys[0]);
        let heads: Vec<&DPat> = rows.iter().map(|r| &r[0]).collect();
        let ctors = match head {
            DPat::Wild => match split_column(&space, &heads) {
                Split::Complete(cs) => cs,
                Split::Incomplete(missing) => {
                    // Some constructor appears in no head, so only the wildcard-headed rows can
                    // cover the values it builds: recurse on the default matrix.
                    let default: Vec<Vec<DPat>> = rows
                        .iter()
                        .filter(|r| matches!(r[0], DPat::Wild))
                        .map(|r| r[1..].to_vec())
                        .collect();
                    return Ok(self.useful(&default, rest, &tys[1..])?.map(|mut w| {
                        w.insert(0, Wit::Missing(missing));
                        w
                    }));
                }
            },
            _ => head_ctors(&space, head, &heads),
        };
        for c in ctors {
            let fields = ctor_fields(&space, c);
            let arity = fields.len();
            let spec: Vec<Vec<DPat>> = rows
                .iter()
                .filter_map(|r| specialize(r, c, arity))
                .collect();
            let Some(q) = specialize(v, c, arity) else {
                continue;
            };
            let mut sub_tys = fields;
            sub_tys.extend_from_slice(&tys[1..]);
            if let Some(mut w) = self.useful(&spec, &q, &sub_tys)? {
                let tail = w.split_off(arity);
                let mut out = Vec::with_capacity(tail.len() + 1);
                out.push(Wit::Ctor(c, w));
                out.extend(tail);
                return Ok(Some(out));
            }
        }
        Ok(None)
    }

    // ---- Rendering witnesses ----

    fn render(&self, w: &Wit, ty: &Ty, more: &mut usize) -> String {
        let space = self.space(ty);
        match w {
            Wit::Wild => "_".to_string(),
            Wit::Ctor(c, fields) => self.render_ctor(&space, *c, fields, more),
            Wit::Missing(cs) if cs.is_empty() => "_".to_string(),
            Wit::Missing(cs) => {
                *more += cs.len().saturating_sub(MAX_LISTED);
                cs.iter()
                    .take(MAX_LISTED)
                    .map(|c| {
                        let wild = vec![Wit::Wild; ctor_fields(&space, *c).len()];
                        self.render_ctor(&space, *c, &wild, more)
                    })
                    .collect::<Vec<_>>()
                    .join(" | ")
            }
        }
    }

    fn render_ctor(&self, space: &Space, c: Ctor, fields: &[Wit], more: &mut usize) -> String {
        let tys = ctor_fields(space, c);
        let subs: Vec<String> = fields
            .iter()
            .zip(&tys)
            .map(|(w, t)| self.render(w, t, more))
            .collect();
        match (space, c) {
            (_, Ctor::Bool(b)) => b.to_string(),
            (Space::Int(s), Ctor::Seg(a, b)) if a == b => int_text(a, *s),
            (Space::Int(s), Ctor::Seg(a, b)) => {
                format!("{}..={}", int_text(a, *s), int_text(b, *s))
            }
            (Space::Enum(es), Ctor::Class(k)) => {
                let v = &es.variants[es.classes[k][0]];
                let path = format!(
                    "{}::{}",
                    self.interner.resolve(es.name),
                    self.interner.resolve(v.name)
                );
                match &v.payload {
                    VariantPayload::Unit => path,
                    VariantPayload::Tuple(_) => format!("{path}({})", subs.join(", ")),
                    VariantPayload::Struct(named) => {
                        let named_subs: Vec<String> = named
                            .iter()
                            .zip(&subs)
                            .filter(|(_, s)| s.as_str() != "_")
                            .map(|((n, _), s)| format!("{}: {s}", self.interner.resolve(*n)))
                            .collect();
                        if named_subs.is_empty() {
                            format!("{path} {{ .. }}")
                        } else if named_subs.len() == named.len() {
                            format!("{path} {{ {} }}", named_subs.join(", "))
                        } else {
                            format!("{path} {{ {}, .. }}", named_subs.join(", "))
                        }
                    }
                }
            }
            (Space::Product(p), Ctor::Single) => match p.kind {
                ProductKind::Unit => "()".to_string(),
                ProductKind::Tuple if subs.len() == 1 => format!("({},)", subs[0]),
                ProductKind::Tuple => format!("({})", subs.join(", ")),
                ProductKind::Array => format!("[{}]", subs.join(", ")),
                ProductKind::Struct(name) => {
                    let sname = self.interner.resolve(name);
                    let named: Vec<String> = p
                        .names
                        .iter()
                        .zip(&subs)
                        .filter(|(_, s)| s.as_str() != "_")
                        .map(|(n, s)| format!("{}: {s}", self.interner.resolve(*n)))
                        .collect();
                    if named.is_empty() {
                        format!("{sname} {{ .. }}")
                    } else if named.len() == p.names.len() {
                        format!("{sname} {{ {} }}", named.join(", "))
                    } else {
                        format!("{sname} {{ {}, .. }}", named.join(", "))
                    }
                }
            },
            (_, Ctor::Str(k)) => match self.strs.borrow().get(k as usize) {
                Some(b) => format!("{:?}", String::from_utf8_lossy(b)),
                None => "_".to_string(),
            },
            (Space::Slice(_), Ctor::SliceFixed(_)) => format!("[{}]", subs.join(", ")),
            (Space::Slice(_), Ctor::SliceVar(p, _)) => {
                let mut parts: Vec<&str> = subs[..p].iter().map(String::as_str).collect();
                parts.push("..");
                parts.extend(subs[p..].iter().map(String::as_str));
                format!("[{}]", parts.join(", "))
            }
            _ => "_".to_string(),
        }
    }
}

/// The constructor space of enum `name`. A tagged union two of whose variants share a
/// discriminant is opaque: the variants' payloads differ, so no one constructor stands for both.
fn enum_space(name: Symbol, vs: &[EnumVariant]) -> Space {
    let c_style = vs.iter().all(|v| v.payload.is_unit());
    let mut classes: Vec<Vec<usize>> = Vec::new();
    for (i, v) in vs.iter().enumerate() {
        match classes.iter_mut().find(|c| vs[c[0]].disc == v.disc) {
            Some(c) if c_style => c.push(i),
            Some(_) => return Space::Opaque,
            None => classes.push(vec![i]),
        }
    }
    Space::Enum(EnumSpace {
        name,
        variants: vs.to_vec(),
        classes,
    })
}

/// The field types of constructor `c` (its sub-columns).
fn ctor_fields(space: &Space, c: Ctor) -> Vec<Ty> {
    match (space, c) {
        (Space::Enum(es), Ctor::Class(k)) => es.variants[es.classes[k][0]].payload.field_tys(),
        (Space::Product(p), Ctor::Single) => p.fields.clone(),
        (Space::Slice(elem), Ctor::SliceFixed(n)) => vec![elem.clone(); n],
        (Space::Slice(elem), Ctor::SliceVar(p, s)) => vec![elem.clone(); p + s],
        _ => Vec::new(),
    }
}

/// Replace each row whose head is an or-pattern by one row per alternative.
fn expand_or_heads(rows: &[Vec<DPat>]) -> Vec<Vec<DPat>> {
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        if let Some((head, rest)) = r.split_first() {
            push_expanded(head, rest, &mut out);
        }
    }
    out
}

fn push_expanded(head: &DPat, rest: &[DPat], out: &mut Vec<Vec<DPat>>) {
    match head {
        DPat::Or(alts) => {
            for a in alts {
                push_expanded(a, rest, out);
            }
        }
        _ => {
            let mut row = Vec::with_capacity(rest.len() + 1);
            row.push(head.clone());
            row.extend_from_slice(rest);
            out.push(row);
        }
    }
}

/// The row `row` specialized by constructor `c` of arity `arity`: its head replaced by the head's
/// fields (wildcards for a wildcard head), or `None` when the head cannot match `c`. The head is
/// never an or-pattern here (`expand_or_heads` ran, and the vector under test peels its own).
fn specialize(row: &[DPat], c: Ctor, arity: usize) -> Option<Vec<DPat>> {
    let (head, rest) = row.split_first()?;
    let mut out = match (head, c) {
        (DPat::Wild, _) => vec![DPat::Wild; arity],
        (DPat::Bool(b), Ctor::Bool(x)) if *b == x => Vec::new(),
        (DPat::Range(lo, hi), Ctor::Seg(a, b)) if *lo <= a && b <= *hi => Vec::new(),
        (DPat::Variant(k, subs), Ctor::Class(x)) if *k == x => subs.clone(),
        (DPat::Product(subs), Ctor::Single) => subs.clone(),
        (DPat::Str(a), Ctor::Str(b)) if *a == b => Vec::new(),
        (
            DPat::Slice {
                prefix, var: false, ..
            },
            Ctor::SliceFixed(n),
        ) if prefix.len() == n => prefix.clone(),
        (
            DPat::Slice {
                prefix,
                suffix,
                var: true,
            },
            Ctor::SliceFixed(n),
        ) if n >= prefix.len() + suffix.len() => {
            let mut out = prefix.clone();
            out.resize(n - suffix.len(), DPat::Wild);
            out.extend_from_slice(suffix);
            out
        }
        // The split made `p`/`s` at least every row's prefix/suffix length.
        (
            DPat::Slice {
                prefix,
                suffix,
                var: true,
            },
            Ctor::SliceVar(p, s),
        ) if p >= prefix.len() && s >= suffix.len() => {
            let mut out = prefix.clone();
            out.resize(p + s - suffix.len(), DPat::Wild);
            out.extend_from_slice(suffix);
            out
        }
        _ => return None,
    };
    out.extend_from_slice(rest);
    Some(out)
}

/// Split a column whose vector entry is a wildcard against the rows' heads.
fn split_column(space: &Space, heads: &[&DPat]) -> Split {
    match space {
        Space::Bool => {
            if !heads.iter().any(|h| matches!(h, DPat::Bool(_))) {
                return Split::Incomplete(Vec::new());
            }
            let all = [Ctor::Bool(false), Ctor::Bool(true)];
            let missing: Vec<Ctor> = all
                .iter()
                .copied()
                .filter(|c| {
                    !heads
                        .iter()
                        .any(|h| matches!((h, c), (DPat::Bool(x), Ctor::Bool(y)) if x == y))
                })
                .collect();
            if missing.is_empty() {
                Split::Complete(all.to_vec())
            } else {
                Split::Incomplete(missing)
            }
        }
        Space::Int(s) => {
            let ranges = head_ranges(heads);
            if ranges.is_empty() {
                return Split::Incomplete(Vec::new());
            }
            let segs = segments(&int_domain(*s), &ranges);
            let covered =
                |seg: &(i128, i128)| ranges.iter().any(|(l, h)| *l <= seg.0 && seg.1 <= *h);
            let missing = merge_adjacent(segs.iter().filter(|seg| !covered(seg)).copied());
            if missing.is_empty() {
                Split::Complete(segs.into_iter().map(|(a, b)| Ctor::Seg(a, b)).collect())
            } else {
                Split::Incomplete(missing.into_iter().map(|(a, b)| Ctor::Seg(a, b)).collect())
            }
        }
        Space::Enum(es) => {
            let present = |k: usize| {
                heads
                    .iter()
                    .any(|h| matches!(h, DPat::Variant(x, _) if *x == k))
            };
            let all: Vec<usize> = (0..es.classes.len()).collect();
            if !all.iter().any(|&k| present(k)) {
                return Split::Incomplete(Vec::new());
            }
            let missing: Vec<Ctor> = all
                .iter()
                .filter(|&&k| !present(k))
                .map(|&k| Ctor::Class(k))
                .collect();
            if missing.is_empty() {
                Split::Complete(all.into_iter().map(Ctor::Class).collect())
            } else {
                Split::Incomplete(missing)
            }
        }
        Space::Product(_) => Split::Complete(vec![Ctor::Single]),
        Space::Slice(_) => {
            if !heads.iter().any(|h| matches!(h, DPat::Slice { .. })) {
                return Split::Incomplete(Vec::new());
            }
            let all = slice_ctors(heads);
            let missing: Vec<Ctor> = all
                .iter()
                .copied()
                .filter(|&c| !heads.iter().any(|h| slice_covers(h, c)))
                .collect();
            if missing.is_empty() {
                Split::Complete(all)
            } else {
                Split::Incomplete(missing)
            }
        }
        // No set of literals covers every `str`: only the wildcard-headed rows can decide.
        Space::Str | Space::Opaque => Split::Incomplete(Vec::new()),
    }
}

/// The constructors of a slice column split against `heads` (see the module docs): one per length
/// below `L`, then the variable-length constructor for every length `>= L`.
fn slice_ctors(heads: &[&DPat]) -> Vec<Ctor> {
    let (mut p, mut s, mut fixed) = (0, 0, None::<usize>);
    for h in heads {
        if let DPat::Slice {
            prefix,
            suffix,
            var,
        } = h
        {
            if *var {
                p = p.max(prefix.len());
                s = s.max(suffix.len());
            } else {
                fixed = Some(fixed.map_or(prefix.len(), |f| f.max(prefix.len())));
            }
        }
    }
    // Past the longest fixed length, so a variable-length slice never equals a listed one.
    if let Some(f) = fixed {
        if f + 1 > p + s {
            p = f + 1 - s;
        }
    }
    let mut ctors: Vec<Ctor> = (0..p + s).map(Ctor::SliceFixed).collect();
    ctors.push(Ctor::SliceVar(p, s));
    ctors
}

/// Whether the head `h` matches every slice constructor `c` stands for.
fn slice_covers(h: &DPat, c: Ctor) -> bool {
    let DPat::Slice {
        prefix,
        suffix,
        var,
    } = h
    else {
        return false;
    };
    let listed = prefix.len() + suffix.len();
    match (var, c) {
        (false, Ctor::SliceFixed(n)) => prefix.len() == n,
        (true, Ctor::SliceFixed(n)) => n >= listed,
        (true, Ctor::SliceVar(p, s)) => listed <= p + s,
        _ => false,
    }
}

/// The constructors a non-wildcard vector head stands for, split against the rows' heads.
fn head_ctors(space: &Space, head: &DPat, heads: &[&DPat]) -> Vec<Ctor> {
    match head {
        DPat::Bool(b) => vec![Ctor::Bool(*b)],
        DPat::Range(lo, hi) => {
            let Space::Int(s) = space else {
                return Vec::new();
            };
            if lo > hi {
                return Vec::new();
            }
            let mut ranges = head_ranges(heads);
            ranges.push((*lo, *hi));
            segments(&int_domain(*s), &ranges)
                .into_iter()
                .filter(|(a, b)| lo <= a && b <= hi)
                .map(|(a, b)| Ctor::Seg(a, b))
                .collect()
        }
        DPat::Variant(k, _) => vec![Ctor::Class(*k)],
        DPat::Product(_) => vec![Ctor::Single],
        DPat::Str(k) => vec![Ctor::Str(*k)],
        // The split counts this head too, so the variable-length constructor's `p`/`s` cover it.
        DPat::Slice { .. } => {
            let mut all: Vec<&DPat> = heads.to_vec();
            all.push(head);
            slice_ctors(&all)
                .into_iter()
                .filter(|&c| slice_covers(head, c))
                .collect()
        }
        DPat::Wild | DPat::Or(_) => Vec::new(),
    }
}

/// The non-empty ranges among the heads.
fn head_ranges(heads: &[&DPat]) -> Vec<(i128, i128)> {
    heads
        .iter()
        .filter_map(|p| match p {
            DPat::Range(l, h) if l <= h => Some((*l, *h)),
            _ => None,
        })
        .collect()
}

/// Split `domain` (sorted, disjoint inclusive ranges) at every bound of `ranges`, so each resulting
/// segment lies wholly inside or wholly outside every range.
fn segments(domain: &[(i128, i128)], ranges: &[(i128, i128)]) -> Vec<(i128, i128)> {
    let mut cuts: Vec<i128> = Vec::with_capacity(ranges.len() * 2);
    for &(lo, hi) in ranges {
        cuts.push(lo);
        // A range up to a 128-bit type's top value ends with its domain: no cut past it.
        if let Some(next) = hi.checked_add(1) {
            cuts.push(next);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    let mut segs = Vec::new();
    for &(dlo, dhi) in domain {
        let mut start = dlo;
        for &c in cuts.iter().filter(|&&c| c > dlo && c <= dhi) {
            segs.push((start, c - 1));
            start = c;
        }
        segs.push((start, dhi));
    }
    segs
}

/// Join touching segments, for a readable witness (`0..=4` rather than `0..=2 | 3..=4`).
fn merge_adjacent(segs: impl Iterator<Item = (i128, i128)>) -> Vec<(i128, i128)> {
    let mut out: Vec<(i128, i128)> = Vec::new();
    for (a, b) in segs {
        match out.last_mut() {
            Some(last) if last.1.checked_add(1) == Some(a) => last.1 = b,
            _ => out.push((a, b)),
        }
    }
    out
}

/// The values a scalar of type `s` can hold, as sorted inclusive ranges. A `char` can hold any
/// 32-bit value: `x as char` does not check that `x` is a Unicode scalar value.
///
/// A 128-bit type's values are its *keys* ([`wrap`]): an `i128` is itself, and a `u128` is offset
/// by `-2^127` so its range fits `i128` in order. Either way the domain is all of `i128`.
fn int_domain(s: Scalar) -> Vec<(i128, i128)> {
    if s.is_128() {
        return vec![(i128::MIN, i128::MAX)];
    }
    let bits = s.size() as u32 * 8;
    if s.is_signed() {
        vec![(-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)]
    } else {
        vec![(0, (1i128 << bits) - 1)]
    }
}

/// A literal value read as the scalar type `s` it is compared against. MIR integers are signless,
/// so mir_build accepts a literal anywhere in the union of the signed and unsigned ranges of the
/// width and compares bit patterns (`-1` on a `u64` scrutinee is `u64::MAX`); outside that band it
/// rejects the pattern, and so does this (`None`).
fn wrap(v: i128, s: Scalar) -> Option<i128> {
    // A 128-bit literal is already its bit pattern (`lit_value`); a `u128`'s key is that pattern
    // offset by `-2^127` (flipping the top bit), which orders the unsigned values as `i128`s.
    if s.is_128() {
        return Some(if s.is_signed() { v } else { v ^ i128::MIN });
    }
    let bits = s.size() as u32 * 8;
    let modulus = 1i128 << bits;
    if v < -(modulus >> 1) || v >= modulus {
        return None;
    }
    let (lo, hi) = int_domain(s)[0];
    Some(if v > hi {
        v - modulus
    } else if v < lo {
        v + modulus
    } else {
        v
    })
}

/// The code point of a char literal's source text (quotes included), or `None` when it is not one
/// the lexer accepts.
fn char_value(text: &str) -> Option<i128> {
    let inner = text.strip_prefix('\'')?.strip_suffix('\'')?;
    let mut chars = inner.chars();
    let c = chars.next()?;
    let v = if c != '\\' {
        if chars.next().is_some() {
            return None;
        }
        c as u32
    } else {
        match chars.next()? {
            'n' => '\n' as u32,
            'r' => '\r' as u32,
            't' => '\t' as u32,
            '\\' => '\\' as u32,
            '\'' => '\'' as u32,
            '"' => '"' as u32,
            '0' => 0,
            'x' => u32::from_str_radix(chars.as_str(), 16).ok()?,
            'u' => {
                let digits = chars.as_str().strip_prefix('{')?.strip_suffix('}')?;
                u32::from_str_radix(digits, 16).ok()?
            }
            _ => return None,
        }
    };
    Some(i128::from(v))
}

/// A witness integer: a domain bound by name (`i32::MIN`, `u8::MAX`), a `char` as a literal.
fn int_text(v: i128, s: Scalar) -> String {
    if s == Scalar::Char {
        let plain = |c: char| c == ' ' || (c.is_ascii_graphic() && !matches!(c, '\'' | '\\' | '`'));
        return match u32::try_from(v).ok().and_then(char::from_u32) {
            Some(c) if plain(c) => format!("'{c}'"),
            Some(c) => format!("'\\u{{{:x}}}'", c as u32),
            // Not a Unicode scalar value: only an integer pattern can name it.
            None => v.to_string(),
        };
    }
    if s == Scalar::U128 {
        // The key back to its value (`wrap`).
        let u = (v ^ i128::MIN) as u128;
        return if u == u128::MAX {
            "u128::MAX".to_string()
        } else {
            u.to_string()
        };
    }
    let (lo, hi) = int_domain(s)[0];
    if v == lo && lo != 0 {
        format!("{}::MIN", s.name())
    } else if v == hi {
        format!("{}::MAX", s.name())
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use wukong_diag::{Diagnostic, Severity};
    use wukong_span::{Interner, SourceId};

    fn check(src: &str) -> Vec<Diagnostic> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        crate::check(&module, &interner).1
    }

    /// The `E0405` messages for `src`.
    fn nonexhaustive(src: &str) -> Vec<String> {
        check(src)
            .into_iter()
            .filter(|d| d.code == Some("E0405"))
            .map(|d| d.message)
            .collect()
    }

    #[track_caller]
    fn assert_missing(src: &str, witness: &str) {
        let msgs = nonexhaustive(src);
        assert!(
            msgs.iter()
                .any(|m| m.contains(&format!("pattern `{witness}`"))),
            "expected E0405 with witness `{witness}` for {src:?}; got {msgs:?}"
        );
    }

    #[track_caller]
    fn assert_exhaustive(src: &str) {
        let d = check(src);
        assert!(
            d.iter().all(|d| !d.is_error()),
            "expected a clean check for {src:?}; got {d:?}"
        );
    }

    #[test]
    fn payload_sub_patterns_are_checked() {
        // `Add(0, y)` covers only the `Add` values whose first field is 0: the variant-name check
        // counted it as covering all of `Add`, and the missed values hit an `Unreachable`.
        let e = "enum Expr { Num(i32), Add(i32, i32), Nil } ";
        assert_missing(
            &format!(
                "{e}fn f(x: Expr) -> i32 {{ return match x {{ Expr::Add(0, y) => y, \
                 Expr::Num(n) => n, Expr::Nil => 0 }}; }}"
            ),
            "Expr::Add(i32::MIN..=-1 | 1..=i32::MAX, _)",
        );
        assert_exhaustive(&format!(
            "{e}fn f(x: Expr) -> i32 {{ return match x {{ Expr::Add(0, y) => y, \
             Expr::Add(a, b) => a + b, Expr::Num(n) => n, Expr::Nil => 0 }}; }}"
        ));
        // A struct payload: unnamed fields are unconstrained.
        let s = "enum S { P { a: bool, b: i32 }, Q } ";
        assert_missing(
            &format!(
                "{s}fn f(x: S) -> i32 {{ return match x {{ S::P {{ a: true }} => 1, \
                 S::Q => 2 }}; }}"
            ),
            "S::P { a: false, .. }",
        );
    }

    #[test]
    fn tuples_are_checked() {
        assert_missing(
            "fn f(p: bool, q: bool) -> i32 { return match (p, q) { (true, _) => 1, \
             (false, true) => 2 }; }",
            "(false, false)",
        );
        assert_missing(
            "fn f(a: i32, b: i32) -> i32 { return match (a, b) { (0, _) => 1, (1, _) => 2 }; }",
            "(i32::MIN..=-1 | 2..=i32::MAX, _)",
        );
        assert_exhaustive(
            "fn f(a: i32, b: i32) -> i32 { return match (a, b) { (0, 0) => 1, (0, _) => 2, \
             (x, y) => x + y }; }",
        );
    }

    #[test]
    fn integer_ranges_cover_the_whole_domain() {
        // A full `0..=255` on a `u8` is exhaustive (it was a false E0405).
        assert_exhaustive(
            "fn f(x: u8) -> i32 { return match x { 0..=99 => 1, 100..=199 => 2, 200..=255 => 3 }; }",
        );
        assert_exhaustive(
            "fn f(x: i8) -> i32 { return match x { -128..=-1 => 1, 0 => 2, 1..=127 => 3 }; }",
        );
        assert_missing(
            "fn f(x: u8) -> i32 { return match x { 0..=99 => 1, 101..=255 => 3 }; }",
            "100",
        );
        assert_missing(
            "fn f(x: i32) -> i32 { return match x { 0 => 1 }; }",
            "i32::MIN..=-1 | 1..=i32::MAX",
        );
        // An exclusive upper bound stops one short.
        assert_missing(
            "fn f(x: u8) -> i32 { return match x { 0..255 => 1 }; }",
            "u8::MAX",
        );
    }

    #[test]
    fn guarded_arms_never_cover() {
        assert_missing(
            "fn f(b: bool) -> i32 { return match b { true => 1, false if 1 > 0 => 2 }; }",
            "false",
        );
        assert_missing(
            "enum E { A, B } fn f(e: E) -> i32 { return match e { E::A => 1, E::B if 1 > 0 => 2 }; }",
            "E::B",
        );
    }

    #[test]
    fn enums_list_the_missing_variants() {
        let e = "enum C { R, G, B, W, K } ";
        assert_missing(
            &format!("{e}fn f(c: C) -> i32 {{ return match c {{ C::R => 1 }}; }}"),
            "C::G | C::B | C::W",
        );
        let msgs = nonexhaustive(&format!(
            "{e}fn f(c: C) -> i32 {{ return match c {{ C::R => 1 }}; }}"
        ));
        assert!(msgs.iter().any(|m| m.contains("(and 1 more)")), "{msgs:?}");
    }

    /// A slice column has one constructor per length; `..` covers every length from its listed
    /// count up.
    #[test]
    fn slice_lengths_are_checked() {
        assert_exhaustive("fn f(s: []i32) -> i32 { return match s { [] => 0, [h, ..] => h }; }");
        assert_exhaustive(
            "fn f(s: []i32) -> i32 { return match s { [] => 0, [x] => x, [a, b] => a + b,              [a, .., z] => a * z }; }",
        );
        assert_missing(
            "fn f(s: []i32) -> i32 { return match s { [] => 0, [x] => x }; }",
            "[_, _, ..]",
        );
        assert_missing(
            "fn f(s: []i32) -> i32 { return match s { [x, ..] => x }; }",
            "[]",
        );
        // Element patterns are sub-columns: the first and last element of a long slice.
        assert_missing(
            "fn f(s: []bool) -> i32 { return match s { [true, ..] => 1, [false, .., true] => 2,              [] => 3 }; }",
            "[false]",
        );
        assert_exhaustive(
            "fn f(s: []bool) -> i32 { return match s { [true, ..] => 1, [false, .., true] => 2,              [false, .., false] => 4, [false] => 5, [] => 3 }; }",
        );
        // A prefix and a suffix of a slice may name the same element: `[_, true]` is not
        // `[.., true]` on a one-element slice.
        assert_missing(
            "fn f(s: []bool) -> i32 { return match s { [] => 0, [_, .., true] => 1,              [.., false] => 2 }; }",
            "[true]",
        );
    }

    #[test]
    fn unmodeled_patterns_stay_silent() {
        // A non-literal range bound cannot be placed: no verdict, so no E0405 (lowering reports it).
        let d = check("const L: i32 = 4; fn f(x: i32) -> i32 { return match x { 0..L => 1 }; }");
        assert!(d.iter().all(|d| d.code != Some("E0405")), "{d:?}");
    }

    /// The W0020 findings for `src`, which must check clean.
    fn unreachable_arms(src: &str) -> usize {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (sema, sd) = crate::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        crate::lints::run(
            &module,
            &sema,
            &interner,
            &crate::lints::LintOptions::default(),
        )
        .into_iter()
        .filter(|d| d.code == Some("W0020") && d.severity == Severity::Warning)
        .count()
    }

    /// How many `match`es of `src` (which must check clean) sema proves exhaustive.
    fn proven(src: &str) -> usize {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (sema, sd) = crate::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        sema.exhaustive_matches.len()
    }

    #[test]
    fn proven_exhaustive_matches_are_published() {
        // Proven: an enum, a bool, a full integer domain, a catch-all.
        assert_eq!(
            proven(
                "enum C { R, G, B } fn f(c: C) -> i32 { return match c { C::R => 1, C::G => 2, \
                 C::B => 3 }; }"
            ),
            1
        );
        assert_eq!(
            proven("fn f(b: bool) -> i32 { return match b { true => 1, false => 0 }; }"),
            1
        );
        assert_eq!(
            proven(
                "fn f(x: u8) -> i32 { return match x { 0..=99 => 1, 100..=199 => 2, \
                 200..=255 => 3 }; }"
            ),
            1
        );
        assert_eq!(
            proven("fn f(x: i32) -> i32 { return match x { 0 => 1, _ => 2 }; }"),
            1
        );
        // A slice: a zero slice is the empty one, which `[]` models.
        assert_eq!(
            proven("fn f(s: []i32) -> i32 { return match s { [] => 0, [h, ..] => h }; }"),
            1
        );
        // Literal inference re-checks this body (`k` is decided by its use as an `i64`); the
        // final pass's verdict is the one published.
        assert_eq!(
            proven(
                "fn f(x: u8) -> i64 { let k = 7; let y: i64 = k; return match x { 0..=127 => y, \
                 128..=255 => 0 }; }"
            ),
            1
        );
        // Here the first pass types `k` as `i32`, over which `0..=255` is NOT exhaustive; the
        // re-pass types it `u8`. Last write wins, so the verdict is the re-pass's (an AND of the
        // two would lose it).
        assert_eq!(
            proven(
                "fn f() -> i32 { let k = 7; let y: u8 = k; return match k { 0..=255 => 1 } + \
                 (y as i32); }"
            ),
            1
        );
    }

    #[test]
    fn unmodeled_zero_values_and_generic_scrutinees_are_not_proven() {
        // `let e: E;` holds discriminant 0, which no variant has.
        let e = "enum E { A = 1, B = 2 } ";
        assert_eq!(
            proven(&format!(
                "{e}fn f(x: E) -> i32 {{ return match x {{ E::A => 1, E::B => 2 }}; }}"
            )),
            0
        );
        // Nor when a struct holds one.
        assert_eq!(
            proven(&format!(
                "{e}struct S {{ e: E, k: i32 }} fn f(s: S) -> i32 {{ return match s {{ \
                 S {{ e: E::A, .. }} => 1, S {{ e: E::B, .. }} => 2 }}; }}"
            )),
            0
        );
        // A scrutinee naming a type parameter is not concrete.
        assert_eq!(
            proven(
                "fn f<T>(x: T, b: bool) -> i32 { return match (x, b) { (_, true) => 1, \
                 (_, false) => 0 }; }"
            ),
            0
        );
    }

    #[test]
    fn zero_value_modeling() {
        let src = "enum Z { A, B } enum N { A = 1, B } enum D { X(N), Y } enum P { Y, X(N) } \
                   struct S { z: Z, n: N } fn main() -> i32 { return 0; }";
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (sema, _) = crate::check(&module, &interner);
        let named = |n: &str| wukong_types::Ty::Named(interner.get(n).expect("interned"));
        let m = |t: &wukong_types::Ty| super::zero_value_modeled(&sema.defs, t);
        assert!(m(&named("Z")));
        assert!(!m(&named("N")));
        // `D`'s zero is `D::X` of a zero `N`; `P`'s zero is the unit `P::Y`.
        assert!(!m(&named("D")));
        assert!(m(&named("P")));
        assert!(!m(&named("S")));
        assert!(m(&wukong_types::Ty::Tuple(vec![
            named("Z"),
            wukong_types::Ty::Scalar(wukong_types::Scalar::F32)
        ])));
        assert!(!m(&wukong_types::Ty::Array {
            elem: Box::new(named("N")),
            len: 4,
            shared: false
        }));
        assert!(!m(&wukong_types::Ty::Unknown));
    }

    #[test]
    fn unreachable_arms_are_found() {
        assert_eq!(
            unreachable_arms(
                "fn f(x: i32) -> i32 { return match x { 0..=9 => 1, 5 => 2, _ => 0 }; }"
            ),
            1
        );
        assert_eq!(
            unreachable_arms(
                "fn f(b: bool) -> i32 { return match b { true => 1, false => 0, _ => 2 }; }"
            ),
            1
        );
        // A guarded arm covers nothing, so the arm after it is still reachable.
        assert_eq!(
            unreachable_arms(
                "fn f(x: i32) -> i32 { return match x { n if n > 0 => 1, 3 => 2, _ => 0 }; }"
            ),
            0
        );
        assert_eq!(
            unreachable_arms(
                "fn f(a: i32, b: i32) -> i32 { return match (a, b) { (_, _) => 1, (0, 0) => 2 }; }"
            ),
            1
        );
        // `[..]` covers every slice; `[_]` after it never fires, `[x, ..]` after `[]` does.
        assert_eq!(
            unreachable_arms("fn f(s: []i32) -> i32 { return match s { [..] => 1, [_] => 2 }; }"),
            1
        );
        assert_eq!(
            unreachable_arms("fn f(s: []i32) -> i32 { return match s { [] => 1, [x, ..] => x }; }"),
            0
        );
    }
}
