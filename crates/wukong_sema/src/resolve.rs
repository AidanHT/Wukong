//! **Name resolution diagnostics** — a name that resolves to nothing is reported here, with a stable
//! resolution code and, where a close name exists, a structured "did you mean" suggestion.
//!
//! Before this module an unresolved callee, field or method typed as the lenient [`Ty::Unknown`] and
//! surfaced in lowering as the codegen limit `C0001` ("`call` is not yet supported by codegen"), and an
//! undeclared type name was accepted as a nominal type and lowered as an `i32` slot. A typo is the
//! programmer's mistake, not a missing compiler feature, and an editor (human or AI) needs a code it
//! can act on:
//!
//! | code | what does not resolve | rustc |
//! |---|---|---|
//! | `E0320` | a called name: no function, function-typed parameter or builtin (`matmull(a, b)`) | E0425 |
//! | `E0321` | a field: `p.xx` on a struct with no `xx`, or any field of a non-struct value | E0609 |
//! | `E0322` | a method of a builtin type: `s.dim(0)` on a slice, `x.sqrt()` on an `f32` | E0599 |
//! | `E0323` | a call of something that is not a function: a local, a constant, a type | E0618 |
//! | `E0324` | a type name: not a scalar, a declared struct/enum or a generic in scope | E0412 |
//!
//! Methods of structs and enums resolve in `methods.rs` (`E0315` there); this module covers the
//! builtin types, whose whole method surface is [`builtin_methods_of`]. The builtin *functions* are
//! registered in `builtins.rs` ([`crate::is_builtin_callee`]), which `wukong_mir_build` reads too.
//!
//! What stays lenient, by design: a receiver or base whose type is `Unknown` (an unmodelled builtin's
//! result) or `Error` (already reported), and one whose type is a generic parameter — a generic body is
//! checked once against an opaque `T`, and each instantiation is lowered with the concrete type.

use wukong_ast::{Expr, ExprKind, Ident, Path, TypeExpr, TypeKind};
use wukong_diag::{did_you_mean, Applicability, Diagnostic};
use wukong_span::{FxHashSet as HashSet, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{DefKind, Sema};

/// Resolution state kept across the body pass.
#[derive(Default)]
pub(crate) struct ResolveState {
    /// Type expressions already reported as unknown (`E0324`). The body pass can lower one type
    /// expression twice — a cast inside a method receiver is typed by the method lookup and again by
    /// the call fallthrough — and the type is reported once.
    reported_types: HashSet<Span>,
}

/// The kind of a builtin (non-nominal) receiver type, for the method surface.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Recv {
    Scalar,
    Vector,
    Array,
    Slice,
    Tensor,
    Tuple,
    Pointer,
    Other,
}

fn recv_of(t: &Ty) -> Recv {
    match t {
        Ty::Scalar(_) => Recv::Scalar,
        Ty::Vector { .. } => Recv::Vector,
        Ty::Array { .. } => Recv::Array,
        Ty::Slice(_) => Recv::Slice,
        Ty::Tensor { .. } => Recv::Tensor,
        Ty::Tuple(_) => Recv::Tuple,
        Ty::Ptr { .. } | Ty::Ref { .. } => Recv::Pointer,
        _ => Recv::Other,
    }
}

/// The methods of a builtin receiver type, sorted. `len` and `dim` lower
/// (`tensor::type_extent_method`); the rest are the explicit-SIMD surface the language guide
/// documents as type-checked but not lowered — a tensor's `row_ptr`, a pointer's `offset`, a vector's
/// `store`/`fma` — which stay the codegen limit `C0001` until they lower.
fn builtin_methods_of(r: Recv) -> &'static [&'static str] {
    match r {
        Recv::Array => &["dim", "len"],
        Recv::Slice => &["len"],
        Recv::Tensor => &["dim", "len", "row_ptr"],
        Recv::Pointer => &["offset"],
        Recv::Vector => &["fma", "store"],
        Recv::Scalar | Recv::Tuple | Recv::Other => &[],
    }
}

/// The scalar type names, for the `E0324` suggestion.
const SCALAR_NAMES: &[&str] = &[
    "bf16", "bool", "char", "f16", "f32", "f64", "i16", "i32", "i64", "i8", "isize", "u16", "u32",
    "u64", "u8", "usize",
];

/// The Wukong spelling of a type name other languages use (`float` is `f32`), for `E0324`.
fn foreign_type_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "float" | "Float" => "f32",
        "double" | "Double" => "f64",
        "half" => "f16",
        "int" | "Int" => "i32",
        "long" => "i64",
        "short" => "i16",
        "byte" | "uchar" => "u8",
        "uint" | "unsigned" => "u32",
        "ulong" => "u64",
        "size_t" => "usize",
        "ssize_t" | "ptrdiff_t" => "isize",
        "boolean" | "Bool" => "bool",
        _ => return None,
    })
}

/// `a, b and c` for a note listing names.
fn list_names(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match quoted.len() {
        0 => String::new(),
        1 => quoted[0].clone(),
        n => format!("{} and {}", quoted[..n - 1].join(", "), quoted[n - 1]),
    }
}

/// What a name that is not a function names, for `E0323`.
enum NotFn {
    /// A local (or a parameter) of this type.
    Local(Ty),
    /// A generic parameter of the enclosing item.
    Generic,
    /// A top-level `const` of this type.
    Const(Ty),
    Struct,
    Enum,
    /// A builtin type name (`f32`, `Tensor`, `f32x8`).
    Type,
}

impl Sema<'_> {
    /// `E0320` / `E0323`: the single-name callee `name` (spanning `span`) is not a function — no
    /// definition, no function-typed parameter and no builtin of any family typed it. Returns the
    /// call's type (`Error`, so nothing cascades from it).
    pub(crate) fn report_unresolved_callee(&mut self, name: Symbol, span: Span) -> Ty {
        let nm = self.sym_str(name).to_string();
        if let Some(what) = self.non_callable(name) {
            let (headline, label, help) = match what {
                // An earlier error already explains this name.
                NotFn::Local(Ty::Error) | NotFn::Const(Ty::Error) => return Ty::Error,
                NotFn::Local(Ty::Unknown) => (
                    format!("expected function, found local variable `{nm}`"),
                    format!("`{nm}` is a local variable, not a function"),
                    None,
                ),
                NotFn::Local(t) => {
                    let ts = t.display(self.interner);
                    (
                        format!("expected function, found `{ts}`"),
                        format!("`{nm}` is a local of type `{ts}`, not a function"),
                        Some(
                            "a `(` that begins a line continues the expression above it as a call; \
                             if a new statement starts there, end the previous one with `;`"
                                .to_string(),
                        ),
                    )
                }
                NotFn::Generic => (
                    format!("expected function, found generic parameter `{nm}`"),
                    format!("`{nm}` is a generic parameter, not a function"),
                    None,
                ),
                NotFn::Const(t) => {
                    let ts = t.display(self.interner);
                    (
                        format!("expected function, found constant `{nm}`"),
                        format!("`{nm}` is a constant of type `{ts}`, not a function"),
                        None,
                    )
                }
                NotFn::Struct => (
                    format!("expected function, found struct `{nm}`"),
                    format!("`{nm}` is a struct, not a function"),
                    Some(format!(
                        "build a value with a struct literal: `{nm} {{ field: value, .. }}`"
                    )),
                ),
                NotFn::Enum => (
                    format!("expected function, found enum `{nm}`"),
                    format!("`{nm}` is an enum, not a function"),
                    Some(format!(
                        "construct one of its variants: `{nm}::Variant` or `{nm}::Variant(..)`"
                    )),
                ),
                NotFn::Type => (
                    format!("expected function, found type `{nm}`"),
                    format!("`{nm}` is a type, not a function"),
                    Scalar::from_name(&nm)
                        .map(|_| format!("convert a value with a cast: `x as {nm}`")),
                ),
            };
            let mut d = Diagnostic::error(headline)
                .with_code("E0323")
                .primary(span, label);
            if let Some(h) = help {
                d = d.help(h);
            }
            self.diags.push(d);
            return Ty::Error;
        }
        let mut d = Diagnostic::error(format!("cannot find function `{nm}` in this scope"))
            .with_code("E0320")
            .primary(span, "not found in this scope");
        let candidates = self.callable_names();
        if let Some(c) = did_you_mean(&nm, candidates.iter().map(String::as_str)) {
            d = d.suggest(span, "a function with a similar name exists", c);
        }
        if let Some((h, path)) = self.callee_elsewhere(&nm) {
            d = d.help(h);
            if let Some(path) = path {
                d = d.suggest(span, "call the associated function through its type", path);
            }
        }
        self.diags.push(d);
        Ty::Error
    }

    /// What `name` names when it is a value or a type rather than a function, or `None` when it
    /// names nothing at all.
    fn non_callable(&self, name: Symbol) -> Option<NotFn> {
        if let Some(t) = self.lookup_local(name) {
            return Some(NotFn::Local(t));
        }
        if self.generics.contains(&name) {
            return Some(NotFn::Generic);
        }
        match self.defs.lookup(name).map(|d| &d.kind) {
            Some(DefKind::Const(t)) => return Some(NotFn::Const(t.clone())),
            Some(DefKind::Struct(_)) => return Some(NotFn::Struct),
            Some(DefKind::Enum(_)) => return Some(NotFn::Enum),
            Some(DefKind::Fn(_)) | None => {}
        }
        let s = self.sym_str(name);
        if Scalar::from_name(s).is_some() || s == "Tensor" || crate::is_vector_name(s) {
            return Some(NotFn::Type);
        }
        None
    }

    /// Every name a call could mean, sorted: the program's functions (not the `Type.method` /
    /// `module.item` spellings no call writes), the function-typed parameters in scope, and every
    /// builtin.
    fn callable_names(&self) -> Vec<String> {
        let mut names = crate::builtins::builtin_callee_names();
        for d in &self.defs.defs {
            if matches!(d.kind, DefKind::Fn(_)) {
                let n = self.sym_str(d.name);
                if !n.contains('.') {
                    names.push(n.to_string());
                }
            }
        }
        for scope in &self.scopes {
            for (n, t) in scope {
                if matches!(t, Ty::Fn { .. }) {
                    names.push(self.sym_str(*n).to_string());
                }
            }
        }
        names.sort();
        names.dedup();
        names
    }

    /// A function named `nm` that exists but is not in scope under that name: a method or associated
    /// function of a type (`Vec2.norm`), or an item of a module this file does not import
    /// (`lib.geom.norm`, the loader's spelling of `norm` in module `lib.geom`). Returns the help
    /// text and, for a type's function, the path that names it (`Vec2::norm`).
    fn callee_elsewhere(&self, nm: &str) -> Option<(String, Option<String>)> {
        let suffix = format!(".{nm}");
        let mut owners: Vec<&str> = self
            .defs
            .defs
            .iter()
            .filter(|d| matches!(d.kind, DefKind::Fn(_)))
            .filter_map(|d| self.sym_str(d.name).strip_suffix(suffix.as_str()))
            .collect();
        owners.sort_unstable();
        let owner = *owners.first()?;
        let is_type = self
            .interner
            .get(owner)
            .is_some_and(|s| self.is_declared_named_ty(s));
        Some(if is_type {
            (
                format!(
                    "`{owner}` has a function named `{nm}`: call it as `{owner}::{nm}(..)`, or as \
                     a method `value.{nm}(..)` when it takes `self`"
                ),
                Some(format!("{owner}::{nm}")),
            )
        } else {
            (
                format!(
                    "a function `{nm}` is defined in module `{owner}`, which this file does not \
                     import: add `import {owner};` (the function must be declared `pub`)"
                ),
                None,
            )
        })
    }

    /// Whether `base` (the left of `base::name` / `base.name`) names a TYPE used as a namespace — a
    /// generic parameter, a struct or enum, or a builtin type name — rather than a value. A local
    /// of the same name is a value. `resolve_value` types a generic as its `usize` value, so this is
    /// what keeps `T::m(..)` from being judged as a method of `usize`.
    pub(crate) fn is_type_namespace(&self, base: &Expr) -> bool {
        let ExprKind::Path(p) = &base.kind else {
            return false;
        };
        if !p.is_single() {
            return false;
        }
        let name = p.first().sym;
        if self.lookup_local(name).is_some() {
            return false;
        }
        matches!(
            self.non_callable(name),
            Some(NotFn::Generic | NotFn::Struct | NotFn::Enum | NotFn::Type)
        )
    }

    /// `base::name` in value position where `base` names a type rather than a value. The `Field` arm
    /// has already resolved an enum's variants, so an enum here has no variant `name`: `E0301`, the
    /// code the call form `E::V(..)` and the pattern form already report, with a suggestion. Any other
    /// namespace (`f32::MAX`, `T::X`) stays the lenient unmodelled path. Returns the access's type.
    pub(crate) fn report_namespace_member(&mut self, base: &Expr, name: Ident) -> Ty {
        let ExprKind::Path(p) = &base.kind else {
            return Ty::Unknown;
        };
        let ename = p.first().sym;
        let variants: Vec<String> = match self.defs.lookup(ename).map(|d| &d.kind) {
            Some(DefKind::Enum(vs)) => vs
                .iter()
                .map(|v| self.sym_str(v.name).to_string())
                .collect(),
            _ => return Ty::Unknown,
        };
        // A function of an `impl` block named without a call is not a variant either, but it is
        // the method machinery's to report (`methods.rs`).
        if self
            .methods
            .resolution
            .table
            .contains_key(&(ename, name.sym))
        {
            return Ty::Unknown;
        }
        let want = self.sym_str(name.sym).to_string();
        let qual = format!("{}::{want}", self.sym_str(ename));
        let mut d = Diagnostic::error(format!("no variant `{qual}`"))
            .with_code("E0301")
            .primary(name.span, "not a variant of this enum");
        if let Some(c) = did_you_mean(&want, variants.iter().map(String::as_str)) {
            d = d.suggest(name.span, "a variant with a similar name exists", c);
        }
        self.diags.push(d);
        Ty::Error
    }

    /// `E0321`: `.name` read from a value of type `base_ty` that has no field `name` — the struct
    /// lookup in the `Field` arm found nothing. Returns the access's type: `Unknown` where the base's
    /// fields are not known (`Unknown`, `Error`, a generic or an undeclared nominal type), `Error`
    /// once reported.
    pub(crate) fn report_missing_field(&mut self, base_ty: &Ty, name: Ident) -> Ty {
        let field = self.sym_str(name.sym).to_string();
        // A field access auto-derefs one pointer or reference to a struct.
        let inner = match base_ty {
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. }
                if matches!(pointee.as_ref(), Ty::Named(_)) =>
            {
                pointee.as_ref()
            }
            t => t,
        };
        let named = match inner {
            Ty::Unknown | Ty::Error => return Ty::Unknown,
            Ty::Named(n) => match self.defs.lookup(*n).map(|def| def.kind.clone()) {
                Some(kind @ (DefKind::Struct(_) | DefKind::Enum(_))) => Some((*n, kind)),
                // A generic type parameter, or an undeclared name (already `E0324`).
                _ => return Ty::Unknown,
            },
            _ => None,
        };
        let shown = inner.display(self.interner);
        let mut d = Diagnostic::error(format!("no field `{field}` on type `{shown}`"))
            .with_code("E0321")
            .primary(name.span, "unknown field");
        match (named, inner) {
            (Some((n, kind)), _) => match kind {
                DefKind::Struct(fields) => {
                    let names: Vec<String> = fields
                        .iter()
                        .map(|(f, _)| self.sym_str(*f).to_string())
                        .collect();
                    if let Some(c) = did_you_mean(&field, names.iter().map(String::as_str)) {
                        d = d.suggest(name.span, "a field with a similar name exists", c);
                    }
                    d = if names.is_empty() {
                        d.note(format!("`{shown}` has no fields"))
                    } else {
                        d.note(format!(
                            "the fields of `{shown}` are {}",
                            list_names(&names)
                        ))
                    };
                    if self.methods.resolution.table.contains_key(&(n, name.sym)) {
                        d = d
                            .help(format!(
                                "`{field}` is a method of `{shown}`: call it, `.{field}(..)`"
                            ))
                            .suggest_with(
                                name.span.shrink_to_hi(),
                                "call the method",
                                "()",
                                Applicability::HasPlaceholders,
                            );
                    }
                }
                _ => {
                    d = d.note(format!(
                        "an enum value has no fields; read a variant's payload with `match`, \
                         or name a variant as `{shown}::Variant`"
                    ));
                }
            },
            (None, Ty::Tuple(items)) => {
                d = d.help(format!(
                    "a tuple's elements are numbered: `.0`{}",
                    if items.len() > 1 {
                        format!(" to `.{}`", items.len() - 1)
                    } else {
                        String::new()
                    }
                ));
            }
            (None, other) => {
                let methods = builtin_methods_of(recv_of(other));
                if methods.contains(&field.as_str()) {
                    d = d
                        .help(format!("`{field}` is a method: call it, `.{field}(..)`"))
                        .suggest_with(
                            name.span.shrink_to_hi(),
                            "call the method",
                            "()",
                            Applicability::HasPlaceholders,
                        );
                } else {
                    d = d.note(format!(
                        "only a struct has named fields; `{shown}` is a builtin type"
                    ));
                }
            }
        }
        self.diags.push(d);
        Ty::Error
    }

    /// `E0321` for `.index` on a value of type `base_ty` that is not a tuple. Returns the access's
    /// type: `Unknown` for a base whose type is not known (`Unknown`, `Error`, a generic, an
    /// undeclared name), `Error` once reported.
    pub(crate) fn report_numbered_field(&mut self, base_ty: &Ty, index: u32, span: Span) -> Ty {
        let kind = match base_ty {
            Ty::Unknown | Ty::Error => return Ty::Unknown,
            Ty::Named(n) => match self.defs.lookup(*n).map(|def| &def.kind) {
                Some(DefKind::Struct(_)) => "a struct's fields are named, not numbered: `.name`",
                Some(DefKind::Enum(_)) => {
                    "an enum value has no fields; read a variant's payload with `match`"
                }
                _ => return Ty::Unknown,
            },
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. }
                if matches!(pointee.as_ref(), Ty::Tuple(_)) =>
            {
                "a numbered field is read from the tuple itself: dereference the pointer first, \
                 `(*p).0`"
            }
            Ty::Array { .. } | Ty::Slice(_) | Ty::Tensor { .. } | Ty::Ptr { .. } => {
                "an element is read by indexing: `[i]`"
            }
            _ => "only a tuple has numbered fields",
        };
        let shown = base_ty.display(self.interner);
        self.diags.push(
            Diagnostic::error(format!("no field `{index}` on type `{shown}`"))
                .with_code("E0321")
                .primary(span, "unknown field")
                .note(kind),
        );
        Ty::Error
    }

    /// `E0324` for a struct literal `Name { .. }` whose `Name` is no struct of this program.
    /// Returns the literal's type (`Error`).
    pub(crate) fn report_unknown_struct(&mut self, path: &Path) -> Ty {
        let Some(last) = path.segments.last().map(|s| s.sym) else {
            return Ty::Error;
        };
        let shown = path
            .segments
            .iter()
            .map(|s| self.sym_str(s.sym))
            .collect::<Vec<_>>()
            .join("::");
        let d = match self.defs.lookup(last).map(|def| &def.kind) {
            Some(DefKind::Enum(_)) => {
                Diagnostic::error(format!("expected struct, found enum `{shown}`"))
                    .with_code("E0324")
                    .primary(path.span, "an enum is not a struct")
                    .help(format!(
                        "a variant with named fields is built as `{shown}::Variant {{ .. }}`"
                    ))
                    .suggest_with(
                        path.span.shrink_to_hi(),
                        "name the variant",
                        "::Variant",
                        Applicability::HasPlaceholders,
                    )
            }
            Some(_) => Diagnostic::error(format!("expected struct, found `{shown}`"))
                .with_code("E0324")
                .primary(path.span, "not a struct"),
            None if self.generics.contains(&last) => {
                Diagnostic::error(format!("expected struct, found type parameter `{shown}`"))
                    .with_code("E0324")
                    .primary(path.span, "a type parameter has no literal")
            }
            None => {
                let mut d =
                    Diagnostic::error(format!("cannot find struct `{shown}` in this scope"))
                        .with_code("E0324")
                        .primary(path.span, "not found in this scope");
                let structs: Vec<String> = {
                    let mut v: Vec<String> = self
                        .defs
                        .defs
                        .iter()
                        .filter(|def| matches!(def.kind, DefKind::Struct(_)))
                        .map(|def| self.sym_str(def.name).to_string())
                        .filter(|n| !n.contains('.'))
                        .collect();
                    v.sort();
                    v
                };
                let want = self.sym_str(last).to_string();
                if let Some(c) = did_you_mean(&want, structs.iter().map(String::as_str)) {
                    d = d.suggest(path.span, "a struct with a similar name exists", c);
                }
                d
            }
        };
        self.diags.push(d);
        Ty::Error
    }

    /// The method call `recv.name(args)` whose receiver type is `recv_ty` and which neither
    /// `methods.rs` (struct and enum methods) nor `tensor::type_extent_method` (`len`/`dim`) typed.
    /// A method of a builtin type outside [`builtin_methods_of`] is `E0322`; a struct FIELD called
    /// like a method is `E0323`. Returns the call's type: `Unknown` on the lenient paths (an
    /// unmodelled receiver, a generic one, or a documented-but-unlowered builtin method), `Error`
    /// once reported.
    pub(crate) fn check_builtin_method(
        &mut self,
        recv_ty: &Ty,
        name: Ident,
        args: &[Expr],
        span: Span,
    ) -> Ty {
        let method = self.sym_str(name.sym).to_string();
        let inner = match recv_ty {
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. }
                if matches!(pointee.as_ref(), Ty::Named(_)) =>
            {
                pointee.as_ref()
            }
            t => t,
        };
        match inner {
            Ty::Unknown | Ty::Error => return Ty::Unknown,
            Ty::Named(n) => {
                // `methods.rs` reports a struct or enum with no such method (`E0315`) and leaves
                // exactly one case here: a struct that has a FIELD of that name.
                let field_ty = match self.defs.lookup(*n).map(|d| &d.kind) {
                    Some(DefKind::Struct(fields)) => fields
                        .iter()
                        .find(|(f, _)| *f == name.sym)
                        .map(|(_, t)| t.clone()),
                    _ => None,
                };
                let Some(fty) = field_ty else {
                    return Ty::Unknown;
                };
                let (ts, sn) = (fty.display(self.interner), self.sym_str(*n).to_string());
                // The call's argument list, `(..)` after the field name, is what the fix deletes.
                let parens = Span::new(name.span.source, name.span.hi, span.hi.max(name.span.hi));
                self.diags.push(
                    Diagnostic::error(format!("expected function, found `{ts}`"))
                        .with_code("E0323")
                        .primary(
                            name.span,
                            format!("`{method}` is a field of `{sn}`, not a method"),
                        )
                        .help(format!(
                            "read the field without parentheses, `.{method}`; a struct has the \
                             methods its `impl {sn}` blocks define"
                        ))
                        .suggest_with(
                            parens,
                            "read the field instead of calling it",
                            "",
                            Applicability::MaybeIncorrect,
                        ),
                );
                return Ty::Error;
            }
            _ => {}
        }
        let methods = builtin_methods_of(recv_of(recv_ty));
        if methods.contains(&method.as_str()) {
            // `len` on a slice with arguments is the only lowered method `type_extent_method`
            // declines; anything else here is the unlowered SIMD surface, typed leniently.
            if method == "len" && !args.is_empty() {
                self.diags.push(
                    Diagnostic::error(format!(
                        "`.len()` takes no arguments, but {} were supplied",
                        args.len()
                    ))
                    .with_code("E0503")
                    .primary(span, "`len` takes no arguments"),
                );
                return Ty::Error;
            }
            return Ty::Unknown;
        }
        let shown = recv_ty.display(self.interner);
        let mut d = Diagnostic::error(format!(
            "no method named `{method}` found for type `{shown}`"
        ))
        .with_code("E0322")
        .primary(name.span, "method not found");
        if let Some(c) = did_you_mean(&method, methods.iter().copied()) {
            d = d.suggest(name.span, "a method with a similar name exists", c);
        }
        if crate::builtins::is_builtin_callee(&method) {
            d = d.help(format!(
                "`{method}` is a builtin function, not a method: pass the value as an argument, \
                 `{method}(..)`"
            ));
        } else if methods.is_empty() {
            d = d.note(format!("`{shown}` has no methods"));
        } else {
            let names: Vec<String> = methods.iter().map(|m| m.to_string()).collect();
            d = d.note(format!(
                "the methods of `{shown}` are {}",
                list_names(&names)
            ));
        }
        self.diags.push(d);
        Ty::Error
    }

    /// `E0323` for a callee that is an expression rather than a name — `a[i](..)`, `p.x(..)` on a
    /// field read by the tuple/field paths, a call's result called again — whose type is known and is
    /// not a function. Returns the call's type.
    pub(crate) fn check_value_callee(&mut self, callee: &Expr, t: &Ty) -> Ty {
        if matches!(t, Ty::Unknown | Ty::Error | Ty::Fn { .. }) {
            return Ty::Unknown;
        }
        let ts = t.display(self.interner);
        self.diags.push(
            Diagnostic::error(format!("expected function, found `{ts}`"))
                .with_code("E0323")
                .primary(
                    callee.span,
                    format!("this is a value of type `{ts}`, not a function"),
                )
                .help(
                    "a `(` that begins a line continues the expression above it as a call; if a \
                     new statement starts there, end the previous one with `;`",
                ),
        );
        Ty::Error
    }

    /// `E0324`: the type path `p` (spanning `span`), whose last segment `name` is not a scalar, names
    /// no declared struct or enum and no generic parameter in scope. Called by `lower_type` once
    /// every item is registered (`checking_bodies`).
    pub(crate) fn check_type_name(&mut self, p: &Path, name: Symbol, span: Span) {
        if self.generics.contains(&name) || self.is_declared_named_ty(name) {
            return;
        }
        if !self.resolve.reported_types.insert(span) {
            return;
        }
        let shown = p
            .segments
            .iter()
            .map(|s| self.sym_str(s.sym))
            .collect::<Vec<_>>()
            .join("::");
        if self.methods.is_trait(name) {
            self.diags.push(
                Diagnostic::error(format!("expected a type, found trait `{shown}`"))
                    .with_code("E0324")
                    .primary(span, "a trait is not a type")
                    .help(format!(
                        "take a generic parameter bounded by the trait instead: \
                         `fn f<T: {shown}>(x: T)`"
                    )),
            );
            return;
        }
        let mut d = Diagnostic::error(format!("cannot find type `{shown}` in this scope"))
            .with_code("E0324")
            .primary(span, "not found in this scope");
        let last = self.sym_str(name).to_string();
        if let Some(w) = foreign_type_name(&last) {
            d = d.suggest(span, format!("Wukong spells this type `{w}`"), w);
        } else if matches!(last.as_str(), "str" | "String" | "string") {
            d = d.help("a string literal has type `*u8` (a NUL-terminated byte pointer)");
        } else {
            let candidates = self.type_names();
            if let Some(c) = did_you_mean(&last, candidates.iter().map(String::as_str)) {
                d = d.suggest(span, "a type with a similar name exists", c);
            }
        }
        if p.segments.len() == 1 {
            d = d.note(
                "a type is a scalar (`f32`, `i64`, …), `Tensor[..]`, `vec[..]`, a struct or enum \
                 declared in this program, or a generic parameter of the enclosing item",
            );
        }
        self.diags.push(d);
    }

    /// [`Self::check_type_name`] over every path inside the type expression `t`, for a type the
    /// checker does not otherwise lower (the operand of `sizeof`/`alignof`).
    pub(crate) fn check_type_expr_names(&mut self, t: &TypeExpr) {
        match &t.kind {
            TypeKind::Path(p) => {
                if let Some(last) = p.segments.last().map(|s| s.sym) {
                    if Scalar::from_name(self.sym_str(last)).is_none() {
                        self.check_type_name(p, last, t.span);
                    }
                }
            }
            TypeKind::Pointer { pointee, .. } | TypeKind::Ref { pointee, .. } => {
                self.check_type_expr_names(pointee)
            }
            TypeKind::Slice(elem)
            | TypeKind::Array { elem, .. }
            | TypeKind::Vector { elem, .. }
            | TypeKind::Tensor { elem, .. } => self.check_type_expr_names(elem),
            TypeKind::Tuple(items) => {
                for it in items {
                    self.check_type_expr_names(it);
                }
            }
            TypeKind::Fn { params, ret } => {
                for p in params {
                    self.check_type_expr_names(p);
                }
                if let Some(r) = ret {
                    self.check_type_expr_names(r);
                }
            }
            _ => {}
        }
    }

    /// Every type name a `TypeKind::Path` could mean here, sorted: the scalars, `Tensor`, the
    /// program's structs and enums, and the generics in scope.
    fn type_names(&self) -> Vec<String> {
        let mut names: Vec<String> = SCALAR_NAMES.iter().map(|s| s.to_string()).collect();
        names.push("Tensor".to_string());
        for d in &self.defs.defs {
            if matches!(d.kind, DefKind::Struct(_) | DefKind::Enum(_)) {
                let n = self.sym_str(d.name);
                if !n.contains('.') {
                    names.push(n.to_string());
                }
            }
        }
        for g in &self.generics {
            names.push(self.sym_str(*g).to_string());
        }
        names.sort();
        names.dedup();
        names
    }
}

#[cfg(test)]
mod tests {
    use wukong_diag::Diagnostic;
    use wukong_span::{Interner, SourceId};

    /// Every diagnostic `check` reports for `src` (which must parse).
    fn diags(src: &str) -> Vec<Diagnostic> {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        crate::check(&module, &interner).1
    }

    fn codes(src: &str) -> Vec<&'static str> {
        diags(src).iter().filter_map(|d| d.code).collect()
    }

    /// The driver expands and removes every alias before sema. A caller that skips that pass
    /// (a direct `check`) gets an ignored alias item and the ordinary E0324 at each use, never a
    /// silent `i32` slot.
    #[test]
    fn an_unexpanded_alias_is_unknown_to_sema() {
        assert_eq!(
            codes("type Real = f32;\nfn f(x: Real) -> f32 { return 1.0; }\n"),
            vec!["E0324"]
        );
        assert_eq!(
            codes("type Real = f32;\nfn f(x: f32) -> f32 { return x; }\n"),
            Vec::<&str>::new()
        );
        assert!(
            codes("type M<N> = [f32; N];\nfn f(x: M<2>) -> f32 { return 1.0; }\n")
                .contains(&"E0324")
        );
    }

    /// The single diagnostic `src` reports, which must carry `code`.
    #[track_caller]
    fn only(src: &str, code: &str) -> Diagnostic {
        let ds = diags(src);
        assert_eq!(
            ds.len(),
            1,
            "expected exactly one diagnostic for {src:?}: {ds:?}"
        );
        assert_eq!(ds[0].code, Some(code), "{ds:?}");
        ds.into_iter().next().expect("one diagnostic")
    }

    /// The first diagnostic carrying `code`.
    #[track_caller]
    fn first(src: &str, code: &str) -> Diagnostic {
        let ds = diags(src);
        let found = ds.iter().find(|d| d.code == Some(code)).cloned();
        found.unwrap_or_else(|| panic!("no {code} for {src:?}: {ds:?}"))
    }

    fn replacement(d: &Diagnostic) -> Option<&str> {
        d.suggestions.first().map(|s| s.replacement.as_str())
    }

    fn notes(d: &Diagnostic) -> String {
        d.notes
            .iter()
            .map(|(_, m)| m.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    }

    #[test]
    fn an_unknown_function_suggests_the_closest_name() {
        let d = only(
            "fn square(x: f32) -> f32 { return x * x; } \
             fn main() -> i32 { return sqare(3.0) as i32; }",
            "E0320",
        );
        assert_eq!(d.message, "cannot find function `sqare` in this scope");
        assert_eq!(replacement(&d), Some("square"));
        // A builtin is a candidate too.
        let d = only(
            "fn main() -> i32 { let x: f32 = 4.0; return sqr(x) as i32; }",
            "E0320",
        );
        assert_eq!(replacement(&d), Some("sqrt"));
        // Nothing close: no suggestion, still the resolution code.
        let d = only("fn main() -> i32 { return zzzzzz(); }", "E0320");
        assert!(d.suggestions.is_empty(), "{d:?}");
    }

    #[test]
    fn a_method_called_as_a_free_function_names_its_type() {
        let d = only(
            "struct V { x: f32 } impl V { fn norm(self) -> f32 { return self.x; } } \
             fn main() -> i32 { let v = V { x: 2.0 }; return norm(v) as i32; }",
            "E0320",
        );
        assert!(notes(&d).contains("`V::norm(..)`"), "{d:?}");
    }

    #[test]
    fn calling_a_value_is_expected_function() {
        let local = only(
            "fn main() -> i32 { let scale: f32 = 2.0; let y: f32 = scale(3.0); return y as i32; }",
            "E0323",
        );
        assert_eq!(local.message, "expected function, found `f32`");
        let konst = only(
            "const K: i32 = 3; fn main() -> i32 { return K(1); }",
            "E0323",
        );
        assert_eq!(konst.message, "expected function, found constant `K`");
        let strukt = only(
            "struct P { x: i32 } fn main() -> i32 { let p = P(1); return 0; }",
            "E0323",
        );
        assert_eq!(strukt.message, "expected function, found struct `P`");
        let ty = only(
            "fn main() -> i32 { let x: i64 = 3; return i32(x); }",
            "E0323",
        );
        assert!(notes(&ty).contains("`x as i32`"), "{ty:?}");
        let generic = only(
            "fn f<N>(a: Tensor[f32, N]) -> f32 { return N(1); }",
            "E0323",
        );
        assert_eq!(
            generic.message,
            "expected function, found generic parameter `N`"
        );
        // A builtin's name bound to a local value is that value, not the builtin.
        assert!(
            codes("fn f(a: [f32; 4], mut o: [f32; 4]) { let relu: f32 = 1.0; relu(a, o); }")
                .contains(&"E0323")
        );
    }

    #[test]
    fn a_line_leading_paren_on_an_element_is_expected_function() {
        let src = "fn main() -> i32 { let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
                   let mut acc: f32 = 0.0; acc = a[1]\n(2.0) * 2.0; return acc as i32; }";
        let d = only(src, "E0323");
        assert!(notes(&d).contains("end the previous one with `;`"), "{d:?}");
    }

    /// The source text a diagnostic's first suggestion replaces, its replacement and how sure the
    /// compiler is of it. An insertion replaces the empty string.
    fn edit(src: &str, d: &Diagnostic) -> (String, String, wukong_diag::Applicability) {
        let s = d.suggestions.first().expect("a suggestion");
        (
            src[s.span.lo as usize..s.span.hi as usize].to_string(),
            s.replacement.clone(),
            s.applicability,
        )
    }

    /// N5: the fixes that the help text describes travel as structured suggestions whose spans
    /// are byte-exact, so an editor can apply them without re-parsing.
    #[test]
    fn resolution_fixes_are_structured_with_exact_spans() {
        use wukong_diag::Applicability::*;
        // E0323: a field called like a method; the fix deletes the argument list.
        let src = "struct S { w: f32 } fn f(s: S) -> f32 { return s.w(); }";
        let d = only(src, "E0323");
        assert_eq!(
            edit(src, &d),
            ("()".to_string(), String::new(), MaybeIncorrect)
        );
        // E0321: a method read like a field; the fix inserts the call right after the name.
        let src = "struct V { x: f32 } impl V { fn norm(self) -> f32 { return self.x; } } \
                   fn f(v: V) -> f32 { return v.norm; }";
        let d = first(src, "E0321");
        let s = &d.suggestions[0];
        let after = (src.find("v.norm;").unwrap() + "v.norm".len()) as u32;
        assert_eq!((s.span.lo, s.span.hi), (after, after));
        assert_eq!(
            (s.replacement.as_str(), s.applicability),
            ("()", HasPlaceholders)
        );
        // E0320: an associated function called bare; the fix names it through its type.
        let src = "struct V { x: f32 } impl V { fn norm(self) -> f32 { return self.x; } } \
                   fn main() -> i32 { let v = V { x: 2.0 }; return norm(v) as i32; }";
        let d = only(src, "E0320");
        let s = d
            .suggestions
            .iter()
            .find(|s| s.replacement == "V::norm")
            .unwrap_or_else(|| panic!("{d:?}"));
        let at = src.find("norm(v)").unwrap() as u32;
        assert_eq!((s.span.lo, s.span.hi), (at, at + 4));
        // E0324: a struct literal of an enum; the fix inserts a variant placeholder.
        let src = "enum E { A, B } fn main() -> i32 { let p = E { x: 1.0 }; return 0; }";
        let d = only(src, "E0324");
        let s = &d.suggestions[0];
        let after = (src.find("E { x").unwrap() + 1) as u32;
        assert_eq!((s.span.lo, s.span.hi), (after, after));
        assert_eq!(
            (s.replacement.as_str(), s.applicability),
            ("::Variant", HasPlaceholders)
        );
        // E0503's caret has a label.
        let d = only("fn f(s: []f32) -> i64 { return s.len(1); }", "E0503");
        assert!(d.labels.iter().all(|l| !l.message.is_empty()), "{d:?}");
    }

    /// N5: E0304 inserts `mut ` before the binding's name — machine-applicable for a rebind,
    /// maybe-incorrect where `mut` makes an aggregate parameter's store reach the caller.
    #[test]
    fn an_immutable_assignment_suggests_mut() {
        use wukong_diag::Applicability::*;
        let src = "fn main() -> i32 { let x: i32 = 5; x = 10; return x; }";
        let d = only(src, "E0304");
        let at = src.find("x: i32").unwrap() as u32;
        let s = &d.suggestions[0];
        assert_eq!((s.span.lo, s.span.hi), (at, at));
        assert_eq!(
            (s.replacement.as_str(), s.applicability),
            ("mut ", MachineApplicable)
        );
        assert!(d.labels.iter().all(|l| !l.message.is_empty()), "{d:?}");
        let src = "fn f(n: i32) -> i32 { n = 2; return n; } fn main() -> i32 { return f(1); }";
        let d = only(src, "E0304");
        let at = src.find("n: i32").unwrap() as u32;
        assert_eq!(
            (d.suggestions[0].span.lo, d.suggestions[0].applicability),
            (at, MachineApplicable)
        );
        let src = "fn f(a: [f32; 2]) { a[0] = 1.0; }";
        let d = only(src, "E0304");
        let at = src.find("a: [f32").unwrap() as u32;
        assert_eq!(
            (d.suggestions[0].span.lo, d.suggestions[0].applicability),
            (at, MaybeIncorrect)
        );
    }

    #[test]
    fn a_struct_field_called_like_a_method_is_expected_function() {
        let d = only(
            "struct S { w: f32 } fn f(s: S) -> f32 { return s.w(1.0); }",
            "E0323",
        );
        assert_eq!(d.message, "expected function, found `f32`");
    }

    #[test]
    fn an_unknown_field_lists_the_fields_and_suggests_one() {
        let d = only(
            "struct Point { x: f32, y: f32 } fn f(p: Point) -> f32 { return p.yy; }",
            "E0321",
        );
        assert_eq!(d.message, "no field `yy` on type `Point`");
        assert_eq!(replacement(&d), Some("y"));
        assert!(notes(&d).contains("`x` and `y`"), "{d:?}");
        // Through a reference, as the access auto-derefs.
        assert!(codes("struct P { x: f32 } fn f(p: &P) -> f32 { return p.z; }").contains(&"E0321"));
        // As an assignment target.
        assert!(codes("struct P { x: f32 } fn f(mut p: P) { p.z = 1.0; }").contains(&"E0321"));
    }

    /// An unknown variant named as a VALUE is the same `E0301` the call and pattern forms report;
    /// it used to type as `Unknown` and fail in lowering as `C0001`.
    #[test]
    fn an_unknown_variant_value_is_a_resolution_error() {
        let d = only(
            "enum Color { Red, Green } fn main() -> i32 { let c = Color::Gren; return 0; }",
            "E0301",
        );
        assert_eq!(d.message, "no variant `Color::Gren`");
        assert_eq!(replacement(&d), Some("Green"));
        // A real variant, and an associated function's namespace, stay accepted.
        let ds = diags(
            "enum Color { Red, Green } impl Color { fn first() -> Color { return Color::Red; } } \
             fn main() -> i32 { let c = Color::Green; let d = Color::first(); return 0; }",
        );
        assert!(ds.is_empty(), "{ds:?}");
    }

    #[test]
    fn a_builtin_type_has_no_named_fields() {
        let scalar = only("fn f(x: f32) -> f32 { return x.re; }", "E0321");
        assert_eq!(scalar.message, "no field `re` on type `f32`");
        let len = only("fn f(s: []f32) -> i64 { return s.len; }", "E0321");
        assert!(notes(&len).contains("`.len(..)`"), "{len:?}");
        let tuple = only("fn f(t: (i32, i32)) -> i32 { return t.x; }", "E0321");
        assert!(notes(&tuple).contains("`.0` to `.1`"), "{tuple:?}");
        let numbered = only("fn f(x: i32) -> i32 { return x.0; }", "E0321");
        assert_eq!(numbered.message, "no field `0` on type `i32`");
        let behind = only("fn f(p: *(i32, i32)) -> i32 { return p.0; }", "E0321");
        assert!(notes(&behind).contains("`(*p).0`"), "{behind:?}");
    }

    #[test]
    fn an_unknown_builtin_method_names_the_real_ones() {
        let d = only("fn f(x: f32) -> f32 { return x.sqrt(); }", "E0322");
        assert_eq!(d.message, "no method named `sqrt` found for type `f32`");
        assert!(notes(&d).contains("`sqrt(..)`"), "{d:?}");
        let slice = only("fn f(s: []f32) -> i64 { return s.dim(0); }", "E0322");
        assert!(notes(&slice).contains("`len`"), "{slice:?}");
        let typo = only(
            "fn f(t: Tensor[f32, 4]) -> i64 { return t.lenn(); }",
            "E0322",
        );
        assert_eq!(replacement(&typo), Some("len"));
        assert_eq!(
            only("fn f(s: []f32) -> i64 { return s.len(1); }", "E0503").message,
            "`.len()` takes no arguments, but 1 were supplied"
        );
    }

    /// A method call's receiver is typed by the method lookup and must not be typed again by the
    /// call fallthrough, or an error inside it is reported twice.
    #[test]
    fn an_error_inside_a_method_receiver_is_reported_once() {
        only(
            "struct P { x: f32 } fn f(p: P) -> i64 { return p.zz.len(); }",
            "E0321",
        );
        only("fn f() -> i64 { return foo(1).len(); }", "E0320");
    }

    #[test]
    fn the_unmodelled_surface_stays_lenient() {
        for src in [
            // The explicit-SIMD spellings the language guide documents as checked, not lowered.
            "fn f(p: *f32) { let v = f32x8::load(p.offset(0)); v.store(p); }",
            "fn f<R, C>(x: Tensor[f32, R, C]) { let row = x.row_ptr(0); }",
            // A generic receiver is checked against each instantiation, not here.
            "fn n<T>(x: T) -> i64 { return x.len(); }",
            "fn n<T>(x: T) -> i64 { return T::make(); }",
            // A type used as a namespace.
            "fn f() -> f32 { return f32::MAX; }",
            // The statement builtins.
            "fn main() -> i32 { print(1); println(2.0); assert(true); return 0; }",
        ] {
            let ds = diags(src);
            assert!(
                !ds.iter().any(|d| matches!(
                    d.code,
                    Some("E0320" | "E0321" | "E0322" | "E0323" | "E0324")
                )),
                "{src}: {ds:?}"
            );
        }
    }

    #[test]
    fn an_unknown_type_is_reported_once_wherever_it_is_written() {
        for src in [
            "fn main() -> i32 { let x: flot = 2.5; return 0; }",
            "fn half(x: flot) -> f32 { return 1.0; }",
            "fn half(x: f32) -> flot { return x; }",
            "struct S { a: flot }",
            "enum E { A(flot), B }",
            "extern { fn ext(a: flot) -> f32; }",
            "trait T { fn m(self) -> flot; }",
            "const C: flot = 1;",
            "fn main() -> i32 { let x: i32 = 3; let y = x as flot; return 0; }",
            "fn main() -> i32 { let n = sizeof[flot]; return 0; }",
        ] {
            let n = codes(src).iter().filter(|c| **c == "E0324").count();
            assert_eq!(n, 1, "{src}: {:?}", diags(src));
        }
    }

    #[test]
    fn an_unknown_type_suggests_a_spelling() {
        let d = first("fn f(x: F32) -> f32 { return 1.0; }", "E0324");
        assert_eq!(d.message, "cannot find type `F32` in this scope");
        assert_eq!(replacement(&d), Some("f32"));
        let c = first("fn f(x: float) -> f32 { return 1.0; }", "E0324");
        assert_eq!(replacement(&c), Some("f32"));
        let s = first(
            "struct Point { x: f32 } fn f(p: Pont) -> f32 { return 1.0; }",
            "E0324",
        );
        assert_eq!(replacement(&s), Some("Point"));
    }

    #[test]
    fn a_trait_is_not_a_type() {
        let d = first(
            "trait Shape { fn area(self) -> f32; } fn f(x: Shape) -> f32 { return 1.0; }",
            "E0324",
        );
        assert_eq!(d.message, "expected a type, found trait `Shape`");
    }

    #[test]
    fn declared_and_generic_types_resolve() {
        for src in [
            // A struct used before its declaration.
            "fn f(p: P) -> i32 { return p.x; } struct P { x: i32 }",
            "fn id<T>(x: T) -> T { return x; }",
            "struct Box1<T> { v: T }",
            "enum Opt<T> { Some(T), None }",
            "trait Shape { fn area(self) -> f32; fn make() -> Self; }",
            "enum C { A, B } fn f(c: C) -> i32 { return 0; }",
            "fn main() -> i32 { let n = sizeof[f32] + sizeof[[i32; 4]]; return 0; }",
        ] {
            let ds = diags(src);
            assert!(!ds.iter().any(|d| d.code == Some("E0324")), "{src}: {ds:?}");
        }
    }

    #[test]
    fn a_literal_of_an_unknown_struct_is_reported() {
        let d = only(
            "struct Point { x: f32 } fn main() -> i32 { let p = Pointt { x: 1.0 }; return 0; }",
            "E0324",
        );
        assert_eq!(d.message, "cannot find struct `Pointt` in this scope");
        assert_eq!(replacement(&d), Some("Point"));
        let e = only(
            "enum E { A, B } fn main() -> i32 { let p = E { x: 1.0 }; return 0; }",
            "E0324",
        );
        assert_eq!(e.message, "expected struct, found enum `E`");
    }
}
