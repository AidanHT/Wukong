//! `wukong_sema` — name resolution, type inference/checking, and shape checking.
//!
//! Sema annotates the AST through side tables keyed by [`NodeId`] (it never rewrites the tree).
//! Its headline job is **shape checking**: tensor dimensions live in the type system, so a shape
//! mismatch at a call site or an index is a compile error, not a runtime crash.
//!
//! The checker is deliberately *lenient* about constructs it does not yet model (builtin methods
//! like `f32x8::load` or `row_ptr`): those yield [`Ty::Unknown`], which unifies with anything and
//! never produces a false positive. Errors are reserved for things we can be sure about — an
//! unresolved name, a `let` whose annotation contradicts its initializer, or a genuine shape
//! conflict.
//!
//! Alongside names, types and shapes, sema owns the rejections that keep the backends from
//! disagreeing on an *accepted* program: `match` exhaustiveness, `mut`/immutability, cast validity,
//! literal well-formedness and range, definite return, and the reserved `wukong_` kernel-name
//! prefix. That is every `E03xx`/`E04xx`/`E05xx` code except `E0305` (unresolved import), which
//! belongs to the driver's import loader.
//!
//! [`check`] is the sole entry point, and its stage order is load-bearing: `collect` (register every
//! top-level definition with its signature lowered) → `refresh_collected_types` (resolve forward
//! extents) → `check_recursive_types` →
//! `check_recursive_consts` → set `checking_bodies` → `recheck_item_signatures` → `check_bodies`.
//! Several diagnostics are gated on the `checking_bodies` flag so they fire only once every generic
//! and `const` is registered.

mod attrs;
mod dependency;
mod layout;
mod shape;

use std::cell::RefCell;

use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet};

use wukong_ast::*;
use wukong_diag::Diagnostic;
use wukong_span::{Interner, Span, Symbol};
use wukong_types::{Dim, Layout, Scalar, Shape, Ty};

/// The element scalar of a typed heap-allocation builtin (`alloc_f32` → `F32`), or `None` for any
/// other name. The v1 heap surface is this per-scalar `alloc_<T>(n) -> []T` family plus `free(s)`
/// (not a generic `alloc<T>`): sema types builtins nominally in one place, and `mir_build` — which
/// shares this table — needs only the element's byte size and float-ness to lower the call. A
/// user-defined function of the same name shadows the builtin (checked before the builtin path).
pub fn heap_alloc_elem(name: &str) -> Option<Scalar> {
    Some(match name {
        "alloc_f32" => Scalar::F32,
        "alloc_f64" => Scalar::F64,
        "alloc_i32" => Scalar::I32,
        "alloc_i64" => Scalar::I64,
        "alloc_i8" => Scalar::I8,
        "alloc_u8" => Scalar::U8,
        "alloc_f16" => Scalar::F16,
        "alloc_bf16" => Scalar::Bf16,
        _ => return None,
    })
}

/// The buffer-element scalar of a file-I/O intrinsic (`read_f32`/`write_f32` → `F32`), or `None`
/// for any other name. The file-I/O surface is the per-scalar `read_<T>(path, buf) -> i64` /
/// `write_<T>(path, buf) -> i64` family over the dtypes {f32, f64, i32, i64, i8, u8}: sema types
/// these builtins nominally in one place (like [`heap_alloc_elem`]), validating that `path` is a
/// `*u8` and `buf` is a `[]T` slice of this element type. `mir_build` shares the same table to
/// select the runtime symbol and element byte size. A user-defined function of the same name
/// shadows the builtin (checked before the builtin path).
pub fn file_io_elem(name: &str) -> Option<Scalar> {
    Some(match name {
        "read_f32" | "write_f32" => Scalar::F32,
        "read_f64" | "write_f64" => Scalar::F64,
        "read_i32" | "write_i32" => Scalar::I32,
        "read_i64" | "write_i64" => Scalar::I64,
        "read_i8" | "write_i8" => Scalar::I8,
        "read_u8" | "write_u8" => Scalar::U8,
        _ => return None,
    })
}

/// Whether `name` is the `now_ns()` timing intrinsic: a zero-argument call returning a monotonic
/// nanosecond `i64` (backed by `wukong_now_ns` in `wukong_runtime`), for benchmarking Wukong
/// programs in-process. Named here alongside [`file_io_elem`] so sema (which types the call `i64`
/// and enforces the zero arity) and `mir_build` (which lowers it to the runtime call) share the one
/// source-level spelling and cannot drift. A user-defined `now_ns` shadows the builtin (checked
/// before the builtin path).
pub fn is_now_ns(name: &str) -> bool {
    name == "now_ns"
}

/// Whether `name` is the cooperative thread-block barrier builtin (`sync_block()`).
pub fn is_sync_block(name: &str) -> bool {
    name == "sync_block"
}

/// A resolved top-level definition.
#[derive(Clone, Debug)]
pub struct FnSig {
    pub generics: Vec<Symbol>,
    pub params: Vec<Ty>,
    pub ret: Ty,
}

#[derive(Clone, Debug)]
pub enum DefKind {
    Fn(FnSig),
    Const(Ty),
    Struct(Vec<(Symbol, Ty)>),
    /// An enum: its variants in declaration order. Each carries a resolved integer discriminant
    /// (auto-incremented from 0, or set by an explicit `= <int>`) and its payload (unit / tuple /
    /// struct, field types already lowered to `Ty`). A **C-style** enum has all-unit variants — a
    /// variant *is* its discriminant, lowered to an i32 scalar. A **data-carrying** (tagged-union)
    /// enum has ≥1 payload variant — every value is a discriminant + padded payload byte buffer.
    Enum(Vec<EnumVariant>),
}

/// A resolved enum variant: name, computed i32 discriminant, and payload with lowered field types.
#[derive(Clone, Debug)]
pub struct EnumVariant {
    pub name: Symbol,
    pub disc: i64,
    pub payload: VariantPayload,
}

/// The data a variant carries. Unit is the C-style variant (no payload); tuple/struct payloads make
/// the enum a tagged union. The tagged-union layout treats every payload as a tuple of `field_tys`.
#[derive(Clone, Debug)]
pub enum VariantPayload {
    Unit,
    Tuple(Vec<Ty>),
    Struct(Vec<(Symbol, Ty)>),
}

impl VariantPayload {
    /// The payload field types in positional order (struct fields in declaration order); empty for a
    /// unit variant. The single source of truth for a variant's payload layout.
    pub fn field_tys(&self) -> Vec<Ty> {
        match self {
            VariantPayload::Unit => Vec::new(),
            VariantPayload::Tuple(ts) => ts.clone(),
            VariantPayload::Struct(fs) => fs.iter().map(|(_, t)| t.clone()).collect(),
        }
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, VariantPayload::Unit)
    }
}

#[derive(Clone, Debug)]
pub struct Def {
    pub name: Symbol,
    pub kind: DefKind,
    pub span: Span,
}

/// All top-level definitions, indexed by name.
#[derive(Clone, Default, Debug)]
pub struct DefMap {
    pub defs: Vec<Def>,
    by_name: HashMap<Symbol, usize>,
}

impl DefMap {
    pub fn lookup(&self, name: Symbol) -> Option<&Def> {
        self.by_name.get(&name).map(|&i| &self.defs[i])
    }
}

/// The result of analyzing a module: types attached to expression nodes, and the def map.
///
/// `Clone` so a consumer that *adds* typed nodes to the AST after checking (the tensor multi-index
/// normalization in `wukong_mir_build`) can extend the type table without needing `&mut` access to
/// the analysis result, which the whole pipeline shares immutably.
#[derive(Clone)]
pub struct SemaResult {
    pub types: HashMap<NodeId, Ty>,
    pub defs: DefMap,
    /// Top-level `const` initializer expressions, by name. Their nodes are type-checked (and adapted
    /// literals retyped) like a `let`, so `mir_build` can lower a const reference by inlining the
    /// initializer with correct types. The `DefMap` records only a const's *type*, not its value.
    pub consts: HashMap<Symbol, Expr>,
}

/// Collect the struct names a type contains *by value* — directly (`Ty::Named`) or nested inside an
/// array/tuple element. Pointer/reference fields are excluded: they have a fixed size and break a
/// size cycle. Non-struct `Named`s are pushed too but resolve to nothing contained (dead ends).
fn collect_value_structs(t: &Ty, out: &mut Vec<Symbol>) {
    match t {
        Ty::Named(n) => out.push(*n),
        Ty::Array { elem, .. } => collect_value_structs(elem, out),
        Ty::Tuple(elems) => {
            for e in elems {
                collect_value_structs(e, out);
            }
        }
        _ => {}
    }
}

/// Collect the names of top-level consts that expression `e` refers to (single-segment paths in the
/// `consts` set). Used to detect a self-referential const initializer before `mir_build` inlines it
/// (which would recurse forever, a compiler stack overflow). Recurses through every sub-expression,
/// including block/`if`/`match` bodies.
fn collect_const_refs(e: &Expr, consts: &HashSet<Symbol>, out: &mut Vec<Symbol>) {
    match &e.kind {
        ExprKind::Path(p) => {
            if p.is_single() && consts.contains(&p.first().sym) {
                out.push(p.first().sym);
            }
        }
        ExprKind::Unary { expr, .. }
        | ExprKind::Cast { expr, .. }
        | ExprKind::Field { base: expr, .. }
        | ExprKind::TupleField { base: expr, .. } => collect_const_refs(expr, consts, out),
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_const_refs(lhs, consts, out);
            collect_const_refs(rhs, consts, out);
        }
        ExprKind::Call { callee, args, .. } => {
            collect_const_refs(callee, consts, out);
            for a in args {
                collect_const_refs(a, consts, out);
            }
        }
        ExprKind::Index { base, indices } => {
            collect_const_refs(base, consts, out);
            for i in indices {
                collect_const_refs(i, consts, out);
            }
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                collect_const_refs(x, consts, out);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            collect_const_refs(value, consts, out);
            collect_const_refs(count, consts, out);
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for f in fields {
                collect_const_refs(&f.value, consts, out);
            }
            if let Some(r) = rest {
                collect_const_refs(r, consts, out);
            }
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_const_refs(cond, consts, out);
            collect_block_const_refs(then_branch, consts, out);
            if let Some(el) = else_branch {
                collect_const_refs(el, consts, out);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            collect_const_refs(scrutinee, consts, out);
            for a in arms {
                if let Some(g) = &a.guard {
                    collect_const_refs(g, consts, out);
                }
                collect_const_refs(&a.body, consts, out);
            }
        }
        ExprKind::Block(b) => collect_block_const_refs(b, consts, out),
        ExprKind::Loop { body, .. } => collect_block_const_refs(body, consts, out),
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => {}
    }
}

/// [`collect_const_refs`] over a block's statements and tail.
fn collect_block_const_refs(b: &Block, consts: &HashSet<Symbol>, out: &mut Vec<Symbol>) {
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Let { init: Some(e), .. }
            | StmtKind::Expr(e)
            | StmtKind::Return(Some(e))
            | StmtKind::Defer(e) => collect_const_refs(e, consts, out),
            StmtKind::Assign { target, value, .. } => {
                collect_const_refs(target, consts, out);
                collect_const_refs(value, consts, out);
            }
            StmtKind::While { cond, body, .. } => {
                collect_const_refs(cond, consts, out);
                collect_block_const_refs(body, consts, out);
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        collect_const_refs(start, consts, out);
                        if let Some(e) = end {
                            collect_const_refs(e, consts, out);
                        }
                        if let Some(s) = step {
                            collect_const_refs(s, consts, out);
                        }
                    }
                    ForIter::Expr(e) => collect_const_refs(e, consts, out),
                }
                collect_block_const_refs(body, consts, out);
            }
            // A `break <value>` may reference a top-level `const`.
            StmtKind::Break(_, Some(e)) => collect_const_refs(e, consts, out),
            _ => {}
        }
    }
    if let Some(t) = &b.tail {
        collect_const_refs(t, consts, out);
    }
}

/// Analyze a module: returns per-expression types (plus the def map and the checked `const`
/// initializers) and every diagnostic collected along the way. Checking does not stop at the first
/// error — a [`SemaResult`] comes back even when `diags` holds errors, and a rejected definition can
/// still be registered (a reserved `wukong_`-prefixed function is) so its call sites do not cascade.
/// The caller stops the pipeline: `wukong_driver::compile` refuses to run `mir_build` if any returned
/// diagnostic `is_error()`.
pub fn check(module: &Module, interner: &Interner) -> (SemaResult, Vec<Diagnostic>) {
    let mut s = Sema {
        interner,
        defs: DefMap::default(),
        diags: Vec::new(),
        types: HashMap::default(),
        scopes: Vec::new(),
        immutable_locals: Vec::new(),
        immutable_params: Vec::new(),
        param_tys: HashMap::default(),
        generics: HashSet::default(),
        ret_ty: Ty::Unit,
        loop_ctx: Vec::new(),
        consts: HashMap::default(),
        const_lengths: RefCell::new(HashMap::default()),
        layout_cache: layout::Cache::default(),
        checking_bodies: false,
        parallel_fn: false,
    };
    // Attribute legality (`attrs`): an attribute names no value and has no type, so this walk needs
    // nothing the collection pass builds and runs first — an unknown `@bogus` is reported even in a
    // module whose bodies do not resolve.
    attrs::check(module, interner, &mut s.diags);
    s.collect(module);
    s.refresh_collected_types(module);
    s.check_recursive_types(module);
    s.check_recursive_consts(module);
    // The undeclared-tensor-dimension check (`lower_dim`) fires only from here on: by the body pass
    // `collect` has registered every top-level `const` and this item's generics are re-established
    // per function, so an unknown dim name can be told from a declared generic / a `const` with no
    // forward-reference false positive (and a function signature, re-lowered here, is reported once).
    s.checking_bodies = true;
    s.layout_cache.prepare(&s.defs);
    for def in s.defs.defs.clone() {
        if matches!(def.kind, DefKind::Struct(_) | DefKind::Enum(_)) {
            s.check_layout(&Ty::Named(def.name), def.span);
        }
    }
    s.recheck_item_signatures(module);
    s.check_bodies(module);
    let result = SemaResult {
        types: s.types,
        defs: s.defs,
        consts: s
            .consts
            .into_iter()
            .map(|(name, init)| (name, init.clone()))
            .collect(),
    };
    (result, s.diags)
}

struct Sema<'a> {
    interner: &'a Interner,
    defs: DefMap,
    diags: Vec<Diagnostic>,
    types: HashMap<NodeId, Ty>,
    scopes: Vec<HashMap<Symbol, Ty>>,
    /// Per-scope set of locals bound *immutably* — a `let` without `mut` that has an initializer —
    /// kept 1:1 with `scopes`. Used to reject reassigning such a binding (`let x = 5; x = 10;`).
    immutable_locals: Vec<HashSet<Symbol>>,
    /// Per-scope set of *parameters* declared without `mut`, kept 1:1 with `scopes` (parameters live
    /// in the function's top scope). Stricter than `immutable_locals`: a non-`mut` parameter rejects
    /// both direct reassignment AND mutation through a projection (`p.f = …`, `p[i] = …`), because an
    /// aggregate parameter is passed by reference — the projection would mutate the *caller's* value.
    immutable_params: Vec<HashSet<Symbol>>,
    /// Declared type of each parameter of the function currently being checked, by name. Queried only
    /// when the name resolves to an immutable parameter (so shadowing by an inner `let` can't confuse
    /// it); used to decide whether a through-projection mutation reaches the caller (aggregate) or a
    /// pointee (pointer/ref — legitimately mutable).
    param_tys: HashMap<Symbol, Ty>,
    generics: HashSet<Symbol>,
    ret_ty: Ty,
    /// The enclosing loops at the current point (innermost last). A `break`/`continue` with an empty
    /// stack is a hard error (E0303) — without it the lowerer emits an `unreachable` terminator,
    /// which the interpreter traps but the native backend turns into a SIGILL, a differential-gate
    /// divergence. A labeled `break`/`continue` `'l` whose label is not on the stack is likewise
    /// E0303. Each frame also carries whether the loop is used as a *value* (so `break v` is legal)
    /// and the running join of its break-value types (the loop's inferred type).
    loop_ctx: Vec<LoopCtx>,
    /// Top-level `const` initializer expressions (by name), borrowed from the immutable input AST.
    /// Literal adaptation updates `types`, never the initializer; clone each surviving initializer
    /// only when publishing the owned `SemaResult`.
    consts: HashMap<Symbol, &'a Expr>,
    /// Length folds by constant name and the initializer's recursion depth. Depth is observable:
    /// the existing evaluator substitutes zero past 64, including in rejected cyclic programs.
    /// Invalidate when a constant initializer changes or a new enum enables discriminant lookup.
    const_lengths: RefCell<HashMap<(Symbol, u32), u64>>,
    /// `false` during the collection pass, `true` once body checking starts. Gates the
    /// undeclared-tensor-dimension diagnostic (`lower_dim`): reporting it only in the body pass means
    /// every top-level `const` is registered and each item's generics are re-established, so an
    /// unknown dim name is distinguished from a declared generic / a `const` with no forward-reference
    /// false positive, and a function signature (re-lowered in `check_fn`) is reported exactly once.
    checking_bodies: bool,
    /// Named layouts are queried only after collection, when definitions are stable.
    layout_cache: layout::Cache,
    /// True while checking a `@parallel` function body (`sync_block()` is legal only here).
    parallel_fn: bool,
}

/// One entry of [`Sema::loop_ctx`] — a loop currently being type-checked.
struct LoopCtx {
    /// The loop's label (`None` for an unlabeled loop), matched by a labeled `break`/`continue`.
    label: Option<Symbol>,
    /// Whether the loop is used in value position (a `let` init, call arg, block tail, `return`, …),
    /// so `break <value>` is permitted. A statement-position loop (`StmtKind::Expr`) and every
    /// `while`/`for` set this `false`: a `break` there must carry no value.
    is_value: bool,
    /// The join of every `break <value>` type seen so far (`None` until the first value break). The
    /// loop's inferred type is this (or unit if it stays `None`). Tensor/vector shapes are unified
    /// across breaks exactly like `if`/`match` arms.
    break_ty: Option<Ty>,
}

impl<'a> Sema<'a> {
    fn error(&mut self, span: Span, code: &'static str, msg: impl Into<String>) {
        self.diags
            .push(Diagnostic::error(msg).with_code(code).primary(span, ""));
    }

    fn sym_str(&self, s: Symbol) -> &str {
        self.interner.resolve(s)
    }

    // ---- Collection pass: register all top-level defs with lowered signatures ----

    fn collect(&mut self, module: &'a Module) {
        for item in &module.items {
            match &item.kind {
                ItemKind::Fn(f) => self.collect_fn(f),
                ItemKind::Const(c) => {
                    let ty = self.lower_type(&c.ty);
                    self.register(c.name, DefKind::Const(ty), item.span);
                    // Record the initializer now (it is re-recorded, with its nodes typed, by
                    // `check_const`) so a `const` used as a tensor/array index can be resolved to its
                    // value by the compile-time bounds check regardless of source order.
                    self.record_const(c);
                }
                ItemKind::Struct(s) => {
                    self.check_unique_decl_names(
                        &Self::generic_decl_names(&s.generics),
                        "generic parameter",
                    );
                    let field_names: Vec<(Symbol, Span)> = s
                        .fields
                        .iter()
                        .map(|fl| (fl.name.sym, fl.name.span))
                        .collect();
                    self.check_unique_decl_names(&field_names, "struct field");
                    self.generics = generic_names(&s.generics);
                    let fields = s
                        .fields
                        .iter()
                        .map(|fl| (fl.name.sym, self.lower_type(&fl.ty)))
                        .collect();
                    self.generics.clear();
                    self.register(s.name, DefKind::Struct(fields), item.span);
                }
                ItemKind::Enum(e) => {
                    self.check_unique_decl_names(
                        &Self::generic_decl_names(&e.generics),
                        "generic parameter",
                    );
                    let variant_names: Vec<(Symbol, Span)> = e
                        .variants
                        .iter()
                        .map(|v| (v.name.sym, v.name.span))
                        .collect();
                    self.check_unique_decl_names(&variant_names, "enum variant");
                    for v in &e.variants {
                        if let VariantData::Struct(fields) = &v.data {
                            let fs: Vec<(Symbol, Span)> =
                                fields.iter().map(|f| (f.name.sym, f.name.span)).collect();
                            self.check_unique_decl_names(&fs, "variant field");
                        }
                    }
                    // Payload field types are lowered in the enum's generic scope (like struct
                    // fields), so a generic-typed payload resolves its param names.
                    self.generics = generic_names(&e.generics);
                    // Resolve each variant's integer discriminant: an explicit `= <int>` sets it,
                    // otherwise it auto-increments from the previous (starting at 0), as in C/Rust.
                    let mut next = 0i64;
                    let mut variants = Vec::with_capacity(e.variants.len());
                    for v in &e.variants {
                        // A written discriminant the folder cannot read — most often a reference to a
                        // top-level `const` — used to be DISCARDED for the auto-increment value with
                        // no diagnostic: `enum Op { Add = 1, Mul = OP_MUL, Div }` numbered Mul 2 and
                        // Div 3 on both backends, silently renumbering an opcode table. Say so
                        // instead. (Resolving the name through `self.consts` here would make the
                        // value depend on source order — `collect` inserts consts as it walks — so
                        // const-valued discriminants need a const pre-pass, not a lookup here.)
                        let disc = match &v.discriminant {
                            None => next,
                            Some(d) => match eval_const_int(d, self.interner) {
                                Some(x) => x,
                                None => {
                                    self.error(
                                        d.span,
                                        "E0401",
                                        "an enum discriminant must be a compile-time integer \
                                         constant (an integer literal, or arithmetic on integer \
                                         literals)",
                                    );
                                    next
                                }
                            },
                        };
                        // An enum value lowers to a 32-bit discriminant (the C-style repr —
                        // mir_build emits `ConstInt(_, I32)`), so a discriminant outside the i32
                        // range would be silently truncated (a max-`i64` sentinel printed as its low
                        // 32 bits). Reject it with a clear diagnostic instead of truncating.
                        if disc < i32::MIN as i64 || disc > i32::MAX as i64 {
                            let sp = v.discriminant.as_ref().map_or(item.span, |d| d.span);
                            self.error(
                                sp,
                                "E0401",
                                format!(
                                    "enum discriminant `{disc}` is out of range for the 32-bit \
                                     enum representation ({}..={})",
                                    i32::MIN,
                                    i32::MAX
                                ),
                            );
                        }
                        let payload = match &v.data {
                            VariantData::Unit => VariantPayload::Unit,
                            VariantData::Tuple(tys) => VariantPayload::Tuple(
                                tys.iter().map(|t| self.lower_type(t)).collect(),
                            ),
                            VariantData::Struct(fields) => VariantPayload::Struct(
                                fields
                                    .iter()
                                    .map(|f| (f.name.sym, self.lower_type(&f.ty)))
                                    .collect(),
                            ),
                        };
                        variants.push(EnumVariant {
                            name: v.name.sym,
                            disc,
                            payload,
                        });
                        // Wrapping, so a sentinel discriminant near `i64::MAX` doesn't panic the
                        // compiler on the auto-increment of the *next* variant before the range
                        // check above fires; wrapping matches two's-complement integer semantics.
                        next = disc.wrapping_add(1);
                    }
                    self.generics.clear();
                    self.register(e.name, DefKind::Enum(variants), item.span);
                }
                ItemKind::Extern(blk) => {
                    for f in &blk.items {
                        self.collect_fn(f);
                    }
                }
                ItemKind::Import(_) => {}
            }
        }
    }

    /// Resolve declaration types again after every const and enum is registered. In the first
    /// collection pass a forward extent is unresolved; it must not remain a recovery type in a
    /// callable signature or a field. Diagnostics still belong to the existing body/recheck passes.
    fn refresh_collected_types(&mut self, module: &Module) {
        let before = self.diags.len();
        // Consts can refer to enums declared after them; discard folds from partial collection.
        self.const_lengths.get_mut().clear();
        for item in &module.items {
            let replacement = match &item.kind {
                ItemKind::Fn(f) => Some((f.name.sym, DefKind::Fn(self.resolved_fn_sig(f)))),
                ItemKind::Const(c) => Some((c.name.sym, DefKind::Const(self.lower_type(&c.ty)))),
                ItemKind::Struct(s) => {
                    self.generics = generic_names(&s.generics);
                    let fields = s
                        .fields
                        .iter()
                        .map(|f| (f.name.sym, self.lower_type(&f.ty)))
                        .collect();
                    self.generics.clear();
                    Some((s.name.sym, DefKind::Struct(fields)))
                }
                ItemKind::Enum(e) => {
                    let Some(DefKind::Enum(mut variants)) =
                        self.defs.lookup(e.name.sym).map(|d| d.kind.clone())
                    else {
                        continue;
                    };
                    self.generics = generic_names(&e.generics);
                    for (variant, decl) in variants.iter_mut().zip(&e.variants) {
                        variant.payload = match &decl.data {
                            VariantData::Unit => VariantPayload::Unit,
                            VariantData::Tuple(ts) => VariantPayload::Tuple(
                                ts.iter().map(|t| self.lower_type(t)).collect(),
                            ),
                            VariantData::Struct(fs) => VariantPayload::Struct(
                                fs.iter()
                                    .map(|f| (f.name.sym, self.lower_type(&f.ty)))
                                    .collect(),
                            ),
                        };
                    }
                    self.generics.clear();
                    Some((e.name.sym, DefKind::Enum(variants)))
                }
                ItemKind::Extern(block) => {
                    for f in &block.items {
                        let sig = self.resolved_fn_sig(f);
                        if let Some(&index) = self.defs.by_name.get(&f.name.sym) {
                            self.defs.defs[index].kind = DefKind::Fn(sig);
                        }
                    }
                    None
                }
                _ => None,
            };
            if let Some((name, kind)) = replacement {
                if let Some(&index) = self.defs.by_name.get(&name) {
                    self.defs.defs[index].kind = kind;
                }
            }
        }
        self.diags.truncate(before);
    }

    fn resolved_fn_sig(&mut self, f: &FnDecl) -> FnSig {
        self.generics = generic_names(&f.generics);
        let params = f.params.iter().map(|p| self.lower_type(&p.ty)).collect();
        let ret = f
            .ret
            .as_ref()
            .map(|t| self.lower_type(t))
            .unwrap_or(Ty::Unit);
        self.generics.clear();
        FnSig {
            generics: f.generics.iter().map(generic_param_sym).collect(),
            params,
            ret,
        }
    }

    /// Re-lower struct field, enum payload and `extern` signature types once the body pass has begun,
    /// purely for the diagnostics gated on `checking_bodies` (the unknown-tensor-dimension E0504 and
    /// the unknown-array-length check).
    ///
    /// Those types are lowered exactly ONCE — during `collect`, with the flag still false — while the
    /// only types the body pass re-lowers on its own are the parameters and return type of functions
    /// WITH a body (`check_fn`) and a top-level `const`'s annotation (`check_const`). So an undeclared
    /// dimension name in a struct field, an enum payload or an `extern` signature was silently
    /// turned into a fresh unconstrained `Dim::Var`, precisely the hole E0504 exists to close:
    /// `extern { fn dot(a: Tensor[f32, K], b: Tensor[f32, KK]) -> f32; }` — a one-character typo —
    /// gave the two parameters independent dims, so a length-4 and a length-8 buffer unified without
    /// complaint and the extern callee read a mismatched buffer.
    ///
    /// The lowered results are discarded (the def map is already populated, and `lower_type` has no
    /// other side effect), and any diagnostic identical to one `collect` already reported is dropped,
    /// so this cannot double-report.
    fn recheck_item_signatures(&mut self, module: &Module) {
        let before = self.diags.len();
        for item in &module.items {
            match &item.kind {
                ItemKind::Struct(s) => {
                    self.generics = generic_names(&s.generics);
                    for fl in &s.fields {
                        self.lower_type(&fl.ty);
                    }
                    self.generics.clear();
                }
                ItemKind::Enum(e) => {
                    self.generics = generic_names(&e.generics);
                    for v in &e.variants {
                        match &v.data {
                            VariantData::Unit => {}
                            VariantData::Tuple(tys) => {
                                for t in tys {
                                    self.lower_type(t);
                                }
                            }
                            VariantData::Struct(fields) => {
                                for f in fields {
                                    self.lower_type(&f.ty);
                                }
                            }
                        }
                    }
                    self.generics.clear();
                }
                // An extern fn declares its own generics, exactly as `collect_fn` establishes them.
                ItemKind::Extern(blk) => {
                    for f in &blk.items {
                        self.generics = generic_names(&f.generics);
                        for p in &f.params {
                            self.lower_type(&p.ty);
                        }
                        if let Some(t) = &f.ret {
                            self.lower_type(t);
                        }
                        self.generics.clear();
                    }
                }
                _ => {}
            }
        }
        if before == 0 || self.diags.len() == before {
            return;
        }
        // The old remove-at-index loop shifted every later diagnostic for every duplicate.
        // Filter the new suffix in order, borrowing message text from the untouched prefix.
        // Only collection-pass duplicates are removed: equal NEW diagnostics must both survive.
        let mut rechecked = self.diags.split_off(before);
        let seen: HashSet<_> = self
            .diags
            .iter()
            .map(|d| (d.code, d.message.as_str(), d.labels.first().map(|l| l.span)))
            .collect();
        rechecked.retain(|d| {
            !seen.contains(&(d.code, d.message.as_str(), d.labels.first().map(|l| l.span)))
        });
        self.diags.extend(rechecked);
    }

    /// Reject a struct that contains itself by value — directly (`struct S { x: S }`) or transitively
    /// (`A` holds `B` holds `A`). Such a type has infinite size; sizing or instantiating it
    /// stack-overflows mir_build's layout pass (a compiler crash on a plausible mistake — forgetting
    /// the indirection). A field behind a pointer/reference has fixed size and breaks the cycle, as
    /// in C/Rust (cf. Rust's E0072). Runs after `collect`, when every struct is registered.
    fn check_recursive_types(&mut self, module: &Module) {
        // Build each containment list once, then classify all cycles together. Repeating a DFS
        // per declaration (with a linear visited list) made long acyclic type chains cubic.
        let graph = self
            .defs
            .defs
            .iter()
            .filter(|d| matches!(d.kind, DefKind::Struct(_) | DefKind::Enum(_)))
            .map(|d| (d.name, self.contained_types(d.name)))
            .collect();
        let cyclic = dependency::cyclic_nodes(&graph);
        for item in &module.items {
            // A struct or a data-carrying enum can be recursive by value; a C-style enum holds
            // nothing (`contained_types` returns empty for it), so it can never be flagged.
            let (root, kind) = match &item.kind {
                ItemKind::Struct(s) => (s.name.sym, "struct"),
                ItemKind::Enum(e) => (e.name.sym, "enum"),
                _ => continue,
            };
            if cyclic.contains(&root) {
                let nm = self.sym_str(root).to_string();
                self.error(
                    item.span,
                    "E0402",
                    format!(
                        "recursive {kind} `{nm}` has infinite size; store the recursive field \
                         behind a pointer (e.g. `*{nm}`) to break the cycle"
                    ),
                );
            }
        }
    }

    /// The named types a struct or enum holds *by value* (a `Named` field / payload field, or one
    /// nested in an array/tuple), for the recursive-type cycle check. A pointer/reference field is
    /// excluded (fixed size, breaks the cycle). A C-style enum (all-unit) and a generic/unknown name
    /// resolve to nothing.
    fn contained_types(&self, name: Symbol) -> Vec<Symbol> {
        let mut out = Vec::new();
        match self.defs.lookup(name).map(|d| &d.kind) {
            Some(DefKind::Struct(fields)) => {
                for (_, fty) in fields {
                    collect_value_structs(fty, &mut out);
                }
            }
            Some(DefKind::Enum(variants)) => {
                for v in variants {
                    match &v.payload {
                        VariantPayload::Unit => {}
                        VariantPayload::Tuple(fields) => {
                            for fty in fields {
                                collect_value_structs(fty, &mut out);
                            }
                        }
                        VariantPayload::Struct(fields) => {
                            for (_, fty) in fields {
                                collect_value_structs(fty, &mut out);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// Reject a `const` whose initializer depends on its own value — directly (`const A = A + 1`) or
    /// transitively (`A` uses `B`, `B` uses `A`). `mir_build` inlines a const's initializer at each
    /// use site and recurses for a const-references-const, so a cycle stack-overflows the compiler
    /// (a crash on a plausible typo). Mirrors `check_recursive_types`: classify cycles in the
    /// const-reference graph. Runs after `collect`, when every const is
    /// registered, and (like the struct check) before `check_bodies`, so the error halts the pipeline
    /// ahead of mir_build's inliner.
    fn check_recursive_consts(&mut self, module: &Module) {
        let mut inits: HashMap<Symbol, &Expr> = HashMap::default();
        for item in &module.items {
            if let ItemKind::Const(c) = &item.kind {
                inits.insert(c.name.sym, &c.value);
            }
        }
        let names: HashSet<Symbol> = inits.keys().copied().collect();
        // Walk each initializer once. Shared dependency tails used to be rescanned, including
        // their entire ASTs, from every root that could reach them.
        let graph: HashMap<Symbol, Vec<Symbol>> = inits
            .iter()
            .map(|(&name, init)| {
                let mut deps = Vec::new();
                collect_const_refs(init, &names, &mut deps);
                (name, deps)
            })
            .collect();
        let cyclic = dependency::cyclic_nodes(&graph);
        for item in &module.items {
            if let ItemKind::Const(c) = &item.kind {
                let root = c.name.sym;
                let recursive = if std::ptr::eq(inits[&root], &c.value) {
                    cyclic.contains(&root)
                } else {
                    // Duplicate names already produce E0300. Preserve the former E0403 recovery
                    // behavior too: start at THIS initializer, then resolve references using the
                    // last initializer registered for each name. Only these invalid duplicate
                    // declarations need a separate reachability walk.
                    let mut stack = Vec::new();
                    collect_const_refs(&c.value, &names, &mut stack);
                    let mut visited = HashSet::default();
                    let mut recursive = false;
                    while let Some(cur) = stack.pop() {
                        if cur == root {
                            recursive = true;
                            break;
                        }
                        if visited.insert(cur) {
                            if let Some(deps) = graph.get(&cur) {
                                stack.extend(deps.iter().copied());
                            }
                        }
                    }
                    recursive
                };
                if recursive {
                    let nm = self.sym_str(root).to_string();
                    self.error(
                        item.span,
                        "E0403",
                        format!(
                            "recursive const `{nm}` depends on its own value; a const must be \
                             evaluable without referring back to itself"
                        ),
                    );
                }
            }
        }
    }

    fn collect_fn(&mut self, f: &FnDecl) {
        self.check_unique_decl_names(&Self::generic_decl_names(&f.generics), "generic parameter");
        self.generics = generic_names(&f.generics);
        let params = f.params.iter().map(|p| self.lower_type(&p.ty)).collect();
        let ret = match &f.ret {
            Some(t) => self.lower_type(t),
            None => Ty::Unit,
        };
        // Declaration order is load-bearing: a turbofish `f::<2, 3>` binds each generic argument to
        // the parameter in the SAME position (`check_fn_call` zips `sig.generics` with the arguments).
        // Building this Vec from the `self.generics` HashSet collected it in per-process-random hash
        // order, so `<M, N>` was paired with `::<2, 3>` as either {M:2,N:3} or {M:3,N:2} from one run
        // to the next on the unchanged source — making a turbofished call's accept/reject (and its
        // runtime result, down to an out-of-bounds trap) nondeterministic, a direct violation of the
        // opt-invariance / backend-agreement gates. Take the order from the AST instead.
        let generics: Vec<Symbol> = f.generics.iter().map(generic_param_sym).collect();
        self.generics.clear();
        self.register(
            f.name,
            DefKind::Fn(FnSig {
                generics,
                params,
                ret,
            }),
            f.name.span,
        );
    }

    /// Report every repeated name in a declaration list, labelling BOTH sites exactly as `register`
    /// does for a duplicate top-level definition and `check_fn` does for a duplicate parameter.
    ///
    /// Struct fields, enum variants and generic parameter lists had no such pass:
    /// `struct S { x: i32, x: i32 }` produced no declaration-site error at all (only the literal
    /// `S { x: 1, x: 2 }` was rejected, so a struct never literal-constructed compiled clean),
    /// `enum E { A, A }` compiled and RAN, and `fn f<N, N>(..)` compiled and ran. A duplicate enum
    /// variant makes `collect_enum_coverage` count one name for two discriminants, so it feeds the
    /// match-exhaustiveness reasoning behind E0405; a duplicate field means the layout carries a
    /// field no expression can ever name; a duplicate generic makes the positional turbofish zip
    /// (`FnSig.generics` × `f::<2, 3>`) bind two different arguments to one name.
    ///
    /// Order comes from the AST slice, never from a hash set, so the diagnostics are deterministic.
    fn check_unique_decl_names(&mut self, names: &[(Symbol, Span)], what: &str) {
        let mut seen: HashMap<Symbol, Span> = HashMap::default();
        for &(sym, span) in names {
            if let Some(&first) = seen.get(&sym) {
                let nm = self.sym_str(sym).to_string();
                let mut d = Diagnostic::error(format!("duplicate {what} `{nm}`"))
                    .with_code("E0300")
                    .primary(span, "redeclared here");
                if !first.is_dummy() {
                    d = d.secondary(first, "first declared here");
                }
                self.diags.push(d);
            } else {
                seen.insert(sym, span);
            }
        }
    }

    /// A declaration's generic parameter list as `(name, span)` pairs, in SOURCE order — the order
    /// `check_fn_call` zips a turbofish against. `generic_names` returns a `HashSet` and silently
    /// absorbs a repeat, which is why the duplicate was invisible.
    fn generic_decl_names(gs: &[GenericParam]) -> Vec<(Symbol, Span)> {
        gs.iter().map(|g| (generic_param_sym(g), g.span)).collect()
    }

    fn register(&mut self, name: Ident, kind: DefKind, span: Span) {
        // `wukong_` is the compiler's own runtime-kernel namespace (~150 symbols emitted by the GEMM
        // / vmath / norm / reduction / transpose / quant recognizers). A user function with one of
        // those names silently HIJACKED the dispatch: `fn wukong_norm_f32(…)` made a hand-written
        // softmax lower to a call to the user's body instead — the normalization never ran, both
        // backends agreed on the wrong answer, and no diagnostic was emitted. With a mismatched
        // arity it instead diverged (interp dropped the kernel and ran on; native aborted with a raw
        // Cranelift signature-incompatibility string, an uncatalogued error leaking backend
        // internals), and with a body that dereferences its arguments it is arbitrary memory
        // corruption — the recognizer passes raw pointers the user declared as `i64`. Reserving the
        // whole prefix closes all of that for every kernel at once; mangling the emitted symbols
        // instead would mean editing every backend's name constants.
        if matches!(kind, DefKind::Fn(_)) {
            let nm = self.sym_str(name.sym);
            if nm.starts_with("wukong_") {
                let nm = nm.to_string();
                self.diags.push(
                    Diagnostic::error(format!(
                        "the name `{nm}` is reserved for a compiler runtime kernel"
                    ))
                    .with_code("E0300")
                    .primary(name.span, "reserved name")
                    .help(
                        "rename this function; the `wukong_` prefix is reserved for the symbols \
                         the kernel recognizers emit",
                    ),
                );
                // Fall through and register it anyway: the compile is already failing, and leaving
                // the def map complete keeps every call site from cascading into E0301.
            }
        }
        if let Some(&idx) = self.defs.by_name.get(&name.sym) {
            // Point at BOTH definitions. In a multi-file program (the driver's import loader
            // splices every imported file's items into one flat namespace) the two can live in
            // different files; each label resolves through the shared SourceMap, so both render
            // with their own file/line.
            let first = self.defs.defs[idx].span;
            let mut d = Diagnostic::error(format!(
                "the name `{}` is defined more than once",
                self.sym_str(name.sym)
            ))
            .with_code("E0300")
            .primary(span, "redefined here");
            if !first.is_dummy() {
                d = d.secondary(first, "first defined here");
            }
            self.diags.push(d);
            return;
        }
        let idx = self.defs.defs.len();
        if matches!(kind, DefKind::Enum(_)) {
            // A previously unresolved E::V in an initializer can now resolve. Non-enum
            // definitions cannot change enum_variant_disc, and duplicates returned above.
            self.const_lengths.get_mut().clear();
        }
        self.defs.defs.push(Def {
            name: name.sym,
            kind,
            span,
        });
        self.defs.by_name.insert(name.sym, idx);
    }

    // ---- Type lowering: AST type syntax -> semantic Ty ----

    fn check_layout(&mut self, ty: &Ty, span: Span) {
        if self.checking_bodies {
            if let Err(limit) = self.layout_cache.extent(ty, &self.defs) {
                self.error(
                    span,
                    "E0401",
                    format!("type layout is too large to represent: exceeds {limit}"),
                );
            }
        }
    }

    fn lower_type(&mut self, t: &TypeExpr) -> Ty {
        let ty = match &t.kind {
            TypeKind::Path(p) => {
                let name = p.segments.last().unwrap().sym;
                let s = self.sym_str(name);
                if let Some(sc) = Scalar::from_name(s) {
                    Ty::Scalar(sc)
                } else {
                    // A generic type variable or a (possibly later-defined) named type.
                    Ty::Named(name)
                }
            }
            TypeKind::Int(_) => Ty::Error,
            TypeKind::Unit => Ty::Unit,
            TypeKind::Pointer { mutable, pointee } => Ty::Ptr {
                mutable: *mutable,
                pointee: Box::new(self.lower_type(pointee)),
            },
            TypeKind::Ref { mutable, pointee } => Ty::Ref {
                mutable: *mutable,
                pointee: Box::new(self.lower_type(pointee)),
            },
            TypeKind::Slice(e) => Ty::Slice(Box::new(self.lower_type(e))),
            TypeKind::Array { elem, len, shared } => {
                let elem = Box::new(self.lower_type(elem));
                let Some(len) = self.array_len(len) else {
                    return Ty::Error;
                };
                Ty::Array {
                    elem,
                    len,
                    shared: *shared,
                }
            }
            TypeKind::Tuple(items) => Ty::Tuple(items.iter().map(|i| self.lower_type(i)).collect()),
            TypeKind::Vector { elem, lanes } => match self.lower_type(elem) {
                Ty::Scalar(s) => Ty::Vector {
                    elem: s,
                    lanes: *lanes,
                },
                _ => {
                    self.error(
                        t.span,
                        "E0302",
                        "a SIMD vector element must be a scalar type",
                    );
                    Ty::Error
                }
            },
            TypeKind::Tensor { elem, dims, layout } => {
                let elem_ty = match self.lower_type(elem) {
                    Ty::Scalar(s) => s,
                    _ => {
                        self.error(t.span, "E0302", "a tensor element must be a scalar type");
                        Scalar::F32
                    }
                };
                // `lower_dim` now reports an undeclared dim name, so it needs `&mut self` — a plain
                // `map` would double-borrow, hence the explicit loop.
                let mut shape_dims = Vec::with_capacity(dims.len());
                for d in dims {
                    shape_dims.push(self.lower_dim(d));
                }
                let shape = Shape(shape_dims);
                let layout = match layout {
                    Some(wukong_ast::Layout::ColMajor) => Layout::ColMajor,
                    Some(wukong_ast::Layout::Strided) => Layout::Strided,
                    Some(wukong_ast::Layout::Tiled(v)) => Layout::Tiled(v.clone()),
                    _ => Layout::Contiguous,
                };
                Ty::Tensor {
                    elem: elem_ty,
                    shape,
                    layout,
                }
            }
        };
        self.check_layout(&ty, t.span);
        ty
    }

    fn lower_dim(&mut self, d: &wukong_ast::Dim) -> Dim {
        match &d.kind {
            DimKind::Int(n) => Dim::Const(*n),
            DimKind::Named(s) => {
                // A declared generic parameter is a *symbolic* dim (bound per call site). It takes
                // precedence over a same-named `const` (a generic shadows an outer const, matching
                // lexical scoping), so check it first and keep it a `Var`.
                if self.generics.contains(s) {
                    return Dim::Var(*s);
                }
                // A top-level `const` used as a dimension resolves to its integer *value* — a fixed
                // dim, exactly like an array length (`eval_usize` / mir_build's `const_usize_expr`).
                // `const D: usize = 3; Tensor[f32, D]` is a size-3 tensor, so an out-of-bounds index
                // `a[7]` is a compile-time error (E0501) and a mismatched concrete size is E0502 —
                // NOT a fresh unconstrained `Var(D)` that silently accepts any index or any size.
                // Without this the flagship shape check had a soundness hole: a const dim escaped
                // bounds/agreement checking entirely (interp trapped a[7] out-of-bounds while native
                // read past the buffer — a differential-gate divergence on plausible code). Mirrors
                // the array-length resolution above so the two agree on the value.
                if self.consts.contains_key(s) {
                    return Dim::Const(self.const_length(*s, 0));
                }
                // Neither a generic nor a const: an undeclared dim name is a typo, not a fresh
                // implicit dim. Previously `Tensor[f32, KK]` (when only `K` was declared) silently
                // introduced a brand-new symbolic dim `KK`, dropping the shared `K` constraint the
                // programmer meant — a shape hole the checker exists to catch. Report E0504 with a
                // did-you-mean hint. Gated on `checking_bodies` so it fires once per site, only after
                // every generic/`const` is known (see the flag's doc-comment).
                if self.checking_bodies {
                    let msg = match self.nearest_generic(*s) {
                        Some(g) => format!(
                            "unknown tensor dimension `{}`; did you mean `{}`? (a dimension must be \
                             a declared generic parameter, an integer literal, `?`, or a `const`)",
                            self.sym_str(*s),
                            self.sym_str(g)
                        ),
                        None => format!(
                            "unknown tensor dimension `{}` (a dimension must be a declared generic \
                             parameter, an integer literal, `?`, or a `const`)",
                            self.sym_str(*s)
                        ),
                    };
                    self.error(d.span, "E0504", msg);
                }
                Dim::Var(*s)
            }
            DimKind::Dynamic => Dim::Dynamic,
        }
    }

    /// The declared generic closest to an unknown dimension name `s`, for the did-you-mean hint: the
    /// minimum edit-distance candidate, tie-broken by name so the suggestion is deterministic across
    /// runs (`generics` is a `HashSet` with no stable order — the sweep-9 determinism discipline).
    /// Only a *reasonably* close name is proposed (distance within half the longer name, min 2), so
    /// an unrelated generic is not suggested.
    fn nearest_generic(&self, s: Symbol) -> Option<Symbol> {
        let want = self.sym_str(s);
        let mut best: Option<(usize, &str, Symbol)> = None;
        for &g in &self.generics {
            let name = self.sym_str(g);
            let dist = levenshtein(want, name);
            best = Some(match best {
                Some(b) if b.0 < dist || (b.0 == dist && b.1 <= name) => b,
                _ => (dist, name, g),
            });
        }
        let (dist, name, g) = best?;
        let threshold = (want.len().max(name.len()) / 2).max(2);
        (dist <= threshold).then_some(g)
    }

    fn eval_usize(&self, e: &Expr) -> u64 {
        self.eval_usize_depth(e, 0)
    }

    /// Fold a recorded initializer at the caller's exact remaining depth budget. Shared DAG
    /// dependencies are expanded once per (name, depth), not once per incoming expression path.
    /// No cache borrow survives the recursive call, so cycles still reach the ordinary depth cap.
    fn const_length(&self, name: Symbol, depth: u32) -> u64 {
        if depth > 64 {
            return 0;
        }
        if let Some(&value) = self.const_lengths.borrow().get(&(name, depth)) {
            return value;
        }
        let Some(init) = self.consts.get(&name) else {
            return 0;
        };
        let value = self.eval_usize_depth(init, depth);
        self.const_lengths.borrow_mut().insert((name, depth), value);
        value
    }

    fn record_const(&mut self, c: &'a ConstDecl) {
        if self
            .consts
            .get(&c.name.sym)
            .is_some_and(|init| std::ptr::eq(*init, &c.value))
        {
            // A valid declaration's body only changes type side tables. Retain cached folds
            // across that visit; duplicate declarations still replace the initializer in order.
            return;
        }
        self.consts.insert(c.name.sym, &c.value);
        self.const_lengths.get_mut().clear();
    }

    /// A fixed-size array's compile-time length, or `None` for an unreadable/out-of-range extent.
    /// A valid zero extent is `Some(0)`; callers use `Ty::Error` for recovery, never a fake length.
    /// MIR array extents are u32; lowering now folds the same
    /// u64 operands and checks the final conversion. Historically, `[i32; 0x1_0000_0001]` was bounds
    /// checked here against 4294967297 while the emitted slot held 1 element. Both point at
    /// the length expression and are gated on `checking_bodies` exactly like the E0504 dim check, so
    /// a signature re-lowered by `check_fn` is reported once (see the flag's doc-comment).
    fn array_len(&mut self, len: &Expr) -> Option<u64> {
        // The length expression must be one the compile-time folder can actually READ, and the same
        // `checking_bodies` gate applies as to the E0504 dim check for the same reason: `collect`
        // registers consts as it walks, so a const declared after this item is only guaranteed
        // visible in the body pass. `array_len_is_const` documents what happens when it cannot.
        if !self.array_len_is_const(len, 0, self.checking_bodies) {
            return None;
        }
        let n = self.eval_usize(len);
        if n > u32::MAX as u64 && self.checking_bodies {
            self.error(
                len.span,
                "E0401",
                format!(
                    "array length {n} is out of range: a fixed-size array may hold at most {} \
                     elements",
                    u32::MAX
                ),
            );
        }
        (n <= u32::MAX as u64).then_some(n)
    }

    /// Whether the array-length expression `e` is a form the compile-time folder can READ, reporting
    /// the first reason it cannot when `report` is set.
    ///
    /// This is the guard on the two-crate const-fold mirror. `eval_usize_depth` here and
    /// `mir_build`'s `const_usize_depth` fold the same four forms — an integer literal, a
    /// single-segment path naming a `const`, `Binary` arithmetic through the shared
    /// `BinOp::fold_const_len`, and an enum variant's discriminant — but they DISAGREE on
    /// everything else: sema ends `_ => 0` and types the annotation `[T; 0]`, while mir_build ends
    /// `_ => None` and `mir_ty_of_ann`'s Array arm turns that into a bare `MirType::Ptr`. So
    /// `let mut a: [i32; 4 as usize]` emitted `v0 = alloca ptr` and then STORED THROUGH the
    /// uninitialized pointer: the interpreter reported `error: expected a pointer` and exited 1
    /// while `--backend=native` printed a value after writing through a garbage stack address —
    /// an accepted program on which the two Executed backends disagree, and a memory-safety
    /// violation on native. `4 as usize` and `sizeof(i64)` are both spellings a user writes without
    /// thinking twice. Report the unreadable length instead; the `None => MirType::Ptr` arm then
    /// becomes unreachable for any program that reaches lowering and stays as the recovery path.
    ///
    /// Three named cases, so the message says which one was hit:
    ///  * a **name that resolves to nothing** — the pre-existing `E0301`, now also found inside
    ///    arithmetic (`fn fill(mut a: [i32; BOGUS + 8])` compiled and RAN, writing 8 elements into
    ///    the caller's buffer, because the old check only looked at a length that was DIRECTLY a
    ///    single-segment path);
    ///  * a **declared generic** — `[T; N]` has no representation (`Ty::Array` carries a `u64`
    ///    len, unlike `Ty::Tensor`'s `Dim::Var`), so it folded to 0 here and to `MirType::Ptr`
    ///    there. It was previously exempted from the name check and then silently evaluated to 0;
    ///  * **any other form** — a cast, `sizeof`/`alignof`, a call, a unary op, a multi-segment
    ///    path, a struct field, or a `const` whose own initializer is unreadable.
    ///
    /// Recursion into a `const`'s initializer does NOT report (that declaration is checked where it
    /// is written); it only decides foldability, and the length site reports against the const name.
    fn array_len_is_const(&mut self, e: &Expr, depth: u32, report: bool) -> bool {
        // Deeper than any real const chain. Both mirrors stop here and they stop DIFFERENTLY:
        // `eval_usize_depth` yields 0 (so sema types the annotation `[T; 0]`) while mir_build's
        // `const_usize_depth` yields `None` (so `mir_ty_of_ann` emits a bare `MirType::Ptr` slot).
        // That IS the desync this function exists to prevent, so the honest answer is "not
        // readable" — reporting a second diagnostic on a cyclic `const` that `check_recursive_consts`
        // already flagged as E0403 is a far smaller cost than a silently uninitialized slot, and a
        // 65-deep chain that is NOT cyclic gets no other diagnostic at all.
        if depth > 64 {
            if report {
                self.error(
                    e.span,
                    "E0406",
                    "array length is nested more than 64 levels deep, past what the compile-time \
                     folder reads; if these `const`s refer to one another, that cycle is the real \
                     error",
                );
            }
            return false;
        }
        match &e.kind {
            ExprKind::Int(_) => true,
            ExprKind::Path(p) if p.is_single() => {
                let sym = p.first().sym;
                // `self.consts` holds AST references borrowed from the module, not from `self`, so
                // the copy here outlives the `&mut self` calls below.
                // A generic shadows a top-level const, just as in lower_dim. Its unreadable
                // extent must not accidentally become that const's (possibly zero) length.
                if let Some(init) = self
                    .consts
                    .get(&sym)
                    .copied()
                    .filter(|_| !self.generics.contains(&sym))
                {
                    if self.array_len_is_const(init, depth + 1, false) {
                        return true;
                    }
                    if report {
                        let nm = self.sym_str(sym).to_string();
                        self.error(
                            e.span,
                            "E0406",
                            format!(
                                "array length `{nm}` is not a compile-time constant: this \
                                 `const`'s initializer is not an integer literal, another `const`, \
                                 an enum variant, or arithmetic on those"
                            ),
                        );
                    }
                    return false;
                }
                if self.generics.contains(&sym) {
                    if report {
                        let nm = self.sym_str(sym).to_string();
                        self.error(
                            e.span,
                            "E0406",
                            format!(
                                "array length `{nm}` is not a compile-time constant: it is a \
                                 generic parameter, and a fixed-size array's length is part of its \
                                 type — use `Tensor[T, {nm}]`, whose dimension is bound per \
                                 call site, or a `const`"
                            ),
                        );
                    }
                    return false;
                }
                if self.lookup_local(sym).is_some() {
                    if report {
                        let nm = self.sym_str(sym).to_string();
                        self.error(
                            e.span,
                            "E0406",
                            format!("array length `{nm}` is a runtime value, not a compile-time constant"),
                        );
                    }
                    return false;
                }
                if report {
                    let nm = self.sym_str(sym).to_string();
                    self.error(
                        e.span,
                        "E0301",
                        format!(
                            "cannot find `{nm}` in this scope (an array length must be an integer \
                             literal, a `const`, an enum variant, or arithmetic on those)"
                        ),
                    );
                }
                false
            }
            // Both operands must be readable. `fold_const_len` treats an unreadable operand as 0 on
            // BOTH sides of the mirror, so the length silently becomes a number the source never
            // wrote — `[i32; (4 as usize) + 1]` is 1, not 5, and both backends agree on it,
            // which is exactly the shape of defect no differential gate can see. Report each bad
            // operand rather than only the first.
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.array_len_is_const(lhs, depth + 1, report);
                let r = self.array_len_is_const(rhs, depth + 1, report);
                if !(l && r) {
                    return false;
                }
                // Readable operands are still not a readable LENGTH. `fold_const_len` has no value
                // for a comparison or a logical operator, nor for checked arithmetic that fails
                // (`2 - 5`, a division by zero, an operand or result above `MAX_CONST_ARRAY_LEN`)
                // — and it reports every one of those as the same 0 that a genuine `[T; 2 - 2]`
                // gives, on BOTH sides of the mirror. So `[i32; 2 - 5]` and `[i32; 4 > 2]` sized a
                // zero-length array with no diagnostic: a number the source never wrote that both
                // backends agree on, the gate-blind class. Ask the shared folder directly.
                let lv = self.eval_usize_depth(lhs, depth + 1);
                let rv = self.eval_usize_depth(rhs, depth + 1);
                if op.try_fold_const_len(lv, rv).is_none() {
                    if report {
                        let g = op.glyph();
                        self.error(
                            e.span,
                            "E0406",
                            format!(
                                "array length `{lv} {g} {rv}` has no compile-time value: `{g}` \
                                 does not fold to a length (a comparison or a logical operator is \
                                 never a length, and the arithmetic must stay within \
                                 0..={MAX_CONST_ARRAY_LEN})"
                            ),
                        );
                    }
                    return false;
                }
                true
            }
            // `[i32; E::V]` — a C-style enum variant's discriminant, the fourth foldable form.
            // A negative discriminant is the 0 fallback in both mirrors, so it is not readable.
            ExprKind::Field { .. } => {
                if self.enum_variant_disc(e).is_some_and(|d| d >= 0) {
                    return true;
                }
                if report {
                    self.error(
                        e.span,
                        "E0406",
                        "an array length must be a compile-time constant (an integer literal, a \
                         `const`, a non-negative enum variant, or arithmetic on those)",
                    );
                }
                false
            }
            _ => {
                if report {
                    self.error(
                        e.span,
                        "E0406",
                        "an array length must be a compile-time constant (an integer literal, a \
                         `const`, an enum variant, or arithmetic on those); a cast, `sizeof`, a \
                         call or any other expression cannot be folded to a length",
                    );
                }
                false
            }
        }
    }

    /// If `e` is an enum-variant access `E::V` (a `Field` whose base is a single-segment path naming
    /// a declared enum), its integer discriminant — the value the variant lowers to. Payload-ness is
    /// not consulted (a data-carrying variant has a discriminant too; using one *bare* is rejected
    /// separately as an incomplete constructor). Mirrors `mir_build`'s `enum_variant_value` exactly,
    /// so compile-time evaluation (array lengths, index bounds) agrees with what lowering emits.
    fn enum_variant_disc(&self, e: &Expr) -> Option<i64> {
        let ExprKind::Field { base, name } = &e.kind else {
            return None;
        };
        let ExprKind::Path(p) = &base.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let DefKind::Enum(variants) = &self.defs.lookup(p.first().sym)?.kind else {
            return None;
        };
        variants.iter().find(|v| v.name == name.sym).map(|v| v.disc)
    }

    /// Evaluate a compile-time array length (or a `const` tensor dimension): a plain integer literal,
    /// a single-segment path naming a top-level `const` whose initializer is itself such a length (so
    /// `const N: usize = 4; [i32; N]` sizes the array), constant arithmetic folded through the shared
    /// `BinOp::fold_const_len`, or an enum variant's discriminant. Anything else — and any
    /// negative discriminant — is the `0` fallback, not an error (`array_len` reports the unresolvable
    /// *name* case). `self.consts` is populated in `collect` before any body is checked, so this is
    /// order-independent. `mir_build`'s `const_array_len_expr`/`const_usize_depth` mirrors this arm for
    /// arm — the two must agree on the length, else the slot size desyncs from these bounds checks.
    /// The depth bound guards against a cyclic const initializer (also rejected by
    /// `check_recursive_consts`).
    fn eval_usize_depth(&self, e: &Expr, depth: u32) -> u64 {
        if depth > 64 {
            return 0;
        }
        match &e.kind {
            // Decode the literal with the SAME rules mir_build's mirrored `const_usize_depth` uses
            // (`parse_int`): radix prefixes `0x`/`0o`/`0b`, `_` digit separators, an integer type
            // suffix. Keeping only the leading run of decimal digits desynced the mirror at the
            // literal leaf — `[i32; 0x10]` was length 0 here and 16 there, `[i32; 1_6]` was 1 here
            // and 16 there — so the alloca'd slot and these bounds checks disagreed: either a valid
            // program rejected as "length 0", or (when sema's short length reached a struct layout)
            // silent stack corruption both backends agreed on.
            ExprKind::Int(s) => parse_u64_text(self.sym_str(*s)).unwrap_or(0),
            ExprKind::Path(p) if p.is_single() => self.const_length(p.first().sym, depth + 1),
            // Const arithmetic in a length (`[i32; 2+2]`, `[i32; N+1]`): fold via the shared
            // `BinOp::fold_const_len` that mir_build's `const_usize_expr` also uses, so the slot size
            // and this bounds check agree. Previously this fell to `_ => 0`, sizing the array 0 (an
            // unsized bare pointer → native segfault / interp != native / -O0 != -O2) and spuriously
            // rejecting valid code as "length 0".
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.eval_usize_depth(lhs, depth + 1);
                let r = self.eval_usize_depth(rhs, depth + 1);
                op.fold_const_len(l, r)
            }
            // A C-style enum variant as a length (`[i32; E::V]`, or via `const K: E = E::V`)
            // resolves to its discriminant. Previously this fell to `_ => 0` — a silent
            // zero-size while mir_build sized the slot differently. Mirrored in mir_build's
            // `const_usize_depth`. A negative discriminant stays the 0 fallback.
            ExprKind::Field { .. } => self
                .enum_variant_disc(e)
                .and_then(|d| u64::try_from(d).ok())
                .unwrap_or(0),
            _ => 0,
        }
    }

    // ---- Body checking ----

    fn check_bodies(&mut self, module: &'a Module) {
        for item in &module.items {
            match &item.kind {
                ItemKind::Fn(f) => {
                    if let Some(body) = &f.body {
                        self.check_fn(f, body, &item.attrs);
                    }
                }
                ItemKind::Const(c) => self.check_const(c, item.span),
                _ => {}
            }
        }
    }

    /// Type-check a top-level `const`'s initializer against its annotation (like a `let`, so an
    /// unsuffixed literal adapts to the annotated type) and record the initializer so `mir_build`
    /// can inline it at each use site. Evaluated at module scope (a const may reference another
    /// const by name, resolved through the def map, not local scopes).
    fn check_const(&mut self, c: &'a ConstDecl, span: Span) {
        self.generics.clear();
        self.scopes.clear();
        self.scopes.push(HashMap::default());
        self.immutable_locals.clear();
        self.immutable_locals.push(HashSet::default());
        self.immutable_params.clear();
        self.immutable_params.push(HashSet::default());
        self.param_tys.clear();
        let ann = self.lower_type(&c.ty);
        let vty = self.type_expr(&c.value);
        if self.let_compatible(&ann, &c.value, &vty) {
            // Only re-stamp an all-literal initializer (see the `let` path): re-stamping a
            // `compatible`-only mixed-width expression like `const C: i64 = A + B` would drop the
            // narrow operand's widening and produce ill-typed MIR (an -O0 verify ICE / -O1+ panic).
            if self.literal_adapts(&ann, &c.value) {
                self.retype_adapted_literal(&c.value, &ann);
                self.range_check_int_literal(&c.value, &ann);
            }
        } else {
            self.error(
                span,
                "E0401",
                format!(
                    "type mismatch: const `{}` is annotated `{}` but the value is `{}`",
                    self.sym_str(c.name.sym),
                    ann.display(self.interner),
                    vty.display(self.interner)
                ),
            );
        }
        self.record_const(c);
    }

    fn check_fn(&mut self, f: &FnDecl, body: &Block, item_attrs: &[Attr]) {
        self.parallel_fn = item_attrs
            .iter()
            .any(|a| self.interner.resolve(a.name.sym) == "parallel");
        self.generics = generic_names(&f.generics);
        self.scopes.clear();
        self.scopes.push(HashMap::default());
        // Mirror the scope reset for immutability tracking (these reset `scopes` directly instead of
        // via `push_scope`, so the two stacks would otherwise desync and the check never fires).
        self.immutable_locals.clear();
        self.immutable_locals.push(HashSet::default());
        self.immutable_params.clear();
        self.immutable_params.push(HashSet::default());
        self.param_tys.clear();
        // Two parameters may not share a name: the second would silently shadow the first in the
        // body scope (a `fn f(a: i32, a: i64)` ran, with `a` resolving to the second), which is a
        // quiet footgun. Duplicate top-level `fn`s are already E0300; parameters get the same code.
        let mut seen_params: HashSet<Symbol> = HashSet::default();
        for p in &f.params {
            if !seen_params.insert(p.name.sym) {
                let nm = self.sym_str(p.name.sym).to_string();
                self.error(
                    p.name.span,
                    "E0300",
                    format!("duplicate parameter name `{nm}`"),
                );
            }
            let ty = self.lower_type(&p.ty);
            // A parameter without `mut` is immutable: track it so reassigning it (`p = …`) or
            // mutating it through a projection (`p.f = …` on an aggregate, which — passed by
            // reference — would reach the caller) is E0304. Record its type either way, so the
            // through-projection check can tell an aggregate (leaks) from a pointer (mutates a
            // pointee, legitimate).
            if !p.mutable {
                if let Some(set) = self.immutable_params.last_mut() {
                    set.insert(p.name.sym);
                }
            }
            self.param_tys.insert(p.name.sym, ty.clone());
            self.bind(p.name.sym, ty);
        }
        self.ret_ty = match &f.ret {
            Some(t) => self.lower_type(t),
            None => Ty::Unit,
        };
        let body_ty = self.type_block(body);
        // The body's tail expression is the implicit return — check its shape against the declared
        // return type (an explicit `return` is checked at its `StmtKind::Return` site).
        if let Some(tail) = &body.tail {
            self.check_return_shape(&body_ty, tail, tail.span);
        }
        // Definite return: a function that promises a value must produce one on every path. If the
        // body can fall off its end (no trailing tail/return, an `if` with no `else`, a breakable or
        // non-exhaustive construct), the value it "returns" is an uninitialized default on both
        // backends — a silent wrong answer. Require a return on all paths, like Rust. Conservative:
        // it only fires when a fall-through path is *certain* (`block_diverges` errs toward "returns"),
        // so a function that does return on every path is never flagged; `()`/`Unknown`/`Error`
        // returns are exempt.
        // The body returns a value if its tail produces one (the block type is non-unit) or every
        // statement path diverges before the (unit) tail.
        let returns_value = !matches!(body_ty, Ty::Unit) || block_diverges(body);
        if !matches!(self.ret_ty, Ty::Unit | Ty::Unknown | Ty::Error) && !returns_value {
            self.error(
                body.span,
                "E0401",
                "not all control-flow paths return a value (this function can fall off its end)"
                    .to_string(),
            );
        }
        self.generics.clear();
    }

    /// Check a returned value's type against the declared return type `self.ret_ty`. It first
    /// ADAPTS an unsuffixed literal to the declared return type (re-stamp + range check, below),
    /// then runs three checks, in order: tensor / vector SHAPE agreement in RIGID mode — a function
    /// must not lie about its output shape, since callers propagate the declared return shape into
    /// downstream shape checks (a single wrong return silently poisons every caller); a
    /// scalar-vs-pointer/aggregate KIND clash; and scalar-type agreement for a non-literal value.
    /// An unsuffixed literal still adapts, so this never over-fires on e.g. `return 5` from
    /// `-> i64`; every other kind stays lenient.
    ///
    /// Both return spellings land here — the explicit `return e;` statement and the body's implicit
    /// TAIL expression — so the adaptation is identical for the two, which it was not before: the
    /// tail form had no range check at all.
    fn check_return_shape(&mut self, val_ty: &Ty, val: &Expr, span: Span) {
        let ret = self.ret_ty.clone();
        // ADAPT FIRST, and adapt COMPLETELY. `literal_adapts` is asked twice further down as a
        // LENIENCY — once inline in the scalar-agreement check, once inside `aggregate_value_clash`
        // — and it is what makes `return 5` from `-> i64` and `fn g() -> [i8; 2] { return [127, 0]; }` legal
        // even though the literal's own type is i32 / `[i32; 2]`. Deciding a literal is ACCEPTED
        // and never recording at what WIDTH it lowers is the silent half of that leniency, which is
        // why every other adapting site (`let`, `const`, a struct-literal field, an assignment, a
        // call argument) pairs the predicate with `retype_adapted_literal` and
        // `range_check_int_literal` — all three, always together:
        //
        //   * without the re-stamp, `fn g() -> (f64, f64) { (0.1, 0.2) }` was accepted and then
        //     lowered as an 8-byte `(f32, f32)` into the caller's 16-byte tuple: `load out of
        //     bounds` on the interpreter, `0.000000013411048210842936` / `NaN` on native — an
        //     interp-vs-native divergence on a clean compile. Scalars lied more quietly:
        //     `fn a() -> f64 { 0.1 }` returned the f32 0.1 widened, `0.10000000149011612`, where
        //     `let a: f64 = 0.1;` returns `0.1`.
        //   * without the range check, the re-stamp itself becomes the hazard, because a re-stamped
        //     literal is materialized as a `ConstInt` OF THE NARROW TYPE instead of an i32 the
        //     backend truncates. `fn a() -> i8 { 300 }` would emit `ConstInt(300, i8)`: wukong's own
        //     verifier only checks that the type is integral, the interpreter masks the value to 44,
        //     and Cranelift's verifier REJECTS it (`iconst_bounds` bounds-masks an `I8` immediate at
        //     `u8::MAX`) — so the interpreter would run a program native cannot build. The explicit
        //     `return` spelling has range-checked since before this branch; the implicit TAIL return
        //     never did, and `fn a() -> i8 { 200 }` silently returned -56.
        if self.literal_adapts(&ret, val) {
            self.retype_adapted_literal(val, &ret);
        }
        self.range_check_int_literal(val, &ret);
        if matches!(ret, Ty::Tensor { .. } | Ty::Vector { .. })
            || matches!(val_ty, Ty::Tensor { .. } | Ty::Vector { .. })
        {
            let mut dims = HashMap::default();
            // Body context: the declared return shape and the returned value's shape are both fully
            // determined, and this function's own generic dims are RIGID — so a generic function
            // cannot declare a return shape its body does not actually produce. (`rigid == true`.)
            self.unify(&ret, val_ty, &mut dims, span, true);
        }
        // Returning a pointer/aggregate where a scalar is declared (or vice versa) is not a numeric
        // coercion — the native backend builds a mismatched return ABI and ICEs (e.g. `return &a`
        // from `-> i32`), while the interpreter silently adapts: a divergence.
        // `aggregate_type_clash` is the second half: `check_return_shape`'s `unify` above runs
        // only for a Tensor/Vector on one side, so an array length, a tuple arity or a struct
        // identity in the return type was never compared against the value's — the callee could
        // declare `-> [i32; 8]` and return a `[i32; 2]`, and every caller then believed the lie.
        // The value's EXPRESSION is passed as well: an array/tuple literal adapts to the declared
        // return type exactly as it does to a `let` annotation, so `fn g() -> [i8; 2] { return
        // [127, 0]; }` stays accepted even though the literal's own type is `[i32; 2]`.
        if self.scalar_aggregate_clash(&ret, val_ty)
            || self.aggregate_value_clash(&ret, val_ty, val)
        {
            self.error(
                span,
                "E0401",
                format!(
                    "type mismatch: this function returns `{}`, but a value of type `{}` is \
                     returned here",
                    ret.display(self.interner),
                    val_ty.display(self.interner)
                ),
            );
        }
        // Scalar-type agreement, the same rule `let`/call-args enforce: a NON-literal value of a
        // different scalar type (`return x` where `x: i64` from a `-> i32` fn, or an `f32` from a
        // `-> i32`) was silently truncated/demoted — both backends agreeing on the lossy value, so
        // the differential gate was blind — where the language otherwise demands an explicit `as`.
        // An unsuffixed literal still adapts (`return 5` from `-> i64`), and an out-of-range literal
        // is caught by `range_check_int_literal`; only a genuinely mismatched non-literal errors.
        if let (Ty::Scalar(rs), Ty::Scalar(vs)) = (&ret, val_ty) {
            if rs != vs && !self.literal_adapts(&ret, val) {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "type mismatch: this function returns `{}`, but a value of type `{}` is \
                         returned here (use `as {}` to convert)",
                        ret.display(self.interner),
                        val_ty.display(self.interner),
                        rs.name()
                    ),
                );
            }
        }
    }

    /// An elementwise binary operator requires its operand *shapes* to agree: adding two tensors of
    /// different shape (`Tensor[f32,2,3] + Tensor[f32,3,2]`) is meaningless, yet the result-type
    /// `join` picks one operand and lets it through. This is the RIGID-mode shape check for every
    /// *body* context where two shapes meet — binary operands, an assignment's place and value,
    /// `if`/`match` arm merges, and a `loop`'s `break` values — while `check_fn_call` covers call
    /// arguments and `check_return_shape` the return. Only fires when *both* sides are tensors, or
    /// both are vectors: a tensor/scalar pairing stays lenient (scalar broadcast), and two scalars
    /// promote via `join` (mixed precision like `(i as f64) + 1.0` must not error).
    fn check_binop_shapes(&mut self, l: &Ty, r: &Ty, span: Span) {
        if matches!(
            (l, r),
            (Ty::Tensor { .. }, Ty::Tensor { .. }) | (Ty::Vector { .. }, Ty::Vector { .. })
        ) {
            let mut dims = HashMap::default();
            // Body context (operator operands, assignment, if/match arm merge): both operand shapes
            // are fully determined and this function's generic dims are RIGID, so distinct generics
            // (`Tensor[f32,M] + Tensor[f32,N]`) no longer "unify" by binding one to the other.
            // (`rigid == true`.)
            self.unify(l, r, &mut dims, span, true);
        }
    }

    /// Whether the named type `sym` is a **data-carrying** (tagged-union) enum — a declared enum with
    /// at least one payload (tuple/struct) variant. Such a value is a discriminant + padded payload
    /// byte buffer (an aggregate, addressed by base pointer), unlike a C-style all-unit enum, whose
    /// value is its i32 discriminant scalar. The distinction drives layout, `==`, and cast validity.
    fn enum_is_data_carrying(&self, sym: Symbol) -> bool {
        matches!(
            self.defs.lookup(sym).map(|d| &d.kind),
            Some(DefKind::Enum(vs)) if vs.iter().any(|v| !v.payload.is_unit())
        )
    }

    /// Whether `t` is an aggregate with no scalar value — a tuple, a fixed-size array, a slice
    /// (a fat `(ptr, len)` view), a *struct*, or a **data-carrying enum** (a tagged union). A
    /// *C-style* (all-unit) enum is **not** an aggregate: a variant is its integer discriminant, so
    /// enum equality is well-defined. Used to reject `==`/`!=` on values whose MIR is a base pointer.
    fn is_aggregate_ty(&self, t: &Ty) -> bool {
        match t {
            Ty::Tuple(_) | Ty::Array { .. } | Ty::Slice(_) => true,
            Ty::Named(n) => {
                matches!(
                    self.defs.lookup(*n).map(|d| &d.kind),
                    Some(DefKind::Struct(_))
                ) || self.enum_is_data_carrying(*n)
            }
            _ => false,
        }
    }

    /// A value that no unary operator and no binary operator besides `==`/`!=` is defined on: an
    /// aggregate (struct/tuple/array), `()` (the unit / a void-returning call's result), or a
    /// function value. Each used to reach mir_build and lower to an op on a base pointer / a dummy
    /// `i32` / a `void` operand — verifier-invalid MIR that crashed the native backend while the
    /// interpreter ran on the placeholder and returned a silently-wrong value (e.g. `nothing() * 2`,
    /// `-nothing()`). Scalars/vectors/tensors/pointers and `Unknown` are *not* flagged.
    fn is_noncomputable_operand(&self, t: &Ty) -> bool {
        self.is_aggregate_ty(t) || matches!(t, Ty::Unit | Ty::Fn { .. })
    }

    /// Type a branch/loop condition and reject a `()` (unit / void-returning call) condition: it
    /// lowers to a `cond_br` on a non-`i1` dummy value — MIR the verifier and Cranelift reject (the
    /// native backend crashed) while the interpreter branched on the placeholder. A non-bool numeric
    /// "truthy" condition like `if 5` is intentionally allowed and stays unaffected.
    fn check_condition(&mut self, cond: &Expr) {
        let t = self.type_expr(cond);
        if matches!(t, Ty::Unit) {
            self.error(
                cond.span,
                "E0401",
                "a condition cannot be `()` (a unit / void value); use a boolean or numeric value"
                    .to_string(),
            );
        } else if self.is_aggregate_ty(&t)
            || matches!(t, Ty::Fn { .. } | Ty::Tensor { .. } | Ty::Vector { .. })
        {
            // A pointer condition is a null test and a number is "nonzero"; an aggregate, tensor
            // view, SIMD vector or function value has no truth value. Each used to reach `cond_br`
            // as a base pointer or vector and crash both backends with raw, unspanned verifier
            // text — the same failure the pointer case had before it became a null test.
            self.error(
                cond.span,
                "E0401",
                format!(
                    "a condition must be a boolean, a number or a pointer, found `{}`",
                    t.display(self.interner)
                ),
            );
        }
    }

    /// True when `a` and `b` are two CONCRETE aggregates that cannot be the same layout — the
    /// byte-reinterpret pairing for which this language has no conversion.
    ///
    /// `scalar_aggregate_clash` below catches scalar-versus-aggregate; every AGGREGATE-versus-
    /// aggregate pairing fell through every body check, because `check_binop_shapes` opens with
    /// `if matches!((l, r), (Tensor, Tensor) | (Vector, Vector))` and is a no-op otherwise. So
    /// `a = b` between two unrelated structs was accepted and reinterpreted the bytes: the
    /// interpreter printed `1` (an `i32` field) where native printed `1065353216` (the bits of
    /// `1.0f32`) — an interp-vs-native divergence on a clean compile. The same hole let a callee
    /// LIE about its declared return type (`fn g() -> [i32; 8] { let a: [i32; 2] = [5, 6]; return a; }`
    /// compiled, and every caller then believed `[i32; 8]`), and let an `if`/`match` arm merge or a
    /// `break` merge two different aggregates into one value.
    ///
    /// The relation is `compatible`, the structural half of the rule `let` already applies through
    /// `let_compatible` — and it is **DIRECTIONAL**, `compatible(place, value)`, not a disjunction
    /// over both orders. `compatible` encodes the language's one implicit aggregate coercion,
    /// `[T; N]` unsizing to `[]T`, in a single direction on purpose (a slice carries no static
    /// length to restore, so there is no slice→array direction). An earlier form of this helper
    /// asked `!(compatible(a, b) || compatible(b, a))` and threw that intent away, re-opening the
    /// forbidden direction in the two contexts where direction is not an accident:
    /// `fn g() -> [f32; 4] { return s; }` for `s: []f32` was accepted, and every caller then read
    /// the 16-byte fat-pointer header as four f32 elements — both backends printing 0, a value the
    /// source never wrote. `let r: [f32; 4] = s;` has always been rejected, and `return`/`=` must
    /// not be weaker than `let`.
    ///
    /// This is the TYPE half only. A place-vs-value site must also grant the literal adaptation
    /// `let` grants — see [`Sema::aggregate_value_clash`] — and a *merge*, where neither side is
    /// the place, is symmetric — see [`Sema::aggregate_merge_clash`].
    ///
    /// Deliberately narrower than `let`'s rule, because this fires in contexts the fixture corpus
    /// exercises far more thinly: only when both sides are aggregates whose layout is fully
    /// determined here (see `is_concrete_layout`). A generic type variable, a tensor, and
    /// `Unknown`/`Error` all stay lenient, per the crate-wide rule that errors are reserved for
    /// what can be PROVED wrong.
    fn aggregate_type_clash(&self, place: &Ty, value: &Ty) -> bool {
        if place.is_unknown() || value.is_unknown() || place.is_error() || value.is_error() {
            return false;
        }
        // A fixed extent/arity is known even when a leaf element is a generic type variable.
        // In particular [T; 0] cannot become [T; 1] while checking a generic function body.
        if fixed_aggregate_shape_clash(place, value) {
            return true;
        }
        if !self.is_concrete_aggregate(place) || !self.is_concrete_aggregate(value) {
            return false;
        }
        !compatible(place, value)
    }

    /// The place-versus-value form, for a `return` and for an assignment: the directional type
    /// clash above, minus the literal adaptation `let` already grants.
    ///
    /// An array/tuple LITERAL of unsuffixed numeric elements takes the annotation's element type
    /// (`let a: [i8; 2] = [127, 0];` and `let t: (u8, u8) = (200, 1);` are deliberately accepted —
    /// see the `ArrayLit`/`ArrayRepeat`/`TupleLit` arms of [`Sema::literal_adapts`]), while the
    /// literal's OWN type is `[i32; 2]` / `(i32, i32)`, because an unsuffixed integer literal
    /// defaults to i32 and a float literal to f32. Comparing types alone therefore made
    /// `fn g() -> [i8; 2] { return [127, 0]; }` a hard error at `return`, and `a = [127, 0];` a
    /// hard error at an assignment, while the identical spelling compiled at a `let` — previously
    /// valid programs rejected by a check that exists to catch layout LIES, not literals. The
    /// relation here is exactly `let_compatible`'s, narrowed to concrete aggregates.
    fn aggregate_value_clash(&self, place: &Ty, value_ty: &Ty, value: &Expr) -> bool {
        self.aggregate_type_clash(place, value_ty) && !self.literal_adapts(place, value)
    }

    /// The merge form, for an `if`/`match` arm merge and a `loop`'s `break` values. Here neither
    /// side is the place — which arm the checker visits first is an accident of the source — so the
    /// array→slice unsizing counts in EITHER direction, unlike a `return` or an assignment. This is
    /// the one context in which the symmetric disjunction is the right relation.
    ///
    /// `incoming` is the expression that produced `r` — the arm being folded in — so an adapting
    /// array/tuple literal there is accepted for the same reason it is at a `let`. The literal
    /// adapts to the type already ESTABLISHED on the left, exactly as it adapts to a `let`'s
    /// annotation, and `l` really is the established type: `join` returns the left operand for
    /// every aggregate, so the merge's type is the first arm's. Hence only one direction is
    /// literal-aware, while the STRUCTURAL half above stays symmetric. A literal in the FIRST arm
    /// (`if c { [127, 0] } else { a }` for `a: [i8; 2]`) therefore establishes `[i32; 2]` and the
    /// `[i8; 2]` arm cannot hold it — which is a real element-width lie, not an adaptation, and it
    /// is rejected by the pre-existing `let` check as well whenever the result is annotated.
    ///
    /// Takes `&mut self` because GRANTING that leniency and RE-STAMPING the literal at the merged
    /// width are ONE operation, not two — they are written in the same branch, below, so they
    /// cannot drift apart. Split across a predicate here and a `retype_adapted_literal` at the call
    /// site (which is how `let` / assignment / call-argument spell it), the pair is one edit away
    /// from becoming an acceptance with no width, which is exactly what shipped:
    /// `let r: (f64, f64) = if c { a } else { (0.1, 0.2) };` compiled, and the `(f32, f32)` arm wrote
    /// 8 bytes into the 16-byte merge slot — `load out of bounds` on the interpreter,
    /// `0.000000013411048210842936` / `NaN` on native.
    fn aggregate_merge_clash(&mut self, l: &Ty, r: &Ty, incoming: Option<&Expr>) -> bool {
        if !(self.aggregate_type_clash(l, r) && self.aggregate_type_clash(r, l)) {
            return false;
        }
        let Some(arm) = incoming else {
            return true;
        };
        if !self.literal_adapts(l, merge_arm_value(arm)) {
            return true;
        }
        // Adapted: re-stamp the arm at the merged type `l` so lowering stores it at that width, and
        // range-check it so the re-stamp cannot mint a `ConstInt` too wide for its own type. The
        // two calls are handed DIFFERENT nodes on purpose. The re-stamp gets the whole ARM, because
        // the arm is normally a block whose recorded type is its tail's and both have to move
        // together — `retype_adapted_literal` peels the block itself, exactly as `merge_arm_value`
        // does. The range check gets the PEELED node, because it dispatches on the expression kind:
        // a `Block` matches none of its `ArrayLit`/`ArrayRepeat`/`TupleLit` arms, so it falls past
        // the element-wise recursion and then out at the `Ty::Scalar` guard — and `l` is always an
        // aggregate here. Handed the arm, it would silently check nothing. Then it is NOT a clash.
        self.retype_adapted_literal(arm, l);
        self.range_check_int_literal(merge_arm_value(arm), l);
        false
    }

    /// An aggregate — array, tuple, slice, or a DECLARED nominal type — whose layout is fully
    /// determined by the type alone. A bare generic type variable is also spelled `Ty::Named`, so a
    /// name that resolves to no declared struct/enum, or that is a generic parameter in scope, is
    /// not one of these.
    fn is_concrete_aggregate(&self, t: &Ty) -> bool {
        match t {
            Ty::Array { .. } | Ty::Tuple(_) | Ty::Slice(_) => self.is_concrete_layout(t),
            Ty::Named(n) => self.is_declared_named_ty(*n) && !self.generics.contains(n),
            _ => false,
        }
    }

    /// Whether every leaf of `t` has a layout known here. Tensors are excluded on purpose: their
    /// agreement is the shape unifier's job (`unify` in rigid mode), which reports the dimension by
    /// name. `Fn`, `Unknown` and `Error` are excluded by the leniency rule. A zero extent is real:
    /// unreadable extents are `Ty::Error`, never a numeric sentinel. Empty literals have an unknown
    /// element type but a known zero-byte layout, so their extent must still agree.
    fn is_concrete_layout(&self, t: &Ty) -> bool {
        match t {
            Ty::Unit | Ty::Scalar(_) | Ty::Vector { .. } => true,
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } => self.is_concrete_layout(pointee),
            Ty::Slice(e) => self.is_concrete_layout(e),
            Ty::Array { elem, len, .. } => *len == 0 || self.is_concrete_layout(elem),
            Ty::Tuple(items) => items.iter().all(|i| self.is_concrete_layout(i)),
            Ty::Named(n) => self.is_declared_named_ty(*n) && !self.generics.contains(n),
            _ => false,
        }
    }

    /// The RIGID agreement check for a *merge*: an `if`/`match` arm merge or a `loop`'s `break`
    /// values, where one value has to hold what every branch produced. `check_binop_shapes` covers
    /// the tensor/vector half; this adds the aggregate half, which had no check at all — a
    /// `let r: A = match k { 0 => a, _ => b }` over two different structs printed `1` on the
    /// interpreter and `1065353216` on native.
    fn check_merge_shapes(&mut self, l: &Ty, r: &Ty, incoming: Option<&Expr>, span: Span) {
        self.check_binop_shapes(l, r, span);
        if self.aggregate_merge_clash(l, r, incoming) {
            self.error(
                span,
                "E0401",
                format!(
                    "type mismatch: `{}` and `{}` are different aggregate types, so one value \
                     cannot hold both",
                    l.display(self.interner),
                    r.display(self.interner)
                ),
            );
        }
    }

    /// True when `a` and `b` are concrete types of incompatible *kind* — one a scalar/vector, the
    /// other a pointer/reference/array/tuple. Such a pairing is never a numeric coercion: it slips
    /// past the lenient checks and then ICEs the native backend (an i64 pointer marshalled into a
    /// 32-bit slot) or silently reinterprets the bytes — a backend divergence. Mirrors the
    /// call-argument cross-kind arm in `unify`. Unknown/Error/Named(generic)/Tensor stay lenient.
    fn scalar_aggregate_clash(&self, a: &Ty, b: &Ty) -> bool {
        let scalarish = |t: &Ty| matches!(t, Ty::Scalar(_) | Ty::Vector { .. });
        let pointerish = |t: &Ty| {
            matches!(
                t,
                Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Array { .. } | Ty::Tuple(_) | Ty::Slice(_)
            )
        };
        (scalarish(a) && pointerish(b)) || (pointerish(a) && scalarish(b))
    }

    /// Validate an `as` cast. The cast operator was an unchecked reinterpret between *any* two types:
    /// `16 as *i32` then `*p` is arbitrary-memory UB (native SIGSEGVs, the interpreter traps — a
    /// divergence), `&a as i64` reads a real address on native but `0` on the interpreter, and
    /// `(&a) as *f32` reinterprets the pointee bytes (interp reads an i32, native an f32). Restrict
    /// casts to the well-defined conversions; reject the byte-reinterprets. Lenient on Unknown/Error.
    fn check_cast(&mut self, from: &Ty, to: &Ty, span: Span) {
        if from.is_unknown() || from.is_error() || to.is_unknown() || to.is_error() {
            return;
        }
        if !self.cast_is_valid(from, to) {
            // A cast whose target is an enum forges a discriminant no variant need hold; give the
            // dedicated hint rather than the generic reinterpret message.
            let to_is_enum = matches!(
                to,
                Ty::Named(n) if matches!(self.defs.lookup(*n).map(|d| &d.kind), Some(DefKind::Enum(_)))
            );
            let msg = if to_is_enum {
                format!(
                    "invalid cast: `{}` cannot be cast to the enum `{}` — an integer cast can forge a \
                     discriminant that no variant holds, which then reaches an exhaustive `match`'s \
                     unreachable arm; `match` on the integer and return the intended variant instead",
                    from.display(self.interner),
                    to.display(self.interner)
                )
            } else {
                format!(
                    "invalid cast: `{}` cannot be cast to `{}`",
                    from.display(self.interner),
                    to.display(self.interner)
                )
            };
            self.error(span, "E0401", msg);
        }
    }

    /// The permitted `as` conversions: identity (`from == to`), scalar→scalar (every numeric/bool/char
    /// pairing — a real numeric conversion), C-style enum→scalar (its discriminant), integer→pointer
    /// (forming an address, including `0 as *T`), and pointer→pointer only when the pointee types
    /// match (a same-layout retype). Everything else — pointer→integer, *scalar→enum* (it would forge
    /// a discriminant no variant holds), aggregate↔scalar, data-carrying-enum↔scalar, differing-element
    /// pointer casts — is a byte reinterpret the two backends disagree on, so it is rejected.
    fn cast_is_valid(&self, from: &Ty, to: &Ty) -> bool {
        if from == to {
            return true;
        }
        // A scalar, or a *C-style* enum `Named` (its integer discriminant) — both integer-
        // representable. A **data-carrying** enum is a byte buffer, not a discriminant scalar, so it
        // is excluded (casting a tagged union to/from an integer is a byte reinterpret, rejected).
        let scalar_like = |t: &Ty| match t {
            Ty::Scalar(_) => true,
            Ty::Named(n) => {
                matches!(
                    self.defs.lookup(*n).map(|d| &d.kind),
                    Some(DefKind::Enum(_))
                ) && !self.enum_is_data_carrying(*n)
            }
            _ => false,
        };
        match (from, to) {
            // A scalar or enum discriminant -> a numeric/bool/char SCALAR: a real, total conversion.
            // Casting TO an enum is deliberately excluded (it falls through to the reject below): an
            // integer cast can forge a discriminant no variant holds, and feeding that value to an
            // exhaustive `match` reaches the `Unreachable` fall-through the enum-exhaustiveness
            // lowering emits — a genuine backend divergence (the interpreter traps with exit 1, native
            // executes an illegal instruction with exit 132). Rust rejects `int as Enum` for the same
            // reason. `E as E` (identity) is already accepted above; `enum -> int` stays valid here.
            (a, b) if scalar_like(a) && matches!(b, Ty::Scalar(_)) => true,
            // Integer -> pointer: forming a pointer from an address, including the null pointer
            // `0 as *T` (the only way to initialize a `*Node` leaf in a linked list / tree). The
            // cast itself never diverges — both backends yield a pointer value; only *dereferencing*
            // an invalid pointer is undefined, which is the programmer's responsibility under manual
            // memory (the same accepted UB as an out-of-bounds index). `ptr as int` stays rejected
            // below: it has no deref to blame yet silently diverges (the interpreter cannot
            // materialise a real address). A float/bool source is rejected as nonsensical.
            (Ty::Scalar(s), Ty::Ptr { .. } | Ty::Ref { .. }) if s.is_int() => true,
            // Pointer/reference retype: only when the pointee types are identical (mutability may
            // differ). A differing pointee reinterprets the referent's bytes — a backend divergence.
            (
                Ty::Ptr { pointee: a, .. } | Ty::Ref { pointee: a, .. },
                Ty::Ptr { pointee: b, .. } | Ty::Ref { pointee: b, .. },
            ) => a == b,
            _ => false,
        }
    }

    /// Check a struct literal's fields against the declaration: every declared field must be
    /// initialized (unless `..rest` supplies the remainder), none twice, and none unknown. A missing
    /// field used to compile and then read an uninitialized slot — the interpreter saw `0`, native
    /// saw stack garbage (a backend divergence); an unknown/duplicate field only erred late at
    /// codegen. `decl` is `(field name, field type)` pairs cloned out of the def map by the caller.
    fn check_struct_literal(
        &mut self,
        decl: &[(Symbol, Ty)],
        inits: &[FieldInit],
        has_rest: bool,
        span: Span,
    ) {
        let mut seen: Vec<Symbol> = Vec::new();
        for f in inits {
            let name = f.name.sym;
            match decl.iter().find(|(dn, _)| *dn == name) {
                Some((_, fty)) => {
                    // A literal field value adapts to (and is range-checked against) the declared
                    // field type, exactly like a `let` annotation: `S { v: 9000000000 }` lowers the
                    // literal at the field's i64 width instead of the default i32, and a
                    // `S { x: 9000000000 }` for an `x: i32` field is a hard E0401, not a silent
                    // low-32-bit truncation both backends agree on. Aggregate fields adapt
                    // element-wise (a `[i8; N]` / tuple field).
                    if self.literal_adapts(fty, &f.value) {
                        self.retype_adapted_literal(&f.value, fty);
                    }
                    self.range_check_int_literal(&f.value, fty);
                    // An array field's initializer must have the declared length. `Buf { data: [11,22] }`
                    // for `data: [i32; 4]` left the tail uninitialized (interp read 0, native read stack
                    // garbage — a backend divergence), and an over-long one silently dropped elements.
                    // `let`/tuple already length-check array initializers; struct fields did not.
                    if let Ty::Array { len, .. } = fty {
                        let init_len = match &f.value.kind {
                            ExprKind::ArrayLit(items) => Some(items.len() as u64),
                            ExprKind::ArrayRepeat { count, .. } => Some(self.eval_usize(count)),
                            _ => None,
                        };
                        if let Some(il) = init_len {
                            if il != *len {
                                self.error(
                                    f.value.span,
                                    "E0401",
                                    format!(
                                        "array field `{}` has length {} but its initializer has {} \
                                         element(s)",
                                        self.sym_str(name),
                                        len,
                                        il
                                    ),
                                );
                            }
                        }
                    }
                    // Scalar-type agreement, the same rule `let`/return/assignment enforce: a
                    // NON-literal field value of a different scalar type (`S { a: x }` with `a: i32`,
                    // `x: f32`) silently truncated/demoted it — both backends agreeing on the lossy
                    // value — where the language demands an explicit `as`. The value's type is already
                    // in the side table (the `StructLit` arm types each field before this runs).
                    let vty = self.types.get(&f.value.id).cloned().unwrap_or(Ty::Unknown);
                    if let (Ty::Scalar(fs), Ty::Scalar(vs)) = (fty, &vty) {
                        if fs != vs && !self.literal_adapts(fty, &f.value) {
                            self.error(
                                f.value.span,
                                "E0401",
                                format!(
                                    "type mismatch: field `{}` has type `{}`, but a value of type \
                                     `{}` is given (use `as {}` to convert)",
                                    self.sym_str(name),
                                    fty.display(self.interner),
                                    vty.display(self.interner),
                                    fs.name()
                                ),
                            );
                        }
                    }
                }
                None => {
                    let nm = self.sym_str(name).to_string();
                    self.error(f.name.span, "E0401", format!("struct has no field `{nm}`"));
                }
            }
            if seen.contains(&name) {
                let nm = self.sym_str(name).to_string();
                self.error(
                    f.name.span,
                    "E0401",
                    format!("field `{nm}` is initialized more than once"),
                );
            } else {
                seen.push(name);
            }
        }
        if !has_rest {
            let missing: Vec<String> = decl
                .iter()
                .filter(|(dn, _)| !seen.contains(dn))
                .map(|(dn, _)| format!("`{}`", self.sym_str(*dn)))
                .collect();
            if !missing.is_empty() {
                let s = if missing.len() == 1 { "" } else { "s" };
                self.error(
                    span,
                    "E0401",
                    format!(
                        "missing field{s} {} in this struct initializer",
                        missing.join(", ")
                    ),
                );
            }
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::default());
        self.immutable_locals.push(HashSet::default());
        self.immutable_params.push(HashSet::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.immutable_locals.pop();
        self.immutable_params.pop();
    }

    /// Whether `name` resolves (innermost scope first) to a local bound immutably. A `mut` binding,
    /// a deferred `let x;` (no initializer — its first assignment is the initialization), a
    /// parameter, and a global are all *not* immutable here, so none of them are rejected.
    fn is_immutable_local(&self, name: Symbol) -> bool {
        for i in (0..self.scopes.len()).rev() {
            if self.scopes[i].contains_key(&name) {
                return self
                    .immutable_locals
                    .get(i)
                    .is_some_and(|s| s.contains(&name));
            }
        }
        false
    }

    /// Whether `name` resolves (innermost scope first) to a parameter declared without `mut`. Uses
    /// the same innermost-first walk as `is_immutable_local`, so an inner `let` shadowing the
    /// parameter (which is mutable-through-projection per the `let` rules) is found first and this
    /// returns `false` for it — the parameter rule only applies where the name really is the param.
    fn is_immutable_param(&self, name: Symbol) -> bool {
        for i in (0..self.scopes.len()).rev() {
            if self.scopes[i].contains_key(&name) {
                return self
                    .immutable_params
                    .get(i)
                    .is_some_and(|s| s.contains(&name));
            }
        }
        false
    }

    /// The root single-name of an assignment place, walking field/index projections but STOPPING at
    /// a dereference: past `*p` you are in a pointee, not the binding's own storage, so `*p = …` and
    /// `(*p).f = …` return `None` (not a mutation of the binding). `None` also for any non-name root.
    fn assign_root_param(&self, e: &Expr) -> Option<Symbol> {
        match &e.kind {
            ExprKind::Field { base, .. }
            | ExprKind::TupleField { base, .. }
            | ExprKind::Index { base, .. } => self.assign_root_param(base),
            ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
            _ => None,
        }
    }

    /// Whether parameter `name`'s declared type is an aggregate passed BY REFERENCE (struct / tuple /
    /// array / tensor / vector — and a `[]T` slice, whose elements live in the caller's buffer) — the
    /// kinds whose through-projection mutation reaches the caller. Implemented as the complement of
    /// scalar / pointer / reference / `()` / `Unknown` / `Error`: a scalar can't be projected, and a
    /// pointer / reference projection dereferences to a pointee that is legitimately mutable.
    fn param_is_aggregate(&self, name: Symbol) -> bool {
        match self.param_tys.get(&name) {
            Some(ty) => !matches!(
                ty,
                Ty::Scalar(_)
                    | Ty::Ptr { .. }
                    | Ty::Ref { .. }
                    | Ty::Unit
                    | Ty::Unknown
                    | Ty::Error
            ),
            None => false,
        }
    }

    fn bind(&mut self, name: Symbol, ty: Ty) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup_local(&self, name: Symbol) -> Option<Ty> {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(&name) {
                return Some(t.clone());
            }
        }
        None
    }

    /// Resolve a single-segment name to a type, or `None` if genuinely unresolved.
    fn resolve_value(&self, name: Symbol) -> Option<Ty> {
        if let Some(t) = self.lookup_local(name) {
            return Some(t);
        }
        if self.generics.contains(&name) {
            // A generic dimension used as a value is a compile-time `usize`.
            return Some(Ty::Scalar(Scalar::Usize));
        }
        if let Some(def) = self.defs.lookup(name) {
            return Some(match &def.kind {
                DefKind::Fn(sig) => Ty::Fn {
                    params: sig.params.clone(),
                    ret: Box::new(sig.ret.clone()),
                },
                DefKind::Const(t) => t.clone(),
                DefKind::Struct(_) | DefKind::Enum(_) => Ty::Unknown, // used as a namespace
            });
        }
        let s = self.sym_str(name);
        if Scalar::from_name(s).is_some() || s == "Tensor" || is_vector_name(s) {
            // A type name used as a namespace/constructor (e.g. `f32x8::splat`).
            return Some(Ty::Unknown);
        }
        None
    }

    fn type_block(&mut self, b: &Block) -> Ty {
        self.push_scope();
        for s in &b.stmts {
            self.type_stmt(s);
        }
        let t = match &b.tail {
            Some(e) => self.type_expr(e),
            None => Ty::Unit,
        };
        self.pop_scope();
        t
    }

    fn type_stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Let {
                pat,
                mutable,
                ty,
                init,
            } => {
                let init_ty = init.as_ref().map(|e| self.type_expr(e));
                let ann_ty = ty.as_ref().map(|t| self.lower_type(t));
                let bound = match (&ann_ty, &init_ty) {
                    (Some(a), Some(i)) => {
                        let init_expr = init.as_ref().unwrap();
                        if self.let_compatible(a, init_expr, i) {
                            // Re-stamp / range-check ONLY a genuinely adapting literal expression. An
                            // untyped literal adopts the annotation top to bottom (so `-1.5` re-stamps
                            // the inner literal, and `0 - 16: i64` re-stamps both literal operands).
                            // But when `let_compatible` holds merely via `compatible` — the operands
                            // already type-check, e.g. `let c: i64 = a + b` with `a: i32` — the
                            // operands are NOT all literals, and re-stamping would overwrite the
                            // narrow variable `a`'s real type with the wide annotation, dropping the
                            // widening `sext` it needs → an operand/result width clash the -O0 verifier
                            // rejects and -O1+ `mem2reg` raw-panics on. `literal_adapts` is true only
                            // for all-literal expressions, exactly where the operand recursion is sound.
                            if self.literal_adapts(a, init_expr) {
                                self.retype_adapted_literal(init_expr, a);
                                self.range_check_int_literal(init_expr, a);
                            }
                        } else {
                            self.error(
                                s.span,
                                "E0401",
                                format!(
                                    "type mismatch: `let` is annotated `{}` but the value is `{}`",
                                    a.display(self.interner),
                                    i.display(self.interner)
                                ),
                            );
                        }
                        a.clone()
                    }
                    (Some(a), None) => a.clone(),
                    (None, Some(i)) => i.clone(),
                    (None, None) => Ty::Unknown,
                };
                self.bind_pattern(pat, &bound);
                if let Ty::Array {
                    shared: true,
                    elem,
                    len,
                    ..
                } = &bound
                {
                    const STATIC_SMEM_CAP: u64 = 48 * 1024;
                    let sizing = Ty::Array {
                        elem: elem.clone(),
                        len: *len,
                        shared: false,
                    };
                    if let Some(bytes) = sizing.size_of() {
                        if bytes > STATIC_SMEM_CAP {
                            self.error(
                                s.span,
                                "E0307",
                                format!(
                                    "shared array requires {bytes} bytes of static shared memory, \
                                     exceeding the {STATIC_SMEM_CAP} byte limit"
                                ),
                            );
                        }
                    }
                }
                // Track immutability for a simple binding: a `let` without `mut` that has an
                // initializer is fully initialized at the binding, so a later `x = …` is a
                // reassignment to reject. A `mut` binding (or a deferred `let x;` with no
                // initializer, whose first assignment is its initialization) is not tracked; a `mut`
                // rebinding also clears a prior immutable mark in this scope (shadowing). Tuple /
                // wildcard patterns stay lenient.
                if let PatKind::Ident(s) = &pat.kind {
                    if let Some(set) = self.immutable_locals.last_mut() {
                        if !*mutable && init.is_some() {
                            set.insert(*s);
                        } else {
                            set.remove(s);
                        }
                    }
                }
            }
            StmtKind::Assign { target, op, value } => {
                let target_ty = self.type_expr(target);
                let value_ty = self.type_expr(value);
                // A literal assigned to a place adapts to (and is range-checked against) the place's
                // type, the same rule `let` and struct-init already apply — so `p = 9000000000` for
                // an i32 place is a hard E0401 instead of a silent low-32-bit truncation both backends
                // agree on, and `p = 9000000000` for an i64 place lowers the literal at i64 width.
                // Plain `=` only (a value store); the compound forms below are a different check, and
                // the helpers no-op on a non-scalar/unknown place type.
                if matches!(op, AssignOp::Assign) {
                    if self.literal_adapts(&target_ty, value) {
                        self.retype_adapted_literal(value, &target_ty);
                    }
                    self.range_check_int_literal(value, &target_ty);
                    // Scalar-type agreement, the same rule `let`/call-args/return enforce: assigning
                    // a NON-literal value of a different scalar type to a scalar place — `y = x` with
                    // `y: i32`, `x: f32`, or a field/element place `s.a = x` / `arr[0] = x` — silently
                    // truncated/demoted it (both backends agreeing on the lossy value) where the
                    // language demands an explicit `as`. An unsuffixed literal still adapts and an
                    // out-of-range literal is caught above; only a non-literal mismatch errors.
                    if let (Ty::Scalar(ts), Ty::Scalar(vs)) = (&target_ty, &value_ty) {
                        if ts != vs && !self.literal_adapts(&target_ty, value) {
                            self.error(
                                target.span,
                                "E0401",
                                format!(
                                    "type mismatch: cannot assign a value of type `{}` to a place of \
                                     type `{}` (use `as {}` to convert)",
                                    value_ty.display(self.interner),
                                    target_ty.display(self.interner),
                                    ts.name()
                                ),
                            );
                        }
                    }
                }
                // A compound assignment `a += b` (and `-= *= /= …`) means `a = a (op) b`. That implied
                // binary operator is undefined on an aggregate (struct/tuple/array) — the explicit
                // `a = a + b` form is already rejected above — but the compound path skipped the
                // check, so mir_build emitted e.g. `add` on the aggregates' base pointers: invalid MIR
                // the verifier and Cranelift reject (a crash), with the two backends disagreeing on
                // the garbage at -O0. Reject it. A *plain* `=` aggregate copy stays valid (it deep-
                // copies), so this fires only for the arithmetic/bitwise compound forms.
                if !matches!(op, AssignOp::Assign) {
                    if let Some(bad) = [&target_ty, &value_ty].into_iter().find(|t| {
                        self.is_noncomputable_operand(t)
                            || matches!(t, Ty::Tensor { .. } | Ty::Ptr { .. } | Ty::Ref { .. })
                    }) {
                        self.error(
                            target.span,
                            "E0401",
                            format!(
                                "compound assignment `{}` is not defined for `{}` values; only \
                                 scalars and SIMD vectors support `{}`",
                                op.glyph(),
                                bad.display(self.interner),
                                op.glyph()
                            ),
                        );
                    }
                    // The implied `a = a (op) b` produces a value of the binary operator's result
                    // type. An integer place with a float operand yields a float that cannot
                    // implicitly narrow back to the integer place — exactly what the plain `a = a + b`
                    // scalar-agreement check above rejects. The compound path silently accepted it and
                    // emitted a float->int conversion (a *negative* float even converted via `fptoui`
                    // to 0, so `acc += -1.0` left `acc` unchanged) — a lossy value both backends
                    // agreed on, but one the language forbids everywhere else. Reject with E0401.
                    if let (Ty::Scalar(ts), Ty::Scalar(vs)) = (&target_ty, &value_ty) {
                        if ts.is_int() && vs.is_float() {
                            self.error(
                                target.span,
                                "E0401",
                                format!(
                                    "compound assignment `{}` cannot apply a `{}` value to an `{}` \
                                     place; the result would narrow to `{}` — convert explicitly \
                                     with `as {}`",
                                    op.glyph(),
                                    vs.name(),
                                    ts.name(),
                                    ts.name(),
                                    ts.name()
                                ),
                            );
                        }
                    }
                }
                // Reassigning an immutable binding requires `mut`, but it was never enforced. Two
                // cases, both E0304:
                //   • Direct rebind of a single name (`x = …`): rejected for a non-`mut` `let` AND a
                //     non-`mut` parameter (the message differs so the fix is obvious).
                //   • Mutation *through* a projection (`p.f = …`, `p[i] = …`): stays lenient for a
                //     `let` (it owns its storage), but a non-`mut` *aggregate parameter* is rejected —
                //     an aggregate is passed by reference, so `p.f = …` would silently mutate the
                //     CALLER's value. `mut` opts into that (visible, in-place). A dereference breaks
                //     the chain (`*p`, `(*p).f`) — that mutates a pointee, not the parameter — so
                //     `assign_root_param` returns `None` and pointer/ref params stay lenient.
                match &target.kind {
                    ExprKind::Path(p) if p.is_single() => {
                        let sym = p.first().sym;
                        if self.is_immutable_param(sym) {
                            let nm = self.sym_str(sym).to_string();
                            self.error(
                                target.span,
                                "E0304",
                                format!(
                                    "cannot assign to immutable parameter `{nm}`; add `mut` to the \
                                     parameter (`fn …(mut {nm}: …)`) to allow mutation"
                                ),
                            );
                        } else if self.is_immutable_local(sym) {
                            let nm = self.sym_str(sym).to_string();
                            self.error(
                                target.span,
                                "E0304",
                                format!(
                                    "cannot assign twice to immutable binding `{nm}`; add `mut` to \
                                     its `let` to allow reassignment"
                                ),
                            );
                        }
                    }
                    ExprKind::Field { .. }
                    | ExprKind::TupleField { .. }
                    | ExprKind::Index { .. } => {
                        if let Some(root) = self.assign_root_param(target) {
                            if self.is_immutable_param(root) && self.param_is_aggregate(root) {
                                let nm = self.sym_str(root).to_string();
                                self.error(
                                    target.span,
                                    "E0304",
                                    format!(
                                        "cannot mutate `{nm}` through a non-`mut` parameter; an \
                                         aggregate parameter is passed by reference, so this would \
                                         mutate the caller's value — add `mut` to the parameter \
                                         (`fn …(mut {nm}: …)`) to allow in-place mutation"
                                    ),
                                );
                            }
                        }
                    }
                    _ => {}
                }
                // Assigning a pointer/aggregate into a scalar place (or vice versa) reinterprets the
                // bits — `x = p` for `x: i32`, `p: *i32` stores a truncated address, which the
                // interpreter and native backend disagree on. Reject the kind clash; numeric
                // coercion across scalars stays lenient (handled at lowering, like `let`).
                // …and assigning an aggregate into a place of a DIFFERENT aggregate type does
                // the same reinterpret without changing width: `a = b` between two unrelated
                // structs printed `1` on the interpreter and `1065353216` on native.
                // Directional, and literal-aware, for the same reasons `check_return_shape` is:
                // the place is the target, so a `[]T` slice must NOT satisfy a `[T; N]` place
                // (`let` has always rejected that direction), while `a = [127, 0];` for
                // `a: [i8; 2]` is the adapting literal `let` accepts.
                if self.scalar_aggregate_clash(&target_ty, &value_ty)
                    || self.aggregate_value_clash(&target_ty, &value_ty, value)
                {
                    self.error(
                        value.span,
                        "E0401",
                        format!(
                            "type mismatch: cannot assign a value of type `{}` to a place of type \
                             `{}`",
                            value_ty.display(self.interner),
                            target_ty.display(self.interner)
                        ),
                    );
                }
                // A whole-tensor (or whole-vector) assignment must agree in shape. `x = b` for
                // `x: Tensor[f32, 4]` and `b: Tensor[f32, 8]` was silently accepted — assignment was
                // the one shape-bearing context with no shape unify (the let-init, binop, and return
                // paths all check) — storing a mis-shaped buffer past the place's length. Unify the
                // place and value shapes (E0501 rank / E0502 dim), the same check the binops apply;
                // a no-op for scalar/aggregate places.
                self.check_binop_shapes(&target_ty, &value_ty, value.span);
            }
            StmtKind::Expr(e) => {
                // A statement-position `loop` (`loop {…};` or `loop {…}` before another statement)
                // discards its value, so a `break <value>` inside it is rejected. Type it in
                // non-value mode; every other expression statement types normally.
                if let ExprKind::Loop { label, body } = &e.kind {
                    let ty = self.type_loop(label.as_ref().map(|l| l.sym), body, false);
                    self.types.insert(e.id, ty);
                } else {
                    self.type_expr(e);
                }
            }
            StmtKind::Return(opt) => {
                let ret = self.ret_ty.clone();
                match opt {
                    Some(e) => {
                        let t = self.type_expr(e);
                        // A value returned from a `-> ()` (or return-less) function: the native
                        // backend builds a `Void` return signature and rejects the value, while the
                        // interpreter silently discards it — a backend divergence. A returned `()`
                        // or void call (type `Unit`) is fine; stay lenient on `Unknown`/`Error`.
                        if matches!(ret, Ty::Unit)
                            && !matches!(t, Ty::Unit | Ty::Unknown | Ty::Error)
                        {
                            self.error(
                                e.span,
                                "E0401",
                                format!(
                                    "this function returns `()`, but a value of type `{}` is \
                                     returned here",
                                    t.display(self.interner)
                                ),
                            );
                        }
                        // The symmetric hole: returning a `()` value where a real type is declared —
                        // `return x;` where `x = if c { 42 }` (an else-less `if` is unit), or
                        // `return (if c { 42 });` directly. mir_build built a `return` of a void value:
                        // the -O0 verifier rejects it (a clean ICE) but -O2 mem2reg *panicked* on the
                        // void branch argument. Reject at the source (E0401), like the bare `return;`
                        // arm below — stay lenient on `Unknown`/`Error`.
                        if !matches!(ret, Ty::Unit | Ty::Unknown | Ty::Error)
                            && matches!(t, Ty::Unit)
                        {
                            self.error(
                                e.span,
                                "E0401",
                                format!(
                                    "this function must return a value of type `{}`, but a `()` \
                                     value is returned here (an `if` with no `else`, or a `match` \
                                     with statement arms, yields `()`)",
                                    ret.display(self.interner)
                                ),
                            );
                        }
                        // Shape / kind / scalar agreement, and the literal adaptation — re-stamp
                        // and range check included, so an out-of-range integer literal returned
                        // where a narrower type is declared (`return 9999999999` from `-> i32`,
                        // once silently truncated to the low bits: exit code 255 from a wrapped
                        // value) is still rejected. That range check used to live here, which left
                        // the implicit TAIL return — the other caller of `check_return_shape` —
                        // without one; it is inside the callee now so both spellings get it exactly
                        // once.
                        self.check_return_shape(&t, e, e.span);
                    }
                    None => {
                        // A bare `return;` where the signature demands a value: the native return
                        // expects an operand and rejects the empty return, while the interpreter
                        // returns a default — again a divergence. A `-> ()` / return-less function
                        // may `return;` freely.
                        if !matches!(ret, Ty::Unit | Ty::Unknown | Ty::Error) {
                            self.error(
                                s.span,
                                "E0401",
                                format!(
                                    "this function must return a value of type `{}`, but this \
                                     `return;` has none",
                                    ret.display(self.interner)
                                ),
                            );
                        }
                    }
                }
            }
            StmtKind::Defer(e) => {
                self.type_expr(e);
            }
            StmtKind::Break(lbl, val) => {
                // Type the value (for its side-table entry) regardless of whether the target loop
                // accepts it, so a malformed value still reports its own errors.
                let vty = val.as_ref().map(|v| self.type_expr(v));
                // Resolve the target loop frame (innermost, or the named label).
                let target = if self.loop_ctx.is_empty() {
                    self.error(s.span, "E0303", "`break` outside of a loop".to_string());
                    None
                } else if let Some(l) = lbl {
                    match self.loop_ctx.iter().rposition(|c| c.label == Some(l.sym)) {
                        Some(i) => Some(i),
                        None => {
                            self.error(
                                l.span,
                                "E0303",
                                format!("use of undeclared loop label `'{}`", self.sym_str(l.sym)),
                            );
                            None
                        }
                    }
                } else {
                    Some(self.loop_ctx.len() - 1)
                };
                if let (Some(idx), Some(v), Some(vt)) = (target, val.as_ref(), vty) {
                    if self.loop_ctx[idx].is_value {
                        // Merge this break value into the loop's running type. Tensor/vector shapes
                        // (and element types) must agree across all breaks — the same unification the
                        // `if`/`match` arms use (`check_binop_shapes`) — so a shape-mismatched break
                        // is an E0502, not a silently mistyped merge that mis-strides or reinterprets.
                        let prev = self.loop_ctx[idx].break_ty.take();
                        let joined = match prev {
                            None => vt,
                            Some(p) => {
                                self.check_merge_shapes(&p, &vt, Some(v), v.span);
                                join(p, vt)
                            }
                        };
                        self.loop_ctx[idx].break_ty = Some(joined);
                    } else {
                        self.error(
                            v.span,
                            "E0401",
                            "`break` with a value is only allowed in a `loop` used as an \
                             expression; this loop's value is discarded"
                                .to_string(),
                        );
                    }
                }
            }
            StmtKind::Continue(lbl) => {
                if self.loop_ctx.is_empty() {
                    self.error(s.span, "E0303", "`continue` outside of a loop".to_string());
                } else if let Some(l) = lbl {
                    // A labeled `continue` must name an enclosing loop's label.
                    if !self.loop_ctx.iter().any(|c| c.label == Some(l.sym)) {
                        self.error(
                            l.span,
                            "E0303",
                            format!("use of undeclared loop label `'{}`", self.sym_str(l.sym)),
                        );
                    }
                }
            }
            StmtKind::While {
                cond, body, label, ..
            } => {
                self.check_condition(cond);
                self.loop_ctx.push(LoopCtx {
                    label: label.as_ref().map(|l| l.sym),
                    is_value: false,
                    break_ty: None,
                });
                self.type_block(body);
                self.loop_ctx.pop();
            }
            StmtKind::For {
                pat,
                iter,
                body,
                label,
                ..
            } => {
                let elem = self.type_for_iter(iter);
                self.push_scope();
                self.bind_pattern(pat, &elem);
                self.loop_ctx.push(LoopCtx {
                    label: label.as_ref().map(|l| l.sym),
                    is_value: false,
                    break_ty: None,
                });
                self.type_block(body);
                self.loop_ctx.pop();
                self.pop_scope();
            }
        }
    }

    fn type_for_iter(&mut self, iter: &ForIter) -> Ty {
        match iter {
            ForIter::Range {
                start, end, step, ..
            } => {
                let t = self.type_expr(start);
                if let Some(e) = end {
                    self.type_expr(e);
                }
                if let Some(st) = step {
                    self.type_expr(st);
                }
                if t.is_unknown() {
                    Ty::Scalar(Scalar::Usize)
                } else {
                    t
                }
            }
            ForIter::Expr(e) => {
                // `for x in arr` binds the loop variable to the array's *element* type, so the
                // body type-checks (`for x in [T; N]` ⇒ `x: T`). A slice iterates the same way
                // (`for x in s` ⇒ `x: T` for `s: []T`). Any other iterand stays `Unknown` (lenient
                // — we don't newly reject other iterables here).
                match self.type_expr(e) {
                    Ty::Array { elem, .. } | Ty::Slice(elem) => *elem,
                    _ => Ty::Unknown,
                }
            }
        }
    }

    fn bind_pattern(&mut self, pat: &Pattern, ty: &Ty) {
        match &pat.kind {
            PatKind::Ident(s) => self.bind(*s, ty.clone()),
            PatKind::Tuple(subs) => {
                if let Ty::Tuple(elems) = ty {
                    for (p, t) in subs.iter().zip(elems) {
                        self.bind_pattern(p, t);
                    }
                } else {
                    for p in subs {
                        self.bind_pattern(p, &Ty::Unknown);
                    }
                }
            }
            // An or-pattern's alternatives are typically binding-free literals/variants; bind each
            // against the same scrutinee type so any shared identifier resolves.
            PatKind::Or(alts) => {
                for a in alts {
                    self.bind_pattern(a, ty);
                }
            }
            PatKind::Wildcard | PatKind::Unit => {}
            // A data-carrying enum-variant pattern: validate the variant + payload shape and bind the
            // inner fields to their declared types (so an arm body sees `x`/`y` typed).
            PatKind::Variant { path, fields } => self.check_variant_pattern(path, fields, pat.span),
            // Literal / enum-variant / range patterns bind nothing — they test the scrutinee's value.
            // An integer literal pattern is materialized by mir_build as `ConstInt(v, <scrutinee
            // width>)` with no range check, so an out-of-range literal was TRUNCATED to that width and
            // matched a different value, shadowing the arm that legitimately covers it: on an `i32`
            // scrutinee `match x { 4294967296 => 1, 0 => 2, _ => 0 }` returned 1 for `x == 0`, and on
            // an `i8` scrutinee `200 => 1` matched `-56`. The same literal in `let`/`const`/`return`/
            // argument position is already rejected — apply that rule here too.
            PatKind::Int { .. } => self.range_check_int_pattern(pat, ty),
            PatKind::Range { lo, hi, .. } => {
                self.range_check_int_pattern(lo, ty);
                self.range_check_int_pattern(hi, ty);
            }
            PatKind::Char(_) | PatKind::Bool(_) | PatKind::Path(_) => {}
        }
    }

    /// The pattern counterpart of [`Sema::range_check_int_literal`]: an integer literal *pattern* that
    /// does not fit the scrutinee's narrow type. Same rule, same code, same wording — only types whose
    /// whole range fits in `i64` are checked (i8..u32), so an `i64`/`u64`/`usize` scrutinee is never
    /// flagged.
    fn range_check_int_pattern(&mut self, pat: &Pattern, ty: &Ty) {
        let Ty::Scalar(sc) = ty else { return };
        let Some((lo, hi)) = int_lit_range(*sc) else {
            return;
        };
        let Some(v) = pat_int_value(pat, self.interner) else {
            return;
        };
        if v < lo || v > hi {
            self.error(
                pat.span,
                "E0401",
                format!(
                    "literal `{v}` is out of range for `{}` ({lo}..={hi})",
                    sc.name()
                ),
            );
        }
    }

    /// Type-check and bind a data-carrying enum-variant pattern `Enum::Variant(..)` / `Enum::Variant
    /// { .. }`. Resolves the variant, rejects an unknown variant (E0301), a payload-kind mismatch or
    /// a wrong tuple arity (E0401), and binds each payload sub-pattern to its declared field type.
    /// On any error it still binds the sub-patterns (to `Unknown`) so the arm body type-checks
    /// without cascading.
    fn check_variant_pattern(&mut self, path: &Path, fields: &VariantPat, span: Span) {
        let (Some(ename), Some(vname)) = (
            path.segments.first().map(|s| s.sym),
            path.segments.last().map(|s| s.sym),
        ) else {
            self.bind_variant_fields_unknown(fields);
            return;
        };
        // Clone the payload out of the def map so the immutable borrow ends before we take `&mut self`.
        let payload = match self.defs.lookup(ename) {
            Some(Def {
                kind: DefKind::Enum(variants),
                ..
            }) => match variants.iter().find(|v| v.name == vname) {
                Some(v) => v.payload.clone(),
                None => {
                    let en = self.sym_str(ename).to_string();
                    let vn = self.sym_str(vname).to_string();
                    self.error(span, "E0301", format!("no variant `{vn}` in enum `{en}`"));
                    self.bind_variant_fields_unknown(fields);
                    return;
                }
            },
            // `path[0]` is not a declared enum: stay lenient (an unmodeled name), bind leniently.
            _ => {
                self.bind_variant_fields_unknown(fields);
                return;
            }
        };
        let qual = format!("{}::{}", self.sym_str(ename), self.sym_str(vname));
        match (fields, &payload) {
            (VariantPat::Tuple(subs), VariantPayload::Tuple(tys)) => {
                if subs.len() != tys.len() {
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "enum variant `{qual}` has {} field(s), but the pattern binds {}",
                            tys.len(),
                            subs.len()
                        ),
                    );
                }
                for (i, sub) in subs.iter().enumerate() {
                    let t = tys.get(i).cloned().unwrap_or(Ty::Unknown);
                    self.bind_pattern(sub, &t);
                }
            }
            (VariantPat::Struct(fps), VariantPayload::Struct(named)) => {
                for fp in fps {
                    match named
                        .iter()
                        .find(|(n, _)| *n == fp.name)
                        .map(|(_, t)| t.clone())
                    {
                        Some(t) => self.bind_pattern(&fp.pat, &t),
                        None => {
                            let fname = self.sym_str(fp.name).to_string();
                            self.error(
                                fp.pat.span,
                                "E0301",
                                format!("no field `{fname}` in enum variant `{qual}`"),
                            );
                            self.bind_pattern(&fp.pat, &Ty::Unknown);
                        }
                    }
                }
            }
            (_, VariantPayload::Unit) => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "enum variant `{qual}` has no payload to destructure; match it as `{qual}`"
                    ),
                );
                self.bind_variant_fields_unknown(fields);
            }
            (VariantPat::Tuple(_), VariantPayload::Struct(_)) => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "enum variant `{qual}` is a struct variant; destructure it with `{{ .. }}`"
                    ),
                );
                self.bind_variant_fields_unknown(fields);
            }
            (VariantPat::Struct(_), VariantPayload::Tuple(_)) => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "enum variant `{qual}` is a tuple variant; destructure it with `( .. )`"
                    ),
                );
                self.bind_variant_fields_unknown(fields);
            }
        }
    }

    /// Bind every sub-pattern of a variant payload pattern to `Unknown` — the error-recovery path so
    /// an ill-formed variant pattern doesn't leave its bindings untyped (a cascade of E0301s).
    fn bind_variant_fields_unknown(&mut self, fields: &VariantPat) {
        match fields {
            VariantPat::Tuple(subs) => {
                for s in subs {
                    self.bind_pattern(s, &Ty::Unknown);
                }
            }
            VariantPat::Struct(fps) => {
                for fp in fps {
                    self.bind_pattern(&fp.pat, &Ty::Unknown);
                }
            }
        }
    }

    /// If `callee(args)` is `Enum::Variant(args)` constructing a **tuple-payload** variant, type-check
    /// it (arity + per-field types) and return the enum's nominal type. Returns `None` when the callee
    /// is not an enum-variant path (an ordinary function/method call — falls through to `type_call`).
    /// A unit / struct variant reached this way is a mis-spelled constructor and is rejected (E0401).
    fn type_variant_construction(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        span: Span,
    ) -> Option<Ty> {
        let ExprKind::Field { base, name } = &callee.kind else {
            return None;
        };
        let ExprKind::Path(p) = &base.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let ename = p.first().sym;
        let vname = name.sym;
        // Resolve under a scoped immutable borrow: `None` = not an enum (defer to `type_call`);
        // `Some(None)` = an enum but an unknown variant; `Some(Some(p))` = the variant's payload.
        let resolved: Option<Option<VariantPayload>> = match self.defs.lookup(ename) {
            Some(Def {
                kind: DefKind::Enum(vs),
                ..
            }) => Some(
                vs.iter()
                    .find(|v| v.name == vname)
                    .map(|v| v.payload.clone()),
            ),
            _ => None,
        };
        let payload_opt = resolved?; // not an enum → ordinary call
        let qual = format!("{}::{}", self.sym_str(ename), self.sym_str(vname));
        let Some(payload) = payload_opt else {
            for a in args {
                self.type_expr(a);
            }
            self.error(span, "E0301", format!("no variant `{qual}`"));
            return Some(Ty::Error);
        };
        match payload {
            VariantPayload::Tuple(tys) => {
                if args.len() != tys.len() {
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "enum variant `{qual}` takes {} field(s), but {} were supplied",
                            tys.len(),
                            args.len()
                        ),
                    );
                }
                for (i, a) in args.iter().enumerate() {
                    let fty = tys.get(i).cloned().unwrap_or(Ty::Unknown);
                    self.check_payload_field(a, &fty);
                }
            }
            VariantPayload::Unit => {
                for a in args {
                    self.type_expr(a);
                }
                self.error(
                    span,
                    "E0401",
                    format!("enum variant `{qual}` has no payload; write it as `{qual}`"),
                );
            }
            VariantPayload::Struct(_) => {
                for a in args {
                    self.type_expr(a);
                }
                self.error(
                    span,
                    "E0401",
                    format!(
                        "enum variant `{qual}` is a struct variant; construct it with `{qual} {{ .. }}`"
                    ),
                );
            }
        }
        Some(Ty::Named(ename))
    }

    /// Type-check one payload/field initializer `arg` against its declared type `fty`: an unsuffixed
    /// numeric literal adapts (and is range-checked), a non-literal scalar of a different type is an
    /// E0401 (the same rule `let`/assign/struct-init apply). Non-scalar fields stay lenient.
    fn check_payload_field(&mut self, arg: &Expr, fty: &Ty) {
        let at = self.type_expr(arg);
        if self.literal_adapts(fty, arg) {
            self.retype_adapted_literal(arg, fty);
        }
        self.range_check_int_literal(arg, fty);
        if let (Ty::Scalar(fs), Ty::Scalar(vs)) = (fty, &at) {
            if fs != vs && !self.literal_adapts(fty, arg) {
                self.error(
                    arg.span,
                    "E0401",
                    format!(
                        "type mismatch: field expects `{}` but the value is `{}` (use `as {}` to convert)",
                        fty.display(self.interner),
                        at.display(self.interner),
                        fs.name()
                    ),
                );
            }
        }
    }

    /// Whether `init` can initialize a `let` annotated `ann`. Besides ordinary compatibility,
    /// an *unsuffixed* numeric literal adapts to any integer/float annotation (Rust's `{integer}`
    /// inference, in miniature) — as does an all-literal arithmetic expression or an array/tuple
    /// literal of adapting elements; see [`Sema::literal_adapts`] for the exact set.
    fn let_compatible(&self, ann: &Ty, init: &Expr, init_ty: &Ty) -> bool {
        if compatible(ann, init_ty) {
            return true;
        }
        self.literal_adapts(ann, init)
    }

    /// Whether `e` is an *unsuffixed* integer literal — its signedness is not pinned by a suffix, so
    /// it adapts to the other operand of a comparison (`u32_var < 5` stays legal: `5` becomes
    /// unsigned). A *suffixed* literal (`5u32`, `3i32`) has a FIXED signedness and is NOT flexible,
    /// so it must still participate in the mixed-signedness ordered-compare check — `i32 < 5u32` is a
    /// genuine sign mismatch (`a < 5u32` and `5u32 > a` disagree), not adaptation.
    fn is_sign_flexible_int_literal(&self, e: &Expr) -> bool {
        matches!(&e.kind, ExprKind::Int(s) if !has_int_suffix(self.sym_str(*s)))
    }

    /// Whether `t` is one of the current function's generic type parameters (`T` in `fn f<T>`) —
    /// a `Ty::Named` whose symbol is in scope as a generic. At check time such a type is abstract
    /// (not a scalar); it becomes concrete only at monomorphization.
    fn is_generic_ty(&self, t: &Ty) -> bool {
        matches!(t, Ty::Named(s) if self.generics.contains(s))
    }

    /// Whether `e` is an *unsuffixed* numeric literal (int or float), optionally under a unary minus —
    /// the type-flexible kind that adapts to the other operand of an arithmetic binop. Used so a
    /// literal adapts to a generic-param operand (`2 * x` where `x: T`) instead of forcing the binop to
    /// the literal's i32 default and truncating the generic value at monomorphization.
    fn is_adaptable_num_literal(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(s) => !has_int_suffix(self.sym_str(*s)),
            ExprKind::Float(s) => !has_float_suffix(self.sym_str(*s)),
            ExprKind::Unary {
                op: UnOp::Neg,
                expr,
            } => self.is_adaptable_num_literal(expr),
            _ => false,
        }
    }

    /// Whether an *unsuffixed* numeric literal — optionally wrapped in a unary minus, e.g.
    /// `let x: f64 = -1.5;` — adapts to the integer/float annotation `ann`. A leading `-`
    /// does not change a literal's kind, so we peel `Neg` and re-check the inner literal.
    fn literal_adapts(&self, ann: &Ty, init: &Expr) -> bool {
        match (&init.kind, ann) {
            (ExprKind::Int(s), Ty::Scalar(sc)) => sc.is_int() && !has_int_suffix(self.sym_str(*s)),
            (ExprKind::Float(s), Ty::Scalar(sc)) => {
                sc.is_float() && !has_float_suffix(self.sym_str(*s))
            }
            (
                ExprKind::Unary {
                    op: UnOp::Neg,
                    expr,
                },
                Ty::Scalar(_),
            ) => self.literal_adapts(ann, expr),
            // A binary arithmetic / bitwise expression of two adapting operands adapts to a scalar
            // annotation: `let v: i64 = 0 - 16` (only the unary form `-16` adapted before, so the
            // binary form was a spurious i32-vs-i64 mismatch). The result type of these ops is the
            // common operand type, so if both sides adapt to `ann`, the whole expression does. A
            // non-literal operand (a variable / call) makes the inner `literal_adapts` false, so
            // only all-constant expressions adapt. Shifts (result = LHS type) and comparisons
            // (result = bool) are excluded. A narrow target still range-checks the FOLDED value
            // (`let v: i8 = 100 + 100` -> 200, out of range) via `range_check_int_literal`.
            (
                ExprKind::Binary {
                    op:
                        BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Rem
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor,
                    lhs,
                    rhs,
                },
                Ty::Scalar(_),
            ) => self.literal_adapts(ann, lhs) && self.literal_adapts(ann, rhs),
            // An array / tuple literal adapts element-wise to a matching aggregate annotation, so a
            // typed buffer can be built from literals — `let a: [i8; 2] = [127, 0]` and
            // `let t: (u8, u8) = (200, 1)` previously failed as `[i32; 2]`/`(i32, i32)` mismatches.
            // Lengths must match and every element must itself adapt (recursively, so nested
            // aggregates work too).
            (ExprKind::ArrayLit(items), Ty::Array { elem, len, .. }) => {
                items.len() as u64 == *len && items.iter().all(|it| self.literal_adapts(elem, it))
            }
            // …and so does a repeat initializer — but its COUNT must match the annotation's length,
            // exactly like the `ArrayLit` arm above and `check_struct_literal`'s array-field check.
            // Without it `let a: [i32; 4] = [7; 2];` compiled clean and mir_build filled the slot to
            // the annotation's 4, silently discarding the count the programmer wrote (the identical
            // mismatch is a hard error as a struct field and as an array literal).
            (ExprKind::ArrayRepeat { value, count }, Ty::Array { elem, len, .. }) => {
                self.eval_usize(count) == *len && self.literal_adapts(elem, value)
            }
            (ExprKind::TupleLit(items), Ty::Tuple(tys)) => {
                items.len() == tys.len()
                    && items
                        .iter()
                        .zip(tys)
                        .all(|(it, t)| self.literal_adapts(t, it))
            }
            _ => false,
        }
    }

    /// Re-stamp an adapted numeric literal — and the literal inside any unary minus — with the
    /// `let`'s annotated type, so MIR lowering sees one consistent width (e.g. `-1.5` becomes
    /// `f64` end to end, not `-(1.5: f32)` widened to `f64` at the `Neg`).
    fn retype_adapted_literal(&mut self, e: &Expr, ann: &Ty) {
        self.types.insert(e.id, ann.clone());
        match (&e.kind, ann) {
            (
                ExprKind::Unary {
                    op: UnOp::Neg,
                    expr,
                },
                _,
            ) => self.retype_adapted_literal(expr, ann),
            // Re-stamp both operands of an adapted binary expression with the annotation, so MIR
            // lowering sees one consistent width (`0 - 16: i64` is `(0: i64) - (16: i64)`, not two
            // i32s widened at the `Sub` — which would be ill-typed MIR). Mirrors `literal_adapts`'s
            // op set; reached only when that returned true (both operands are adapting constants).
            (
                ExprKind::Binary {
                    op:
                        BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Rem
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor,
                    lhs,
                    rhs,
                },
                Ty::Scalar(_),
            ) => {
                self.retype_adapted_literal(lhs, ann);
                self.retype_adapted_literal(rhs, ann);
            }
            // Re-stamp each element of an adapted aggregate literal with the annotation's element
            // type so MIR lowering stores it at the right width (`[127, 0]: [i8; 2]` writes two i8s,
            // not i32s narrowed at the store). The aggregate node itself takes `ann` (above).
            (ExprKind::ArrayLit(items), Ty::Array { elem, .. }) => {
                for it in items {
                    self.retype_adapted_literal(it, elem);
                }
            }
            (ExprKind::ArrayRepeat { value, .. }, Ty::Array { elem, .. }) => {
                self.retype_adapted_literal(value, elem);
            }
            (ExprKind::TupleLit(items), Ty::Tuple(tys)) => {
                for (it, t) in items.iter().zip(tys) {
                    self.retype_adapted_literal(it, t);
                }
            }
            // A merge arm is normally spelled as a BLOCK whose tail is the literal, and both nodes
            // carry the type — the block's recorded type is its tail's. Peel to the tail and
            // re-stamp it too, mirroring `merge_arm_value`, which is how `literal_adapts` was asked
            // the question in the first place. Unreachable from every other caller: they all guard
            // with `literal_adapts`, which has no `Block` arm and so answers `false` for one.
            (ExprKind::Block(b), _) => {
                if let Some(tail) = &b.tail {
                    self.retype_adapted_literal(tail, ann);
                }
            }
            _ => {}
        }
    }

    /// Diagnose an unsuffixed integer literal that does not fit the narrow type it adapts to
    /// (`let x: i8 = 200;`). Only types whose whole range fits in `i64` are checked (i8..u32), and
    /// only a literal `eval_const_int` can read (incl. a leading `-`) — so a runtime value, a float,
    /// or an `i64`/`u64`/`usize` literal is never flagged. Rust rejects this; without it the value
    /// silently wraps (`200 as i8 == -56`), a quiet footgun in a safety-first language. Call at each
    /// site a literal adapts to a type — `let`, `const`, a struct-literal field, an argument, an
    /// assignment, a `return` (both spellings, inside `check_return_shape`) and a merge arm
    /// (`aggregate_merge_clash`) — right where `retype_adapted_literal` re-stamps it, since the
    /// re-stamp is what makes an out-of-range literal a constant too wide for its own type.
    fn range_check_int_literal(&mut self, e: &Expr, ty: &Ty) {
        // Aggregate literals recurse element-wise (`[300, 0]: [i8; 2]` flags the `300`), mirroring
        // how `retype_adapted_literal`/`literal_adapts` descend into them.
        match (&e.kind, ty) {
            (ExprKind::ArrayLit(items), Ty::Array { elem, .. }) => {
                for it in items {
                    self.range_check_int_literal(it, elem);
                }
                return;
            }
            (ExprKind::ArrayRepeat { value, .. }, Ty::Array { elem, .. }) => {
                self.range_check_int_literal(value, elem);
                return;
            }
            (ExprKind::TupleLit(items), Ty::Tuple(tys)) => {
                for (it, t) in items.iter().zip(tys) {
                    self.range_check_int_literal(it, t);
                }
                return;
            }
            _ => {}
        }
        let Ty::Scalar(sc) = ty else { return };
        let Some((lo, hi)) = int_lit_range(*sc) else {
            return;
        };
        if let Some(v) = eval_const_int(e, self.interner) {
            if v < lo || v > hi {
                self.error(
                    e.span,
                    "E0401",
                    format!(
                        "literal `{v}` is out of range for `{}` ({lo}..={hi})",
                        sc.name()
                    ),
                );
            }
        }
    }

    /// Validate a char/string literal's escapes, keeping the established split (sema validates
    /// literals, mir_build decodes them) alongside `int_literal_well_formed` /
    /// `float_literal_well_formed`. Two rules:
    ///
    ///  * the escape must be one the decoder KNOWS. `decode_escape` ends `Some(c) => c as u32`, so
    ///    every unknown escape silently became the escaped character itself and `println("A\dB")`
    ///    printed `AdB` — a value the source never wrote, agreed on by both backends, so no
    ///    differential gate can see it. C and Rust both reject an unknown escape.
    ///  * a `\u` must be followed IMMEDIATELY by `{`. The decoder's `\u` arm is
    ///    `chars.by_ref().skip_while(|&c| c != '{')` over the shared iterator, so a `\u` with no
    ///    brace consumes the entire remainder of the literal and folds to 0: `println("A\uZZF")`
    ///    printed `A` and silently deleted the tail. `bad_unicode_escape` below has the identical
    ///    `skip_while`, which is why it saw code point 0 and reported nothing.
    fn check_unicode_escapes(&mut self, sym: Symbol, span: Span) {
        if let Some(msg) = malformed_escape(self.sym_str(sym)) {
            self.error(span, "E0401", msg);
        }
        if let Some(cp) = bad_unicode_escape(self.sym_str(sym)) {
            self.error(
                span,
                "E0401",
                format!(
                    "`\\u{{{cp:X}}}` is not a Unicode scalar value (the maximum is \\u{{10FFFF}}, \
                     and \\u{{D800}}..=\\u{{DFFF}} are surrogates)"
                ),
            );
        }
    }

    fn type_expr(&mut self, e: &Expr) -> Ty {
        let ty = self.type_expr_inner(e);
        self.check_layout(&ty, e.span);
        self.types.insert(e.id, ty.clone());
        ty
    }

    fn type_expr_inner(&mut self, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Int(s) => {
                // A malformed integer literal — a mistyped radix like `0z123`, an empty radix `0x`,
                // a bad digit `0b2`, or a value beyond u64 — lexes as one `Int` token, fails to
                // parse, and used to lower *silently to 0* (both backends agreed on the wrong value,
                // so the differential gate was blind). Reject it instead of miscompiling.
                let text = self.sym_str(*s).to_string();
                if int_literal_well_formed(&text) {
                    Ty::Scalar(int_lit_scalar(&text))
                } else {
                    self.error(e.span, "E0401", format!("invalid integer literal `{text}`"));
                    Ty::Error
                }
            }
            ExprKind::Float(s) => {
                // Likewise a malformed float literal (`1.5z`, an incomplete exponent `1.5e`) that
                // the greedy suffix scan swept into one `Float` token; it parsed to 0.0 silently.
                let text = self.sym_str(*s).to_string();
                if float_literal_well_formed(&text) {
                    Ty::Scalar(float_lit_scalar(&text))
                } else {
                    self.error(e.span, "E0401", format!("invalid float literal `{text}`"));
                    Ty::Error
                }
            }
            ExprKind::Bool(_) => Ty::Scalar(Scalar::Bool),
            ExprKind::Str(s) => {
                self.check_unicode_escapes(*s, e.span);
                Ty::Ptr {
                    mutable: false,
                    pointee: Box::new(Ty::Scalar(Scalar::U8)),
                }
            }
            ExprKind::Char(s) => {
                self.check_unicode_escapes(*s, e.span);
                Ty::Scalar(Scalar::Char)
            }
            ExprKind::Path(p) => {
                if p.is_single() {
                    match self.resolve_value(p.first().sym) {
                        Some(t) => t,
                        None => {
                            self.error(
                                p.span,
                                "E0301",
                                format!(
                                    "cannot find `{}` in this scope",
                                    self.sym_str(p.first().sym)
                                ),
                            );
                            Ty::Error
                        }
                    }
                } else {
                    Ty::Unknown
                }
            }
            ExprKind::Unary { op, expr } => {
                let t = self.type_expr(expr);
                match op {
                    // `*x` requires a pointer/reference operand. Dereferencing a definite
                    // non-pointer (a scalar, struct, tuple, or array) used to fall through to
                    // `Ty::Unknown`, then mir_build lowered `*scalar` to a load off a non-pointer
                    // operand — invalid MIR the verifier/Cranelift reject (an ICE on a program sema
                    // had accepted). Stay lenient for `Unknown`/`Error` and pointer-like types.
                    UnOp::Deref => match t {
                        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } => *pointee,
                        Ty::Scalar(_) | Ty::Named(_) | Ty::Tuple(_) | Ty::Array { .. } => {
                            self.error(
                                expr.span,
                                "E0401",
                                format!(
                                    "cannot dereference a value of type `{}`; only a pointer or \
                                     reference can be dereferenced with `*`",
                                    t.display(self.interner)
                                ),
                            );
                            Ty::Unknown
                        }
                        _ => Ty::Unknown,
                    },
                    UnOp::Ref => Ty::Ptr {
                        mutable: false,
                        pointee: Box::new(t),
                    },
                    UnOp::RefMut => Ty::Ptr {
                        mutable: true,
                        pointee: Box::new(t),
                    },
                    // Unary `-`/`!` on an aggregate (struct/tuple/array), `()` (a void call's
                    // result), or a function value is meaningless and used to pass through unchanged,
                    // then mir_build emitted an arithmetic op on a base pointer / a dummy / a void
                    // operand — invalid MIR (`-nothing()` crashed native). Scalars/vectors/tensors
                    // negate fine; stay lenient for `Unknown`.
                    UnOp::Neg | UnOp::Not => {
                        if self.is_noncomputable_operand(&t) {
                            let glyph = if matches!(op, UnOp::Neg) { "-" } else { "!" };
                            self.error(
                                expr.span,
                                "E0401",
                                format!(
                                    "cannot apply unary `{glyph}` to a value of type `{}`",
                                    t.display(self.interner)
                                ),
                            );
                            Ty::Unknown
                        } else if matches!(op, UnOp::Not)
                            && matches!(&t, Ty::Scalar(s) if s.is_float())
                        {
                            // `~`/`!` is a bitwise complement — undefined on a float. mir_build emitted
                            // a bitwise-not on float SSA (the interpreter truncates to int, Cranelift
                            // takes the raw IEEE bits then fptosi — a gate-blind interp!=native). E0401.
                            self.error(
                                expr.span,
                                "E0401",
                                "bitwise complement `!`/`~` requires an integer operand, not a float"
                                    .to_string(),
                            );
                            Ty::Unknown
                        } else if matches!(op, UnOp::Neg)
                            && matches!(&expr.kind, ExprKind::Int(s)
                                if !has_int_suffix(self.sym_str(*s))
                                    && parse_u64_text(self.sym_str(*s)) == Some(1u64 << 63))
                        {
                            // `-9223372036854775808` is i64::MIN. The magnitude 2^63 alone overflows
                            // i64, so `int_lit_scalar` defaults the bare literal to u64 (right for the
                            // positive form `print(9223372036854775808)`) — but under a unary minus it
                            // is i64::MIN. Re-type the inner literal (and the negation) as i64 so
                            // mir_build bakes the i64::MIN constant and `print` treats it as signed;
                            // otherwise the u64 negate wrapped back to +2^63 and printed positive — a
                            // gate-blind wrong sign both backends agreed on. Mirrors how the annotated
                            // `let x: i64 = -9223372036854775808` already lowers correctly.
                            self.types.insert(expr.id, Ty::Scalar(Scalar::I64));
                            Ty::Scalar(Scalar::I64)
                        } else {
                            t
                        }
                    }
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.type_expr(lhs);
                let r = self.type_expr(rhs);
                // Operand shapes must agree (tensor-tensor / vector-vector); a mismatch is a real
                // error regardless of whether the operator yields a value or a bool.
                self.check_binop_shapes(&l, &r, e.span);
                use BinOp::*;
                // No binary operator other than `==`/`!=` (which has its own message below) is
                // defined on an aggregate (struct/tuple/array), `()` (a void call's result), or a
                // function value: arithmetic/bitwise/shift/ordering/logical on one lowers to an op on
                // a base pointer / dummy / void operand — invalid MIR, an ICE on accepted input
                // (`nothing() * 2`). Tensors and vectors are *not* aggregates here, so tensor/vector
                // arithmetic is unaffected; `Unknown` stays lenient.
                if !matches!(op, Eq | Ne) {
                    if let Some(bad) = [&l, &r]
                        .into_iter()
                        .find(|t| self.is_noncomputable_operand(t))
                    {
                        self.error(
                            e.span,
                            "E0401",
                            format!(
                                "binary operator `{}` is not defined for `{}` values; operate on \
                                 their fields or elements instead",
                                op.glyph(),
                                bad.display(self.interner)
                            ),
                        );
                        return Ty::Unknown;
                    }
                }
                // Pointer arithmetic (`p + 1`) and whole-tensor arithmetic (`a + b` on `Tensor`
                // values) are not supported: both lowered to an `add` on a base pointer that the MIR
                // verifier and Cranelift reject (an ICE / a backend-specific error, the two backends
                // disagreeing). Reject them cleanly here, only for the arithmetic/bitwise/shift
                // operators. `==`/`!=` and ordering on pointers are untouched; elementwise tensor work
                // uses indexed scalars (`a[i] + b[i]`), and SIMD `Vector` arithmetic still lowers.
                if matches!(
                    op,
                    Add | Sub | Mul | Div | Rem | BitAnd | BitOr | BitXor | Shl | Shr
                ) {
                    if let Some(bad) = [&l, &r]
                        .into_iter()
                        .find(|t| matches!(t, Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Tensor { .. }))
                    {
                        let msg = if matches!(bad, Ty::Tensor { .. }) {
                            "whole-tensor arithmetic is not supported; operate on elements in a loop \
                             (e.g. `c[i] = a[i] + b[i]`)"
                        } else {
                            "pointer arithmetic is not supported; take an element address with \
                             `&arr[i]`"
                        };
                        self.error(e.span, "E0401", msg.to_string());
                        return Ty::Unknown;
                    }
                }
                // bool is not a number: reject it as an operand to a true arithmetic or shift
                // operator (it would silently coerce to 0/1 — `true + 1` evaluated to `2`). Bitwise
                // `& | ^` (non-short-circuit boolean ops), `&&`/`||`, `==`/`!=`, ordered comparison
                // (rejected above), and unary `!` all still take bool. Concrete-bool-only, so an
                // `Unknown`/`Error` operand stays lenient (no false positives on unmodeled values).
                if matches!(op, Add | Sub | Mul | Div | Rem | Shl | Shr)
                    && [&l, &r]
                        .into_iter()
                        .any(|t| matches!(t, Ty::Scalar(Scalar::Bool)))
                {
                    self.error(
                        e.span,
                        "E0401",
                        format!(
                            "arithmetic operator `{}` is not defined for `bool`",
                            op.glyph()
                        ),
                    );
                    return Ty::Unknown;
                }
                // Bitwise (`& | ^`) and shift (`<< >>`) operators require INTEGER operands. A float
                // operand reached mir_build, which emitted an integer bitwise/shift MIR op on float SSA
                // values: the interpreter truncated-to-int-then-bitwise while Cranelift took the raw
                // IEEE bits then fptosi (a gate-blind interp!=native), and a float SHIFT crashed the
                // verifier at -O0 / panicked mem2reg at -O2. Reject with E0401 (Rust rejects `f32 & f32`
                // likewise). Concrete-float-only, so Unknown/Error stay lenient.
                if matches!(op, BitAnd | BitOr | BitXor | Shl | Shr)
                    && [&l, &r]
                        .into_iter()
                        .any(|t| matches!(t, Ty::Scalar(s) if s.is_float()))
                {
                    self.error(
                        e.span,
                        "E0401",
                        format!(
                            "bitwise/shift operator `{}` requires integer operands, not float",
                            op.glyph()
                        ),
                    );
                    return Ty::Unknown;
                }
                match op {
                    // `==`/`!=` on an aggregate (struct/tuple/array) silently lowers to a
                    // base-pointer compare — two distinct values are *always* "not equal" — a wrong
                    // answer with no diagnostic. Reject it (compare the fields/elements instead).
                    // An enum (a C-style integer discriminant) and a pointer compare fine, so only
                    // struct/tuple/array operands are rejected.
                    Eq | Ne => {
                        if self.is_aggregate_ty(&l) || self.is_aggregate_ty(&r) {
                            self.error(
                                e.span,
                                "E0401",
                                "`==`/`!=` is not defined for aggregate (struct/tuple/array) \
                                 values; compare their fields or elements instead"
                                    .to_string(),
                            );
                        } else if let Some(other) = match (&l, &r) {
                            // A `bool` compared for equality against a NON-bool scalar is almost
                            // always a chained comparison: `a == b == c` parses as `(a == b) == c`,
                            // and `(a == b)` is a bool, so `== c` silently compared a bool against
                            // `c` (coerced to 0/1) — `5 == 3 == 0` evaluated to `true`. Reject the
                            // mixed compare (ordered chains are rejected below). Concrete-scalar-only,
                            // so `Unknown` stays lenient and `bool == bool` / same-kind compares are
                            // unaffected.
                            (Ty::Scalar(Scalar::Bool), Ty::Scalar(o)) if *o != Scalar::Bool => {
                                Some(*o)
                            }
                            (Ty::Scalar(o), Ty::Scalar(Scalar::Bool)) if *o != Scalar::Bool => {
                                Some(*o)
                            }
                            _ => None,
                        } {
                            self.error(
                                e.span,
                                "E0401",
                                format!(
                                    "cannot compare `bool` with `{}`; a chained comparison like \
                                     `a == b == c` parses as `(a == b) == c` — write \
                                     `a == b && b == c`",
                                    other.name()
                                ),
                            );
                        }
                        Ty::Scalar(Scalar::Bool)
                    }
                    // Ordered comparison is not defined on `bool`. The usual trigger is a chained
                    // comparison: `a < b < c` parses left-associatively as `(a < b) < c`, and
                    // `(a < b)` is a `bool`, so the outer `< c` silently compared a bool against an
                    // int (the bool coerced to 0/1) — a wrong answer with no diagnostic. Reject a
                    // concrete bool operand; `Unknown`/`Error` stay lenient (no false positives on
                    // unmodeled operands). `And`/`Or` legitimately take bool, so they keep returning
                    // bool unchecked.
                    Lt | Le | Gt | Ge => {
                        if matches!(l, Ty::Scalar(Scalar::Bool))
                            || matches!(r, Ty::Scalar(Scalar::Bool))
                        {
                            self.error(
                                e.span,
                                "E0401",
                                format!(
                                    "ordered comparison `{}` is not defined for `bool`; a chained \
                                     comparison like `a < b < c` parses as `(a < b) < c` — write \
                                     `a < b && b < c` instead",
                                    op.glyph()
                                ),
                            );
                        } else if let (Ty::Scalar(ls), Ty::Scalar(rs)) = (&l, &r) {
                            // A mixed signed/unsigned integer comparison lowers to a `Cmp` whose
                            // signedness is taken from ONE operand, so `a < b` (i32 vs u32) and
                            // `b > a` give CONTRADICTORY answers (`slt` says -1 < 1, `ugt` says
                            // 1 < 4294967295) — a gate-blind logic bug both backends agree on. Reject
                            // a concrete opposite-signedness integer pair. Only an *unsuffixed* int
                            // LITERAL is sign-flexible (it adapts to the other operand), so skip when
                            // either side is one — `b < 5` / `5 < a` stay fine. A *suffixed* literal
                            // (`5u32`, `200u8`, `3i32`) pins its signedness and must still be checked:
                            // `i32 < 5u32` is a genuine mismatch that made `a < 5u32` (=1) and
                            // `5u32 > a` (=0) disagree — the guard was bypassed whenever the literal
                            // adopted the peer type. Rust rejects mixed signed/unsigned compares too.
                            if ls.is_int()
                                && rs.is_int()
                                && ls.is_signed() != rs.is_signed()
                                && !self.is_sign_flexible_int_literal(lhs)
                                && !self.is_sign_flexible_int_literal(rhs)
                            {
                                let (s, u) = if ls.is_signed() {
                                    (ls.name(), rs.name())
                                } else {
                                    (rs.name(), ls.name())
                                };
                                self.error(
                                    e.span,
                                    "E0401",
                                    format!(
                                        "comparison mixes signed `{s}` and unsigned `{u}`: the result \
                                         depends on which operand's signedness the compare uses, so \
                                         `a {} b` and `b {} a` can disagree — cast one side so both \
                                         are the same signedness",
                                        op.glyph(),
                                        match op {
                                            Lt => ">",
                                            Le => ">=",
                                            Gt => "<",
                                            _ => "<=",
                                        }
                                    ),
                                );
                            }
                        }
                        Ty::Scalar(Scalar::Bool)
                    }
                    And | Or => Ty::Scalar(Scalar::Bool),
                    _ => {
                        // For arithmetic, adapt an unsuffixed numeric literal to a generic-param
                        // operand on EITHER side. In a generic body `T` isn't a scalar, so `join`
                        // (which returns its LEFT arg when the pair isn't two scalars) mis-typed
                        // `2 * x` (i32-literal · T) as i32 — and mir_build then truncated the
                        // monomorphized f32/i64 operand (`fptoui f32 -> i32`), a gate-blind wrong
                        // answer. `x * 2` already worked (`join(T, i32)` returns the left T); this
                        // makes it symmetric, so `2 * x` types as `T` and the literal re-adapts at
                        // monomorphization.
                        let arith = matches!(op, Add | Sub | Mul | Div | Rem);
                        if arith && self.is_adaptable_num_literal(lhs) && self.is_generic_ty(&r) {
                            r
                        } else if arith
                            && self.is_adaptable_num_literal(rhs)
                            && self.is_generic_ty(&l)
                        {
                            l
                        } else {
                            join(l, r)
                        }
                    }
                }
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                // `Enum::Variant(args)` — construct a tuple-payload variant. Checked before
                // `type_call` (which would type the `Enum::Variant` callee as a bare value and
                // reject it as an incomplete constructor).
                if let Some(t) = self.type_variant_construction(callee, args, e.span) {
                    t
                } else {
                    self.type_call(callee, generic_args, args, e.span)
                }
            }
            ExprKind::Index { base, indices } => self.type_index(base, indices, e.span),
            ExprKind::Field { base, name } => {
                // `E::B` parses as a field access on the enum-name path `E`. If `E` is a declared
                // enum and `B` is one of its variants, the whole expression has the enum's nominal
                // type (its runtime value is the variant's integer discriminant, filled in by
                // `mir_build`). Checked before the struct path so the enum name isn't typed as a
                // value.
                if let ExprKind::Path(p) = &base.kind {
                    if p.is_single() {
                        if let Some(Def {
                            kind: DefKind::Enum(variants),
                            ..
                        }) = self.defs.lookup(p.first().sym)
                        {
                            if let Some(v) = variants.iter().find(|v| v.name == name.sym) {
                                // A payload-carrying variant used *bare* (not `E::V(..)` / `E::V { .. }`,
                                // which are a `Call`/`StructLit` intercepted before this) is an
                                // incomplete constructor — reject it with a hint rather than lowering a
                                // value with no payload. A unit variant is a complete value.
                                if !v.payload.is_unit() {
                                    let ename = self.sym_str(p.first().sym).to_string();
                                    let vname = self.sym_str(name.sym).to_string();
                                    self.error(
                                        e.span,
                                        "E0401",
                                        format!(
                                            "enum variant `{ename}::{vname}` carries a payload; \
                                             construct it with its fields (`{ename}::{vname}(..)` or \
                                             `{ename}::{vname} {{ .. }}`)"
                                        ),
                                    );
                                }
                                return Ty::Named(p.first().sym);
                            }
                        }
                    }
                }
                let t = self.type_expr(base);
                // A field access on a struct value — or on a pointer/reference to a struct, which
                // auto-derefs (`p.x` on a `&Pt` / `*mut Pt`) — resolves to the declared field type.
                // Anything else (a method, an unmodeled builtin) stays lenient (`Unknown`).
                let struct_name = match &t {
                    Ty::Named(n) => Some(*n),
                    Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } => match pointee.as_ref() {
                        Ty::Named(n) => Some(*n),
                        _ => None,
                    },
                    _ => None,
                };
                match struct_name {
                    Some(struct_name) => self
                        .defs
                        .lookup(struct_name)
                        .and_then(|d| match &d.kind {
                            DefKind::Struct(fields) => fields
                                .iter()
                                .find(|(fname, _)| *fname == name.sym)
                                .map(|(_, fty)| fty.clone()),
                            _ => None,
                        })
                        .unwrap_or(Ty::Unknown),
                    None => Ty::Unknown,
                }
            }
            ExprKind::TupleField { base, index } => {
                let t = self.type_expr(base);
                match t {
                    // A known tuple type range-checks the field index: `t.9` on a 3-tuple is a hard
                    // E0501, mirroring array/tensor out-of-bounds. Without it the access silently
                    // typed as `Unknown` and fell through to mir_build's generic "unsupported
                    // construct" fallback, reporting a misleading `C0001: tuple field access is not
                    // yet supported by codegen` when tuple access IS supported — it was just OOB.
                    Ty::Tuple(elems) => match elems.get(*index as usize) {
                        Some(ty) => ty.clone(),
                        None => {
                            self.error(
                                e.span,
                                "E0501",
                                format!(
                                    "tuple index {index} is out of range for a {}-element tuple",
                                    elems.len()
                                ),
                            );
                            Ty::Error
                        }
                    },
                    _ => Ty::Unknown,
                }
            }
            ExprKind::Cast { expr, ty } => {
                let from = self.type_expr(expr);
                let to = self.lower_type(ty);
                self.check_cast(&from, &to, e.span);
                to
            }
            ExprKind::StructLit { path, fields, rest } => {
                // Type each field value (populates their NodeId side-table entries for lowering).
                for f in fields {
                    self.type_expr(&f.value);
                }
                if let Some(r) = rest {
                    self.type_expr(r);
                }
                // `Enum::Variant { .. }` — a struct-payload variant literal (a multi-segment path
                // whose head is a declared enum). Checked before the plain struct path.
                if path.segments.len() >= 2 {
                    let ename = path.segments[0].sym;
                    let vname = path.segments.last().unwrap().sym;
                    let vp: Option<Option<VariantPayload>> = match self.defs.lookup(ename) {
                        Some(Def {
                            kind: DefKind::Enum(vs),
                            ..
                        }) => Some(
                            vs.iter()
                                .find(|v| v.name == vname)
                                .map(|v| v.payload.clone()),
                        ),
                        _ => None,
                    };
                    if let Some(payload_opt) = vp {
                        let qual = format!("{}::{}", self.sym_str(ename), self.sym_str(vname));
                        match payload_opt {
                            // A struct-payload variant's named fields are checked exactly like a
                            // struct literal's (same `(name, ty)` shape) — arity, unknown/duplicate
                            // fields, per-field type adaptation, and completeness.
                            Some(VariantPayload::Struct(named)) => {
                                self.check_struct_literal(&named, fields, rest.is_some(), e.span);
                            }
                            Some(_) => self.error(
                                e.span,
                                "E0401",
                                format!(
                                    "enum variant `{qual}` is not a struct variant; construct it as \
                                     `{qual}(..)` or `{qual}`"
                                ),
                            ),
                            None => self.error(e.span, "E0301", format!("no variant `{qual}`")),
                        }
                        return Ty::Named(ename);
                    }
                }
                // A `Name { … }` whose `Name` resolves to a declared struct has that nominal type
                // (and its fields are checked for completeness); an unknown name stays lenient.
                match path.segments.last().map(|s| s.sym) {
                    Some(n) => {
                        // Clone the declared field list out of the def map so the immutable borrow
                        // ends before `check_struct_literal` takes `&mut self` to emit diagnostics.
                        let decl = match self.defs.lookup(n) {
                            Some(Def {
                                kind: DefKind::Struct(decl),
                                ..
                            }) => Some(decl.clone()),
                            _ => None,
                        };
                        match decl {
                            Some(decl) => {
                                self.check_struct_literal(&decl, fields, rest.is_some(), e.span);
                                Ty::Named(n)
                            }
                            None => Ty::Unknown,
                        }
                    }
                    None => Ty::Unknown,
                }
            }
            ExprKind::ArrayLit(items) => {
                let mut elem = Ty::Unknown;
                for it in items {
                    let t = self.type_expr(it);
                    if elem.is_unknown() {
                        elem = t;
                    }
                }
                Ty::Array {
                    elem: Box::new(elem),
                    len: items.len() as u64,
                    shared: false,
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                let elem = self.type_expr(value);
                self.type_expr(count);
                let Some(len) = self.array_len(count) else {
                    return Ty::Error;
                };
                Ty::Array {
                    elem: Box::new(elem),
                    len,
                    shared: false,
                }
            }
            ExprKind::TupleLit(items) => {
                if items.is_empty() {
                    Ty::Unit
                } else {
                    Ty::Tuple(items.iter().map(|i| self.type_expr(i)).collect())
                }
            }
            ExprKind::Block(b) => self.type_block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.check_condition(cond);
                let then_ty = self.type_block(then_branch);
                if let Some(else_expr) = else_branch {
                    let else_ty = self.type_expr(else_expr);
                    // The two arms merge into ONE value, so — like a binary operator's operands, a
                    // call's arguments, and a `return` — their tensor/vector SHAPES (and element
                    // types) must agree. `join` alone picks the then-arm's type and lets a mismatched
                    // else-arm through: `if c { t: Tensor[f32,4] } else { u: Tensor[i32,4] }` typed as
                    // f32[4] makes native reinterpret u's i32 bits while interp reads the i32 (a
                    // divergence); a [2,2]-vs-[2,4] mismatch silently mis-strides the result; a
                    // [1024]-vs-[2] lie turns a statically-valid index into a runtime OOB segfault.
                    // Unify them — the 4th shape-bearing context, the one `check_binop_shapes` missed.
                    self.check_merge_shapes(&then_ty, &else_ty, Some(else_expr), else_expr.span);
                    join(then_ty, else_ty)
                } else {
                    Ty::Unit
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                let scrut_ty = self.type_expr(scrutinee);
                let mut result = Ty::Unknown;
                for arm in arms {
                    // Each arm gets its own scope: an `Ident` pattern binds the scrutinee value for
                    // the arm's guard and body; a literal/`_` pattern binds nothing.
                    self.push_scope();
                    self.bind_pattern(&arm.pat, &scrut_ty);
                    if let Some(g) = &arm.guard {
                        self.type_expr(g);
                    }
                    let t = self.type_expr(&arm.body);
                    self.pop_scope();
                    // Arms merge into one value: their tensor/vector shapes (and element types) must
                    // agree, exactly as the `if` arms above (see `check_binop_shapes`). Unify each arm
                    // against the running result before `join` folds it in, so a shape-mismatched arm
                    // is an E0502 — not a silently mistyped result that mis-strides or reinterprets.
                    self.check_merge_shapes(&result, &t, Some(&arm.body), arm.body.span);
                    result = join(result, t);
                }
                // Exhaustiveness: a `match` with no catch-all that provably misses a case is rejected
                // (E0405), like Rust. A *value* match would synthesize a typed-zero default in mir_build
                // (a silent wrong answer, or invalid MIR for an aggregate); a *statement* / empty match
                // still lowers a value-merge whose "no arm matched" fallthrough is `Unreachable`, which
                // the interpreter traps (exit 1) but the native backend hits as an illegal instruction
                // (a backend divergence) — so the check must NOT be gated on the result type being a
                // value. Skip only an `Error` result (its arm already reported, avoid cascading); an
                // unmodeled scrutinee stays lenient inside `match_is_provably_nonexhaustive`.
                if !matches!(result, Ty::Error)
                    && self.match_is_provably_nonexhaustive(&scrut_ty, arms)
                {
                    self.error(
                        e.span,
                        "E0405",
                        "non-exhaustive `match`: no arm covers all possible values; add a `_` arm \
                         (or cover every enum variant / both `bool` cases)"
                            .to_string(),
                    );
                }
                result
            }
            // A value-position `loop` (a `let` init, call arg, block tail, `return`, …): its type is
            // the join of every `break <value>`, inferred while checking the body.
            ExprKind::Loop { label, body } => {
                self.type_loop(label.as_ref().map(|l| l.sym), body, true)
            }
            ExprKind::SizeOf(_) | ExprKind::AlignOf(_) => Ty::Scalar(Scalar::Usize),
        }
    }

    /// Type-check a `loop`, inferring its type from its `break` values. `is_value` says whether the
    /// loop's value is consumed (a value-position loop, where `break v` is legal). A break-less loop
    /// is infinite and types unit; a statement loop always types unit.
    fn type_loop(&mut self, label: Option<Symbol>, body: &Block, is_value: bool) -> Ty {
        self.loop_ctx.push(LoopCtx {
            label,
            is_value,
            break_ty: None,
        });
        self.type_block(body);
        let ctx = self.loop_ctx.pop().expect("loop_ctx push/pop balanced");
        if is_value {
            ctx.break_ty.unwrap_or(Ty::Unit)
        } else {
            Ty::Unit
        }
    }

    /// Whether `arms` provably fail to cover every value of `scrut_ty` (used by E0405). Conservative:
    /// returns `true` only when incompleteness is *certain*, so a valid match is never rejected. A
    /// guard-less `_`/identifier arm covers everything (checked first). Otherwise: an `enum` is
    /// covered iff every variant appears in a guard-less variant arm; a `bool` iff both `true` and
    /// `false` appear; any other scalar (`int`/`char`/`usize` — an effectively unbounded domain)
    /// needs a catch-all. Unmodeled scrutinees (tuple/array/tensor/struct/pointer/unknown/…) stay
    /// lenient (`false`), matching sema's overall leniency. Guarded arms never prove coverage.
    fn match_is_provably_nonexhaustive(&self, scrut_ty: &Ty, arms: &[MatchArm]) -> bool {
        if arms.iter().any(arm_is_catch_all) {
            return false;
        }
        if arms.is_empty() {
            // An empty `match x {}` covers nothing, so it is non-exhaustive for any modeled, inhabited
            // scrutinee (Wukong has no uninhabited types). The `Ty::Scalar(_)` case below already
            // catches an empty int/char match; this also catches an empty enum / `bool` match (whose
            // coverage arms below would read "no cases seen" as lenient). Unmodeled scrutinees stay
            // lenient, matching the rest of this function.
            return match scrut_ty {
                Ty::Named(n) => {
                    matches!(
                        self.defs.lookup(*n).map(|d| &d.kind),
                        Some(DefKind::Enum(_))
                    )
                }
                Ty::Scalar(_) => true,
                _ => false,
            };
        }
        match scrut_ty {
            Ty::Named(n) => match self.defs.lookup(*n).map(|d| &d.kind) {
                Some(DefKind::Enum(variants)) => {
                    // Coverage per variant: a variant pattern covers its name; an int-literal /
                    // range pattern covers the variants whose DISCRIMINANT it matches (an enum
                    // value IS its discriminant at runtime, and `int as Enum` is rejected, so the
                    // declared discriminants are the whole domain). Previously a match by raw
                    // discriminant stayed lenient — `match e { 0 => .., 1 => .. }` on a
                    // three-variant enum compiled, and the missed variant silently took
                    // mir_build's zero default on BOTH backends (a gate-blind wrong answer). It
                    // also falsely rejected a mixed exhaustive match (`E::A | 1 | 2`), whose int
                    // arms contributed nothing. Any guard-less pattern the collector can't reason
                    // about keeps the whole match lenient, so incompleteness stays *certain*.
                    let mut covered = HashSet::default();
                    let mut spans: Vec<(i64, i64)> = Vec::new();
                    for a in arms {
                        if a.guard.is_none()
                            && !collect_enum_coverage(
                                &a.pat,
                                &mut covered,
                                &mut spans,
                                self.interner,
                            )
                        {
                            return false;
                        }
                    }
                    // All arms guarded (guards never prove coverage): lenient.
                    if covered.is_empty() && spans.is_empty() {
                        return false;
                    }
                    variants.iter().any(|v| {
                        !covered.contains(&v.name)
                            && !spans.iter().any(|(l, h)| *l <= v.disc && v.disc <= *h)
                    })
                }
                _ => false, // a non-enum `Named` (struct / generic / forward ref): lenient
            },
            Ty::Scalar(Scalar::Bool) => {
                let (mut t, mut f) = (false, false);
                for a in arms {
                    if a.guard.is_none() {
                        collect_bool_cases(&a.pat, &mut t, &mut f);
                    }
                }
                (t || f) && !(t && f) // uses bool patterns but misses one case
            }
            Ty::Scalar(_) => true, // int/char/usize/…: unbounded, and no catch-all reached here
            _ => false,
        }
    }
}

/// The symbol naming a single generic parameter (a type/dim var `N` or a const generic `const N`).
fn generic_param_sym(g: &GenericParam) -> Symbol {
    match &g.kind {
        GenericParamKind::Type(id) => id.sym,
        GenericParamKind::Const { name, .. } => name.sym,
    }
}

/// Classic Levenshtein edit distance (insert/delete/substitute), for the unknown-tensor-dimension
/// did-you-mean hint. Dimension names are short, so the simple two-row `O(len_a·len_b)` form is fine.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// The generic parameter names as an unordered set, for membership tests (`is this name a generic
/// of the current item?`). Order is irrelevant here — use `generic_param_sym` over the AST slice
/// directly where declaration order matters (e.g. binding a turbofish `f::<2, 3>` by position).
fn generic_names(gs: &[GenericParam]) -> HashSet<Symbol> {
    gs.iter().map(generic_param_sym).collect()
}

/// Pick a "more concrete" type when joining two (used for arithmetic results and if/match arms).
/// Leniently prefers a known type over Unknown/Error so we never invent false mismatches.
/// The result type of a binary arithmetic op on `a` and `b`. Unknown/Error defer to the other
/// side; two numeric scalars promote to the wider type (and float beats int), so a mixed-precision
/// expression like `(i as f64) + 1.0` is `f64` rather than picking an operand arbitrarily. This
/// keeps the lowered MIR well-typed once operands are coerced to the result.
fn join(a: Ty, b: Ty) -> Ty {
    if a.is_unknown() || a.is_error() {
        return b;
    }
    if b.is_unknown() || b.is_error() {
        return a;
    }
    if let (Ty::Scalar(sa), Ty::Scalar(sb)) = (&a, &b) {
        return Ty::Scalar(join_scalar(*sa, *sb));
    }
    a
}

fn scalar_bits(s: Scalar) -> u32 {
    use Scalar::*;
    match s {
        Bool => 1,
        I8 | U8 => 8,
        I16 | U16 | F16 | Bf16 => 16,
        I32 | U32 | F32 | Char => 32,
        I64 | U64 | Usize | Isize | F64 => 64,
    }
}

fn join_scalar(a: Scalar, b: Scalar) -> Scalar {
    if a == b {
        return a;
    }
    match (a.is_float(), b.is_float()) {
        (true, false) => a,
        (false, true) => b,
        _ => {
            if scalar_bits(a) >= scalar_bits(b) {
                a
            } else {
                b
            }
        }
    }
}

/// The expression a merge arm actually *produces*. An `if` or `match` arm is normally spelled as
/// a block, and `check_merge_shapes` receives that block — whose `ExprKind` is `Block`, never
/// `ArrayLit` — so an adapting array/tuple literal written as an arm would look like a plain type
/// mismatch and be rejected where the identical literal at a `let` is accepted.
///
/// Peel to the block's tail expression, and ONLY that: a block with no tail (its value comes from a
/// `break` or a `return`) is returned unchanged, and no other wrapper is unwrapped. It answers a
/// `literal_adapts` question and then names the node `range_check_int_literal` must look at; the
/// re-stamp that follows an adapting answer is handed the ARM instead, because the block and its
/// tail both carry the type and both have to move (see `aggregate_merge_clash`). This function
/// itself is pure — it changes no recorded type.
fn merge_arm_value(e: &Expr) -> &Expr {
    match &e.kind {
        ExprKind::Block(b) => match &b.tail {
            Some(t) => merge_arm_value(t),
            None => e,
        },
        _ => e,
    }
}

/// Loose compatibility used only where we are confident enough to diagnose (e.g. `let`
/// annotation vs initializer). Unknown/Error are compatible with anything.
fn compatible(a: &Ty, b: &Ty) -> bool {
    if same_stored_type(a, b) {
        return true;
    }
    // Unsizing: a fixed-size array `[T; N]` coerces to a slice `[]T` of the same element type — the
    // one implicit array↔slice conversion (it drops the static length to a runtime one). The slice
    // must be the *target* (`a`), e.g. `let s: []T = arr` or passing `[T; N]` where `[]T` is
    // expected; there is no slice→array direction (a slice has no static length to restore).
    match (a, b) {
        (Ty::Slice(ea), Ty::Slice(eb)) | (Ty::Slice(ea), Ty::Array { elem: eb, .. }) => {
            compatible(ea, eb)
        }
        _ => false,
    }
}

/// An empty literal has no element from which to infer a type. Match stored aggregate types
/// recursively so it can acquire that type inside tuples/arrays or a first merge arm. Every
/// extent/arity must agree. Unlike `compatible`, this does not grant implicit array-to-slice
/// conversion to a nested field: that would require new storage for the slice header.
fn same_stored_type(a: &Ty, b: &Ty) -> bool {
    if a == b || a.is_unknown() || b.is_unknown() || a.is_error() || b.is_error() {
        return true;
    }
    match (a, b) {
        (
            Ty::Array {
                elem: ea,
                len: la,
                shared: sa,
            },
            Ty::Array {
                elem: eb,
                len: lb,
                shared: sb,
            },
        ) if la == lb && sa == sb => same_stored_type(ea, eb),
        (Ty::Tuple(a), Ty::Tuple(b)) if a.len() == b.len() => {
            a.iter().zip(b).all(|(a, b)| same_stored_type(a, b))
        }
        _ => false,
    }
}

fn fixed_aggregate_shape_clash(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (
            Ty::Array {
                elem: ea, len: la, ..
            },
            Ty::Array {
                elem: eb, len: lb, ..
            },
        ) => la != lb || fixed_aggregate_shape_clash(ea, eb),
        (Ty::Tuple(a), Ty::Tuple(b)) => {
            a.len() != b.len()
                || a.iter()
                    .zip(b)
                    .any(|(a, b)| fixed_aggregate_shape_clash(a, b))
        }
        _ => false,
    }
}

fn is_vector_name(s: &str) -> bool {
    if let Some(idx) = s.rfind('x') {
        let (left, right) = (&s[..idx], &s[idx + 1..]);
        return !right.is_empty()
            && right.bytes().all(|b| b.is_ascii_digit())
            && Scalar::from_name(left).is_some();
    }
    false
}

/// Whether `b` is guaranteed to diverge on every path — return from the function or loop forever —
/// so control never falls off its end. The basis for the definite-return check. **Conservative
/// toward `true`**: it only reports `false` when a fall-through path is *certain*, so a function that
/// does return on every path is never flagged. A trailing tail expression is the block's value, so
/// it counts as a return.
fn block_diverges(b: &Block) -> bool {
    // A diverging statement makes the rest of the block unreachable, so the block diverges.
    if b.stmts.iter().any(stmt_diverges) {
        return true;
    }
    // The tail contributes divergence only if it is itself a diverging control-flow expression
    // (`if a { return } else { return }` as the last expression). A plain *value* tail means the
    // block completes normally and yields a value — which the function-level check treats as a
    // return via the block's type, not here.
    b.tail.as_deref().map(expr_diverges).unwrap_or(false)
}

fn stmt_diverges(s: &Stmt) -> bool {
    match &s.kind {
        StmtKind::Return(_) => true,
        StmtKind::Expr(e) => expr_diverges(e),
        _ => false,
    }
}

fn expr_diverges(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Block(b) => block_diverges(b),
        // A `loop` with no `break` anywhere in its body never exits normally (it loops forever or
        // returns from inside) — it diverges, and its value never materializes. Any `break` means it
        // may fall through to its merge (be lenient). Covers both statement and value loops.
        ExprKind::Loop { body, .. } => !block_contains_break(body),
        // Both arms must diverge; an `if` with no `else` can fall through (it takes the final
        // `_ => false` arm below).
        ExprKind::If {
            then_branch,
            else_branch: Some(els),
            ..
        } => block_diverges(then_branch) && expr_diverges(els),
        // A `match` diverges only if it is exhaustive (some unconditional catch-all arm) and every
        // arm body diverges; a non-exhaustive match falls through to a default.
        ExprKind::Match { arms, .. } => {
            !arms.is_empty()
                && arms.iter().any(arm_is_catch_all)
                && arms.iter().all(|a| expr_diverges(&a.body))
        }
        _ => false,
    }
}

fn arm_is_catch_all(a: &MatchArm) -> bool {
    a.guard.is_none() && matches!(a.pat.kind, PatKind::Wildcard | PatKind::Ident(_))
}

/// Enum-coverage contribution of one (guard-less) pattern, for `match_is_provably_nonexhaustive`:
/// a variant pattern covers its NAME (the path's last segment, `Color::Red` → `Red`), an int-literal
/// / range pattern covers a numeric SPAN of discriminants (an enum matches by its discriminant), a
/// wildcard/ident alternative covers everything, and or-patterns recurse.
/// Returns `false` — "cannot reason, stay lenient" — for any other pattern kind (char/bool/tuple:
/// ill-typed for an enum scrutinee or unmodeled here), so E0405 only fires on *certain* misses.
fn collect_enum_coverage(
    p: &Pattern,
    names: &mut HashSet<Symbol>,
    spans: &mut Vec<(i64, i64)>,
    interner: &Interner,
) -> bool {
    match &p.kind {
        PatKind::Path(path) | PatKind::Variant { path, .. } => {
            if let Some(seg) = path.segments.last() {
                names.insert(seg.sym);
            }
            true
        }
        PatKind::Int { .. } => match pat_int_value(p, interner) {
            Some(v) => {
                spans.push((v, v));
                true
            }
            None => false,
        },
        PatKind::Range { lo, hi, inclusive } => {
            let (Some(l), Some(h)) = (pat_int_value(lo, interner), pat_int_value(hi, interner))
            else {
                return false;
            };
            spans.push((l, if *inclusive { h } else { h.saturating_sub(1) }));
            true
        }
        PatKind::Or(alts) => alts
            .iter()
            .all(|a| collect_enum_coverage(a, names, spans, interner)),
        PatKind::Wildcard | PatKind::Ident(_) => {
            spans.push((i64::MIN, i64::MAX));
            true
        }
        _ => false,
    }
}

/// The integer value of an int-literal pattern (with its folded-in sign), or `None`.
fn pat_int_value(p: &Pattern, interner: &Interner) -> Option<i64> {
    match &p.kind {
        PatKind::Int { sym, neg } => {
            parse_int_text(interner.resolve(*sym)).map(|v| if *neg { v.wrapping_neg() } else { v })
        }
        _ => None,
    }
}

/// Record whether a (guard-less) pattern covers the `true` and/or `false` case, flattening
/// or-patterns. Used by `match_is_provably_nonexhaustive` for `bool` coverage.
fn collect_bool_cases(p: &Pattern, t: &mut bool, f: &mut bool) {
    match &p.kind {
        PatKind::Bool(true) => *t = true,
        PatKind::Bool(false) => *f = true,
        PatKind::Or(alts) => alts.iter().for_each(|a| collect_bool_cases(a, t, f)),
        _ => {}
    }
}

/// Whether `b` contains a `break` anywhere (recursively). Used to decide whether a `loop` is
/// infinite. Conservative: descending into nested loops may count a `break` bound to an inner loop,
/// which only makes the outer analysis *more* lenient (assume it can exit), never causing a false
/// definite-return error. `break`/`continue` are statements (never inside a value expression), so
/// only statement positions and the block/if/match that hold statements need scanning.
fn block_contains_break(b: &Block) -> bool {
    b.stmts.iter().any(stmt_contains_break)
        || b.tail.as_deref().map(expr_contains_break).unwrap_or(false)
}

fn stmt_contains_break(s: &Stmt) -> bool {
    match &s.kind {
        StmtKind::Break(..) => true,
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_contains_break(e),
        StmtKind::Return(opt) => opt.as_ref().map(expr_contains_break).unwrap_or(false),
        StmtKind::Let { init, .. } => init.as_ref().map(expr_contains_break).unwrap_or(false),
        StmtKind::Assign { value, .. } => expr_contains_break(value),
        StmtKind::While { body, .. } | StmtKind::For { body, .. } => block_contains_break(body),
        StmtKind::Continue(_) => false,
    }
}

fn expr_contains_break(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Block(b) => block_contains_break(b),
        // A nested `loop` is itself breakable, so descending counts a `break` bound to it — the same
        // lenient over-count the statement-loop arm made before `loop` became an expression (only
        // ever makes the enclosing loop analysis assume it *can* exit, never a false definite-return).
        ExprKind::Loop { body, .. } => block_contains_break(body),
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            block_contains_break(then_branch)
                || else_branch
                    .as_deref()
                    .map(expr_contains_break)
                    .unwrap_or(false)
        }
        ExprKind::Match { arms, .. } => arms.iter().any(|a| expr_contains_break(&a.body)),
        _ => false,
    }
}

/// The inclusive value range of a narrow integer scalar, as `i64` bounds. `None` for `bool`/`char`/
/// floats and for `i64`/`u64`/`usize`/`isize` — a literal that parses to an `i64` always fits those,
/// and a `u64` near its top doesn't fit an `i64` to compare, so they are left unchecked rather than
/// mis-flagged.
fn int_lit_range(sc: Scalar) -> Option<(i64, i64)> {
    use Scalar::*;
    Some(match sc {
        I8 => (i8::MIN as i64, i8::MAX as i64),
        U8 => (0, u8::MAX as i64),
        I16 => (i16::MIN as i64, i16::MAX as i64),
        U16 => (0, u16::MAX as i64),
        I32 => (i32::MIN as i64, i32::MAX as i64),
        U32 => (0, u32::MAX as i64),
        _ => return None,
    })
}

fn int_lit_scalar(text: &str) -> Scalar {
    for (suf, sc) in [
        ("usize", Scalar::Usize),
        ("isize", Scalar::Isize),
        ("i8", Scalar::I8),
        ("i16", Scalar::I16),
        ("i32", Scalar::I32),
        ("i64", Scalar::I64),
        ("u8", Scalar::U8),
        ("u16", Scalar::U16),
        ("u32", Scalar::U32),
        ("u64", Scalar::U64),
    ] {
        if text.ends_with(suf) {
            return sc;
        }
    }
    // Unsuffixed: default to i32, but a literal that does not fit i32 widens to i64 so it is never
    // *silently* truncated. `9000000000` baked as a `const.i32` wraps to 410065408 on BOTH backends,
    // so the interp-vs-native differential gate cannot see the error — the only defense is to not
    // mis-default in the first place. Growing the default to the narrowest type that holds the value
    // mirrors an unconstrained `{integer}` literal and keeps every existing narrowing check honest: a
    // pinned annotation still wins (`let x: i32 = 9000000000` re-adapts the literal back to i32 and
    // range-checks it -> E0401), while a wider return / field / bare-expression context now lowers
    // the true value. A magnitude past i64 but within u64 (e.g. `9223372036854775808`) widens one
    // more rung to `u64` — it must not silently truncate to i32 either (`… as u64` was baking
    // `const.i32 0` on both backends, gate-blind).
    match parse_int_text(text) {
        Some(v) if v < i32::MIN as i64 || v > i32::MAX as i64 => Scalar::I64,
        Some(_) => Scalar::I32,
        // `parse_int_text` returns None for a magnitude that overflows i64. If it still fits u64 (a
        // literal in `(i64::MAX, u64::MAX]`), default to u64; a value past u64 is malformed and is
        // already rejected by `int_literal_well_formed`, so the i32 fallback there is unreachable.
        None if parse_u64_text(text).is_some() => Scalar::U64,
        None => Scalar::I32,
    }
}

fn float_lit_scalar(text: &str) -> Scalar {
    if text.ends_with("bf16") {
        Scalar::Bf16
    } else if text.ends_with("f16") {
        Scalar::F16
    } else if text.ends_with("f64") {
        Scalar::F64
    } else {
        Scalar::F32
    }
}

fn has_int_suffix(text: &str) -> bool {
    [
        "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "usize", "isize",
    ]
    .iter()
    .any(|s| text.ends_with(s))
}

fn has_float_suffix(text: &str) -> bool {
    ["f16", "bf16", "f32", "f64"]
        .iter()
        .any(|s| text.ends_with(s))
}

/// Evaluate a constant integer expression to its value: an integer literal, a unary `-`/`!` of one,
/// or folded arithmetic / bitwise / shift over such operands (wrapping, matching the optimizer's
/// constant folder; a division or remainder by zero declines). Returns `None` for anything else — a
/// name, a call, a comparison. Used for an enum variant's explicit discriminant (`A = 10`, which then
/// auto-increments when this declines), for `range_check_int_literal`, and for the compile-time
/// index-bounds check via `eval_index_const`. NOTE: this is *not* the array-length evaluator —
/// lengths go through `eval_usize_depth`, which is the arm-for-arm mirror of `mir_build`.
fn eval_const_int(e: &Expr, interner: &Interner) -> Option<i64> {
    match &e.kind {
        ExprKind::Int(s) => parse_int_text(interner.resolve(*s)),
        ExprKind::Unary {
            op: UnOp::Neg,
            expr,
        } => eval_const_int(expr, interner).map(|v| v.wrapping_neg()),
        ExprKind::Unary {
            op: UnOp::Not,
            expr,
        } => eval_const_int(expr, interner).map(|v| !v),
        // Fold constant integer arithmetic / bitwise / shift — the *same* ops the optimizer's
        // constant folder evaluates before the backend runs. Without this, a constant-arithmetic
        // index like `xs[2 + 3]` (vs the literal `xs[5]`) returned `None` here, so the compile-time
        // bounds check silently skipped it: the optimizer then folded `2 + 3` -> 5 and the
        // out-of-bounds access reached the backend (interp trap vs native past-the-buffer read — a
        // divergence; or, for a multi-dim index that lands in-buffer, a silently wrong element on
        // both). Wrapping semantics match the runtime/folder; div/rem by zero stays `None` (the
        // runtime guards it to 0, in bounds, so skipping the check is safe and consistent).
        ExprKind::Binary { op, lhs, rhs } => {
            let l = eval_const_int(lhs, interner)?;
            let r = eval_const_int(rhs, interner)?;
            Some(match op {
                BinOp::Add => l.wrapping_add(r),
                BinOp::Sub => l.wrapping_sub(r),
                BinOp::Mul => l.wrapping_mul(r),
                BinOp::Div => {
                    if r == 0 {
                        return None;
                    }
                    l.wrapping_div(r)
                }
                BinOp::Rem => {
                    if r == 0 {
                        return None;
                    }
                    l.wrapping_rem(r)
                }
                BinOp::BitAnd => l & r,
                BinOp::BitOr => l | r,
                BinOp::BitXor => l ^ r,
                BinOp::Shl => l.wrapping_shl(r as u32),
                BinOp::Shr => l.wrapping_shr(r as u32),
                // comparisons / logical ops don't yield an integer index value.
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Whether an integer literal's text denotes a value Wukong can represent — it parses, after the
/// optional sign / type suffix / `_` separators and in its radix, as an i64 *or* a u64. A mistyped
/// radix like `0z123` (lexed as one `Int` token with a bogus `z123` suffix), an empty/garbled radix
/// body, or a value past u64 parses as neither; such a literal used to lower silently to 0. Mirrors
/// `parse_int_text`'s stripping so the two agree on what a well-formed literal is.
fn int_literal_well_formed(text: &str) -> bool {
    let mut s = text.trim();
    if s.starts_with('-') || s.starts_with('+') {
        s = &s[1..];
    }
    for suf in [
        "usize", "isize", "u128", "i128", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
    ] {
        if let Some(x) = s.strip_suffix(suf) {
            s = x;
            break;
        }
    }
    // Keep separator-free spellings borrowed, as the parser's extent decoder does.
    let normalized;
    let body = if s.contains('_') {
        normalized = s.replace('_', "");
        normalized.as_str()
    } else {
        s
    };
    let (digits, radix): (&str, u32) =
        if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
            (h, 16)
        } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
            (o, 8)
        } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
            (b, 2)
        } else {
            (body, 10)
        };
    !digits.is_empty()
        && (i64::from_str_radix(digits, radix).is_ok()
            || u64::from_str_radix(digits, radix).is_ok())
}

/// Whether a float literal's text parses as an `f64` after stripping the optional sign, a float type
/// suffix, and `_` separators — so a garbled literal the greedy suffix scan swept into one `Float`
/// token (`1.5z`, an incomplete exponent `1.5e`) is rejected instead of silently lowering to 0.0.
fn float_literal_well_formed(text: &str) -> bool {
    let mut s = text.trim();
    if s.starts_with('-') || s.starts_with('+') {
        s = &s[1..];
    }
    // `f` alone is the C-style float suffix (`5f` -> f32); try the longer suffixes first.
    for suf in ["bf16", "f16", "f32", "f64", "f"] {
        if let Some(x) = s.strip_suffix(suf) {
            s = x;
            break;
        }
    }
    // Keep separator-free spellings borrowed, as the parser's extent decoder does.
    let normalized;
    let body = if s.contains('_') {
        normalized = s.replace('_', "");
        normalized.as_str()
    } else {
        s
    };
    !body.is_empty() && body.parse::<f64>().is_ok()
}

/// The first malformed escape in a char/string literal's raw source text, as a ready diagnostic
/// message, or `None` when every escape is one `decode_escape` can read.
///
/// The accepted set is exactly `decode_escape`'s: `\n \r \t \\ \' \" \0`, `\xHH`, and
/// `\u{…}`. The scan consumes the escaped character (so a literal `\\` is not mistaken for the
/// start of a new escape) and, for a well-formed `\u{…}`, consumes through the closing brace, so
/// hex digits inside it are never read as source text.
fn malformed_escape(text: &str) -> Option<String> {
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            continue;
        }
        match chars.next() {
            // `\xHH`'s digits are plain characters to this scan; the lexer already bounded the
            // escape to at most two of them, and a non-hex digit there folds like the decoder does.
            Some('n' | 'r' | 't' | '\\' | '\'' | '"' | '0' | 'x') => {}
            Some('u') => match chars.next() {
                Some('{') => {
                    // The closing brace is not optional. Without the flag this loop ran the
                    // iterator to the end of the literal and returned `None`, and
                    // `bad_unicode_escape`'s `take_while(|&c| c != '}')` did the same — its
                    // `map_or(v, ..)` skips the non-hex characters, so `"A\u{41Zoo"` folded to a
                    // perfectly valid 0x41 and BOTH validators stayed silent while mir_build's
                    // `decode_escape` deleted the rest of the string. That is the worse half of the
                    // very finding the `\u`-with-no-brace arm below closes.
                    let mut closed = false;
                    for c in chars.by_ref() {
                        if c == '}' {
                            closed = true;
                            break;
                        }
                    }
                    if !closed {
                        return Some(
                            "unterminated `\\u{…}` escape: the closing `}` is missing, so the \
                             decoder swallows the rest of the literal — write it as `\\u{1F600}`"
                                .to_string(),
                        );
                    }
                }
                _ => {
                    return Some(
                        "malformed `\\u` escape: `\\u` must be followed immediately by `{`, as in \
                         `\\u{1F600}` — without the brace the decoder swallows the rest of the \
                         literal"
                            .to_string(),
                    )
                }
            },
            Some(c) => {
                return Some(format!(
                    "unknown escape `\\{c}` in a literal (the escapes Wukong decodes are `\\n` \
                     `\\r` `\\t` `\\\\` `\\'` `\\\"` `\\0` `\\xHH` `\\u{{…}}`); write `\\\\{c}` for a \
                     backslash followed by `{c}`"
                ))
            }
            None => {
                return Some(
                    "a literal ends with a lone `\\`; write `\\\\` for a backslash".to_string(),
                )
            }
        }
    }
    None
}

/// The first `\u{…}` escape in a char/string literal's raw source text that is NOT a Unicode scalar
/// value (above `0x10FFFF`, or a surrogate `0xD800..=0xDFFF`), or `None` if every escape is fine.
///
/// mir_build's `decode_escape` saturates the accumulator, `char::from_u32` then returns `None`, and
/// `decode_string_literal`'s fallback pushes `cp as u8`: `"a\u{110000}b"` pushed a NUL, so a
/// `println` of it stopped before the `b` and the rest of the string was silently lost, while the
/// char form printed 1114112 — a value the language guide says a `char` cannot hold. This mirrors
/// `decode_escape`'s scan exactly, so it accepts precisely the set the decoder can represent.
fn bad_unicode_escape(text: &str) -> Option<u32> {
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            continue;
        }
        // Every other escape's body is plain text to this scan; consuming the escaped character is
        // what matters, so a literal `\\` is not mistaken for the start of a new escape.
        if chars.next() != Some('u') {
            continue;
        }
        let cp = chars
            .by_ref()
            .skip_while(|&c| c != '{')
            .skip(1)
            .take_while(|&c| c != '}')
            .fold(0u32, |v, c| {
                c.to_digit(16)
                    .map_or(v, |d| v.saturating_mul(16).saturating_add(d))
            });
        if char::from_u32(cp).is_none() {
            return Some(cp);
        }
    }
    None
}

/// Parse an integer literal's source text (decimal, `0x`/`0o`/`0b` radix, `_` separators, optional
/// type suffix, optional leading sign) to an `i64`, or `None` if it isn't a valid integer literal —
/// including a magnitude that overflows `i64` (see [`parse_u64_text`] for that rung).
fn parse_int_text(text: &str) -> Option<i64> {
    let mut s = text.trim();
    let neg = s.starts_with('-');
    if neg || s.starts_with('+') {
        s = &s[1..];
    }
    for suf in [
        "usize", "isize", "u128", "i128", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
    ] {
        if let Some(x) = s.strip_suffix(suf) {
            s = x;
            break;
        }
    }
    // Keep separator-free spellings borrowed, as the parser's extent decoder does.
    let normalized;
    let body = if s.contains('_') {
        normalized = s.replace('_', "");
        normalized.as_str()
    } else {
        s
    };
    let v = if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        i64::from_str_radix(h, 16)
    } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        i64::from_str_radix(o, 8)
    } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        i64::from_str_radix(b, 2)
    } else {
        body.parse::<i64>()
    }
    .ok()?;
    Some(if neg { -v } else { v })
}

/// Parse an integer literal's MAGNITUDE as a `u64` (no sign — a negative value fits `i64` and is
/// handled by `parse_int_text`). Used to recognize a literal in `(i64::MAX, u64::MAX]` so it defaults
/// to `u64` instead of silently truncating to i32. Mirrors `parse_int_text`'s radix/suffix handling.
pub(crate) fn parse_u64_text(text: &str) -> Option<u64> {
    let mut s = text.trim();
    if s.starts_with('-') {
        return None;
    }
    if s.starts_with('+') {
        s = &s[1..];
    }
    for suf in [
        "usize", "isize", "u128", "i128", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
    ] {
        if let Some(x) = s.strip_suffix(suf) {
            s = x;
            break;
        }
    }
    // Keep separator-free spellings borrowed, as the parser's extent decoder does.
    let normalized;
    let body = if s.contains('_') {
        normalized = s.replace('_', "");
        normalized.as_str()
    } else {
        s
    };
    if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).ok()
    } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        u64::from_str_radix(o, 8).ok()
    } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        u64::from_str_radix(b, 2).ok()
    } else {
        body.parse::<u64>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::SourceId;

    fn analyze(src: &str) -> (Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (_res, diags) = check(&module, &interner);
        (diags, interner)
    }

    fn errors(src: &str) -> Vec<&'static str> {
        analyze(src).0.into_iter().filter_map(|d| d.code).collect()
    }

    /// The headline messages of every diagnostic carrying `code`. The shape diagnostics are asserted
    /// on their TEXT and not only their code: naming the disagreeing dimension is the contract, and
    /// a message that regresses to a bare "expected 512, found 513" still carries the right code.
    fn messages_for(src: &str, code: &str) -> Vec<String> {
        analyze(src)
            .0
            .into_iter()
            .filter(|d| d.code == Some(code))
            .map(|d| d.message)
            .collect()
    }

    /// Assert that some diagnostic with `code` contains every one of `needles`.
    #[track_caller]
    fn assert_message_has(src: &str, code: &str, needles: &[&str]) {
        let msgs = messages_for(src, code);
        assert!(
            msgs.iter().any(|m| needles.iter().all(|n| m.contains(n))),
            "no {code} message contains {needles:?}; got {msgs:?} for {src:?}"
        );
    }

    #[test]
    fn aggregate_layout_overflow_diagnostics() {
        for src in [
            "fn f() { let t: ([[i64;4000000000];4000000000], i32); }",
            "fn f() { let s: shared [[i64;4000000000];4000000000]; }",
            "fn f() { let t: ([u8;4294967295], u8); }",
            "fn f() { let t: ([u8;4294967293], i16); }",
            "struct S { a: [u8;4294967295], b: u8 }",
            "enum E { A([u8;4294967295]), B }",
            "struct Outer { a: [Inner;4000000000] } struct Inner { x: i64 }",
            "fn f(a: [i64;4000000000]) { let t = (a, 1); }",
        ] {
            assert_message_has(src, "E0401", &["layout is too large to represent", "byte"]);
        }
    }

    #[test]
    fn aggregate_layout_overflow_named_dependency_controls() {
        for src in ["struct S { x: S }", "enum E { A(S) } struct S { e: E }"] {
            let diags = errors(src);
            assert!(diags.contains(&"E0402"), "{src}: {diags:?}");
            assert!(!diags.contains(&"E0401"), "{src}: {diags:?}");
        }
        let mut source = String::from("struct S0 { x: i32 }");
        for i in 1..2048 {
            source.push_str(&format!("struct S{i} {{ x: S{} }}", i - 1));
        }
        let (diags, _) = analyze(&source);
        assert!(diags.is_empty(), "{diags:?}");
    }

    #[test]
    fn aggregate_layout_overflow_controls() {
        for src in [
            "fn f(a: [i64;4000000000]) {}",
            "fn f(a: *[i64;4000000000]) {}",
            "fn f() { let t: ([[i64;0];4000000000], i32); }",
            "fn f() { let t: ([();4000000000], i32); }",
            "fn f(a: []i32, b: Tensor[f32, ?]) { let t: ([]i32, i32) = (a, 7); }",
            "struct S { a: u8, b: i64 } enum E { A(S), B } fn f() { let s: (S, [u8;0]); }",
            "fn f() { let t: ([u8;4294967295], ()); }",
        ] {
            let (diags, _) = analyze(src);
            assert!(diags.is_empty(), "{src}: {diags:?}");
        }
    }

    #[test]
    fn clean_kernel_has_no_errors() {
        let src = "fn add<N>(x: Tensor[f32, N], y: Tensor[f32, N]) { \
                   let mut i: usize = 0; while i < N { let v = x[i] + y[i]; i += 1; } }";
        let (diags, _) = analyze(src);
        assert!(diags.is_empty(), "unexpected: {diags:?}");
    }

    #[test]
    fn unresolved_name_errors() {
        let (diags, _) = analyze("fn f() { let x = nonexistent; }");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, Some("E0301"));
    }

    #[test]
    fn builtin_methods_are_lenient() {
        // f32x8::load and .offset are not modeled, but must not error.
        let src = "fn f(p: *f32) { let v = f32x8::load(p.offset(0)); }";
        let (diags, _) = analyze(src);
        assert!(diags.is_empty(), "unexpected: {diags:?}");
    }

    #[test]
    fn math_intrinsics_are_typed_as_float() {
        // sqrt/rsqrt/exp/fmax/fmin are lowered directly by the backends, so sema gives them a real
        // float result type (not the lenient `Unknown`): kernels type-check and lowering knows the
        // result is a float.
        let ok = "fn f(x: f32, y: f32) -> f32 { \
                  let a = sqrt(x); let b = exp(y); let c = fmax(a, b); return rsqrt(c); }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        // Because the result is `f32` (not `Unknown`), binding it to a non-float annotation must
        // conflict — this distinguishes the modeled signature from the lenient fallback.
        let bad = "fn f(x: f32) { let n: i64 = exp(x); }";
        assert!(
            errors(bad).contains(&"E0401"),
            "expected a type mismatch: {:?}",
            errors(bad)
        );
    }

    #[test]
    fn heap_builtins_are_typed() {
        // `alloc_<T>(n)` types as `[]T` (so it binds/passes as a slice) and `free(s)` as unit; any
        // integer length is accepted, including one inferred from an unsuffixed literal.
        let ok = "fn take(s: []f32) -> i64 { return s.len(); } \
                  fn f() -> i64 { let n: i64 = 12; let s: []f32 = alloc_f32(n); \
                  let t = alloc_i32(4); let r = take(s); free(s); free(t); return r; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
    }

    #[test]
    fn slice_len_is_typed_i64() {
        // `s.len()` is a modeled builtin method on slices (result `i64`), so binding it to a
        // conflicting annotation errors — distinguishing it from the lenient `Unknown` fallback
        // (under which `for i in 0..s.len()` ICE'd on a mixed-width Cmp in MIR).
        let ok = "fn f(s: []f32) -> i64 { let n: i64 = s.len(); return n; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        let bad = "fn f(s: []f32) { let n: f32 = s.len(); }";
        assert!(
            errors(bad).contains(&"E0401"),
            "expected a type mismatch: {:?}",
            errors(bad)
        );
    }

    #[test]
    fn heap_builtin_misuse_is_rejected() {
        // A non-integer length is an E0401 (the `alloc("x")` class of misuse) …
        assert!(errors("fn f() { let s = alloc_f32(\"x\"); }").contains(&"E0401"));
        assert!(errors("fn f() { let s = alloc_i32(1.5); }").contains(&"E0401"));
        // … the wrong arity is an E0503 …
        assert!(errors("fn f() { let s = alloc_f32(); }").contains(&"E0503"));
        assert!(errors("fn f() { let s = alloc_f32(1, 2); }").contains(&"E0503"));
        assert!(errors("fn f() { free(); }").contains(&"E0503"));
        // … and freeing a non-slice is an E0401.
        assert!(errors("fn f() { free(5); }").contains(&"E0401"));
    }

    #[test]
    fn file_io_builtins_are_typed() {
        // `read_<T>(path, buf)` / `write_<T>(path, buf)` type as `i64` (the element count / return
        // code), so the result composes in arithmetic — mirroring the `alloc_*` heap family. `path`
        // is a `*u8` (a string literal); `buf` is a `[]T` slice from an `alloc_<T>` builtin.
        let ok = "fn f() -> i64 { \
                  let s: []f32 = alloc_f32(4); \
                  let n: i64 = read_f32(\"in.bin\", s); \
                  let w = write_i32(\"out.bin\", alloc_i32(8)); \
                  return n + w; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        // The later-wave dtypes (f64, i8) type identically: a well-typed `read_f64` / `write_i8`
        // over a `[]f64` / `[]i8` buffer composes as `i64` just like the core family.
        let ok2 = "fn f() -> i64 { \
                   let s: []f64 = alloc_f64(4); \
                   let n: i64 = read_f64(\"in.bin\", s); \
                   let w = write_i8(\"out.bin\", alloc_i8(8)); \
                   return n + w; }";
        assert!(errors(ok2).is_empty(), "unexpected: {:?}", errors(ok2));
        // Because the result is `i64` (not the lenient `Unknown`), binding it to a conflicting
        // annotation must error — this distinguishes the modeled signature from the fallback.
        let bad = "fn f() { let n: f32 = read_u8(\"in.bin\", alloc_u8(4)); }";
        assert!(
            errors(bad).contains(&"E0401"),
            "expected a type mismatch: {:?}",
            errors(bad)
        );
    }

    #[test]
    fn file_io_builtin_misuse_is_rejected() {
        // A non-`*u8` path is an E0401 …
        assert!(
            errors("fn f() { let s = alloc_f32(4); let n = read_f32(5, s); }").contains(&"E0401")
        );
        // … a buffer whose element type does not match the name's dtype is an E0401 …
        assert!(
            errors("fn f() { let s = alloc_i32(4); let n = read_f32(\"in.bin\", s); }")
                .contains(&"E0401")
        );
        assert!(
            errors("fn f() { let s = alloc_f32(4); let n = write_u8(\"o.bin\", s); }")
                .contains(&"E0401")
        );
        // … and the later-wave dtypes reject a mismatched buffer element the same way (a `[]f64`
        // handed to `read_i8`, or a `[]i32` handed to `write_f64`, is an E0401) …
        assert!(
            errors("fn f() { let s = alloc_f64(4); let n = read_i8(\"in.bin\", s); }")
                .contains(&"E0401")
        );
        assert!(
            errors("fn f() { let s = alloc_i32(4); let n = write_f64(\"o.bin\", s); }")
                .contains(&"E0401")
        );
        // … a non-slice buffer is an E0401 …
        assert!(errors("fn f() { let n = read_f32(\"in.bin\", 3); }").contains(&"E0401"));
        // … and the wrong arity is an E0503.
        assert!(errors("fn f() { let n = read_f32(\"in.bin\"); }").contains(&"E0503"));
        assert!(
            errors("fn f() { let s = alloc_f32(4); let n = write_f32(\"o.bin\", s, s); }")
                .contains(&"E0503")
        );
    }

    #[test]
    fn let_annotation_mismatch_errors() {
        let (diags, _) = analyze("fn f() { let x: f32 = true; }");
        assert!(diags.iter().any(|d| d.code == Some("E0401")));
    }

    #[test]
    fn negated_literal_adapts_to_let_annotation() {
        // A leading `-` does not change a literal's kind, so an unsuffixed negated literal must
        // still adopt the annotation (`f64`/`i64` here), not stay at the default `f32`/`i32`.
        for src in [
            "fn f() { let x: f64 = -1.5; }",
            "fn f() { let x: i64 = -7; }",
            "fn f() { let x: f64 = 2.0; }",
        ] {
            let (diags, _) = analyze(src);
            assert!(diags.is_empty(), "unexpected for {src:?}: {diags:?}");
        }
        // A *suffixed* literal still pins its type and must conflict.
        let (diags, _) = analyze("fn f() { let x: f64 = -1.5f32; }");
        assert!(diags.iter().any(|d| d.code == Some("E0401")));
    }

    #[test]
    fn match_nonexhaustive_in_all_positions() {
        // A provably-non-exhaustive match is E0405 in EVERY position — value, statement (unit arms),
        // and empty — because the "no arm matched" fallthrough lowers to `Unreachable`, which the
        // interpreter and native backend trap differently (a divergence). Was value-position only.
        for src in [
            "fn f(n: i32) { match n { 0 => {}, 1 => {} } }", // statement, unit arms
            "fn f(n: i32) -> i32 { let x = match n {}; return x; }", // empty match
            "fn f(n: i32) -> i32 { return match n { 0 => 1, 1 => 2 }; }", // value (regression)
            "enum E { A, B } fn f(e: E) { match e {} }",     // empty enum match
            "fn f(b: bool) { match b { true => {} } }",      // bool missing a case
        ] {
            assert!(errors(src).contains(&"E0405"), "expected E0405 for {src:?}");
        }
        // Exhaustive matches stay valid in every position (no false positive).
        for src in [
            "fn f(b: bool) { match b { true => {}, false => {} } }",
            "enum E { A, B, C } fn f(e: E) { match e { E::A => {}, E::B => {}, E::C => {} } }",
            "fn f(n: i32) { match n { 0 => {}, _ => {} } }",
            "fn f(n: i32) -> i32 { return match n { 0 => 1, _ => 2 }; }",
        ] {
            assert!(
                !errors(src).contains(&"E0405"),
                "unexpected E0405 for {src:?}"
            );
        }
    }

    #[test]
    fn malformed_numeric_literals_are_rejected() {
        // A mistyped radix / empty radix / bad digit / value past u64 / garbled float lexes as one
        // numeric token, fails to parse, and used to lower *silently to 0* (both backends agreed on
        // the wrong value) — now a hard E0401.
        for src in [
            "fn f() { let x = 0z123; }",
            "fn f() { let x = 0x; }",
            "fn f() { let x = 0b2; }",
            "fn f() { let x = 1.5z; }",
        ] {
            assert!(errors(src).contains(&"E0401"), "expected E0401 for {src:?}");
        }
        // Every well-formed literal stays clean (guards against over-rejection): radix, separators,
        // type suffixes, u64::MAX, and the float forms.
        for src in [
            "fn f() { let x = 0xFF; }",
            "fn f() { let x = 0o17; }",
            "fn f() { let x = 0b1010; }",
            "fn f() { let x = 1_000_000; }",
            "fn f() { let x: u8 = 250u8; }",
            "fn f() { let x: u64 = 18446744073709551615; }",
            "fn f() { let x = 1.5e3; }",
            "fn f() { let x = 5f32; }",
            "fn f() { let x = 9000000000; }",
        ] {
            assert!(
                !errors(src).contains(&"E0401"),
                "unexpected E0401 for {src:?}"
            );
        }
    }

    #[test]
    fn if_match_arm_shapes_must_agree() {
        // if/match arms merge into one value, so their tensor shapes / element types must agree — the
        // 4th shape-bearing context after binop/call/return. Mismatch -> E0502.
        for src in [
            "fn p(c: bool, a: Tensor[f32,2,2], b: Tensor[f32,2,4]) -> f32 { let x = if c {a} else {b}; return x[0,0]; }",
            "fn p(c: bool, a: Tensor[f32,4], b: Tensor[i32,4]) -> f32 { let x = if c {a} else {b}; return x[0]; }",
            "fn p(s: i32, a: Tensor[f32,4], b: Tensor[i32,4]) -> f32 { let x = match s { 0 => a, _ => b }; return x[0]; }",
        ] {
            assert!(errors(src).contains(&"E0502"), "expected E0502 for {src:?}");
        }
        // No false positives: same-shape tensor arms, and scalar/float arms (which `join` handles).
        for src in [
            "fn p(c: bool, a: Tensor[f32,2,4], b: Tensor[f32,2,4]) -> f32 { let x = if c {a} else {b}; return x[0,0]; }",
            "fn f(c: bool) -> i32 { let x = if c { 1 } else { 2 }; return x; }",
            "fn f(s: i32) -> i32 { let x = match s { 0 => 10, _ => 20 }; return x; }",
            "fn f(c: bool) -> f64 { let x = if c { 1.0 } else { 2.0 }; return x as f64; }",
        ] {
            assert!(!errors(src).contains(&"E0502"), "unexpected E0502 for {src:?}");
        }
    }

    #[test]
    fn ordered_comparison_on_bool_is_rejected() {
        // A chained comparison `a < b < c` parses left-associatively as `(a < b) < c`; the inner
        // `<` yields a bool, so the outer silently compares a bool against an int. Reject a concrete
        // bool operand to an ordered comparison (`< <= > >=`). E0401.
        for src in [
            "fn f(a: i32) -> bool { return 1 < a < 2; }",
            "fn f(a: i32) -> bool { return a > 0 > 1; }",
            "fn f() -> bool { let x = true; let y = false; return x < y; }",
        ] {
            assert!(errors(src).contains(&"E0401"), "expected E0401 for {src:?}");
        }
        // No false positives: ordinary int/float ordering, the correct `&&` chain, and `==`/`!=`
        // and `&&`/`||`/`!` on bool all stay clean (only ORDERED comparison rejects bool).
        for src in [
            "fn f(a: i32, b: i32) -> bool { return a < b; }",
            "fn f(a: i32) -> bool { return 1 < a && a < 2; }",
            "fn f(a: f32, b: f32) -> bool { return a >= b; }",
            "fn f() -> bool { let x = true; let y = false; return x == y || !x; }",
        ] {
            assert!(
                !errors(src).contains(&"E0401"),
                "unexpected E0401 for {src:?}"
            );
        }
    }

    #[test]
    fn chained_equality_rejected() {
        // `a == b == c` parses as `(a == b) == c`; the inner `==` yields a bool, so the outer
        // compares a bool against a non-bool scalar (the bool coerced to 0/1) — `5 == 3 == 0`
        // evaluated to `true`. Equality between bool and a non-bool scalar is rejected, completing
        // the chained-comparison guard for `==`/`!=`. E0401.
        for src in [
            "fn f() -> bool { return 5 == 3 == 0; }",
            "fn f() -> bool { return 1 != 2 != 3; }",
            "fn f(a: i32) -> bool { return 1 < a == 0; }", // (1 < a) == 0  ->  bool == int
        ] {
            assert!(errors(src).contains(&"E0401"), "expected E0401 for {src:?}");
        }
        // No false positives: `bool == bool`, `int == int`, and comparison-of-comparisons stay clean.
        for src in [
            "fn f() -> bool { let a = true; let b = false; return a == b; }",
            "fn f(a: i32, b: i32) -> bool { return a == b; }",
            "fn f(a: i32, b: i32, c: i32, d: i32) -> bool { return (a < b) == (c < d); }",
        ] {
            assert!(
                !errors(src).contains(&"E0401"),
                "unexpected E0401 for {src:?}"
            );
        }
    }

    #[test]
    fn bool_arithmetic_and_index_rejected() {
        // bool is not a number: it cannot be an arithmetic / shift operand or an array index (it
        // would silently coerce to 0/1 — `true + 1 == 2`, `a[true] == a[1]`). E0401.
        for src in [
            "fn f() -> i32 { let x = true; return x + 1; }",
            "fn f() -> i32 { let x = true; return x * 3; }",
            "fn f() -> i32 { let a: [i32; 2] = [1, 2]; return a[true]; }",
        ] {
            assert!(errors(src).contains(&"E0401"), "expected E0401 for {src:?}");
        }
        // Still fine: bitwise `& | ^` and `&&`/`||`/`!` on bool, and an integer index.
        for src in [
            "fn f() -> bool { let a = true; let b = false; return a & b | (a ^ b); }",
            "fn f() -> bool { let a = true; return !a && (a || a); }",
            "fn f() -> i32 { let a: [i32; 2] = [10, 20]; let i = 1; return a[i]; }",
        ] {
            assert!(
                !errors(src).contains(&"E0401"),
                "unexpected E0401 for {src:?}"
            );
        }
    }

    #[test]
    fn char_type_is_usable() {
        // A char literal types as `char`, the `char` annotation resolves to the same scalar, and
        // char <-> int casts are valid in both directions — none of these should error.
        for src in [
            "fn f() -> i32 { let c: char = 'A'; return c as i32; }",
            "fn f() -> char { return 66 as char; }",
            "fn f() -> bool { let a: char = 'a'; let b: char = 'b'; return a < b; }",
            "struct G { code: char } fn f() -> i32 { let g = G { code: 'Z' }; return g.code as i32; }",
        ] {
            assert!(errors(src).is_empty(), "unexpected errors for {src:?}: {:?}", errors(src));
        }
    }

    #[test]
    fn array_kernel_has_no_false_errors() {
        // Array declaration, indexed store/load, casts, and a reduction must type-check cleanly.
        let src = "fn main() -> i32 { \
                   let mut xs: [i32; 4] = [0, 0, 0, 0]; \
                   let mut i: i32 = 0; \
                   while i < 4 { xs[i] = i * i; i = i + 1; } \
                   let f: f32 = xs[2] as f32; let n: i32 = f as i32; \
                   return xs[0] + n; }";
        let (diags, _) = analyze(src);
        assert!(diags.is_empty(), "unexpected: {diags:?}");
    }

    // ---- Shape checking (the headline feature) ----

    const MATMUL: &str = "fn matmul<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N], \
                          c: Tensor[f32, M, N]) {}";

    #[test]
    fn matmul_call_with_consistent_shapes_ok() {
        let src = format!(
            "{MATMUL}\nfn driver(a: Tensor[f32, 512, 512], b: Tensor[f32, 512, 512], \
             c: Tensor[f32, 512, 512]) {{ matmul::<512, 512, 512>(a, b, c); }}"
        );
        assert!(errors(&src).is_empty(), "unexpected: {:?}", errors(&src));
    }

    #[test]
    fn matmul_call_with_bad_inner_dim_errors() {
        // c is 512x513 but the signature forces c: [M, N] = [512, 512].
        let src = format!(
            "{MATMUL}\nfn driver(a: Tensor[f32, 512, 512], b: Tensor[f32, 512, 512], \
             c: Tensor[f32, 512, 513]) {{ matmul::<512, 512, 512>(a, b, c); }}"
        );
        assert!(
            errors(&src).contains(&"E0502"),
            "expected a dimension mismatch: {:?}",
            errors(&src)
        );
    }

    #[test]
    fn matmul_shape_inference_detects_k_conflict() {
        // Without turbofish: a binds K=512, but b's first dim is 500 -> K conflict.
        let src = format!(
            "{MATMUL}\nfn driver(a: Tensor[f32, 512, 512], b: Tensor[f32, 500, 512], \
             c: Tensor[f32, 512, 512]) {{ matmul(a, b, c); }}"
        );
        assert!(
            errors(&src).contains(&"E0502"),
            "expected a K conflict: {:?}",
            errors(&src)
        );
    }

    #[test]
    fn generic_function_cannot_lie_about_its_shape() {
        // A generic function's own dimension variables are RIGID inside its body: the function must
        // not declare a return shape (or merge / operate on operand shapes) that its body does not
        // actually produce. Were these treated like call-site inference vars, `grow` below would
        // type-check, and a turbofished caller (`grow::<2, 2>`) would then propagate a
        // [2,5]=10-element claim from a 4-element buffer and index out of bounds — the interpreter
        // traps while native reads past the buffer (a backend divergence on a program that should
        // never have compiled). Each of these is a shape lie -> E0502.
        for src in [
            // Return-shape lie: a distinct generic (`N` vs `M`) AND a constant (`5`) vs a generic.
            "fn grow<M, N>(a: Tensor[f32, M, N]) -> Tensor[f32, N, 5] { return a; }",
            // Rank-1 lie: a constant length claimed from an arbitrary generic.
            "fn g<N>(a: Tensor[f32, N]) -> Tensor[f32, 8] { return a; }",
            // Swapped dims: returning `Tensor[M, N]` as `Tensor[N, M]` (mis-strides for M != N).
            "fn t<M, N>(a: Tensor[f32, M, N]) -> Tensor[f32, N, M] { return a; }",
            // An `if`-arm merge of two distinct generic shapes is the same lie in the merge context.
            "fn pick<M, N>(c: bool, a: Tensor[f32, M], b: Tensor[f32, N]) -> f32 \
             { let x = if c { a } else { b }; return x[0]; }",
        ] {
            assert!(
                errors(src).contains(&"E0502"),
                "expected E0502 for {src:?}: {:?}",
                errors(src)
            );
        }
        // No false positives — the patterns real generic kernels actually use stay clean: returning /
        // merging the function's OWN declared shape (identity, multi-dim identity, same-generic merge)
        // and reducing a generic tensor to a scalar (the declared return is not even a tensor).
        for src in [
            "fn id<N>(a: Tensor[f32, N]) -> Tensor[f32, N] { return a; }",
            "fn id2<M, N>(a: Tensor[f32, M, N]) -> Tensor[f32, M, N] { return a; }",
            "fn pick<N>(c: bool, a: Tensor[f32, N], b: Tensor[f32, N]) -> f32 \
             { let x = if c { a } else { b }; return x[0]; }",
            "fn sum<N>(a: Tensor[f32, N]) -> f32 { return a[0]; }",
        ] {
            assert!(
                errors(src).is_empty(),
                "unexpected errors for {src:?}: {:?}",
                errors(src)
            );
        }
    }

    #[test]
    fn generic_param_order_is_deterministic() {
        // A function's generic parameter list must be recorded in SOURCE declaration order. It used
        // to be collected from the `self.generics` HashSet, so `FnSig.generics` came out in
        // per-process hash order — and since a turbofish `f::<2, 3>` binds each argument to the
        // generic in the SAME position (`check_fn_call` zips `sig.generics` with the arguments), the
        // pairing of `<M, N>` with `2, 3` flipped between runs of the *unchanged* source: a call's
        // accept-vs-`E0502` (and its runtime value, down to an out-of-bounds trap) became
        // nondeterministic, a direct violation of the determinism / backend-agreement gates.
        let mut interner = Interner::new();
        let src = "fn f<M, N, K>(a: Tensor[f32, M, N], b: Tensor[f32, N, K]) {}";
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (res, _diags) = check(&module, &interner);
        let def = res
            .defs
            .lookup(interner.intern("f"))
            .expect("fn f registered");
        let names: Vec<&str> = match &def.kind {
            DefKind::Fn(sig) => sig.generics.iter().map(|s| interner.resolve(*s)).collect(),
            _ => panic!("f is not a function"),
        };
        assert_eq!(
            names,
            vec!["M", "N", "K"],
            "generic params must be in declaration order, not hash order"
        );

        // Observable consequence: an order-sensitive turbofish binds by position. `g::<2, 3>` with
        // params `Tensor[M, N]` and `Tensor[N, M]` fed `[2,3]` and `[3,2]` type-checks only if
        // M := 2 and N := 3 (a swapped binding makes the first parameter `Tensor[3, 2]` conflict).
        let ok = "fn g<M, N>(a: Tensor[f32, M, N], b: Tensor[f32, N, M]) {} \
                  fn driver(a: Tensor[f32, 2, 3], b: Tensor[f32, 3, 2]) { g::<2, 3>(a, b); }";
        assert!(
            !errors(ok).contains(&"E0502"),
            "an order-sensitive turbofish must bind by position: {:?}",
            errors(ok)
        );
    }

    #[test]
    fn rigid_shape_mismatch_names_the_axis() {
        // A body-context (RIGID) shape mismatch must say WHICH axis disagrees. `grow` returns
        // `Tensor[f32, M, N]` as `Tensor[f32, N, 5]`: axis 0 is `N` against `M`, axis 1 is the bare
        // constant `5` against `N`. Reporting only "expected N, found M" left the reader counting
        // axes across two shapes.
        let src = "fn grow<M, N>(a: Tensor[f32, M, N]) -> Tensor[f32, N, 5] { return a; }";
        assert_message_has(src, "E0502", &["`N` (axis 0)", "expected N, found M"]);
        assert_message_has(src, "E0502", &["axis 1", "expected 5, found N"]);
    }

    #[test]
    fn inferred_dim_conflict_names_the_declared_dim() {
        // The flagship message (master plan §7.1 `wrong_k.wk`): `a` binds `K := 512`, `b`'s first
        // axis is 513. `apply_subst` has already turned the parameter into `Tensor[f32, 512, N]` by
        // then, so the name `K` can only come from the DECLARED signature — that is what
        // `unify_decl` carries.
        let src = format!(
            "{MATMUL}\nfn driver(a: Tensor[f32, 512, 512], b: Tensor[f32, 513, 512], \
             c: Tensor[f32, 512, 512]) {{ matmul(a, b, c); }}"
        );
        assert_message_has(&src, "E0502", &["`K` (axis 0)", "expected 512, found 513"]);
    }

    #[test]
    fn turbofished_dim_conflict_names_the_declared_dim() {
        // Same, with the dims pinned by a turbofish rather than inferred: `c`'s axis 1 must be `N`.
        let src = format!(
            "{MATMUL}\nfn driver(a: Tensor[f32, 512, 512], b: Tensor[f32, 512, 512], \
             c: Tensor[f32, 512, 513]) {{ matmul::<512, 512, 512>(a, b, c); }}"
        );
        assert_message_has(&src, "E0502", &["`N` (axis 1)", "expected 512, found 513"]);
    }

    #[test]
    fn constant_dim_mismatch_at_a_call_names_the_axis() {
        // A fully concrete call: the callee's axis 1 is 8, the argument's is 9. Both numbers were
        // already in the message; the axis was not.
        let src = "fn takes(a: Tensor[f32, 4, 8]) {} \
                   fn f(a: Tensor[f32, 4, 9]) { takes(a); }";
        assert_message_has(src, "E0502", &["axis 1", "expected 8, found 9"]);
    }

    #[test]
    fn generic_dim_against_a_fixed_size_names_the_axis() {
        // Forwarding a caller's own universally-quantified dim into a fixed-size parameter: the
        // message now says which axis carries the unprovable claim, alongside the size and the
        // generic's name.
        let src = "fn takes64(a: Tensor[f32, 64]) -> f32 { return a[63]; } \
                   fn fwd<N>(a: Tensor[f32, N]) -> f32 { return takes64(a); }";
        assert_message_has(
            src,
            "E0502",
            &["axis 0", "expected 64", "the generic dimension `N`"],
        );
    }

    #[test]
    fn already_bound_dim_conflict_names_the_dim() {
        // The one call-site path where a callee dim is still a VARIABLE when it is compared: a
        // turbofish binds `N` to the caller's own generic of the same name, so the second argument
        // meets `Var(N)` rather than a substituted constant.
        let src = "fn takes<N>(a: Tensor[f32, N], b: Tensor[f32, N]) {} \
                   fn fwd<N>(a: Tensor[f32, N], b: Tensor[f32, 4]) { takes::<N>(a, b); }";
        assert_message_has(src, "E0502", &["`N` (axis 0)", "expected N, found 4"]);
    }

    #[test]
    fn array_decay_length_error_names_the_shape() {
        // Array -> tensor decay has no single disagreeing axis (an array has one length against a
        // whole shape), so this one names the shape and the element count, honestly.
        let exact = "fn take(a: Tensor[f32, 2, 4]) {} \
                     fn f() { let b: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; take(b); }";
        assert_message_has(
            exact,
            "E0501",
            &["array of length 6", "shape [2, 4]", "8 elements"],
        );
        // With a free dim there is no exact count, only divisibility by the fixed axes.
        let free = "fn take<N>(a: Tensor[f32, N, 4]) {} \
                    fn f() { let b: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; take(b); }";
        assert_message_has(
            free,
            "E0501",
            &["array of length 6", "shape [N, 4]", "fixed axes 4"],
        );
    }

    #[test]
    fn out_of_bounds_tensor_index_names_the_axis() {
        // With one message per axis, `a[0, 5]` on a `Tensor[f32, 2, 2]` has to say WHICH index is
        // the offending one — both are spelled `0`/`5` in source but only axis 1 is out of range.
        let src = "fn f(a: Tensor[f32, 2, 2]) { let x = a[0, 5]; }";
        assert_message_has(src, "E0501", &["index 5", "axis 1", "length 2"]);
    }

    #[test]
    fn rank_mismatch_names_both_shapes() {
        // A rank mismatch has no single disagreeing dim to name, so it names rank against rank —
        // and now the two shapes as well, so the reader sees which signature is rank-2.
        let src = "fn use2d<M, N>(a: Tensor[f32, M, N]) {} \
                   fn f(a: Tensor[f32, 4, 4, 4]) { use2d(a); }";
        assert_message_has(
            src,
            "E0501",
            &["expected rank 2 [M, N]", "found rank 3 [4, 4, 4]"],
        );
    }

    #[test]
    fn tensor_index_rank_mismatch_errors() {
        let src = "fn f(a: Tensor[f32, 4, 4]) { let x = a[0]; }";
        assert!(
            errors(src).contains(&"E0501"),
            "expected a rank mismatch: {:?}",
            errors(src)
        );
        // An index-count mismatch names how many indices were written against the tensor's rank —
        // and the shape, so the reader knows what the missing index should range over.
        assert_message_has(src, "E0501", &["rank 2 [4, 4]", "indexed with 1 index"]);
    }

    #[test]
    fn tensor_index_correct_rank_ok() {
        let src = "fn f(a: Tensor[f32, 4, 4]) { let x = a[0, 0]; }";
        assert!(errors(src).is_empty(), "unexpected: {:?}", errors(src));
    }

    #[test]
    fn const_length_decodes_radix_and_separators() {
        // §5: const-array-length folding is mirrored with mir_build's `const_usize_depth`, which
        // decodes the literal with the full radix/`_`/suffix-aware `parse_int`. Decoding only the
        // leading run of decimal digits here made `[i32; 0x10]` length 0 and `[i32; 1_6]` length 1,
        // desyncing the bounds check from the emitted slot size.
        for len in ["0x10", "0o20", "0b10000", "1_6", "16usize"] {
            let ok = format!("fn f() {{ let mut a: [i32; {len}] = [0; {len}]; a[15] = 7; }}");
            assert!(
                errors(&ok).is_empty(),
                "`[i32; {len}]` must be 16 elements: {:?}",
                errors(&ok)
            );
            // …and still exactly 16, not "anything goes".
            let bad = format!("fn f() {{ let mut a: [i32; {len}] = [0; {len}]; a[16] = 7; }}");
            assert!(
                errors(&bad).contains(&"E0501"),
                "`[i32; {len}]` must still reject index 16: {:?}",
                errors(&bad)
            );
        }
        // Via a `const`, and as a tensor dimension (both route through `eval_usize_depth`).
        let via_const = "const N: usize = 1_6; fn f() { let mut a: [i32; N] = [0; N]; a[15] = 7; }";
        assert!(errors(via_const).is_empty(), "{:?}", errors(via_const));
        let dim = "const N: usize = 0x10; fn f(a: Tensor[f32, N]) { let x = a[15]; }";
        assert!(errors(dim).is_empty(), "{:?}", errors(dim));
    }

    #[test]
    fn turbofish_dim_decodes_radix_and_separators() {
        // The turbofish dim must decode like mir_build's `turbofish_dim_value` (the hidden
        // symbolic-dim ABI): `::<3_0>` bound 3 in sema while codegen passed 30, so sema shape-checked
        // one dimension and the callee addressed with another.
        let bad = "fn get2<M, N>(x: Tensor[f32, M, N], i: i32) -> f32 { return x[i, 0]; } \
                   fn f(a: Tensor[f32, 6]) { let v = get2::<2, 3_0>(a, 1); }";
        assert!(
            errors(bad).contains(&"E0501") || errors(bad).contains(&"E0502"),
            "a 6-element argument cannot satisfy `::<2, 3_0>` (= 2x30): {:?}",
            errors(bad)
        );
        let ok = "fn get1<N>(x: Tensor[f32, N], i: i32) -> f32 { return x[i]; } \
                  fn f() { let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; let v = get1::<0x4>(a, 0); }";
        assert!(
            errors(ok).is_empty(),
            "`::<0x4>` must bind N := 4: {:?}",
            errors(ok)
        );
    }

    #[test]
    fn unknown_array_length_name_is_reported() {
        // `[i32; NOPE]` typed the parameter `[i32; 0]` and compiled clean, so every compile-time
        // bounds check on it was vacuous. Tensor dims already had this check (E0504).
        let bad = "fn sum(a: [i32; NOPE]) -> i32 { return a[0]; }";
        assert!(
            errors(bad).contains(&"E0301"),
            "expected an unknown array length: {:?}",
            errors(bad)
        );
        // A `const` (declared BEFORE or AFTER the use) and a declared generic both resolve.
        let after = "fn sum(a: [i32; N]) -> i32 { return a[0]; } const N: usize = 4;";
        assert!(errors(after).is_empty(), "unexpected: {:?}", errors(after));
        // A declared generic is consulted first, exactly as `lower_dim` does, so it is not reported
        // as an unknown name. (Such a length still evaluates to 0 — pre-existing behaviour, which is
        // why the E0501 below fires — but that is not this check's business.)
        let gen = "fn sum<N>(a: [i32; N]) -> i32 { return a[0]; }";
        assert!(
            !errors(gen).contains(&"E0301"),
            "a declared generic is not an unknown length: {:?}",
            errors(gen)
        );
    }

    #[test]
    fn struct_and_extern_signature_dims_are_checked() {
        // Struct fields, enum payloads and `extern` signatures are lowered once, in `collect`, with
        // `checking_bodies` still false — so an undeclared dim name in one of them silently became a
        // fresh unconstrained `Dim::Var`, the exact hole E0504 exists to close.
        for src in [
            "fn main() -> i32 { return 0; } struct S { t: Tensor[f32, KK] }",
            "fn main() -> i32 { return 0; } enum E { V(Tensor[f32, KK]) }",
            "fn main() -> i32 { return 0; } extern { fn ext(a: Tensor[f32, KK]) -> f32; }",
            "fn main() -> i32 { return 0; } struct S { a: [i32; NOPE] }",
        ] {
            assert!(
                errors(src).iter().any(|c| *c == "E0504" || *c == "E0301"),
                "expected an unknown dimension/length in `{src}`: {:?}",
                errors(src)
            );
        }
        // An extern fn's own generics still bind, and a re-lowered signature must not double-report
        // a diagnostic `collect` already emitted.
        let ok = "extern { fn dot<K>(a: Tensor[f32, K], b: Tensor[f32, K]) -> f32; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        let dup = "struct P { a: i32 } struct Q { t: Tensor[P, 4] }";
        assert_eq!(
            errors(dup),
            vec!["E0302"],
            "the diagnostic re-pass must not double-report"
        );
    }

    #[test]
    fn out_of_range_unicode_escape_is_rejected() {
        // mir_build's decoder saturates and then falls back to `cp as u8`, so `\u{110000}` became a
        // NUL inside the string (the `println` stopped there) and printed 1114112 as a `char`.
        for src in [
            r#"fn f() { println("a\u{110000}b"); }"#,
            r#"fn f() { let c = '\u{110000}'; }"#,
            r#"fn f() { let c = '\u{D800}'; }"#,
        ] {
            assert!(
                errors(src).contains(&"E0401"),
                "expected an invalid Unicode escape: {:?}",
                errors(src)
            );
        }
        // Valid escapes — including the largest scalar value and an escaped backslash followed by a
        // literal `u{...}` — stay accepted.
        for src in [
            r#"fn f() { let c = '\u{1F600}'; }"#,
            r#"fn f() { let c = '\u{10FFFF}'; }"#,
            r#"fn f() { println("a\\u{110000}b"); }"#,
            r#"fn f() { println("tab\there\n"); }"#,
        ] {
            assert!(errors(src).is_empty(), "unexpected: {:?}", errors(src));
        }
    }

    #[test]
    fn wukong_prefixed_function_name_is_reserved() {
        // A user function named after a recognizer-emitted runtime kernel HIJACKED the dispatch: the
        // recognized softmax window was replaced by a call to the user's body, silently.
        let bad = "fn wukong_norm_f32(a: i64, b: i64) { } fn main() -> i32 { return 0; }";
        assert!(
            errors(bad).contains(&"E0300"),
            "expected a reserved-name error: {:?}",
            errors(bad)
        );
        // Only the prefix is reserved, and only for functions.
        let ok = "fn wukongish(a: i64) {} fn my_wukong_norm(a: i64) {}";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
    }

    #[test]
    fn out_of_range_match_pattern_literal_is_rejected() {
        // `pattern_cond` materializes the literal as `ConstInt(v, <scrutinee width>)` with no range
        // check, so it truncated and matched a DIFFERENT value — stealing the arm that covers it.
        let wide = "fn f(x: i32) -> i32 { return match x { 4294967296 => 1, 0 => 2, _ => 0 }; }";
        assert!(
            errors(wide).contains(&"E0401"),
            "expected an out-of-range pattern literal: {:?}",
            errors(wide)
        );
        let narrow = "fn g(x: i8) -> i32 { return match x { 200 => 1, _ => 0 }; }";
        assert!(
            errors(narrow).contains(&"E0401"),
            "expected an out-of-range pattern literal: {:?}",
            errors(narrow)
        );
        // Range-pattern bounds are checked the same way …
        let rng = "fn h(x: i8) -> i32 { return match x { 0..=200 => 1, _ => 0 }; }";
        assert!(
            errors(rng).contains(&"E0401"),
            "expected an out-of-range range bound: {:?}",
            errors(rng)
        );
        // … and every in-range pattern is unaffected, including a negative one.
        let ok = "fn k(x: i8) -> i32 { return match x { -128 => 1, 0..=127 => 2, _ => 0 }; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
    }

    #[test]
    fn unfoldable_enum_discriminant_is_reported() {
        // A discriminant the folder cannot read was silently replaced by the auto-increment value,
        // renumbering the rest of the enum with no diagnostic at all.
        let bad = "const OP_MUL: i32 = 10; enum Op { Add = 1, Mul = OP_MUL, Div }";
        assert!(
            errors(bad).contains(&"E0401"),
            "expected a non-constant discriminant error: {:?}",
            errors(bad)
        );
        // Literal discriminants — including negative, hex and folded arithmetic — still work.
        let ok = "enum E { Neg = -5, Pos = 7, Hex = 0x10, Sum = 2 + 3, Auto }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
    }

    #[test]
    fn array_repeat_count_must_match_the_annotation() {
        // The `ArrayLit` arm and `check_struct_literal` both length-check; only the `let`/`const`
        // repeat form swallowed the mismatch and let mir_build fill the slot to the annotation.
        let short = "fn f() { let a: [i32; 4] = [7; 2]; }";
        assert!(
            errors(short).contains(&"E0401"),
            "expected a length mismatch: {:?}",
            errors(short)
        );
        let long = "fn f() { let a: [i32; 2] = [7; 8]; }";
        assert!(
            errors(long).contains(&"E0401"),
            "expected a length mismatch: {:?}",
            errors(long)
        );
        // Matching counts still adapt — including through a `const` and an enum-variant length.
        let ok =
            "const N: usize = 4; fn f() { let a: [f32; N] = [0.0; 4]; let b: [i8; 2] = [7; 2]; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
    }

    #[test]
    fn slice_and_aggregate_do_not_satisfy_a_tensor_param() {
        // A `[]T` slice is a `(ptr, len)` fat pointer, not a tensor base pointer; the clash had no
        // arm in `unify` and fell to the lenient `_`, so the callee indexed the slice HEADER.
        let sl = "fn get1(a: Tensor[f32, 4], i: i32) -> f32 { return a[i]; } \
                  fn f(s: []f32) -> f32 { return get1(s, 2); }";
        assert!(
            errors(sl).contains(&"E0501"),
            "a slice must not satisfy a tensor parameter: {:?}",
            errors(sl)
        );
        let st = "struct S { a: f32 } \
                  fn get1(a: Tensor[f32, 4], i: i32) -> f32 { return a[i]; } \
                  fn f(s: S) -> f32 { return get1(s, 2); }";
        assert!(
            errors(st).contains(&"E0501"),
            "a struct must not satisfy a tensor parameter: {:?}",
            errors(st)
        );
        // A bare generic TYPE variable is also spelled `Ty::Named` and must stay lenient.
        let gen = "fn take<T>(x: T) {} fn f(a: Tensor[f32, 4]) { take(a); }";
        assert!(errors(gen).is_empty(), "unexpected: {:?}", errors(gen));
        // Array → tensor decay is the documented, intentionally lenient path.
        let arr = "fn get1(a: Tensor[f32, 4], i: i32) -> f32 { return a[i]; } \
                   fn f() -> f32 { let b: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; return get1(b, 2); }";
        assert!(errors(arr).is_empty(), "unexpected: {:?}", errors(arr));
    }

    #[test]
    fn generic_dim_does_not_satisfy_a_const_dim_param() {
        // A caller's universally-quantified `N` is not a proof that the buffer is 64 long. Rigid mode
        // already rejected this; the call-site (inference) path silently accepted it.
        let bad = "fn takes64(a: Tensor[f32, 64]) -> f32 { return a[63]; } \
                   fn fwd<N>(a: Tensor[f32, N]) -> f32 { return takes64(a); }";
        assert!(
            errors(bad).contains(&"E0502"),
            "expected a dimension mismatch: {:?}",
            errors(bad)
        );
        // `?` stays the documented escape hatch, and Var→Var forwarding is unaffected.
        let dyn_ok = "fn takes_any(a: Tensor[f32, ?]) -> f32 { return a[0]; } \
                      fn fwd<N>(a: Tensor[f32, N]) -> f32 { return takes_any(a); }";
        assert!(
            errors(dyn_ok).is_empty(),
            "unexpected: {:?}",
            errors(dyn_ok)
        );
        let var_ok = "fn inner<P>(a: Tensor[f32, P]) -> f32 { return a[0]; } \
                      fn fwd<N>(a: Tensor[f32, N]) -> f32 { return inner(a); }";
        assert!(
            errors(var_ok).is_empty(),
            "unexpected: {:?}",
            errors(var_ok)
        );
    }

    #[test]
    fn tensor_layout_must_match_at_a_call() {
        // Lowering DECLINES a non-contiguous tensor (C0001), but `unify` dropped `layout` with `..`,
        // so routing a `.col_major` value through a contiguous-typed callee silently reinterpreted it
        // row-major.
        let bad = "fn getrm(a: Tensor[f32, 2, 3], i: i32, j: i32) -> f32 { return a[i, j]; } \
                   fn pass(a: Tensor[f32, 2, 3, .col_major], i: i32, j: i32) -> f32 { \
                   return getrm(a, i, j); }";
        assert!(
            errors(bad).contains(&"E0502"),
            "expected a layout mismatch: {:?}",
            errors(bad)
        );
        // A fixed-size array is row-major, so it cannot decay to a column-major tensor.
        let decay = "fn getcm(a: Tensor[f32, 2, 3, .col_major]) -> f32 { return a[0, 0]; } \
                     fn f() -> f32 { let b: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; \
                     return getcm(b); }";
        assert!(
            errors(decay).contains(&"E0502"),
            "expected a layout mismatch on array decay: {:?}",
            errors(decay)
        );
        // Matching layouts (including the default contiguous one) still unify.
        let ok = "fn getcm(a: Tensor[f32, 2, 3, .col_major], i: i32) -> f32 { return a[i, 0]; } \
                  fn pass(a: Tensor[f32, 2, 3, .col_major], i: i32) -> f32 { return getcm(a, i); }";
        assert!(
            !errors(ok).contains(&"E0502"),
            "identical layouts must unify: {:?}",
            errors(ok)
        );
    }

    #[test]
    fn array_len_not_constant() {
        // sema's `eval_usize_depth` ends `_ => 0` and mir_build's mirrored `const_usize_depth` ends
        // `_ => None`, which `mir_ty_of_ann` turns into a bare `MirType::Ptr`. So `[i32; 4 as usize]`
        // typed as `[i32; 0]` here while lowering emitted `alloca ptr` and STORED THROUGH the
        // uninitialized pointer — interp `error: expected a pointer` (exit 1), native printed a
        // value after writing to a garbage stack address. The desync is unreachable once the
        // unreadable length is reported.
        for len in ["4 as usize", "sizeof(i64)", "alignof(i64)", "-4", "f()"] {
            let bad = format!("fn f() -> i32 {{ return 4; }} fn g(a: [i32; {len}]) {{ }}");
            assert!(
                errors(&bad).contains(&"E0406"),
                "`[i32; {len}]` must be rejected as unreadable: {:?}",
                errors(&bad)
            );
        }
        // An undefined name inside ARITHMETIC: the old check only looked at a length that was
        // DIRECTLY a single-segment path, so `[i32; BOGUS + 8]` compiled and RAN (exit 99), writing
        // 8 elements into the caller's buffer with `BOGUS` never resolved.
        let arith = "fn fill(mut a: [i32; BOGUS + 8]) { a[0] = 1; }";
        assert!(
            errors(arith).contains(&"E0301"),
            "an undefined name inside a length must be reported: {:?}",
            errors(arith)
        );
        // A cast inside arithmetic folds to 0 on BOTH sides of the mirror, so `(4 as usize) + 1` is
        // length 1, not 5, and both backends agree on it — gate-blind. Report it too.
        let mixed = "fn f(a: [i32; (4 as usize) + 1]) { }";
        assert!(
            errors(mixed).contains(&"E0406"),
            "an unreadable operand must be reported: {:?}",
            errors(mixed)
        );
        // A `const` whose own initializer is unreadable.
        let bad_const = "const K: usize = 4 as usize; fn f(a: [i32; K]) { }";
        assert!(
            errors(bad_const).contains(&"E0406"),
            "an unreadable const initializer must be reported: {:?}",
            errors(bad_const)
        );
        // Controls — every form the two crates DO fold must stay accepted.
        for ok in [
            "fn f(a: [i32; 4]) { }",
            "fn f(a: [i32; 0x10]) { }",
            "const N: usize = 4; fn f(a: [i32; N]) { }",
            "const A: usize = 2; const B: usize = A; fn f(a: [i32; B]) { }",
            "const A: usize = 2; const B: usize = 3; fn f(a: [i32; A * B]) { }",
            "fn f(a: [i32; 2 + 2]) { }",
            "enum Size { Small = 2, Big = 6 } fn f(a: [i32; Size::Big]) { }",
            "struct S { d: [i32; 4] } fn f(s: S) -> i32 { return s.d[0]; }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "a readable length must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn generic_array_length() {
        // `Ty::Array` carries a plain `u64` length with no representation for a symbolic extent
        // (unlike `Ty::Tensor`'s `Dim::Var`), so `[T; N]` was exempted from the name check and then
        // folded to 0 here while mir_build lowered `MirType::Ptr` — the same two-crate desync.
        let gen = "fn sum<N>(a: [f32; N], n: i64) -> f32 { return a[0]; }";
        assert!(
            errors(gen).contains(&"E0406"),
            "a generic array length must be reported: {:?}",
            errors(gen)
        );
        // It is still not an unresolved NAME — the distinction the older check drew stands.
        assert!(
            !errors(gen).contains(&"E0301"),
            "a declared generic is not an unknown name: {:?}",
            errors(gen)
        );
        // The spelling the diagnostic points at must itself be clean: a symbolic tensor extent is
        // bound per call site and shape-checked, which is the whole point of `Tensor[T, N]`.
        let ok = "fn sum<N>(a: Tensor[f32, N]) -> f32 { return a[0]; }";
        assert!(
            errors(ok).is_empty(),
            "the Tensor spelling must stay clean: {:?}",
            errors(ok)
        );
        // Recovery must NOT cascade to call sites. An unreadable extent is now Ty::Error;
        // treating it as zero would report a false array-length mismatch against the caller.
        let call = "fn sum<N>(a: [f32; N], n: i64) -> i64 { return n; } \
                    fn main() -> i32 { let x: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
                    return sum(x, 4) as i32; }";
        assert!(
            errors(call).contains(&"E0406"),
            "the generic length must still be reported at the declaration: {:?}",
            errors(call)
        );
        assert!(
            !errors(call).contains(&"E0501"),
            "an unreadable extent must not cascade into a call-site length mismatch: {:?}",
            errors(call)
        );
        // A generic ELEMENT type with a literal length is unaffected (tests/run/generic_typed_local.wk).
        let elem = "fn f<T>(x: T) -> T { let a: [T; 2] = [x, x]; return a[0] + a[1]; }";
        assert!(
            errors(elem).is_empty(),
            "a generic element type with a literal length must stay clean: {:?}",
            errors(elem)
        );
    }

    #[test]
    fn zero_extent_call_mismatch() {
        let src = "fn get(a: [i32; 1]) -> i32 { return a[0]; } \
                   fn bad(a: [i32; 0]) -> i32 { return get(a); }";
        assert_message_has(src, "E0501", &["expected an array of length 1", "length 0"]);
    }

    #[test]
    fn zero_extent_return_mismatch() {
        let src = "fn bad(a: [i32; 0]) -> [i32; 1] { return a; }";
        assert!(errors(src).contains(&"E0401"), "{:?}", analyze(src).0);
    }

    #[test]
    fn zero_extent_assignment_mismatch() {
        let src = "fn bad(mut a: [i32; 1], b: [i32; 0]) { a = b; }";
        assert!(errors(src).contains(&"E0401"), "{:?}", analyze(src).0);
    }

    #[test]
    fn zero_extent_nested_calls_and_merges() {
        for src in [
            "fn get(a: ([i32; 1], i32)) {} fn bad(a: ([i32; 0], i32)) { get(a); }",
            "fn get(a: [[i32; 1]; 2]) {} fn bad(a: [[i32; 0]; 2]) { get(a); }",
            "fn get(a: [i32; 0]) {} fn bad(a: [i32; 1]) { get(a); }",
        ] {
            assert!(
                errors(src).contains(&"E0501"),
                "{src}: {:?}",
                analyze(src).0
            );
        }
        for src in [
            "fn f(k: i32, a: [i32; 1], b: [i32; 0]) { let r = if k { a } else { b }; }",
            "fn f(k: i32, a: [i32; 0], b: [i32; 1]) { let r = match k { 0 => a, _ => b }; }",
            "fn f(k: i32, a: [i32; 1], b: [i32; 0]) { let r = loop { if k { break a; } break b; }; }",
            "fn f(a: ([i32; 0], i32)) -> ([i32; 1], i32) { return a; }",
            "fn f() -> [i32; 1] { return []; }",
            "fn f(mut a: [i32; 1]) { a = []; }",
            "fn f(a: [i32; 1]) -> [i32; 0] { return a; }",
            "fn f(a: ([i32; 0], i32)) -> ([f32; 0], i32) { return a; }",
            "fn f(a: [[i32; 0]; 1]) -> [[]i32; 1] { return a; }",
            "fn f<T>(a: [T; 0]) -> [T; 1] { return a; }",
            "fn f<T>(mut a: [T; 1], b: [T; 0]) { a = b; }",
            "fn f<T>(k: i32, a: [T; 1], b: [T; 0]) { let r = if k { a } else { b }; }",
        ] {
            assert!(errors(src).contains(&"E0401"), "{src}: {:?}", analyze(src).0);
        }
    }

    #[test]
    fn zero_extent_valid_empty_values_and_forward_extents() {
        for src in [
            "fn id(a: [i32; 0]) -> [i32; 0] { return a; } \
             fn f() { let mut a: [i32; 0] = []; a = [0; 0]; let b = id(a); let t = (); }",
            "fn f(k: i32, a: [i32; 0]) -> [i32; 0] { return if k { a } else { [] }; }",
            "fn f(k: i32, a: [i32; 0]) -> [i32; 0] { return if k { [] } else { a }; }",
            "fn f(k: i32, a: [i32; 0]) { let r = match k { 0 => [], _ => a }; }",
            "fn f(k: i32, a: [i32; 0]) { let r = loop { if k { break []; } break a; }; }",
            "fn f() -> [i32; 0] { let a = []; return a; }",
            "fn f() -> ([i32; 0], i32) { let t = ([], 1); return t; }",
            "fn f() -> [[i32; 0]; 2] { let a = [[], []]; return a; }",
            "fn id(t: ([i32; 0], i32)) -> ([i32; 0], i32) { return t; }",
            "fn id(t: ((), i32)) -> ((), i32) { return t; }",
            "fn get(a: [i32; N]) -> i32 { return a[0]; } \
             fn f(a: [i32; 1]) -> i32 { return get(a); } const N: usize = 1;",
            "fn id(a: [i32; N]) -> [i32; N] { return a; } \
             fn f(a: [i32; 0]) -> [i32; 0] { return id(a); } const N: usize = 0;",
            "fn get(a: [i32; Size::N]) {} fn f(a: [i32; 1]) { get(a); } enum Size { N = 1 }",
        ] {
            assert!(errors(src).is_empty(), "{src}: {:?}", analyze(src).0);
        }
    }

    #[test]
    fn zero_extent_forward_declarations_cannot_hide_mismatches() {
        for src in [
            "fn get(a: [i32; N]) {} fn bad(a: [i32; 0]) { get(a); } const N: usize = 1;",
            "extern { fn get(a: [i32; N]); } fn bad(a: [i32; 0]) { get(a); } const N: usize = 1;",
            "struct S { a: [i32; N] } fn get(a: [i32; 1]) {} \
             fn bad(s: S) { get(s.a); } const N: usize = 0;",
            "fn get(a: [i32; Size::N]) {} fn bad(a: [i32; 0]) { get(a); } enum Size { N = 1 }",
        ] {
            assert!(
                errors(src).contains(&"E0501"),
                "{src}: {:?}",
                analyze(src).0
            );
        }
    }

    #[test]
    fn zero_extent_recovery_remains_distinct_from_empty_arrays() {
        for (src, code) in [
            (
                "fn get<N>(a: [i32; N]) {} fn f(a: [i32; 1]) { get(a); }",
                "E0406",
            ),
            ("const N: usize = 0; fn get<N>(a: [i32; N]) {}", "E0406"),
            (
                "fn get(a: [i32; MISSING]) {} fn f(a: [i32; 1]) { get(a); }",
                "E0301",
            ),
            (
                "fn get(a: [i32; 4 as usize]) {} fn f(a: [i32; 1]) { get(a); }",
                "E0406",
            ),
            ("fn f(n: usize) -> [i32; 0] { return [1; n]; }", "E0406"),
        ] {
            assert_eq!(errors(src), vec![code], "{src}: {:?}", analyze(src).0);
        }
    }

    #[test]
    fn zero_extent_refresh_records_real_types_in_every_declaration() {
        let src = "fn id(a: [i32; N]) -> [i32; N] { return a; } \
                   struct S { a: [i32; N] } \
                   enum E { A([i32; N]), B { a: [i32; N] } } \
                   extern { fn ext(a: [i32; N]); } \
                   const A: [i32; N] = [1, 2]; const N: usize = 2;";
        let mut interner = Interner::new();
        let (module, parse_diags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse_diags.is_empty(), "{parse_diags:?}");
        let (result, diags) = check(&module, &interner);
        assert!(diags.is_empty(), "{diags:?}");
        let expected = Ty::Array {
            elem: Box::new(Ty::Scalar(Scalar::I32)),
            len: 2,
            shared: false,
        };
        for def in &result.defs.defs {
            match &def.kind {
                DefKind::Fn(sig) => {
                    assert_eq!(sig.params, vec![expected.clone()]);
                    if interner.resolve(def.name) == "id" {
                        assert_eq!(sig.ret, expected);
                    }
                }
                DefKind::Struct(fields) => assert_eq!(fields[0].1, expected),
                DefKind::Enum(variants) => {
                    for variant in variants {
                        assert_eq!(variant.payload.field_tys(), vec![expected.clone()]);
                    }
                }
                DefKind::Const(ty) if interner.resolve(def.name) == "A" => {
                    assert_eq!(*ty, expected)
                }
                _ => {}
            }
        }
    }

    #[test]
    fn array_arg_length_and_element_are_checked() {
        // `unify_decl` had no (Array, Array) arm at all, so an array argument for an array parameter
        // fell to the lenient `_ => {}`: `fn f(a: [i32; 8])` fed a `[i32; 2]` compiled clean, interp
        // trapped `load out of bounds` and native printed 32758 after reading past the caller's
        // stack buffer — an interp-vs-native divergence AND a memory-safety violation.
        let short = "fn f(a: [i32; 8]) -> i32 { return a[7]; } \
                     fn g() -> i32 { let x: [i32; 2] = [1, 2]; return f(x); }";
        assert!(
            errors(short).contains(&"E0501"),
            "a short array argument must be rejected: {:?}",
            errors(short)
        );
        // The element recursion: `[i32; 4]` for `[f32; 4]` read the i32 bit pattern as an f32
        // (interp 3, native 4.203895e-45).
        let elem = "fn f(a: [f32; 4]) -> f32 { return a[0]; } \
                    fn g() -> f32 { let x: [i32; 4] = [1, 2, 3, 4]; return f(x); }";
        assert!(
            errors(elem).contains(&"E0401"),
            "a wrong element type must be rejected: {:?}",
            errors(elem)
        );
        // Nested: the outer lengths agree, the inner ones do not.
        let nested = "fn f(a: [[i32; 3]; 2]) -> i32 { return a[0][2]; } \
                      fn g() -> i32 { let x: [[i32; 2]; 2] = [[1, 2], [3, 4]]; return f(x); }";
        assert!(
            errors(nested).contains(&"E0501"),
            "a nested length mismatch must be rejected: {:?}",
            errors(nested)
        );
        // A slice argument for an ARRAY parameter handed the callee the fat-pointer header as data;
        // both backends printed 0, so no differential gate could see it.
        let fat = "fn f(a: [i32; 4]) -> i32 { return a[0]; } \
                   fn g(s: []i32) -> i32 { return f(s); }";
        assert!(
            errors(fat).contains(&"E0401"),
            "a slice argument for an array parameter must be rejected: {:?}",
            errors(fat)
        );
        // Controls: the exact match, array->slice unsizing, and array->tensor decay must all stay
        // accepted (tests/run/tensor_add.wk feeds a `[f32; 6]` to a `Tensor[f32, 2, 3]`).
        for ok in [
            "fn f(a: [i32; 4]) -> i32 { return a[3]; } \
             fn g() -> i32 { let x: [i32; 4] = [1, 2, 3, 4]; return f(x); }",
            "fn f(a: []i32) -> i32 { return a[0]; } \
             fn g() -> i32 { let x: [i32; 4] = [1, 2, 3, 4]; return f(x); }",
            "fn f(a: Tensor[f32, 2, 3]) -> f32 { return a[1, 2]; } \
             fn g() -> f32 { let x: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; return f(x); }",
            "fn f(mut a: [[i32; 2]; 2]) -> i32 { return a[1][1]; } \
             fn g() -> i32 { let mut x: [[i32; 2]; 2] = [[1, 2], [3, 4]]; return f(x); }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "a well-typed array argument must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn struct_arg_wrong_nominal() {
        // No (Named, Named) arm: `struct A` satisfied a `B` parameter, the callee GEPed the
        // caller's differently-laid-out buffer, interp trapped `load out of bounds` and native
        // printed 0.
        let bad = "struct A { x: i32 } struct B { y: f32, z: f32 } \
                   fn f(v: B) -> f32 { return v.z; } \
                   fn g() -> f32 { let a: A = A { x: 5 }; return f(a); }";
        assert!(
            errors(bad).contains(&"E0401"),
            "a different struct must not satisfy a nominal parameter: {:?}",
            errors(bad)
        );
        // A declared nominal against a structural kind is the same confusion with one side spelled
        // `Named`.
        let kind = "struct A { x: i32 } \
                    fn f(v: A) -> i32 { return v.x; } \
                    fn g() -> i32 { return f(5); }";
        assert!(
            errors(kind).contains(&"E0401"),
            "a scalar must not satisfy a struct parameter: {:?}",
            errors(kind)
        );
        // Controls: the right struct, and a bare generic TYPE variable, which is ALSO spelled
        // `Ty::Named` and must stay lenient (that is what `is_declared_named_ty` filters).
        for ok in [
            "struct B { y: f32, z: f32 } \
             fn f(v: B) -> f32 { return v.z; } \
             fn g() -> f32 { let b: B = B { y: 1.0, z: 2.0 }; return f(b); }",
            "fn id<T>(x: T) -> T { return x; } fn g() -> i32 { return id(5); }",
            "struct A { x: i32 } fn id<T>(x: T) -> T { return x; } \
             fn g() -> i32 { let a: A = A { x: 5 }; let b: A = id(a); return b.x; }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "a well-typed nominal argument must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn tuple_arg_arity_and_elements_are_checked() {
        // No (Tuple, Tuple) arm: `fn f(t: (i32, i32, i32))` fed a `(i32, i32)` compiled clean and
        // `t.2` read past the caller's buffer (interp trapped, native printed -1037373571).
        let arity = "fn f(t: (i32, i32, i32)) -> i32 { return t.2; } \
                     fn g() -> i32 { let p: (i32, i32) = (1, 2); return f(p); }";
        assert!(
            errors(arity).contains(&"E0501"),
            "a tuple of the wrong arity must be rejected: {:?}",
            errors(arity)
        );
        let elem = "fn f(t: (i32, f32)) -> f32 { return t.1; } \
                    fn g() -> f32 { let p: (i32, i32) = (1, 2); return f(p); }";
        assert!(
            errors(elem).contains(&"E0401"),
            "a tuple element mismatch must be rejected: {:?}",
            errors(elem)
        );
        let ok = "fn f(t: (i32, i32)) -> i32 { return t.1; } \
                  fn g() -> i32 { let p: (i32, i32) = (1, 2); return f(p); }";
        assert!(
            errors(ok).is_empty(),
            "a matching tuple argument must stay accepted: {:?}",
            errors(ok)
        );
    }

    #[test]
    fn array_return_length_lie() {
        // `check_return_shape` ran its `unify` only when one side was a Tensor/Vector, so a callee
        // could declare `-> [i32; 8]` and return a `[i32; 2]`; every caller then believed the lie
        // and `r[7]` was interp-trap / native-0.
        let arr = "fn g() -> [i32; 8] { let a: [i32; 2] = [5, 6]; return a; }";
        assert!(
            errors(arr).contains(&"E0401"),
            "a function must not lie about its array return length: {:?}",
            errors(arr)
        );
        let tup = "fn g() -> (i32, i32) { let t: (i32, i32, i32) = (1, 2, 3); return t; }";
        assert!(
            errors(tup).contains(&"E0401"),
            "a function must not lie about its tuple arity: {:?}",
            errors(tup)
        );
        let nominal = "struct A { x: i32 } struct B { y: i32 } \
                       fn g() -> B { let a: A = A { x: 1 }; return a; }";
        assert!(
            errors(nominal).contains(&"E0401"),
            "a function must not lie about its struct identity: {:?}",
            errors(nominal)
        );
        for ok in [
            "fn g() -> [i32; 2] { let a: [i32; 2] = [5, 6]; return a; }",
            "struct A { x: i32 } fn g() -> A { let a: A = A { x: 1 }; return a; }",
            "fn g<T>(x: T) -> T { return x; }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "an honest return must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn assign_struct_type_mismatch() {
        // `check_binop_shapes` opens with `if matches!((l, r), (Tensor, Tensor) | (Vector, Vector))`
        // and is a no-op otherwise, so `a = b` between two unrelated structs reinterpreted the
        // bytes: interp printed 1 (the i32 field), native 1065353216 (the bits of 1.0f32).
        let structs = "struct A { x: i32 } struct B { y: f32, z: f32 } \
                       fn f() { let mut a: A = A { x: 1 }; let b: B = B { y: 1.0, z: 2.0 }; a = b; }";
        assert!(
            errors(structs).contains(&"E0401"),
            "assigning a different struct must be rejected: {:?}",
            errors(structs)
        );
        let arrays =
            "fn f() { let mut a: [i32; 2] = [1, 2]; let b: [i32; 4] = [1, 2, 3, 4]; a = b; }";
        assert!(
            errors(arrays).contains(&"E0401"),
            "assigning a different array length must be rejected: {:?}",
            errors(arrays)
        );
        // The same hole in the other three merge contexts `check_binop_shapes` guards.
        let arm = "struct A { x: i32 } struct B { y: f32 } \
                   fn f(k: i32) { let a: A = A { x: 1 }; let b: B = B { y: 1.0 }; \
                                  let r: A = match k { 0 => a, _ => b }; }";
        assert!(
            errors(arm).contains(&"E0401"),
            "a match-arm merge of two structs must be rejected: {:?}",
            errors(arm)
        );
        let brk = "struct A { x: i32 } struct B { y: f32 } \
                   fn f(k: i32) { let a: A = A { x: 1 }; let b: B = B { y: 1.0 }; \
                                  let r: A = loop { if k == 0 { break a; } break b; }; }";
        assert!(
            errors(brk).contains(&"E0401"),
            "a break merge of two structs must be rejected: {:?}",
            errors(brk)
        );
        for ok in [
            "struct A { x: i32 } fn f() { let mut a: A = A { x: 1 }; let b: A = A { x: 2 }; a = b; }",
            "fn f() { let mut a: [i32; 2] = [1, 2]; let b: [i32; 2] = [3, 4]; a = b; }",
            "struct A { x: i32 } fn f(k: i32) -> i32 { let a: A = A { x: 1 }; let b: A = A { x: 2 }; \
             let r: A = match k { 0 => a, _ => b }; return r.x; }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "a same-type assignment/merge must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn aggregate_literal_adapts_at_a_return_an_assignment_and_a_merge() {
        // OVER-REJECTION control for `aggregate_type_clash`. `literal_adapts`'s ArrayLit/TupleLit
        // arms exist so a typed buffer can be built from bare literals — `let a: [i8; 2] = [127, 0]`
        // is DELIBERATELY accepted — but an unsuffixed integer literal's own type is i32 and a float
        // literal's is f32, so the literal's type is `[i32; 2]`. Comparing types alone made the same
        // spelling a hard E0401 at a `return`, at an assignment and at an arm merge: previously
        // valid programs rejected by a check whose job is to catch layout LIES.
        for ok in [
            "fn g() -> [i8; 2] { return [127, 0]; }",
            "fn g() -> [f64; 2] { return [1.0, 2.0]; }",
            "fn g() -> (u8, u8) { return (200, 1); }",
            "fn g() -> [u8; 3] { return [1; 3]; }",
            "fn f() { let mut a: [i8; 2] = [1, 2]; a = [127, 0]; }",
            "fn f() { let mut t: (u8, u8) = (1, 2); t = (200, 1); }",
            "fn f(k: i32) { let a: [i8; 2] = [1, 2]; let r: [i8; 2] = match k { 0 => a, _ => [127, 0] }; }",
            "fn f(k: i32) { let a: [i8; 2] = [1, 2]; let r: [i8; 2] = if k == 0 { a } else { [127, 0] }; }",
            "fn f(k: i32) { let a: [i8; 2] = [1, 2]; \
             let r: [i8; 2] = loop { if k == 0 { break a; } break [127, 0]; }; }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "an adapting aggregate literal must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
        // …and the lie the same check exists to catch must still be caught, with a literal on the
        // value side: the LENGTH and the ARITY never adapt, only the element type does.
        for bad in [
            "fn g() -> [i8; 2] { return [1, 2, 3]; }",
            // And the asymmetry is deliberate: `join` returns the LEFT type for an aggregate, so a
            // literal in the FIRST arm establishes `[i32; 2]` and the `[i8; 2]` arm genuinely
            // cannot hold it. The literal adapts to the type already established, exactly as it
            // adapts to a `let` annotation — it does not retroactively set one.
            "fn f(k: i32, a: [i8; 2]) { let r: [i32; 2] = if k == 0 { [127, 0] } else { a }; }",
            "fn g() -> (u8, u8) { return (1, 2, 3); }",
            "fn f() { let mut a: [i8; 2] = [1, 2]; a = [1, 2, 3]; }",
        ] {
            assert!(
                errors(bad).contains(&"E0401"),
                "a wrong-length literal must still be rejected (`{bad}`): {:?}",
                errors(bad)
            );
        }
    }

    /// Every type this module RECORDED, as display strings. `mir_build` materializes a numeric
    /// literal as a constant of its recorded type (`lower_expr`'s `Int`/`Float` arms read
    /// `expr_mir`), so this is the WIDTH half of an adaptation — the half no "does it compile"
    /// assertion can see, and the half that was missing at a `return` and at a merge.
    fn recorded_tys(src: &str) -> Vec<String> {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (res, diags) = check(&module, &interner);
        assert!(
            !diags.iter().any(|d| d.is_error()),
            "unexpected errors for {src:?}: {diags:?}"
        );
        res.types.values().map(|t| t.display(&interner)).collect()
    }

    #[test]
    fn adapted_literal_is_restamped_at_the_return_and_the_merge_width() {
        // The WIDTH half of the leniency the test above proves is granted. An unsuffixed float
        // literal's own type is f32, so accepting `fn g() -> (f64, f64) { (0.1, 0.2) }` without
        // re-stamping it left an 8-byte `(f32, f32)` written into the caller's 16-byte tuple: the
        // interpreter reported `load out of bounds`, native printed
        // `0.000000013411048210842936` and `NaN`. The scalar form lied quietly instead of loudly —
        // `fn a() -> f64 { 0.1 }` returned `0.10000000149011612`, the f32 0.1 widened, where
        // `let a: f64 = 0.1;` returns `0.1`. Assert on the recorded type, because a compile-only
        // assertion is blind to exactly this.
        for src in [
            "fn g() -> f64 { 0.1 }",
            "fn g() -> f64 { return 0.1; }",
            "fn g() -> [f64; 2] { [0.1, 0.2] }",
            "fn g() -> [f64; 2] { return [0.1, 0.2]; }",
            "fn g() -> (f64, f64) { (0.1, 0.2) }",
            "fn g() -> (f64, f64) { return (0.1, 0.2); }",
            "fn g() -> [f64; 3] { [0.1; 3] }",
            "fn f(k: i32) -> f64 { let a: (f64, f64) = (0.5, 0.25); let r: (f64, f64) = if k == 0 { a } else { (0.1, 0.2) }; return r.0; }",
            "fn f(k: i32) -> f64 { let a: [f64; 2] = [0.5, 0.25]; let r: [f64; 2] = match k { 0 => a, _ => [0.1, 0.2] }; return r[0]; }",
        ] {
            let tys = recorded_tys(src);
            assert!(
                !tys.iter().any(|t| t.contains("f32")),
                "an adapted float literal must be RE-STAMPED at the declared width, not left f32 for lowering to reinterpret or widen (`{src}`): {tys:?}"
            );
        }
        // …and the range check is the rest of that pair. Once a literal is re-stamped it is
        // materialized as a constant OF THE NARROW TYPE rather than an i32 the backend truncates,
        // so an out-of-range one has to be a diagnostic: `ConstInt(300, i8)` passes wukong's
        // verifier (which only checks the type is integral), masks to 44 on the interpreter, and is
        // REJECTED by Cranelift's `iconst_bounds`. Accepting it would be a program the oracle runs
        // and native cannot build.
        for bad in [
            "fn g() -> i8 { 200 }",
            "fn g() -> [i8; 2] { [300, 0] }",
            "fn g() -> (u8, u8) { (300, 1) }",
            "fn f(k: i32, a: [i8; 2]) -> i8 { let r: [i8; 2] = if k == 0 { a } else { [300, 0] }; return r[0]; }",
        ] {
            assert!(
                errors(bad).contains(&"E0401"),
                "an out-of-range literal that ADAPTS must be rejected, not re-stamped into a constant too wide for its own type (`{bad}`): {:?}",
                errors(bad)
            );
        }
    }

    #[test]
    fn slice_does_not_satisfy_an_array_place() {
        // UNDER-REJECTION control. `compatible` encodes `[T; N]` → `[]T` unsizing in ONE direction
        // on purpose ("there is no slice→array direction (a slice has no static length to
        // restore)"), and an earlier `aggregate_type_clash` asked
        // `!(compatible(a, b) || compatible(b, a))`, which threw that intent away exactly where
        // direction is not an accident. `return s` from `-> [f32; 4]` was accepted and the caller
        // then read the 16-byte fat-pointer header as four f32 elements — both backends printing 0,
        // a value the source never wrote. `let r: [f32; 4] = s;` has always been rejected through
        // `let_compatible`; `return` and `=` must not be weaker than `let`.
        let ret = "fn g(s: []f32) -> [f32; 4] { return s; }";
        assert!(
            errors(ret).contains(&"E0401"),
            "a slice must not satisfy an array return type: {:?}",
            errors(ret)
        );
        let asg = "fn f(s: []f32) { let mut a: [f32; 4] = [0.0; 4]; a = s; }";
        assert!(
            errors(asg).contains(&"E0401"),
            "a slice must not be assigned into an array place: {:?}",
            errors(asg)
        );
        // The reference point: `let` rejects the same direction, and always has.
        let lt = "fn f(s: []f32) { let r: [f32; 4] = s; }";
        assert!(
            errors(lt).contains(&"E0401"),
            "the `let` this must match still rejects it: {:?}",
            errors(lt)
        );
        // The LEGAL direction — the one real coercion — must stay accepted at every one of those
        // sites, and a MERGE stays symmetric, because there neither side is the place.
        for ok in [
            "fn g(a: [f32; 4]) -> []f32 { return a; }",
            "fn f(a: [f32; 4]) { let mut s: []f32 = a; s = a; }",
            "fn f(k: i32, a: [f32; 4], s: []f32) { let r: []f32 = match k { 0 => a, _ => s }; }",
            "fn f(k: i32, a: [f32; 4], s: []f32) { let r: []f32 = if k == 0 { s } else { a }; }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "array→slice unsizing must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn array_length_operator_must_fold() {
        // `array_len_is_const` proved the OPERANDS were readable and stopped there, but
        // `fold_const_len` has no value for a comparison or a logical operator, and none for
        // checked arithmetic that fails — and it reports all of them as the same 0 a genuine
        // `[T; 2 - 2]` gives, on BOTH sides of the two-crate mirror. So the array silently took a
        // length the source never wrote, with both backends agreeing on it.
        for len in [
            "2 - 5",
            "4 > 2",
            "1 && 1",
            "4 / 0",
            "4 % 0",
            "4294967295 + 1",
        ] {
            let bad = format!("fn f(a: [i32; {len}]) {{ }}");
            assert!(
                errors(&bad).contains(&"E0406"),
                "`[i32; {len}]` does not fold and must be reported: {:?}",
                errors(&bad)
            );
        }
        // Everything the shared folder really does fold stays accepted, including a real `[T; 0]`.
        for ok in [
            "fn f(a: [i32; 2 + 2]) { }",
            "fn f(a: [i32; 8 - 3]) { }",
            "fn f(a: [i32; 2 * 3]) { }",
            "fn f(a: [i32; 8 / 2]) { }",
            "fn f(a: [i32; 1 << 3]) { }",
            "fn f(a: [i32; 2 - 2]) { }",
            "const A: usize = 2; const B: usize = 3; fn f(a: [i32; A * B]) { }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "a folding length must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn deeply_nested_const_length_is_not_readable() {
        // The depth cutoff is where the two mirrors stop, and they stopped DIFFERENTLY: sema's
        // `eval_usize_depth` yields 0 (type `[T; 0]`) past depth 64 while mir_build's
        // `const_usize_depth` yields `None` (a bare `MirType::Ptr` slot). `array_len_is_const`
        // returned `true` there to avoid stacking a diagnostic on a cyclic const, so a NON-cyclic
        // chain deeper than 64 — which `check_recursive_consts` never reports — kept the exact
        // desync this whole function exists to remove, silently.
        let mut deep = String::from("const C0: usize = 4;");
        for i in 1..=70 {
            deep.push_str(&format!("const C{i}: usize = C{};", i - 1));
        }
        deep.push_str("fn f(a: [i32; C70]) { }");
        assert!(
            errors(&deep).contains(&"E0406"),
            "a const chain deeper than the fold cutoff is not a readable length: {:?}",
            errors(&deep)
        );
        // A chain the folder really does read must stay accepted.
        let mut shallow = String::from("const D0: usize = 4;");
        for i in 1..=10 {
            shallow.push_str(&format!("const D{i}: usize = D{};", i - 1));
        }
        shallow.push_str("fn f(a: [i32; D10]) { }");
        assert!(
            errors(&shallow).is_empty(),
            "a chain within the fold cutoff must stay accepted: {:?}",
            errors(&shallow)
        );
    }

    #[test]
    fn array_index_count() {
        // shape.rs's module header promises "a wrong index count is E0501", but only the tensor arm
        // enforced it: the array arm gated its bounds check on `indices.len() == 1` and the
        // slice/pointer arm had no count check at all, so `a[i, j]` on a `[i32; 4]` came back as the
        // generic `C0001 'expression' is not yet supported by codegen` — the compiler blaming itself
        // for the user's error.
        let arr = "fn f(a: [i32; 4]) -> i32 { return a[1, 2]; }";
        assert!(
            errors(arr).contains(&"E0501"),
            "a rank-2 index on an array must be E0501: {:?}",
            errors(arr)
        );
        let slice = "fn f(s: []i32) -> i32 { return s[1, 2]; }";
        assert!(
            errors(slice).contains(&"E0501"),
            "a rank-2 index on a slice must be E0501: {:?}",
            errors(slice)
        );
        let ptr = "fn f(p: *i32) -> i32 { return p[1, 2]; }";
        assert!(
            errors(ptr).contains(&"E0501"),
            "a rank-2 index on a pointer must be E0501: {:?}",
            errors(ptr)
        );
        for ok in [
            "fn f(a: [i32; 4]) -> i32 { return a[1]; }",
            "fn f(s: []i32) -> i32 { return s[1]; }",
            "fn f(p: *i32) -> i32 { return p[1]; }",
            "fn f(a: Tensor[f32, 2, 2]) -> f32 { return a[1, 1]; }",
        ] {
            assert!(
                errors(ok).is_empty(),
                "a correct index count must stay accepted (`{ok}`): {:?}",
                errors(ok)
            );
        }
    }

    #[test]
    fn duplicate_struct_field() {
        // Duplicate PARAMETER names were already E0300; the same discipline was never applied to
        // struct fields, enum variants or generic lists. `struct S { x: i32, x: i32 }` produced no
        // declaration-site error, `enum E { A, A }` compiled and ran printing 0, and `fn f<N, N>`
        // compiled and ran.
        let field = "struct S { x: i32, x: i32 } fn f(s: S) -> i32 { return s.x; }";
        assert!(
            errors(field).contains(&"E0300"),
            "a duplicate struct field must be reported: {:?}",
            errors(field)
        );
        let variant = "enum E { A, A } fn f() -> i32 { return E::A as i32; }";
        assert!(
            errors(variant).contains(&"E0300"),
            "a duplicate enum variant must be reported: {:?}",
            errors(variant)
        );
        let generic = "fn f<N, N>(a: Tensor[f32, N]) -> f32 { return a[0]; }";
        assert!(
            errors(generic).contains(&"E0300"),
            "a duplicate generic parameter must be reported: {:?}",
            errors(generic)
        );
        let ok = "struct S { x: i32, y: i32 } enum E { A, B } \
                  fn f<M, N>(a: Tensor[f32, M, N]) -> f32 { return a[0, 0]; }";
        assert!(
            errors(ok).is_empty(),
            "distinct declaration names must stay accepted: {:?}",
            errors(ok)
        );
    }

    #[test]
    fn unknown_escape() {
        // mir_build's `decode_escape` ends `Some(c) => c as u32`, so an unknown escape silently
        // became the escaped character: `println("A\dB")` printed `AdB`, a value the source never
        // wrote, with both backends agreeing so no differential gate could see it.
        let unknown = "fn f() { println(\"A\\dB\"); }";
        assert!(
            errors(unknown).contains(&"E0401"),
            "an unknown escape must be reported: {:?}",
            errors(unknown)
        );
        // Worse: the `\u` arm is `skip_while(|&c| c != '{')` over the SHARED iterator, so a `\u`
        // with no brace eats the whole remainder of the literal and folds to 0 —
        // `println("AAF")` printed just `A`.
        let no_brace = "fn f() { println(\"A\\u0041F\"); }";
        assert!(
            errors(no_brace).contains(&"E0401"),
            "a `\\u` with no brace must be reported: {:?}",
            errors(no_brace)
        );
        // And the worse half of the same mechanism: a brace that is OPENED and never CLOSED runs
        // the scan to the end of the literal. `malformed_escape`'s loop had no found-brace flag, so
        // it returned `None`; `bad_unicode_escape`'s `take_while(|&c| c != '}')` ran to the end too
        // and its `map_or(v, ..)` skipped the non-hex characters, folding `\u{41Zoo` to a perfectly
        // valid 0x41 — so BOTH validators stayed silent while `decode_escape` deleted `Zoo`.
        let unclosed = "fn f() { println(\"A\\u{41Zoo\"); }";
        assert!(
            errors(unclosed).contains(&"E0401"),
            "an unterminated `\\u{{` must be reported: {:?}",
            errors(unclosed)
        );
        let lone = "fn f() -> char { return '\\d'; }";
        assert!(
            errors(lone).contains(&"E0401"),
            "an unknown escape in a char literal must be reported: {:?}",
            errors(lone)
        );
        // Every escape the decoder really reads must stay accepted, `\\` included (the scan has to
        // consume the escaped character or a literal backslash starts a phantom escape).
        let ok = "fn f() { println(\"a\\tb\\nc\\\\d\\\"e\\u{41}\\x41\\0\\r\"); }";
        assert!(
            errors(ok).is_empty(),
            "the decodable escapes must stay accepted: {:?}",
            errors(ok)
        );
        // The pre-existing out-of-range check must still fire, and must not be masked by the new one.
        let surrogate = "fn f() { println(\"A\\u{D800}B\"); }";
        assert!(
            errors(surrogate).contains(&"E0401"),
            "a surrogate code point must still be reported: {:?}",
            errors(surrogate)
        );
    }

    #[test]
    fn array_length_above_u32_max_is_rejected() {
        // MIR array extents are u32. Historically, unchecked narrowing made a longer length wrap
        // to a small slot while sema bounds-checked the full value. Keep the explicit diagnostic
        // even though lowering now checks its final conversion too.
        let bad = "fn f() { let mut a: [i32; 0x1_0000_0001] = [0; 0x1_0000_0001]; a[0] = 1; }";
        assert!(
            errors(bad).contains(&"E0401"),
            "expected an out-of-range array length: {:?}",
            errors(bad)
        );
        // The largest representable length is still accepted (no off-by-one at the boundary).
        let ok = "fn f(a: [i32; 4294967295]) -> i32 { return a[0]; }";
        assert!(
            !errors(ok).contains(&"E0401"),
            "u32::MAX must remain a legal length: {:?}",
            errors(ok)
        );
    }
}
