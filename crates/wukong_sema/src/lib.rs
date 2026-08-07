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
//! top-level definition with its signature lowered) → `check_recursive_types` →
//! `check_recursive_consts` → set `checking_bodies` → `recheck_item_signatures` → `check_bodies`.
//! Several diagnostics are gated on the `checking_bodies` flag so they fire only once every generic
//! and `const` is registered.

mod shape;

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
        checking_bodies: false,
    };
    s.collect(module);
    s.check_recursive_types(module);
    s.check_recursive_consts(module);
    // The undeclared-tensor-dimension check (`lower_dim`) fires only from here on: by the body pass
    // `collect` has registered every top-level `const` and this item's generics are re-established
    // per function, so an unknown dim name can be told from a declared generic / a `const` with no
    // forward-reference false positive (and a function signature, re-lowered here, is reported once).
    s.checking_bodies = true;
    s.recheck_item_signatures(module);
    s.check_bodies(module);
    let result = SemaResult {
        types: s.types,
        defs: s.defs,
        consts: s.consts,
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
    /// Top-level `const` initializer expressions (by name), accumulated as their bodies are checked.
    consts: HashMap<Symbol, Expr>,
    /// `false` during the collection pass, `true` once body checking starts. Gates the
    /// undeclared-tensor-dimension diagnostic (`lower_dim`): reporting it only in the body pass means
    /// every top-level `const` is registered and each item's generics are re-established, so an
    /// unknown dim name is distinguished from a declared generic / a `const` with no forward-reference
    /// false positive, and a function signature (re-lowered in `check_fn`) is reported exactly once.
    checking_bodies: bool,
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

impl Sema<'_> {
    fn error(&mut self, span: Span, code: &'static str, msg: impl Into<String>) {
        self.diags
            .push(Diagnostic::error(msg).with_code(code).primary(span, ""));
    }

    fn sym_str(&self, s: Symbol) -> &str {
        self.interner.resolve(s)
    }

    // ---- Collection pass: register all top-level defs with lowered signatures ----

    fn collect(&mut self, module: &Module) {
        for item in &module.items {
            match &item.kind {
                ItemKind::Fn(f) => self.collect_fn(f),
                ItemKind::Const(c) => {
                    let ty = self.lower_type(&c.ty);
                    self.register(c.name, DefKind::Const(ty), item.span);
                    // Record the initializer now (it is re-recorded, with its nodes typed, by
                    // `check_const`) so a `const` used as a tensor/array index can be resolved to its
                    // value by the compile-time bounds check regardless of source order.
                    self.consts.insert(c.name.sym, c.value.clone());
                }
                ItemKind::Struct(s) => {
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
        let key = |d: &Diagnostic| (d.code, d.message.clone(), d.labels.first().map(|l| l.span));
        let seen: HashSet<(Option<&'static str>, String, Option<Span>)> =
            self.diags[..before].iter().map(key).collect();
        let mut i = before;
        while i < self.diags.len() {
            if seen.contains(&key(&self.diags[i])) {
                self.diags.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Reject a struct that contains itself by value — directly (`struct S { x: S }`) or transitively
    /// (`A` holds `B` holds `A`). Such a type has infinite size; sizing or instantiating it
    /// stack-overflows mir_build's layout pass (a compiler crash on a plausible mistake — forgetting
    /// the indirection). A field behind a pointer/reference has fixed size and breaks the cycle, as
    /// in C/Rust (cf. Rust's E0072). Runs after `collect`, when every struct is registered.
    fn check_recursive_types(&mut self, module: &Module) {
        for item in &module.items {
            // A struct or a data-carrying enum can be recursive by value; a C-style enum holds
            // nothing (`contained_types` returns empty for it), so it can never be flagged.
            let (root, kind) = match &item.kind {
                ItemKind::Struct(s) => (s.name.sym, "struct"),
                ItemKind::Enum(e) => (e.name.sym, "enum"),
                _ => continue,
            };
            // DFS over by-value containment from `root`; reaching `root` means infinite size.
            let mut stack = self.contained_types(root);
            let mut visited: Vec<Symbol> = Vec::new();
            let mut recursive = false;
            while let Some(cur) = stack.pop() {
                if cur == root {
                    recursive = true;
                    break;
                }
                if visited.contains(&cur) {
                    continue;
                }
                visited.push(cur);
                stack.extend(self.contained_types(cur));
            }
            if recursive {
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
                    for fty in v.payload.field_tys() {
                        collect_value_structs(&fty, &mut out);
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
    /// (a crash on a plausible typo). Mirrors `check_recursive_types`: a DFS over the
    /// const-reference graph that flags reaching the root. Runs after `collect`, when every const is
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
        for item in &module.items {
            if let ItemKind::Const(c) = &item.kind {
                let root = c.name.sym;
                let mut stack: Vec<Symbol> = Vec::new();
                collect_const_refs(&c.value, &names, &mut stack);
                let mut visited: Vec<Symbol> = Vec::new();
                let mut recursive = false;
                while let Some(cur) = stack.pop() {
                    if cur == root {
                        recursive = true;
                        break;
                    }
                    if visited.contains(&cur) {
                        continue;
                    }
                    visited.push(cur);
                    if let Some(e) = inits.get(&cur) {
                        collect_const_refs(e, &names, &mut stack);
                    }
                }
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
        self.defs.defs.push(Def {
            name: name.sym,
            kind,
            span,
        });
        self.defs.by_name.insert(name.sym, idx);
    }

    // ---- Type lowering: AST type syntax -> semantic Ty ----

    fn lower_type(&mut self, t: &TypeExpr) -> Ty {
        match &t.kind {
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
            TypeKind::Array { elem, len } => {
                let elem = Box::new(self.lower_type(elem));
                Ty::Array {
                    elem,
                    len: self.array_len(len),
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
        }
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
                if let Some(init) = self.consts.get(s).cloned() {
                    return Dim::Const(self.eval_usize_depth(&init, 0));
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

    /// A fixed-size array's compile-time length, with the two diagnostics `eval_usize` cannot report
    /// (it is `&self`, and every caller wants a number): an unresolvable length NAME, and a length
    /// beyond what lowering can represent — mir_build's mirrored `const_usize_expr` is `u32` end to
    /// end and narrows the folded value with an unchecked `as u32`, so `[i32; 0x1_0000_0001]` was
    /// bounds checked here against 4294967297 while the emitted slot held 1 element. Both point at
    /// the length expression and are gated on `checking_bodies` exactly like the E0504 dim check, so
    /// a signature re-lowered by `check_fn` is reported once (see the flag's doc-comment).
    fn array_len(&mut self, len: &Expr) -> u64 {
        // An array length naming an identifier that is neither a declared generic nor a top-level
        // `const` is a typo, not a length. `fn sum(a: [i32; NOPE])` typed the parameter `[i32; 0]`
        // and compiled clean — mir_build's `const_usize_expr` returned `None` and lowered a bare
        // pointer — so every compile-time bounds check on that array was vacuous or nonsensical
        // (`a[0]` reported "index 0 is out of bounds for an array of length 0", naming neither
        // `NOPE` nor the real cause). Tensor dimensions already get exactly this check (E0504);
        // array lengths had no equivalent. Generics are consulted first, like `lower_dim` does, and
        // the same `checking_bodies` gate applies for the same reason: `collect` registers consts as
        // it walks, so a const declared after this item is only guaranteed visible in the body pass.
        if self.checking_bodies {
            if let ExprKind::Path(p) = &len.kind {
                if p.is_single() {
                    let sym = p.first().sym;
                    if !self.generics.contains(&sym) && !self.consts.contains_key(&sym) {
                        let nm = self.sym_str(sym).to_string();
                        self.error(
                            len.span,
                            "E0301",
                            format!(
                                "cannot find `{nm}` in this scope (an array length must be an \
                                 integer literal, a `const`, or a declared generic parameter)"
                            ),
                        );
                    }
                }
            }
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
        n
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
    /// order-independent. `mir_build`'s `const_usize_expr`/`const_usize_depth` mirrors this arm for
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
            ExprKind::Path(p) if p.is_single() => match self.consts.get(&p.first().sym) {
                Some(init) => self.eval_usize_depth(init, depth + 1),
                None => 0,
            },
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

    fn check_bodies(&mut self, module: &Module) {
        for item in &module.items {
            match &item.kind {
                ItemKind::Fn(f) => {
                    if let Some(body) = &f.body {
                        self.check_fn(f, body);
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
    fn check_const(&mut self, c: &ConstDecl, span: Span) {
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
        self.consts.insert(c.name.sym, c.value.clone());
    }

    fn check_fn(&mut self, f: &FnDecl, body: &Block) {
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

    /// Check a returned value's type against the declared return type `self.ret_ty`. Three checks, in
    /// order: tensor / vector SHAPE agreement in RIGID mode — a function must not lie about its output
    /// shape, since callers propagate the declared return shape into downstream shape checks (a single
    /// wrong return silently poisons every caller); a scalar-vs-pointer/aggregate KIND clash; and
    /// scalar-type agreement for a non-literal value. An unsuffixed literal still adapts, so this
    /// never over-fires on e.g. `return 5` from `-> i64`; every other kind stays lenient.
    fn check_return_shape(&mut self, val_ty: &Ty, val: &Expr, span: Span) {
        let ret = self.ret_ty.clone();
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
        if self.scalar_aggregate_clash(&ret, val_ty) {
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
                    .map_or(false, |s| s.contains(&name));
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
                    .map_or(false, |s| s.contains(&name));
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
                if self.scalar_aggregate_clash(&target_ty, &value_ty) {
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
                        self.check_return_shape(&t, e, e.span);
                        // An out-of-range integer literal returned where a narrower type is declared
                        // (`return 9999999999` from `-> i32`) was silently truncated to the low bits
                        // (exit code 255 from a wrapped value). Range-check it against the return
                        // type, the same rule `let`/`const` already apply to their annotation.
                        self.range_check_int_literal(e, &ret);
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
                                self.check_binop_shapes(&p, &vt, v.span);
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
            (ExprKind::Binary { op, lhs, rhs }, Ty::Scalar(_))
                if matches!(
                    op,
                    BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Rem
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor
                ) =>
            {
                self.literal_adapts(ann, lhs) && self.literal_adapts(ann, rhs)
            }
            // An array / tuple literal adapts element-wise to a matching aggregate annotation, so a
            // typed buffer can be built from literals — `let a: [i8; 2] = [127, 0]` and
            // `let t: (u8, u8) = (200, 1)` previously failed as `[i32; 2]`/`(i32, i32)` mismatches.
            // Lengths must match and every element must itself adapt (recursively, so nested
            // aggregates work too).
            (ExprKind::ArrayLit(items), Ty::Array { elem, len }) => {
                items.len() as u64 == *len && items.iter().all(|it| self.literal_adapts(elem, it))
            }
            // …and so does a repeat initializer — but its COUNT must match the annotation's length,
            // exactly like the `ArrayLit` arm above and `check_struct_literal`'s array-field check.
            // Without it `let a: [i32; 4] = [7; 2];` compiled clean and mir_build filled the slot to
            // the annotation's 4, silently discarding the count the programmer wrote (the identical
            // mismatch is a hard error as a struct field and as an array literal).
            (ExprKind::ArrayRepeat { value, count }, Ty::Array { elem, len }) => {
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
            (ExprKind::Binary { op, lhs, rhs }, Ty::Scalar(_))
                if matches!(
                    op,
                    BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Rem
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor
                ) =>
            {
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
            _ => {}
        }
    }

    /// Diagnose an unsuffixed integer literal that does not fit the narrow type it adapts to
    /// (`let x: i8 = 200;`). Only types whose whole range fits in `i64` are checked (i8..u32), and
    /// only a literal `eval_const_int` can read (incl. a leading `-`) — so a runtime value, a float,
    /// or an `i64`/`u64`/`usize` literal is never flagged. Rust rejects this; without it the value
    /// silently wraps (`200 as i8 == -56`), a quiet footgun in a safety-first language. Call at each
    /// site a literal adapts to a type (`let`/`const`/argument).
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

    /// Reject a `\u{…}` escape that is not a Unicode scalar value, keeping the established split
    /// (sema validates literals, mir_build decodes them) alongside `int_literal_well_formed` /
    /// `float_literal_well_formed`.
    fn check_unicode_escapes(&mut self, sym: Symbol, span: Span) {
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
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                let elem = self.type_expr(value);
                let len = self.eval_usize(count);
                self.type_expr(count);
                Ty::Array {
                    elem: Box::new(elem),
                    len,
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
                    self.check_binop_shapes(&then_ty, &else_ty, else_expr.span);
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
                    self.check_binop_shapes(&result, &t, arm.body.span);
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

/// Loose compatibility used only where we are confident enough to diagnose (e.g. `let`
/// annotation vs initializer). Unknown/Error are compatible with anything.
fn compatible(a: &Ty, b: &Ty) -> bool {
    if a.is_unknown() || b.is_unknown() || a.is_error() || b.is_error() {
        return true;
    }
    if a == b {
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
        // Both arms must diverge; an `if` with no `else` can fall through.
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => match else_branch {
            Some(els) => block_diverges(then_branch) && expr_diverges(els),
            None => false,
        },
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
    let body = s.replace('_', "");
    let (digits, radix): (&str, u32) =
        if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
            (h, 16)
        } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
            (o, 8)
        } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
            (b, 2)
        } else {
            (&body, 10)
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
    let body = s.replace('_', "");
    !body.is_empty() && body.parse::<f64>().is_ok()
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
    let body = s.replace('_', "");
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
    let body = s.replace('_', "");
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
    fn tensor_index_rank_mismatch_errors() {
        let src = "fn f(a: Tensor[f32, 4, 4]) { let x = a[0]; }";
        assert!(
            errors(src).contains(&"E0501"),
            "expected a rank mismatch: {:?}",
            errors(src)
        );
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
    fn array_length_above_u32_max_is_rejected() {
        // mir_build's mirrored `const_usize_expr` is u32 end to end and narrows with an unchecked
        // `as u32`, so a longer length wrapped to a small slot while sema bounds-checked the full
        // value. Reject it here so the truncating path is unreachable.
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
