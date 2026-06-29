//! `mercury_sema` — name resolution, type inference/checking, and shape checking.
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

mod shape;

use std::collections::{HashMap, HashSet};

use mercury_ast::*;
use mercury_diag::Diagnostic;
use mercury_span::{Interner, Span, Symbol};
use mercury_types::{Dim, Layout, Scalar, Shape, Ty};

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
    /// A C-style enum: its variants in declaration order with their resolved integer discriminants
    /// (auto-incremented from 0, or set by an explicit `= <int>`). `mir_build` lowers `E::Variant`
    /// to its discriminant constant.
    Enum(Vec<(Symbol, i64)>),
}

#[derive(Clone, Debug)]
pub struct Def {
    pub name: Symbol,
    pub kind: DefKind,
    pub span: Span,
}

/// All top-level definitions, indexed by name.
#[derive(Default, Debug)]
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
pub struct SemaResult {
    pub types: HashMap<NodeId, Ty>,
    pub defs: DefMap,
    /// Top-level `const` initializer expressions, by name. Their nodes are type-checked (and adapted
    /// literals retyped) like a `let`, so `mir_build` can lower a const reference by inlining the
    /// initializer with correct types. The `DefMap` records only a const's *type*, not its value.
    pub consts: HashMap<Symbol, Expr>,
}

/// Analyze a module, returning per-expression types and any diagnostics.
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
            StmtKind::Loop { body, .. } => collect_block_const_refs(body, consts, out),
            _ => {}
        }
    }
    if let Some(t) = &b.tail {
        collect_const_refs(t, consts, out);
    }
}

pub fn check(module: &Module, interner: &Interner) -> (SemaResult, Vec<Diagnostic>) {
    let mut s = Sema {
        interner,
        defs: DefMap::default(),
        diags: Vec::new(),
        types: HashMap::new(),
        scopes: Vec::new(),
        immutable_locals: Vec::new(),
        generics: HashSet::new(),
        ret_ty: Ty::Unit,
        loop_labels: Vec::new(),
        consts: HashMap::new(),
    };
    s.collect(module);
    s.check_recursive_structs(module);
    s.check_recursive_consts(module);
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
    generics: HashSet<Symbol>,
    ret_ty: Ty,
    /// The labels of the enclosing loops at the current point (innermost last; `None` for an
    /// unlabeled loop). A `break`/`continue` with an empty stack is a hard error (E0303) — without it
    /// the lowerer emits an `unreachable` terminator, which the interpreter traps but the native
    /// backend turns into a SIGILL, a differential-gate divergence. A labeled `break`/`continue` `'l`
    /// whose label is not on the stack is likewise E0303 (an undeclared label would otherwise leave
    /// mir_build with a terminator-less block).
    loop_labels: Vec<Option<Symbol>>,
    /// Top-level `const` initializer expressions (by name), accumulated as their bodies are checked.
    consts: HashMap<Symbol, Expr>,
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
                    // Resolve each variant's integer discriminant: an explicit `= <int>` sets it,
                    // otherwise it auto-increments from the previous (starting at 0), as in C/Rust.
                    let mut next = 0i64;
                    let mut variants = Vec::with_capacity(e.variants.len());
                    for v in &e.variants {
                        let disc = v
                            .discriminant
                            .as_ref()
                            .and_then(|d| eval_const_int(d, self.interner))
                            .unwrap_or(next);
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
                        variants.push((v.name.sym, disc));
                        // Wrapping, so a sentinel discriminant near `i64::MAX` doesn't panic the
                        // compiler on the auto-increment of the *next* variant before the range
                        // check above fires; wrapping matches two's-complement integer semantics.
                        next = disc.wrapping_add(1);
                    }
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

    /// Reject a struct that contains itself by value — directly (`struct S { x: S }`) or transitively
    /// (`A` holds `B` holds `A`). Such a type has infinite size; sizing or instantiating it
    /// stack-overflows mir_build's layout pass (a compiler crash on a plausible mistake — forgetting
    /// the indirection). A field behind a pointer/reference has fixed size and breaks the cycle, as
    /// in C/Rust (cf. Rust's E0072). Runs after `collect`, when every struct is registered.
    fn check_recursive_structs(&mut self, module: &Module) {
        for item in &module.items {
            if let ItemKind::Struct(s) = &item.kind {
                let root = s.name.sym;
                // DFS over by-value containment from `root`; reaching `root` means infinite size.
                let mut stack = self.contained_structs(root);
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
                    stack.extend(self.contained_structs(cur));
                }
                if recursive {
                    let nm = self.sym_str(root).to_string();
                    self.error(
                        item.span,
                        "E0402",
                        format!(
                            "recursive struct `{nm}` has infinite size; store the recursive field \
                             behind a pointer (e.g. `*{nm}`) to break the cycle"
                        ),
                    );
                }
            }
        }
    }

    /// The struct names a struct holds *by value* (a `Named` field, or one nested in an array/tuple
    /// field). A non-struct `Named` (an enum, a generic param, an unknown) resolves to nothing.
    fn contained_structs(&self, name: Symbol) -> Vec<Symbol> {
        let mut out = Vec::new();
        if let Some(Def {
            kind: DefKind::Struct(fields),
            ..
        }) = self.defs.lookup(name)
        {
            for (_, fty) in fields {
                collect_value_structs(fty, &mut out);
            }
        }
        out
    }

    /// Reject a `const` whose initializer depends on its own value — directly (`const A = A + 1`) or
    /// transitively (`A` uses `B`, `B` uses `A`). `mir_build` inlines a const's initializer at each
    /// use site and recurses for a const-references-const, so a cycle stack-overflows the compiler
    /// (a crash on a plausible typo). Mirrors `check_recursive_structs`: a DFS over the
    /// const-reference graph that flags reaching the root. Runs after `collect`, when every const is
    /// registered, and (like the struct check) before `check_bodies`, so the error halts the pipeline
    /// ahead of mir_build's inliner.
    fn check_recursive_consts(&mut self, module: &Module) {
        let mut inits: HashMap<Symbol, &Expr> = HashMap::new();
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
        let generics = self.generics.iter().copied().collect();
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
        if self.defs.by_name.contains_key(&name.sym) {
            self.error(
                span,
                "E0300",
                format!(
                    "the name `{}` is defined more than once",
                    self.sym_str(name.sym)
                ),
            );
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
            TypeKind::Array { elem, len } => Ty::Array {
                elem: Box::new(self.lower_type(elem)),
                len: self.eval_usize(len),
            },
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
                let shape = Shape(dims.iter().map(|d| self.lower_dim(d)).collect());
                let layout = match layout {
                    Some(mercury_ast::Layout::ColMajor) => Layout::ColMajor,
                    Some(mercury_ast::Layout::Strided) => Layout::Strided,
                    Some(mercury_ast::Layout::Tiled(v)) => Layout::Tiled(v.clone()),
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

    fn lower_dim(&self, d: &mercury_ast::Dim) -> Dim {
        match &d.kind {
            DimKind::Int(n) => Dim::Const(*n),
            DimKind::Named(s) => Dim::Var(*s),
            DimKind::Dynamic => Dim::Dynamic,
        }
    }

    fn eval_usize(&self, e: &Expr) -> u64 {
        self.eval_usize_depth(e, 0)
    }

    /// Evaluate a compile-time array length: a plain integer literal, or a single-segment path
    /// naming a top-level `const` whose initializer is itself such a length (so `const N: usize = 4;
    /// [i32; N]` sizes the array). `self.consts` is populated in `collect` before any body is
    /// checked, so this is order-independent. `mir_build`'s `const_usize_expr` mirrors this exactly
    /// — the two must agree on the length, else the slot size desyncs from these bounds checks. The
    /// depth bound guards against a cyclic const initializer (also rejected by `check_recursive_consts`).
    fn eval_usize_depth(&self, e: &Expr, depth: u32) -> u64 {
        if depth > 64 {
            return 0;
        }
        match &e.kind {
            ExprKind::Int(s) => {
                let text = self.sym_str(*s);
                text.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0)
            }
            ExprKind::Path(p) if p.is_single() => match self.consts.get(&p.first().sym) {
                Some(init) => self.eval_usize_depth(init, depth + 1),
                None => 0,
            },
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
        self.scopes.push(HashMap::new());
        self.immutable_locals.clear();
        self.immutable_locals.push(HashSet::new());
        let ann = self.lower_type(&c.ty);
        let vty = self.type_expr(&c.value);
        if self.let_compatible(&ann, &c.value, &vty) {
            self.retype_adapted_literal(&c.value, &ann);
            self.range_check_int_literal(&c.value, &ann);
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
        self.scopes.push(HashMap::new());
        // Mirror the scope reset for immutability tracking (these reset `scopes` directly instead of
        // via `push_scope`, so the two stacks would otherwise desync and the check never fires).
        self.immutable_locals.clear();
        self.immutable_locals.push(HashSet::new());
        for p in &f.params {
            let ty = self.lower_type(&p.ty);
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
            self.check_return_shape(&body_ty, tail.span);
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

    /// Check a returned value's type against the declared return type `self.ret_ty`. Only tensor /
    /// vector SHAPE agreement is enforced — a function must not lie about its output shape, since
    /// callers propagate the declared return shape into downstream shape checks (a single wrong
    /// return silently poisons every caller). Scalars and other kinds stay lenient (numeric coercion
    /// at lowering, like `let`/assignment), so this never over-fires on e.g. `return 5` from `-> i64`.
    fn check_return_shape(&mut self, val_ty: &Ty, span: Span) {
        let ret = self.ret_ty.clone();
        if matches!(ret, Ty::Tensor { .. } | Ty::Vector { .. })
            || matches!(val_ty, Ty::Tensor { .. } | Ty::Vector { .. })
        {
            let mut dims = HashMap::new();
            self.unify(&ret, val_ty, &mut dims, span);
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
    }

    /// An elementwise binary operator requires its operand *shapes* to agree: adding two tensors of
    /// different shape (`Tensor[f32,2,3] + Tensor[f32,3,2]`) is meaningless, yet the result-type
    /// `join` picks one operand and lets it through. This applies the headline shape check to
    /// operators — the call-site unifier already covers function arguments, so this closes the last
    /// of the three shape-bearing contexts. Only fires when *both* sides are tensors, or both are
    /// vectors: a tensor/scalar pairing stays lenient (scalar broadcast), and two scalars promote via
    /// `join` (mixed precision like `(i as f64) + 1.0` must not error).
    fn check_binop_shapes(&mut self, l: &Ty, r: &Ty, span: Span) {
        if matches!(
            (l, r),
            (Ty::Tensor { .. }, Ty::Tensor { .. }) | (Ty::Vector { .. }, Ty::Vector { .. })
        ) {
            let mut dims = HashMap::new();
            self.unify(l, r, &mut dims, span);
        }
    }

    /// Whether `t` is an aggregate with no scalar value — a tuple, a fixed-size array, or a *struct*
    /// (`Ty::Named` resolving to a `DefKind::Struct`). An *enum* `Ty::Named` is **not** an aggregate:
    /// a C-style variant is its integer discriminant, so enum equality is well-defined. Used to
    /// reject `==`/`!=` on values whose MIR is a base pointer (where a compare would be meaningless).
    fn is_aggregate_ty(&self, t: &Ty) -> bool {
        match t {
            Ty::Tuple(_) | Ty::Array { .. } => true,
            Ty::Named(n) => matches!(
                self.defs.lookup(*n).map(|d| &d.kind),
                Some(DefKind::Struct(_))
            ),
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
        let pointerish =
            |t: &Ty| matches!(t, Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Array { .. } | Ty::Tuple(_));
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
            self.error(
                span,
                "E0401",
                format!(
                    "invalid cast: `{}` cannot be cast to `{}`",
                    from.display(self.interner),
                    to.display(self.interner)
                ),
            );
        }
    }

    /// The permitted `as` conversions: scalar↔scalar (every numeric/bool/char pairing — a real
    /// numeric conversion), scalar↔enum (a C-style discriminant), and pointer→pointer only when the
    /// pointee types match (a same-layout retype). Everything else — pointer↔integer,
    /// aggregate↔scalar, differing-element pointer casts — is a byte reinterpret the two backends
    /// disagree on, so it is rejected.
    fn cast_is_valid(&self, from: &Ty, to: &Ty) -> bool {
        if from == to {
            return true;
        }
        // A scalar, or an enum `Named` (its integer discriminant) — both integer-representable.
        let scalar_like = |t: &Ty| match t {
            Ty::Scalar(_) => true,
            Ty::Named(n) => matches!(
                self.defs.lookup(*n).map(|d| &d.kind),
                Some(DefKind::Enum(_))
            ),
            _ => false,
        };
        match (from, to) {
            (a, b) if scalar_like(a) && scalar_like(b) => true,
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
                    format!("missing field{s} {} in this struct initializer", missing.join(", ")),
                );
            }
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.immutable_locals.push(HashSet::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.immutable_locals.pop();
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
                            // An untyped literal adopts the annotated type, top to bottom (so a
                            // negated literal like `-1.5` re-stamps the inner literal too).
                            self.retype_adapted_literal(init_expr, a);
                            self.range_check_int_literal(init_expr, a);
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
                }
                // Reassigning an immutable binding (`let x = 5; x = 10;`): the language requires
                // `mut` for reassignment, but it was never enforced. Reject a direct assignment to
                // an immutable local (a single-name target). Mutating *through* an immutable binding
                // (`a[i] = …`, `s.f = …`, `*p = …`) stays lenient — those are place projections, not
                // a rebinding, so this conservative check never over-fires.
                if let ExprKind::Path(p) = &target.kind {
                    if p.is_single() && self.is_immutable_local(p.first().sym) {
                        let nm = self.sym_str(p.first().sym).to_string();
                        self.error(
                            target.span,
                            "E0304",
                            format!(
                                "cannot assign twice to immutable binding `{nm}`; add `mut` to its \
                                 `let` to allow reassignment"
                            ),
                        );
                    }
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
                self.type_expr(e);
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
                        if matches!(ret, Ty::Unit) && !matches!(t, Ty::Unit | Ty::Unknown | Ty::Error)
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
                        self.check_return_shape(&t, e.span);
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
            StmtKind::Break(lbl) | StmtKind::Continue(lbl) => {
                let kw = if matches!(s.kind, StmtKind::Break(_)) {
                    "break"
                } else {
                    "continue"
                };
                if self.loop_labels.is_empty() {
                    self.error(s.span, "E0303", format!("`{kw}` outside of a loop"));
                } else if let Some(l) = lbl {
                    // A labeled `break`/`continue` must name an enclosing loop's label.
                    if !self.loop_labels.iter().any(|x| *x == Some(l.sym)) {
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
                self.loop_labels.push(label.as_ref().map(|l| l.sym));
                self.type_block(body);
                self.loop_labels.pop();
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
                self.loop_labels.push(label.as_ref().map(|l| l.sym));
                self.type_block(body);
                self.loop_labels.pop();
                self.pop_scope();
            }
            StmtKind::Loop { body, label, .. } => {
                self.loop_labels.push(label.as_ref().map(|l| l.sym));
                self.type_block(body);
                self.loop_labels.pop();
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
                // body type-checks (`for x in [T; N]` ⇒ `x: T`). Any non-array iterand stays
                // `Unknown` (lenient — we don't newly reject other iterables here).
                match self.type_expr(e) {
                    Ty::Array { elem, .. } => *elem,
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
            // Literal / enum-variant / range patterns bind nothing — they test the scrutinee's value.
            PatKind::Int { .. }
            | PatKind::Char(_)
            | PatKind::Bool(_)
            | PatKind::Path(_)
            | PatKind::Range { .. } => {}
        }
    }

    /// Whether `init` can initialize a `let` annotated `ann`. Besides ordinary compatibility,
    /// an *unsuffixed* numeric literal adapts to any integer/float annotation (Rust's `{integer}`
    /// inference, in miniature).
    fn let_compatible(&self, ann: &Ty, init: &Expr, init_ty: &Ty) -> bool {
        if compatible(ann, init_ty) {
            return true;
        }
        self.literal_adapts(ann, init)
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
            (ExprKind::ArrayRepeat { value, .. }, Ty::Array { elem, .. }) => {
                self.literal_adapts(elem, value)
            }
            (ExprKind::TupleLit(items), Ty::Tuple(tys)) => {
                items.len() == tys.len()
                    && items.iter().zip(tys).all(|(it, t)| self.literal_adapts(t, it))
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
            ExprKind::Str(_) => Ty::Ptr {
                mutable: false,
                pointee: Box::new(Ty::Scalar(Scalar::U8)),
            },
            ExprKind::Char(_) => Ty::Scalar(Scalar::Char),
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
                if matches!(op, Add | Sub | Mul | Div | Rem | BitAnd | BitOr | BitXor | Shl | Shr) {
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
                        format!("arithmetic operator `{}` is not defined for `bool`", op.glyph()),
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
                        }
                        Ty::Scalar(Scalar::Bool)
                    }
                    And | Or => Ty::Scalar(Scalar::Bool),
                    _ => join(l, r),
                }
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => self.type_call(callee, generic_args, args, e.span),
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
                            if variants.iter().any(|(vname, _)| *vname == name.sym) {
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
                    Ty::Tuple(elems) => elems.get(*index as usize).cloned().unwrap_or(Ty::Unknown),
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
            ExprKind::SizeOf(_) | ExprKind::AlignOf(_) => Ty::Scalar(Scalar::Usize),
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
            // scrutinee (Mercury has no uninhabited types). The `Ty::Scalar(_)` case below already
            // catches an empty int/char match; this also catches an empty enum / `bool` match (whose
            // coverage arms below would read "no cases seen" as lenient). Unmodeled scrutinees stay
            // lenient, matching the rest of this function.
            return match scrut_ty {
                Ty::Named(n) => {
                    matches!(self.defs.lookup(*n).map(|d| &d.kind), Some(DefKind::Enum(_)))
                }
                Ty::Scalar(_) => true,
                _ => false,
            };
        }
        match scrut_ty {
            Ty::Named(n) => match self.defs.lookup(*n).map(|d| &d.kind) {
                Some(DefKind::Enum(variants)) => {
                    let mut covered = HashSet::new();
                    for a in arms {
                        if a.guard.is_none() {
                            collect_variant_names(&a.pat, &mut covered);
                        }
                    }
                    // Only reason about coverage when the arms actually use variant patterns; a match
                    // by raw discriminant (or some unmodeled spelling) stays lenient.
                    !covered.is_empty() && variants.iter().any(|(v, _)| !covered.contains(v))
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

fn generic_names(gs: &[GenericParam]) -> HashSet<Symbol> {
    gs.iter()
        .map(|g| match &g.kind {
            GenericParamKind::Type(id) => id.sym,
            GenericParamKind::Const { name, .. } => name.sym,
        })
        .collect()
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
    a == b
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

/// The inclusive value range of a narrow integer scalar, as `i64` bounds. `None` for `bool`/floats
/// and for `i64`/`u64`/`usize`/`isize` — a literal that parses to an `i64` always fits those, and a
/// `u64` near its top doesn't fit an `i64` to compare, so they are left unchecked rather than
/// mis-flagged.
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
        // A `loop` with no `break` anywhere in its body never exits normally (it loops forever or
        // returns from inside) — it diverges. Any `break` means it may fall through (be lenient).
        StmtKind::Loop { body, .. } => !block_contains_break(body),
        _ => false,
    }
}

fn expr_diverges(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Block(b) => block_diverges(b),
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

/// Record every enum-variant name a (guard-less) pattern covers, flattening or-patterns. A `Path`
/// pattern's last segment is the variant name (`Color::Red` → `Red`); other pattern kinds contribute
/// nothing. Used by `match_is_provably_nonexhaustive` for enum coverage.
fn collect_variant_names(p: &Pattern, out: &mut HashSet<Symbol>) {
    match &p.kind {
        PatKind::Path(path) => {
            if let Some(seg) = path.segments.last() {
                out.insert(seg.sym);
            }
        }
        PatKind::Or(alts) => alts.iter().for_each(|a| collect_variant_names(a, out)),
        _ => {}
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
        StmtKind::Break(_) => true,
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_contains_break(e),
        StmtKind::Return(opt) => opt.as_ref().map(expr_contains_break).unwrap_or(false),
        StmtKind::Let { init, .. } => init.as_ref().map(expr_contains_break).unwrap_or(false),
        StmtKind::Assign { value, .. } => expr_contains_break(value),
        StmtKind::While { body, .. } | StmtKind::For { body, .. } | StmtKind::Loop { body, .. } => {
            block_contains_break(body)
        }
        StmtKind::Continue(_) => false,
    }
}

fn expr_contains_break(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Block(b) => block_contains_break(b),
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

/// Evaluate a constant integer expression — an integer literal (with an optional unary minus) — to
/// its value, for resolving an enum variant's explicit discriminant (`A = 10`). Returns `None` for
/// anything not a compile-time integer literal (the variant then auto-increments).
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

/// Parse an integer literal's source text (decimal, `0x`/`0o`/`0b` radix, `_` separators, optional
/// type suffix, optional leading sign) to an `i64`, or `None` if it isn't a valid integer literal.
/// Whether an integer literal's text denotes a value Mercury can represent — it parses, after the
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
        && (i64::from_str_radix(digits, radix).is_ok() || u64::from_str_radix(digits, radix).is_ok())
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
fn parse_u64_text(text: &str) -> Option<u64> {
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
    use mercury_span::SourceId;

    fn analyze(src: &str) -> (Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
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
            "fn f(n: i32) { match n { 0 => {}, 1 => {} } }",            // statement, unit arms
            "fn f(n: i32) -> i32 { let x = match n {}; return x; }",    // empty match
            "fn f(n: i32) -> i32 { return match n { 0 => 1, 1 => 2 }; }", // value (regression)
            "enum E { A, B } fn f(e: E) { match e {} }",               // empty enum match
            "fn f(b: bool) { match b { true => {} } }",                // bool missing a case
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
            assert!(!errors(src).contains(&"E0405"), "unexpected E0405 for {src:?}");
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
            assert!(!errors(src).contains(&"E0401"), "unexpected E0401 for {src:?}");
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
            assert!(!errors(src).contains(&"E0401"), "unexpected E0401 for {src:?}");
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
            assert!(!errors(src).contains(&"E0401"), "unexpected E0401 for {src:?}");
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
            assert!(!errors(src).contains(&"E0401"), "unexpected E0401 for {src:?}");
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
}
