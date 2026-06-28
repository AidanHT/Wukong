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
pub fn check(module: &Module, interner: &Interner) -> (SemaResult, Vec<Diagnostic>) {
    let mut s = Sema {
        interner,
        defs: DefMap::default(),
        diags: Vec::new(),
        types: HashMap::new(),
        scopes: Vec::new(),
        generics: HashSet::new(),
        ret_ty: Ty::Unit,
        loop_labels: Vec::new(),
        consts: HashMap::new(),
    };
    s.collect(module);
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
                        variants.push((v.name.sym, disc));
                        next = disc + 1;
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
        match &e.kind {
            ExprKind::Int(s) => {
                let text = self.sym_str(*s);
                text.chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0)
            }
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
            // Pointer/reference retype: only when the pointee types are identical (mutability may
            // differ). A differing pointee reinterprets the referent's bytes — a backend divergence.
            (
                Ty::Ptr { pointee: a, .. } | Ty::Ref { pointee: a, .. },
                Ty::Ptr { pointee: b, .. } | Ty::Ref { pointee: b, .. },
            ) => a == b,
            _ => false,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
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
            StmtKind::Let { pat, ty, init, .. } => {
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
            }
            StmtKind::Assign { target, value, .. } => {
                let target_ty = self.type_expr(target);
                let value_ty = self.type_expr(value);
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
                self.type_expr(cond);
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
                self.type_expr(e);
                Ty::Unknown
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
            PatKind::Int { .. } | PatKind::Bool(_) | PatKind::Path(_) | PatKind::Range { .. } => {}
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
            _ => false,
        }
    }

    /// Re-stamp an adapted numeric literal — and the literal inside any unary minus — with the
    /// `let`'s annotated type, so MIR lowering sees one consistent width (e.g. `-1.5` becomes
    /// `f64` end to end, not `-(1.5: f32)` widened to `f64` at the `Neg`).
    fn retype_adapted_literal(&mut self, e: &Expr, ann: &Ty) {
        self.types.insert(e.id, ann.clone());
        if let ExprKind::Unary {
            op: UnOp::Neg,
            expr,
        } = &e.kind
        {
            self.retype_adapted_literal(expr, ann);
        }
    }

    /// Diagnose an unsuffixed integer literal that does not fit the narrow type it adapts to
    /// (`let x: i8 = 200;`). Only types whose whole range fits in `i64` are checked (i8..u32), and
    /// only a literal `eval_const_int` can read (incl. a leading `-`) — so a runtime value, a float,
    /// or an `i64`/`u64`/`usize` literal is never flagged. Rust rejects this; without it the value
    /// silently wraps (`200 as i8 == -56`), a quiet footgun in a safety-first language. Call at each
    /// site a literal adapts to a type (`let`/`const`/argument).
    fn range_check_int_literal(&mut self, e: &Expr, ty: &Ty) {
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
            ExprKind::Int(s) => Ty::Scalar(int_lit_scalar(self.sym_str(*s))),
            ExprKind::Float(s) => Ty::Scalar(float_lit_scalar(self.sym_str(*s))),
            ExprKind::Bool(_) => Ty::Scalar(Scalar::Bool),
            ExprKind::Str(_) => Ty::Ptr {
                mutable: false,
                pointee: Box::new(Ty::Scalar(Scalar::U8)),
            },
            ExprKind::Char(_) => Ty::Scalar(Scalar::U32),
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
                    UnOp::Deref => match t {
                        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } => *pointee,
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
                    UnOp::Neg | UnOp::Not => t,
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.type_expr(lhs);
                let r = self.type_expr(rhs);
                // Operand shapes must agree (tensor-tensor / vector-vector); a mismatch is a real
                // error regardless of whether the operator yields a value or a bool.
                self.check_binop_shapes(&l, &r, e.span);
                use BinOp::*;
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
                        }
                        Ty::Scalar(Scalar::Bool)
                    }
                    Lt | Le | Gt | Ge | And | Or => Ty::Scalar(Scalar::Bool),
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
                // A `Name { … }` whose `Name` resolves to a declared struct has that nominal type;
                // an unknown name stays lenient.
                match path.segments.last().map(|s| s.sym) {
                    Some(n)
                        if matches!(self.defs.lookup(n), Some(d) if matches!(d.kind, DefKind::Struct(_))) =>
                    {
                        Ty::Named(n)
                    }
                    _ => Ty::Unknown,
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
                self.type_expr(cond);
                let then_ty = self.type_block(then_branch);
                if let Some(e) = else_branch {
                    let else_ty = self.type_expr(e);
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
                    result = join(result, t);
                }
                result
            }
            ExprKind::SizeOf(_) | ExprKind::AlignOf(_) => Ty::Scalar(Scalar::Usize),
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
        I32 | U32 | F32 => 32,
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
    Scalar::I32
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
        } => eval_const_int(expr, interner).map(|v| -v),
        _ => None,
    }
}

/// Parse an integer literal's source text (decimal, `0x`/`0o`/`0b` radix, `_` separators, optional
/// type suffix, optional leading sign) to an `i64`, or `None` if it isn't a valid integer literal.
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
