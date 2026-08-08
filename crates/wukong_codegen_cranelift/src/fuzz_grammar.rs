//! Grammar-based **whole-program** differential fuzzer: random well-typed Wukong programs
//! through the REAL pipeline (parse → sema → mir_build → optimize → interp / Cranelift JIT),
//! asserting the two hard invariants on every one:
//!
//!   1. interp == native on (exit code, stdout), and
//!   2. -O0 == -O2 == -O3 on (exit code, stdout),
//!
//! plus, per compile, that every function passes `wukong_mir::verify` after the pass pipeline (a
//! verify failure is reported like a compile failure). The five executions compared are interp at
//! -O0/-O2/-O3 and native at -O0/-O3.
//!
//! Sixteen manual "gap-hunt sweeps" found real miscompiles by hand-writing probe programs in
//! exactly this space (scalar arithmetic, casts, control flow, arrays, calls, if-values); the
//! generator has since grown past it — every program also carries a by-reference `mut` aggregate
//! param (`h0`) and an aggregate return by value (`h1`, the sret path), and the body may draw struct,
//! nested-aggregate and enum-`match` locals. This
//! automates that: a deterministic SplitMix64-seeded generator (a failure reproduces exactly —
//! the panic prints the seed and full source) emits programs that sema must ACCEPT — the
//! generator respects every deliberate rejection (same-type comparisons only, no chained
//! comparison, guarded integer division by positive literals, casts on every narrowing, bounded
//! loops, in-bounds constant/loop-var indexing, definite returns) — so a sema error is itself a
//! finding (a generator/checker disagreement), and a backend/opt divergence is a real bug.
//!
//! Float ops are included deliberately: scalar float arithmetic, `sqrt`/`abs`/`fmax`/`fmin`,
//! saturating float→int casts, and division by zero (guarded to 0 for ints, IEEE for floats) are
//! all documented deterministic-and-identical across backends, so they are fair game. Values are
//! printed via `(expr) as i32` / integer prints so stdout is exact.

use wukong_span::{Interner, SourceId};

/// SplitMix64 (same as `fuzz.rs`) — dependency-free determinism.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    I32,
    I64,
    F32,
    Bool,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::F32 => "f32",
            Ty::Bool => "bool",
        }
    }
    fn pick_numeric(rng: &mut Rng) -> Ty {
        match rng.below(3) {
            0 => Ty::I32,
            1 => Ty::I64,
            _ => Ty::F32,
        }
    }
}

#[derive(Clone)]
struct Var {
    name: String,
    ty: Ty,
    mutable: bool,
    /// A loop counter the loop's own step owns — never reassigned by generated statements.
    protected: bool,
}

/// One array local: name, element type, and (const) length.
#[derive(Clone)]
struct Arr {
    name: String,
    elem: Ty,
    len: usize,
}

/// Everything a lexical scope can shadow — snapshot on block entry, restore on exit.
struct Marks(usize, usize, usize, usize, usize, usize);

struct Gen {
    rng: Rng,
    vars: Vec<Var>,
    arrs: Vec<Arr>,
    /// `Pt { x: f32, y: i32 }` locals (all mutable).
    pts: Vec<String>,
    /// `Box2 { a: Pt, k: i64 }` locals (all mutable) — nested-aggregate coverage.
    boxes: Vec<String>,
    /// `Col { R, G, B }` enum locals (all mutable).
    cols: Vec<String>,
    /// `(i32, f32)` tuple locals (immutable).
    tups: Vec<String>,
    /// Loop variables of enclosing `for i in 0..N` loops: (name, exclusive bound).
    loop_vars: Vec<(String, usize)>,
    /// Whether helper-fn calls may appear (false inside the helpers themselves).
    allow_calls: bool,
    next_id: usize,
    depth: u32,
    src: String,
    indent: usize,
    prints: usize,
}

impl Gen {
    fn new(seed: u64) -> Self {
        Gen {
            rng: Rng(seed),
            vars: Vec::new(),
            arrs: Vec::new(),
            pts: Vec::new(),
            boxes: Vec::new(),
            cols: Vec::new(),
            tups: Vec::new(),
            loop_vars: Vec::new(),
            allow_calls: true,
            next_id: 0,
            depth: 0,
            src: String::new(),
            indent: 1,
            prints: 0,
        }
    }

    fn marks(&self) -> Marks {
        Marks(
            self.vars.len(),
            self.arrs.len(),
            self.pts.len(),
            self.boxes.len(),
            self.cols.len(),
            self.tups.len(),
        )
    }

    fn truncate_to(&mut self, m: &Marks) {
        self.vars.truncate(m.0);
        self.arrs.truncate(m.1);
        self.pts.truncate(m.2);
        self.boxes.truncate(m.3);
        self.cols.truncate(m.4);
        self.tups.truncate(m.5);
    }

    fn fresh(&mut self, prefix: &str) -> String {
        let n = self.next_id;
        self.next_id += 1;
        format!("{prefix}{n}")
    }

    fn line(&mut self, s: String) {
        for _ in 0..self.indent {
            self.src.push_str("    ");
        }
        self.src.push_str(&s);
        self.src.push('\n');
    }

    /// A literal of type `t`, kept small so annotated narrows never range-error.
    fn literal(&mut self, t: Ty) -> String {
        match t {
            Ty::I32 | Ty::I64 => {
                let v = self.rng.below(201) as i64 - 100;
                format!("{v}")
            }
            Ty::F32 => {
                let v = (self.rng.below(4001) as f64 - 2000.0) / 100.0;
                format!("{v:.2}")
            }
            Ty::Bool => if self.rng.chance(50) { "true" } else { "false" }.into(),
        }
    }

    /// An in-scope scalar variable of type `t`, if any.
    fn var_of(&mut self, t: Ty) -> Option<String> {
        let cands: Vec<&Var> = self.vars.iter().filter(|v| v.ty == t).collect();
        if cands.is_empty() {
            return None;
        }
        let i = self.rng.below(cands.len() as u64) as usize;
        Some(cands[i].name.clone())
    }

    /// An array element read whose index is provably in bounds: a literal `< len`, or an
    /// enclosing loop var whose bound `<= len`.
    fn arr_read(&mut self, t: Ty) -> Option<String> {
        let cands: Vec<Arr> = self.arrs.iter().filter(|a| a.elem == t).cloned().collect();
        if cands.is_empty() {
            return None;
        }
        let a = &cands[self.rng.below(cands.len() as u64) as usize];
        let idx = self.index_for(a.len);
        Some(format!("{}[{}]", a.name, idx))
    }

    fn index_for(&mut self, len: usize) -> String {
        let loops: Vec<(String, usize)> = self
            .loop_vars
            .iter()
            .filter(|(_, b)| *b <= len)
            .cloned()
            .collect();
        if !loops.is_empty() && self.rng.chance(70) {
            loops[self.rng.below(loops.len() as u64) as usize].0.clone()
        } else {
            format!("{}", self.rng.below(len as u64))
        }
    }

    /// An expression of type `t`. `depth` bounds recursion.
    fn expr(&mut self, t: Ty, depth: u32) -> String {
        if depth == 0 {
            return match self.var_of(t) {
                Some(v) if self.rng.chance(60) => v,
                _ => self.literal(t),
            };
        }
        match t {
            Ty::Bool => match self.rng.below(4) {
                // Same-type comparison (mixed-sign / bool-vs-int comparisons are rejected).
                0 | 1 => {
                    let nt = Ty::pick_numeric(&mut self.rng);
                    let a = self.expr(nt, depth - 1);
                    let b = self.expr(nt, depth - 1);
                    let op = ["<", "<=", ">", ">=", "==", "!="][self.rng.below(6) as usize];
                    format!("(({a}) {op} ({b}))")
                }
                2 => {
                    let a = self.expr(Ty::Bool, depth - 1);
                    let b = self.expr(Ty::Bool, depth - 1);
                    let op = ["&&", "||"][self.rng.below(2) as usize];
                    format!("(({a}) {op} ({b}))")
                }
                _ => {
                    let a = self.expr(Ty::Bool, depth - 1);
                    format!("(!({a}))")
                }
            },
            Ty::F32 => match self.rng.below(8) {
                0 => self.leaf(t),
                1 | 2 => {
                    let a = self.expr(Ty::F32, depth - 1);
                    let b = self.expr(Ty::F32, depth - 1);
                    // Float division (even by zero) is IEEE-deterministic on both backends.
                    let op = ["+", "-", "*", "/"][self.rng.below(4) as usize];
                    format!("(({a}) {op} ({b}))")
                }
                3 => {
                    let a = self.expr(Ty::F32, depth - 1);
                    let f = ["sqrt", "abs", "exp", "tanh"][self.rng.below(4) as usize];
                    format!("{f}(({a}))")
                }
                4 => {
                    let a = self.expr(Ty::F32, depth - 1);
                    let b = self.expr(Ty::F32, depth - 1);
                    let f = ["fmax", "fmin"][self.rng.below(2) as usize];
                    format!("{f}(({a}), ({b}))")
                }
                5 => {
                    let it = [Ty::I32, Ty::I64][self.rng.below(2) as usize];
                    let a = self.expr(it, depth - 1);
                    format!("(({a}) as f32)")
                }
                6 => {
                    let c = self.expr(Ty::Bool, depth - 1);
                    let a = self.expr(Ty::F32, depth - 1);
                    let b = self.expr(Ty::F32, depth - 1);
                    format!("(if ({c}) {{ {a} }} else {{ {b} }})")
                }
                _ => self.leaf(t),
            },
            Ty::I32 | Ty::I64 => match self.rng.below(10) {
                0 => self.leaf(t),
                1 | 2 => {
                    let a = self.expr(t, depth - 1);
                    let b = self.expr(t, depth - 1);
                    let op = ["+", "-", "*"][self.rng.below(3) as usize];
                    format!("(({a}) {op} ({b}))")
                }
                3 => {
                    // Division/remainder by a POSITIVE literal: no div-by-zero, no MIN/-1
                    // overflow, both defined identically on the two backends.
                    let a = self.expr(t, depth - 1);
                    let d = self.rng.below(9) + 1;
                    let op = ["/", "%"][self.rng.below(2) as usize];
                    format!("(({a}) {op} {d})")
                }
                4 => {
                    let a = self.expr(t, depth - 1);
                    let b = self.expr(t, depth - 1);
                    let op = ["&", "|", "^"][self.rng.below(3) as usize];
                    format!("(({a}) {op} ({b}))")
                }
                5 => {
                    // Shift by a small literal (< width of the narrower type in play).
                    let a = self.expr(t, depth - 1);
                    let s = self.rng.below(15);
                    let op = ["<<", ">>"][self.rng.below(2) as usize];
                    format!("(({a}) {op} {s})")
                }
                6 => {
                    // Saturating float->int cast: NaN/Inf/out-of-range are all pinned bit-exact.
                    let a = self.expr(Ty::F32, depth - 1);
                    format!("(({a}) as {})", t.name())
                }
                7 => {
                    let other = if t == Ty::I32 { Ty::I64 } else { Ty::I32 };
                    let a = self.expr(other, depth - 1);
                    format!("(({a}) as {})", t.name())
                }
                8 => {
                    let c = self.expr(Ty::Bool, depth - 1);
                    let a = self.expr(t, depth - 1);
                    let b = self.expr(t, depth - 1);
                    format!("(if ({c}) {{ {a} }} else {{ {b} }})")
                }
                _ => self.leaf(t),
            },
        }
    }

    /// A leaf: variable, array element, aggregate projection, or literal.
    fn leaf(&mut self, t: Ty) -> String {
        if self.rng.chance(25) {
            if let Some(r) = self.agg_read(t) {
                return r;
            }
        }
        if self.rng.chance(35) {
            if let Some(r) = self.arr_read(t) {
                return r;
            }
        }
        match self.var_of(t) {
            Some(v) if self.rng.chance(65) => v,
            _ => self.literal(t),
        }
    }

    /// A projection out of an aggregate local of the requested type, if one is in scope:
    /// struct fields (incl. nested), tuple fields, and the enum-as-discriminant cast.
    fn agg_read(&mut self, t: Ty) -> Option<String> {
        let mut opts: Vec<String> = Vec::new();
        match t {
            Ty::F32 => {
                for p in &self.pts {
                    opts.push(format!("{p}.x"));
                }
                for b in &self.boxes {
                    opts.push(format!("{b}.a.x"));
                }
                for tu in &self.tups {
                    opts.push(format!("{tu}.1"));
                }
            }
            Ty::I32 => {
                for p in &self.pts {
                    opts.push(format!("{p}.y"));
                }
                for b in &self.boxes {
                    opts.push(format!("{b}.a.y"));
                }
                for tu in &self.tups {
                    opts.push(format!("{tu}.0"));
                }
                for e in &self.cols {
                    opts.push(format!("({e} as i32)"));
                }
            }
            Ty::I64 => {
                for b in &self.boxes {
                    opts.push(format!("{b}.k"));
                }
            }
            Ty::Bool => {}
        }
        if opts.is_empty() {
            return None;
        }
        let i = self.rng.below(opts.len() as u64) as usize;
        Some(opts.swap_remove(i))
    }

    fn stmt(&mut self) {
        match self.rng.below(18) {
            // let
            0..=2 => {
                let t = if self.rng.chance(15) {
                    Ty::Bool
                } else {
                    Ty::pick_numeric(&mut self.rng)
                };
                let e = self.expr(t, 2);
                let name = self.fresh("x");
                let mutable = self.rng.chance(70);
                let mu = if mutable { "mut " } else { "" };
                // Literal adaptation is shallow (a compound expr of unsuffixed literals types
                // as i32 even under an i64 annotation), so pin compound inits with a cast; a
                // depth-0 pure literal keeps exercising the adaptation path.
                let init = if t == Ty::Bool || !e.contains(' ') {
                    e
                } else {
                    format!("(({e}) as {})", t.name())
                };
                self.line(format!("let {mu}{name}: {} = {init};", t.name()));
                self.vars.push(Var {
                    name,
                    ty: t,
                    mutable,
                    protected: false,
                });
            }
            // assign to an existing mutable var
            3 | 4 => {
                let cands: Vec<Var> = self
                    .vars
                    .iter()
                    .filter(|v| v.mutable && !v.protected)
                    .cloned()
                    .collect();
                if let Some(v) = (!cands.is_empty())
                    .then(|| cands[self.rng.below(cands.len() as u64) as usize].clone())
                {
                    let e = self.expr(v.ty, 2);
                    // A bare unsuffixed literal only ADAPTS in a `let` (assign positions are
                    // strict E0401), so pin the RHS type with an explicit no-op-when-equal cast.
                    let e = if v.ty == Ty::Bool {
                        e
                    } else {
                        format!("(({e}) as {})", v.ty.name())
                    };
                    if v.ty != Ty::Bool && self.rng.chance(40) {
                        let op = ["+=", "-=", "*="][self.rng.below(3) as usize];
                        self.line(format!("{} {op} {e};", v.name));
                    } else {
                        self.line(format!("{} = {e};", v.name));
                    }
                }
            }
            // array declaration
            5 => {
                let t = Ty::pick_numeric(&mut self.rng);
                // `index_for` only lets a loop var index an array whose length covers its bound, so
                // these must reach the raised for-bounds above or a trip-17 loop can never index.
                let len = [4usize, 5, 8, 17, 24][self.rng.below(5) as usize];
                let init = self.literal(t);
                let name = self.fresh("a");
                self.line(format!(
                    "let mut {name}: [{}; {len}] = [{init}; {len}];",
                    t.name()
                ));
                self.arrs.push(Arr { name, elem: t, len });
            }
            // array store
            6 => {
                let cands: Vec<Arr> = self.arrs.clone();
                if let Some(a) = (!cands.is_empty())
                    .then(|| cands[self.rng.below(cands.len() as u64) as usize].clone())
                {
                    let idx = self.index_for(a.len);
                    let e = self.expr(a.elem, 2);
                    // Same strict-assign rule as scalar assigns: pin the element type.
                    self.line(format!(
                        "{}[{}] = (({e}) as {});",
                        a.name,
                        idx,
                        a.elem.name()
                    ));
                }
            }
            // if / else
            7 | 8 => {
                if self.depth >= 3 {
                    return self.emit_print();
                }
                let c = self.expr(Ty::Bool, 2);
                self.line(format!("if ({c}) {{"));
                self.depth += 1;
                self.indent += 1;
                let m = self.marks();
                let n = self.rng.below(3) + 1;
                for _ in 0..n {
                    self.stmt();
                }
                self.truncate_to(&m);
                self.indent -= 1;
                if self.rng.chance(60) {
                    self.line("} else {".into());
                    self.indent += 1;
                    let n = self.rng.below(3) + 1;
                    for _ in 0..n {
                        self.stmt();
                    }
                    self.truncate_to(&m);
                    self.indent -= 1;
                }
                self.line("}".into());
                self.depth -= 1;
            }
            // for over a small range
            9 | 10 => {
                if self.depth >= 3 {
                    return self.emit_print();
                }
                // 3/4/5 alone kept every fuzzed loop below 8, the lane group, so no trip-dependent
                // lowering (counting-while normalization, the vector strip's `trip & !7`, a
                // non-empty tail) was ever reached. 8 is one whole group and 17 is two groups plus a
                // one-element tail. NOT sufficient to reach the vectorizer itself: measured over 400
                // generated programs, zero contain a `vec_kernels` entry or SIMD MIR, because the
                // grammar's loop bodies are scalar `let`s and cast-heavy stores, not the f32
                // elementwise nests a recipe claims. Reaching that needs a grammar that emits
                // `arr[i] = f(arr2[i], …)` bodies.
                let bound = [3usize, 4, 5, 8, 17][self.rng.below(5) as usize];
                let iv = self.fresh("i");
                self.line(format!("for {iv} in 0..{bound} {{"));
                self.depth += 1;
                self.indent += 1;
                self.loop_vars.push((iv.clone(), bound));
                self.vars.push(Var {
                    name: iv,
                    ty: Ty::I32,
                    mutable: false,
                    protected: true,
                });
                let m = self.marks();
                let n = self.rng.below(3) + 1;
                for _ in 0..n {
                    self.stmt();
                }
                self.truncate_to(&m);
                self.vars.pop();
                self.loop_vars.pop();
                self.indent -= 1;
                self.depth -= 1;
                self.line("}".into());
            }
            // bounded while with a protected counter
            11 => {
                if self.depth >= 3 {
                    return self.emit_print();
                }
                let c = self.fresh("c");
                let bound = self.rng.below(5) + 1;
                self.line(format!("let mut {c}: i32 = 0;"));
                self.line(format!("while {c} < {bound} {{"));
                self.depth += 1;
                self.indent += 1;
                self.vars.push(Var {
                    name: c.clone(),
                    ty: Ty::I32,
                    mutable: true,
                    protected: true,
                });
                let m = self.marks();
                let n = self.rng.below(2) + 1;
                for _ in 0..n {
                    self.stmt();
                }
                self.truncate_to(&m);
                self.line(format!("{c} = {c} + 1;"));
                self.vars.pop();
                self.indent -= 1;
                self.depth -= 1;
                self.line("}".into());
            }
            // Pt struct local — from a literal, or from the sret-returning helper h1.
            12 => {
                let name = self.fresh("p");
                if self.allow_calls && self.rng.chance(35) {
                    let a = self.expr(Ty::F32, 1);
                    let b = self.expr(Ty::I32, 1);
                    self.line(format!("let mut {name}: Pt = h1(({a}), ({b}));"));
                } else {
                    let x = self.expr(Ty::F32, 1);
                    let y = self.expr(Ty::I32, 1);
                    self.line(format!(
                        "let mut {name}: Pt = Pt {{ x: ({x}) as f32, y: ({y}) as i32 }};"
                    ));
                }
                self.pts.push(name);
            }
            // Pt mutation: field write, whole-struct copy, or the SELF-REFERENTIAL literal
            // assign (`p = Pt { x: …p.y…, y: …p.x… }` — the sweep-14 build-into-temp bug class).
            13 => {
                let cands = self.pts.clone();
                if let Some(p) = (!cands.is_empty())
                    .then(|| cands[self.rng.below(cands.len() as u64) as usize].clone())
                {
                    match self.rng.below(4) {
                        0 => {
                            let e = self.expr(Ty::F32, 2);
                            self.line(format!("{p}.x = ({e}) as f32;"));
                        }
                        1 => {
                            let e = self.expr(Ty::I32, 2);
                            self.line(format!("{p}.y = ({e}) as i32;"));
                        }
                        2 => {
                            let q = cands[self.rng.below(cands.len() as u64) as usize].clone();
                            self.line(format!("{p} = {q};"));
                        }
                        _ => {
                            // Deliberately read the destination's own fields in the initializer.
                            self.line(format!(
                                "{p} = Pt {{ x: (({p}.y) as f32) + 0.5, y: (({p}.x) as i32) - 1 }};"
                            ));
                        }
                    }
                }
            }
            // Nested-aggregate Box2 local + a nested field write.
            14 => {
                if self.boxes.is_empty() || self.rng.chance(50) {
                    let x = self.expr(Ty::F32, 1);
                    let y = self.expr(Ty::I32, 1);
                    let k = self.expr(Ty::I64, 1);
                    let name = self.fresh("bx");
                    self.line(format!(
                        "let mut {name}: Box2 = Box2 {{ a: Pt {{ x: ({x}) as f32, y: ({y}) as i32 }}, k: ({k}) as i64 }};"
                    ));
                    self.boxes.push(name);
                } else {
                    let cands = self.boxes.clone();
                    let b = cands[self.rng.below(cands.len() as u64) as usize].clone();
                    match self.rng.below(3) {
                        0 => {
                            let e = self.expr(Ty::F32, 2);
                            self.line(format!("{b}.a.x = ({e}) as f32;"));
                        }
                        1 => {
                            let e = self.expr(Ty::I64, 2);
                            self.line(format!("{b}.k = ({e}) as i64;"));
                        }
                        _ => {
                            if let Some(p) = (!self.pts.is_empty()).then(|| {
                                self.pts[self.rng.below(self.pts.len() as u64) as usize].clone()
                            }) {
                                self.line(format!("{b}.a = {p};"));
                            }
                        }
                    }
                }
            }
            // Enum local + exhaustive VALUE match over it (covers the value-merge path).
            15 => {
                if self.cols.is_empty() || self.rng.chance(40) {
                    let v = ["Col::R", "Col::G", "Col::B"][self.rng.below(3) as usize];
                    let name = self.fresh("en");
                    self.line(format!("let mut {name}: Col = {v};"));
                    self.cols.push(name);
                } else {
                    let cands = self.cols.clone();
                    let e = cands[self.rng.below(cands.len() as u64) as usize].clone();
                    if self.rng.chance(50) {
                        let v = ["Col::R", "Col::G", "Col::B"][self.rng.below(3) as usize];
                        self.line(format!("{e} = {v};"));
                    } else {
                        let a0 = self.expr(Ty::I32, 1);
                        let a1 = self.expr(Ty::I32, 1);
                        let a2 = self.expr(Ty::I32, 1);
                        let name = self.fresh("x");
                        self.line(format!(
                            "let {name}: i32 = match {e} {{ Col::R => ({a0}) as i32, Col::G => ({a1}) as i32, Col::B => ({a2}) as i32 }};"
                        ));
                        self.vars.push(Var {
                            name,
                            ty: Ty::I32,
                            mutable: false,
                            protected: false,
                        });
                    }
                }
            }
            // Tuple local (i32, f32) — plus occasional destructuring back into scalars.
            16 => {
                if self.tups.is_empty() || self.rng.chance(60) {
                    let a = self.expr(Ty::I32, 1);
                    let b = self.expr(Ty::F32, 1);
                    let name = self.fresh("t");
                    self.line(format!(
                        "let {name}: (i32, f32) = (({a}) as i32, ({b}) as f32);"
                    ));
                    self.tups.push(name);
                } else {
                    let cands = self.tups.clone();
                    let t = cands[self.rng.below(cands.len() as u64) as usize].clone();
                    let u = self.fresh("u");
                    let v = self.fresh("v");
                    self.line(format!("let ({u}, {v}) = {t};"));
                    self.vars.push(Var {
                        name: u,
                        ty: Ty::I32,
                        mutable: false,
                        protected: false,
                    });
                    self.vars.push(Var {
                        name: v,
                        ty: Ty::F32,
                        mutable: false,
                        protected: false,
                    });
                }
            }
            // Call a helper for its value (h0: (Pt, i32) -> f32, exercising by-ref aggregate args).
            17 => {
                if self.allow_calls {
                    if let Some(p) = (!self.pts.is_empty())
                        .then(|| self.pts[self.rng.below(self.pts.len() as u64) as usize].clone())
                    {
                        let k = self.expr(Ty::I32, 1);
                        let name = self.fresh("x");
                        self.line(format!("let {name}: f32 = h0({p}, ({k}) as i32);"));
                        self.vars.push(Var {
                            name,
                            ty: Ty::F32,
                            mutable: false,
                            protected: false,
                        });
                    } else {
                        self.emit_print();
                    }
                } else {
                    self.emit_print();
                }
            }
            _ => unreachable!(),
        }
    }

    fn emit_print(&mut self) {
        // Print an i32 projection of anything in scope: exact stdout is the strong signal.
        let t = Ty::pick_numeric(&mut self.rng);
        let e = self.expr(t, 2);
        let proj = match t {
            Ty::I32 => format!("print({e});"),
            Ty::I64 => format!("print(({e}) as i32);"),
            Ty::F32 => format!("print((({e}) * 100.0) as i32);"),
            Ty::Bool => unreachable!(),
        };
        self.line(proj);
        self.prints += 1;
    }

    /// Emit `stmts` statements (with sprinkled prints) into `self.src`.
    fn gen_body(&mut self, stmts: u64, print_chance: u64) {
        for _ in 0..stmts {
            self.stmt();
            if self.rng.chance(print_chance) {
                self.emit_print();
            }
        }
    }
}

/// Assemble one whole program: shared aggregate types, two helper fns (h0 takes a by-ref `mut Pt`
/// — caller-visible writes; h1 returns a `Pt` by value — the sret path), and `main`.
fn program(seed: u64) -> String {
    let header = format!(
        "module fuzz.p{seed}\n\n\
         struct Pt {{ x: f32, y: i32 }}\n\
         struct Box2 {{ a: Pt, k: i64 }}\n\
         enum Col {{ R, G, B }}\n\n"
    );

    // h0(mut p: Pt, k: i32) -> f32 — mutating a by-reference aggregate param.
    let mut g0 = Gen::new(seed ^ 0xA5A5_5A5A_0000_0001);
    g0.allow_calls = false;
    g0.pts.push("p".into());
    g0.vars.push(Var {
        name: "k".into(),
        ty: Ty::I32,
        mutable: false,
        protected: true,
    });
    g0.gen_body(3, 20);
    let r0 = g0.expr(Ty::F32, 2);
    g0.line(format!("return (({r0})) as f32;"));
    let h0 = format!("fn h0(mut p: Pt, k: i32) -> f32 {{\n{}}}\n\n", g0.src);

    // h1(a: f32, b: i32) -> Pt — aggregate return by value (sret).
    let mut g1 = Gen::new(seed ^ 0xA5A5_5A5A_0000_0002);
    g1.allow_calls = false;
    g1.vars.push(Var {
        name: "a".into(),
        ty: Ty::F32,
        mutable: false,
        protected: true,
    });
    g1.vars.push(Var {
        name: "b".into(),
        ty: Ty::I32,
        mutable: false,
        protected: true,
    });
    g1.gen_body(3, 20);
    let rx = g1.expr(Ty::F32, 2);
    let ry = g1.expr(Ty::I32, 2);
    g1.line(format!(
        "return Pt {{ x: ({rx}) as f32, y: ({ry}) as i32 }};"
    ));
    let h1 = format!("fn h1(a: f32, b: i32) -> Pt {{\n{}}}\n\n", g1.src);

    let mut g = Gen::new(seed);
    let stmts = g.rng.below(18) + 8;
    g.gen_body(stmts, 30);
    // Guarantee a strong stdout signal even if the RNG never rolled a print.
    while g.prints < 3 {
        g.emit_print();
    }
    // Exit code: a small, always-valid projection.
    let ret = g.expr(Ty::I32, 2);
    g.line(format!("return (({ret}) & 63);"));
    format!("{header}{h0}{h1}fn main() -> i32 {{\n{}}}\n", g.src)
}

/// Compile `src` through the real pipeline at `opt`, then run it on the given backend.
/// `Err` means the program did not compile — a generator bug worth failing loudly on.
fn run_at(src: &str, opt: u8, native: bool) -> Result<Result<(i64, Vec<u8>), String>, String> {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return Err(format!(
            "parse error: {:?}",
            pd.iter().find(|d| d.is_error())
        ));
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return Err(format!(
            "sema error: {:?}",
            sd.iter().find(|d| d.is_error())
        ));
    }
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        return Err(format!(
            "lowering error: {:?}",
            ld.iter().find(|d| d.is_error())
        ));
    }
    wukong_opt::optimize(&mut program, opt);
    for f in &program.funcs {
        let issues = wukong_mir::verify::verify_function(f);
        if !issues.is_empty() {
            return Err(format!("MIR verify failed at -O{opt}: {issues:?}"));
        }
    }
    let main = interner.intern("main");
    Ok(if native {
        crate::jit_run(&program, main, &interner)
    } else {
        wukong_interp::run_with_output(&program, main, &interner)
    })
}

/// Debugging aid: dump a few generated programs (`cargo test ... dump_generated -- --ignored
/// --nocapture`) to eyeball the grammar's coverage.
#[test]
#[ignore]
fn dump_generated_programs() {
    let base: u64 = 0x4D45_5243_5552_5931;
    for i in 0..3u64 {
        let seed = base ^ (i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        println!("=== program {i} (seed {seed}) ===\n{}", program(seed));
    }
}

/// The fuzzer proper: N seeded programs, each checked interp==native and -O0==-O2==-O3.
#[test]
fn fuzz_grammar_differential() {
    // 48 programs (~16 s) by default; WUKONG_FUZZ_PROGRAMS=N deepens a dedicated fuzz run
    // (120+ verified green). Seeds are fixed offsets from `base`, so N programs are always the
    // same N programs — a failure reproduces exactly from its printed seed.
    // `WUKONG_FUZZ_PROGRAMS=0` used to report `ok` in 0.00 s having compiled and run nothing —
    // verified — which is exactly what a "turn the slow fuzzer off" gesture in a CI job would do.
    // A zero (or unparseable) override falls back to the default instead of silently disabling it.
    let programs: u64 = std::env::var("WUKONG_FUZZ_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(48);
    // How many programs actually had two executions compared. Asserted against `programs` below: if
    // every execution of a program errors, `first_ok` is `None`, the comparison arm is empty, and the
    // fuzzer reports `ok` without having checked one exit code or one stdout byte.
    let mut compared = 0u64;
    let base: u64 = 0x4D45_5243_5552_5931; // fixed tag so failures reproduce.
    for i in 0..programs {
        let seed = base ^ (i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let src = program(seed);
        let fail = |what: &str, detail: String| -> ! {
            panic!(
                "grammar fuzzer: {what} (seed {seed}, program {i})\n--- detail ---\n{detail}\n--- source ---\n{src}"
            )
        };
        // Compile+run the matrix. A compile error at any level is a finding.
        let i0 = match run_at(&src, 0, false) {
            Ok(r) => r,
            Err(e) => fail("did not compile at -O0", e),
        };
        let i2 = match run_at(&src, 2, false) {
            Ok(r) => r,
            Err(e) => fail("did not compile at -O2", e),
        };
        let i3 = match run_at(&src, 3, false) {
            Ok(r) => r,
            Err(e) => fail("did not compile at -O3", e),
        };
        let n0 = match run_at(&src, 0, true) {
            Ok(r) => r,
            Err(e) => fail("did not compile at -O0 (native)", e),
        };
        let n3 = match run_at(&src, 3, true) {
            Ok(r) => r,
            Err(e) => fail("did not compile at -O3 (native)", e),
        };
        // All five executions must agree on (exit, stdout) — or all fail.
        let outs = [&i0, &i2, &i3, &n0, &n3];
        let names = [
            "interp -O0",
            "interp -O2",
            "interp -O3",
            "native -O0",
            "native -O3",
        ];
        let first_ok = outs.iter().position(|r| r.is_ok());
        match first_ok {
            None => {} // all five failed — counted as uncompared by the assert after the loop.
            Some(k) => {
                compared += 1;
                let golden = outs[k].as_ref().unwrap();
                for (r, name) in outs.iter().zip(names) {
                    match r {
                        Err(e) => fail(
                            &format!("{name} failed while {} succeeded", names[k]),
                            e.clone(),
                        ),
                        Ok(got) if got != golden => fail(
                            &format!("{name} != {}", names[k]),
                            format!(
                                "{name}: exit={} stdout={:?}\n{}: exit={} stdout={:?}",
                                got.0,
                                String::from_utf8_lossy(&got.1),
                                names[k],
                                golden.0,
                                String::from_utf8_lossy(&golden.1)
                            ),
                        ),
                        Ok(_) => {}
                    }
                }
            }
        }
    }
    assert_eq!(
        compared, programs,
        "grammar fuzzer compared {compared} of {programs} programs — every execution of the rest \
         errored, so the corpus or the runner is broken, not the programs"
    );
}
