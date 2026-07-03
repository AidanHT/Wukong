//! Grammar-based **whole-program** differential fuzzer: random well-typed Mercury programs
//! through the REAL pipeline (parse → sema → mir_build → optimize → interp / Cranelift JIT),
//! asserting the two hard invariants on every one:
//!
//!   1. interp == native on (exit code, stdout), and
//!   2. -O0 == -O2 == -O3 on (exit code, stdout),
//!
//! Sixteen manual "gap-hunt sweeps" found real miscompiles by hand-writing probe programs in
//! exactly this space (scalar arithmetic, casts, control flow, arrays, calls, if-values). This
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

use mercury_span::{Interner, SourceId};

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

struct Gen {
    rng: Rng,
    vars: Vec<Var>,
    arrs: Vec<Arr>,
    /// Loop variables of enclosing `for i in 0..N` loops: (name, exclusive bound).
    loop_vars: Vec<(String, usize)>,
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
            loop_vars: Vec::new(),
            next_id: 0,
            depth: 0,
            src: String::new(),
            indent: 1,
            prints: 0,
        }
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
                    let op = ["<", "<=", ">", ">=", "==", "!="]
                        [self.rng.below(6) as usize];
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

    /// A leaf: variable, array element, or literal.
    fn leaf(&mut self, t: Ty) -> String {
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

    fn stmt(&mut self) {
        match self.rng.below(12) {
            // let
            0 | 1 | 2 => {
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
                let len = [4usize, 5, 8][self.rng.below(3) as usize];
                let init = self.literal(t);
                let name = self.fresh("a");
                self.line(format!("let mut {name}: [{}; {len}] = [{init}; {len}];", t.name()));
                self.arrs.push(Arr {
                    name,
                    elem: t,
                    len,
                });
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
                    self.line(format!("{}[{}] = (({e}) as {});", a.name, idx, a.elem.name()));
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
                let vars_mark = self.vars.len();
                let arrs_mark = self.arrs.len();
                let n = self.rng.below(3) + 1;
                for _ in 0..n {
                    self.stmt();
                }
                self.vars.truncate(vars_mark);
                self.arrs.truncate(arrs_mark);
                self.indent -= 1;
                if self.rng.chance(60) {
                    self.line("} else {".into());
                    self.indent += 1;
                    let n = self.rng.below(3) + 1;
                    for _ in 0..n {
                        self.stmt();
                    }
                    self.vars.truncate(vars_mark);
                    self.arrs.truncate(arrs_mark);
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
                let bound = [3usize, 4, 5][self.rng.below(3) as usize];
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
                let vars_mark = self.vars.len();
                let arrs_mark = self.arrs.len();
                let n = self.rng.below(3) + 1;
                for _ in 0..n {
                    self.stmt();
                }
                self.vars.truncate(vars_mark);
                self.arrs.truncate(arrs_mark);
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
                let vars_mark = self.vars.len();
                let arrs_mark = self.arrs.len();
                let n = self.rng.below(2) + 1;
                for _ in 0..n {
                    self.stmt();
                }
                self.vars.truncate(vars_mark);
                self.arrs.truncate(arrs_mark);
                self.line(format!("{c} = {c} + 1;"));
                self.vars.pop();
                self.indent -= 1;
                self.depth -= 1;
                self.line("}".into());
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

    fn program(mut self, seed: u64) -> String {
        self.src = format!("module fuzz.p{seed}\n\nfn main() -> i32 {{\n");
        let stmts = self.rng.below(18) + 8;
        for _ in 0..stmts {
            self.stmt();
            if self.rng.chance(30) {
                self.emit_print();
            }
        }
        // Guarantee a strong stdout signal even if the RNG never rolled a print.
        while self.prints < 3 {
            self.emit_print();
        }
        // Exit code: a small, always-valid projection.
        let ret = self.expr(Ty::I32, 2);
        self.line(format!("return (({ret}) & 63);"));
        self.src.push_str("}\n");
        self.src
    }
}

/// Compile `src` through the real pipeline at `opt`, then run it on the given backend.
/// `Err` means the program did not compile — a generator bug worth failing loudly on.
fn run_at(
    src: &str,
    opt: u8,
    native: bool,
) -> Result<Result<(i64, Vec<u8>), String>, String> {
    let mut interner = Interner::new();
    let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return Err(format!("parse error: {:?}", pd.iter().find(|d| d.is_error())));
    }
    let (sema, sd) = mercury_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return Err(format!(
            "sema error: {:?}",
            sd.iter().find(|d| d.is_error())
        ));
    }
    let (mut program, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        return Err(format!(
            "lowering error: {:?}",
            ld.iter().find(|d| d.is_error())
        ));
    }
    mercury_opt::optimize(&mut program, opt);
    for f in &program.funcs {
        let issues = mercury_mir::verify::verify_function(f);
        if !issues.is_empty() {
            return Err(format!("MIR verify failed at -O{opt}: {issues:?}"));
        }
    }
    let main = interner.intern("main");
    Ok(if native {
        crate::jit_run(&program, main, &interner)
    } else {
        mercury_interp::run_with_output(&program, main, &interner)
    })
}

/// The fuzzer proper: N seeded programs, each checked interp==native and -O0==-O2==-O3.
#[test]
fn fuzz_grammar_differential() {
    // 48 programs (~16 s) by default; MERCURY_FUZZ_PROGRAMS=N deepens a dedicated fuzz run
    // (120+ verified green). Seeds are fixed offsets from `base`, so N programs are always the
    // same N programs — a failure reproduces exactly from its printed seed.
    let programs: u64 = std::env::var("MERCURY_FUZZ_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let base: u64 = 0x4D45_5243_5552_5931; // fixed tag so failures reproduce.
    for i in 0..programs {
        let seed = base ^ (i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let src = Gen::new(seed).program(seed);
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
        let names = ["interp -O0", "interp -O2", "interp -O3", "native -O0", "native -O3"];
        let first_ok = outs.iter().position(|r| r.is_ok());
        match first_ok {
            None => {} // all five failed — consistent (e.g. a runtime trap); acceptable.
            Some(k) => {
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
}
