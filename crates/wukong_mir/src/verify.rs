//! The MIR verifier — checks well-formedness invariants. A failure here is an *internal compiler
//! error* (a bug in a pass), not a user error, so violations are returned as plain strings.
//!
//! It runs before every backend entry and before printing MIR (`wukong_driver::verify_or_ice`, which
//! gates `--run` and `--emit=llvm-ir`/`obj`/`exe`, plus `emit_mir` after the dump so a broken program
//! still shows why), and after each optimization pass that ran — but that per-pass "verify-each" is a
//! `#[cfg(debug_assertions)]` block inside `wukong_opt::optimize`, **not** a CLI flag: there is no
//! `--verify-each` option and the check is compiled out of release, so a release-only invalid form can
//! slip past it.
//!
//! Checks: every value is defined exactly once, inside the value arena, and
//! before — in the dominance order — every use of it; the entry block matches the function's
//! parameters and has no predecessors; block ids match their index; block-parameter arities and types
//! line up across edges; per-operation operand/result types are consistent; a `veckernel` index is in
//! range and its result presence matches the recipe's kind; and terminators are type-correct
//! (including the function return type). Given a whole [`Program`], [`verify_program`] additionally
//! checks each call against its callee's declared signature — but only [`verify_function`] is wired
//! into the pipeline, so that call-vs-callee cross-check runs in tests only (see its doc).

use std::collections::{HashMap, HashSet};

use crate::{BinOp, Function, MirType, Op, Program, Terminator, ValueId};
use wukong_span::Symbol;

/// Declared signatures (parameter types, return type) of the functions defined in a program, so a
/// call can be checked against the function it names. Only callees with a MIR body are in here:
/// `print` and the `wukong_*` runtime kernels are external symbols whose signatures live outside
/// the IR, and a call to one is left unchecked.
type FnSigs = HashMap<Symbol, (Vec<MirType>, MirType)>;

/// Verify every function in a program. Returns a list of human-readable problems (empty = ok).
/// Unlike [`verify_function`] this also cross-checks each call against its callee's signature,
/// which needs the whole program in hand.
///
/// LANDMINE: no compiler path calls this. `wukong_driver::verify_or_ice` and `wukong_opt`'s
/// debug-only per-pass check both loop over the functions calling [`verify_function`], so the
/// call-vs-callee signature cross-check exists only in this crate's `mod tests` and one
/// `wukong_interp` test. Change a callee's MIR signature and nothing in the pipeline objects — the
/// symptom is an interp/native divergence with a silent zero on the oracle side, so run the
/// differential gate by hand.
pub fn verify_program(p: &Program) -> Vec<String> {
    let mut sigs: FnSigs = HashMap::new();
    for f in &p.funcs {
        // A parameter outside the value arena is reported separately; skip registering such a
        // function rather than indexing past the end here.
        let params: Option<Vec<MirType>> = f
            .params
            .iter()
            .map(|v| f.value_types.get(v.0 as usize).cloned())
            .collect();
        if let Some(params) = params {
            // `Program::function` resolves a name to the FIRST function carrying it; mirror that.
            sigs.entry(f.name).or_insert((params, f.ret.clone()));
        }
    }
    let mut errors = Vec::new();
    for f in &p.funcs {
        errors.extend(verify(f, Some(&sigs)));
    }
    errors
}

/// Verify a single function. This is the entry point the pipeline uses (`wukong_driver::verify_or_ice`
/// and `wukong_opt`'s debug-only per-pass check both call it per function). Calls are checked for
/// operand definition only — cross-checking a call against its callee needs [`verify_program`].
pub fn verify_function(f: &Function) -> Vec<String> {
    verify(f, None)
}

fn verify(f: &Function, sigs: Option<&FnSigs>) -> Vec<String> {
    let mut v = Verifier {
        f,
        sigs,
        errors: Vec::new(),
        defined: HashSet::new(),
        live: HashSet::new(),
        dom_check: false,
    };
    v.run();
    v.errors
}

struct Verifier<'a> {
    f: &'a Function,
    /// Callee signatures, when the verifier was entered with a whole program.
    sigs: Option<&'a FnSigs>,
    errors: Vec<String>,
    defined: HashSet<u32>,
    /// Values whose definition dominates the point currently being checked. Maintained only while
    /// walking the reachable blocks in dominator-tree preorder; see [`Verifier::walk_dominated`].
    live: HashSet<u32>,
    /// Whether `live` is meaningful right now (false while checking unreachable blocks).
    dom_check: bool,
}

/// The at-most-two blocks a terminator can transfer control to.
fn successors(t: &Terminator) -> [Option<u32>; 2] {
    match t {
        Terminator::Br { target, .. } => [Some(target.0), None],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => [Some(then_blk.0), Some(else_blk.0)],
        Terminator::Ret(_) | Terminator::Unreachable => [None, None],
    }
}

/// Sentinel for "no immediate dominator yet" in the Cooper–Harvey–Kennedy fixpoint below.
const NO_IDOM: u32 = u32::MAX;

/// Walk up the partially built dominator tree from two nodes until they meet.
fn intersect(mut a: u32, mut b: u32, idom: &[u32], rpo_num: &[usize]) -> u32 {
    while a != b {
        while rpo_num[a as usize] > rpo_num[b as usize] {
            a = idom[a as usize];
        }
        while rpo_num[b as usize] > rpo_num[a as usize] {
            b = idom[b as usize];
        }
    }
    a
}

impl Verifier<'_> {
    fn err(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    fn run(&mut self) {
        let f = self.f;
        let nblocks = f.blocks.len() as u32;

        // Collect all defined values (block params + instruction results), checking as we go that
        // each is defined exactly once and lies inside the value arena. Both are load-bearing:
        // Cranelift's `vmap` keeps whichever definition RPO lowered last, and `Function::value_type`
        // and the printer index `value_types` directly.
        for b in &f.blocks {
            for p in &b.params {
                self.define(*p);
            }
            for inst in &b.insts {
                if let Some(r) = inst.result {
                    self.define(r);
                }
            }
        }

        // Both backends bind the entry block's parameters by zipping them against `Function::params`
        // and both zips truncate silently (Cranelift then aborts on the first unbound value, the
        // interpreter reads it as 0), so the two lists must agree.
        match f.blocks.get(f.entry.0 as usize) {
            None => self.err(format!("entry block bb{} does not exist", f.entry.0)),
            Some(entry) if entry.params != f.params => self.err(
                "the entry block's parameters do not match the function's parameters".to_string(),
            ),
            Some(_) => {}
        }

        for (bi, b) in f.blocks.iter().enumerate() {
            if b.id.0 != bi as u32 {
                self.err(format!("block {} has inconsistent id {}", bi, b.id.0));
            }
        }

        // Check the reachable blocks in dominator-tree preorder so a use can be required to be
        // dominated by its definition, then the rest with that rule off: nothing lowers a block the
        // entry cannot reach (Cranelift's RPO skips it, the interpreter never enters it), so
        // dominance is vacuous there — but its operand and result types are still checked.
        let reachable = self.walk_dominated(nblocks);
        for (bi, b) in f.blocks.iter().enumerate() {
            if reachable[bi] {
                continue;
            }
            for inst in &b.insts {
                self.check_op(&inst.op, inst.result);
            }
            self.check_term(&b.term, nblocks);
        }
    }

    /// Walk the blocks reachable from the entry in dominator-tree preorder, checking each and
    /// carrying `live` = the definitions that dominate the current point. Returns reachability.
    fn walk_dominated(&mut self, nblocks: u32) -> Vec<bool> {
        let f = self.f;
        let n = f.blocks.len();
        let entry = f.entry.0 as usize;
        if entry >= n {
            return vec![false; n];
        }

        // Depth-first search from the entry: postorder (reversed below) and the predecessor map,
        // both restricted to the reachable subgraph.
        let mut visited = vec![false; n];
        let mut post: Vec<u32> = Vec::with_capacity(n);
        let mut stack: Vec<(u32, usize)> = vec![(entry as u32, 0)];
        visited[entry] = true;
        while let Some(&mut (b, i)) = stack.last_mut() {
            if i >= 2 {
                post.push(b);
                stack.pop();
                continue;
            }
            stack.last_mut().expect("stack is non-empty here").1 += 1;
            if let Some(s) = successors(&f.blocks[b as usize].term)[i] {
                if (s as usize) < n && !visited[s as usize] {
                    visited[s as usize] = true;
                    stack.push((s, 0));
                }
            }
        }
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (b, blk) in f.blocks.iter().enumerate() {
            if !visited[b] {
                continue;
            }
            for s in successors(&blk.term).into_iter().flatten() {
                if (s as usize) < n {
                    preds[s as usize].push(b as u32);
                }
            }
        }
        // `Builder::alloca` parks stack slots in the entry block because it runs exactly once, and
        // mem2reg never places a merge parameter there only because nothing branches back to it.
        if !preds[entry].is_empty() {
            self.err(format!(
                "the entry block bb{entry} has {} predecessor(s); it must have none",
                preds[entry].len()
            ));
        }

        // Immediate dominators over the reachable subgraph (Cooper–Harvey–Kennedy), then the
        // dominator tree's children in ascending block order so the walk is deterministic.
        let mut rpo_num = vec![usize::MAX; n];
        for (i, &b) in post.iter().rev().enumerate() {
            rpo_num[b as usize] = i;
        }
        let mut idom = vec![NO_IDOM; n];
        idom[entry] = entry as u32;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in post.iter().rev() {
                if b as usize == entry {
                    continue;
                }
                let mut new_idom = NO_IDOM;
                for &p in &preds[b as usize] {
                    if rpo_num[p as usize] == usize::MAX || idom[p as usize] == NO_IDOM {
                        continue;
                    }
                    new_idom = if new_idom == NO_IDOM {
                        p
                    } else {
                        intersect(p, new_idom, &idom, &rpo_num)
                    };
                }
                if new_idom != NO_IDOM && idom[b as usize] != new_idom {
                    idom[b as usize] = new_idom;
                    changed = true;
                }
            }
        }
        let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
        for b in 0..n as u32 {
            let id = idom[b as usize];
            if id != NO_IDOM && id != b {
                children[id as usize].push(b);
            }
        }

        // Preorder walk with an explicit stack (a recursive one would be as deep as the block chain
        // of a long function). `undo` records what this subtree added to `live` so exiting a node
        // restores exactly the state its siblings must see.
        enum Step {
            Enter(u32),
            Exit(usize),
        }
        let mut undo: Vec<u32> = Vec::new();
        let mut work = vec![Step::Enter(entry as u32)];
        self.dom_check = true;
        while let Some(step) = work.pop() {
            match step {
                Step::Exit(mark) => {
                    while undo.len() > mark {
                        let v = undo.pop().expect("undo mark is a prefix");
                        self.live.remove(&v);
                    }
                }
                Step::Enter(b) => {
                    let blk = &f.blocks[b as usize];
                    work.push(Step::Exit(undo.len()));
                    for p in &blk.params {
                        if self.live.insert(p.0) {
                            undo.push(p.0);
                        }
                    }
                    for inst in &blk.insts {
                        // Check before defining, so a use of the instruction's own result — or of
                        // anything later in this block — is reported.
                        self.check_op(&inst.op, inst.result);
                        if let Some(r) = inst.result {
                            if self.live.insert(r.0) {
                                undo.push(r.0);
                            }
                        }
                    }
                    self.check_term(&blk.term, nblocks);
                    for &c in children[b as usize].iter().rev() {
                        work.push(Step::Enter(c));
                    }
                }
            }
        }
        self.dom_check = false;
        visited
    }

    /// Record a definition, rejecting a second definition of the same value and one that lies past
    /// the end of the value arena.
    fn define(&mut self, v: ValueId) {
        if self.ty(v).is_none() {
            self.err(format!(
                "{v:?} is defined but lies outside the function's value arena"
            ));
        }
        if !self.defined.insert(v.0) {
            self.err(format!("value {v:?} is defined more than once"));
        }
    }

    fn ty(&self, v: ValueId) -> Option<&MirType> {
        self.f.value_types.get(v.0 as usize)
    }

    fn use_val(&mut self, v: ValueId) -> bool {
        if self.ty(v).is_none() {
            self.err(format!(
                "use of {v:?}, which lies outside the function's value arena"
            ));
            return false;
        }
        if !self.defined.contains(&v.0) {
            self.err(format!("use of undefined value {v:?}"));
            return false;
        }
        // The value exists and is typed, so the remaining checks are still worth running; only its
        // position is wrong.
        if self.dom_check && !self.live.contains(&v.0) {
            self.err(format!("use of {v:?} is not dominated by its definition"));
        }
        true
    }

    fn expect_ty(&mut self, v: ValueId, want: &MirType, ctx: &str) {
        if let Some(t) = self.ty(v) {
            if t != want {
                self.err(format!(
                    "{ctx}: {v:?} has type {} but expected {}",
                    t.display(),
                    want.display()
                ));
            }
        }
    }

    fn result_ty(&self, result: Option<ValueId>) -> Option<MirType> {
        result.and_then(|r| self.ty(r).cloned())
    }

    fn check_op(&mut self, op: &Op, result: Option<ValueId>) {
        // `Store` never produces a result; `Call` may be void (e.g. the `print` intrinsic) or
        // value-producing; `VecKernelCall` is void when elementwise and f32 when a reduction (checked
        // against the recipe below). Every other op must produce exactly one result.
        let must_produce = !matches!(op, Op::Store { .. } | Op::Call { .. } | Op::VecKernelCall { .. });
        if must_produce && result.is_none() {
            self.err(format!("operation {op:?} must produce a result value"));
        }
        if matches!(op, Op::Store { .. }) && result.is_some() {
            self.err("Store must not produce a result value".to_string());
        }

        match op {
            Op::ConstInt(_, ty) => {
                if !ty.is_int() {
                    self.err(format!("const int has non-integer type {}", ty.display()));
                }
                self.check_result_is(result, ty);
            }
            Op::ConstFloat(_, ty) => {
                if !ty.is_float() {
                    self.err(format!("const float has non-float type {}", ty.display()));
                }
                self.check_result_is(result, ty);
            }
            Op::Bin(b, l, r) => {
                if self.use_val(*l) & self.use_val(*r) {
                    if let Some(res) = self.result_ty(result) {
                        self.expect_ty(*l, &res, b.name());
                        self.expect_ty(*r, &res, b.name());
                        // Classify by lane type so vector arithmetic (`<4 x f32> fadd`) is checked
                        // against its `f32` lanes, not the aggregate `Vec` type.
                        let lane = res.lane_type();
                        let float_op = b.is_float();
                        if float_op && !lane.is_float() {
                            self.err(format!(
                                "float op {} on non-float type {}",
                                b.name(),
                                res.display()
                            ));
                        }
                        if !float_op
                            && !lane.is_int()
                            && !matches!(b, BinOp::Xor | BinOp::And | BinOp::Or)
                        {
                            self.err(format!(
                                "int op {} on non-int type {}",
                                b.name(),
                                res.display()
                            ));
                        }
                    }
                }
            }
            Op::Cmp(c, l, r) => {
                if self.use_val(*l) & self.use_val(*r) {
                    if let (Some(lt), Some(rt)) = (self.ty(*l).cloned(), self.ty(*r).cloned()) {
                        if lt != rt {
                            self.err(format!(
                                "cmp operands differ: {} vs {}",
                                lt.display(),
                                rt.display()
                            ));
                        }
                        // Classify by lane type so a `<4 x f32>` compare reads as a float compare.
                        if c.is_float() != lt.lane_type().is_float() {
                            self.err(format!(
                                "cmp predicate {} mismatches operand type {}",
                                c.name(),
                                lt.display()
                            ));
                        }
                        // A vector compare yields a same-width lane mask; a scalar compare yields i1.
                        if let MirType::Vec(_, n) = lt {
                            match self.result_ty(result) {
                                Some(MirType::Vec(_, m)) if m == n => {}
                                Some(other) => self.err(format!(
                                    "vector cmp result must be an {n}-lane mask, got {}",
                                    other.display()
                                )),
                                None => {}
                            }
                            return;
                        }
                    }
                }
                self.check_result_is(result, &MirType::I1);
            }
            Op::Neg(v) | Op::Not(v) => {
                if self.use_val(*v) {
                    if let Some(res) = self.result_ty(result) {
                        self.expect_ty(*v, &res, "neg/not");
                    }
                }
            }
            Op::Cast(_, v, to) => {
                self.use_val(*v);
                self.check_result_is(result, to);
            }
            Op::Select(c, a, b) => {
                if self.use_val(*a) & self.use_val(*b) {
                    if let Some(res) = self.result_ty(result) {
                        self.expect_ty(*a, &res, "select");
                        self.expect_ty(*b, &res, "select");
                        // A vector select takes a same-width lane mask; a scalar select takes i1.
                        if let MirType::Vec(_, n) = &res {
                            if self.use_val(*c) {
                                match self.ty(*c) {
                                    Some(MirType::Vec(_, m)) if m == n => {}
                                    Some(other) => {
                                        let other = other.display();
                                        self.err(format!(
                                            "vector select mask must be an {n}-lane vector, got {other}"
                                        ));
                                    }
                                    None => {}
                                }
                            }
                            return;
                        }
                    }
                }
                if self.use_val(*c) {
                    self.expect_ty(*c, &MirType::I1, "select condition");
                }
            }
            Op::Alloca(_) => self.check_result_is(result, &MirType::Ptr),
            Op::Load(p, ty) => {
                if self.use_val(*p) {
                    self.expect_ty(*p, &MirType::Ptr, "load pointer");
                }
                self.check_result_is(result, ty);
            }
            Op::Store { ptr, value } => {
                if self.use_val(*ptr) {
                    self.expect_ty(*ptr, &MirType::Ptr, "store pointer");
                }
                self.use_val(*value);
            }
            Op::Gep { ptr, index, .. } => {
                if self.use_val(*ptr) {
                    self.expect_ty(*ptr, &MirType::Ptr, "gep base");
                }
                if self.use_val(*index) {
                    if let Some(t) = self.ty(*index) {
                        if !t.is_int() {
                            self.err(format!("gep index has non-int type {}", t.display()));
                        }
                    }
                }
                self.check_result_is(result, &MirType::Ptr);
            }
            Op::Call { func, args } => {
                for a in args {
                    self.use_val(*a);
                }
                // Both backends consume a call's arity and types as fact, and both truncate
                // silently on mismatch: the interpreter zips the callee's params with the args, so
                // surplus parameters keep reading as 0, while Cranelift builds the call against the
                // callee's signature and rejects or panics inside its own IR. An ABI desync is
                // therefore an interp-vs-native divergence with a silent zero on the oracle side.
                let sig = self.sigs.and_then(|s| s.get(func)).cloned();
                if let Some((params, ret)) = sig {
                    if params.len() != args.len() {
                        self.err(format!(
                            "call to {func:?} passes {} args but the function has {} parameters",
                            args.len(),
                            params.len()
                        ));
                    } else {
                        for (i, (a, pt)) in args.iter().zip(&params).enumerate() {
                            self.expect_ty(*a, pt, &format!("call argument {i} to {func:?}"));
                        }
                    }
                    // Discarding a returned value is legal (`result` absent), but taking one from a
                    // `void` callee is not, and a taken result must have the callee's return type.
                    if ret == MirType::Void {
                        if result.is_some() {
                            self.err(format!(
                                "call to {func:?} takes a result but the function returns void"
                            ));
                        }
                    } else {
                        self.check_result_is(result, &ret);
                    }
                }
            }
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            } => {
                // `kernel` is a bare index into the *owning* function's `vec_kernels`; nothing else
                // in the instruction can recover it, and Cranelift indexes its kernel table with it
                // unchecked (`kernel_refs[k]`, a slice panic on a stale index; the interpreter is the
                // gentler of the two — it `get`s and reports "unknown kernel index"). A pass that
                // re-parents the call into another function leaves it dangling, so range-check it
                // here — this is the one operand the type rules below cannot see.
                let kind = self
                    .f
                    .vec_kernels
                    .get(*kernel as usize)
                    .map(|k| k.reduce.is_some());
                match kind {
                    None => {
                        let have = self.f.vec_kernels.len();
                        self.err(format!(
                            "veckernel kernel index {kernel} is out of range \
                             (the function has {have} kernels)"
                        ));
                    }
                    // A reduction kernel returns its horizontal fold as `f32`; an elementwise one
                    // writes through its output streams and yields nothing. The two have different
                    // backend signatures, so the call's result must agree with the recipe.
                    Some(true) if result.is_none() => self.err(format!(
                        "veckernel #{kernel} is a reduction but the call takes no result"
                    )),
                    Some(true) => self.check_result_is(result, &MirType::F32),
                    Some(false) if result.is_some() => self.err(format!(
                        "veckernel #{kernel} is elementwise but the call takes a result"
                    )),
                    Some(false) => {}
                }
                if self.use_val(*ptrs) {
                    self.expect_ty(*ptrs, &MirType::Ptr, "veckernel ptrs");
                }
                if self.use_val(*scalars) {
                    self.expect_ty(*scalars, &MirType::Ptr, "veckernel scalars");
                }
                if self.use_val(*n) {
                    if let Some(t) = self.ty(*n) {
                        if !t.is_int() {
                            self.err(format!("veckernel n has non-int type {}", t.display()));
                        }
                    }
                }
            }
            Op::FuncAddr(_) => {
                self.check_result_is(result, &MirType::Ptr);
            }
            Op::GlobalAddr(_) => {
                self.check_result_is(result, &MirType::Ptr);
            }
            Op::Fma(a, b, c) => {
                let ok = self.use_val(*a) & self.use_val(*b) & self.use_val(*c);
                if let Some(res) = self.result_ty(result) {
                    if !res.lane_type().is_float() {
                        self.err(format!("fma on non-float type {}", res.display()));
                    }
                    if ok {
                        self.expect_ty(*a, &res, "fma");
                        self.expect_ty(*b, &res, "fma");
                        self.expect_ty(*c, &res, "fma");
                    }
                }
            }
            Op::Sqrt(a) => {
                let ok = self.use_val(*a);
                if let Some(res) = self.result_ty(result) {
                    if !res.lane_type().is_float() {
                        self.err(format!("sqrt on non-float type {}", res.display()));
                    }
                    if ok {
                        self.expect_ty(*a, &res, "sqrt");
                    }
                }
            }
            Op::Round(_, a) => {
                let ok = self.use_val(*a);
                if let Some(res) = self.result_ty(result) {
                    if !res.lane_type().is_float() {
                        self.err(format!("round on non-float type {}", res.display()));
                    }
                    if ok {
                        self.expect_ty(*a, &res, "round");
                    }
                }
            }
            Op::Splat(v) => {
                if self.use_val(*v) {
                    if let (Some(vt), Some(res)) = (self.ty(*v).cloned(), self.result_ty(result)) {
                        match &res {
                            MirType::Vec(lane, _) if **lane == vt => {}
                            MirType::Vec(lane, _) => self.err(format!(
                                "splat: operand {} does not match result lane type {}",
                                vt.display(),
                                lane.display()
                            )),
                            _ => self.err(format!(
                                "splat result must be a vector, got {}",
                                res.display()
                            )),
                        }
                    }
                }
            }
            // The operand must really be a vector and the index must really be in range: the
            // interpreter indexes its lane arena directly, so an out-of-range lane there would be a
            // panic rather than a diagnostic, and a scalar operand would read a lane that does not
            // exist.
            Op::ExtractLane(v, k) => {
                if self.use_val(*v) {
                    if let (Some(vt), Some(res)) = (self.ty(*v).cloned(), self.result_ty(result)) {
                        match &vt {
                            MirType::Vec(lane, n) => {
                                if *k >= *n {
                                    self.err(format!(
                                        "extractlane index {k} is out of range for {}",
                                        vt.display()
                                    ));
                                }
                                if **lane != res {
                                    self.err(format!(
                                        "extractlane of {} must yield {}, got {}",
                                        vt.display(),
                                        lane.display(),
                                        res.display()
                                    ));
                                }
                            }
                            _ => self.err(format!(
                                "extractlane operand must be a vector, got {}",
                                vt.display()
                            )),
                        }
                    }
                }
            }
            // The lane ramp. Its type is carried on the op *and* is the result type, so the two are
            // checked against each other: a mismatch would have the interpreter build one lane count
            // and the backend read another. Integer lanes only — a float ramp has no use and would
            // need a float-literal constant pool entry in the Cranelift path.
            Op::Iota(ty) => {
                if let Some(res) = self.result_ty(result) {
                    if res != *ty {
                        self.err(format!(
                            "iota {} must yield {}, got {}",
                            ty.display(),
                            ty.display(),
                            res.display()
                        ));
                    }
                }
                match ty {
                    MirType::Vec(lane, n) if lane.is_int() && **lane != MirType::I1 && *n > 0 => {}
                    _ => self.err(format!(
                        "iota must be a vector of integer lanes, got {}",
                        ty.display()
                    )),
                }
            }
        }
    }

    fn check_result_is(&mut self, result: Option<ValueId>, want: &MirType) {
        if let Some(r) = result {
            if let Some(t) = self.ty(r) {
                if t != want {
                    self.err(format!(
                        "result {r:?} has type {} but operation yields {}",
                        t.display(),
                        want.display()
                    ));
                }
            }
        }
    }

    fn check_term(&mut self, t: &Terminator, nblocks: u32) {
        match t {
            Terminator::Ret(Some(v)) => {
                if self.use_val(*v) {
                    self.expect_ty(*v, &self.f.ret.clone(), "return value");
                }
            }
            Terminator::Ret(None) => {
                if self.f.ret != MirType::Void {
                    self.err(format!(
                        "`ret` with no value but function returns {}",
                        self.f.ret.display()
                    ));
                }
            }
            Terminator::Br { target, args } => {
                self.check_edge(*target, args, nblocks);
            }
            Terminator::CondBr {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                if self.use_val(*cond) {
                    self.expect_ty(*cond, &MirType::I1, "cond_br condition");
                }
                self.check_edge(*then_blk, then_args, nblocks);
                self.check_edge(*else_blk, else_args, nblocks);
            }
            Terminator::Unreachable => {}
        }
    }

    fn check_edge(&mut self, target: crate::BlockId, args: &[ValueId], nblocks: u32) {
        if target.0 >= nblocks {
            self.err(format!("branch to nonexistent block bb{}", target.0));
            return;
        }
        let tparams = self.f.blocks[target.0 as usize].params.clone();
        if tparams.len() != args.len() {
            self.err(format!(
                "branch to bb{} passes {} args but the block has {} parameters",
                target.0,
                args.len(),
                tparams.len()
            ));
            return;
        }
        for (a, p) in args.iter().zip(&tparams) {
            if self.use_val(*a) {
                if let (Some(at), Some(pt)) = (self.ty(*a).cloned(), self.ty(*p).cloned()) {
                    if at != pt {
                        self.err(format!(
                            "branch arg {a:?} ({}) does not match bb{} param {p:?} ({})",
                            at.display(),
                            target.0,
                            pt.display()
                        ));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinOp, Builder, MirType, Op};
    use wukong_span::Interner;

    #[test]
    fn valid_function_verifies() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, one));
        b.ret(Some(y));
        assert!(verify_function(&b.finish()).is_empty());
    }

    #[test]
    fn detects_type_mismatch() {
        // add an i32 and an i64 -> operand type mismatch against the i32 result.
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let big = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, big));
        b.ret(Some(y));
        assert!(!verify_function(&b.finish()).is_empty());
    }

    #[test]
    fn void_call_verifies() {
        // A result-less call (e.g. the `print` intrinsic) is valid and must not be flagged as
        // "must produce a result" — regression for the void-call verifier rule.
        use crate::Op;
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("main"), MirType::Void);
        let arg = b.build(MirType::I32, Op::ConstInt(42, MirType::I32));
        b.build_void(Op::Call {
            func: i.intern("print"),
            args: vec![arg],
        });
        b.ret(None);
        assert!(verify_function(&b.finish()).is_empty());
    }

    #[test]
    fn detects_block_param_arity() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        let target = b.new_block();
        let _p = b.block_param(target, MirType::I32);
        // branch with no args to a block expecting one parameter.
        b.br(target, vec![]);
        b.switch_to(target);
        b.ret(None);
        let errs = verify_function(&b.finish());
        assert!(errs.iter().any(|e| e.contains("parameters")), "{errs:?}");
    }

    /// `kernel` is a bare index into the *owning* function's `vec_kernels`; nothing else in the
    /// instruction can recover it, and Cranelift indexes `kernel_refs` with it unchecked. A pass
    /// that re-parents the call into a function that owns no kernels leaves it dangling.
    #[test]
    fn detects_veckernel_index_out_of_range() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        let ptrs = b.alloca(MirType::Ptr);
        let scalars = b.alloca(MirType::F32);
        let n = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        b.build_void(Op::VecKernelCall {
            kernel: 0,
            ptrs,
            scalars,
            n,
        });
        b.ret(None);
        // The function registered no kernels, so `#0` refers to nothing.
        let errs = verify_function(&b.finish());
        assert!(
            errs.iter().any(|e| e.contains("out of range")),
            "{errs:?}"
        );
    }

    // ---- rules the verifier already enforced but that nothing pinned ----

    #[test]
    fn detects_use_of_undefined_value() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        // A value that lives in the arena but that no block ever defines.
        let ghost = b.new_value(MirType::I32);
        b.ret(Some(x));
        let mut f = b.finish();
        f.blocks[0].term = Terminator::Ret(Some(ghost));
        let errs = verify_function(&f);
        assert!(
            errs.iter().any(|e| e.contains("undefined value")),
            "{errs:?}"
        );
    }

    #[test]
    fn detects_branch_to_nonexistent_block() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        b.br(crate::BlockId(99), vec![]);
        let errs = verify_function(&b.finish());
        assert!(errs.iter().any(|e| e.contains("nonexistent")), "{errs:?}");
    }

    #[test]
    fn detects_block_with_inconsistent_id() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        let t = b.new_block();
        b.br(t, vec![]);
        b.switch_to(t);
        b.ret(None);
        let mut f = b.finish();
        f.blocks[1].id = crate::BlockId(7);
        let errs = verify_function(&f);
        assert!(
            errs.iter().any(|e| e.contains("inconsistent id")),
            "{errs:?}"
        );
    }

    #[test]
    fn detects_store_with_a_result() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        let slot = b.alloca(MirType::I32);
        let v = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        b.push(
            Some(MirType::I32),
            Op::Store {
                ptr: slot,
                value: v,
            },
        );
        b.ret(None);
        let errs = verify_function(&b.finish());
        assert!(
            errs.iter().any(|e| e.contains("Store must not produce")),
            "{errs:?}"
        );
    }

    // ---- SSA structure: single definition, dominance, entry-block invariants ----

    /// A use whose definition lives in a sibling block is *defined somewhere*, so the flat-set check
    /// accepted it. The two backends then disagree: the interpreter reads its pre-initialized
    /// `Value::Unit` slot as 0, Cranelift aborts with "MIR value used before definition".
    #[test]
    fn detects_use_not_dominated_by_its_definition() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let c = b.add_param(MirType::I1);
        let t = b.new_block();
        let e = b.new_block();
        b.cond_br(c, t, vec![], e, vec![]);
        b.switch_to(t);
        let x = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        b.ret(Some(x));
        b.switch_to(e);
        // `x` is defined in `t`, which does not dominate `e`.
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, x));
        b.ret(Some(y));
        let errs = verify_function(&b.finish());
        assert!(errs.iter().any(|e| e.contains("not dominated")), "{errs:?}");
    }

    /// Within a block the definition must precede the use, not merely appear in the same block.
    #[test]
    fn detects_use_before_definition_in_the_same_block() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let x = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, x));
        b.ret(Some(y));
        let mut f = b.finish();
        f.blocks[0].insts.swap(0, 1); // the add now reads `x` before the const defines it
        let errs = verify_function(&f);
        assert!(errs.iter().any(|e| e.contains("not dominated")), "{errs:?}");
    }

    /// A natural loop (a back edge into a block with a parameter) must still verify — the dominance
    /// rule must not reject legitimate loop-carried values.
    #[test]
    fn loop_with_back_edge_still_verifies() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let n = b.add_param(MirType::I32);
        let head = b.new_block();
        let body = b.new_block();
        let done = b.new_block();
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        b.br(head, vec![zero]);
        let iv = b.block_param(head, MirType::I32);
        b.switch_to(head);
        let cmp = b.build(MirType::I1, Op::Cmp(crate::CmpOp::Slt, iv, n));
        b.cond_br(cmp, body, vec![], done, vec![]);
        b.switch_to(body);
        let next = b.build(MirType::I32, Op::Bin(BinOp::Add, iv, one));
        b.br(head, vec![next]);
        b.switch_to(done);
        b.ret(Some(iv));
        let errs = verify_function(&b.finish());
        assert!(errs.is_empty(), "{errs:?}");
    }

    /// Dominance is vacuous for a block the entry cannot reach: nothing lowers it (Cranelift's RPO
    /// skips it, the interpreter never enters it), so it must not be reported.
    #[test]
    fn unreachable_block_is_not_dominance_checked() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let dead = b.new_block();
        let x = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        b.ret(Some(x));
        b.switch_to(dead);
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, x));
        b.ret(Some(y));
        let errs = verify_function(&b.finish());
        assert!(errs.is_empty(), "{errs:?}");
    }

    #[test]
    fn detects_value_defined_more_than_once() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let a = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let _c = b.build(MirType::I32, Op::ConstInt(2, MirType::I32));
        b.ret(Some(a));
        let mut f = b.finish();
        f.blocks[0].insts[1].result = Some(a); // two instructions now define `a`
        let errs = verify_function(&f);
        assert!(
            errs.iter().any(|e| e.contains("defined more than once")),
            "{errs:?}"
        );
    }

    /// `Function::value_type` and the printer index `value_types` directly, so a value id past the
    /// end of the arena is a latent panic; `ty()` returned `None` and every type check passed.
    #[test]
    fn detects_definition_past_the_end_of_the_value_arena() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let _dead = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let y = b.build(MirType::I32, Op::ConstInt(2, MirType::I32));
        b.ret(Some(y));
        let mut f = b.finish();
        f.blocks[0].insts[0].result = Some(ValueId(999));
        let errs = verify_function(&f);
        assert!(errs.iter().any(|e| e.contains("arena")), "{errs:?}");
    }

    /// Both backends bind the entry block's parameters by zipping against `Function::params` and
    /// both zips truncate silently, so the two lists must agree.
    #[test]
    fn detects_entry_block_params_out_of_sync_with_function_params() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let _y = b.add_param(MirType::I32);
        b.ret(Some(x));
        let mut f = b.finish();
        f.params.pop(); // the entry block still has two parameters
        let errs = verify_function(&f);
        assert!(errs.iter().any(|e| e.contains("entry block")), "{errs:?}");
    }

    #[test]
    fn detects_branch_back_to_the_entry_block() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        let entry = b.entry();
        let body = b.new_block();
        b.br(body, vec![]);
        b.switch_to(body);
        b.br(entry, vec![]);
        let errs = verify_function(&b.finish());
        assert!(errs.iter().any(|e| e.contains("entry block")), "{errs:?}");
    }

    // ---- cross-function call checks (`verify_program`) ----

    fn two_ptr_void_callee(i: &mut Interner) -> Function {
        let mut b = Builder::new(i.intern("callee"), MirType::Void);
        let _a = b.add_param(MirType::Ptr);
        let _c = b.add_param(MirType::Ptr);
        b.ret(None);
        b.finish()
    }

    fn program_of(funcs: Vec<Function>) -> Program {
        Program {
            funcs,
            statics: Vec::new(),
            level: crate::MirLevel::Low,
        }
    }

    /// The exact shape a monomorphization-key collision produces: one argument passed to a
    /// two-parameter function, and a result taken from a `void` callee. Per-function verification
    /// accepted it; the interpreter then reported `expected a pointer` and the native backend
    /// aborted with `MIR value used before definition`.
    #[test]
    fn program_detects_call_arity_and_void_result_mismatch() {
        let mut i = Interner::new();
        let callee = two_ptr_void_callee(&mut i);
        let mut b = Builder::new(i.intern("main"), MirType::I32);
        let slot = b.alloca(MirType::I32);
        let bad = b.build(
            MirType::I32,
            Op::Call {
                func: i.intern("callee"),
                args: vec![slot],
            },
        );
        b.ret(Some(bad));
        let errs = verify_program(&program_of(vec![callee, b.finish()]));
        assert!(
            errs.iter().any(|e| e.contains("passes 1 args")),
            "arity: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("returns void")),
            "void result: {errs:?}"
        );
    }

    #[test]
    fn program_detects_call_argument_type_mismatch() {
        let mut i = Interner::new();
        let callee = two_ptr_void_callee(&mut i);
        let mut b = Builder::new(i.intern("main"), MirType::I32);
        let slot = b.alloca(MirType::I32);
        let wrong = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        b.build_void(Op::Call {
            func: i.intern("callee"),
            args: vec![slot, wrong],
        });
        let z = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(z));
        let errs = verify_program(&program_of(vec![callee, b.finish()]));
        assert!(
            errs.iter().any(|e| e.contains("call argument 1")),
            "{errs:?}"
        );
    }

    /// A call to a symbol with no MIR body (the `print` intrinsic, a `wukong_*` runtime kernel) is
    /// external and must be skipped, a well-formed internal call must stay clean, and discarding a
    /// non-void callee's result is legal.
    #[test]
    fn program_accepts_external_well_formed_and_discarded_calls() {
        let mut i = Interner::new();
        let callee = two_ptr_void_callee(&mut i);
        let mut vb = Builder::new(i.intern("answer"), MirType::I32);
        let a = vb.build(MirType::I32, Op::ConstInt(42, MirType::I32));
        vb.ret(Some(a));
        let answer = vb.finish();

        let mut b = Builder::new(i.intern("main"), MirType::I32);
        let p1 = b.alloca(MirType::I32);
        let p2 = b.alloca(MirType::I32);
        b.build_void(Op::Call {
            func: i.intern("callee"),
            args: vec![p1, p2],
        });
        b.build_void(Op::Call {
            func: i.intern("answer"),
            args: vec![],
        });
        let z = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.build_void(Op::Call {
            func: i.intern("print"),
            args: vec![z],
        });
        b.ret(Some(z));
        let p = program_of(vec![callee, answer, b.finish()]);
        assert!(verify_program(&p).is_empty(), "{:?}", verify_program(&p));
    }
}
