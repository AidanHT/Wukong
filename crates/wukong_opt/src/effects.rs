//! Interprocedural call effects: which user functions a call to can be value-numbered, deleted or
//! hoisted like an ordinary instruction.
//!
//! Without this every `Op::Call` is an opaque side effect. The inliner hides that for small
//! helpers, so the cost lands exactly on the functions too large to inline — the expensive ones:
//! `heavy(a) + heavy(a)` called `heavy` twice, a call whose result was never used survived, and a
//! loop recomputed an invariant `heavy(c)` every iteration.
//!
//! # The summary
//!
//! A function is summarized only when calling it has **no side effect** — LLVM's `readonly` /
//! `readnone`, here derived bottom-up over the call graph:
//!
//!  * every `store` writes this function's own stack frame (provenance [`Prov::Alloca`]), which
//!    dies with the call;
//!  * every call is to a function already summarized, or to a member of the function's own
//!    strongly-connected component (see below) — never to an external `print`/`wukong_*` symbol;
//!  * no barrier, no shared-memory slot, no element-wise vector kernel (it stores through its
//!    streams); a reduction kernel only reads;
//!  * it returns no pointer, compares no pointer and converts no pointer to an integer unless that
//!    pointer was handed in or names static data: anything else may expose an address in this
//!    call's frame, which differs from one call to the next.
//!
//! On top of that, a summary records whether the function **reads** memory its caller can observe
//! (a load not provably from its own frame or from read-only static data), whether it **always
//! returns** (no loop, no recursion, no `unreachable`, no store that may fall outside its slot —
//! the store `dse` refuses to delete for the same reason — and only callees that always return),
//! and whether it is **speculatable** — always returns, reads nothing, and holds nothing LICM would
//! refuse to speculate in its own loop: no integer division, and no load or store outside a known
//! in-bounds offset of its own frame (a frame slot indexed by an argument may fault for arguments a
//! zero-trip loop never passes).
//!
//! Recursion is handled the standard way for a property only a write can break: the members of one
//! call-graph SCC are assumed side-effect free while their calls to one another are examined (the
//! greatest fixpoint), and the assumption is then checked against every member's body. A recursive
//! function can therefore still be value-numbered; it is never "always returns".
//!
//! # How it is used
//!
//! | effect | consumer |
//! |---|---|
//! | any summary | `cse` keeps its memory table across the call; `licm` stops treating the call as a writer |
//! | reads nothing | `cse` value-numbers identical calls like any pure instruction |
//! | reads memory | `cse` forwards an identical call's result like a load, until something may write memory the callee can see |
//! | always returns | `dce` deletes the call once its result is unused |
//! | speculatable | `licm` hoists an invariant call into the preheader |
//!
//! The pass manager optimizes functions callees-first and summarizes each SCC from its *optimized*
//! bodies, so a caller sees its callees at their most analyzable. A function's semantics — and so
//! its summary — is preserved by every later transform, which is what makes it sound to use the
//! table while other functions are still being optimized.
//!
//! Every consumer asks the table about one symbol at a time, and only about a symbol an `Op::Call`
//! of the function being optimized names — never "what is in the table". That is what lets the
//! parallel schedule ([`crate::parallel`]) optimize a whole call-graph *level* at once against the
//! table as it stands after the levels below: each function's callees sit in lower levels, so their
//! entries are exactly the ones the one-SCC-at-a-time order would have shown it, and an entry for
//! anything else is never read.

use wukong_mir::{AddrSpace, CastKind, Function, MirType, Op, Program, Terminator, ValueId};
use wukong_span::Symbol;

use crate::alias::{AliasInfo, Prov};
use crate::fxhash::{FxHashMap, FxHashSet};

/// What calling one user function can do, as far as its caller is concerned. Only functions whose
/// calls have no side effect have one; see the module note.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct CallEffect {
    /// May read memory the caller can observe. `false`: the result is a function of the argument
    /// values alone.
    pub(crate) reads_memory: bool,
    /// Every call returns normally.
    pub(crate) will_return: bool,
    /// Safe to execute where the program would not have.
    pub(crate) speculatable: bool,
}

/// The summaries of the user functions proved free of side effects. A symbol absent from the table
/// — an impure function, an external runtime or `print` symbol — is opaque.
#[derive(Clone, Default)]
pub(crate) struct CallEffects {
    map: FxHashMap<Symbol, CallEffect>,
}

impl CallEffects {
    /// Summarize every function of `program` from its body as it stands now, callees first. The
    /// pass manager builds its table incrementally instead, as each SCC finishes optimizing; this
    /// is for a pass over functions whose callees are already done (the post-unroll cleanup).
    pub(crate) fn of(program: &Program) -> CallEffects {
        let graph = CallGraph::of(program);
        let mut table = CallEffects::default();
        for scc in 0..graph.sccs.len() {
            graph.summarize(program, scc, &mut table);
        }
        table
    }

    pub(crate) fn get(&self, func: Symbol) -> Option<CallEffect> {
        self.map.get(&func).copied()
    }

    /// May a call whose result is unused be deleted?
    pub(crate) fn removable(&self, func: Symbol) -> bool {
        self.get(func).is_some_and(|e| e.will_return)
    }

    /// May a call with loop-invariant arguments run in a preheader, even on a zero-trip entry?
    pub(crate) fn hoistable(&self, func: Symbol) -> bool {
        self.get(func).is_some_and(|e| e.speculatable)
    }
}

/// The program's call graph, condensed into strongly-connected components in callees-first order.
pub(crate) struct CallGraph {
    /// Function indices per SCC; every SCC appears after every SCC it calls.
    pub(crate) sccs: Vec<Vec<u32>>,
    /// Per SCC, the length of the longest chain of calls out of it into other SCCs: 0 for an SCC
    /// that calls none, otherwise one more than the deepest SCC it calls. Two SCCs of one level
    /// never call each other, directly or not, so once every lower level is summarized their
    /// members can be optimized in any order — or at once ([`CallGraph::levels`]).
    level: Vec<u32>,
    /// Whether a function outside the SCC calls one of its members — the only case in which its
    /// summary is ever consulted, so the only case worth computing it.
    called: Vec<bool>,
    /// Names carried by more than one function. A call names a function by symbol, so such a name
    /// is never summarized.
    duplicated: FxHashSet<Symbol>,
}

impl CallGraph {
    pub(crate) fn of(program: &Program) -> CallGraph {
        let mut index: FxHashMap<Symbol, u32> = FxHashMap::default();
        let mut duplicated = FxHashSet::default();
        for (i, f) in program.funcs.iter().enumerate() {
            if index.insert(f.name, i as u32).is_some() {
                duplicated.insert(f.name);
            }
        }
        let adj: Vec<Vec<u32>> = program
            .funcs
            .iter()
            .map(|f| {
                let mut out: Vec<u32> = Vec::new();
                for inst in f.blocks.iter().flat_map(|b| &b.insts) {
                    if let Op::Call { func, .. } = &inst.op {
                        if let Some(&c) = index.get(func) {
                            if !out.contains(&c) {
                                out.push(c);
                            }
                        }
                    }
                }
                out
            })
            .collect();
        let sccs = crate::inline::strongly_connected(&adj);
        let mut comp = vec![0usize; adj.len()];
        for (c, members) in sccs.iter().enumerate() {
            for &m in members {
                comp[m as usize] = c;
            }
        }
        let mut called = vec![false; sccs.len()];
        for (caller, callees) in adj.iter().enumerate() {
            for &callee in callees {
                if comp[callee as usize] != comp[caller] {
                    called[comp[callee as usize]] = true;
                }
            }
        }
        // Callees first, so every SCC a component calls already has its level.
        let mut level = vec![0u32; sccs.len()];
        for (c, members) in sccs.iter().enumerate() {
            for &m in members {
                for &callee in &adj[m as usize] {
                    let d = comp[callee as usize];
                    if d != c {
                        level[c] = level[c].max(level[d] + 1);
                    }
                }
            }
        }
        CallGraph {
            sccs,
            level,
            called,
            duplicated,
        }
    }

    /// The SCC indices grouped by [`CallGraph::level`], lowest level first, each group in SCC
    /// order. Every SCC an SCC calls is in an earlier group.
    pub(crate) fn levels(&self) -> Vec<Vec<usize>> {
        let depth = self.level.iter().max().map_or(0, |&l| l as usize + 1);
        let mut groups = vec![Vec::new(); depth];
        for (scc, &l) in self.level.iter().enumerate() {
            groups[l as usize].push(scc);
        }
        groups
    }

    /// Summarize SCC `scc` from its members' current bodies and record the result in `table`. A
    /// no-op when nothing outside the SCC calls it, or when any member has a side effect.
    pub(crate) fn summarize(&self, program: &Program, scc: usize, table: &mut CallEffects) {
        if !self.called[scc] {
            return;
        }
        let members: Vec<Symbol> = self.sccs[scc]
            .iter()
            .map(|&i| program.funcs[i as usize].name)
            .collect();
        if members.iter().any(|m| self.duplicated.contains(m)) {
            return;
        }
        let mut total = Local {
            reads: false,
            returns: true,
            speculatable: true,
            recursive: members.len() > 1,
        };
        for &i in &self.sccs[scc] {
            let Some(local) = classify(&program.funcs[i as usize], table, &members) else {
                return;
            };
            total.reads |= local.reads;
            total.returns &= local.returns;
            total.speculatable &= local.speculatable;
            total.recursive |= local.recursive;
        }
        let will_return = total.returns && !total.recursive;
        let effect = CallEffect {
            reads_memory: total.reads,
            will_return,
            speculatable: will_return && !total.reads && total.speculatable,
        };
        for name in members {
            table.map.insert(name, effect);
        }
    }
}

/// One function's own contribution to its SCC's summary.
#[derive(Clone, Copy)]
struct Local {
    reads: bool,
    returns: bool,
    speculatable: bool,
    recursive: bool,
}

/// Classify `f`, or `None` when calling it may have a side effect. Calls to members of `scc` are
/// assumed side-effect free (the caller checks every member).
fn classify(f: &Function, table: &CallEffects, scc: &[Symbol]) -> Option<Local> {
    // A returned pointer may name this call's own frame, which is different on every call.
    if f.ret == MirType::Ptr {
        return None;
    }
    let mut local = Local {
        reads: false,
        returns: true,
        speculatable: true,
        recursive: false,
    };
    let mut facts: Option<AliasInfo> = None;
    // A pointer handed in, or the address of static data: the same on every call with these
    // arguments, unlike anything in the frame.
    let stable = |prov: Prov| matches!(prov, Prov::Param(_) | Prov::Global(_));
    // Could this access fault where the program would not have made it? Only an access LICM itself
    // would speculate — inside this function's own slot at a known in-bounds offset — is safe to
    // run from a preheader on a zero-trip entry; a frame slot indexed by an argument is not.
    let in_bounds = |alias: &AliasInfo, ptr: ValueId, op: &Op| {
        AliasInfo::access_bytes(f, op).is_some_and(|bytes| alias.is_dereferenceable(ptr, bytes))
    };
    for b in &f.blocks {
        if matches!(b.term, Terminator::Unreachable) {
            local.returns = false;
        }
        for inst in &b.insts {
            match &inst.op {
                Op::Store { ptr, .. } => {
                    let alias: &AliasInfo = facts.get_or_insert_with(|| AliasInfo::analyze(f));
                    if !matches!(alias.prov(*ptr), Prov::Alloca(_)) {
                        return None;
                    }
                    // A store `dse` would keep for the same reason: at an argument-chosen offset
                    // it may fault, and deleting the call must not hide that.
                    if !in_bounds(alias, *ptr, &inst.op) {
                        local.returns = false;
                    }
                }
                Op::Load(ptr, _) => {
                    let alias: &AliasInfo = facts.get_or_insert_with(|| AliasInfo::analyze(f));
                    if !matches!(alias.prov(*ptr), Prov::Alloca(_) | Prov::Global(_)) {
                        local.reads = true;
                    }
                    if !in_bounds(alias, *ptr, &inst.op) {
                        local.speculatable = false;
                    }
                }
                Op::Cast(CastKind::PtrToInt, v, _) => {
                    let alias = facts.get_or_insert_with(|| AliasInfo::analyze(f));
                    if !stable(alias.prov(*v)) {
                        return None;
                    }
                }
                Op::Cmp(_, a, c) if *f.value_type(*a) == MirType::Ptr => {
                    let alias = facts.get_or_insert_with(|| AliasInfo::analyze(f));
                    if !stable(alias.prov(*a)) || !stable(alias.prov(*c)) {
                        return None;
                    }
                }
                Op::Call { func, .. } => {
                    if scc.contains(func) {
                        local.recursive = true;
                        continue;
                    }
                    let effect = table.get(*func)?;
                    local.reads |= effect.reads_memory;
                    local.returns &= effect.will_return;
                    local.speculatable &= effect.speculatable;
                }
                Op::VecKernelCall { kernel, .. } => match f.vec_kernels.get(*kernel as usize) {
                    // A reduction recipe only reads its streams; an element-wise one writes them.
                    Some(recipe) if recipe.reduce.is_some() => local.reads = true,
                    _ => return None,
                },
                Op::Alloca {
                    space: AddrSpace::Shared,
                    ..
                } => return None,
                // Every other effect is a side effect here (`Op::effects`): an atomic RMW (a write
                // wherever it points, which exists to publish to other threads), a barrier, any
                // other kind of call. Without this arm an atomics-only callee would be summarized
                // as pure and its unused call deleted.
                op if op.may_read_memory() || op.has_side_effects() => return None,
                // An integer division or remainder traps on a zero divisor.
                op if op.may_trap() => local.speculatable = false,
                _ => {}
            }
        }
    }
    if crate::inline::loop_blocks(f).iter().any(|&in_loop| in_loop) {
        local.returns = false;
    }
    Some(local)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{BinOp, Builder};
    use wukong_span::Interner;

    /// `leaf(x) = x * x` (pure), `reader(p) = *p` (reads), `writer(p) = { *p = 1 }` (impure),
    /// `looper(n)` (pure, but may not return), `printer(x)` (calls an external symbol), `scratch`
    /// and `indexer` (frame slots, at a fixed and at an argument-chosen position), plus a caller of
    /// each so every one is summarized.
    fn zoo(it: &mut Interner) -> Program {
        let mut funcs = Vec::new();
        let mut leaf = Builder::new(it.intern("leaf"), MirType::I64);
        let x = leaf.add_param(MirType::I64);
        let sq = leaf.build(MirType::I64, Op::Bin(BinOp::Mul, x, x));
        leaf.ret(Some(sq));
        funcs.push(leaf.finish());

        let mut reader = Builder::new(it.intern("reader"), MirType::I64);
        let p = reader.add_param(MirType::Ptr);
        let v = reader.build(MirType::I64, Op::Load(p, MirType::I64));
        reader.ret(Some(v));
        funcs.push(reader.finish());

        let mut writer = Builder::new(it.intern("writer"), MirType::Void);
        let p = writer.add_param(MirType::Ptr);
        let one = writer.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        writer.build_void(Op::Store { ptr: p, value: one });
        writer.ret(None);
        funcs.push(writer.finish());

        let mut looper = Builder::new(it.intern("looper"), MirType::I64);
        let n = looper.add_param(MirType::I64);
        let header = looper.new_block();
        let exit = looper.new_block();
        let i = looper.block_param(header, MirType::I64);
        let zero = looper.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        looper.br(header, vec![zero]);
        looper.switch_to(header);
        let one = looper.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let next = looper.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        let more = looper.build(MirType::I1, Op::Cmp(wukong_mir::CmpOp::Slt, next, n));
        looper.cond_br(more, header, vec![next], exit, vec![]);
        looper.switch_to(exit);
        looper.ret(Some(i));
        funcs.push(looper.finish());

        let mut printer = Builder::new(it.intern("printer"), MirType::Void);
        let x = printer.add_param(MirType::I64);
        printer.build_void(Op::Call {
            func: it.intern("print"),
            args: vec![x],
        });
        printer.ret(None);
        funcs.push(printer.finish());

        // A frame-local scratch slot is fine: it dies with the call.
        let mut scratch = Builder::new(it.intern("scratch"), MirType::I64);
        let x = scratch.add_param(MirType::I64);
        let slot = scratch.alloca(MirType::I64);
        scratch.build_void(Op::Store {
            ptr: slot,
            value: x,
        });
        let back = scratch.build(MirType::I64, Op::Load(slot, MirType::I64));
        let call = scratch.build(
            MirType::I64,
            Op::Call {
                func: it.intern("leaf"),
                args: vec![back],
            },
        );
        scratch.ret(Some(call));
        funcs.push(scratch.finish());

        // A frame slot read at a position the argument picks: invisible to the caller like
        // `scratch`, but an out-of-range argument may fault, so not something to speculate.
        let mut indexer = Builder::new(it.intern("indexer"), MirType::I64);
        let i = indexer.add_param(MirType::I64);
        let table = indexer.alloca(MirType::Array(Box::new(MirType::I64), 4));
        let at = indexer.build(
            MirType::Ptr,
            Op::Gep {
                ptr: table,
                index: i,
                elem: MirType::I64,
            },
        );
        let got = indexer.build(MirType::I64, Op::Load(at, MirType::I64));
        indexer.ret(Some(got));
        funcs.push(indexer.finish());

        let mut main = Builder::new(it.intern("main"), MirType::I64);
        let slot = main.alloca(MirType::I64);
        let k = main.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        for (name, args) in [
            ("leaf", vec![k]),
            ("reader", vec![slot]),
            ("writer", vec![slot]),
            ("looper", vec![k]),
            ("printer", vec![k]),
            ("scratch", vec![k]),
            ("indexer", vec![k]),
        ] {
            main.build_void(Op::Call {
                func: it.intern(name),
                args,
            });
        }
        main.ret(Some(k));
        funcs.push(main.finish());
        let mut program = Program::new();
        program.funcs = funcs;
        assert!(wukong_mir::verify::verify_program(&program).is_empty());
        program
    }

    #[test]
    fn summaries_separate_pure_reading_looping_and_effectful_functions() {
        let mut it = Interner::new();
        let program = zoo(&mut it);
        let table = CallEffects::of(&program);
        let mut get = |name: &str| table.get(it.intern(name));
        let pure = CallEffect {
            reads_memory: false,
            will_return: true,
            speculatable: true,
        };
        assert_eq!(get("leaf"), Some(pure));
        assert_eq!(
            get("scratch"),
            Some(pure),
            "frame-local memory is invisible"
        );
        assert_eq!(
            get("indexer"),
            Some(CallEffect {
                reads_memory: false,
                will_return: true,
                speculatable: false,
            }),
            "an argument-indexed frame access is invisible but may fault"
        );
        assert_eq!(
            get("reader"),
            Some(CallEffect {
                reads_memory: true,
                will_return: true,
                speculatable: false,
            })
        );
        assert_eq!(
            get("looper"),
            Some(CallEffect {
                reads_memory: false,
                will_return: false,
                speculatable: false,
            })
        );
        assert_eq!(get("writer"), None);
        assert_eq!(get("printer"), None);
        assert_eq!(
            get("main"),
            None,
            "nothing calls main, so it is never summarized"
        );
        assert_eq!(get("print"), None);
    }

    /// Mutual recursion: both members are side-effect free, so both are summarized, but neither
    /// is known to return.
    #[test]
    fn a_recursive_component_is_summarized_but_never_assumed_to_return() {
        let mut it = Interner::new();
        let (even, odd) = (it.intern("even"), it.intern("odd"));
        let mut funcs = Vec::new();
        for (name, other) in [(even, odd), (odd, even)] {
            let mut b = Builder::new(name, MirType::I64);
            let n = b.add_param(MirType::I64);
            let r = b.build(
                MirType::I64,
                Op::Call {
                    func: other,
                    args: vec![n],
                },
            );
            b.ret(Some(r));
            funcs.push(b.finish());
        }
        let mut main = Builder::new(it.intern("main"), MirType::I64);
        let k = main.build(MirType::I64, Op::ConstInt(4, MirType::I64));
        let r = main.build(
            MirType::I64,
            Op::Call {
                func: even,
                args: vec![k],
            },
        );
        main.ret(Some(r));
        funcs.push(main.finish());
        let mut program = Program::new();
        program.funcs = funcs;
        let table = CallEffects::of(&program);
        for name in [even, odd] {
            assert_eq!(
                table.get(name),
                Some(CallEffect {
                    reads_memory: false,
                    will_return: false,
                    speculatable: false,
                })
            );
        }
    }
}
