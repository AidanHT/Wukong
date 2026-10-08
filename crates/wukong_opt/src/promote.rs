//! Scalar promotion of loop-invariant memory locations — LLVM's `promoteLoopAccessesToScalars`.
//!
//! A reduction written through memory — `out[0] = out[0] + a[i]`, `c[i*n + j] += a[i*n + k] *
//! b[k*n + j]` in the `k` loop of a naive matmul, `hist[b] += 1` with `b` fixed in the inner loop —
//! loads and stores one invariant address every iteration. LICM cannot hoist the load (the loop
//! writes the location) and nothing sinks the store, so the loop runs at store-to-load-forwarding
//! latency with a store port busy per element, where a register accumulator would run at the latency
//! of the add.
//!
//! This pass keeps the location in a register for the duration of the loop: one load in the
//! preheader, one store on every exit, and every access in between redirected to a fresh scalar
//! stack slot that mem2reg, at the head of the next sweep, turns into a loop-carried block
//! parameter. Building on mem2reg rather than threading the value by hand is what makes arbitrary
//! control flow inside the loop (conditional updates, several exits) correct by construction.
//!
//! # Legality
//!
//! A location — every access in the loop through a pointer defined outside it that is *must-alias*
//! to one address (the same SSA value, or the same base object at the same constant offset) — is
//! promoted when:
//!
//! * the loop has a preheader, no `return` inside it, no barrier, and every exit block is
//!   **dedicated** (all its predecessors are in the loop), so the exit store runs only on paths out
//!   of the loop;
//! * the location is written in the loop, every access to it moves one scalar type (an integer,
//!   `f32` or `f64` — not a pointer, which mem2reg would not promote back, nor `bf16`/`f16`, which
//!   round on their way through memory), and no other access in the loop may alias it, nor any call
//!   unless the location is a stack slot no callee can name; it is not in a `.shared` slot, which
//!   every thread of the block can see;
//! * the **preheader load** reads nothing the program would not have read (LICM's condition 3): the
//!   location is a dereferenceable stack slot, or it is accessed in the header, or it is accessed in
//!   a block that runs on every iteration, in which case the load (and the exit store) go through
//!   `select(first_test, addr, dummy)` exactly as LICM's guarded hoist does;
//! * the **exit store** writes nothing the program would not have written: the location is a stack
//!   slot no other code can see, or a store to it runs on every entry (in the header) or on every
//!   iteration (under the guard, so a zero-trip entry writes the dummy instead). Writing back an
//!   unchanged value to memory another thread can see would be a new data race. A location the
//!   header reads keeps an unguarded preheader load even when its store is guarded: the header runs
//!   on a zero-trip entry too and now reads the slot, so the slot must hold memory's value there.
//!
//! Loops are visited innermost first. A promotion moves the location's accesses into the
//! preheader and exits of its loop, which lie in the enclosing loop; that loop is skipped for the
//! rest of the run and considered again on the next sweep, so a location migrates outward one loop
//! level at a time.

use crate::alias::{type_bytes, AliasInfo, Prov};
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::licm::{fresh_value, Dummies, LoopMem, Spec};
use crate::loop_info::{self, LoopId, NaturalLoop};

use wukong_mir::{AddrSpace, Function, Inst, MirType, Op, ValueId};

use crate::{cfg, CfgAnalyses, Pass};

pub struct Promote;

impl Pass for Promote {
    fn name(&self) -> &'static str {
        "promote"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        if !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate();
        }
        if f.blocks.len() < 2 {
            return false;
        }
        let alias = AliasInfo::analyze(f);
        let (idom, preds) = cache.idoms_and_preds(f);
        let forest = loop_info::structure_with(f, idom, preds);
        if forest.is_empty() {
            return false;
        }
        // `.shared` slots are visible to every thread of the block: a write-back there is a store
        // another thread can observe, so they are never promoted.
        let shared: FxHashSet<u32> = f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|i| match (i.result, &i.op) {
                (
                    Some(r),
                    Op::Alloca {
                        space: AddrSpace::Shared,
                        ..
                    },
                ) => Some(r.0),
                _ => None,
            })
            .collect();
        let mut touched: Vec<LoopId> = Vec::new();
        let mut slots: Vec<Inst> = Vec::new();
        let mut dummies = Dummies::default();
        for id in forest.innermost_first() {
            if touched.iter().any(|&t| related(&forest, t, id)) {
                continue;
            }
            let l = forest.get(id);
            let Some(plan) = plan_loop(f, l, &alias, &shared, idom, preds) else {
                continue;
            };
            rewrite(f, &plan, &mut slots, &mut dummies);
            touched.push(id);
        }
        if slots.is_empty() {
            return false;
        }
        dummies.install(f);
        let entry = f.entry.0 as usize;
        slots.append(&mut f.blocks[entry].insts);
        f.blocks[entry].insts = slots;
        true
    }
}

/// Is one of these loops nested in (or equal to) the other?
fn related(forest: &loop_info::LoopForest, a: LoopId, b: LoopId) -> bool {
    let inside = |inner: LoopId, outer: LoopId| {
        let mut cur = Some(inner);
        while let Some(c) = cur {
            if c == outer {
                return true;
            }
            cur = forest.get(c).parent;
        }
        false
    };
    inside(a, b) || inside(b, a)
}

/// One location to promote.
struct Location {
    /// An address of the location that is available in the preheader and every exit block.
    addr: ValueId,
    ty: MirType,
    /// The preheader load reads through `select(first_test, addr, dummy)` rather than `addr`.
    guard_load: bool,
    /// The exit stores write through that `select` rather than `addr`. Always set when
    /// `guard_load` is: a zero-trip slot holds the dummy's value, which must not reach `addr`.
    guard_store: bool,
    /// `(block, instruction index)` of every access to it in the loop.
    accesses: Vec<(u32, usize)>,
}

struct Plan {
    locations: Vec<Location>,
    /// The first-iteration test, when any location is guarded.
    first_test: Option<(wukong_mir::CmpOp, ValueId, ValueId)>,
    preheader: u32,
    /// Distinct exit blocks, ascending.
    exits: Vec<u32>,
}

/// A location key: must-alias addresses share one.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Based(u8, u32, i64),
    Value(u32),
}

fn key_of(alias: &AliasInfo, p: ValueId) -> Key {
    match alias.base_and_offset(p) {
        Some((Prov::Alloca(a), off)) => Key::Based(0, a.0, off),
        Some((Prov::Param(i), off)) => Key::Based(1, i, off),
        Some((Prov::Global(g), off)) => Key::Based(2, g.0, off),
        _ => Key::Value(p.0),
    }
}

/// Can a value of this type live in a promoted slot that mem2reg will turn into a register?
fn promotable(ty: &MirType) -> bool {
    ty.is_int() || matches!(ty, MirType::F32 | MirType::F64)
}

fn plan_loop(
    f: &Function,
    l: &NaturalLoop,
    alias: &AliasInfo,
    shared: &FxHashSet<u32>,
    idom: &[u32],
    preds: &[Vec<u32>],
) -> Option<Plan> {
    let preheader = l.preheader?.0;
    if !l.abnormal_exits.is_empty() || l.exits.is_empty() {
        return None;
    }
    let body: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
    let mut exits: Vec<u32> = l.exits.iter().map(|&(_, to)| to.0).collect();
    exits.sort_unstable();
    exits.dedup();
    if exits
        .iter()
        .any(|&e| preds[e as usize].iter().any(|p| !body.contains(p)))
    {
        return None; // an exit block also reached from outside the loop
    }
    let mem = LoopMem::of(f, &body, None);
    if mem.has_barrier {
        return None;
    }

    // Every memory access in the loop, and which of them could belong to an invariant location.
    let mut defined: FxHashSet<u32> = FxHashSet::default();
    for &b in &body {
        let bb = &f.blocks[b as usize];
        defined.extend(bb.params.iter().map(|p| p.0));
        defined.extend(bb.insts.iter().filter_map(|i| i.result.map(|r| r.0)));
    }
    // (block, index, address, type, is_store)
    let mut all: Vec<(u32, usize, ValueId, MirType, bool)> = Vec::new();
    for b in &l.blocks {
        for (i, inst) in f.blocks[b.0 as usize].insts.iter().enumerate() {
            match &inst.op {
                Op::Load(p, ty) => all.push((b.0, i, *p, ty.clone(), false)),
                Op::Store { ptr, value } => {
                    all.push((b.0, i, *ptr, f.value_type(*value).clone(), true))
                }
                _ => {}
            }
        }
    }
    let mut groups: FxHashMap<Key, Vec<usize>> = FxHashMap::default();
    let mut order: Vec<Key> = Vec::new();
    for (k, a) in all.iter().enumerate() {
        if defined.contains(&a.2 .0) {
            continue;
        }
        let key = key_of(alias, a.2);
        groups
            .entry(key)
            .or_insert_with(|| {
                order.push(key);
                Vec::new()
            })
            .push(k);
    }

    let spec = Spec::of(f, l, &body, idom);
    let mut locations = Vec::new();
    for key in order {
        let members = &groups[&key];
        let first = &all[members[0]];
        let (addr, ty) = (first.2, first.3.clone());
        if !promotable(&ty) || members.iter().any(|&k| all[k].3 != ty) {
            continue;
        }
        if !members.iter().any(|&k| all[k].4) {
            continue; // never written: plain LICM's business
        }
        let bytes = type_bytes(&ty)?;
        let mine: FxHashSet<usize> = members.iter().copied().collect();
        let clobbered = all.iter().enumerate().any(|(k, a)| {
            !mine.contains(&k)
                && alias.may_alias_sized(addr, bytes, a.2, type_bytes(&a.3).unwrap_or(u32::MAX))
        });
        if matches!(alias.prov(addr), Prov::Alloca(a) if shared.contains(&a.0)) {
            continue;
        }
        if clobbered || (mem.has_call && !alias.is_private_stack(addr)) {
            continue;
        }
        let in_header = |store_only: bool| {
            members
                .iter()
                .any(|&k| all[k].0 == spec.header && (!store_only || all[k].4))
        };
        let every_iteration =
            |k: usize| all[k].0 == spec.header || spec.every_iteration.contains(&all[k].0);
        let stored_every_iteration = members.iter().any(|&k| all[k].4 && every_iteration(k));
        let accessed_every_iteration = members.iter().any(|&k| every_iteration(k));
        let private = alias.is_private_stack(addr);
        let guardable = spec.first_test.is_some() && !matches!(alias.prov(addr), Prov::Alloca(_));
        // The preheader load is safe un-guarded for a stack slot, a location the header reads, or
        // one every iteration reads when the body provably runs; the exit store, for a slot nobody
        // else sees, a location the header writes, or one every iteration writes when the body
        // provably runs. Otherwise the unsafe half goes through the first-iteration guard, which
        // needs a store on every iteration: it proves the location is both read-safe and written
        // whenever the loop runs at all.
        //
        // Only the unsafe half: a location the header *reads* is read on a zero-trip entry too, and
        // the header's read now reads the slot, so the slot must start from memory even when the
        // body never runs — a guarded load would hand the header (and any exit argument it feeds)
        // the dummy's value. A guarded load in turn forces a guarded store, so a zero-trip slot
        // holding the dummy's value is never written back to `addr`.
        let plain_load = alias.is_dereferenceable(addr, bytes)
            || in_header(false)
            || (spec.runs_once && accessed_every_iteration);
        let plain_store = private || in_header(true) || (spec.runs_once && stored_every_iteration);
        let (guard_load, guard_store) = if plain_load && plain_store {
            (false, false)
        } else if guardable && stored_every_iteration {
            (!plain_load, true)
        } else {
            continue;
        };
        locations.push(Location {
            addr,
            ty,
            guard_load,
            guard_store,
            accesses: members.iter().map(|&k| (all[k].0, all[k].1)).collect(),
        });
    }
    if locations.is_empty() {
        return None;
    }
    Some(Plan {
        first_test: spec.first_test,
        locations,
        preheader,
        exits,
    })
}

/// Apply a plan: a slot per location (returned in `slots` for the entry block), the preheader load,
/// the redirected accesses, and the exit stores. A guarded location's zero-trip target comes from
/// `dummies`, shared with every other guarded location of its type.
fn rewrite(f: &mut Function, plan: &Plan, slots: &mut Vec<Inst>, dummies: &mut Dummies) {
    let mut guard: Option<ValueId> = None;
    let mut pre: Vec<Inst> = Vec::new();
    // (exit slot load, exit store) per location, built once and copied into every exit.
    let mut tails: Vec<(ValueId, MirType, ValueId)> = Vec::new();
    for loc in &plan.locations {
        let slot = fresh_value(f, MirType::Ptr);
        slots.push(Inst {
            result: Some(slot),
            op: Op::Alloca {
                ty: loc.ty.clone(),
                space: AddrSpace::Local,
            },
        });
        let guarded_addr = if loc.guard_load || loc.guard_store {
            let cond = match guard {
                Some(c) => c,
                None => {
                    let (pred, a, b) = plan
                        .first_test
                        .expect("a guarded location requires a first-iteration test");
                    let c = fresh_value(f, MirType::I1);
                    pre.push(Inst {
                        result: Some(c),
                        op: Op::Cmp(pred, a, b),
                    });
                    guard = Some(c);
                    c
                }
            };
            let dummy = dummies.slot(f, &loc.ty);
            let sel = fresh_value(f, MirType::Ptr);
            pre.push(Inst {
                result: Some(sel),
                op: Op::Select(cond, loc.addr, dummy),
            });
            sel
        } else {
            loc.addr
        };
        let load_addr = if loc.guard_load {
            guarded_addr
        } else {
            loc.addr
        };
        let store_addr = if loc.guard_store {
            guarded_addr
        } else {
            loc.addr
        };
        let init = fresh_value(f, loc.ty.clone());
        pre.push(Inst {
            result: Some(init),
            op: Op::Load(load_addr, loc.ty.clone()),
        });
        pre.push(Inst {
            result: None,
            op: Op::Store {
                ptr: slot,
                value: init,
            },
        });
        for &(b, i) in &loc.accesses {
            match &mut f.blocks[b as usize].insts[i].op {
                Op::Load(p, _) => *p = slot,
                Op::Store { ptr, .. } => *ptr = slot,
                _ => unreachable!("a promoted access is a load or a store"),
            }
        }
        tails.push((slot, loc.ty.clone(), store_addr));
    }
    f.blocks[plan.preheader as usize].insts.extend(pre);
    for &e in &plan.exits {
        let mut head: Vec<Inst> = Vec::with_capacity(2 * tails.len());
        for (slot, ty, addr) in &tails {
            let v = fresh_value(f, ty.clone());
            head.push(Inst {
                result: Some(v),
                op: Op::Load(*slot, ty.clone()),
            });
            head.push(Inst {
                result: None,
                op: Op::Store {
                    ptr: *addr,
                    value: v,
                },
            });
        }
        head.append(&mut f.blocks[e as usize].insts);
        f.blocks[e as usize].insts = head;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mem2Reg;
    use wukong_mir::{BinOp, Builder, CmpOp, Program};
    use wukong_span::{Interner, Symbol};

    /// The body of an accumulate-through-memory loop, for [`accumulate`].
    #[derive(Clone, Copy, PartialEq)]
    enum Body {
        /// `out[0] = out[0] + x[i]` with `x` a private array: promotable.
        Private,
        /// `out[0] = out[0] + w[0]` with `w` another parameter, which may be `out`.
        AliasedParam,
        /// `if i is odd { out[0] = out[0] + x[i] }`: the store does not run every iteration.
        Conditional,
    }

    /// `fn f(out: *i64, w: *i64, n: i64) { for i in 0..n { body } }`, plus a `main` that runs it
    /// with `n = 4` and `n = 0` and returns `out_a * 1000 + out_b`.
    fn accumulate(it: &mut Interner, body: Body) -> (Function, Function) {
        let fname = it.intern("f");
        let mut b = Builder::new(fname, MirType::Void);
        let out = b.add_param(MirType::Ptr);
        let w = b.add_param(MirType::Ptr);
        let n = b.add_param(MirType::I64);
        let x = b.alloca(MirType::Array(Box::new(MirType::I64), 4));
        for k in 0..4 {
            let kk = b.build(MirType::I64, Op::ConstInt(k, MirType::I64));
            let p = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: x,
                    index: kk,
                    elem: MirType::I64,
                },
            );
            let v = b.build(MirType::I64, Op::ConstInt(k + 1, MirType::I64));
            b.build_void(Op::Store { ptr: p, value: v });
        }
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let blk = b.new_block();
        let upd = b.new_block();
        let latch = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, blk, vec![], exit, vec![]);
        b.switch_to(blk);
        if body == Body::Conditional {
            let odd = b.build(MirType::I64, Op::Bin(BinOp::And, i, one));
            let take = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, odd, one));
            b.cond_br(take, upd, vec![], latch, vec![]);
        } else {
            b.br(upd, vec![]);
        }
        b.switch_to(upd);
        let addend = if body == Body::AliasedParam {
            b.build(MirType::I64, Op::Load(w, MirType::I64))
        } else {
            let p = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: x,
                    index: i,
                    elem: MirType::I64,
                },
            );
            b.build(MirType::I64, Op::Load(p, MirType::I64))
        };
        let old = b.build(MirType::I64, Op::Load(out, MirType::I64));
        let new = b.build(MirType::I64, Op::Bin(BinOp::Add, old, addend));
        b.build_void(Op::Store {
            ptr: out,
            value: new,
        });
        b.br(latch, vec![]);
        b.switch_to(latch);
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(None);
        let f = b.finish();
        (f, harness(it, fname))
    }

    /// `main`: `a = 100; f(&a, &a, 4); b = 5; f(&b, &b, 0); return a * 1000 + b`.
    fn harness(it: &mut Interner, fname: Symbol) -> Function {
        let mut m = Builder::new(it.intern("main"), MirType::I64);
        let a = m.alloca(MirType::I64);
        let bslot = m.alloca(MirType::I64);
        let hundred = m.build(MirType::I64, Op::ConstInt(100, MirType::I64));
        let five = m.build(MirType::I64, Op::ConstInt(5, MirType::I64));
        let four = m.build(MirType::I64, Op::ConstInt(4, MirType::I64));
        let zero = m.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        m.build_void(Op::Store {
            ptr: a,
            value: hundred,
        });
        m.build_void(Op::Store {
            ptr: bslot,
            value: five,
        });
        m.build_void(Op::Call {
            func: fname,
            args: vec![a, a, four],
        });
        m.build_void(Op::Call {
            func: fname,
            args: vec![bslot, bslot, zero],
        });
        let va = m.build(MirType::I64, Op::Load(a, MirType::I64));
        let vb = m.build(MirType::I64, Op::Load(bslot, MirType::I64));
        let k = m.build(MirType::I64, Op::ConstInt(1000, MirType::I64));
        let hi = m.build(MirType::I64, Op::Bin(BinOp::Mul, va, k));
        let r = m.build(MirType::I64, Op::Bin(BinOp::Add, hi, vb));
        m.ret(Some(r));
        m.finish()
    }

    fn run(main: &Function, f: &Function, it: &Interner) -> i64 {
        let mut p = Program::new();
        p.funcs.push(main.clone());
        p.funcs.push(f.clone());
        let errors = wukong_mir::verify::verify_program(&p);
        assert!(errors.is_empty(), "{errors:?}");
        wukong_interp::run(&p, main.name, it).unwrap()
    }

    /// Loads and stores through `out` inside the loop (the header, body and latch blocks).
    fn out_traffic(f: &Function) -> usize {
        let out = f.params[0];
        [1usize, 2, 3, 4]
            .iter()
            .flat_map(|&b| &f.blocks[b].insts)
            .filter(|i| match &i.op {
                Op::Load(p, _) => *p == out,
                Op::Store { ptr, .. } => *ptr == out,
                _ => false,
            })
            .count()
    }

    #[test]
    fn an_out_parameter_accumulator_is_carried_in_a_register() {
        let mut it = Interner::new();
        let (mut f, main) = accumulate(&mut it, Body::Private);
        // a = 100 + (1 + 2 + 3 + 4) = 110; b = 5 untouched by the zero-trip call.
        assert_eq!(run(&main, &f, &it), 110_005);
        assert_eq!(out_traffic(&f), 2);
        let mut cache = CfgAnalyses::default();
        assert!(Promote.run_function(&mut f, &mut cache));
        assert_eq!(
            out_traffic(&f),
            0,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(run(&main, &f, &it), 110_005);
        // mem2reg turns the promotion slot into a loop-carried value.
        assert!(Mem2Reg.run_function(&mut f, &mut cache));
        assert_eq!(run(&main, &f, &it), 110_005);
        // Nothing left to promote.
        assert!(!Promote.run_function(&mut f, &mut cache));
    }

    /// `w` may be `out` (main passes the same pointer twice), and the loop never writes `out` on
    /// some iterations: neither location may be kept in a register.
    #[test]
    fn aliased_or_conditionally_written_locations_stay_in_memory() {
        for body in [Body::AliasedParam, Body::Conditional] {
            let mut it = Interner::new();
            let (mut f, main) = accumulate(&mut it, body);
            let before = run(&main, &f, &it);
            assert!(
                !Promote.run_function(&mut f, &mut CfgAnalyses::default()),
                "{}",
                wukong_mir::print::print_function(&f, &it)
            );
            assert_eq!(out_traffic(&f), 2);
            assert_eq!(run(&main, &f, &it), before);
        }
    }

    /// A private slot read and written in the loop needs no guard: it is in bounds, and nobody else
    /// can observe the value written back on a zero-trip entry.
    #[test]
    fn a_private_slot_is_promoted_without_a_guard() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let s = b.alloca(MirType::Array(Box::new(MirType::I64), 2));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let cell = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: s,
                index: one,
                elem: MirType::I64,
            },
        );
        let three = b.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        b.build_void(Op::Store {
            ptr: cell,
            value: three,
        });
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let n = b.build(MirType::I64, Op::ConstInt(5, MirType::I64));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        let old = b.build(MirType::I64, Op::Load(cell, MirType::I64));
        let new = b.build(MirType::I64, Op::Bin(BinOp::Mul, old, three));
        b.build_void(Op::Store {
            ptr: cell,
            value: new,
        });
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        let r = b.build(MirType::I64, Op::Load(cell, MirType::I64));
        b.ret(Some(r));
        let mut f = b.finish();
        let run1 = |f: &Function| {
            let mut p = Program::new();
            p.funcs.push(f.clone());
            assert!(wukong_mir::verify::verify_function(f).is_empty());
            wukong_interp::run(&p, f.name, &it).unwrap()
        };
        // 3 · 3^5 = 729.
        assert_eq!(run1(&f), 729);
        assert!(Promote.run_function(&mut f, &mut CfgAnalyses::default()));
        assert!(
            !f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(i.op, Op::Select(..))),
            "a private slot needs no guard"
        );
        assert_eq!(run1(&f), 729);
    }

    /// `for i in 0..4 { out[0] = out[0] + i }`: the first test `0 < 4` is a true constant, so the
    /// body provably runs and a parameter location written on every iteration is promoted with a
    /// plain preheader load and exit store — no guard, no dummy slot.
    #[test]
    fn a_constant_trip_loop_promotes_without_a_guard() {
        let mut it = Interner::new();
        let fname = it.intern("f");
        let mut b = Builder::new(fname, MirType::Void);
        let out = b.add_param(MirType::Ptr);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let four = b.build(MirType::I64, Op::ConstInt(4, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, four));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        let old = b.build(MirType::I64, Op::Load(out, MirType::I64));
        let new = b.build(MirType::I64, Op::Bin(BinOp::Add, old, i));
        b.build_void(Op::Store {
            ptr: out,
            value: new,
        });
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(None);
        let mut f = b.finish();

        let mut m = Builder::new(it.intern("main"), MirType::I64);
        let a = m.alloca(MirType::I64);
        let hundred = m.build(MirType::I64, Op::ConstInt(100, MirType::I64));
        m.build_void(Op::Store {
            ptr: a,
            value: hundred,
        });
        m.build_void(Op::Call {
            func: fname,
            args: vec![a],
        });
        let r = m.build(MirType::I64, Op::Load(a, MirType::I64));
        m.ret(Some(r));
        let main = m.finish();

        // 100 + 0 + 1 + 2 + 3.
        assert_eq!(run(&main, &f, &it), 106);
        assert!(Promote.run_function(&mut f, &mut CfgAnalyses::default()));
        let body_traffic = f.blocks[body.0 as usize]
            .insts
            .iter()
            .filter(|i| {
                matches!(&i.op, Op::Load(p, _) if *p == out)
                    || matches!(&i.op, Op::Store { ptr, .. } if *ptr == out)
            })
            .count();
        assert_eq!(body_traffic, 0);
        assert!(
            !f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(i.op, Op::Select(..))),
            "a loop that provably runs needs no guard: {}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(run(&main, &f, &it), 106);
    }

    /// `fn f(out, n) -> i64 { i = 0; loop { v = out[0]; if !(i < n) { return v } out[0] = v + 10;
    /// i += 1 } }`: the header reads the location and passes it out of the loop, and the body writes
    /// it on every iteration. The write-back needs the guard, but the preheader load must not take
    /// it: on a zero-trip entry the header still reads, and it must see memory, not the dummy.
    #[test]
    fn a_location_the_header_reads_is_loaded_unguarded_on_a_zero_trip_entry() {
        let mut it = Interner::new();
        let fname = it.intern("f");
        let mut b = Builder::new(fname, MirType::I64);
        let out = b.add_param(MirType::Ptr);
        let n = b.add_param(MirType::I64);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let ten = b.build(MirType::I64, Op::ConstInt(10, MirType::I64));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let r = b.block_param(exit, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let v = b.build(MirType::I64, Op::Load(out, MirType::I64));
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![v]);
        b.switch_to(body);
        let old = b.build(MirType::I64, Op::Load(out, MirType::I64));
        let new = b.build(MirType::I64, Op::Bin(BinOp::Add, old, ten));
        b.build_void(Op::Store {
            ptr: out,
            value: new,
        });
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(Some(r));
        let mut f = b.finish();

        // `main`: a = 100, b = 5; return f(&a, 3) * 1000 + f(&b, 0) + a * 10 + b.
        let mut m = Builder::new(it.intern("main"), MirType::I64);
        let a = m.alloca(MirType::I64);
        let bs = m.alloca(MirType::I64);
        let hundred = m.build(MirType::I64, Op::ConstInt(100, MirType::I64));
        let five = m.build(MirType::I64, Op::ConstInt(5, MirType::I64));
        let three = m.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        let none = m.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        m.build_void(Op::Store {
            ptr: a,
            value: hundred,
        });
        m.build_void(Op::Store {
            ptr: bs,
            value: five,
        });
        let r3 = m.build(
            MirType::I64,
            Op::Call {
                func: fname,
                args: vec![a, three],
            },
        );
        let r0 = m.build(
            MirType::I64,
            Op::Call {
                func: fname,
                args: vec![bs, none],
            },
        );
        let va = m.build(MirType::I64, Op::Load(a, MirType::I64));
        let vb = m.build(MirType::I64, Op::Load(bs, MirType::I64));
        let k = m.build(MirType::I64, Op::ConstInt(1000, MirType::I64));
        let tenk = m.build(MirType::I64, Op::ConstInt(10, MirType::I64));
        let t = m.build(MirType::I64, Op::Bin(BinOp::Mul, r3, k));
        let t = m.build(MirType::I64, Op::Bin(BinOp::Add, t, r0));
        let ta = m.build(MirType::I64, Op::Bin(BinOp::Mul, va, tenk));
        let t = m.build(MirType::I64, Op::Bin(BinOp::Add, t, ta));
        let t = m.build(MirType::I64, Op::Bin(BinOp::Add, t, vb));
        m.ret(Some(t));
        let main = m.finish();

        // f(&a, 3): the header reads 100, 110, 120, 130 and returns 130, leaving a = 130.
        // f(&b, 0): the header reads 5 and returns it, leaving b = 5.
        // 130·1000 + 5 + 130·10 + 5 = 131_310.
        assert_eq!(run(&main, &f, &it), 131_310);
        assert!(Promote.run_function(&mut f, &mut CfgAnalyses::default()));
        let in_loop = [header, body]
            .iter()
            .flat_map(|blk| &f.blocks[blk.0 as usize].insts)
            .filter(|i| {
                matches!(&i.op, Op::Load(p, _) if *p == out)
                    || matches!(&i.op, Op::Store { ptr, .. } if *ptr == out)
            })
            .count();
        assert_eq!(in_loop, 0);
        // The write-back is guarded; the preheader load reads `out` itself.
        assert!(f.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::Load(p, _) if p == out)));
        assert_eq!(run(&main, &f, &it), 131_310);
    }
}
