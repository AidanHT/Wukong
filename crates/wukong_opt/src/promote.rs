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
//! * the loop has a preheader, no `return` inside it, and no barrier. The exit store must run only
//!   on paths out of the loop, so it goes in each exit block that is **dedicated** (all its
//!   predecessors are in the loop); an exit block other code also reaches gets a dedicated copy
//!   first -- a new block on the loop's edges into it, forwarding their arguments (LLVM's
//!   `formDedicatedExitBlocks`; `loop-canon` leaves exits alone, see its docs). Bounds-check
//!   elimination is what makes this common: its checked and check-free versions of a loop share
//!   one exit block, so neither version's exit is ever dedicated;
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

use wukong_mir::{AddrSpace, BlockId, Function, Inst, MirType, Op, ValueId};

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
        let blocks_before = f.blocks.len();
        for id in forest.innermost_first() {
            if touched.iter().any(|&t| related(&forest, t, id)) {
                continue;
            }
            let l = forest.get(id);
            // A dedicated exit made for an earlier loop leaves `idom` and `preds` stale, but not
            // wrong for this one: splitting an edge keeps every dominance relation between the
            // blocks that were already there, and the loops visited after it are disjoint from it,
            // so the only change they can see is a predecessor of a shared exit block moving from
            // the earlier loop to its new block -- outside this loop either way.
            let Some(plan) = plan_loop(f, l, &alias, &shared, idom, preds) else {
                continue;
            };
            rewrite(f, &plan, &mut slots, &mut dummies);
            touched.push(id);
        }
        if f.blocks.len() != blocks_before {
            cache.invalidate();
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
    exits: Vec<Exit>,
}

/// One exit block of the loop.
struct Exit {
    block: u32,
    /// The loop's blocks that branch to `block`, when blocks outside the loop branch there too: the
    /// write-back then goes in a new block on just these edges.
    shared_from: Option<Vec<u32>>,
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
    let mut exit_blocks: Vec<u32> = l.exits.iter().map(|&(_, to)| to.0).collect();
    exit_blocks.sort_unstable();
    exit_blocks.dedup();
    let exits: Vec<Exit> = exit_blocks
        .into_iter()
        .map(|e| {
            let dedicated = preds[e as usize].iter().all(|p| body.contains(p));
            let shared_from = (!dedicated).then(|| {
                let mut from: Vec<u32> = l
                    .exits
                    .iter()
                    .filter(|&&(_, to)| to.0 == e)
                    .map(|&(inside, _)| inside.0)
                    .collect();
                from.dedup(); // `l.exits` is sorted
                from
            });
            Exit {
                block: e,
                shared_from,
            }
        })
        .collect();
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
                align: 0,
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
    for exit in &plan.exits {
        let e = match &exit.shared_from {
            None => exit.block,
            Some(from) => {
                // A new block on the loop's edges into the shared exit, taking each edge's
                // arguments and passing them on: it runs exactly when the loop is left that way.
                let to = BlockId(exit.block);
                let dedicated = crate::loop_canon::forwarding_block(f, to);
                for &b in from {
                    crate::loop_canon::redirect(&mut f.blocks[b as usize].term, to, dedicated);
                }
                dedicated.0
            }
        };
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

    /// Loads and stores through `ptr` in the given blocks.
    fn traffic_through(f: &Function, ptr: ValueId, blocks: &[BlockId]) -> usize {
        blocks
            .iter()
            .flat_map(|b| &f.blocks[b.0 as usize].insts)
            .filter(|i| {
                matches!(&i.op, Op::Load(p, _) if *p == ptr)
                    || matches!(&i.op, Op::Store { ptr: q, .. } if *q == ptr)
            })
            .count()
    }

    /// The blocks promotion added: each must be a dedicated exit holding a write-back, forwarding
    /// its parameters to `exit`, and reached only from `from`.
    fn dedicated_exits(f: &Function, before: usize, exit: BlockId, from: &[BlockId]) -> usize {
        let preds = cfg::predecessors(f);
        for b in &f.blocks[before..] {
            assert!(
                matches!(&b.term, wukong_mir::Terminator::Br { target, args }
                    if *target == exit && *args == b.params),
                "a new block forwards its parameters to the shared exit"
            );
            assert!(b.insts.iter().any(|i| matches!(i.op, Op::Store { .. })));
            let mut want: Vec<u32> = from.iter().map(|b| b.0).collect();
            want.sort_unstable();
            assert_eq!(preds[b.id.0 as usize], want);
        }
        f.blocks.len() - before
    }

    /// `fn f(out: *i64, n: i64) -> i64`, whose loop exit block is also where the function goes
    /// when it skips the loop:
    ///
    /// ```text
    ///   entry: x = [1, 2, 3, 4]; cond_br (n > 4), E(-1), pre
    ///   pre:   br H(0)
    ///   H(i):  cond_br (i < n), B, E(i)
    ///   B:     out[0] = out[0] + x[i]; br H(i + 1)
    ///   E(r):  ret r
    /// ```
    ///
    /// `E` is not a dedicated exit, so the write-back cannot go there: it would store on the path
    /// that skips the loop, a write the program never made. It goes in a new block on the exit
    /// edge instead.
    #[test]
    fn a_shared_exit_block_gets_a_dedicated_copy_for_the_write_back() {
        let mut it = Interner::new();
        let fname = it.intern("f");
        let mut b = Builder::new(fname, MirType::I64);
        let out = b.add_param(MirType::Ptr);
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
        let four = b.build(MirType::I64, Op::ConstInt(4, MirType::I64));
        let skip = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, n, four));
        let minus1 = b.build(MirType::I64, Op::ConstInt(-1, MirType::I64));
        let pre = b.new_block();
        let header = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let body = b.new_block();
        let exit = b.new_block();
        let r = b.block_param(exit, MirType::I64);
        b.cond_br(skip, exit, vec![minus1], pre, vec![]);
        b.switch_to(pre);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![i]);
        b.switch_to(body);
        let p = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: x,
                index: i,
                elem: MirType::I64,
            },
        );
        let v = b.build(MirType::I64, Op::Load(p, MirType::I64));
        let old = b.build(MirType::I64, Op::Load(out, MirType::I64));
        let new = b.build(MirType::I64, Op::Bin(BinOp::Add, old, v));
        b.build_void(Op::Store {
            ptr: out,
            value: new,
        });
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(Some(r));
        let mut f = b.finish();

        // `main`: a = 100, b = 5, c = 7; r1 = f(&a, 3), r2 = f(&b, 9), r3 = f(&c, 0);
        // return a·10^7 + b·10^5 + c·10^3 + r1·100 + (r2 + 5)·10 + r3.
        let mut m = Builder::new(it.intern("main"), MirType::I64);
        let mut slots = Vec::new();
        for init in [100, 5, 7] {
            let s = m.alloca(MirType::I64);
            let v = m.build(MirType::I64, Op::ConstInt(init, MirType::I64));
            m.build_void(Op::Store { ptr: s, value: v });
            slots.push(s);
        }
        let mut rs = Vec::new();
        for (s, trip) in slots.iter().zip([3, 9, 0]) {
            let t = m.build(MirType::I64, Op::ConstInt(trip, MirType::I64));
            rs.push(m.build(
                MirType::I64,
                Op::Call {
                    func: fname,
                    args: vec![*s, t],
                },
            ));
        }
        let five = m.build(MirType::I64, Op::ConstInt(5, MirType::I64));
        rs[1] = m.build(MirType::I64, Op::Bin(BinOp::Add, rs[1], five));
        let mut terms = Vec::new();
        for s in &slots {
            terms.push(m.build(MirType::I64, Op::Load(*s, MirType::I64)));
        }
        terms.extend(rs);
        let mut acc = m.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        for (t, scale) in terms
            .into_iter()
            .zip([10_000_000, 100_000, 1_000, 100, 10, 1])
        {
            let k = m.build(MirType::I64, Op::ConstInt(scale, MirType::I64));
            let w = m.build(MirType::I64, Op::Bin(BinOp::Mul, t, k));
            acc = m.build(MirType::I64, Op::Bin(BinOp::Add, acc, w));
        }
        m.ret(Some(acc));
        let main = m.finish();

        // f(&a, 3): a = 100 + 1 + 2 + 3 = 106, returns 3. f(&b, 9): skips, b = 5, returns -1.
        // f(&c, 0): zero trips, c = 7, returns 0.
        let expected = 106 * 10_000_000 + 5 * 100_000 + 7 * 1_000 + 3 * 100 + 4 * 10;
        assert_eq!(run(&main, &f, &it), expected);
        let before = f.blocks.len();
        let mut cache = CfgAnalyses::default();
        assert!(
            Promote.run_function(&mut f, &mut cache),
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        let text = wukong_mir::print::print_function(&f, &it);
        assert_eq!(traffic_through(&f, out, &[header, body]), 0, "{text}");
        assert!(
            !f.blocks[exit.0 as usize]
                .insts
                .iter()
                .any(|i| matches!(i.op, Op::Store { .. })),
            "the shared exit is also the skip path's: no write-back there\n{text}"
        );
        assert_eq!(dedicated_exits(&f, before, exit, &[header]), 1, "{text}");
        assert_eq!(run(&main, &f, &it), expected);
        assert!(Mem2Reg.run_function(&mut f, &mut cache));
        assert_eq!(run(&main, &f, &it), expected);
        assert!(!Promote.run_function(&mut f, &mut cache));
    }

    /// `fn g(n, lim) -> i64` doubling a private slot, with two exits (the header's test and a
    /// `break` once the slot passes `lim`) into one block that the path around the loop reaches too:
    ///
    /// ```text
    ///   entry: s = 1; cond_br (n < 0), E(n), pre
    ///   pre:   br H(0)
    ///   H(i):  cond_br (i < n), B, E(i)
    ///   B:     s = s * 2; cond_br (s > lim), E(100), L
    ///   L:     br H(i + 1)
    ///   E(r):  ret s * 1000 + r
    /// ```
    ///
    /// Both exiting edges go through one new block, which forwards each edge's argument.
    #[test]
    fn several_exit_edges_into_a_shared_block_share_one_dedicated_exit() {
        let mut it = Interner::new();
        let gname = it.intern("g");
        let mut b = Builder::new(gname, MirType::I64);
        let n = b.add_param(MirType::I64);
        let lim = b.add_param(MirType::I64);
        let s = b.alloca(MirType::I64);
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        b.build_void(Op::Store { ptr: s, value: one });
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let neg = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, n, zero));
        let pre = b.new_block();
        let header = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let body = b.new_block();
        let latch = b.new_block();
        let exit = b.new_block();
        let r = b.block_param(exit, MirType::I64);
        b.cond_br(neg, exit, vec![n], pre, vec![]);
        b.switch_to(pre);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![i]);
        b.switch_to(body);
        let old = b.build(MirType::I64, Op::Load(s, MirType::I64));
        let two = b.build(MirType::I64, Op::ConstInt(2, MirType::I64));
        let new = b.build(MirType::I64, Op::Bin(BinOp::Mul, old, two));
        b.build_void(Op::Store { ptr: s, value: new });
        let big = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, new, lim));
        let hundred = b.build(MirType::I64, Op::ConstInt(100, MirType::I64));
        b.cond_br(big, exit, vec![hundred], latch, vec![]);
        b.switch_to(latch);
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        let v = b.build(MirType::I64, Op::Load(s, MirType::I64));
        let k = b.build(MirType::I64, Op::ConstInt(1000, MirType::I64));
        let t = b.build(MirType::I64, Op::Bin(BinOp::Mul, v, k));
        let u = b.build(MirType::I64, Op::Bin(BinOp::Add, t, r));
        b.ret(Some(u));
        let mut g = b.finish();

        let call = |it: &mut Interner, n: i128, lim: i128| -> Function {
            let mut m = Builder::new(it.intern("main"), MirType::I64);
            let a = m.build(MirType::I64, Op::ConstInt(n, MirType::I64));
            let l = m.build(MirType::I64, Op::ConstInt(lim, MirType::I64));
            let r = m.build(
                MirType::I64,
                Op::Call {
                    func: gname,
                    args: vec![a, l],
                },
            );
            m.ret(Some(r));
            m.finish()
        };
        // g(-3, 50): skips, 1·1000 - 3. g(4, 50): s = 16 after four trips, 16_000 + 4.
        // g(10, 50): s = 64 > 50 on the sixth trip, 64_000 + 100. g(0, 50): zero trips, 1000.
        let cases = [
            ((-3, 50), 997),
            ((4, 50), 16_004),
            ((10, 50), 64_100),
            ((0, 50), 1_000),
        ];
        let mains: Vec<(Function, i64)> = cases
            .iter()
            .map(|&((n, lim), want)| (call(&mut it, n, lim), want))
            .collect();
        for (main, want) in &mains {
            assert_eq!(run(main, &g, &it), *want);
        }
        let before = g.blocks.len();
        let mut cache = CfgAnalyses::default();
        assert!(
            Promote.run_function(&mut g, &mut cache),
            "{}",
            wukong_mir::print::print_function(&g, &it)
        );
        let text = wukong_mir::print::print_function(&g, &it);
        assert_eq!(traffic_through(&g, s, &[header, body, latch]), 0, "{text}");
        assert_eq!(
            dedicated_exits(&g, before, exit, &[header, body]),
            1,
            "{text}"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(i.op, Op::Select(..))),
            "a private slot needs no guard\n{text}"
        );
        for (main, want) in &mains {
            assert_eq!(run(main, &g, &it), *want, "{text}");
        }
        assert!(Mem2Reg.run_function(&mut g, &mut cache));
        for (main, want) in &mains {
            assert_eq!(run(main, &g, &it), *want);
        }
        assert!(!Promote.run_function(&mut g, &mut cache));
    }

    /// From source: `out[0] += a[i]` over `@noalias` slices. Bounds-check elimination versions
    /// the loop into a checked copy and a check-free copy that share one exit block, so the
    /// check-free loop's exit is never dedicated -- the case the shared-exit split exists for. At
    /// `-O2` no single-exit loop (the check-free copy, and whatever unrolling makes of it) loads
    /// or stores memory other than `a`'s elements, and every level prints every lane of `out`.
    #[test]
    fn a_slice_reduction_is_carried_in_a_register_past_bce_versioning() {
        let src = concat!(
            "fn red(@noalias mut out: []f32, a: []f32, n: i64) { for i in 0..n { out[0] += a[i]; } }\n",
            "fn main() -> i32 {\n",
            "  let mut o: [f32; 3] = [0.5, 7.0, 9.0];\n",
            "  let a: [f32; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];\n",
            "  let so: []f32 = o; let sa: []f32 = a;\n",
            "  red(so, sa, 6); print(o[0]); print(o[1]); print(o[2]);\n",
            "  red(so, sa, 0); print(o[0]); print(o[1]); print(o[2]);\n",
            "  red(so, sa, 3); print(o[0]); print(o[1]); print(o[2]);\n",
            "  return 0;\n",
            "}\n",
        );
        let mut interner = Interner::new();
        let (module, parse) =
            wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        // 0.5 + 63 = 63.5, then unchanged by zero trips, then + 1 + 2 + 4 = 70.5.
        let expected = "63.5\n7\n9\n63.5\n7\n9\n70.5\n7\n9\n";
        let main = interner.intern("main");
        for level in [0u8, 1, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            assert_eq!(String::from_utf8(stdout).unwrap(), expected, "O{level}");
        }
        let mut optimized = program.clone();
        crate::PassManager::standard(2).run(&mut optimized);
        let red = optimized.function(interner.intern("red")).unwrap();
        let text = wukong_mir::print::print_function(red, &interner);
        let forest = loop_info::analyze_function(red);
        let mut single_exit = 0;
        for id in forest.innermost_first() {
            let l = forest.get(id);
            if l.exits.len() != 1 {
                continue; // the checked copy: its bounds checks are exits too
            }
            single_exit += 1;
            let stores = l
                .blocks
                .iter()
                .flat_map(|b| &red.blocks[b.0 as usize].insts)
                .filter(|i| matches!(i.op, Op::Store { .. }))
                .count();
            assert_eq!(stores, 0, "loop at {:?} still stores\n{text}", l.header);
        }
        assert!(single_exit > 0, "no check-free loop\n{text}");
    }
}
