//! Loop-invariant code motion.
//!
//! Computations inside a loop whose operands do not change across iterations are hoisted to the
//! loop's preheader so they run once instead of every iteration. For kernels this is a major win:
//! the inner loop of a matmul recomputes row/column base addresses (`a[i*n + k]`) whose `i*n` term
//! is invariant in the `k` loop — LICM lifts exactly that out.
//!
//! Natural loops are found from the dominator tree (a back edge `n -> h` where `h` dominates `n`).
//! We hoist only into a loop that already has a preheader — a single out-of-loop predecessor that
//! reaches the header by an unconditional branch and dominates it — rather than synthesizing one,
//! which keeps the transform simple and always legal. Only side-effect-free, non-trapping
//! operations are moved (see `safe_to_hoist`: no stores, calls, vector kernels, allocas, or integer
//! division/remainder), so hoisting a computation onto a path that would not have executed it can
//! never change observable behavior.
//!
//! # Loads
//!
//! A **load** is hoisted only when three separate things hold, because a load is both a memory read
//! and a speculation:
//!
//!  1. its address is loop-invariant (the ordinary operand test);
//!  2. nothing in the loop body can write what it reads — [`crate::alias`] is asked about every
//!     store in the body, and about whether a `call` in the body could reach the location at all;
//!  3. the access is **dereferenceable** — [`AliasInfo::is_dereferenceable`] — because the preheader
//!     runs even when the loop trips zero times, so the hoisted load happens on a path the original
//!     program might never have taken. That is the whole reason the pass refused loads before: an
//!     address the caller handed us may be valid only under the loop's own guard, and speculating a
//!     read through it would turn a program that ran into one that faults.
//!
//! Condition 3 is the tight one, and it is deliberately tight. It holds today only for an address
//! inside an `alloca` of this function at a known in-bounds offset: a stack slot is live for the
//! whole function and default-initialized by both backends, so reading it early can neither fault
//! nor observe anything new. Every `[]T`/struct/tuple local is such a slot, which is where the
//! loop-invariant loads in Wukong code actually live — the fat-pointer header's `data` field, reread
//! on every subscript.

use crate::alias::{type_bytes, AliasInfo};
use crate::fxhash::{FxHashMap, FxHashSet};

use wukong_mir::{BinOp, Function, Inst, Op, Terminator, ValueId};

use crate::{cfg, each_op_use, CfgAnalyses, Pass};

pub struct Licm;

impl Pass for Licm {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // Skip the reachability DFS when the CFG is already fully reachable (the steady state);
        // `all_reachable` is free off the cached rpo `idoms` needs anyway.
        if !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate();
        }
        if f.blocks.len() < 2 {
            return false;
        }
        let alias = AliasInfo::analyze(f);
        // `idom`/`preds` borrow the cache; `hoist_from_loop` mutates `f` (a different object), so
        // the two coexist for the whole loop.
        let (idom, preds) = cache.idoms_and_preds(f);
        let loops = natural_loops(f, idom, preds);

        let mut changed = false;
        for (header, body) in loops {
            if let Some(preheader) = preheader(f, header, &body, preds, idom) {
                let mem = LoopMem::of(f, &body);
                changed |= hoist_from_loop(f, &body, preheader, &alias, &mem);
            }
        }
        changed
    }
}

/// Everything in one loop body that can write memory, summarized once so the hoist test does not
/// rescan the body per candidate. Stores and calls are never hoisted, so this stays valid across
/// every round of `hoist_from_loop`.
struct LoopMem {
    /// `(destination, bytes written)` for every store in the body.
    stores: Vec<(ValueId, u32)>,
    /// Does the body contain a call or a vector-kernel call? Either may write through any pointer
    /// it was handed, so only a stack slot that never escaped this function survives one.
    has_call: bool,
}

impl LoopMem {
    fn of(f: &Function, body: &FxHashSet<u32>) -> LoopMem {
        let mut mem = LoopMem {
            stores: Vec::new(),
            has_call: false,
        };
        for &blk in body {
            for inst in &f.blocks[blk as usize].insts {
                match &inst.op {
                    Op::Store { ptr, value } => mem.stores.push((
                        *ptr,
                        type_bytes(f.value_type(*value)).unwrap_or(u32::MAX),
                    )),
                    Op::Call { .. } | Op::VecKernelCall { .. } => mem.has_call = true,
                    _ => {}
                }
            }
        }
        mem
    }
}

/// May a `load` of `bytes` bytes at `ptr` be lifted out of a loop whose memory effects are `mem`?
///
/// See the module note: no writer in the body may reach it, and the access must be safe to perform
/// speculatively because the preheader also runs on a zero-trip entry.
fn load_hoistable(alias: &AliasInfo, ptr: ValueId, bytes: u32, mem: &LoopMem) -> bool {
    if !alias.is_dereferenceable(ptr, bytes) {
        return false;
    }
    if mem.has_call && !alias.is_private_stack(ptr) {
        return false;
    }
    !mem.stores
        .iter()
        .any(|&(q, qb)| alias.may_alias_sized(ptr, bytes, q, qb))
}

/// Natural loops keyed by header, each mapped to the set of blocks in the loop. Loops that share a
/// header (multiple back edges) are merged.
fn natural_loops(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> Vec<(u32, FxHashSet<u32>)> {
    let mut by_header: FxHashMap<u32, FxHashSet<u32>> = FxHashMap::default();
    for b in &f.blocks {
        let n = b.id.0;
        for s in cfg::successors(&b.term) {
            let h = s.0;
            if dominates(h, n, idom) {
                // Back edge n -> h: collect the nodes that reach n without passing through h.
                let body = by_header.entry(h).or_insert_with(|| {
                    let mut s = FxHashSet::default();
                    s.insert(h);
                    s
                });
                if n != h {
                    let mut stack = vec![n];
                    body.insert(n);
                    while let Some(x) = stack.pop() {
                        for &p in &preds[x as usize] {
                            if body.insert(p) {
                                stack.push(p);
                            }
                        }
                    }
                }
            }
        }
    }
    // Process loops in a deterministic (header-id) order. `HashMap` iteration order is randomized
    // per run, and loop processing order can change what a single LICM pass hoists (e.g. nested
    // loops) — so leaving it unordered makes the emitted MIR nondeterministic run-to-run (M12).
    let mut loops: Vec<(u32, FxHashSet<u32>)> = by_header.into_iter().collect();
    loops.sort_by_key(|(h, _)| *h);
    loops
}

/// Does `a` dominate `b`? Walks the immediate-dominator chain from `b` up to the entry.
fn dominates(a: u32, b: u32, idom: &[u32]) -> bool {
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        let id = idom[x as usize];
        if id == x {
            return false; // reached the entry
        }
        x = id;
    }
}

/// The loop's preheader, if it has a usable one: a single predecessor outside the loop that
/// branches unconditionally to the header and dominates it.
fn preheader(
    f: &Function,
    header: u32,
    body: &FxHashSet<u32>,
    preds: &[Vec<u32>],
    idom: &[u32],
) -> Option<u32> {
    let outside: Vec<u32> = preds[header as usize]
        .iter()
        .copied()
        .filter(|p| !body.contains(p))
        .collect();
    if outside.len() != 1 {
        return None;
    }
    let p = outside[0];
    let unconditional =
        matches!(&f.blocks[p as usize].term, Terminator::Br { target, .. } if target.0 == header);
    if unconditional && dominates(p, header, idom) {
        Some(p)
    } else {
        None
    }
}

/// May this operation be speculatively executed in the preheader (no side effects, cannot trap)?
///
/// `Op::Load` is decided separately by [`load_hoistable`], which needs the loop's memory effects and
/// the alias analysis; everything here is judged on the opcode alone.
fn safe_to_hoist(f: &Function, op: &Op, alias: &AliasInfo, mem: &LoopMem) -> bool {
    match op {
        Op::ConstInt(..)
        | Op::ConstFloat(..)
        | Op::Cmp(..)
        | Op::Neg(..)
        | Op::Not(..)
        | Op::Cast(..)
        | Op::Select(..)
        | Op::Gep { .. }
        | Op::FuncAddr(..)
        | Op::GlobalAddr(..)
        | Op::Splat(..)
        | Op::ExtractLane(..)
        | Op::Iota(..)
        | Op::Fma(..)
        | Op::Sqrt(..)
        | Op::Round(..) => true,
        // Integer division/remainder can trap on a zero divisor, so they are not speculatable.
        Op::Bin(b, ..) => !matches!(b, BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem),
        Op::Load(p, ty) => {
            let _ = f;
            load_hoistable(alias, *p, type_bytes(ty).unwrap_or(u32::MAX), mem)
        }
        Op::Store { .. } | Op::Call { .. } | Op::VecKernelCall { .. } | Op::Alloca(..) => false,
    }
}

fn hoist_from_loop(
    f: &mut Function,
    body: &FxHashSet<u32>,
    preheader: u32,
    alias: &AliasInfo,
    mem: &LoopMem,
) -> bool {
    // Iterate the loop body in a deterministic (block-id) order. The order in which hoisted
    // instructions are appended to the preheader must not depend on `HashSet` iteration order —
    // that order is randomized per run, so using it directly emits nondeterministic MIR (M12). The
    // hoisted instructions in a round are mutually independent, so any fixed order is equally valid.
    let mut body_blocks: Vec<u32> = body.iter().copied().collect();
    body_blocks.sort_unstable();

    // Values defined inside the loop (block parameters and instruction results).
    let mut defined_in_loop: FxHashSet<u32> = FxHashSet::default();
    for &blk in &body_blocks {
        let b = &f.blocks[blk as usize];
        for p in &b.params {
            defined_in_loop.insert(p.0);
        }
        for inst in &b.insts {
            if let Some(r) = inst.result {
                defined_in_loop.insert(r.0);
            }
        }
    }

    // An operand is available in the preheader if it is not defined in the loop, or it has already
    // been hoisted there.
    let mut hoisted: FxHashSet<u32> = FxHashSet::default();
    let mut moved_ops: Vec<Inst> = Vec::new();

    loop {
        // Read phase: which still-in-loop instructions are now invariant?
        let mut found: FxHashSet<u32> = FxHashSet::default();
        for &blk in &body_blocks {
            for inst in &f.blocks[blk as usize].insts {
                let Some(r) = inst.result else { continue };
                if !safe_to_hoist(f, &inst.op, alias, mem) {
                    continue;
                }
                let mut invariant = true;
                each_op_use(&inst.op, &mut |v| {
                    if defined_in_loop.contains(&v.0) && !hoisted.contains(&v.0) {
                        invariant = false;
                    }
                });
                if invariant {
                    found.insert(r.0);
                }
            }
        }
        if found.is_empty() {
            break;
        }
        hoisted.extend(found.iter().copied());

        // Mutate phase: pull the newly invariant instructions out of their loop blocks. All their
        // operands were available *before* this round, so there are no intra-round dependencies and
        // appending them in (deterministic block-id) discovery order is valid.
        for &blk in &body_blocks {
            let insts = std::mem::take(&mut f.blocks[blk as usize].insts);
            let mut kept = Vec::with_capacity(insts.len());
            for inst in insts {
                if inst.result.is_some_and(|r| found.contains(&r.0)) {
                    moved_ops.push(inst);
                } else {
                    kept.push(inst);
                }
            }
            f.blocks[blk as usize].insts = kept;
        }
    }

    if moved_ops.is_empty() {
        return false;
    }
    f.blocks[preheader as usize].insts.extend(moved_ops);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{Builder, CmpOp, MirType};
    use wukong_span::Interner;

    /// Build `bb0 -> bb1(header, guard) -> bb2(body) -> bb1`, running `body` inside bb2. The
    /// candidate load's address is whatever `addr` returns, built in bb0 (so it is loop-invariant).
    /// Returns the finished function and how many loads sit inside the loop.
    fn loop_fn(
        it: &mut Interner,
        name: &str,
        build: impl FnOnce(&mut Builder) -> ValueId,
        body: impl FnOnce(&mut Builder, ValueId),
    ) -> Function {
        let mut b = Builder::new(it.intern(name), MirType::Void);
        let n = b.add_param(MirType::I64);
        let addr = build(&mut b);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let bodyb = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, bodyb, vec![], exit, vec![]);
        b.switch_to(bodyb);
        body(&mut b, addr);
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(None);
        b.finish()
    }

    fn loads_in(f: &Function, blk: usize) -> usize {
        f.blocks[blk]
            .insts
            .iter()
            .filter(|i| matches!(i.op, Op::Load(..)))
            .count()
    }

    /// The `[]T` shape: the fat-pointer header is a stack slot, so re-reading its `data` field every
    /// iteration is both invariant and safe to speculate. The element store goes through the loaded
    /// pointer — unknown provenance — but it cannot reach a slot whose address never escaped.
    #[test]
    fn a_private_stack_load_is_hoisted_out_of_the_loop() {
        let mut it = Interner::new();
        let mut f = loop_fn(
            &mut it,
            "f",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, hdr| {
                let p = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
                let x = b.build(MirType::F32, Op::Load(p, MirType::F32));
                b.build_void(Op::Store { ptr: p, value: x });
            },
        );
        assert_eq!(loads_in(&f, 2), 2);
        let mut cache = CfgAnalyses::default();
        assert!(Licm.run_function(&mut f, &mut cache));
        // The header load moved to the preheader; the element load stays (its address varies).
        assert_eq!(loads_in(&f, 2), 1, "{}", wukong_mir::print::print_function(&f, &it));
        assert_eq!(loads_in(&f, 0), 1, "{}", wukong_mir::print::print_function(&f, &it));
    }

    /// The identical loop, but the address is a *parameter*. Nothing writes it, so it is invariant —
    /// yet the preheader runs even when the loop trips zero times, and whether that pointer is
    /// dereferenceable at all is the caller's business. It must stay put.
    #[test]
    fn a_parameter_load_is_not_speculated_into_the_preheader() {
        let mut it = Interner::new();
        let mut f = loop_fn(
            &mut it,
            "g",
            |b| b.add_param(MirType::Ptr),
            |b, hdr| {
                let p = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
                let _ = b.build(MirType::F32, Op::Load(p, MirType::F32));
            },
        );
        assert_eq!(loads_in(&f, 2), 2);
        let mut cache = CfgAnalyses::default();
        Licm.run_function(&mut f, &mut cache);
        assert_eq!(loads_in(&f, 2), 2, "{}", wukong_mir::print::print_function(&f, &it));
    }

    /// A store in the loop that may alias the slot keeps its load inside.
    #[test]
    fn a_may_aliasing_store_in_the_loop_pins_the_load() {
        let mut it = Interner::new();
        let mut f = loop_fn(
            &mut it,
            "h",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, slot| {
                let x = b.build(MirType::I64, Op::Load(slot, MirType::I64));
                b.build_void(Op::Store { ptr: slot, value: x });
            },
        );
        assert_eq!(loads_in(&f, 2), 1);
        let mut cache = CfgAnalyses::default();
        Licm.run_function(&mut f, &mut cache);
        assert_eq!(loads_in(&f, 2), 1, "{}", wukong_mir::print::print_function(&f, &it));
    }

    /// A call in the loop is opaque, but it still cannot reach a slot this function never published.
    /// The same load behind an escaped slot must stay.
    #[test]
    fn a_call_in_the_loop_pins_only_reachable_loads() {
        let mut it = Interner::new();
        let sink = it.intern("sink");
        let mut f = loop_fn(
            &mut it,
            "k",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, slot| {
                let _ = b.build(MirType::I64, Op::Load(slot, MirType::I64));
                b.build_void(Op::Call {
                    func: sink,
                    args: vec![],
                });
            },
        );
        let mut cache = CfgAnalyses::default();
        assert!(Licm.run_function(&mut f, &mut cache));
        assert_eq!(loads_in(&f, 2), 0, "{}", wukong_mir::print::print_function(&f, &it));

        let mut escaped = loop_fn(
            &mut it,
            "k2",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, slot| {
                let _ = b.build(MirType::I64, Op::Load(slot, MirType::I64));
                b.build_void(Op::Call {
                    func: sink,
                    args: vec![slot],
                });
            },
        );
        let mut cache = CfgAnalyses::default();
        Licm.run_function(&mut escaped, &mut cache);
        assert_eq!(
            loads_in(&escaped, 2),
            1,
            "{}",
            wukong_mir::print::print_function(&escaped, &it)
        );
    }
}
