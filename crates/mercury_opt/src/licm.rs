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
//! operations are moved (no loads, stores, calls, or integer division), so hoisting a computation
//! onto a path that would not have executed it can never change observable behavior.

use std::collections::{HashMap, HashSet};

use mercury_mir::{BinOp, Function, Inst, Op, Terminator};

use crate::{cfg, dom, each_op_use, Pass};

pub struct Licm;

impl Pass for Licm {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run_function(&self, f: &mut Function) -> bool {
        cfg::prune_unreachable(f);
        if f.blocks.len() < 2 {
            return false;
        }
        let idom = dom::idoms(f);
        let preds = cfg::predecessors(f);
        let loops = natural_loops(f, &idom, &preds);

        let mut changed = false;
        for (header, body) in loops {
            if let Some(preheader) = preheader(f, header, &body, &preds, &idom) {
                changed |= hoist_from_loop(f, &body, preheader);
            }
        }
        changed
    }
}

/// Natural loops keyed by header, each mapped to the set of blocks in the loop. Loops that share a
/// header (multiple back edges) are merged.
fn natural_loops(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> Vec<(u32, HashSet<u32>)> {
    let mut by_header: HashMap<u32, HashSet<u32>> = HashMap::new();
    for b in &f.blocks {
        let n = b.id.0;
        for s in cfg::successors(&b.term) {
            let h = s.0;
            if dominates(h, n, idom) {
                // Back edge n -> h: collect the nodes that reach n without passing through h.
                let body = by_header.entry(h).or_insert_with(|| {
                    let mut s = HashSet::new();
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
    by_header.into_iter().collect()
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
    body: &HashSet<u32>,
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
fn safe_to_hoist(op: &Op) -> bool {
    match op {
        Op::ConstInt(..)
        | Op::ConstFloat(..)
        | Op::Cmp(..)
        | Op::Neg(..)
        | Op::Not(..)
        | Op::Cast(..)
        | Op::Select(..)
        | Op::Gep { .. }
        | Op::FuncAddr(..) => true,
        // Integer division/remainder can trap on a zero divisor, so they are not speculatable.
        Op::Bin(b, ..) => !matches!(b, BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem),
        Op::Load(..) | Op::Store { .. } | Op::Call { .. } | Op::Alloca(..) => false,
    }
}

fn hoist_from_loop(f: &mut Function, body: &HashSet<u32>, preheader: u32) -> bool {
    // Values defined inside the loop (block parameters and instruction results).
    let mut defined_in_loop: HashSet<u32> = HashSet::new();
    for &blk in body {
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
    let mut hoisted: HashSet<u32> = HashSet::new();
    let mut moved_ops: Vec<Inst> = Vec::new();

    loop {
        // Read phase: which still-in-loop instructions are now invariant?
        let mut found: HashSet<u32> = HashSet::new();
        for &blk in body {
            for inst in &f.blocks[blk as usize].insts {
                let Some(r) = inst.result else { continue };
                if !safe_to_hoist(&inst.op) {
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
        // appending them in discovery order is valid.
        for &blk in body {
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
