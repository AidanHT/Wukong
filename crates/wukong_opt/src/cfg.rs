//! Control-flow-graph analyses shared across optimization passes.
//!
//! Successors/predecessors, reachability, reverse postorder, and unreachable-block pruning.
//! These are the building blocks for the dominator analysis (`dom`) and the SSA-construction
//! pass (`mem2reg`). Everything operates on the block-parameter CFG the front-end emits.

use wukong_mir::{BlockId, Function, Terminator};

/// The successor blocks of a terminator. A `cond_br` whose two arms target the same block is
/// reported once (it is a single CFG edge as far as reachability/dominance are concerned), and so
/// is a block that several `switch` edges reach: the distinct targets, in first-edge order.
pub(crate) fn successors(t: &Terminator) -> impl DoubleEndedIterator<Item = BlockId> {
    // Every terminator but a `switch` has at most two distinct successors. Keep those inline
    // instead of allocating a Vec for every visited block in every CFG consumer.
    let targets = match t {
        Terminator::Br { target, .. } => [Some(*target), None],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => [Some(*then_blk), (then_blk != else_blk).then_some(*else_blk)],
        Terminator::Ret(_) | Terminator::Unreachable => [None, None],
        Terminator::Switch { .. } => {
            let mut distinct: Vec<BlockId> = Vec::new();
            t.for_each_edge(|s, _| {
                if !distinct.contains(&s) {
                    distinct.push(s);
                }
            });
            return Successors::Many(distinct.into_iter());
        }
    };
    Successors::Inline(targets.into_iter().flatten())
}

/// [`successors`]' iterator: the inline pair every two-way terminator fits in, or the distinct
/// targets of a `switch`.
enum Successors {
    Inline(std::iter::Flatten<std::array::IntoIter<Option<BlockId>, 2>>),
    Many(std::vec::IntoIter<BlockId>),
}

impl Iterator for Successors {
    type Item = BlockId;

    fn next(&mut self) -> Option<BlockId> {
        match self {
            Successors::Inline(it) => it.next(),
            Successors::Many(it) => it.next(),
        }
    }
}

impl DoubleEndedIterator for Successors {
    fn next_back(&mut self) -> Option<BlockId> {
        match self {
            Successors::Inline(it) => it.next_back(),
            Successors::Many(it) => it.next_back(),
        }
    }
}

/// Predecessor blocks (unique, in source order) for every block, indexed by block id.
pub(crate) fn predecessors(f: &Function) -> Vec<Vec<u32>> {
    let mut preds = vec![Vec::new(); f.blocks.len()];
    for b in &f.blocks {
        for s in successors(&b.term) {
            // Each source block is visited once and successors() already deduplicates its arms.
            // Searching the growing predecessor list adds quadratic work at a large join.
            preds[s.0 as usize].push(b.id.0);
        }
    }
    preds
}

/// Reverse postorder of the blocks reachable from the entry (entry first). Every block reachable
/// from entry appears exactly once; unreachable blocks are omitted.
pub(crate) fn reverse_postorder(f: &Function) -> Vec<u32> {
    let mut visited = vec![false; f.blocks.len()];
    let mut post = Vec::with_capacity(f.blocks.len());
    // Explicit enter/exit events reproduce recursive DFS exactly, including then-before-else
    // traversal. Deep generated control flow consumes heap space instead of the host call stack.
    let mut work = vec![(f.entry.0, false)];
    while let Some((b, exiting)) = work.pop() {
        if exiting {
            post.push(b);
        } else if !visited[b as usize] {
            visited[b as usize] = true;
            work.push((b, true));
            for s in successors(&f.blocks[b as usize].term).rev() {
                work.push((s.0, false));
            }
        }
    }
    post.reverse();
    post
}

/// A per-block flag: is the block reachable from the entry?
pub(crate) fn reachable(f: &Function) -> Vec<bool> {
    let mut visited = vec![false; f.blocks.len()];
    let mut work = vec![f.entry.0];
    visited[f.entry.0 as usize] = true;
    while let Some(b) = work.pop() {
        for s in successors(&f.blocks[b as usize].term) {
            if !visited[s.0 as usize] {
                visited[s.0 as usize] = true;
                work.push(s.0);
            }
        }
    }
    visited
}

/// Remove blocks not reachable from the entry, compacting and renumbering the remainder. Returns
/// whether anything was removed. Terminator targets and the function entry are rewritten to the
/// new ids. Required before dominance analysis, which is only defined on reachable blocks.
pub(crate) fn prune_unreachable(f: &mut Function) -> bool {
    let reach = reachable(f);
    if reach.iter().all(|&r| r) {
        return false;
    }

    // Old block id -> new (compacted) id, preserving order.
    let mut remap = vec![u32::MAX; f.blocks.len()];
    let mut next = 0u32;
    for (i, &live) in reach.iter().enumerate() {
        if live {
            remap[i] = next;
            next += 1;
        }
    }

    let mut new_blocks = Vec::with_capacity(next as usize);
    for (i, mut b) in f.blocks.drain(..).enumerate() {
        if !reach[i] {
            continue;
        }
        b.id = BlockId(remap[i]);
        remap_term(&mut b.term, &remap);
        new_blocks.push(b);
    }
    f.blocks = new_blocks;
    f.entry = BlockId(remap[f.entry.0 as usize]);
    true
}

fn remap_term(t: &mut Terminator, remap: &[u32]) {
    match t {
        Terminator::Br { target, .. } => target.0 = remap[target.0 as usize],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => {
            then_blk.0 = remap[then_blk.0 as usize];
            else_blk.0 = remap[else_blk.0 as usize];
        }
        Terminator::Switch { cases, default, .. } => {
            for c in cases.iter_mut() {
                c.target.0 = remap[c.target.0 as usize];
            }
            default.0 = remap[default.0 as usize];
        }
        Terminator::Ret(_) | Terminator::Unreachable => {}
    }
}
