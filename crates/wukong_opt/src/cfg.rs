//! Control-flow-graph analyses shared across optimization passes.
//!
//! Successors/predecessors, reachability, reverse postorder, and unreachable-block pruning.
//! These are the building blocks for the dominator analysis (`dom`) and the SSA-construction
//! pass (`mem2reg`). Everything operates on the block-parameter CFG the front-end emits.

use std::collections::BTreeMap;

use wukong_mir::{BlockId, Function, Terminator};

/// The successor blocks of a terminator. A `cond_br` whose two arms target the same block is
/// reported once (it is a single CFG edge as far as reachability/dominance are concerned).
pub(crate) fn successors(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => {
            if then_blk == else_blk {
                vec![*then_blk]
            } else {
                vec![*then_blk, *else_blk]
            }
        }
        Terminator::Ret(_) | Terminator::Unreachable => vec![],
    }
}

/// Predecessor blocks (unique, in source order) for every block, indexed by block id.
pub(crate) fn predecessors(f: &Function) -> Vec<Vec<u32>> {
    let mut preds = vec![Vec::new(); f.blocks.len()];
    for b in &f.blocks {
        for s in successors(&b.term) {
            let list = &mut preds[s.0 as usize];
            if !list.contains(&b.id.0) {
                list.push(b.id.0);
            }
        }
    }
    preds
}

fn post_dfs(f: &Function, b: u32, visited: &mut [bool], post: &mut Vec<u32>) {
    if visited[b as usize] {
        return;
    }
    visited[b as usize] = true;
    for s in successors(&f.blocks[b as usize].term) {
        post_dfs(f, s.0, visited, post);
    }
    post.push(b);
}

/// Reverse postorder of the blocks reachable from the entry (entry first). Every block reachable
/// from entry appears exactly once; unreachable blocks are omitted.
pub(crate) fn reverse_postorder(f: &Function) -> Vec<u32> {
    let mut visited = vec![false; f.blocks.len()];
    let mut post = Vec::new();
    post_dfs(f, f.entry.0, &mut visited, &mut post);
    post.reverse();
    post
}

/// A per-block flag: is the block reachable from the entry?
pub(crate) fn reachable(f: &Function) -> Vec<bool> {
    let mut visited = vec![false; f.blocks.len()];
    let mut post = Vec::new();
    post_dfs(f, f.entry.0, &mut visited, &mut post);
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
    let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
    let mut next = 0u32;
    for (i, &live) in reach.iter().enumerate() {
        if live {
            remap.insert(i as u32, next);
            next += 1;
        }
    }

    let mut new_blocks = Vec::with_capacity(next as usize);
    for (i, mut b) in f.blocks.drain(..).enumerate() {
        if !reach[i] {
            continue;
        }
        b.id = BlockId(remap[&(i as u32)]);
        remap_term(&mut b.term, &remap);
        new_blocks.push(b);
    }
    f.blocks = new_blocks;
    f.entry = BlockId(remap[&f.entry.0]);
    true
}

fn remap_term(t: &mut Terminator, remap: &BTreeMap<u32, u32>) {
    match t {
        Terminator::Br { target, .. } => target.0 = remap[&target.0],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => {
            then_blk.0 = remap[&then_blk.0];
            else_blk.0 = remap[&else_blk.0];
        }
        Terminator::Ret(_) | Terminator::Unreachable => {}
    }
}
