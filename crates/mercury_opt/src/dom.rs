//! Dominator analysis: immediate dominators, dominance frontiers, and the dominator tree.
//!
//! Uses the Cooper–Harvey–Kennedy iterative algorithm ("A Simple, Fast Dominance Algorithm"),
//! which is compact, fast in practice, and easy to verify. All blocks must be reachable from the
//! entry — run [`crate::cfg::prune_unreachable`] first. The dominance frontier is what `mem2reg`
//! uses to decide where SSA merges (block parameters) are needed.

use mercury_mir::Function;

const NONE: u32 = u32::MAX;

/// Immediate dominators, indexed by block id, from a precomputed reverse postorder and predecessor
/// map. `idom[entry] == entry`. Every block's idom is set (the input must be fully reachable). The
/// analysis cache holds `rpo`/`preds` and reuses them across passes rather than recomputing them
/// here per call.
pub(crate) fn idoms_from(f: &Function, rpo: &[u32], preds: &[Vec<u32>]) -> Vec<u32> {
    let n = f.blocks.len();
    let mut rpo_num = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_num[b as usize] = i;
    }

    let mut idom = vec![NONE; n];
    let entry = f.entry.0;
    idom[entry as usize] = entry;

    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo {
            if b == entry {
                continue;
            }
            let mut new_idom = NONE;
            for &p in &preds[b as usize] {
                // Skip predecessors that are themselves unreachable (no rpo number) or not yet
                // processed in this round (no idom assigned).
                if rpo_num[p as usize] == usize::MAX || idom[p as usize] == NONE {
                    continue;
                }
                new_idom = if new_idom == NONE {
                    p
                } else {
                    intersect(p, new_idom, &idom, &rpo_num)
                };
            }
            if new_idom != NONE && idom[b as usize] != new_idom {
                idom[b as usize] = new_idom;
                changed = true;
            }
        }
    }
    idom
}

/// Walk up the dominator tree from two nodes until they meet (their common dominator).
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

/// Dominance frontiers, indexed by block id, from precomputed idoms and predecessors: `df[b]` is
/// the set of blocks where `b`'s dominance ends — exactly the blocks that may need a phi for a value
/// defined in `b`.
pub(crate) fn dominance_frontiers_from(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = f.blocks.len();
    let mut df: Vec<Vec<u32>> = vec![Vec::new(); n];
    for b in 0..n as u32 {
        let ps = &preds[b as usize];
        if ps.len() < 2 {
            continue; // a join requires at least two distinct predecessors
        }
        let id_b = idom[b as usize];
        if id_b == NONE {
            continue;
        }
        for &p in ps {
            let mut runner = p;
            while runner != id_b && runner != NONE {
                if !df[runner as usize].contains(&b) {
                    df[runner as usize].push(b);
                }
                runner = idom[runner as usize];
            }
        }
    }
    df
}

/// Children of each block in the dominator tree, indexed by block id, in ascending child order.
pub(crate) fn dom_children(f: &Function, idom: &[u32]) -> Vec<Vec<u32>> {
    let n = f.blocks.len();
    let mut children = vec![Vec::new(); n];
    for b in 0..n as u32 {
        let id = idom[b as usize];
        if id != NONE && id != b {
            children[id as usize].push(b);
        }
    }
    children
}
