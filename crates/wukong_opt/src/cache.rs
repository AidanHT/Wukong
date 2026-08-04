//! A per-function CFG/dominator analysis cache for the pass fixpoint.
//!
//! `predecessors`, `reverse_postorder`, `idoms`, `dominance_frontiers`, and `dom_children` all
//! depend **only** on the block set and each terminator's successor edges — never on instructions,
//! block parameters, or edge arguments. Across the standard pipeline the only transforms that change
//! that structure are `simplify-cfg` (folding a `cond_br` to a `br`, merging straight-line blocks,
//! then pruning) and the unreachable-block pruning that mem2reg/cse/licm run first to make dominance
//! well-defined — which is a no-op, and so free, once nothing is unreachable. Everything else
//! (mem2reg's phi placement and renaming, simplify, simplify-phis, dce, dse, and licm's hoisting)
//! rewrites/moves/removes instructions and rewires block parameters but leaves the block graph
//! intact.
//!
//! So instead of every pass rebuilding these analyses from scratch on each fixpoint iteration (the
//! old code recomputed `idoms` — an O(V·E) iterative fixpoint — and `predecessors` several times per
//! round), a single cache computes each lazily on first use and hands the same result to every pass
//! until a structural mutation calls [`CfgAnalyses::invalidate`]. Over the tail of the fixpoint,
//! where cse/licm/dse keep firing on a CFG that no longer changes shape, the dominator analyses are
//! computed once and reused across all remaining iterations.
//!
//! The cache is created once per function by the pass manager and threaded through every pass; the
//! passes that restructure the CFG are responsible for invalidating it (see their `run_function`).

use wukong_mir::Function;

use crate::{cfg, dom};

/// Lazily-computed, structurally-invalidated CFG/dominator analyses for one function. Opaque to
/// callers outside the crate — it is a parameter of the (crate-internal) pass fixpoint only.
#[derive(Default)]
pub struct CfgAnalyses {
    preds: Option<Vec<Vec<u32>>>,
    rpo: Option<Vec<u32>>,
    idom: Option<Vec<u32>>,
    df: Option<Vec<Vec<u32>>>,
    children: Option<Vec<Vec<u32>>>,
}

impl CfgAnalyses {
    /// Drop every cached analysis. Must be called after any mutation that changes the block set or
    /// a terminator's successor edges (pruning that removed blocks, a `cond_br` folded to a `br`, a
    /// straight-line block merge). Cheap when already empty.
    pub(crate) fn invalidate(&mut self) {
        self.preds = None;
        self.rpo = None;
        self.idom = None;
        self.df = None;
        self.children = None;
    }

    fn ensure_preds(&mut self, f: &Function) {
        if self.preds.is_none() {
            self.preds = Some(cfg::predecessors(f));
        }
    }

    fn ensure_rpo(&mut self, f: &Function) {
        if self.rpo.is_none() {
            self.rpo = Some(cfg::reverse_postorder(f));
        }
    }

    fn ensure_idom(&mut self, f: &Function) {
        if self.idom.is_none() {
            self.ensure_rpo(f);
            self.ensure_preds(f);
            let idom = dom::idoms_from(f, self.rpo.as_ref().unwrap(), self.preds.as_ref().unwrap());
            self.idom = Some(idom);
        }
    }

    fn ensure_df(&mut self, f: &Function) {
        if self.df.is_none() {
            self.ensure_idom(f);
            self.ensure_preds(f);
            let df =
                dom::dominance_frontiers_from(f, self.idom.as_ref().unwrap(), self.preds.as_ref().unwrap());
            self.df = Some(df);
        }
    }

    fn ensure_children(&mut self, f: &Function) {
        if self.children.is_none() {
            self.ensure_idom(f);
            let children = dom::dom_children(f, self.idom.as_ref().unwrap());
            self.children = Some(children);
        }
    }

    /// Whether every block is reachable from the entry — i.e. `prune_unreachable` would be a no-op.
    /// The reverse postorder visits exactly the reachable blocks, so this is a free comparison once
    /// `rpo` is cached (and `idoms`, which every dominance user needs, computes `rpo` anyway). Lets
    /// the prune-first passes skip their reachability DFS in the steady state where nothing is dead.
    pub(crate) fn all_reachable(&mut self, f: &Function) -> bool {
        self.ensure_rpo(f);
        self.rpo.as_ref().unwrap().len() == f.blocks.len()
    }

    /// Predecessor lists, indexed by block id.
    pub(crate) fn predecessors(&mut self, f: &Function) -> &[Vec<u32>] {
        self.ensure_preds(f);
        self.preds.as_ref().unwrap()
    }

    /// Dominator-tree children, indexed by block id.
    pub(crate) fn dom_children(&mut self, f: &Function) -> &[Vec<u32>] {
        self.ensure_children(f);
        self.children.as_ref().unwrap()
    }

    /// Immediate dominators and predecessors together (LICM needs both live at once).
    pub(crate) fn idoms_and_preds(&mut self, f: &Function) -> (&[u32], &[Vec<u32>]) {
        self.ensure_idom(f);
        self.ensure_preds(f);
        (self.idom.as_ref().unwrap(), self.preds.as_ref().unwrap())
    }

    /// Dominance frontiers and dominator-tree children together (mem2reg needs both live at once).
    pub(crate) fn df_and_children(&mut self, f: &Function) -> (&[Vec<u32>], &[Vec<u32>]) {
        self.ensure_df(f);
        self.ensure_children(f);
        (self.df.as_ref().unwrap(), self.children.as_ref().unwrap())
    }
}
