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
//!
//! # Alias facts
//!
//! The cache also carries the function's [`AliasInfo`], which `cse`, `dse` and `licm` all consult.
//! Unlike the CFG analyses it depends on *instructions*, so it is only valid until the next
//! mutation of any kind. The pass manager knows exactly when that happens — a pass reports it — so
//! the alias facts are shared only in a cache the manager drives ([`CfgAnalyses::managed`]): built
//! on first request, dropped by [`CfgAnalyses::note_mutation`] after any pass that changed the
//! function, and by [`CfgAnalyses::invalidate`] (pruning removes values). Within one sweep where
//! the earlier memory passes find nothing to do, the later ones reuse the first one's facts instead
//! of each re-running the provenance, constant and escape sweeps — the case the final, all-no-op
//! confirmation sweep of every function always hits. A cache built by hand (a test, or a pass
//! driven outside the manager) is unmanaged and recomputes on every request, exactly as before.

use std::rc::Rc;

use wukong_mir::Function;

use crate::alias::AliasInfo;
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
    /// Provenance facts, retained only in a managed cache (see the module note).
    alias: Option<Rc<AliasInfo>>,
    managed: bool,
}

impl CfgAnalyses {
    /// A cache driven by the pass manager, which reports every mutation through
    /// [`CfgAnalyses::note_mutation`] and may therefore share instruction-level facts across passes.
    pub(crate) fn managed() -> CfgAnalyses {
        CfgAnalyses {
            managed: true,
            ..CfgAnalyses::default()
        }
    }

    /// Drop every cached analysis. Must be called after any mutation that changes the block set or
    /// a terminator's successor edges (pruning that removed blocks, a `cond_br` folded to a `br`, a
    /// straight-line block merge). Cheap when already empty.
    pub(crate) fn invalidate(&mut self) {
        self.preds = None;
        self.rpo = None;
        self.idom = None;
        self.df = None;
        self.children = None;
        self.alias = None;
    }

    /// A pass changed the function's instructions: instruction-level facts are stale. The CFG
    /// analyses stay — a pass that changed edges has already called [`CfgAnalyses::invalidate`].
    pub(crate) fn note_mutation(&mut self) {
        self.alias = None;
    }

    /// The function's alias facts. Shared across passes in a managed cache until the next mutation;
    /// computed afresh on every call otherwise. Returned by `Rc` so the caller holds no borrow of the
    /// cache while it also asks for dominators.
    pub(crate) fn alias(&mut self, f: &Function) -> Rc<AliasInfo> {
        if !self.managed {
            return Rc::new(AliasInfo::analyze(f));
        }
        self.alias
            .get_or_insert_with(|| Rc::new(AliasInfo::analyze(f)))
            .clone()
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
            let df = dom::dominance_frontiers_from(
                f,
                self.idom.as_ref().unwrap(),
                self.preds.as_ref().unwrap(),
            );
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

    /// Dominator children and predecessors together, for forwarding facts across a unique edge.
    pub(crate) fn children_and_preds(&mut self, f: &Function) -> (&[Vec<u32>], &[Vec<u32>]) {
        self.ensure_children(f);
        self.ensure_preds(f);
        (
            self.children.as_ref().unwrap(),
            self.preds.as_ref().unwrap(),
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{Builder, MirType, Op};
    use wukong_span::Interner;

    #[test]
    fn alias_facts_are_shared_only_inside_a_managed_cache_and_until_a_mutation() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I64);
        let slot = b.alloca(MirType::I64);
        let v = b.build(MirType::I64, Op::Load(slot, MirType::I64));
        b.ret(Some(v));
        let f = b.finish();

        let mut managed = CfgAnalyses::managed();
        let first = managed.alias(&f);
        assert!(Rc::ptr_eq(&first, &managed.alias(&f)), "reused while clean");
        managed.note_mutation();
        let second = managed.alias(&f);
        assert!(!Rc::ptr_eq(&first, &second), "a mutation drops the facts");
        managed.invalidate();
        assert!(
            !Rc::ptr_eq(&second, &managed.alias(&f)),
            "so does a CFG change"
        );

        // A hand-driven cache cannot know when its function changed: never share.
        let mut unmanaged = CfgAnalyses::default();
        let a = unmanaged.alias(&f);
        assert!(!Rc::ptr_eq(&a, &unmanaged.alias(&f)));
    }
}
