//! The memo that keeps the GEMM bias-alias preflight (`GemmBiasAlias` in `lib.rs`) linear.
//!
//! The preflight decides, before any kernel is substituted, whether a recognized GEMM inside a
//! loop's subtree rereads its own output as the per-column bias (a fused epilogue that would see
//! columns the kernel already overwrote). It has to look at the WHOLE subtree, because outlining
//! or dispatching an enclosing loop first would hide the dependency from the inner loop's decline,
//! and `FnLowerer::lower_for` asks it at every loop. Unmemoized, a loop at depth `k` of a nest was
//! walked again by each of its `k` ancestors and its own visit: every guard probe -- one run of the
//! GEMM recognizers on one loop or one statement window -- happened once per enclosing `for`, so the
//! recognizer work of a `d`-deep nest grew as `d * (d + 1) / 2` (N65).
//!
//! # The key is the node's address, scoped to the borrow that keeps it alive
//!
//! A `NodeId` key is unsound: a fused loop body (`try_fuse_run`) reuses the first member's id,
//! synthesized tails carry `NodeId::DUMMY`, and both are different code under the same id. What
//! does identify a node is its address -- but only while the node is alive and unchanged, since a
//! temporary built later can be allocated where a dropped one was. Both are exactly what a borrow
//! guarantees, and every walk that records an entry starts from the `pat` / `iter` / `body`
//! references one `lower_for` call holds: the AST has no interior mutability, so for that call's
//! whole dynamic extent every node reachable from them stays put and keeps its contents. Entries
//! are therefore released when the `lower_for` whose walk recorded them returns (`mark` /
//! `release`, strictly LIFO), and an address can never be looked up after the node it named is
//! gone.
//!
//! # What an entry asserts
//!
//! Only a NEGATIVE result is recorded: "the walk of this loop (or this block) found no GEMM whose
//! bias can overlap its output, under this `alias_free` bit". The preflight is a pure function of
//! the nodes, `SemaResult`, the interner and that bit (no scope or MIR state reaches it), and an
//! `||` that comes back `false` evaluated every operand, so each sub-loop and sub-block the walk
//! reached is clean in its own right. That is what lets `lower_for` skip the walk for a loop an
//! ancestor's walk already covered, `lower_block_stmts` skip the per-statement window probe of a
//! block whose windows were all probed, and a walk skip a sub-block an earlier walk proved. A
//! positive result is never cached: it turns recognition and outlining off for the subtree, so the
//! nested `lower_for` visits do not ask again anyway.
//!
//! The memo changes no answer, only how often one is computed: the recognizer census is
//! byte-identical with and without it, and `bias_memo_tests.rs` lowers the whole corpus both ways
//! and compares the MIR. [`count_probe`] counts the guard's recognizer runs for its linear-growth
//! test.

use std::cell::RefCell;

use wukong_ast::{Block, ForIter, Pattern};
use wukong_span::FxHashSet as HashSet;

/// One proven-clean node: a loop (its pattern, iterator and body, which together are what the
/// preflight reads) or a block, under the `alias_free` bit the walk ran with.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Clean {
    Loop {
        pat: usize,
        iter: usize,
        body: usize,
        alias_free: bool,
    },
    Block {
        block: usize,
        alias_free: bool,
    },
}

impl Clean {
    fn of_loop(pat: &Pattern, iter: &ForIter, body: &Block, alias_free: bool) -> Clean {
        Clean::Loop {
            pat: std::ptr::from_ref(pat).addr(),
            iter: std::ptr::from_ref(iter).addr(),
            body: std::ptr::from_ref(body).addr(),
            alias_free,
        }
    }

    fn of_block(block: &Block, alias_free: bool) -> Clean {
        Clean::Block {
            block: std::ptr::from_ref(block).addr(),
            alias_free,
        }
    }
}

/// The clean set plus the insertion log that scopes it (see the module docs). Interior mutability
/// because the preflight walks through `&self`; one per `FnLowerer`.
#[derive(Default)]
pub(crate) struct BiasAliasMemo {
    clean: RefCell<HashSet<Clean>>,
    log: RefCell<Vec<Clean>>,
}

impl BiasAliasMemo {
    /// The scope a `lower_for` opens: everything recorded after this is released by
    /// [`Self::release`] with the same mark.
    pub(crate) fn mark(&self) -> usize {
        self.log.borrow().len()
    }

    /// Forget every entry recorded since `mark` -- the nodes they name may not outlive the call
    /// that opened the scope.
    pub(crate) fn release(&self, mark: usize) {
        let mut log = self.log.borrow_mut();
        if log.len() > mark {
            let mut clean = self.clean.borrow_mut();
            for entry in log.drain(mark..) {
                clean.remove(&entry);
            }
        }
    }

    pub(crate) fn loop_clean(
        &self,
        pat: &Pattern,
        iter: &ForIter,
        body: &Block,
        alias_free: bool,
    ) -> bool {
        enabled()
            && self
                .clean
                .borrow()
                .contains(&Clean::of_loop(pat, iter, body, alias_free))
    }

    pub(crate) fn block_clean(&self, block: &Block, alias_free: bool) -> bool {
        enabled()
            && self
                .clean
                .borrow()
                .contains(&Clean::of_block(block, alias_free))
    }

    pub(crate) fn record_loop(
        &self,
        pat: &Pattern,
        iter: &ForIter,
        body: &Block,
        alias_free: bool,
    ) {
        self.record(Clean::of_loop(pat, iter, body, alias_free));
    }

    pub(crate) fn record_block(&self, block: &Block, alias_free: bool) {
        self.record(Clean::of_block(block, alias_free));
    }

    /// Log only a NEW entry, so a nested scope's release never removes one an enclosing scope
    /// recorded first.
    fn record(&self, entry: Clean) {
        if enabled() && self.clean.borrow_mut().insert(entry) {
            self.log.borrow_mut().push(entry);
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Guard probes run on this thread: one per run of the GEMM recognizers on a loop
    /// (`GemmBiasAlias::loop_or_body`) or a statement window (`GemmBiasAlias::window`).
    static PROBES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Test builds can switch the memo off on this thread, so a test can lower the same program
    /// both ways and compare the MIR and the probe count.
    static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Is the memo consulted and filled? Always outside test builds.
#[inline]
fn enabled() -> bool {
    #[cfg(test)]
    {
        !DISABLED.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// Switch the memo off (`true`) or back on for this thread; returns the previous setting.
#[cfg(test)]
pub(crate) fn set_disabled(off: bool) -> bool {
    DISABLED.with(|d| d.replace(off))
}

/// Count one guard probe (test builds only; a no-op otherwise).
#[inline]
pub(crate) fn count_probe() {
    #[cfg(test)]
    PROBES.with(|p| p.set(p.get() + 1));
}

/// Drain this thread's guard-probe count.
#[cfg(test)]
pub(crate) fn take_probes() -> u64 {
    PROBES.with(|p| p.replace(0))
}
