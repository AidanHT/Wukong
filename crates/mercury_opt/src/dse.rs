//! Dead-store elimination (intra-block).
//!
//! The dual of CSE's load forwarding: a `store slot, v1` is dead if a later `store slot, v2` in the
//! same block overwrites it with no intervening read of `slot`. We track, per alloca slot, the
//! index of a pending (not-yet-read) store; a second store to the same slot marks the first dead.
//!
//! Reads clear a slot's pending store (the value was observed): a `load slot`, or — conservatively
//! — any `load`/`store` through an unknown pointer, or any `call` (which may read through pointers).
//! Pending stores that survive to the end of the block are kept: a successor block may read them.

use std::collections::{HashMap, HashSet};

use mercury_mir::{Function, Op};

use crate::Pass;

pub struct Dse;

impl Pass for Dse {
    fn name(&self) -> &'static str {
        "dse"
    }

    fn run_function(&self, f: &mut Function) -> bool {
        // Allocas are function-global value ids; collect them once.
        let mut allocas: HashSet<u32> = HashSet::new();
        for b in &f.blocks {
            for inst in &b.insts {
                if let (Some(r), Op::Alloca(_)) = (inst.result, &inst.op) {
                    allocas.insert(r.0);
                }
            }
        }

        let mut changed = false;
        for b in &mut f.blocks {
            // slot value-id -> index of the last store to it that has not yet been read.
            let mut pending: HashMap<u32, usize> = HashMap::new();
            let mut dead: Vec<usize> = Vec::new();

            for (i, inst) in b.insts.iter().enumerate() {
                match &inst.op {
                    Op::Store { ptr, .. } => {
                        if allocas.contains(&ptr.0) {
                            // Overwrites any prior unread store to the same slot.
                            if let Some(prev) = pending.insert(ptr.0, i) {
                                dead.push(prev);
                            }
                        } else {
                            // Unknown destination may alias anything; conservatively forget.
                            pending.clear();
                        }
                    }
                    Op::Load(ptr, _) => {
                        if allocas.contains(&ptr.0) {
                            pending.remove(&ptr.0);
                        } else {
                            pending.clear();
                        }
                    }
                    Op::Call { .. } => {
                        // A call may read through any pointer it was given.
                        pending.clear();
                    }
                    _ => {}
                }
            }

            if !dead.is_empty() {
                let dead: HashSet<usize> = dead.into_iter().collect();
                let mut i = 0;
                b.insts.retain(|_| {
                    let keep = !dead.contains(&i);
                    i += 1;
                    keep
                });
                changed = true;
            }
        }
        changed
    }
}
