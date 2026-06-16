//! Local value numbering (LVN) with alloca-aware load forwarding.
//!
//! The front-end emits one `alloca` per local and `load`/`store` on every use (mem2reg is left to
//! LLVM). Pure common-subexpression elimination alone is therefore weak here: two textually equal
//! expressions read their operands through *separate* `load` instructions, so their operand value
//! ids differ. To make CSE bite, this pass also forwards loads.
//!
//! Within a single basic block we track, per `alloca` slot, the value it currently holds:
//!   * a `store slot, v` makes `v` the current value of `slot`;
//!   * the first `load slot` becomes the current value; later `load slot` reuse it;
//!   * a store through an *unknown* pointer (e.g. a `gep` result) or any `call` conservatively
//!     forgets all slot contents, since it may alias anything.
//!
//! With loads forwarded to a common value, ordinary value numbering then collapses the redundant
//! pure ops built on top of them. We stay intra-block, so the earlier (canonical) definition always
//! dominates the uses we redirect — no dominator analysis required. DCE deletes the dead remains.

use std::collections::{HashMap, HashSet};

use mercury_mir::{Function, Op, ValueId};

use crate::{map_op_uses, map_term_uses, Pass};

pub struct Cse;

impl Pass for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run_function(&self, f: &mut Function) -> bool {
        let mut rewrite: HashMap<u32, u32> = HashMap::new();

        for b in &f.blocks {
            // Value-number table for pure ops, current value per alloca slot, and the set of
            // value ids that name an alloca's base pointer.
            let mut vn: HashMap<String, u32> = HashMap::new();
            let mut slot_val: HashMap<u32, u32> = HashMap::new();
            let mut allocas: HashSet<u32> = HashSet::new();

            for inst in &b.insts {
                match &inst.op {
                    Op::Alloca(_) => {
                        if let Some(res) = inst.result {
                            allocas.insert(res.0);
                        }
                    }
                    Op::Load(ptr, _) => {
                        let p = resolve(&rewrite, ptr.0);
                        let Some(res) = inst.result else { continue };
                        if allocas.contains(&p) {
                            match slot_val.get(&p) {
                                Some(&v) => {
                                    rewrite.insert(res.0, v);
                                }
                                None => {
                                    slot_val.insert(p, res.0);
                                }
                            }
                        }
                    }
                    Op::Store { ptr, value } => {
                        let p = resolve(&rewrite, ptr.0);
                        let v = resolve(&rewrite, value.0);
                        if allocas.contains(&p) {
                            slot_val.insert(p, v);
                        } else {
                            // Unknown pointer: may alias any slot.
                            slot_val.clear();
                        }
                    }
                    Op::Call { .. } => {
                        // A call may store through pointers it was given.
                        slot_val.clear();
                    }
                    _ => {
                        let Some(res) = inst.result else { continue };
                        let Some(key) = pure_key(&inst.op, &rewrite) else { continue };
                        match vn.get(&key) {
                            Some(&canon) => {
                                rewrite.insert(res.0, canon);
                            }
                            None => {
                                vn.insert(key, res.0);
                            }
                        }
                    }
                }
            }
        }

        if rewrite.is_empty() {
            return false;
        }

        for b in &mut f.blocks {
            for inst in &mut b.insts {
                map_op_uses(&mut inst.op, |v| ValueId(resolve(&rewrite, v.0)));
            }
            map_term_uses(&mut b.term, |v| ValueId(resolve(&rewrite, v.0)));
        }
        true
    }
}

/// Follow rewrite chains (`a -> b -> c`) to the final target.
fn resolve(rewrite: &HashMap<u32, u32>, mut v: u32) -> u32 {
    let mut guard = 0;
    while let Some(&n) = rewrite.get(&v) {
        if n == v || guard > 10_000 {
            break;
        }
        v = n;
        guard += 1;
    }
    v
}

/// A canonical string key for a pure op, operands mapped through prior rewrites so equal
/// computations hash identically. Returns `None` for impure/uncacheable ops (handled separately).
fn pure_key(op: &Op, rewrite: &HashMap<u32, u32>) -> Option<String> {
    let m = |v: ValueId| -> u32 { resolve(rewrite, v.0) };
    Some(match op {
        Op::ConstInt(n, ty) => format!("ci:{n}:{ty:?}"),
        Op::ConstFloat(x, ty) => format!("cf:{}:{ty:?}", x.to_bits()),
        Op::Bin(o, a, b) => format!("bin:{o:?}:{}:{}", m(*a), m(*b)),
        Op::Cmp(o, a, b) => format!("cmp:{o:?}:{}:{}", m(*a), m(*b)),
        Op::Neg(a) => format!("neg:{}", m(*a)),
        Op::Not(a) => format!("not:{}", m(*a)),
        Op::Cast(k, a, ty) => format!("cast:{k:?}:{}:{ty:?}", m(*a)),
        Op::Select(c, a, b) => format!("sel:{}:{}:{}", m(*c), m(*a), m(*b)),
        Op::Gep { ptr, index, elem } => format!("gep:{}:{}:{elem:?}", m(*ptr), m(*index)),
        Op::Load(..) | Op::Store { .. } | Op::Call { .. } | Op::Alloca(..) => return None,
    })
}
