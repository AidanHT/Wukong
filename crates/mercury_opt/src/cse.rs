//! Common-subexpression elimination: dominator-tree value numbering with load forwarding.
//!
//! Pure operations are value-numbered over the **dominator tree**: a computation in a block is
//! reused by any block it dominates, since the dominating definition reaches all those uses. Walking
//! the tree with a scoped table (entries added on entry to a block, removed on exit) keeps reuse
//! legal without re-checking dominance per candidate. This catches redundancy across blocks, not
//! just within one.
//!
//! Loads are forwarded **within a block**: per `alloca` slot we track the value it currently holds —
//! a `store slot, v` makes `v` current, the first `load slot` becomes current and later loads reuse
//! it. A store through an unknown pointer or any call conservatively forgets all slots (they may
//! alias). Cross-block memory forwarding needs memory SSA and is left to DSE/the LLVM backend.
//!
//! Forwarding loads to a common value lets the pure value-numbering then collapse the expressions
//! built on top of them. DCE deletes the dead remains.

use std::collections::{HashMap, HashSet};

use mercury_mir::{Function, Op, ValueId};

use crate::{cfg, dom, map_op_uses, map_term_uses, Pass};

pub struct Cse;

impl Pass for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run_function(&self, f: &mut Function) -> bool {
        cfg::prune_unreachable(f); // dominance requires a clean CFG
        let idom = dom::idoms(f);
        let children = dom::dom_children(f, &idom);

        // Alloca base pointers are function-global value ids; collect them once.
        let mut allocas: HashSet<u32> = HashSet::new();
        for b in &f.blocks {
            for inst in &b.insts {
                if let (Some(r), Op::Alloca(_)) = (inst.result, &inst.op) {
                    allocas.insert(r.0);
                }
            }
        }

        let mut cx = Numbering {
            f,
            children: &children,
            allocas: &allocas,
            vn: HashMap::new(),
            rewrite: HashMap::new(),
        };
        cx.visit(f.entry.0);
        let rewrite = cx.rewrite;

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

struct Numbering<'a> {
    f: &'a Function,
    children: &'a [Vec<u32>],
    allocas: &'a HashSet<u32>,
    /// pure-op key -> canonical value id, scoped to the current dominator-tree path.
    vn: HashMap<String, u32>,
    /// value id -> the value it is replaced by (load forwards and CSE rewrites).
    rewrite: HashMap<u32, u32>,
}

impl Numbering<'_> {
    fn visit(&mut self, blk: u32) {
        // Keys this block introduced into `vn`, to remove when we leave its subtree.
        let mut added: Vec<String> = Vec::new();
        // Load forwarding is intra-block: the current value of each slot, reset per block.
        let mut slot_val: HashMap<u32, u32> = HashMap::new();

        for inst in &self.f.blocks[blk as usize].insts {
            match &inst.op {
                Op::Alloca(_) => {}
                Op::Load(ptr, _) => {
                    let p = resolve(&self.rewrite, ptr.0);
                    let Some(res) = inst.result else { continue };
                    if self.allocas.contains(&p) {
                        match slot_val.get(&p) {
                            Some(&v) => {
                                self.rewrite.insert(res.0, v);
                            }
                            None => {
                                slot_val.insert(p, res.0);
                            }
                        }
                    }
                }
                Op::Store { ptr, value } => {
                    let p = resolve(&self.rewrite, ptr.0);
                    let v = resolve(&self.rewrite, value.0);
                    if self.allocas.contains(&p) {
                        slot_val.insert(p, v);
                    } else {
                        slot_val.clear(); // unknown pointer may alias any slot
                    }
                }
                Op::Call { .. } => {
                    slot_val.clear(); // a call may store through pointers it was given
                }
                _ => {
                    let Some(res) = inst.result else { continue };
                    let Some(key) = pure_key(&inst.op, &self.rewrite) else {
                        continue;
                    };
                    match self.vn.get(&key) {
                        Some(&canon) => {
                            self.rewrite.insert(res.0, canon);
                        }
                        None => {
                            self.vn.insert(key.clone(), res.0);
                            added.push(key);
                        }
                    }
                }
            }
        }

        for i in 0..self.children[blk as usize].len() {
            let c = self.children[blk as usize][i];
            self.visit(c);
        }

        for k in added {
            self.vn.remove(&k);
        }
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
        Op::FuncAddr(s) => format!("faddr:{s:?}"),
        Op::Splat(a) => format!("splat:{}", m(*a)),
        Op::Load(..) | Op::Store { .. } | Op::Call { .. } | Op::Alloca(..) => return None,
    })
}
