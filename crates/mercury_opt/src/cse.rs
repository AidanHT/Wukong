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

use mercury_mir::{Function, MirType, Op, ValueId};

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
    vn: HashMap<Key, u32>,
    /// value id -> the value it is replaced by (load forwards and CSE rewrites).
    rewrite: HashMap<u32, u32>,
}

/// A canonical, allocation-free value-numbering key for a pure op. One variant per cacheable
/// `Op`, carrying exactly the fields the previous `format!`-string key encoded: the op
/// discriminant (the enum variant itself), every operand `ValueId` (as its inner `u32`, mapped
/// through prior rewrites), the result/operand `MirType` where the op carried one, immediates,
/// and op sub-kinds (`BinOp`/`CmpOp`/`CastKind`/`RoundMode`) as their stable `as u8` discriminant.
/// `Eq`/`Hash` are derived, so two ops compare equal here iff their old strings were equal —
/// no more, no less. `MirType` is `Hash + Eq` and a bitwise copy for scalars (only `Vec`/`Array`
/// operand types touch the heap), so this avoids the per-instruction `String` the old key built.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    ConstInt(i128, MirType),
    /// Float immediate keyed by its raw bits (matches the old `x.to_bits()`), so `-0.0`/`NaN`
    /// payloads stay distinct exactly as before.
    ConstFloat(u64, MirType),
    Bin(u8, u32, u32),
    Cmp(u8, u32, u32),
    Neg(u32),
    Not(u32),
    Cast(u8, u32, MirType),
    Select(u32, u32, u32),
    Gep(u32, u32, MirType),
    FuncAddr(u32),
    GlobalAddr(u32),
    Splat(u32),
    Fma(u32, u32, u32),
    Sqrt(u32),
    Round(u8, u32),
}

impl Numbering<'_> {
    fn visit(&mut self, blk: u32) {
        // Keys this block introduced into `vn`, to remove when we leave its subtree.
        let mut added: Vec<Key> = Vec::new();
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

/// A canonical [`Key`] for a pure op, operands mapped through prior rewrites so equal
/// computations hash identically. Returns `None` for impure/uncacheable ops (handled separately).
/// The op sub-kinds (`BinOp`/`CmpOp`/`CastKind`/`RoundMode`) are fieldless C-like enums; `as u8`
/// is their stable discriminant, which distinguishes the same variants the old `{:?}` did.
fn pure_key(op: &Op, rewrite: &HashMap<u32, u32>) -> Option<Key> {
    let m = |v: ValueId| -> u32 { resolve(rewrite, v.0) };
    Some(match op {
        Op::ConstInt(n, ty) => Key::ConstInt(*n, ty.clone()),
        Op::ConstFloat(x, ty) => Key::ConstFloat(x.to_bits(), ty.clone()),
        Op::Bin(o, a, b) => Key::Bin(*o as u8, m(*a), m(*b)),
        Op::Cmp(o, a, b) => Key::Cmp(*o as u8, m(*a), m(*b)),
        Op::Neg(a) => Key::Neg(m(*a)),
        Op::Not(a) => Key::Not(m(*a)),
        Op::Cast(k, a, ty) => Key::Cast(*k as u8, m(*a), ty.clone()),
        Op::Select(c, a, b) => Key::Select(m(*c), m(*a), m(*b)),
        Op::Gep { ptr, index, elem } => Key::Gep(m(*ptr), m(*index), elem.clone()),
        Op::FuncAddr(s) => Key::FuncAddr(s.0),
        Op::GlobalAddr(s) => Key::GlobalAddr(s.0),
        Op::Splat(a) => Key::Splat(m(*a)),
        Op::Fma(a, b, c) => Key::Fma(m(*a), m(*b), m(*c)),
        Op::Sqrt(a) => Key::Sqrt(m(*a)),
        Op::Round(mode, a) => Key::Round(*mode as u8, m(*a)),
        Op::Load(..) | Op::Store { .. } | Op::Call { .. } | Op::Alloca(..) => return None,
    })
}
