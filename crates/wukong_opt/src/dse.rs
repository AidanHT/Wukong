//! Dead-store elimination (intra-block).
//!
//! The dual of CSE's load forwarding: a `store p, v1` is dead if a later `store p, v2` in the same
//! block overwrites it with no intervening read that could observe it. We track, per location, the
//! index of a pending (not-yet-read) store; a second store to the *same* location — same address
//! value and same stored type, so the byte range is identical — marks the first dead.
//!
//! Which reads kill which pending stores is decided by [`crate::alias`]: a `load` clears only the
//! pending stores it `may_alias`, and a `call` (or vector-kernel call, which reads through the
//! pointers it was given) clears everything except stores into a stack slot of this function whose
//! address never escaped — no callee can hold a pointer to one. Before that analysis existed this
//! pass tracked only pointers that were *literally* an `alloca` result and cleared the whole table
//! on any other load or store, so a `[]T` element write, which goes through a pointer loaded out of
//! the slice header, was never a candidate.
//!
//! Pending stores that survive to the end of the block are kept: a successor block may read them.

use crate::alias::{type_bytes, AliasInfo};
use crate::fxhash::{FxHashMap, FxHashSet};

use wukong_mir::{Function, MirType, Op, ValueId};

use crate::{CfgAnalyses, Pass};

pub struct Dse;

/// One tracked store destination: the address value plus the type stored there. Two stores share a
/// key only when they cover exactly the same bytes, which is what makes the earlier one dead.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StoreKey {
    addr: u32,
    ty: MirType,
}

impl Pass for Dse {
    fn name(&self) -> &'static str {
        "dse"
    }

    fn run_function(&self, f: &mut Function, _cache: &mut CfgAnalyses) -> bool {
        let alias = AliasInfo::analyze(f);

        let mut changed = false;
        for b in &mut f.blocks {
            // location -> index of the last store to it that has not yet been read.
            let mut pending: FxHashMap<StoreKey, usize> = FxHashMap::default();
            let mut dead: Vec<usize> = Vec::new();

            for (i, inst) in b.insts.iter().enumerate() {
                match &inst.op {
                    Op::Store { ptr, value } => {
                        // A store never *reads*, so it cannot make another store observable; it can
                        // only fully overwrite one, which needs the identical byte range.
                        let key = StoreKey {
                            addr: ptr.0,
                            ty: f.value_types[value.0 as usize].clone(),
                        };
                        if let Some(prev) = pending.insert(key, i) {
                            dead.push(prev);
                        }
                    }
                    Op::Load(ptr, ty) => {
                        let bytes = type_bytes(ty).unwrap_or(u32::MAX);
                        pending.retain(|k, _| {
                            !alias.may_alias_sized(
                                ValueId(k.addr),
                                type_bytes(&k.ty).unwrap_or(u32::MAX),
                                *ptr,
                                bytes,
                            )
                        });
                    }
                    Op::Call { .. } | Op::VecKernelCall { .. } => {
                        // A call may read through any pointer it was given, or any pointer reachable
                        // from one — but never a stack slot of this function that never escaped.
                        pending.retain(|k, _| alias.is_private_stack(ValueId(k.addr)));
                    }
                    _ => {}
                }
            }

            if !dead.is_empty() {
                let dead: FxHashSet<usize> = dead.into_iter().collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Builder;
    use wukong_span::Interner;

    fn count_stores(f: &Function) -> usize {
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, Op::Store { .. }))
            .count()
    }

    /// Two writes to one `[]T` element, with a read of a *disjoint* slot in between. The address is
    /// a pointer loaded out of the slice header, so the old alloca-only rule tracked nothing.
    #[test]
    fn a_store_through_a_loaded_pointer_is_killed_by_its_overwrite() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::Void);
        let hdr = b.add_param(MirType::Ptr);
        let other = b.alloca(MirType::Array(Box::new(MirType::I8), 8));
        let data = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
        let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let two = b.build(MirType::F32, Op::ConstFloat(2.0, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: one,
        });
        // Reading a private stack slot cannot observe the store above.
        let _ = b.build(MirType::F32, Op::Load(other, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: two,
        });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_stores(&f), 2);
        let mut cache = CfgAnalyses::default();
        assert!(Dse.run_function(&mut f, &mut cache));
        assert_eq!(
            count_stores(&f),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// The same pair, but the intervening read goes through a pointer that may well be the same
    /// element. The first store is observable and must stay.
    #[test]
    fn an_aliasing_read_keeps_the_first_store() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("g"), MirType::Void);
        let hdr = b.add_param(MirType::Ptr);
        let other = b.add_param(MirType::Ptr);
        let data = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
        let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let two = b.build(MirType::F32, Op::ConstFloat(2.0, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: one,
        });
        let _ = b.build(MirType::F32, Op::Load(other, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: two,
        });
        b.ret(None);
        let mut f = b.finish();

        let mut cache = CfgAnalyses::default();
        Dse.run_function(&mut f, &mut cache);
        assert_eq!(
            count_stores(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// A call may read anything it was handed, so it keeps an escaped store alive — but a slot whose
    /// address never left this function is out of its reach.
    #[test]
    fn a_call_keeps_escaped_stores_but_not_private_ones() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("h"), MirType::Void);
        let out = b.add_param(MirType::Ptr);
        let priv_slot = b.alloca(MirType::Array(Box::new(MirType::I8), 8));
        let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let two = b.build(MirType::F32, Op::ConstFloat(2.0, MirType::F32));
        b.build_void(Op::Store {
            ptr: out,
            value: one,
        });
        b.build_void(Op::Store {
            ptr: priv_slot,
            value: one,
        });
        b.build_void(Op::Call {
            func: it.intern("sink"),
            args: vec![out],
        });
        b.build_void(Op::Store {
            ptr: out,
            value: two,
        });
        b.build_void(Op::Store {
            ptr: priv_slot,
            value: two,
        });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_stores(&f), 4);
        let mut cache = CfgAnalyses::default();
        assert!(Dse.run_function(&mut f, &mut cache));
        // Only the private slot's first store dies; the parameter's survives the call.
        assert_eq!(
            count_stores(&f),
            3,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }
}
