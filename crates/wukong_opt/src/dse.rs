//! Dead-store elimination for overwritten stores and unread private stack slots.
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
//! A whole-function scan also removes in-bounds stores to local allocas whose addresses never
//! escape and which have no loads anywhere in the function, including through derived pointers.
//! This needs neither postdominance nor speculation: no execution can observe those bytes. Other
//! pending stores that survive to the end of the block are kept: a successor may read them.
//! Address observations count as escapes. Shared storage and functions with barriers are excluded
//! from lifetime elimination; MIR currently has no volatile load/store variant. DCE can subsequently
//! remove unused allocas and addresses, but this pass does not delete empty initialization loops.

use crate::alias::{type_bytes, AliasInfo, Prov};
use crate::fxhash::FxHashMap;

use wukong_mir::{AddrSpace, Function, MirType, Op, ValueId};

use crate::{CfgAnalyses, Pass};

pub struct Dse;

/// One tracked store destination: the address value plus the type stored there. Two stores share a
/// key only when they cover exactly the same bytes, which is what makes the earlier one dead.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StoreKey {
    addr: u32,
    ty: MirType,
}

/// Indexed by alloca result, not by individual address: even a read from a disjoint offset keeps
/// the entire slot live. Unknown pointers cannot reach a private slot; any publication of its
/// address (including pointer tables passed to VecKernelCall) already disqualifies it in AliasInfo.
fn unread_private_slots(f: &Function, alias: &AliasInfo) -> Vec<bool> {
    let mut unread = vec![false; f.value_types.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        match inst.op {
            Op::Alloca {
                space: AddrSpace::Local,
                ..
            } => {
                if let Some(v) = inst.result {
                    unread[v.0 as usize] = alias.is_private_stack(v);
                }
            }
            Op::Barrier => return vec![false; f.value_types.len()],
            _ => {}
        }
    }
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        if let Op::Load(ptr, _) = inst.op {
            if let Prov::Alloca(a) = alias.prov(ptr) {
                unread[a.0 as usize] = false;
            }
        }
    }
    unread
}

impl Pass for Dse {
    fn name(&self) -> &'static str {
        "dse"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let alias = cache.alias(f);
        let unread = unread_private_slots(f, &alias);

        let mut changed = false;
        // Reuse deletion storage between blocks. The bitmap is allocated only after
        // finding a redundant store; a retained instruction needs one bit lookup, not hashing.
        let mut dead: Vec<u64> = Vec::new();
        for b in &mut f.blocks {
            // location -> index of the last store to it that has not yet been read.
            let mut pending: FxHashMap<StoreKey, usize> = FxHashMap::default();
            dead.clear();

            for (i, inst) in b.insts.iter().enumerate() {
                match &inst.op {
                    Op::Store { ptr, value } => {
                        if matches!(alias.prov(*ptr), Prov::Alloca(a) if unread[a.0 as usize])
                            && type_bytes(&f.value_types[value.0 as usize])
                                .is_some_and(|bytes| alias.is_dereferenceable(*ptr, bytes))
                        {
                            // Keep potentially invalid or dynamically indexed stores: deletion
                            // must not hide a memory fault just because the slot has no reader.
                            if dead.is_empty() {
                                dead.resize(b.insts.len().div_ceil(64), 0);
                            }
                            dead[i / 64] |= 1u64 << (i % 64);
                            continue;
                        }
                        // A store never *reads*, so it cannot make another store observable; it can
                        // only fully overwrite one, which needs the identical byte range.
                        let key = StoreKey {
                            addr: ptr.0,
                            ty: f.value_types[value.0 as usize].clone(),
                        };
                        if let Some(prev) = pending.insert(key, i) {
                            if dead.is_empty() {
                                dead.resize(b.insts.len().div_ceil(64), 0);
                            }
                            dead[prev / 64] |= 1u64 << (prev % 64);
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
                    Op::Barrier => pending.clear(),
                    _ => {}
                }
            }

            if !dead.is_empty() {
                let mut i = 0;
                b.insts.retain(|_| {
                    let keep = dead[i / 64] & (1u64 << (i % 64)) == 0;
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
        // Both private stores now die: no instruction ever reads the slot. The parameter's
        // first store still survives the call, and its final store remains caller-observable.
        assert_eq!(
            count_stores(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    fn verify(f: &Function) {
        let errors = wukong_mir::verify::verify_function(f);
        assert!(errors.is_empty(), "{errors:?}");
    }

    fn run(f: &Function, it: &Interner) -> i64 {
        let mut p = wukong_mir::Program::new();
        p.funcs.push(f.clone());
        wukong_interp::run(&p, f.name, it).unwrap()
    }

    #[test]
    fn unread_private_array_stores_die_across_branches_but_read_slot_survives() {
        for take_left in [false, true] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("main"), MirType::I64);
            let dead = b.alloca(MirType::Array(Box::new(MirType::I64), 2));
            let live = b.alloca(MirType::I64);
            let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
            let tail = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: dead,
                    index: one,
                    elem: MirType::I64,
                },
            );
            let a = b.build(MirType::I64, Op::ConstInt(23, MirType::I64));
            let c = b.build(MirType::I64, Op::ConstInt(41, MirType::I64));
            let cond = b.build(
                MirType::I1,
                Op::ConstInt(i128::from(take_left), MirType::I1),
            );
            b.build_void(Op::Store {
                ptr: dead,
                value: one,
            });
            let left = b.new_block();
            let right = b.new_block();
            let exit = b.new_block();
            b.cond_br(cond, left, vec![], right, vec![]);
            b.switch_to(left);
            b.build_void(Op::Store {
                ptr: tail,
                value: a,
            });
            b.build_void(Op::Store {
                ptr: live,
                value: a,
            });
            b.br(exit, vec![]);
            b.switch_to(right);
            b.build_void(Op::Store {
                ptr: tail,
                value: c,
            });
            b.build_void(Op::Store {
                ptr: live,
                value: c,
            });
            b.br(exit, vec![]);
            b.switch_to(exit);
            let answer = b.build(MirType::I64, Op::Load(live, MirType::I64));
            b.ret(Some(answer));
            let mut f = b.finish();
            verify(&f);
            assert_eq!(run(&f, &it), if take_left { 23 } else { 41 });
            assert_eq!(count_stores(&f), 5);
            let mut cache = CfgAnalyses::default();
            assert!(Dse.run_function(&mut f, &mut cache));
            assert_eq!(count_stores(&f), 2);
            verify(&f);
            assert_eq!(run(&f, &it), if take_left { 23 } else { 41 });
            assert!(crate::dce::Dce.run_function(&mut f, &mut cache));
            assert!(!f
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.result == Some(dead)));
            verify(&f);
            assert_eq!(run(&f, &it), if take_left { 23 } else { 41 });
        }
    }

    #[test]
    fn successor_read_through_distinct_gep_keeps_the_store() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let slot = b.alloca(MirType::Array(Box::new(MirType::I64), 2));
        let index = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let p = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index,
                elem: MirType::I64,
            },
        );
        let value = b.build(MirType::I64, Op::ConstInt(73, MirType::I64));
        b.build_void(Op::Store { ptr: p, value });
        let next = b.new_block();
        b.br(next, vec![]);
        b.switch_to(next);
        let q = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index,
                elem: MirType::I64,
            },
        );
        let read = b.build(MirType::I64, Op::Load(q, MirType::I64));
        b.ret(Some(read));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 73);
        assert!(!Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 1);
        verify(&f);
        assert_eq!(run(&f, &it), 73);
    }

    #[test]
    fn published_or_observed_addresses_keep_unread_stores() {
        for observer in ["call", "stored", "ptrtoint", "return", "edge"] {
            let mut it = Interner::new();
            let ret = if observer == "return" {
                MirType::Ptr
            } else {
                MirType::Void
            };
            let mut b = Builder::new(it.intern("f"), ret);
            let out = b.add_param(MirType::Ptr);
            let slot = b.alloca(MirType::I64);
            let value = b.build(MirType::I64, Op::ConstInt(73, MirType::I64));
            b.build_void(Op::Store { ptr: slot, value });
            match observer {
                "call" => b.build_void(Op::Call {
                    func: it.intern("sink"),
                    args: vec![slot],
                }),
                "stored" => b.build_void(Op::Store {
                    ptr: out,
                    value: slot,
                }),
                "ptrtoint" => {
                    let addr = b.build(
                        MirType::I64,
                        Op::Cast(wukong_mir::CastKind::PtrToInt, slot, MirType::I64),
                    );
                    b.build_void(Op::Store {
                        ptr: out,
                        value: addr,
                    });
                }
                "edge" => {
                    let next = b.new_block();
                    let _ = b.block_param(next, MirType::Ptr);
                    b.br(next, vec![slot]);
                    b.switch_to(next);
                }
                "return" => {}
                _ => unreachable!(),
            }
            b.ret((observer == "return").then_some(slot));
            let mut f = b.finish();
            verify(&f);
            let before = count_stores(&f);
            assert!(
                !Dse.run_function(&mut f, &mut CfgAnalyses::default()),
                "{observer}"
            );
            assert_eq!(count_stores(&f), before, "{observer}");
            verify(&f);
        }
    }

    #[test]
    fn vector_kernel_pointer_table_and_scalar_arguments_keep_stores() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let data = b.alloca(MirType::Array(Box::new(MirType::F32), 8));
        let ptrs = b.alloca(MirType::Ptr);
        let scalars = b.alloca(MirType::F32);
        let value = b.build(MirType::F32, Op::ConstFloat(7.0, MirType::F32));
        // The kernel ABI processes complete eight-lane groups; initialize every lane.
        for lane in 0..8 {
            let index = b.build(MirType::I64, Op::ConstInt(lane, MirType::I64));
            let ptr = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: data,
                    index,
                    elem: MirType::F32,
                },
            );
            b.build_void(Op::Store { ptr, value });
        }
        b.build_void(Op::Store {
            ptr: ptrs,
            value: data,
        });
        b.build_void(Op::Store {
            ptr: scalars,
            value,
        });
        let n = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let kernel = b.add_vec_kernel(wukong_mir::VecKernel {
            name: it.intern("kernel"),
            streams: 1,
            scalars: 1,
            unroll: 1,
            ops: vec![
                wukong_mir::VecOp::Load { stream: 0 },
                wukong_mir::VecOp::Splat { scalar: 0 },
                wukong_mir::VecOp::Bin {
                    op: wukong_mir::VecBin::Add,
                    a: 0,
                    b: 1,
                },
            ],
            reduce: Some(wukong_mir::VecReduce {
                op: wukong_mir::VecRedOp::Add,
                value: 2,
                fma: None,
            }),
        });
        let result = b.build(
            MirType::F32,
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            },
        );
        let answer = b.build(
            MirType::I64,
            Op::Cast(wukong_mir::CastKind::FpToSi, result, MirType::I64),
        );
        b.ret(Some(answer));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 8 * (7 + 7));
        assert!(!Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 10);
        verify(&f);
        assert_eq!(run(&f, &it), 8 * (7 + 7));
    }

    #[test]
    fn barriers_keep_stores_and_block_local_overwrite_elimination() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::Void);
        let slot = b.alloca(MirType::I64);
        let value = b.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        b.build_void(Op::Store { ptr: slot, value });
        b.barrier();
        b.build_void(Op::Store { ptr: slot, value });
        b.ret(None);
        let mut f = b.finish();
        verify(&f);
        assert!(!Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 2);
        verify(&f);
    }

    #[test]
    fn unproven_store_extents_are_retained() {
        for offset in [None, Some(-1), Some(1)] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("f"), MirType::Void);
            let index = match offset {
                None => b.add_param(MirType::I64),
                Some(v) => b.build(MirType::I64, Op::ConstInt(v, MirType::I64)),
            };
            let slot = b.alloca(MirType::I64);
            let p = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: slot,
                    index,
                    elem: MirType::I64,
                },
            );
            let value = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
            b.build_void(Op::Store { ptr: p, value });
            b.ret(None);
            let mut f = b.finish();
            verify(&f);
            assert!(
                !Dse.run_function(&mut f, &mut CfgAnalyses::default()),
                "{offset:?}"
            );
            assert_eq!(count_stores(&f), 1);
            verify(&f);
        }
    }
}
