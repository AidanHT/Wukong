//! Common-subexpression elimination: dominator-tree value numbering with load forwarding.
//!
//! Pure operations are value-numbered over the **dominator tree**: a computation in a block is
//! reused by any block it dominates, since the dominating definition reaches all those uses. Walking
//! the tree with a scoped table (entries added on entry to a block, removed on exit) keeps reuse
//! legal without re-checking dominance per candidate. This catches redundancy across blocks, not
//! just within one.
//!
//! Loads are forwarded **within a block**, keyed by the SSA address value and the accessed type:
//! a `store p, v` makes `v` the current contents of `(p, typeof v)`, the first `load p` becomes the
//! current contents of `(p, ty)`, and a later load of the same address and type reuses it. Which
//! entries a store or call invalidates is decided by [`crate::alias`] — a store only forgets the
//! entries it `may_alias`, and a call forgets everything except addresses inside a stack slot whose
//! address never escaped this function (no callee can hold a pointer to one). Before that analysis
//! existed this pass tracked only pointers that were *literally* an `alloca` result and cleared the
//! whole table on any other store; the difference is visible on every `[]T` slice loop, whose
//! fat-pointer header loads sit behind a `gep`.
//!
//! Cross-block memory forwarding would need memory SSA, which this pass does not build, so it is
//! simply not attempted here — whatever survives is left to the native backend's own optimizer.
//!
//! Forwarding loads to a common value lets the pure value-numbering then collapse the expressions
//! built on top of them. DCE deletes the dead remains.

use crate::alias::{type_bytes, AliasInfo};
use crate::fxhash::FxHashMap;

use wukong_mir::{Function, MirType, Op, ValueId};

use crate::{cfg, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// Cap on the per-block memory table. A store invalidates by scanning the table, so an unbounded
/// table would make a block with many loads and many stores quadratic. Recognizer-generated blocks
/// can be large; past the cap the pass simply stops recording new locations (already-recorded ones
/// keep working), which only costs optimization, never correctness.
const MAX_TRACKED_LOCATIONS: usize = 256;

pub struct Cse;

impl Pass for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // Dominance requires a clean CFG. Skip the reachability DFS when nothing is unreachable
        // (the steady state) — `all_reachable` is free off the cached rpo `idoms` needs anyway.
        if !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate(); // pruning renumbered blocks
        }
        let children = cache.dom_children(f);
        let alias = AliasInfo::analyze(f);

        let mut cx = Numbering {
            f,
            children,
            alias: &alias,
            vn: FxHashMap::default(),
            rewrite: FxHashMap::default(),
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
    /// Provenance facts for `f`, computed once per pass run. Decides which memory-table entries a
    /// store or a call has to forget.
    alias: &'a AliasInfo,
    /// pure-op key -> canonical value id, scoped to the current dominator-tree path.
    vn: FxHashMap<Key, u32>,
    /// value id -> the value it is replaced by (load forwards and CSE rewrites).
    rewrite: FxHashMap<u32, u32>,
}

/// One tracked memory location: an SSA address value plus the type read/written there. The type is
/// part of the key because a `store f32` and a `load i32` at one address are not the same value —
/// keying on the address alone would forward the raw bits of one as the other.
#[derive(Clone, PartialEq, Eq, Hash)]
struct MemKey {
    addr: u32,
    ty: MirType,
}

/// May a load or store of `ty` participate in forwarding at all?
///
/// `bf16`/`f16` are excluded: their register form is `f32` and the narrowing happens on the way to
/// (2-byte) storage, so forwarding a stored `f32` register straight into a later load would skip the
/// rounding and change the value. That is the open `-O0` != `-O2` half-float defect (gap-hunt 15D);
/// this pass must not widen it.
fn forwardable(ty: &MirType) -> bool {
    !matches!(ty, MirType::F16 | MirType::BF16)
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
    ExtractLane(u32, u32),
    /// Keyed by the whole vector type: `iota <4 x i32>` and `iota <2 x i64>` are different
    /// constants, and only the type distinguishes them (the op has no operands).
    Iota(MirType),
    Fma(u32, u32, u32),
    Sqrt(u32),
    Round(u8, u32),
}

impl Numbering<'_> {
    fn visit(&mut self, blk: u32) {
        // Keys this block introduced into `vn`, to remove when we leave its subtree.
        let mut added: Vec<Key> = Vec::new();
        // Load forwarding is intra-block: what each tracked location currently holds, reset per
        // block.
        let mut mem: FxHashMap<MemKey, u32> = FxHashMap::default();

        for inst in &self.f.blocks[blk as usize].insts {
            match &inst.op {
                Op::Alloca(_) => {}
                Op::Load(ptr, ty) => {
                    let p = resolve(&self.rewrite, ptr.0);
                    let Some(res) = inst.result else { continue };
                    if !forwardable(ty) {
                        continue;
                    }
                    let key = MemKey {
                        addr: p,
                        ty: ty.clone(),
                    };
                    match mem.get(&key) {
                        Some(&v) => {
                            self.rewrite.insert(res.0, v);
                        }
                        None if mem.len() < MAX_TRACKED_LOCATIONS => {
                            mem.insert(key, res.0);
                        }
                        None => {}
                    }
                }
                Op::Store { ptr, value } => {
                    let p = resolve(&self.rewrite, ptr.0);
                    let v = resolve(&self.rewrite, value.0);
                    let sty = self.f.value_type(*value).clone();
                    let sbytes = AliasInfo::access_bytes(self.f, &inst.op).unwrap_or(u32::MAX);
                    // Forget exactly the locations this store may have written.
                    mem.retain(|k, _| {
                        !self.alias.may_alias_sized(
                            ValueId(k.addr),
                            type_bytes(&k.ty).unwrap_or(u32::MAX),
                            ValueId(p),
                            sbytes,
                        )
                    });
                    if forwardable(&sty) && mem.len() < MAX_TRACKED_LOCATIONS {
                        mem.insert(MemKey { addr: p, ty: sty }, v);
                    }
                }
                Op::Call { .. } | Op::VecKernelCall { .. } => {
                    // A call (or vector kernel) may store through any pointer it was given, or any
                    // pointer reachable from one — but never into a stack slot of *this* function
                    // whose address never escaped.
                    mem.retain(|k, _| self.alias.is_private_stack(ValueId(k.addr)));
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
fn resolve(rewrite: &FxHashMap<u32, u32>, mut v: u32) -> u32 {
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
fn pure_key(op: &Op, rewrite: &FxHashMap<u32, u32>) -> Option<Key> {
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
        Op::ExtractLane(a, k) => Key::ExtractLane(m(*a), *k),
        Op::Iota(ty) => Key::Iota(ty.clone()),
        Op::Fma(a, b, c) => Key::Fma(m(*a), m(*b), m(*c)),
        Op::Sqrt(a) => Key::Sqrt(m(*a)),
        Op::Round(mode, a) => Key::Round(*mode as u8, m(*a)),
        Op::Load(..)
        | Op::Store { .. }
        | Op::Call { .. }
        | Op::VecKernelCall { .. }
        | Op::Alloca(..) => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CfgAnalyses, Dce, Pass};
    use wukong_mir::Builder;
    use wukong_span::Interner;

    fn count_loads(f: &Function) -> usize {
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, Op::Load(..)))
            .count()
    }

    fn blob(bytes: u32) -> MirType {
        MirType::Array(Box::new(MirType::I8), bytes)
    }

    /// The shape every `[]T` loop lowers to: a slice fat-pointer header reached through a `gep`,
    /// its `data` field re-loaded per use, with an element store in between. The store writes into
    /// a *different* slot, but before the alias analysis existed this pass could not tell — the
    /// pointer was not literally an `alloca` result, so it cleared its whole table and forwarded
    /// nothing.
    #[test]
    fn header_loads_behind_a_gep_survive_a_store_to_a_disjoint_slot() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::Void);
        let hdr = b.alloca(blob(16));
        let out = b.alloca(blob(16));
        let z = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let data = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: hdr,
                index: z,
                elem: MirType::I8,
            },
        );
        let p1 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        let x = b.build(MirType::F32, Op::Load(p1, MirType::F32));
        b.build_void(Op::Store { ptr: out, value: x });
        let p2 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        let y = b.build(MirType::F32, Op::Load(p2, MirType::F32));
        b.build_void(Op::Store { ptr: out, value: y });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_loads(&f), 4);
        let mut cache = CfgAnalyses::default();
        assert!(Cse.run_function(&mut f, &mut cache));
        Dce.run_function(&mut f, &mut cache);
        // The second header load forwards to the first, and with it the element load behind it.
        assert_eq!(
            count_loads(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// The same shape, but the intervening store goes through a pointer of unknown provenance —
    /// which may well be the header itself. Nothing may be forwarded.
    #[test]
    fn a_store_through_an_unknown_pointer_still_kills_the_header() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("g"), MirType::Void);
        let hdr = b.add_param(MirType::Ptr);
        let z = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let data = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: hdr,
                index: z,
                elem: MirType::I8,
            },
        );
        let p1 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        b.build_void(Op::Store { ptr: p1, value: p1 });
        let p2 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        b.build_void(Op::Store { ptr: p2, value: p2 });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_loads(&f), 2);
        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        assert_eq!(
            count_loads(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// Two loads of one address at different widths are two different values: forwarding the i64
    /// one into the f32 one would hand over raw bits.
    #[test]
    fn a_load_is_keyed_by_its_type_not_only_its_address() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("h"), MirType::Void);
        let slot = b.alloca(blob(8));
        let a = b.build(MirType::I64, Op::Load(slot, MirType::I64));
        let c = b.build(MirType::F32, Op::Load(slot, MirType::F32));
        b.build_void(Op::Store {
            ptr: slot,
            value: a,
        });
        b.build_void(Op::Store {
            ptr: slot,
            value: c,
        });
        b.ret(None);
        let mut f = b.finish();

        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        assert_eq!(
            count_loads(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    use wukong_mir::{BasicBlock, BlockId, Inst, Terminator};

    /// Two loads of *different types* from one slot must not be forwarded to each other.
    ///
    /// The forwarding table used to be keyed by the slot alone, so the second load below was
    /// rewritten to the first one's value and an `f32` operand ended up holding a `<4 x f32>`.
    /// Nothing in the corpus reached it — the front end always addresses a slot through a
    /// `gep slot, 0 : T`, and the two geps are distinct *values*, so the table was never consulted
    /// with two types for one slot. Folding that zero-index gep to the slot itself (a
    /// canonicalization) makes both loads name the slot directly and reaches it immediately.
    #[test]
    fn loads_of_different_types_from_one_slot_are_not_forwarded() {
        let v4 = MirType::Vec(Box::new(MirType::F32), 4);
        let mut interner = Interner::new();
        let mut f = Function {
            name: interner.intern("t"),
            params: Vec::new(),
            ret: MirType::F32,
            blocks: vec![BasicBlock {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![
                    // v0 = alloca [4 x f32]
                    Inst {
                        result: Some(ValueId(0)),
                        op: Op::Alloca(MirType::Array(Box::new(MirType::F32), 4)),
                    },
                    // v1 = load <4 x f32> v0   (the vectorizer's lane load)
                    Inst {
                        result: Some(ValueId(1)),
                        op: Op::Load(ValueId(0), v4.clone()),
                    },
                    // v2 = load f32 v0         (the same address, read as one scalar)
                    Inst {
                        result: Some(ValueId(2)),
                        op: Op::Load(ValueId(0), MirType::F32),
                    },
                ],
                term: Terminator::Ret(Some(ValueId(2))),
            }],
            value_types: vec![MirType::Ptr, v4, MirType::F32],
            entry: BlockId(0),
            vec_kernels: Vec::new(),
        };
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "the input is well-formed"
        );
        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        let errs = wukong_mir::verify::verify_function(&f);
        assert!(errs.is_empty(), "cse produced invalid MIR: {errs:?}");
        assert!(
            matches!(f.blocks[0].term, Terminator::Ret(Some(v)) if v == ValueId(2)),
            "the f32 load must survive as its own value"
        );
    }

    /// A store invalidates *every* view of the slot, not just the one at the stored type: writing
    /// one `f32` lane leaves a previously loaded `<4 x f32>` stale in three lanes.
    #[test]
    fn a_scalar_store_invalidates_a_wider_cached_load() {
        let v4 = MirType::Vec(Box::new(MirType::F32), 4);
        let mut interner = Interner::new();
        let mut f = Function {
            name: interner.intern("t"),
            params: Vec::new(),
            ret: MirType::Void,
            blocks: vec![BasicBlock {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![
                    Inst {
                        result: Some(ValueId(0)),
                        op: Op::Alloca(MirType::Array(Box::new(MirType::F32), 4)),
                    },
                    // v1 = load <4 x f32> v0
                    Inst {
                        result: Some(ValueId(1)),
                        op: Op::Load(ValueId(0), v4.clone()),
                    },
                    // v2 = 1.0f32 ; store v2 -> v0
                    Inst {
                        result: Some(ValueId(2)),
                        op: Op::ConstFloat(1.0, MirType::F32),
                    },
                    Inst {
                        result: None,
                        op: Op::Store {
                            ptr: ValueId(0),
                            value: ValueId(2),
                        },
                    },
                    // v3 = load <4 x f32> v0  -- must NOT be forwarded to v1
                    Inst {
                        result: Some(ValueId(3)),
                        op: Op::Load(ValueId(0), v4.clone()),
                    },
                    Inst {
                        result: None,
                        op: Op::Store {
                            ptr: ValueId(0),
                            value: ValueId(3),
                        },
                    },
                ],
                term: Terminator::Ret(None),
            }],
            value_types: vec![MirType::Ptr, v4.clone(), MirType::F32, v4],
            entry: BlockId(0),
            vec_kernels: Vec::new(),
        };
        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        let reload_survives = f.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(&i.op, Op::Store { value, .. } if *value == ValueId(3)));
        assert!(
            reload_survives,
            "the reload after the scalar store must not be forwarded to the pre-store value"
        );
    }
}
