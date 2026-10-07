//! Common-subexpression elimination: dominator-tree value numbering with load forwarding.
//!
//! Pure operations are value-numbered over the **dominator tree**: a computation in a block is
//! reused by any block it dominates, since the dominating definition reaches all those uses. Walking
//! the tree with a scoped table (entries added on entry to a block, removed on exit) keeps reuse
//! legal without re-checking dominance per candidate. This catches redundancy across blocks, not
//! just within one.
//!
//! Loads are forwarded within a block and along **single-predecessor edges**, keyed by the SSA
//! address value and the accessed type:
//! a `store p, v` makes `v` the current contents of `(p, typeof v)`, the first `load p` becomes the
//! current contents of `(p, ty)`, and a later load of the same address and type reuses it. Which
//! entries a store or call invalidates is decided by [`crate::alias`] — a store only forgets the
//! entries it `may_alias`, and a call forgets everything except addresses inside a stack slot whose
//! address never escaped this function (no callee can hold a pointer to one). Before that analysis
//! existed this pass tracked only pointers that were *literally* an `alloca` result and cleared the
//! whole table on any other store; the difference is visible on every `[]T` slice loop, whose
//! fat-pointer header loads sit behind a `gep`.
//!
//! A dominator child inherits its parent's final memory table only when that parent is its only
//! predecessor. Every execution of the child then sees exactly the state the table describes:
//! there is no intervening block or alternate incoming edge. Joins and loop headers start with an
//! empty table, so back-edge writes cannot be missed. This limited cross-block forwarding needs
//! neither memory SSA nor a speculative load; it removes a load only after the same address was
//! already read or written on the path actually taken.
//!
//! Forwarding loads to a common value lets the pure value-numbering then collapse the expressions
//! built on top of them. DCE deletes the dead remains.

use crate::alias::{type_bytes, AliasInfo};
use crate::fxhash::FxHashMap;

use wukong_mir::{BinOp, CmpOp, Function, MirType, Op, ValueId};

use crate::{cfg, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// Cap on the per-block memory table. A store invalidates by scanning the table, so an unbounded
/// table would make a block with many loads and many stores quadratic. Recognizer-generated blocks
/// can be large; past the cap the pass simply stops recording new locations (already-recorded ones
/// keep working), which only costs optimization, never correctness.
const MAX_TRACKED_LOCATIONS: usize = 256;

/// "No rewrite" in the dense rewrite table.
const NO_REWRITE: u32 = u32::MAX;

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
        // Taken before the dominator borrows below: an `Rc`, so it borrows nothing.
        let alias = cache.alias(f);
        let (children, preds) = cache.children_and_preds(f);

        let mut cx = Numbering {
            f,
            children,
            preds,
            alias: &alias,
            vn: FxHashMap::default(),
            rewrite: vec![NO_REWRITE; f.value_types.len()],
            rewrites: 0,
            types: TyIds::default(),
        };
        cx.visit(f.entry.0, FxHashMap::default());
        let (rewrite, rewrites) = (cx.rewrite, cx.rewrites);

        if rewrites == 0 {
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
    preds: &'a [Vec<u32>],
    /// Provenance facts for `f`, computed once per pass run. Decides which memory-table entries a
    /// store or a call has to forget.
    alias: &'a AliasInfo,
    /// pure-op key -> canonical value id, scoped to the current dominator-tree path.
    vn: FxHashMap<Key, u32>,
    /// value id -> the value it is replaced by (load forwards and CSE rewrites), dense over the
    /// value arena with [`NO_REWRITE`] for "kept". A target is always a value that was already
    /// visited and kept, so it is final when recorded: every chain is one step long.
    rewrite: Vec<u32>,
    rewrites: usize,
    /// Interned key types (see [`TyId`]).
    types: TyIds,
}

/// State suspended while a dominator child's subtree is numbered. Keep pending siblings in
/// their parent frame instead of eagerly copying a memory table for every child.
struct Scope {
    block: u32,
    next_child: usize,
    last_inheriting: Option<usize>,
    mem: FxHashMap<MemKey, u32>,
    added_start: usize,
}

/// A [`MirType`] folded into one word, so a key is `Copy` and building or probing one never clones a
/// boxed element type (`Vec`/`Array` types carry a `Box`). Scalars, and vectors or arrays of
/// scalars — every type value numbering meets in practice — encode directly; anything nested
/// deeper is interned per pass run. Equal types always map to equal ids and distinct types to
/// distinct ids, so keys compare exactly as the `MirType`s they replace.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TyId(u64);

#[derive(Default)]
struct TyIds {
    nested: FxHashMap<MirType, u64>,
}

/// A scalar type's code; `None` for `Vec`/`Array`.
fn scalar_code(ty: &MirType) -> Option<u64> {
    Some(match ty {
        MirType::I1 => 1,
        MirType::I8 => 2,
        MirType::I16 => 3,
        MirType::I32 => 4,
        MirType::I64 => 5,
        MirType::F16 => 6,
        MirType::BF16 => 7,
        MirType::F32 => 8,
        MirType::F64 => 9,
        MirType::Ptr => 10,
        MirType::Void => 11,
        MirType::Vec(..) | MirType::Array(..) => return None,
    })
}

impl TyIds {
    /// Bits 62-63 tag the form (scalar, vector-of-scalar, array-of-scalar, interned); a direct
    /// encoding keeps the element code in bits 32-35 and the length in bits 0-31.
    fn id(&mut self, ty: &MirType) -> TyId {
        if let Some(code) = scalar_code(ty) {
            return TyId(code);
        }
        match ty {
            MirType::Vec(elem, n) => {
                if let Some(code) = scalar_code(elem) {
                    return TyId((1 << 62) | (code << 32) | u64::from(*n));
                }
            }
            MirType::Array(elem, n) => {
                if let Some(code) = scalar_code(elem) {
                    return TyId((2 << 62) | (code << 32) | u64::from(*n));
                }
            }
            _ => {}
        }
        if let Some(&id) = self.nested.get(ty) {
            return TyId((3 << 62) | id);
        }
        let id = self.nested.len() as u64;
        self.nested.insert(ty.clone(), id);
        TyId((3 << 62) | id)
    }
}

/// One tracked memory location: an SSA address value plus the type read/written there. The type is
/// part of the key because a `store f32` and a `load i32` at one address are not the same value —
/// keying on the address alone would forward the raw bits of one as the other. `bytes` is the
/// access width that type implies (a function of `ty`, so it adds no distinction), kept here so an
/// invalidating store can ask the alias analysis without recovering the `MirType`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MemKey {
    addr: u32,
    ty: TyId,
    bytes: u32,
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

/// A typed value-numbering key for a pure op. One variant per cacheable `Op`, carrying its
/// discriminant, operands mapped through prior rewrites, immediate values, and every type that
/// cannot be recovered from the operands. In particular a splat needs its result vector type.
/// `Eq`/`Hash` are derived and no textual key is built; types are [`TyId`]s, so a key is `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    ConstInt(i128, TyId),
    /// Float immediate keyed by its raw bits (matches the old `x.to_bits()`), so `-0.0`/`NaN`
    /// payloads stay distinct exactly as before.
    ConstFloat(u64, TyId),
    Bin(u8, u32, u32),
    /// Vector comparisons may have different mask element types for the same operands and lane
    /// count. Keep the result type so a comparison never substitutes a differently typed mask.
    Cmp(u8, u32, u32, TyId),
    Neg(u32),
    Not(u32),
    Cast(u8, u32, TyId),
    Select(u32, u32, u32),
    Gep(u32, u32, TyId),
    FuncAddr(u32),
    GlobalAddr(u32),
    /// The scalar operand does not encode the destination lane count. Two splats of one value
    /// into different vector widths are distinct computations even in the same dominator scope.
    Splat(u32, TyId),
    ExtractLane(u32, u32),
    /// Keyed by the whole vector type: `iota <4 x i32>` and `iota <2 x i64>` are different
    /// constants, and only the type distinguishes them (the op has no operands).
    Iota(TyId),
    Fma(u32, u32, u32),
    Sqrt(u32),
    Round(u8, u32),
}

impl Numbering<'_> {
    fn visit(&mut self, entry: u32, mem: FxHashMap<MemKey, u32>) {
        // One undo log serves the whole DFS. Each scope removes its suffix on exit, so sibling
        // computations never become candidates for one another. Reusing the log also avoids a
        // separate allocation for every block that contributes a pure expression.
        let mut added: Vec<Key> = Vec::new();
        let mut stack = vec![self.enter(entry, mem, &mut added)];
        while let Some(scope) = stack.last_mut() {
            let children = &self.children[scope.block as usize];
            if let Some(&child) = children.get(scope.next_child) {
                let i = scope.next_child;
                scope.next_child += 1;
                // Preserve the recursive walk's child order and move the table into its last
                // inheriting child. Straight-line regions still incur no table copy.
                let inherited = if Some(i) == scope.last_inheriting {
                    std::mem::take(&mut scope.mem)
                } else if self.preds[child as usize].as_slice() == [scope.block] {
                    scope.mem.clone()
                } else {
                    FxHashMap::default()
                };
                stack.push(self.enter(child, inherited, &mut added));
            } else {
                for key in added.drain(scope.added_start..) {
                    self.vn.remove(&key);
                }
                stack.pop();
            }
        }
    }

    fn resolve(&self, v: ValueId) -> u32 {
        resolve(&self.rewrite, v.0)
    }

    fn set_rewrite(&mut self, v: ValueId, to: u32) {
        if let Some(slot) = self.rewrite.get_mut(v.0 as usize) {
            *slot = to;
            self.rewrites += 1;
        }
    }

    fn mem_key(&mut self, addr: u32, ty: &MirType) -> MemKey {
        MemKey {
            addr,
            ty: self.types.id(ty),
            bytes: type_bytes(ty).unwrap_or(u32::MAX),
        }
    }

    fn enter(&mut self, blk: u32, mut mem: FxHashMap<MemKey, u32>, added: &mut Vec<Key>) -> Scope {
        let added_start = added.len();
        let f = self.f;
        for inst in &f.blocks[blk as usize].insts {
            match &inst.op {
                Op::Alloca {
                    ty: _,
                    space: wukong_mir::AddrSpace::Local,
                } => {}
                Op::Load(ptr, ty) => {
                    let p = self.resolve(*ptr);
                    let Some(res) = inst.result else { continue };
                    if !forwardable(ty) {
                        continue;
                    }
                    let key = self.mem_key(p, ty);
                    match mem.get(&key) {
                        Some(&v) => self.set_rewrite(res, v),
                        None if mem.len() < MAX_TRACKED_LOCATIONS => {
                            mem.insert(key, res.0);
                        }
                        None => {}
                    }
                }
                Op::Store { ptr, value } => {
                    let p = self.resolve(*ptr);
                    let v = self.resolve(*value);
                    let sty = f.value_type(*value);
                    let sbytes = AliasInfo::access_bytes(f, &inst.op).unwrap_or(u32::MAX);
                    // Forget exactly the locations this store may have written.
                    let alias = self.alias;
                    mem.retain(|k, _| {
                        !alias.may_alias_sized(ValueId(k.addr), k.bytes, ValueId(p), sbytes)
                    });
                    if forwardable(sty) && mem.len() < MAX_TRACKED_LOCATIONS {
                        let key = self.mem_key(p, sty);
                        mem.insert(key, v);
                    }
                }
                Op::Call { .. } | Op::VecKernelCall { .. } => {
                    // A call (or vector kernel) may store through any pointer it was given, or any
                    // pointer reachable from one — but never into a stack slot of *this* function
                    // whose address never escaped.
                    let alias = self.alias;
                    mem.retain(|k, _| alias.is_private_stack(ValueId(k.addr)));
                }
                _ => {
                    let Some(res) = inst.result else { continue };
                    let Some(key) = self.pure_key(&inst.op, f.value_type(res)) else {
                        continue;
                    };
                    match self.vn.get(&key) {
                        Some(&canon) => self.set_rewrite(res, canon),
                        None => {
                            self.vn.insert(key, res.0);
                            added.push(key);
                        }
                    }
                }
            }
        }

        // At most the actual CFG successors can inherit this table. Clone only when more than
        // one child needs it, then move it into the last such child. Straight-line regions incur
        // no table copy, and the existing location cap bounds the cost at conditional branches.
        let last_inheriting = self.children[blk as usize]
            .iter()
            .rposition(|&c| self.preds[c as usize].as_slice() == [blk]);
        Scope {
            block: blk,
            next_child: 0,
            last_inheriting,
            mem,
            added_start,
        }
    }

    /// A canonical [`Key`] for a pure op, operands mapped through prior rewrites so equal
    /// computations hash identically. Returns `None` for impure/uncacheable ops (handled
    /// separately). The op sub-kinds (`BinOp`/`CmpOp`/`CastKind`/`RoundMode`) are fieldless C-like
    /// enums; `as u8` is their stable discriminant, which distinguishes the same variants the old
    /// `{:?}` did.
    fn pure_key(&mut self, op: &Op, result_ty: &MirType) -> Option<Key> {
        let rw = &self.rewrite;
        let m = |v: ValueId| -> u32 { resolve(rw, v.0) };
        Some(match op {
            Op::ConstInt(n, ty) => Key::ConstInt(*n, self.types.id(ty)),
            Op::ConstFloat(x, ty) => Key::ConstFloat(x.to_bits(), self.types.id(ty)),
            Op::Bin(o, a, b) => {
                let (mut a, mut b) = (m(*a), m(*b));
                // Wrapping integer addition/multiplication and bitwise operations commute at every
                // scalar and vector lane width. Canonicalize only their lookup key, retaining the
                // first dominating computation's operand order in the emitted MIR. Floating-point
                // arithmetic is deliberately excluded: swapping NaN operands may change payloads.
                if matches!(
                    o,
                    BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor
                ) && a > b
                {
                    std::mem::swap(&mut a, &mut b);
                }
                Key::Bin(*o as u8, a, b)
            }
            Op::Cmp(o, a, b) => {
                let (mut a, mut b) = (m(*a), m(*b));
                if matches!(o, CmpOp::Eq | CmpOp::Ne) && a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                Key::Cmp(*o as u8, a, b, self.types.id(result_ty))
            }
            Op::Neg(a) => Key::Neg(m(*a)),
            Op::Not(a) => Key::Not(m(*a)),
            Op::Cast(k, a, ty) => {
                let a = m(*a);
                Key::Cast(*k as u8, a, self.types.id(ty))
            }
            Op::Select(c, a, b) => Key::Select(m(*c), m(*a), m(*b)),
            Op::Gep { ptr, index, elem } => {
                let (ptr, index) = (m(*ptr), m(*index));
                Key::Gep(ptr, index, self.types.id(elem))
            }
            Op::FuncAddr(s) => Key::FuncAddr(s.0),
            Op::GlobalAddr(s) => Key::GlobalAddr(s.0),
            Op::Splat(a) => {
                let a = m(*a);
                Key::Splat(a, self.types.id(result_ty))
            }
            Op::ExtractLane(a, k) => Key::ExtractLane(m(*a), *k),
            Op::Iota(ty) => Key::Iota(self.types.id(ty)),
            Op::Fma(a, b, c) => Key::Fma(m(*a), m(*b), m(*c)),
            Op::Sqrt(a) => Key::Sqrt(m(*a)),
            Op::Round(mode, a) => Key::Round(*mode as u8, m(*a)),
            Op::Load(..)
            | Op::Store { .. }
            | Op::Call { .. }
            | Op::VecKernelCall { .. }
            | Op::Alloca { .. }
            | Op::Barrier => return None,
        })
    }
}

/// Follow the rewrite table to the final target. Targets are final when recorded (see
/// [`Numbering::rewrite`]), so this is one lookup; the bound only guards a malformed chain.
fn resolve(rewrite: &[u32], mut v: u32) -> u32 {
    for _ in 0..rewrite.len() {
        match rewrite.get(v as usize) {
            Some(&n) if n != NO_REWRITE && n != v => v = n,
            _ => break,
        }
    }
    v
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
                        op: Op::Alloca {
                            ty: MirType::Array(Box::new(MirType::F32), 4),
                            space: wukong_mir::AddrSpace::Local,
                        },
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
                        op: Op::Alloca {
                            ty: MirType::Array(Box::new(MirType::F32), 4),
                            space: wukong_mir::AddrSpace::Local,
                        },
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

    /// Keys carry folded type ids instead of `MirType`s. Equal types must still collide and
    /// distinct ones must not — including a vector width the operand cannot encode and an element
    /// type nested too deeply for the direct encoding.
    #[test]
    fn type_keyed_values_merge_only_at_identical_types() {
        let mut it = Interner::new();
        let v4 = MirType::Vec(Box::new(MirType::F32), 4);
        let v8 = MirType::Vec(Box::new(MirType::F32), 8);
        let nested = |n| MirType::Array(Box::new(MirType::Array(Box::new(MirType::I8), 4)), n);
        let mut b = Builder::new(it.intern("t"), MirType::Void);
        let x = b.add_param(MirType::F32);
        let base = b.add_param(MirType::Ptr);
        let i = b.add_param(MirType::I64);
        let s4a = b.build(v4.clone(), Op::Splat(x));
        let s4b = b.build(v4.clone(), Op::Splat(x));
        let s8 = b.build(v8.clone(), Op::Splat(x));
        let gep = |b: &mut Builder, elem: MirType| {
            b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base,
                    index: i,
                    elem,
                },
            )
        };
        let g2a = gep(&mut b, nested(2));
        let g2b = gep(&mut b, nested(2));
        let g3 = gep(&mut b, nested(3));
        let sink = it.intern("sink");
        b.build_void(Op::Call {
            func: sink,
            args: vec![s4a, s4b, s8, g2a, g2b, g3],
        });
        b.ret(None);
        let mut f = b.finish();
        assert!(Cse.run_function(&mut f, &mut CfgAnalyses::default()));
        let args = f.blocks[0]
            .insts
            .iter()
            .find_map(|inst| match &inst.op {
                Op::Call { args, .. } => Some(args.clone()),
                _ => None,
            })
            .expect("the call survives");
        assert_eq!(args, vec![s4a, s4a, s8, g2a, g2a, g3]);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
    }
}
