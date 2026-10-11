//! Straight-line SLP vectorization: pack isomorphic scalar stores into one 128-bit vector store.
//!
//! The loop vectorizer ([`crate::vectorize`]) widens *iterations*. This pass widens *statements*:
//! `W` stores in one block to consecutive elements of one object, whose values are computed by the
//! same operations, become one `Vec(T, W)` store of one vector computation (Larsen and
//! Amarasinghe's superword-level parallelism, seeded from stores as LLVM's SLP vectorizer is).
//! The shapes it exists for are the ones loops leave behind or never had: the register tile a
//! complete unroll leaves (`for k in 0..8 step 2`, which the loop vectorizer declines for its
//! step), hand-written `vec4`/quaternion/complex-pair arithmetic, and small fixed-array kernels.
//!
//! # Placement
//!
//! A post-pipeline stage at `-O2` and above, after [`crate::unroll_program`] and its cleanup — so
//! a complete unroll's copies are visible with their induction variables folded to constants — and
//! before switch formation. The functions it changes get the same reduced cleanup sweep unrolling
//! does ([`crate::PassManager::unroll_cleanup`]), whose `dce` removes the scalar computations the
//! packed tree replaced. `-O0` and `-O1` never run it, and `-O3` is `-O2`.
//!
//! # The algorithm
//!
//! Deterministic throughout: blocks in index order, and within a block the seeds sorted by their
//! first store's index. After every pack the function is rescanned from that block with fresh
//! facts (definitions, use counts and a fresh [`AliasInfo`]), at most [`MAX_PACKS_PER_FN`] times,
//! and a function above [`MAX_FN_INSTS`] instructions is skipped.
//!
//! * **Seeds.** The block's stores of an `f32`, `f64`, `i32` or `i64` value, grouped by the
//!   address decomposition `(base, symbolic index, element type)` ([`addr_of`]): a `gep` off
//!   `base` whose index is a constant, `x + c`, `x - c` or `x`, with `x` an `i64` (an `i32`
//!   `x + c` can wrap, and then the elements are not contiguous). A group storing one element
//!   twice is skipped. Runs of consecutive constants split into chunks of `W = 16 / bytes(T)`
//!   lanes, lane `j` storing element `k0 + j`.
//! * **The tree**, built bottom-up from the stored values in lane order (at most [`MAX_DEPTH`]
//!   deep and [`MAX_NODES`] nodes). A bundle of `W` values is a **VECTOR** node when every value
//!   is a distinct, single-use result of this block computed by the same lane-wise operation — a
//!   `Bin` the loop vectorizer would widen at `T` (no shift, no division:
//!   `vectorize::bin_widenable`), an `fma`/`sqrt`/`round` of floats, a `not` of integers, or a
//!   `neg` — whose operand bundles are its children, position by position. Commutative operands
//!   are **never swapped**: a float swap would change which NaN payload survives. A **LOAD** node
//!   is `W` loads of this block from consecutive elements; their scalar loads stay for any other
//!   reader. A **SPLAT** is one value, or one constant, in every lane. Anything else is a
//!   **BUILD**: a splat of lane 0 and `W - 1` lane inserts.
//! * **Cost** ([`tree_cost`]): the vector side is the store, one per VECTOR, LOAD and SPLAT node,
//!   and `W` per BUILD; the scalar side is the `W` stores, `W` per VECTOR node and every
//!   single-use LOAD lane. A tree with no VECTOR or LOAD node, or one that does not cost strictly
//!   less, is declined. Untuned, like the loop vectorizer's model: a speed claim needs a same-run
//!   release A/B against `WUKONG_NO_SLP`.
//!
//! # Soundness: what moves, and the crossing rule
//!
//! Every new instruction is inserted before the chunk's **last** store (index `P`), and the `W`
//! scalar stores are deleted. So the chunk's stores move *later*, to `P`, and so do the LOAD
//! nodes' loads (re-read at `P` as one vector load). Every value computation is pure, and every
//! value the new code reads is defined before its use in the scalar tree, which is before `P`. Two
//! orders can change, and [`legal`] checks both with [`AliasInfo::may_alias_sized`] on the exact
//! access widths:
//!
//! * a moved access against every *unmoved* instruction it moves past (strictly between the first
//!   moved access and `P`): a store moved past a load or store, or a load moved past a store, must
//!   be proved disjoint; a call (or vector-kernel call) in between declines unless the moved
//!   access is to storage no callee can name ([`AliasInfo::is_call_proof`]); a barrier declines;
//! * the moved accesses among themselves: one vector load at `P` now runs *before* the vector
//!   store at `P`, so a chunk store that originally preceded a LOAD lane must be proved disjoint
//!   from it — otherwise the tree may read back what the bundle stores.
//!
//! Lane-wise vector arithmetic is bit-exact against the scalar operations it replaces (no
//! reassociation, no operand swap, the same rounding per lane), and every vector access touches
//! exactly the elements its `W` scalar accesses touched, so nothing is speculated. No new MIR
//! operation exists: the output is `Load`/`Store`/`Splat`/`InsertLane` and the widened arithmetic
//! the loop vectorizer already emits, so no backend changed.
//!
//! **The PTX caveat.** These are 128-bit vector loads and stores **outside loops**, including on
//! frame slots. `gpu-native` lowers a vector access it cannot prove 16-byte aligned through its
//! generic scalar-lane path, but only the on-device suite can prove that (`CLAUDE.md`, "PTX has
//! three address windows").
//!
//! # Switches
//!
//! `WUKONG_NO_SLP` (presence) turns the pass off, read here so every caller — `optimize`, the
//! verified and timed paths, and the measurement replicas — honours it identically.
//! `WUKONG_SLP_TRACE=1` prints one stderr line per decision: `slp: PACK bb<N> ty=<T> w=<W>
//! nodes=<n> cost=<V>/<S>`, or `slp: declined bb<N>: <reason>` (each declined seed once, although
//! a rescan after a pack re-tries it).

use wukong_mir::{BinOp, Function, Inst, MirType, Op, Program, RoundMode, ValueId};

use crate::alias::{type_bytes, AliasInfo};
use crate::fxhash::{FxHashMap, FxHashSet};

/// The vector register width this pass packs into, as the loop vectorizer's: Cranelift's x64 ISA
/// legalizes 128-bit vectors only.
const VEC_BYTES: u32 = 16;
/// At most this many packs per function, so a bug that keeps finding the same seed degrades into
/// "stopped early" rather than a compile that never ends.
const MAX_PACKS_PER_FN: usize = 64;
/// A function larger than this is skipped: each pack rescans the whole function.
const MAX_FN_INSTS: usize = 6000;
/// The deepest a tree is expanded; below it every bundle is built from its scalars.
const MAX_DEPTH: u32 = 8;
/// The most nodes one tree may have before the seed is declined.
const MAX_NODES: usize = 32;

/// Pack the straight-line store bundles of every function. Returns the indices of the functions it
/// changed, for the caller's cleanup sweep. Empty when `WUKONG_NO_SLP` is set.
pub fn slp_program(program: &mut Program) -> Vec<usize> {
    if std::env::var_os("WUKONG_NO_SLP").is_some() {
        return Vec::new();
    }
    let trace = std::env::var_os("WUKONG_SLP_TRACE").is_some();
    let mut changed = Vec::new();
    for (i, f) in program.funcs.iter_mut().enumerate() {
        if slp_function(f, trace) {
            changed.push(i);
        }
    }
    changed
}

/// The lane types this pass packs, and their width in bytes.
fn lane_bytes(t: &MirType) -> Option<u32> {
    match t {
        MirType::I32 | MirType::F32 => Some(4),
        MirType::I64 | MirType::F64 => Some(8),
        _ => None,
    }
}

/// Pack one function's bundles. Returns whether anything changed.
fn slp_function(f: &mut Function, trace: bool) -> bool {
    let size: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
    if size > MAX_FN_INSTS {
        return false;
    }
    // Cheap structural pre-filter, before any table is built: a block with two stores of a packed
    // lane type. Most functions have none.
    let candidate = f.blocks.iter().any(|b| {
        b.insts
            .iter()
            .filter(|i| match &i.op {
                Op::Store { value, .. } => lane_bytes(f.value_type(*value)).is_some(),
                _ => false,
            })
            .nth(1)
            .is_some()
    });
    if !candidate {
        return false;
    }
    let mut packs = 0;
    let mut from_block = 0;
    // Declined seeds already reported, by their first store's `(ptr, value)`: a rescan after a pack
    // re-tries them, but the trace names each once.
    let mut reported: FxHashSet<(u32, u32)> = FxHashSet::default();
    while packs < MAX_PACKS_PER_FN {
        let facts = Facts::of(f);
        let Some((b, seed, tree, facts)) = first_pack(f, facts, from_block, trace, &mut reported)
        else {
            break;
        };
        emit_pack(f, &facts, b, &seed, &tree);
        packs += 1;
        // Blocks before `b` were declined in full and a pack in `b` cannot change their answer
        // (it only adds values and deletes `b`'s stores), so the rescan starts at `b`.
        from_block = b;
    }
    packs > 0
}

/// The first seed from block `from` on that packs: its block, the seed, its plan and the facts it
/// was planned with (which `emit_pack` reads before it mutates anything).
fn first_pack(
    f: &Function,
    mut facts: Facts,
    from: usize,
    trace: bool,
    reported: &mut FxHashSet<(u32, u32)>,
) -> Option<(usize, Seed, Tree, Facts)> {
    for b in from..f.blocks.len() {
        for seed in seeds_of(f, &facts, b) {
            match plan_pack(f, &mut facts, b, &seed) {
                Ok(tree) => {
                    if trace {
                        eprintln!(
                            "slp: PACK bb{b} ty={} w={} nodes={} cost={}/{}",
                            seed.ty.display(),
                            seed.lanes.len(),
                            tree.nodes.len(),
                            tree.cost.0,
                            tree.cost.1
                        );
                    }
                    return Some((b, seed, tree, facts));
                }
                Err(why) => {
                    let first = &seed.lanes[0];
                    if trace && reported.insert((first.ptr.0, first.value.0)) {
                        eprintln!("slp: declined bb{b}: {why}");
                    }
                }
            }
        }
    }
    None
}

/// What one scan of a function knows: where each value is defined, how often it is read, and —
/// built on the first legality question — the alias facts. Rebuilt after every pack, because a
/// pack adds values and deletes stores.
struct Facts {
    /// `ValueId -> (block, instruction index)` for every instruction result.
    def: Vec<Option<(u32, u32)>>,
    /// How many operand positions (instructions and terminators) read each value.
    uses: Vec<u32>,
    alias: Option<AliasInfo>,
}

impl Facts {
    fn of(f: &Function) -> Facts {
        let n = f.value_types.len();
        let mut def = vec![None; n];
        let mut uses = vec![0u32; n];
        for (bi, b) in f.blocks.iter().enumerate() {
            for (ii, inst) in b.insts.iter().enumerate() {
                if let Some(r) = inst.result {
                    if let Some(d) = def.get_mut(r.0 as usize) {
                        *d = Some((bi as u32, ii as u32));
                    }
                }
                crate::each_op_use(&inst.op, &mut |u| {
                    if let Some(c) = uses.get_mut(u.0 as usize) {
                        *c += 1;
                    }
                });
            }
            crate::each_term_use(&b.term, &mut |u| {
                if let Some(c) = uses.get_mut(u.0 as usize) {
                    *c += 1;
                }
            });
        }
        Facts {
            def,
            uses,
            alias: None,
        }
    }

    /// The instruction defining `v`, anywhere in the function.
    fn op<'f>(&self, f: &'f Function, v: ValueId) -> Option<&'f Op> {
        let (b, i) = (*self.def.get(v.0 as usize)?)?;
        Some(&f.blocks[b as usize].insts[i as usize].op)
    }

    /// The index of `v`'s defining instruction, when it is in block `b`.
    fn index_in(&self, b: usize, v: ValueId) -> Option<usize> {
        match self.def.get(v.0 as usize).copied().flatten() {
            Some((db, i)) if db as usize == b => Some(i as usize),
            _ => None,
        }
    }

    fn uses(&self, v: ValueId) -> u32 {
        self.uses.get(v.0 as usize).copied().unwrap_or(u32::MAX)
    }

    fn alias(&mut self, f: &Function) -> &AliasInfo {
        self.alias.get_or_insert_with(|| AliasInfo::analyze(f))
    }
}

/// An address as `base + (sym + k)` elements of `elem`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Addr {
    base: ValueId,
    /// The symbolic part of the index, an `i64`; `None` for a constant index.
    sym: Option<ValueId>,
    k: i64,
}

/// Decompose a pointer for a `T`-wide access: `gep b, c` is `(b, None, c)`; `gep b, x + c` is
/// `(b, x, c)` (and `x - c` is `-c`, `c + x` is `c`); `gep b, x` is `(b, x, 0)`; any other pointer
/// `p` is `(p, None, 0)`. The `gep`'s element type must be `T` itself, and a symbolic index an
/// `i64`: an `i32` `x + c` that wraps would not address the element after `x + c - 1`.
fn addr_of(f: &Function, facts: &Facts, p: ValueId, ty: &MirType) -> Option<Addr> {
    let Some(Op::Gep { ptr, index, elem }) = facts.op(f, p) else {
        return Some(Addr {
            base: p,
            sym: None,
            k: 0,
        });
    };
    if elem != ty {
        return None;
    }
    let konst = |v: ValueId| match facts.op(f, v) {
        Some(Op::ConstInt(c, _)) => i64::try_from(*c).ok(),
        _ => None,
    };
    let (sym, k) = if let Some(c) = konst(*index) {
        (None, c)
    } else {
        match facts.op(f, *index) {
            Some(Op::Bin(BinOp::Add, x, y)) => match (konst(*x), konst(*y)) {
                (_, Some(c)) => (Some(*x), c),
                (Some(c), None) => (Some(*y), c),
                (None, None) => (Some(*index), 0),
            },
            Some(Op::Bin(BinOp::Sub, x, y)) => match konst(*y).and_then(i64::checked_neg) {
                Some(c) => (Some(*x), c),
                None => (Some(*index), 0),
            },
            _ => (Some(*index), 0),
        }
    };
    if let Some(s) = sym {
        if *f.value_type(s) != MirType::I64 {
            return None;
        }
    }
    Some(Addr { base: *ptr, sym, k })
}

/// One scalar store of a chunk.
#[derive(Clone, Copy, Debug)]
struct Lane {
    /// Its index in the block.
    idx: usize,
    ptr: ValueId,
    value: ValueId,
}

/// `W` stores to consecutive elements, in lane order (lane `j` stores element `k0 + j`).
#[derive(Clone, Debug)]
struct Seed {
    ty: MirType,
    lanes: Vec<Lane>,
}

/// The block's seeds, sorted by their first store's index ([`Seed`]).
fn seeds_of(f: &Function, facts: &Facts, b: usize) -> Vec<Seed> {
    type Key = (u32, Option<u32>, MirType);
    let mut order: Vec<(Key, Vec<(i64, Lane)>)> = Vec::new();
    let mut slot: FxHashMap<Key, usize> = FxHashMap::default();
    for (idx, inst) in f.blocks[b].insts.iter().enumerate() {
        let Op::Store { ptr, value } = &inst.op else {
            continue;
        };
        let ty = f.value_type(*value);
        if lane_bytes(ty).is_none() {
            continue;
        }
        let Some(a) = addr_of(f, facts, *ptr, ty) else {
            continue;
        };
        let key = (a.base.0, a.sym.map(|s| s.0), ty.clone());
        let lane = Lane {
            idx,
            ptr: *ptr,
            value: *value,
        };
        match slot.get(&key) {
            Some(&g) => order[g].1.push((a.k, lane)),
            None => {
                slot.insert(key.clone(), order.len());
                order.push((key, vec![(a.k, lane)]));
            }
        }
    }
    let mut seeds = Vec::new();
    for ((_, _, ty), mut stores) in order {
        let w = (VEC_BYTES / lane_bytes(&ty).expect("a seed's type packs")) as usize;
        if stores.len() < w {
            continue;
        }
        stores.sort_by_key(|(k, _)| *k);
        // One element stored twice: which store wins is program order, which a pack would change.
        if stores.windows(2).any(|p| p[0].0 == p[1].0) {
            continue;
        }
        let mut run_start = 0;
        for i in 1..=stores.len() {
            let breaks = i == stores.len() || stores[i].0 != stores[i - 1].0 + 1;
            if !breaks {
                continue;
            }
            let run = &stores[run_start..i];
            for chunk in run.chunks_exact(w) {
                seeds.push(Seed {
                    ty: ty.clone(),
                    lanes: chunk.iter().map(|(_, l)| *l).collect(),
                });
            }
            run_start = i;
        }
    }
    seeds.sort_by_key(|s| s.lanes.iter().map(|l| l.idx).min());
    seeds
}

/// What a bundle of `W` values becomes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// One lane-wise operation over the children.
    Vector,
    /// One vector load of consecutive elements.
    Load,
    /// One value in every lane.
    Splat,
    /// A splat of lane 0 and a lane insert per further lane.
    Build,
}

#[derive(Clone, Debug)]
struct Node {
    kind: Kind,
    lanes: Vec<ValueId>,
    /// Operand bundles, in operand order (VECTOR only).
    children: Vec<usize>,
}

/// A planned pack: the nodes in post-order (every child before its parent, the root last) and the
/// `(vector, scalar)` cost estimate.
#[derive(Debug)]
struct Tree {
    nodes: Vec<Node>,
    cost: (u64, u64),
}

/// The lane-wise operation shapes a VECTOR node may have.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Bin(BinOp),
    Fma,
    Sqrt,
    Round(RoundMode),
    Neg,
    Not,
}

/// The shape of `op` and its operands in operand order, when it is one a VECTOR node takes at
/// lane type `ty`.
fn shape_of(op: &Op, ty: &MirType) -> Option<(Shape, Vec<ValueId>)> {
    let float = ty.is_float();
    Some(match op {
        Op::Bin(b, x, y) if crate::vectorize::bin_widenable(*b, ty) => {
            (Shape::Bin(*b), vec![*x, *y])
        }
        Op::Fma(a, b, c) if float => (Shape::Fma, vec![*a, *b, *c]),
        Op::Sqrt(a) if float => (Shape::Sqrt, vec![*a]),
        Op::Round(m, a) if float => (Shape::Round(*m), vec![*a]),
        Op::Neg(a) => (Shape::Neg, vec![*a]),
        Op::Not(a) if ty.is_int() => (Shape::Not, vec![*a]),
        _ => return None,
    })
}

/// `op` with its operands replaced, in [`shape_of`]'s order.
fn with_operands(op: &Op, xs: &[ValueId]) -> Op {
    match op {
        Op::Bin(b, ..) => Op::Bin(*b, xs[0], xs[1]),
        Op::Fma(..) => Op::Fma(xs[0], xs[1], xs[2]),
        Op::Sqrt(_) => Op::Sqrt(xs[0]),
        Op::Round(m, _) => Op::Round(*m, xs[0]),
        Op::Neg(_) => Op::Neg(xs[0]),
        Op::Not(_) => Op::Not(xs[0]),
        _ => unreachable!("shape_of admitted only these operations"),
    }
}

/// Builds one seed's tree.
struct TreeBuilder<'a> {
    f: &'a Function,
    facts: &'a Facts,
    b: usize,
    ty: &'a MirType,
    nodes: Vec<Node>,
}

impl TreeBuilder<'_> {
    fn build(&mut self, lanes: Vec<ValueId>, depth: u32) -> Result<usize, &'static str> {
        if self.nodes.len() >= MAX_NODES {
            return Err("the tree has more than 32 nodes");
        }
        if lanes.iter().any(|v| self.f.value_type(*v) != self.ty) {
            return Err("a bundle mixes lane types");
        }
        let (kind, children) = if self.is_splat(&lanes) {
            (Kind::Splat, Vec::new())
        } else if self.is_load(&lanes) {
            (Kind::Load, Vec::new())
        } else if let Some(operands) = self.vector_operands(&lanes, depth) {
            let mut children = Vec::with_capacity(operands.len());
            for bundle in operands {
                children.push(self.build(bundle, depth + 1)?);
            }
            (Kind::Vector, children)
        } else {
            (Kind::Build, Vec::new())
        };
        self.nodes.push(Node {
            kind,
            lanes,
            children,
        });
        Ok(self.nodes.len() - 1)
    }

    /// One value in every lane, or one constant spelled by several equal instructions.
    fn is_splat(&self, lanes: &[ValueId]) -> bool {
        if lanes.iter().all(|v| *v == lanes[0]) {
            return true;
        }
        let konst = |v: ValueId| match self.facts.op(self.f, v) {
            Some(Op::ConstInt(c, t)) => Some((Some(*c), None, t)),
            Some(Op::ConstFloat(x, t)) => Some((None, Some(x.to_bits()), t)),
            _ => None,
        };
        let Some(first) = konst(lanes[0]) else {
            return false;
        };
        lanes[1..].iter().all(|v| konst(*v) == Some(first))
    }

    /// `W` loads of this block from consecutive elements of one object, lane `j` at `k0 + j`.
    fn is_load(&self, lanes: &[ValueId]) -> bool {
        let mut first: Option<Addr> = None;
        for (j, v) in lanes.iter().enumerate() {
            if self.facts.index_in(self.b, *v).is_none() {
                return false;
            }
            let Some(Op::Load(p, t)) = self.facts.op(self.f, *v) else {
                return false;
            };
            if t != self.ty {
                return false;
            }
            let Some(a) = addr_of(self.f, self.facts, *p, self.ty) else {
                return false;
            };
            match first {
                None => first = Some(a),
                Some(a0) => {
                    let want = a0.k.checked_add(j as i64);
                    if a.base != a0.base || a.sym != a0.sym || Some(a.k) != want {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// The operand bundles of a VECTOR node, when the lanes are distinct single-use results of this
    /// block computed by one lane-wise operation shape.
    fn vector_operands(&self, lanes: &[ValueId], depth: u32) -> Option<Vec<Vec<ValueId>>> {
        if depth >= MAX_DEPTH {
            return None;
        }
        let mut shape: Option<Shape> = None;
        let mut operands: Vec<Vec<ValueId>> = Vec::new();
        for (j, v) in lanes.iter().enumerate() {
            self.facts.index_in(self.b, *v)?;
            if self.facts.uses(*v) != 1 || lanes[..j].contains(v) {
                return None;
            }
            let (s, xs) = shape_of(self.facts.op(self.f, *v)?, self.ty)?;
            match shape {
                None => {
                    shape = Some(s);
                    operands = xs.iter().map(|x| vec![*x]).collect();
                }
                Some(s0) if s0 == s => {
                    for (bundle, x) in operands.iter_mut().zip(xs) {
                        bundle.push(x);
                    }
                }
                Some(_) => return None,
            }
        }
        Some(operands)
    }
}

/// `(vector, scalar)` for one tree of `w` lanes: the vector side is the store, one per VECTOR,
/// LOAD and SPLAT node, and `w` per BUILD; the scalar side is the `w` stores, `w` per VECTOR node,
/// and each LOAD lane nothing else reads (its scalar load dies with the stores).
fn tree_cost(nodes: &[Node], facts: &Facts, w: u64) -> (u64, u64) {
    let mut vector = 1;
    let mut scalar = w;
    for n in nodes {
        match n.kind {
            Kind::Vector => {
                vector += 1;
                scalar += w;
            }
            Kind::Load => {
                vector += 1;
                scalar += n.lanes.iter().filter(|v| facts.uses(**v) == 1).count() as u64;
            }
            Kind::Splat => vector += 1,
            Kind::Build => vector += w,
        }
    }
    (vector, scalar)
}

/// Plan one seed: build its tree, cost it, and check the moves it makes ([`legal`]).
fn plan_pack(f: &Function, facts: &mut Facts, b: usize, seed: &Seed) -> Result<Tree, &'static str> {
    let mut tb = TreeBuilder {
        f,
        facts,
        b,
        ty: &seed.ty,
        nodes: Vec::new(),
    };
    tb.build(seed.lanes.iter().map(|l| l.value).collect(), 0)?;
    let nodes = tb.nodes;
    if !nodes
        .iter()
        .any(|n| matches!(n.kind, Kind::Vector | Kind::Load))
    {
        return Err("the stored values share no vector operation and no load");
    }
    let cost = tree_cost(&nodes, facts, seed.lanes.len() as u64);
    if cost.0 >= cost.1 {
        return Err("the packed tree costs no less than its scalar stores");
    }
    legal(f, facts, b, seed, &nodes)?;
    Ok(Tree { nodes, cost })
}

/// A memory access the pack moves: its index in the block, pointer, width, and whether it stores.
#[derive(Clone, Copy)]
struct Moved {
    idx: usize,
    ptr: ValueId,
    bytes: u32,
    store: bool,
}

/// The crossing rule (module docs): every access the pack moves to the chunk's last store must not
/// pass an unmoved access, call or barrier that could observe the move, and the vector loads now
/// ahead of the vector store must not read what an earlier chunk store wrote.
fn legal(
    f: &Function,
    facts: &mut Facts,
    b: usize,
    seed: &Seed,
    nodes: &[Node],
) -> Result<(), &'static str> {
    let bytes = type_bytes(&seed.ty).expect("a packed lane type has a width");
    let mut moved: Vec<Moved> = seed
        .lanes
        .iter()
        .map(|l| Moved {
            idx: l.idx,
            ptr: l.ptr,
            bytes,
            store: true,
        })
        .collect();
    let insts = &f.blocks[b].insts;
    for n in nodes.iter().filter(|n| n.kind == Kind::Load) {
        for v in &n.lanes {
            let idx = facts
                .index_in(b, *v)
                .expect("a LOAD lane is defined in this block");
            if moved.iter().any(|m| m.idx == idx) {
                continue; // the same load in two LOAD nodes
            }
            let Op::Load(p, _) = &insts[idx].op else {
                unreachable!("a LOAD lane is a load")
            };
            moved.push(Moved {
                idx,
                ptr: *p,
                bytes,
                store: false,
            });
        }
    }
    let at = moved.iter().filter(|m| m.store).map(|m| m.idx).max();
    let at = at.expect("a seed has stores");
    let lo = moved
        .iter()
        .map(|m| m.idx)
        .min()
        .expect("a seed has stores");
    let alias = facts.alias(f);
    let width = |op: &Op| AliasInfo::access_bytes(f, op).unwrap_or(u32::MAX);
    for (j, inst) in insts.iter().enumerate().take(at).skip(lo + 1) {
        if moved.iter().any(|m| m.idx == j) {
            continue;
        }
        let mut passed = moved.iter().filter(|m| m.idx < j);
        match &inst.op {
            Op::Call { .. } | Op::VecKernelCall { .. } => {
                if passed.any(|m| !alias.is_call_proof(m.ptr)) {
                    return Err("a call between the members may touch a moved access");
                }
            }
            Op::Barrier => return Err("a barrier between the members"),
            Op::Store { ptr, .. } => {
                let w = width(&inst.op);
                if passed.any(|m| alias.may_alias_sized(m.ptr, m.bytes, *ptr, w)) {
                    return Err("a store between the members may alias a moved access");
                }
            }
            Op::Load(p, _) => {
                let w = width(&inst.op);
                if passed.any(|m| m.store && alias.may_alias_sized(m.ptr, m.bytes, *p, w)) {
                    return Err("a load between the members may read a moved store");
                }
            }
            _ => {}
        }
    }
    for s in moved.iter().filter(|m| m.store) {
        for l in moved.iter().filter(|m| !m.store && m.idx > s.idx) {
            if alias.may_alias_sized(s.ptr, s.bytes, l.ptr, l.bytes) {
                return Err("a bundle store may be read back by a load of its own tree");
            }
        }
    }
    Ok(())
}

/// Emit the planned pack before the chunk's last store and delete the chunk's `W` scalar stores.
/// The scalar computations they leave dead are the cleanup's `dce`'s.
fn emit_pack(f: &mut Function, facts: &Facts, b: usize, seed: &Seed, tree: &Tree) {
    let w = seed.lanes.len();
    let vty = MirType::Vec(Box::new(seed.ty.clone()), w as u32);
    // The defining op of each lane-0 value, read before anything moves.
    let lane0_op: Vec<Option<Op>> = tree
        .nodes
        .iter()
        .map(|n| facts.op(f, n.lanes[0]).cloned())
        .collect();
    let mut code: Vec<Inst> = Vec::new();
    let push = |f: &mut Function, code: &mut Vec<Inst>, ty: MirType, op: Op| {
        let id = ValueId(f.value_types.len() as u32);
        f.value_types.push(ty);
        code.push(Inst {
            result: Some(id),
            op,
        });
        id
    };
    let mut vals: Vec<ValueId> = Vec::with_capacity(tree.nodes.len());
    for (n, op0) in tree.nodes.iter().zip(&lane0_op) {
        let v = match n.kind {
            Kind::Load => {
                let Some(Op::Load(p, _)) = op0 else {
                    unreachable!("a LOAD node's lanes are loads")
                };
                push(f, &mut code, vty.clone(), Op::Load(*p, vty.clone()))
            }
            Kind::Vector => {
                let xs: Vec<ValueId> = n.children.iter().map(|c| vals[*c]).collect();
                let op = with_operands(op0.as_ref().expect("a VECTOR lane is an instruction"), &xs);
                push(f, &mut code, vty.clone(), op)
            }
            Kind::Splat => push(f, &mut code, vty.clone(), Op::Splat(n.lanes[0])),
            Kind::Build => {
                let mut v = push(f, &mut code, vty.clone(), Op::Splat(n.lanes[0]));
                for (j, x) in n.lanes.iter().enumerate().skip(1) {
                    v = push(f, &mut code, vty.clone(), Op::InsertLane(v, *x, j as u32));
                }
                v
            }
        };
        vals.push(v);
    }
    let root = *vals.last().expect("a tree has a root");
    code.push(Inst {
        result: None,
        op: Op::Store {
            ptr: seed.lanes[0].ptr,
            value: root,
        },
    });
    let at = seed
        .lanes
        .iter()
        .map(|l| l.idx)
        .max()
        .expect("a seed has stores");
    let gone: FxHashSet<usize> = seed.lanes.iter().map(|l| l.idx).collect();
    let old = std::mem::take(&mut f.blocks[b].insts);
    let mut out = Vec::with_capacity(old.len() + code.len());
    let mut code = Some(code);
    for (i, inst) in old.into_iter().enumerate() {
        if i == at {
            out.extend(code.take().expect("emitted once"));
        }
        if !gone.contains(&i) {
            out.push(inst);
        }
    }
    f.blocks[b].insts = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        (program, interner)
    }

    fn optimized(src: &str, level: u8) -> (Program, Interner) {
        let (mut program, interner) = lower(src);
        crate::optimize(&mut program, level);
        for f in &program.funcs {
            let errs = wukong_mir::verify::verify_function(f);
            assert!(errs.is_empty(), "MIR invalid after -O{level}: {errs:?}");
        }
        (program, interner)
    }

    /// The `-O2` instructions of one function, flattened.
    fn ops_of(src: &str, func: &str) -> (Vec<Op>, Vec<MirType>) {
        let (program, mut interner) = optimized(src, 2);
        let f = program
            .function(interner.intern(func))
            .unwrap_or_else(|| panic!("no fn {func}"));
        let ops = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .map(|i| i.op.clone())
            .collect();
        (ops, f.value_types.clone())
    }

    fn vec_ty(t: MirType, n: u32) -> MirType {
        MirType::Vec(Box::new(t), n)
    }

    /// Stores of `ty` (scalar or vector) in the ops of a function.
    fn stores_of(ops: &[Op], types: &[MirType], ty: &MirType) -> usize {
        ops.iter()
            .filter(|op| matches!(op, Op::Store { value, .. } if types[value.0 as usize] == *ty))
            .count()
    }

    fn loads_of(ops: &[Op], ty: &MirType) -> usize {
        ops.iter()
            .filter(|op| matches!(op, Op::Load(_, t) if t == ty))
            .count()
    }

    fn has_vectors(src: &str, func: &str) -> bool {
        let (_, types) = ops_of(src, func);
        types.iter().any(|t| t.is_vector())
    }

    /// `o[k] = a[k]*s + b[k]` four times: the front end contracts each into one `fma`, so the
    /// bundle is one vector `fma` over two vector loads and a splat of `s`, and one vector store
    /// replaces the four scalar ones.
    #[test]
    fn four_fma_stores_pack_into_one_vector_store() {
        let src = "fn axpy4(@noalias mut o: [f32; 4], a: [f32; 4], b: [f32; 4], s: f32) { \
                   o[0] = a[0] * s + b[0]; o[1] = a[1] * s + b[1]; \
                   o[2] = a[2] * s + b[2]; o[3] = a[3] * s + b[3]; } \
                   fn main() -> i32 { return 0; }";
        let (ops, types) = ops_of(src, "axpy4");
        let f32x4 = vec_ty(MirType::F32, 4);
        assert_eq!(stores_of(&ops, &types, &f32x4), 1, "{ops:?}");
        assert_eq!(stores_of(&ops, &types, &MirType::F32), 0, "{ops:?}");
        assert_eq!(loads_of(&ops, &f32x4), 2, "{ops:?}");
        assert!(ops.iter().any(|op| matches!(op, Op::Splat(_))), "{ops:?}");
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Fma(x, ..) if types[x.0 as usize] == f32x4)),
            "{ops:?}"
        );
    }

    /// The other two widths: an `i32` quad (`<4 x i32>` add of a load and a multiply) and an `f64`
    /// pair (`<2 x f64>`).
    #[test]
    fn an_i32_quad_and_an_f64_pair_pack() {
        let src = "fn iq(@noalias mut o: [i32; 4], a: [i32; 4], b: [i32; 4]) { \
                   o[0] = a[0] + b[0] * 3; o[1] = a[1] + b[1] * 3; \
                   o[2] = a[2] + b[2] * 3; o[3] = a[3] + b[3] * 3; } \
                   fn dp(@noalias mut o: [f64; 2], a: [f64; 2], s: f64) { \
                   o[0] = a[0] * s; o[1] = a[1] * s; } \
                   fn main() -> i32 { return 0; }";
        let (ops, types) = ops_of(src, "iq");
        let i32x4 = vec_ty(MirType::I32, 4);
        assert_eq!(stores_of(&ops, &types, &i32x4), 1, "{ops:?}");
        assert_eq!(stores_of(&ops, &types, &MirType::I32), 0, "{ops:?}");
        let (ops, types) = ops_of(src, "dp");
        let f64x2 = vec_ty(MirType::F64, 2);
        assert_eq!(stores_of(&ops, &types, &f64x2), 1, "{ops:?}");
        assert_eq!(stores_of(&ops, &types, &MirType::F64), 0, "{ops:?}");
    }

    /// `for k in 0..8 step 2` is declined by the loop vectorizer for its step; the unroller then
    /// replaces it by its four iterations with `k` folded, which leaves eight stores to `o[0..8]`:
    /// two chunks, two vector stores.
    #[test]
    fn a_fully_unrolled_step_two_loop_packs() {
        let src = "fn k(@noalias mut o: [f32; 8], a: [f32; 8], s: f32) { \
                   for k in 0..8 step 2 { o[k] = a[k] * s; o[k + 1] = a[k + 1] * s; } } \
                   fn main() -> i32 { return 0; }";
        let (ops, types) = ops_of(src, "k");
        assert_eq!(
            stores_of(&ops, &types, &vec_ty(MirType::F32, 4)),
            2,
            "{ops:?}"
        );
        assert_eq!(stores_of(&ops, &types, &MirType::F32), 0, "{ops:?}");
    }

    /// No promise: `o` and `a` may be one buffer. The loads all come before the first store, so
    /// re-reading them as one vector load at the last store moves no load past a store, and the
    /// pack is legal anyway.
    #[test]
    fn loads_ahead_of_all_stores_pack_without_a_promise() {
        let src = "fn k(mut o: [f32; 4], a: [f32; 4], s: f32) { \
                   let x0: f32 = a[0]; let x1: f32 = a[1]; let x2: f32 = a[2]; let x3: f32 = a[3]; \
                   o[0] = x0 * s; o[1] = x1 * s; o[2] = x2 * s; o[3] = x3 * s; } \
                   fn main() -> i32 { return 0; }";
        let (ops, types) = ops_of(src, "k");
        assert_eq!(
            stores_of(&ops, &types, &vec_ty(MirType::F32, 4)),
            1,
            "{ops:?}"
        );
    }

    /// The same loads, but a store through `p` — which may be `a` — sits between them and the
    /// bundle's stores. Re-reading `a` after it could see `p`'s write, so nothing packs.
    #[test]
    fn a_load_crossing_a_possibly_aliased_store_is_not_packed() {
        let src = "fn k(@noalias mut o: [f32; 4], a: [f32; 4], mut p: [f32; 4], s: f32) { \
                   let x0: f32 = a[0]; let x1: f32 = a[1]; let x2: f32 = a[2]; let x3: f32 = a[3]; \
                   p[2] = 9.0; \
                   o[0] = x0 * s; o[1] = x1 * s; o[2] = x2 * s; o[3] = x3 * s; } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !has_vectors(src, "k"),
            "a load moved past a possibly-aliased store"
        );
    }

    /// `o[i + c] = o[m + c] * s`: with `m = i - 1` lane 1 reads the element lane 0 just stored —
    /// a recurrence through memory. The one vector load at the last store would read it before
    /// the vector store writes it, so the pack is declined, and the program agrees with `-O0`.
    /// (An array parameter, so every access is one `gep` off the same base: a slice would re-read
    /// its data pointer around each store and give each access a base of its own.)
    #[test]
    fn a_bundle_store_read_back_by_its_own_tree_is_not_packed() {
        let k = "@unchecked fn k(mut o: [f32; 8], i: i64, m: i64, s: f32) { \
                 o[i] = o[m] * s; o[i + 1] = o[m + 1] * s; \
                 o[i + 2] = o[m + 2] * s; o[i + 3] = o[m + 3] * s; }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            !has_vectors(&shape, "k"),
            "a store read back by its own tree"
        );
        scalar_and_slp_agree(
            &format!(
                "{k} @unchecked fn main() -> i32 {{ let mut o: [f32; 8] = [0.0; 8]; \
                 let mut j: i64 = 0; \
                 while j < 8 {{ o[j] = (j as f32) + 1.0; j = j + 1; }} k(o, 2, 1, 2.0); \
                 j = 0; while j < 8 {{ print(o[j]); j = j + 1; }} return 0; }}"
            ),
            false,
        );
    }

    /// A call between the members may read the stored elements before they are written: `o` is a
    /// plain parameter, so the callee can name it. Without the call the same bundle packs.
    #[test]
    fn a_call_between_members_blocks_packing() {
        let with_call = "fn k(mut o: [f32; 4], s: f32) { \
                         o[0] = s * 1.5; o[1] = s * 2.5; print(0); \
                         o[2] = s * 3.5; o[3] = s * 4.5; } \
                         fn main() -> i32 { return 0; }";
        assert!(!has_vectors(with_call, "k"), "stores moved past a call");
        let without = "fn k(mut o: [f32; 4], s: f32) { \
                       o[0] = s * 1.5; o[1] = s * 2.5; o[2] = s * 3.5; o[3] = s * 4.5; } \
                       fn main() -> i32 { return 0; }";
        assert!(has_vectors(without, "k"), "the control bundle did not pack");
        // `@noalias` storage nothing derived from escapes is storage no callee can name, so the
        // same call is no obstacle there.
        let proof = "fn k(@noalias mut o: [f32; 4], s: f32) { \
                     o[0] = s * 1.5; o[1] = s * 2.5; print(0); o[2] = s * 3.5; o[3] = s * 4.5; } \
                     fn main() -> i32 { return 0; }";
        assert!(has_vectors(proof, "k"), "a call blocked call-proof storage");
    }

    /// Moving the first two stores down to the last one passes a load and a store through `p`,
    /// which may be `o`: the load could read a value the moved store no longer wrote, the store
    /// could be overwritten by the moved one. Both decline.
    #[test]
    fn a_load_or_store_between_members_that_may_alias_blocks_packing() {
        let load = "fn k(mut o: [f32; 4], s: f32, p: [f32; 4]) -> f32 { \
                    o[0] = s * 1.5; o[1] = s * 2.5; let y: f32 = p[1]; \
                    o[2] = s * 3.5; o[3] = s * 4.5; return y; } \
                    fn main() -> i32 { return 0; }";
        assert!(
            !has_vectors(load, "k"),
            "a store moved past a possibly-aliased load"
        );
        let store = "fn k(mut o: [f32; 4], s: f32, mut p: [f32; 4]) { \
                     o[0] = s * 1.5; o[1] = s * 2.5; p[1] = 7.0; o[2] = s * 3.5; o[3] = s * 4.5; } \
                     fn main() -> i32 { return 0; }";
        assert!(
            !has_vectors(store, "k"),
            "a store moved past a possibly-aliased store"
        );
    }

    /// A barrier orders memory between the threads of a block; no access moves across one.
    #[test]
    fn a_barrier_between_members_blocks_packing() {
        let src = "@parallel fn k(@noalias mut o: [f32; 4], s: f32) { \
                   o[0] = s * 1.5; o[1] = s * 2.5; sync_block(); o[2] = s * 3.5; o[3] = s * 4.5; } \
                   fn main() -> i32 { return 0; }";
        assert!(!has_vectors(src, "k"), "stores moved across a barrier");
    }

    /// `+`, `-`, `+`, `-` is not one operation: the root is a BUILD of four scalars with no vector
    /// or load node under it, which is not worth packing.
    #[test]
    fn mixed_opcodes_are_not_packed() {
        let src = "fn k(@noalias mut o: [f32; 4], a: [f32; 4], b: [f32; 4]) { \
                   o[0] = a[0] + b[0]; o[1] = a[1] - b[1]; \
                   o[2] = a[2] + b[2]; o[3] = a[3] - b[3]; } \
                   fn main() -> i32 { return 0; }";
        assert!(!has_vectors(src, "k"), "mixed opcodes packed");
    }

    /// A stored value that is also returned has two uses: it must stay a scalar, so its lane
    /// cannot be part of a VECTOR node, and here that leaves the root a BUILD — nothing packs.
    #[test]
    fn a_multi_use_lane_is_not_a_vector_node() {
        let src = "fn k(@noalias mut o: [f32; 4], a: [f32; 4], s: f32) -> f32 { \
                   let t: f32 = a[0] * s; o[0] = t; o[1] = a[1] * s; o[2] = a[2] * s; \
                   o[3] = a[3] * s; return t; } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !has_vectors(src, "k"),
            "a multi-use lane became a vector node"
        );
    }

    /// `-O0` (no SLP) and `-O2` (packed) must print the same bytes. `with_vectors` asserts that
    /// something is vector-typed at `-O2`, so a positive case cannot pass by packing nothing.
    fn scalar_and_slp_agree(src: &str, with_vectors: bool) {
        let mut outs = Vec::new();
        for level in [0u8, 2u8] {
            let (program, mut interner) = optimized(src, level);
            if level == 2 && with_vectors {
                assert!(
                    program
                        .funcs
                        .iter()
                        .any(|f| f.value_types.iter().any(|t| t.is_vector())),
                    "vacuous differential: nothing is vector-typed at -O2"
                );
            }
            let main = interner.intern("main");
            let (code, bytes) = wukong_interp::run_with_output(&program, main, &interner)
                .unwrap_or_else(|e| panic!("-O{level}: {e}"));
            outs.push((code, String::from_utf8_lossy(&bytes).to_string()));
        }
        assert_eq!(outs[0], outs[1], "-O0 and -O2 disagree");
    }

    /// Every packed shape, run: an f32 fma, an i32 add/sub/mul tree, an f64 pair, a copy and the
    /// unrolled step-2 loop, each out of line (the kernels call each other, so nothing inlines and
    /// the arrays stay in memory) and every element printed through a variable index.
    #[test]
    fn slp_output_matches_the_scalar_program() {
        let k = "fn fm(@noalias mut o: [f32; 4], a: [f32; 4], b: [f32; 4], s: f32, d: i64) { \
                   if d > 0 { again(d - 1); return; } \
                   o[0] = a[0] * s + b[0]; o[1] = a[1] * s + b[1]; \
                   o[2] = a[2] * s + b[2]; o[3] = a[3] * s + b[3]; } \
                 fn im(@noalias mut o: [i32; 4], a: [i32; 4], b: [i32; 4], d: i64) { \
                   if d > 0 { again(d - 1); return; } \
                   o[0] = (a[0] - b[0]) * a[0]; o[1] = (a[1] - b[1]) * a[1]; \
                   o[2] = (a[2] - b[2]) * a[2]; o[3] = (a[3] - b[3]) * a[3]; } \
                 fn dm(@noalias mut o: [f64; 2], a: [f64; 2], s: f64, d: i64) { \
                   if d > 0 { again(d - 1); return; } o[0] = a[0] * s; o[1] = a[1] * s; } \
                 fn cp(@noalias mut o: [f32; 8], a: [f32; 8], d: i64) { \
                   if d > 0 { again(d - 1); return; } \
                   for k in 0..8 step 2 { o[k] = a[k]; o[k + 1] = a[k + 1]; } } \
                 fn again(d: i64) { let mut f4: [f32; 4] = [0.0; 4]; let g4: [f32; 4] = [0.0; 4]; \
                   let mut i4: [i32; 4] = [0; 4]; let j4: [i32; 4] = [0; 4]; \
                   let mut d2: [f64; 2] = [0.0; 2]; let e2: [f64; 2] = [0.0; 2]; \
                   let mut f8: [f32; 8] = [0.0; 8]; let g8: [f32; 8] = [0.0; 8]; \
                   fm(f4, g4, g4, 1.0, d); im(i4, j4, j4, d); dm(d2, e2, 1.0, d); cp(f8, g8, d); }";
        let src = format!(
            "{k} fn main() -> i32 {{ \
             let mut fo: [f32; 4] = [0.0; 4]; let mut fa: [f32; 4] = [0.0; 4]; \
             let mut fb: [f32; 4] = [0.0; 4]; let mut io: [i32; 4] = [0; 4]; \
             let mut ia: [i32; 4] = [0; 4]; let mut ib: [i32; 4] = [0; 4]; \
             let mut dout: [f64; 2] = [0.0; 2]; let mut da: [f64; 2] = [0.0; 2]; \
             let mut co: [f32; 8] = [0.0; 8]; let mut ca: [f32; 8] = [0.0; 8]; \
             let mut j: i64 = 0; \
             while j < 4 {{ fa[j] = (j as f32) * 0.5 - 1.0; fb[j] = 3.0 - (j as f32); \
               ia[j] = (j as i32) * 7 - 9; ib[j] = 5 - (j as i32) * 3; j = j + 1; }} \
             j = 0; while j < 2 {{ da[j] = (j as f64) * 0.25 + 1.5; j = j + 1; }} \
             j = 0; while j < 8 {{ ca[j] = (j as f32) * 1.25 - 2.0; j = j + 1; }} \
             fm(fo, fa, fb, 2.0, 0); im(io, ia, ib, 0); dm(dout, da, -4.0, 0); cp(co, ca, 0); \
             j = 0; while j < 4 {{ print(fo[j]); print(io[j]); j = j + 1; }} \
             j = 0; while j < 2 {{ print(dout[j]); j = j + 1; }} \
             j = 0; while j < 8 {{ print(co[j]); j = j + 1; }} \
             return 0; }}"
        );
        scalar_and_slp_agree(&src, true);
    }
}
