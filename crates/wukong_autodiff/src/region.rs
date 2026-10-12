//! Reverse mode through `@parallel` loop regions (N87).
//!
//! `wukong_mir_build` outlines a provably independent loop of a `@parallel` function into its own
//! function `P(start, end, env)` and replaces the loop with an env pack plus one
//! `wukong_parallel_for(n, &P, env)` call (`try_emit_parallel_region`; a whole-function `@parallel`
//! helper spliced in by [`crate::inline_calls`] has the same shape). The forward is unchanged; this
//! module gives that call a VJP.
//!
//! # The backward of one region
//!
//! * **Per-iteration recompute.** `P`'s loop body becomes a single-block function `F(hh, env)` —
//!   one iteration — which the ordinary single-block transform differentiates in a **region mode**
//!   ([`RegionCtx`]): the captured buffers (env slots) are buffer bases whose gradient pointers
//!   arrive through a third parameter `gc`, every contribution to them ACCUMULATES (later forward
//!   ops already put adjoints there), and the replay never writes a captured buffer again (the
//!   forward left exactly those values behind). The result is `{loss}_grad$par<i>$it(hh, env, gc)`.
//! * **The same partition, deterministically.** `{loss}_grad$par<i>(cs, ce, benv)` is the parallel
//!   driver: the loss's backward hands it to `wukong_parallel_for(K, ..)` over `K` virtual chunks,
//!   chunk `c` running its share of `lo + [0, N)` in order (`N / K` iterations, one more for the
//!   first `N mod K` chunks). Which iterations form a chunk is a function of `N` alone — never of
//!   the thread count — so every float sum below is taken in one fixed order on every backend and
//!   every pool width.
//! * **Privatized shared adjoints.** A buffer written by the region is written in a per-iteration
//!   disjoint slice (the outliner's proof), and so is its adjoint. A buffer only READ may be read by
//!   many iterations (a broadcast weight, a captured scalar): unless the per-iteration index proof
//!   below places every read of it in the iteration's own band, it gets `K` private zeroed copies of
//!   its adjoint (`K = min(N, PRIVATE_CHUNKS)`), chunk `c` accumulating into copy `c`, which the
//!   loss's backward then adds into the real adjoint in chunk order `0, 1, …, K-1`. No float adjoint
//!   is ever accumulated with an atomic: the bits are the same across thread counts and between the
//!   interpreter (which runs the chunks in order) and the native pool.
//!
//! # What is supported, stated exactly
//!
//! The outlined body at `-O1` (the driver re-optimizes outlined bodies at `-O1` for a gradient
//! compile, so `-O2`'s unrolling never reaches here) must be the canonical counted loop: an entry
//! block of pure work (the env loads, the grain mapping, the body-local `alloca`s), a header
//! `cmp.slt|ult hh, bound`, ONE straight-line body block and a `ret`. An iteration with an inner
//! loop, a run-time bounds check or an overlap guard has control flow in it and declines by name.
//! Every other refusal is loud too: a captured buffer the region both reads and writes (an in-place
//! update: the recompute would read the forward's result), a captured buffer something after the
//! region writes again, a captured slice whose elements need a gradient, a shared buffer of no known
//! size, and second-order differentiation.

use crate::{map_operands, Mode, Vjp};
use std::collections::{HashMap, HashSet};
use wukong_mir::{
    BinOp, Builder, CastKind, CmpOp, Function, Inst, MirType, Op, Program, Terminator, ValueId,
};
use wukong_span::{Interner, Symbol};

/// The most private adjoint copies one shared buffer gets (`K = min(N, PRIVATE_CHUNKS)`), and so
/// the widest the backward of a region with a shared read runs. A constant on purpose: `K` decides
/// the order of every privatized float sum, so it may depend on the trip count and nothing else.
pub(crate) const PRIVATE_CHUNKS: i64 = 16;

const PTR: MirType = MirType::Ptr;
const I64: MirType = MirType::I64;

/// One `wukong_parallel_for` call of the forward, with its outlined body taken apart.
pub(crate) struct RegionCall<'a> {
    /// Index of the call in the forward block.
    pub(crate) call: usize,
    /// The call's chunk count (forward value).
    pub(crate) n: ValueId,
    /// The env pointer table (forward value, an `alloca [k x ptr]`).
    pub(crate) env: ValueId,
    /// The forward value stored in each env slot (`None`: a slot nothing stores).
    pub(crate) slots: Vec<Option<ValueId>>,
    pub(crate) body: IterBody<'a>,
    /// `{grad}$par<i>`: the chunk driver handed to `wukong_parallel_for`.
    pub(crate) bwd_name: Symbol,
    /// `{grad}$par<i>$it`: one iteration's recompute + reverse.
    pub(crate) it_name: Symbol,
}

/// The outlined body `P`, split into its iteration range and one iteration.
pub(crate) struct IterBody<'a> {
    pub(crate) p: &'a Function,
    /// `F(hh, env) -> void`: the loop body as a single-block function of the iteration index.
    pub(crate) f: Function,
    /// The loop counter's type (the header parameter's).
    pub(crate) hty: MirType,
    /// The header compares unsigned (`cmp.ult`).
    pub(crate) unsigned: bool,
    /// `P`'s instructions computing the iteration range from `(start, end, env)`, in order.
    pub(crate) range: Vec<&'a Inst>,
    /// The first iteration (the header's entry argument) and the bound, as `P` values.
    pub(crate) init: ValueId,
    pub(crate) bound: ValueId,
    /// `F` values that are an env slot's pointer (`load ptr env[k]`) -> `k`.
    pub(crate) cap_of: HashMap<ValueId, usize>,
    /// Every `F` pointer derived from an env slot's pointer (the slot pointers included) -> slot.
    pub(crate) root: HashMap<ValueId, usize>,
    /// How the iteration uses each env slot.
    pub(crate) slots: Vec<SlotUse>,
    /// `F`'s iteration-index parameter.
    pub(crate) hh: ValueId,
}

/// How one iteration uses one env slot's buffer.
#[derive(Clone, Default, Debug)]
pub(crate) struct SlotUse {
    /// Some load, kernel input or other read reaches it.
    pub(crate) read: bool,
    /// A store, a kernel output or an atomic add reaches it.
    pub(crate) write: bool,
    /// A kernel (or another call) writes it: that rule reads the forward's output back.
    pub(crate) call_write: bool,
    /// A float load or a kernel buffer operand reads it (a gradient may flow into it).
    pub(crate) float_read: bool,
    /// It is a kernel operand (the whole-buffer form the index proof cannot place).
    pub(crate) kernel: bool,
    /// Some other use the index proof cannot place (a non-kernel call, a pointer comparison).
    pub(crate) other: bool,
    /// Its elements are reached through a slice's data pointer.
    pub(crate) via_slice: bool,
    /// The element type of its float loads, if they agree.
    pub(crate) elem: Option<MirType>,
    pub(crate) mixed_elem: bool,
    /// The pointers of its float loads (for the per-iteration index proof).
    pub(crate) loads: Vec<ValueId>,
}

/// What the gradient of one env slot is, inside an iteration's backward.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CapGrad {
    /// Not a buffer the iteration reads as float data or writes (an integer slot, an unused one).
    Unused,
    /// Written by the region, and its adjoint is live in the loss: the iteration reads it as the
    /// output adjoint and zeroes the slice it overwrote.
    Live,
    /// Written by the region, but nothing downstream reads it with a gradient.
    Dead,
    /// Read by the region, and the loss keeps (or can make) a gradient buffer for it.
    HasGrad,
    /// Read by the region, but it carries no gradient (an input outside `wrt`, e.g. labels).
    NoGrad,
    /// Read as float data, but its storage has no gradient model (a slice, an integer or `f64`
    /// local array, a derived pointer): routing a gradient into it is an error.
    Unknown,
}

/// The region mode of a [`Vjp`] (see the module docs): the env slots of one iteration.
pub(crate) struct RegionCtx {
    pub(crate) cap_of: HashMap<ValueId, usize>,
    pub(crate) root: HashMap<ValueId, usize>,
    pub(crate) kinds: Vec<CapGrad>,
    /// The `gc` parameter: one gradient (or private-copy) pointer per env slot.
    pub(crate) gc: ValueId,
    /// The pointer loaded from `gc[k]` (new value space), for every slot with a gradient.
    pub(crate) gptr: Vec<Option<ValueId>>,
    /// Slots whose gradient the iteration actually accumulated into.
    pub(crate) needed: Vec<bool>,
    /// The `@parallel` region's name, for messages.
    pub(crate) name: String,
}

impl RegionCtx {
    /// The env slot `v` (an `F` pointer) derives from.
    pub(crate) fn slot_of(&self, v: ValueId) -> Option<usize> {
        self.root.get(&v).copied()
    }
}

/// The indices of every `wukong_parallel_for` call in the forward block, in order.
pub(crate) fn locate(fwd: &Function, pfor: Symbol) -> Vec<usize> {
    fwd.blocks[0]
        .insts
        .iter()
        .enumerate()
        .filter(|(_, inst)| matches!(&inst.op, Op::Call { func, .. } if *func == pfor))
        .map(|(i, _)| i)
        .collect()
}

/// Take every region call of `fwd` apart: its env slots, its outlined body (from `program`), and
/// the loss-level legality of recomputing it (nothing after the call may write what it captured).
pub(crate) fn scan<'a>(
    fwd: &Function,
    program: &'a Program,
    calls: &[usize],
    names: &[(Symbol, Symbol)],
    it: &Interner,
) -> Result<Vec<RegionCall<'a>>, String> {
    let block = &fwd.blocks[0];
    let def: HashMap<ValueId, &Op> = block
        .insts
        .iter()
        .filter_map(|i| Some((i.result?, &i.op)))
        .collect();
    let mut out = Vec::with_capacity(calls.len());
    for (&ci, &(bwd_name, it_name)) in calls.iter().zip(names) {
        let Op::Call { args, .. } = &block.insts[ci].op else {
            unreachable!("located a call")
        };
        if args.len() != 3 {
            return Err(format!(
                "autodiff: no VJP rule for a `wukong_parallel_for` call with {} argument(s) \
                 (the runtime ABI is `(n, body, env)`)",
                args.len()
            ));
        }
        let Some(Op::FuncAddr(body_sym)) = def.get(&args[1]).copied() else {
            return Err(
                "autodiff: no VJP rule for a `wukong_parallel_for` call whose body is not a \
                 function address"
                    .into(),
            );
        };
        let body_name = it.resolve(*body_sym).to_string();
        let p = program.function(*body_sym).ok_or_else(|| {
            format!(
                "autodiff: no VJP rule for the @parallel region `{body_name}`: its outlined body \
                 is not in the program handed to the gradient (differentiate the loss through \
                 `wukong_autodiff::grad_program` with the program that defines it)"
            )
        })?;
        let env = args[2];
        let nslots = match def.get(&env).copied() {
            Some(Op::Alloca {
                ty: MirType::Array(elem, k),
                ..
            }) if **elem == PTR => *k as usize,
            _ => {
                return Err(format!(
                    "autodiff: no VJP rule for the @parallel region `{body_name}`: its env is not \
                     a local pointer table"
                ))
            }
        };
        let mut slots: Vec<Option<ValueId>> = vec![None; nslots];
        for inst in &block.insts[..ci] {
            let Op::Store { ptr, value } = &inst.op else {
                continue;
            };
            let k = if *ptr == env {
                Some(0)
            } else {
                match def.get(ptr).copied() {
                    Some(Op::Gep {
                        ptr: base,
                        index,
                        elem: MirType::Ptr,
                    }) if *base == env => match def.get(index).copied() {
                        Some(Op::ConstInt(k, _)) => Some(*k as usize),
                        _ => {
                            return Err(format!(
                                "autodiff: the env of the @parallel region `{body_name}` is \
                                 written at a non-constant slot"
                            ))
                        }
                    },
                    _ => None,
                }
            };
            let Some(k) = k else {
                continue;
            };
            if k >= nslots || slots[k].replace(*value).is_some() {
                return Err(format!(
                    "autodiff: env slot {k} of the @parallel region `{body_name}` is not stored \
                     exactly once"
                ));
            }
        }
        let body = extract(p, nslots, it)?;
        out.push(RegionCall {
            call: ci,
            n: args[0],
            env,
            slots,
            body,
            bwd_name,
            it_name,
        });
    }
    check_no_later_writes(fwd, &out, it)?;
    Ok(out)
}

/// The base buffer a forward pointer addresses (every `gep` peeled).
fn base_of(def: &HashMap<ValueId, &Op>, mut v: ValueId) -> ValueId {
    while let Some(Op::Gep { ptr, .. }) = def.get(&v) {
        v = *ptr;
    }
    v
}

/// The iteration's backward recomputes it from the buffers the forward left behind, so every buffer
/// a region READS must hold, at the end of the forward, what it held when the region ran: nothing
/// after the call may store into it, write it as a kernel output, atomically update it, or write it
/// from a later region. So must a buffer a kernel inside the region writes, whose rule reads that
/// output back. (A buffer the region only stores into is free to change afterwards: the iteration's
/// backward takes the stored values from its own recompute, never from memory.)
fn check_no_later_writes(
    fwd: &Function,
    regions: &[RegionCall<'_>],
    it: &Interner,
) -> Result<(), String> {
    let block = &fwd.blocks[0];
    let def: HashMap<ValueId, &Op> = block
        .insts
        .iter()
        .filter_map(|i| Some((i.result?, &i.op)))
        .collect();
    for r in regions {
        let captured: HashSet<ValueId> = r
            .slots
            .iter()
            .enumerate()
            .filter(|(k, _)| r.body.slots.get(*k).is_some_and(|u| u.read || u.call_write))
            .filter_map(|(_, v)| v.map(|v| base_of(&def, v)))
            .collect();
        let name = it.resolve(r.body.p.name);
        for (i, inst) in block.insts.iter().enumerate().skip(r.call + 1) {
            let written: Vec<ValueId> = match &inst.op {
                Op::Store { ptr, .. } | Op::AtomicAdd { ptr, .. } => vec![base_of(&def, *ptr)],
                Op::Call { func, args } => {
                    if let Some(later) = regions.iter().find(|q| q.call == i) {
                        later
                            .slots
                            .iter()
                            .enumerate()
                            .filter(|(k, _)| later.body.slots.get(*k).is_some_and(|s| s.write))
                            .filter_map(|(_, v)| v.map(|v| base_of(&def, v)))
                            .collect()
                    } else {
                        match crate::tape::kernel_by_name(it.resolve(*func)) {
                            Some(kind) if kind.arity() == args.len() => kind
                                .out_index()
                                .map(|oi| vec![base_of(&def, args[oi])])
                                .unwrap_or_default(),
                            _ => Vec::new(),
                        }
                    }
                }
                _ => Vec::new(),
            };
            if let Some(b) = written.iter().find(|b| captured.contains(b)) {
                return Err(format!(
                    "autodiff: buffer v{} is captured by the @parallel region `{name}` and \
                     written again after it; the region's backward recomputes each iteration \
                     from the buffers the forward leaves behind, so a captured buffer must not \
                     change after the region (write the later value into a separate buffer)",
                    b.0
                ));
            }
        }
    }
    Ok(())
}

/// Split the outlined body `p` into its iteration range and one iteration (see [`IterBody`]).
fn extract<'a>(p: &'a Function, nslots: usize, it: &Interner) -> Result<IterBody<'a>, String> {
    let name = it.resolve(p.name).to_string();
    let shape = |why: &str| {
        format!(
            "autodiff: the @parallel region `{name}` cannot be differentiated: its iteration is \
             not one straight-line block ({why}); the region backward recomputes one iteration \
             at a time, which needs an iteration of scalar arithmetic, loads and stores of \
             captured arrays and recognized kernel calls — an inner loop, a run-time bounds \
             check (a slice index) or an overlap guard puts control flow in it"
        )
    };
    if p.params.len() != 3 {
        return Err(shape("it is not a `(start, end, env)` body"));
    }
    if p.blocks.len() != 4 {
        return Err(shape(&format!("{} blocks", p.blocks.len())));
    }
    let (start, end, env) = (p.params[0], p.params[1], p.params[2]);
    let blk = |id| p.blocks.iter().find(|b| b.id == id);
    let entry = blk(p.entry).ok_or_else(|| shape("no entry block"))?;
    let Terminator::Br {
        target: h_id,
        args: eargs,
    } = &entry.term
    else {
        return Err(shape("the entry does not branch to a loop header"));
    };
    let [init] = eargs[..] else {
        return Err(shape("the loop header does not take one counter"));
    };
    let header = blk(*h_id).ok_or_else(|| shape("no loop header"))?;
    let [hh] = header.params[..] else {
        return Err(shape("the loop header does not take one counter"));
    };
    let Terminator::CondBr {
        cond,
        then_blk,
        then_args,
        else_blk,
        else_args,
    } = &header.term
    else {
        return Err(shape("the loop header does not test the counter"));
    };
    if !then_args.is_empty() || !else_args.is_empty() {
        return Err(shape("the loop exits carry values"));
    }
    let body = blk(*then_blk).ok_or_else(|| shape("no loop body"))?;
    let exit = blk(*else_blk).ok_or_else(|| shape("no loop exit"))?;
    let ids = [entry.id, header.id, body.id, exit.id];
    if (1..4).any(|i| ids[..i].contains(&ids[i])) {
        return Err(shape("the loop blocks are not four distinct blocks"));
    }
    if !exit.params.is_empty()
        || !exit.insts.is_empty()
        || !matches!(exit.term, Terminator::Ret(None))
    {
        return Err(shape("the loop exit does more than return"));
    }
    if !body.params.is_empty() {
        return Err(shape("the loop body takes block parameters"));
    }
    let Terminator::Br {
        target: back,
        args: bargs,
    } = &body.term
    else {
        return Err(shape("the loop body does not branch back"));
    };
    let [next] = bargs[..] else {
        return Err(shape("the back edge does not carry one counter"));
    };
    if back != h_id {
        return Err(shape("the loop body does not branch back to the header"));
    }
    let pdef: HashMap<ValueId, &Op> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| Some((i.result?, &i.op)))
        .collect();
    let is_one = |v: ValueId| matches!(pdef.get(&v), Some(Op::ConstInt(1, _)));
    match pdef.get(&next) {
        Some(Op::Bin(BinOp::Add, a, b)) if (*a == hh && is_one(*b)) || (*b == hh && is_one(*a)) => {
        }
        _ => return Err(shape("the counter does not step by one")),
    }
    let (unsigned, bound) = match pdef.get(cond) {
        Some(Op::Cmp(CmpOp::Slt, a, b)) if *a == hh => (false, *b),
        Some(Op::Cmp(CmpOp::Ult, a, b)) if *a == hh => (true, *b),
        _ => return Err(shape("the header is not `counter < bound`")),
    };
    let pure = |op: &Op| {
        !matches!(
            op,
            Op::Store { .. }
                | Op::AtomicAdd { .. }
                | Op::Call { .. }
                | Op::Barrier
                | Op::VecKernelCall { .. }
        )
    };
    if !entry.insts.iter().all(|i| pure(&i.op)) {
        return Err(shape("the entry block has side effects"));
    }
    if !header
        .insts
        .iter()
        .all(|i| pure(&i.op) && !matches!(i.op, Op::Load(..) | Op::Alloca { .. }))
    {
        return Err(shape("the loop header reads memory"));
    }
    if body
        .insts
        .iter()
        .any(|i| matches!(i.op, Op::VecKernelCall { .. } | Op::Barrier))
    {
        return Err(shape("the iteration calls a synthesized vector kernel"));
    }

    // Values defined outside the body (entry and header), by id.
    let outer: HashMap<ValueId, &Inst> = entry
        .insts
        .iter()
        .chain(&header.insts)
        .filter_map(|i| Some((i.result?, i)))
        .collect();
    // The outer values the body needs, transitively. Reading the chunk bounds would make an
    // iteration depend on how the runtime split the range.
    let mut need: HashSet<ValueId> = HashSet::new();
    let mut stack: Vec<ValueId> = Vec::new();
    for inst in &body.insts {
        let _ = map_operands(&inst.op, |v| {
            stack.push(v);
            v
        });
    }
    while let Some(v) = stack.pop() {
        if v == start || v == end {
            return Err(shape("the iteration reads its chunk bounds"));
        }
        if let Some(inst) = outer.get(&v) {
            if need.insert(v) {
                let _ = map_operands(&inst.op, |u| {
                    stack.push(u);
                    u
                });
            }
        }
    }
    // The range: the outer values `init` and `bound` need (the chunk bounds included).
    let mut rneed: HashSet<ValueId> = HashSet::new();
    let mut stack = vec![init, bound];
    while let Some(v) = stack.pop() {
        if let Some(inst) = outer.get(&v) {
            if rneed.insert(v) {
                let _ = map_operands(&inst.op, |u| {
                    stack.push(u);
                    u
                });
            }
        }
    }
    if outer
        .iter()
        .any(|(v, i)| rneed.contains(v) && matches!(i.op, Op::Alloca { .. }))
    {
        return Err(shape("the iteration range reads a local"));
    }
    let range: Vec<&Inst> = entry
        .insts
        .iter()
        .chain(&header.insts)
        .filter(|i| i.result.is_some_and(|r| rneed.contains(&r)))
        .collect();

    // F(hh, env): the needed outer values in order, then the body, then `ret`.
    let hty = p.value_type(hh).clone();
    let mut fb = Builder::new(p.name, MirType::Void);
    let nhh = fb.add_param(hty.clone());
    let nenv = fb.add_param(PTR);
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    map.insert(hh, nhh);
    map.insert(env, nenv);
    let outer_needed = entry
        .insts
        .iter()
        .chain(&header.insts)
        .filter(|i| i.result.is_some_and(|r| need.contains(&r)));
    for inst in outer_needed.chain(&body.insts) {
        let op = map_operands(&inst.op, |v| {
            *map.get(&v)
                .expect("an operand of the iteration is defined before it")
        });
        match inst.result {
            Some(r) => {
                let nr = fb.build(p.value_type(r).clone(), op);
                map.insert(r, nr);
            }
            None => fb.build_void(op),
        }
    }
    fb.ret(None);
    let f = fb.finish();
    let (cap_of, root, slots) = slot_uses(&f, nenv, nslots, it)?;
    for (k, s) in slots.iter().enumerate() {
        if s.read && s.write {
            return Err(format!(
                "autodiff: the @parallel region `{name}` both reads and writes captured buffer \
                 #{k} (an in-place update); its backward recomputes each iteration from the \
                 buffers the forward left behind, so a region's captured inputs and outputs must \
                 be distinct buffers (write the result into a separate array)"
            ));
        }
    }
    Ok(IterBody {
        p,
        f,
        hty,
        unsigned,
        range,
        init,
        bound,
        cap_of,
        root,
        slots,
        hh: nhh,
    })
}

type SlotMaps = (
    HashMap<ValueId, usize>,
    HashMap<ValueId, usize>,
    Vec<SlotUse>,
);

/// Classify how the single-block iteration `f` uses each env slot.
fn slot_uses(f: &Function, env: ValueId, nslots: usize, it: &Interner) -> Result<SlotMaps, String> {
    let insts = &f.blocks[0].insts;
    let def: HashMap<ValueId, &Op> = insts
        .iter()
        .filter_map(|i| Some((i.result?, &i.op)))
        .collect();
    let mut cap_of: HashMap<ValueId, usize> = HashMap::new();
    let mut root: HashMap<ValueId, usize> = HashMap::new();
    let mut slots = vec![SlotUse::default(); nslots];
    let slot_of_addr = |addr: ValueId| -> Option<usize> {
        if addr == env {
            return Some(0);
        }
        match def.get(&addr) {
            Some(Op::Gep {
                ptr,
                index,
                elem: MirType::Ptr,
            }) if *ptr == env => match def.get(index) {
                Some(Op::ConstInt(k, _)) => Some(*k as usize),
                _ => None,
            },
            _ => None,
        }
    };
    let name = it.resolve(f.name);
    for inst in insts {
        let r = inst.result;
        match &inst.op {
            Op::Load(addr, MirType::Ptr) => {
                if let Some(k) = slot_of_addr(*addr) {
                    if k >= nslots {
                        return Err(format!(
                            "autodiff: the @parallel region `{name}` reads env slot {k}, which \
                             its call never fills"
                        ));
                    }
                    let r = r.expect("a load has a result");
                    cap_of.insert(r, k);
                    root.insert(r, k);
                } else if let Some(&k) = root.get(addr) {
                    // A pointer loaded out of a captured buffer: a slice's data pointer.
                    slots[k].read = true;
                    slots[k].via_slice = true;
                    root.insert(r.expect("a load has a result"), k);
                }
            }
            Op::Gep { ptr, .. } => {
                if let Some(&k) = root.get(ptr) {
                    root.insert(r.expect("a gep has a result"), k);
                }
            }
            Op::Load(addr, ty) => {
                if let Some(&k) = root.get(addr) {
                    let s = &mut slots[k];
                    s.read = true;
                    if ty.is_float() {
                        s.float_read = true;
                        s.loads.push(*addr);
                        match &s.elem {
                            None => s.elem = Some(ty.clone()),
                            Some(e) if e != ty => s.mixed_elem = true,
                            _ => {}
                        }
                    }
                }
            }
            Op::Store { ptr, value } => {
                if let Some(&k) = root.get(value) {
                    return Err(format!(
                        "autodiff: the @parallel region `{name}` stores a pointer into captured \
                         buffer #{k}; a pointer that escapes the iteration cannot be followed by \
                         the region backward"
                    ));
                }
                if let Some(&k) = root.get(ptr) {
                    slots[k].write = true;
                }
            }
            Op::AtomicAdd { ptr, .. } => {
                if let Some(&k) = root.get(ptr) {
                    slots[k].write = true;
                }
            }
            Op::Call { func, args } => {
                let callee = it.resolve(*func);
                let kernel =
                    crate::tape::kernel_by_name(callee).filter(|kind| kind.arity() == args.len());
                for (i, a) in args.iter().enumerate() {
                    let Some(&k) = root.get(a) else {
                        continue;
                    };
                    let s = &mut slots[k];
                    match kernel {
                        Some(kind) if kind.out_index() == Some(i) => {
                            s.write = true;
                            s.call_write = true;
                        }
                        Some(_) => {
                            s.read = true;
                            s.float_read = true;
                            s.kernel = true;
                        }
                        // `print`, `assert`, a file write: they read the buffer.
                        None if crate::coverage::is_effect_only(callee) => {
                            s.read = true;
                            s.other = true;
                        }
                        // Any other call may write what it is handed.
                        None => {
                            s.write = true;
                            s.call_write = true;
                            s.other = true;
                        }
                    }
                }
            }
            other => {
                let _ = map_operands(other, |v| {
                    if let Some(&k) = root.get(&v) {
                        slots[k].other = true;
                    }
                    v
                });
            }
        }
    }
    Ok((cap_of, root, slots))
}

impl IterBody<'_> {
    fn def(&self) -> HashMap<ValueId, &Op> {
        self.f.blocks[0]
            .insts
            .iter()
            .filter_map(|i| Some((i.result?, &i.op)))
            .collect()
    }

    /// `v = a·hh + c` over the integers, through casts and constant arithmetic; `None` otherwise.
    fn affine(&self, def: &HashMap<ValueId, &Op>, v: ValueId, depth: u32) -> Option<(i128, i128)> {
        if v == self.hh {
            return Some((1, 0));
        }
        if depth > 64 {
            return None;
        }
        let d = depth + 1;
        match def.get(&v)? {
            Op::ConstInt(c, _) => Some((0, *c)),
            Op::Cast(CastKind::SExt | CastKind::ZExt | CastKind::Trunc, a, _) => {
                self.affine(def, *a, d)
            }
            Op::Bin(BinOp::Add, a, b) => {
                let (x, y) = (self.affine(def, *a, d)?, self.affine(def, *b, d)?);
                Some((x.0.checked_add(y.0)?, x.1.checked_add(y.1)?))
            }
            Op::Bin(BinOp::Sub, a, b) => {
                let (x, y) = (self.affine(def, *a, d)?, self.affine(def, *b, d)?);
                Some((x.0.checked_sub(y.0)?, x.1.checked_sub(y.1)?))
            }
            Op::Bin(BinOp::Mul, a, b) => {
                let (x, y) = (self.affine(def, *a, d)?, self.affine(def, *b, d)?);
                match (x, y) {
                    ((0, k), (a, c)) | ((a, c), (0, k)) => {
                        Some((a.checked_mul(k)?, c.checked_mul(k)?))
                    }
                    _ => None,
                }
            }
            Op::Bin(BinOp::Shl, a, b) => {
                let x = self.affine(def, *a, d)?;
                let (0, s) = self.affine(def, *b, d)? else {
                    return None;
                };
                if !(0..63).contains(&s) {
                    return None;
                }
                let k = 1i128 << s;
                Some((x.0.checked_mul(k)?, x.1.checked_mul(k)?))
            }
            _ => None,
        }
    }

    /// The element index a load pointer addresses in its slot's buffer, as `a·hh + c`: the sum of
    /// its `gep` chain's indices, every link over the loaded element type.
    fn load_index(
        &self,
        def: &HashMap<ValueId, &Op>,
        mut p: ValueId,
        ty: &MirType,
    ) -> Option<(i128, i128)> {
        let mut acc = (0i128, 0i128);
        while !self.cap_of.contains_key(&p) {
            let Some(Op::Gep { ptr, index, elem }) = def.get(&p) else {
                return None;
            };
            if elem != ty {
                return None;
            }
            let (a, c) = self.affine(def, *index, 0)?;
            acc = (acc.0.checked_add(a)?, acc.1.checked_add(c)?);
            p = *ptr;
        }
        Some(acc)
    }

    /// Whether every iteration reads its own elements of the buffer behind `slots`: each read is a
    /// float load at `C·hh + c` with one stride `C > 0` and every `c` in one band `[k·C, k·C + C)`.
    /// Then iteration `hh` touches only `[(hh+k)·C, (hh+k)·C + C)`, two iterations never share an
    /// element, and their adjoint contributions need no private copy.
    pub(crate) fn partitioned(&self, slots: &[usize]) -> bool {
        let def = self.def();
        let mut stride: Option<i128> = None;
        let mut band: Option<i128> = None;
        for &k in slots {
            let s = &self.slots[k];
            if s.kernel || s.other || s.via_slice || s.mixed_elem {
                return false;
            }
            let Some(ty) = &s.elem else {
                continue;
            };
            for &p in &s.loads {
                let Some((a, c)) = self.load_index(&def, p, ty) else {
                    return false;
                };
                if a <= 0 || stride.is_some_and(|st| st != a) {
                    return false;
                }
                stride = Some(a);
                let b = c.div_euclid(a);
                if band.is_some_and(|x| x != b) {
                    return false;
                }
                band = Some(b);
            }
        }
        stride.is_some()
    }
}

/// The chunk driver `name(cs, ce, benv) -> void` of one region's backward. `benv` is
/// `[env, gtab, ints]`: the forward env, one gradient (or private-copy base) pointer per env slot,
/// and `[lo, N, K, size_0, size_1, …]`. For each virtual chunk `c` in `[cs, ce)` it fills `gc` —
/// `gtab[k]`, or `gtab[k] + c·size_k` for a privatized slot `k` (element type `private[k]`) — and
/// runs its iterations in order through `it(hh, env, gc)`: `N / K` of them, one more for the
/// first `N mod K` chunks.
pub(crate) fn build_bwd(
    name: Symbol,
    it_name: Symbol,
    private: &[Option<MirType>],
    hty: &MirType,
) -> Function {
    let nslots = private.len();
    let mut b = Builder::new(name, MirType::Void);
    let cs = b.add_param(I64);
    let ce = b.add_param(I64);
    let benv = b.add_param(PTR);
    let cint = |b: &mut Builder, v: i64| b.build(I64, Op::ConstInt(v as i128, I64));
    let slot = |b: &mut Builder, base: ValueId, k: i64, elem: MirType| {
        let kv = b.build(I64, Op::ConstInt(k as i128, I64));
        b.build(
            PTR,
            Op::Gep {
                ptr: base,
                index: kv,
                elem,
            },
        )
    };
    let env = b.build(PTR, Op::Load(benv, PTR));
    let gp = slot(&mut b, benv, 1, PTR);
    let gtab = b.build(PTR, Op::Load(gp, PTR));
    let ip = slot(&mut b, benv, 2, PTR);
    let ints = b.build(PTR, Op::Load(ip, PTR));
    let lo = b.build(I64, Op::Load(ints, I64));
    let np = slot(&mut b, ints, 1, I64);
    let n = b.build(I64, Op::Load(np, I64));
    let kp = slot(&mut b, ints, 2, I64);
    let kk = b.build(I64, Op::Load(kp, I64));
    let gc = b.alloca(MirType::Array(Box::new(PTR), nslots.max(1) as u32));

    let ch = b.new_block();
    let cb = b.new_block();
    let cx = b.new_block();
    let ih = b.new_block();
    let ib = b.new_block();
    let cl = b.new_block();
    let c = b.block_param(ch, I64);
    let h = b.block_param(ih, I64);
    b.br(ch, vec![cs]);

    // ch(c): c < ce ?
    b.switch_to(ch);
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, c, ce));
    b.cond_br(more, cb, vec![], cx, vec![]);

    // cb: fill gc for chunk c, then its iteration range.
    b.switch_to(cb);
    for (k, priv_elem) in private.iter().enumerate() {
        let src = slot(&mut b, gtab, k as i64, PTR);
        let g = b.build(PTR, Op::Load(src, PTR));
        let val = match priv_elem {
            Some(elem) => {
                let szp = slot(&mut b, ints, 3 + k as i64, I64);
                let sz = b.build(I64, Op::Load(szp, I64));
                let off = b.build(I64, Op::Bin(BinOp::Mul, c, sz));
                b.build(
                    PTR,
                    Op::Gep {
                        ptr: g,
                        index: off,
                        elem: elem.clone(),
                    },
                )
            }
            None => g,
        };
        let dst = slot(&mut b, gc, k as i64, PTR);
        b.build_void(Op::Store {
            ptr: dst,
            value: val,
        });
    }
    // Chunk c starts at `c·q + min(c, r)` with `q = N / K`, `r = N % K`: the first `r` chunks take
    // `q + 1` iterations and the rest `q`, and no product exceeds `N`.
    let one = cint(&mut b, 1);
    let q = b.build(I64, Op::Bin(BinOp::SDiv, n, kk));
    let r = b.build(I64, Op::Bin(BinOp::SRem, n, kk));
    let start_of = |b: &mut Builder, c: ValueId| {
        let cq = b.build(I64, Op::Bin(BinOp::Mul, c, q));
        let below = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, c, r));
        let extra = b.build(I64, Op::Select(below, c, r));
        let off = b.build(I64, Op::Bin(BinOp::Add, cq, extra));
        b.build(I64, Op::Bin(BinOp::Add, lo, off))
    };
    let first = start_of(&mut b, c);
    let c1 = b.build(I64, Op::Bin(BinOp::Add, c, one));
    let last = start_of(&mut b, c1);
    b.br(ih, vec![first]);

    // ih(h): h < last ?
    b.switch_to(ih);
    let go = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, h, last));
    b.cond_br(go, ib, vec![], cl, vec![]);

    // ib: one iteration.
    b.switch_to(ib);
    let hh = if *hty == I64 {
        h
    } else {
        b.build(hty.clone(), Op::Cast(CastKind::Trunc, h, hty.clone()))
    };
    b.build_void(Op::Call {
        func: it_name,
        args: vec![hh, env, gc],
    });
    let h1 = b.build(I64, Op::Bin(BinOp::Add, h, one));
    b.br(ih, vec![h1]);

    // cl: next chunk.
    b.switch_to(cl);
    b.br(ch, vec![c1]);

    b.switch_to(cx);
    b.ret(None);
    b.finish()
}

impl<'a> Vjp<'a> {
    /// Region mode: which `F` instructions the recompute skips. A store, a kernel output, an atomic
    /// add or an unrecognized void call into a captured buffer already happened in the forward, and
    /// the buffer still holds its result (`check_no_later_writes`); writing it again would only race
    /// another chunk's reads of the same values. (A region never reads what it writes, so no
    /// recomputed value depends on the skipped write.)
    pub(crate) fn region_skips(&self, op: &Op) -> bool {
        let Some(rc) = &self.region else {
            return false;
        };
        let captured = |v: &ValueId| rc.root.contains_key(v);
        match op {
            Op::Store { ptr, .. } | Op::AtomicAdd { ptr, .. } => captured(ptr),
            Op::Call { func, args } => {
                if let Some((kind, _)) = self.syms.kernel(*func) {
                    kind.arity() == args.len()
                        && kind.out_index().is_some_and(|oi| captured(&args[oi]))
                } else {
                    let name = self.it.resolve(*func);
                    !crate::coverage::is_effect_only(name) && args.iter().any(captured)
                }
            }
            _ => false,
        }
    }

    /// Region mode, after the replay: load every slot's gradient pointer from `gc`, and key the
    /// output adjoint of each written slot (and of each kernel output slice of it) so the rules of
    /// the iteration find it. Every captured key accumulates.
    pub(crate) fn region_setup(&mut self) {
        let Some(rc) = self.region.as_ref() else {
            return;
        };
        let gc = rc.gc;
        let kinds = rc.kinds.clone();
        let mut caps: Vec<(ValueId, usize)> = rc.cap_of.iter().map(|(&v, &k)| (v, k)).collect();
        caps.sort_by_key(|&(v, _)| v.0);
        let mut gptr = vec![None; kinds.len()];
        for (k, kind) in kinds.iter().enumerate() {
            if matches!(kind, CapGrad::Live | CapGrad::HasGrad) {
                let kv = self.cint(k as i64);
                let at = self.b.build(
                    PTR,
                    Op::Gep {
                        ptr: gc,
                        index: kv,
                        elem: PTR,
                    },
                );
                gptr[k] = Some(self.b.build(PTR, Op::Load(at, PTR)));
            }
        }
        if let Some(rc) = self.region.as_mut() {
            rc.gptr = gptr.clone();
        }
        for (v, k) in caps {
            if kinds[k] == CapGrad::Live {
                let g = gptr[k].expect("a live slot has a pointer");
                self.buf_adj.insert(v, g);
                self.contributed.insert(v);
            }
        }
        // Kernel outputs that are a slice of a live written slot.
        let mut outs: Vec<ValueId> = Vec::new();
        for inst in &self.entry_block().insts {
            let Op::Call { func, args } = &inst.op else {
                continue;
            };
            let Some((kind, _)) = self.syms.kernel(*func) else {
                continue;
            };
            if kind.arity() != args.len() {
                continue;
            }
            if let Some(oi) = kind.out_index() {
                outs.push(self.canon(args[oi]));
            }
        }
        for out in outs {
            if self.buf_adj.contains_key(&out) {
                continue;
            }
            let Some(k) = self.region.as_ref().and_then(|rc| rc.slot_of(out)) else {
                continue;
            };
            if kinds[k] != CapGrad::Live {
                continue;
            }
            if let Ok(Some(g)) = self.region_slice_grad(out, gptr[k].expect("live")) {
                self.buf_adj.insert(out, g);
                self.contributed.insert(out);
            }
        }
    }

    /// Region mode: the gradient pointer of `buf`, a captured slot pointer or a `gep` chain over one
    /// — `None` when the slot is not a capture (the ordinary rules apply), `Some(None)` when it
    /// carries no gradient. Records that the iteration used a read slot's gradient.
    pub(crate) fn region_grad_target(
        &mut self,
        buf: ValueId,
    ) -> Option<Result<Option<ValueId>, String>> {
        let rc = self.region.as_ref()?;
        let k = rc.slot_of(buf)?;
        let kind = rc.kinds[k];
        let name = rc.name.clone();
        let g = match kind {
            CapGrad::Live | CapGrad::HasGrad => rc.gptr[k].expect("a slot with a gradient"),
            CapGrad::Dead | CapGrad::NoGrad | CapGrad::Unused => return Some(Ok(None)),
            CapGrad::Unknown => {
                return Some(Err(format!(
                    "autodiff: captured buffer #{k} of the @parallel region `{name}` needs a \
                     gradient, but its storage has none (a slice, a derived pointer, or a local \
                     array that is not `f32`); capture an `f32` array or a buffer parameter"
                )))
            }
        };
        if kind == CapGrad::HasGrad {
            if let Some(rc) = self.region.as_mut() {
                rc.needed[k] = true;
            }
        }
        let is_cap = self
            .region
            .as_ref()
            .is_some_and(|rc| rc.cap_of.contains_key(&buf));
        let target = if is_cap {
            Ok(Some(g))
        } else {
            self.region_slice_grad(buf, g)
        };
        if let Ok(Some(t)) = target {
            self.buf_adj.insert(buf, t);
            self.contributed.insert(buf);
        }
        Some(target)
    }

    /// The gradient of a `gep` chain over a captured slot pointer: the same chain over the slot's
    /// gradient pointer `g` (every link over one element type).
    fn region_slice_grad(&mut self, buf: ValueId, g: ValueId) -> Result<Option<ValueId>, String> {
        let mut links: Vec<(ValueId, MirType)> = Vec::new();
        let mut cur = buf;
        loop {
            if self
                .region
                .as_ref()
                .is_some_and(|rc| rc.cap_of.contains_key(&cur))
            {
                break;
            }
            match self.def_op.get(&cur) {
                Some(Op::Gep { ptr, index, elem }) => {
                    links.push((*index, elem.clone()));
                    cur = *ptr;
                }
                _ => {
                    return Err(format!(
                        "autodiff: cannot route the gradient of v{} to the captured buffer it \
                         views (not a gep chain over the env slot)",
                        buf.0
                    ))
                }
            }
        }
        let mut p = g;
        for (index, elem) in links.into_iter().rev() {
            let i = self.remap(index);
            p = self.b.build(
                PTR,
                Op::Gep {
                    ptr: p,
                    index: i,
                    elem,
                },
            );
        }
        Ok(Some(p))
    }

    /// Whether `out` is (a slice of) a buffer the region writes — the iteration overwrote it, so
    /// the adjoint of what it held before is zero over the written extent.
    pub(crate) fn region_written(&self, out: ValueId) -> bool {
        self.region.as_ref().is_some_and(|rc| {
            rc.slot_of(out)
                .is_some_and(|k| matches!(rc.kinds[k], CapGrad::Live | CapGrad::Dead))
        })
    }

    /// The VJP of the region call `self.regions[idx]` (see the module docs): build the iteration
    /// backward and its chunk driver, hand the gradients (or private copies) of the captured
    /// buffers to `wukong_parallel_for`, then add the private copies back in chunk order.
    pub(crate) fn diff_region(&mut self, idx: usize) -> Result<(), String> {
        let regions = self.regions;
        let rc: &'a RegionCall<'a> = &regions[idx];
        let name = self.it.resolve(rc.body.p.name).to_string();
        if self.mode != Mode::Grad {
            return Err(format!(
                "autodiff: second-order differentiation (grad_dot / hvp) supports scalar-path \
                 losses only; `{name}` is a @parallel region"
            ));
        }
        let nslots = rc.slots.len();
        // 1. What each slot's gradient is in the loss.
        let mut kinds = vec![CapGrad::Unused; nslots];
        let mut bases: Vec<Option<ValueId>> = vec![None; nslots];
        for k in 0..nslots {
            let (Some(v), Some(u)) = (rc.slots[k], rc.body.slots.get(k)) else {
                continue;
            };
            let base = self.canon(v);
            bases[k] = Some(base);
            kinds[k] = if u.write {
                if self.buf_adj.contains_key(&base) || self.grad_buf.contains_key(&base) {
                    CapGrad::Live
                } else {
                    CapGrad::Dead
                }
            } else if u.float_read {
                self.region_read_kind(base)
            } else {
                CapGrad::Unused
            };
        }
        // Two env slots naming one buffer, one of them written: an in-place region by another name.
        for a in 0..nslots {
            for b in 0..nslots {
                if a != b
                    && bases[a].is_some()
                    && bases[a] == bases[b]
                    && rc.body.slots[a].write
                    && rc.body.slots[b].read
                {
                    return Err(format!(
                        "autodiff: the @parallel region `{name}` reads and writes buffer v{} \
                         through two names (an in-place update); its backward recomputes each \
                         iteration from the buffers the forward left behind, so a region's \
                         captured inputs and outputs must be distinct buffers",
                        bases[a].map_or(0, |v| v.0)
                    ));
                }
            }
        }
        // 2. The iteration backward.
        let (it_fn, needed) = self.iteration_grad(rc, &kinds, &name)?;
        // 3. The loss-side gradient pointer of every slot the iteration uses.
        let mut gloss: Vec<Option<ValueId>> = vec![None; nslots];
        for k in 0..nslots {
            let Some(base) = bases[k] else {
                continue;
            };
            let used = match kinds[k] {
                CapGrad::Live => true,
                CapGrad::HasGrad => needed[k],
                _ => false,
            };
            if used {
                gloss[k] = Some(self.grad_target(base)?.ok_or_else(|| {
                    format!(
                        "autodiff: internal: captured buffer v{} lost its gradient",
                        base.0
                    )
                })?);
            }
        }
        // 4. Group the read slots by buffer; privatize the groups whose reads are shared.
        let mut groups: Vec<(ValueId, Vec<usize>)> = Vec::new();
        for k in 0..nslots {
            if kinds[k] != CapGrad::HasGrad || !needed[k] {
                continue;
            }
            let base = bases[k].expect("a needed slot has a base");
            match groups.iter_mut().find(|(b, _)| *b == base) {
                Some((_, ks)) => ks.push(k),
                None => groups.push((base, vec![k])),
            }
        }
        // (base, slots, element type, size) of every privatized group.
        let mut private: Vec<(ValueId, Vec<usize>, MirType, u64)> = Vec::new();
        for (base, ks) in &groups {
            if rc.body.partitioned(ks) {
                continue;
            }
            let mut elem: Option<MirType> = None;
            for &k in ks {
                let s = &rc.body.slots[k];
                let e = if s.kernel {
                    Some(MirType::F32)
                } else {
                    s.elem.clone()
                };
                if s.mixed_elem || (e.is_some() && elem.is_some() && e != elem) {
                    return Err(format!(
                        "autodiff: the @parallel region `{name}` reads buffer v{} as two \
                         element types; its shared adjoint cannot be privatized",
                        base.0
                    ));
                }
                elem = elem.or(e);
            }
            let elem = elem.unwrap_or(MirType::F32);
            let size = self.region_buffer_size(*base, &elem).ok_or_else(|| {
                format!(
                    "autodiff: buffer v{} is read by every iteration of the @parallel region \
                     `{name}`, so its adjoint is privatized per chunk, but its extent is not \
                     known (capture a fixed-size array or a local `f32` array)",
                    base.0
                )
            })?;
            private.push((*base, ks.clone(), elem, size));
        }
        // 5. The iteration range, N and K.
        let p = rc.body.p;
        let mut map: HashMap<ValueId, ValueId> = HashMap::new();
        let zero = self.cint(0);
        map.insert(p.params[0], zero);
        map.insert(p.params[1], self.remap(rc.n));
        map.insert(p.params[2], self.remap(rc.env));
        for inst in &rc.body.range {
            let op = map_operands(&inst.op, |v| map[&v]);
            let r = inst.result.expect("a range value");
            let nr = self.b.build(p.value_type(r).clone(), op);
            map.insert(r, nr);
        }
        let widen = |me: &mut Self, v: ValueId| {
            if rc.body.hty == I64 {
                v
            } else {
                let kind = if rc.body.unsigned {
                    CastKind::ZExt
                } else {
                    CastKind::SExt
                };
                me.b.build(I64, Op::Cast(kind, v, I64))
            }
        };
        let lo = widen(self, map[&rc.body.init]);
        let hi = widen(self, map[&rc.body.bound]);
        let span = self.b.build(I64, Op::Bin(BinOp::Sub, hi, lo));
        let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, span, zero));
        let n_iter = self.b.build(I64, Op::Select(pos, span, zero));
        let chunks = if private.is_empty() {
            n_iter
        } else {
            let cap = self.cint(PRIVATE_CHUNKS);
            let small = self.b.build(MirType::I1, Op::Cmp(CmpOp::Slt, n_iter, cap));
            self.b.build(I64, Op::Select(small, n_iter, cap))
        };
        // 6. The tables: one pointer per slot, then `[lo, N, K, size_k...]`.
        let null = self.b.build(PTR, Op::Cast(CastKind::IntToPtr, zero, PTR));
        let mut table: Vec<ValueId> = gloss.iter().map(|g| g.unwrap_or(null)).collect();
        let mut sizes: Vec<Option<ValueId>> = vec![None; nslots];
        let mut private_elem: Vec<Option<MirType>> = vec![None; nslots];
        for (_, ks, elem, size) in &private {
            let one = self.cint(1);
            let some = self.b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, chunks, one));
            let copies = self.b.build(I64, Op::Select(some, chunks, one));
            let sz = self.cint(*size as i64);
            let count = self.mul64(copies, sz);
            let width = self.cint(if *elem == MirType::F64 { 8 } else { 4 });
            let is_float = self.cint(1);
            let f = self.syms.rt_alloc;
            let pbuf = self.b.build(
                PTR,
                Op::Call {
                    func: f,
                    args: vec![count, width, is_float],
                },
            );
            self.scratch.push(pbuf);
            for &k in ks {
                table[k] = pbuf;
                sizes[k] = Some(sz);
                private_elem[k] = Some(elem.clone());
            }
        }
        let gtab = self
            .b
            .alloca(MirType::Array(Box::new(PTR), nslots.max(1) as u32));
        for (k, &v) in table.iter().enumerate() {
            self.store_slot(gtab, k, v, PTR);
        }
        let ints = self
            .b
            .alloca(MirType::Array(Box::new(I64), 3 + nslots as u32));
        self.store_slot(ints, 0, lo, I64);
        self.store_slot(ints, 1, n_iter, I64);
        self.store_slot(ints, 2, chunks, I64);
        for (k, s) in sizes.iter().enumerate() {
            let v = match s {
                Some(v) => *v,
                None => zero,
            };
            self.store_slot(ints, 3 + k, v, I64);
        }
        let benv = self.b.alloca(MirType::Array(Box::new(PTR), 3));
        let env_new = self.remap(rc.env);
        self.store_slot(benv, 0, env_new, PTR);
        self.store_slot(benv, 1, gtab, PTR);
        self.store_slot(benv, 2, ints, PTR);
        let addr = self.b.build(PTR, Op::FuncAddr(rc.bwd_name));
        let pfor = self.syms.pfor;
        self.b.build_void(Op::Call {
            func: pfor,
            args: vec![chunks, addr, benv],
        });
        // 7. Add the private copies into the real adjoints, chunk by chunk, in order.
        for (base, ks, elem, size) in &private {
            let g = gloss[ks[0]].expect("a privatized slot has a gradient");
            let pbuf = table[ks[0]];
            let sz = self.cint(*size as i64);
            let outer = self.open_loop(chunks);
            let c = outer.idx;
            let off = self.mul64(c, sz);
            let inner = self.open_loop(sz);
            let e = inner.idx;
            let at = self.b.build(I64, Op::Bin(BinOp::Add, off, e));
            let pp = self.b.build(
                PTR,
                Op::Gep {
                    ptr: pbuf,
                    index: at,
                    elem: elem.clone(),
                },
            );
            let pv = self.b.build(elem.clone(), Op::Load(pp, elem.clone()));
            let gp = self.b.build(
                PTR,
                Op::Gep {
                    ptr: g,
                    index: e,
                    elem: elem.clone(),
                },
            );
            let gv = self.b.build(elem.clone(), Op::Load(gp, elem.clone()));
            let sum = self.b.build(elem.clone(), Op::Bin(BinOp::FAdd, gv, pv));
            self.b.build_void(Op::Store {
                ptr: gp,
                value: sum,
            });
            self.close_loop(inner);
            self.close_loop(outer);
            self.contributed.insert(*base);
        }
        for (base, _) in &groups {
            self.contributed.insert(*base);
        }
        let bwd = build_bwd(rc.bwd_name, rc.it_name, &private_elem, &rc.body.hty);
        self.support.push(it_fn);
        self.support.push(bwd);
        Ok(())
    }

    /// `ptr[k] = v` for a table of `elem`.
    fn store_slot(&mut self, table: ValueId, k: usize, v: ValueId, elem: MirType) {
        let kv = self.cint(k as i64);
        let at = self.b.build(
            PTR,
            Op::Gep {
                ptr: table,
                index: kv,
                elem,
            },
        );
        self.b.build_void(Op::Store { ptr: at, value: v });
    }

    /// What the gradient of a buffer a region only reads is, in the loss (see [`CapGrad`]).
    fn region_read_kind(&self, base: ValueId) -> CapGrad {
        if self.buf_adj.contains_key(&base)
            || self.grad_buf.contains_key(&base)
            || self.buf_count.contains_key(&base)
        {
            CapGrad::HasGrad
        } else if self.fwd.params.contains(&base) {
            if self.kernel_out.contains_key(&base) || self.region_out.contains(&base) {
                CapGrad::HasGrad
            } else {
                CapGrad::NoGrad
            }
        } else {
            CapGrad::Unknown
        }
    }

    /// The element count of the forward buffer `base` holding `elem`s: a local array's length, or
    /// a fixed-extent parameter's.
    pub(crate) fn region_buffer_size(&self, base: ValueId, elem: &MirType) -> Option<u64> {
        if let Some(&n) = self.buf_count.get(&base) {
            return (*elem == MirType::F32).then_some(u64::from(n));
        }
        let i = self.fwd.params.iter().position(|&p| p == base)?;
        let ext = self.fwd.param_attr(i).extent?;
        let width = match elem {
            MirType::F64 => 8,
            MirType::F32 => 4,
            _ => return None,
        };
        (ext.elem_bytes == width).then_some(ext.elems)
    }

    /// Build `{grad}$par<i>$it(hh, env, gc)`: one iteration's recompute and reverse, in region
    /// mode. Returns the function and which read slots' gradients it accumulated into.
    fn iteration_grad(
        &self,
        rc: &'a RegionCall<'a>,
        kinds: &[CapGrad],
        name: &str,
    ) -> Result<(Function, Vec<bool>), String> {
        let f = &rc.body.f;
        let mut v = Vjp::new(
            f,
            rc.it_name,
            &[],
            self.syms.clone(),
            self.it,
            Mode::Grad,
            MirType::Void,
            &[],
        );
        let gc = v.b.add_param(PTR);
        v.region = Some(RegionCtx {
            cap_of: rc.body.cap_of.clone(),
            root: rc.body.root.clone(),
            kinds: kinds.to_vec(),
            gc,
            gptr: vec![None; kinds.len()],
            needed: vec![false; kinds.len()],
            name: name.to_string(),
        });
        v.scan_kernel_outputs();
        v.replay();
        v.region_setup();
        v.reverse()?;
        let needed = v
            .region
            .as_ref()
            .map(|r| r.needed.clone())
            .unwrap_or_default();
        let (func, support) = v.finish();
        debug_assert!(support.is_empty(), "an iteration has no nested region");
        Ok((func, needed))
    }
}
