//! Function inlining (a program-level transform).
//!
//! Inlining replaces a call with a copy of the callee's body, removing the call/return overhead and
//! exposing the callee's code to the caller's optimizer (constant args become constants, results
//! feed local CSE, a callee's loop nest becomes visible to LICM and the unroller). It runs once
//! before the function-level pipeline so that mem2reg and the rest then optimize the spliced-in code
//! in context.
//!
//! # Bottom-up over the call graph
//!
//! Callees are inlined **before** their callers, in reverse topological order of the call graph. That
//! is what lets a chain of helpers collapse in a single sweep: by the time `outer` is considered,
//! `mid` has already absorbed `leaf`, so splicing `mid` brings the whole chain. The previous
//! implementation snapshotted every callee body up front, inlined **leaf functions only**, and never
//! revisited a function afterwards — so `leaf` landed in `mid` and stopped there, and any user who
//! factored their kernel into two layers of helpers paid an abstraction tax that blocked every
//! downstream transform. Worse, the tax was *non-monotone*: whether a helper was inlined depended on
//! how deep it happened to sit, which is not something a user can predict.
//!
//! Order comes from Tarjan's strongly-connected components over the call graph, which emits each SCC
//! only after every SCC it calls. A function in a non-trivial SCC — or with a self-edge — is
//! recursive, and is never inlined into anything (splicing it could not terminate). That replaces the
//! old leaf-only rule, which forbade recursion by forbidding *all* calls.
//!
//! # The cost model
//!
//! [`should_inline`] scores each site rather than applying one blanket size cap:
//!
//!  * a callee with a **single call site** program-wide gets a much larger budget, because splicing
//!    it duplicates nothing — the original is deleted afterwards, so code size does not grow;
//!  * a call site **inside a loop** doubles the budget, since the win recurs every iteration;
//!  * a callee that **contains a loop** earns a bonus: inlining it hands a whole loop nest to the
//!    caller's LICM/CSE/unroller, which is worth far more than the call overhead alone;
//!  * a call whose **arguments are all constants** earns a bonus, since the body will likely fold.
//!
//! Growth is bounded three ways so compile time — one of this compiler's few genuine strengths —
//! cannot regress: no caller may exceed [`GROWTH_LIMIT`] instructions, no caller may more than
//! [`GROWTH_FACTOR`]× its original size, and no caller may take more than [`MAX_SPLICES`] splices.
//!
//! # The splice
//!
//! Standard SSA inlining over block-parameter MIR: the callee's values and blocks are copied with
//! fresh ids; the call's block is split so the call's successors live in a new continuation block
//! that takes the result as a parameter; the caller branches into the callee's (copied) entry passing
//! the arguments; and each `ret` in the copy becomes a branch to the continuation carrying the
//! returned value.

use crate::fxhash::{FxHashMap, FxHashSet};

use wukong_mir::{BasicBlock, BlockId, Function, Inst, Op, Program, Terminator, ValueId};
use wukong_span::Symbol;

use crate::{cfg, map_op_uses};

/// Baseline budget (in callee instructions) for a call site with nothing else going for it. This is
/// the old unconditional `SIZE_LIMIT`, kept as the floor so no site that used to be inlined stops
/// being inlined.
const BASE_LIMIT: usize = 40;
/// Budget for a callee with exactly one call site in the whole program. Splicing it duplicates
/// nothing — the sole call disappears and the original body is dropped — so code size does not grow
/// and only the growth caps below need to hold.
const SOLE_SITE_LIMIT: usize = 200;
/// Added when the callee contains a loop: inlining exposes a whole loop nest to the caller's LICM,
/// CSE and unroller, which is worth much more than the call overhead itself.
const LOOP_NEST_BONUS: usize = 40;
/// Added when every argument at the site is a compile-time constant: the body is likely to fold.
const CONST_ARG_BONUS: usize = 24;
/// Multiplier applied when the call site sits inside a loop in the caller.
const IN_LOOP_FACTOR: usize = 2;

/// Don't grow any one caller past this many instructions via inlining.
const GROWTH_LIMIT: usize = 5000;
/// Nor past this multiple of its own original size — a bound that scales with the function instead
/// of letting a small function balloon to `GROWTH_LIMIT`.
const GROWTH_FACTOR: usize = 6;
/// Nor via more than this many splices, which bounds the work per caller outright.
const MAX_SPLICES: u32 = 400;

/// Inline throughout the program, bottom-up over the call graph. Returns whether anything changed.
pub fn inline_program(program: &mut Program) -> bool {
    let n = program.funcs.len();
    if n == 0 {
        return false;
    }
    // Symbol -> index. Names are unique per program (the driver merges imports by qualified name).
    let index: FxHashMap<Symbol, u32> = program
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name, i as u32))
        .collect();

    // Call-graph adjacency over user functions only; a call to an intrinsic names no `Function`.
    let adj: Vec<Vec<u32>> = program
        .funcs
        .iter()
        .map(|f| {
            let mut out = Vec::new();
            let mut seen = FxHashSet::default();
            for b in &f.blocks {
                for inst in &b.insts {
                    if let Op::Call { func, .. } = &inst.op {
                        if let Some(&c) = index.get(func) {
                            if seen.insert(c) {
                                out.push(c);
                            }
                        }
                    }
                }
            }
            out
        })
        .collect();

    // Reverse-topological SCC order: every function is emitted after all the functions it calls.
    let sccs = strongly_connected(&adj);
    // A function is recursive — and so never inlinable — if it shares an SCC with anything else or
    // calls itself directly.
    let mut recursive = vec![false; n];
    for comp in &sccs {
        if comp.len() > 1 {
            for &m in comp {
                recursive[m as usize] = true;
            }
        } else {
            let m = comp[0];
            if adj[m as usize].contains(&m) {
                recursive[m as usize] = true;
            }
        }
    }

    // How many call sites each function has program-wide, before any inlining. A static heuristic:
    // it is only used to widen the budget for a sole-call-site callee, and re-counting after every
    // splice would cost more than it could possibly buy.
    let mut site_count: FxHashMap<Symbol, u32> = FxHashMap::default();
    for f in &program.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::Call { func, .. } = &inst.op {
                    *site_count.entry(*func).or_insert(0) += 1;
                }
            }
        }
    }

    // Finalized callee bodies, cloned on first use. Filling this lazily is correct *because* of the
    // bottom-up order: a function is fully inlined before any caller of it is processed, so the body
    // we clone is already final and never needs re-cloning.
    let mut bodies: FxHashMap<Symbol, Function> = FxHashMap::default();
    // Whether each callee's own body contains a loop — a cost-model input, computed once.
    let mut has_loop: FxHashMap<Symbol, bool> = FxHashMap::default();

    let mut changed = false;
    let mut inlined: FxHashSet<Symbol> = FxHashSet::default();

    for comp in &sccs {
        for &ci in comp {
            let ci = ci as usize;
            let original_size = func_insts(&program.funcs[ci]);
            let size_cap =
                GROWTH_LIMIT.min(original_size.saturating_mul(GROWTH_FACTOR).max(BASE_LIMIT));
            // Which of the caller's blocks sit inside a loop; extended in place as we splice.
            let mut in_loop = loop_blocks(&program.funcs[ci]);
            let mut splices = 0u32;

            loop {
                if splices >= MAX_SPLICES || func_insts(&program.funcs[ci]) > size_cap {
                    break;
                }
                let Some(site) = pick_site(
                    &program.funcs[ci],
                    &in_loop,
                    &index,
                    &recursive,
                    &site_count,
                    &mut bodies,
                    &mut has_loop,
                    program_ref(program),
                    size_cap,
                ) else {
                    break;
                };
                // Never splice a function into itself, even though the SCC rule already excludes it.
                if site.callee == program.funcs[ci].name {
                    break;
                }
                let callee = bodies
                    .get(&site.callee)
                    .expect("pick_site caches the callee body")
                    .clone();
                let site_in_loop = in_loop[site.block];
                let added = callee.blocks.len() + 1; // callee blocks plus the continuation
                inline_call_site(&mut program.funcs[ci], &callee, site.block, site.inst);
                // The spliced blocks and the continuation are all as loop-resident as the call was.
                in_loop.resize(in_loop.len() + added, site_in_loop);
                inlined.insert(site.callee);
                changed = true;
                splices += 1;
            }
        }
    }

    // Drop callees that we inlined and that are now referenced nowhere. `main` is never a call
    // target, so it is never in `inlined` and is always kept — which makes this safe without knowing
    // which function is the entry. `FuncAddr` counts as a reference: the outlined body of a
    // `@parallel for` is reached *only* by its address, so a name-based liveness scan that looked at
    // `Call` alone could delete a live function out from under the runtime.
    if changed {
        let mut still_referenced: FxHashSet<Symbol> = FxHashSet::default();
        for f in &program.funcs {
            for b in &f.blocks {
                for inst in &b.insts {
                    match &inst.op {
                        Op::Call { func, .. } => {
                            still_referenced.insert(*func);
                        }
                        Op::FuncAddr(s) => {
                            still_referenced.insert(*s);
                        }
                        _ => {}
                    }
                }
            }
        }
        program
            .funcs
            .retain(|f| !inlined.contains(&f.name) || still_referenced.contains(&f.name));
    }
    changed
}

/// A borrow of the program's function list, so `pick_site` can read callee bodies while the caller
/// is being mutated by index. Kept as a helper to make the aliasing obvious at the call site.
fn program_ref(program: &Program) -> &[Function] {
    &program.funcs
}

/// A chosen call site: where it is in the caller, and whom it calls.
struct Site {
    block: usize,
    inst: usize,
    callee: Symbol,
}

/// The first call site in `f` that the cost model approves, caching the callee's body and
/// loop-shape as it goes.
#[allow(clippy::too_many_arguments)]
fn pick_site(
    f: &Function,
    in_loop: &[bool],
    index: &FxHashMap<Symbol, u32>,
    recursive: &[bool],
    site_count: &FxHashMap<Symbol, u32>,
    bodies: &mut FxHashMap<Symbol, Function>,
    has_loop: &mut FxHashMap<Symbol, bool>,
    funcs: &[Function],
    size_cap: usize,
) -> Option<Site> {
    // Values the caller defines as compile-time constants, for the all-constant-args bonus. Built
    // once per scan, which costs the same asymptotically as the scan itself.
    let mut consts: FxHashSet<u32> = FxHashSet::default();
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::ConstInt(..) | Op::ConstFloat(..)) = (inst.result, &inst.op) {
                consts.insert(r.0);
            }
        }
    }

    let caller_size = func_insts(f);
    for (bi, b) in f.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            let Op::Call { func, args } = &inst.op else {
                continue;
            };
            let Some(&idx) = index.get(func) else {
                continue; // an intrinsic — no body to splice
            };
            if recursive[idx as usize] || *func == f.name {
                continue;
            }
            let body = bodies
                .entry(*func)
                .or_insert_with(|| funcs[idx as usize].clone());
            let callee_size = func_insts(body);
            let callee_loops = *has_loop
                .entry(*func)
                .or_insert_with(|| !loop_blocks(&funcs[idx as usize]).iter().all(|x| !x));
            if caller_size + callee_size > size_cap {
                continue;
            }
            let approved = should_inline(
                callee_size,
                site_count.get(func).copied().unwrap_or(1),
                in_loop.get(bi).copied().unwrap_or(false),
                callee_loops,
                !args.is_empty() && args.iter().all(|a| consts.contains(&a.0)),
            );
            if approved {
                return Some(Site {
                    block: bi,
                    inst: ii,
                    callee: *func,
                });
            }
        }
    }
    None
}

/// The cost model: is a callee of `callee_size` instructions worth splicing at this site?
///
/// Pure arithmetic on the four site facts, so it is trivially deterministic and easy to retune. The
/// budget starts at [`BASE_LIMIT`] — the old blanket cap — so this can only ever approve *more*
/// sites than the previous rule, never fewer.
fn should_inline(
    callee_size: usize,
    call_sites: u32,
    in_loop: bool,
    callee_has_loop: bool,
    all_const_args: bool,
) -> bool {
    let mut budget = if call_sites <= 1 {
        SOLE_SITE_LIMIT
    } else {
        BASE_LIMIT
    };
    if in_loop {
        budget *= IN_LOOP_FACTOR;
    }
    if callee_has_loop {
        budget += LOOP_NEST_BONUS;
    }
    if all_const_args {
        budget += CONST_ARG_BONUS;
    }
    callee_size <= budget
}

fn func_insts(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

/// Which blocks of `f` lie on a cycle in its CFG — i.e. sit inside a loop. Computed as the
/// strongly-connected components of the CFG: a block is in a loop exactly when its component has
/// more than one member, or it branches to itself. This needs no dominator information, which keeps
/// the inliner off the (more expensive) analyses the function-level pipeline builds later.
fn loop_blocks(f: &Function) -> Vec<bool> {
    let adj: Vec<Vec<u32>> = f
        .blocks
        .iter()
        .map(|b| cfg::successors(&b.term).iter().map(|s| s.0).collect())
        .collect();
    let mut out = vec![false; f.blocks.len()];
    for comp in strongly_connected(&adj) {
        if comp.len() > 1 {
            for m in comp {
                out[m as usize] = true;
            }
        } else {
            let m = comp[0];
            if adj[m as usize].contains(&m) {
                out[m as usize] = true;
            }
        }
    }
    out
}

/// Tarjan's strongly-connected components, iterative.
///
/// Emitted in reverse topological order: a component appears only after every component reachable
/// from it. The recursion is turned into an explicit stack because both call graphs and CFGs here
/// can be thousands of nodes deep in generated model code, and blowing the *compiler's* stack is not
/// an acceptable failure mode.
fn strongly_connected(adj: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = adj.len();
    const UNVISITED: u32 = u32::MAX;
    let mut index = vec![UNVISITED; n];
    let mut lowlink = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut next_index = 0u32;
    let mut out: Vec<Vec<u32>> = Vec::new();

    // Each frame is (node, index of the next successor to visit).
    let mut work: Vec<(u32, usize)> = Vec::new();

    for root in 0..n as u32 {
        if index[root as usize] != UNVISITED {
            continue;
        }
        work.push((root, 0));
        while let Some(&mut (v, ref mut si)) = work.last_mut() {
            if *si == 0 {
                index[v as usize] = next_index;
                lowlink[v as usize] = next_index;
                next_index += 1;
                stack.push(v);
                on_stack[v as usize] = true;
            }
            if *si < adj[v as usize].len() {
                let w = adj[v as usize][*si];
                *si += 1;
                if index[w as usize] == UNVISITED {
                    work.push((w, 0));
                } else if on_stack[w as usize] {
                    lowlink[v as usize] = lowlink[v as usize].min(index[w as usize]);
                }
                continue;
            }
            // v is finished: pop it and, if it is a root, emit its component.
            work.pop();
            if let Some(&(p, _)) = work.last() {
                lowlink[p as usize] = lowlink[p as usize].min(lowlink[v as usize]);
            }
            if lowlink[v as usize] == index[v as usize] {
                let mut comp = Vec::new();
                while let Some(w) = stack.pop() {
                    on_stack[w as usize] = false;
                    comp.push(w);
                    if w == v {
                        break;
                    }
                }
                out.push(comp);
            }
        }
    }
    out
}

fn inline_call_site(caller: &mut Function, callee: &Function, bi: usize, ii: usize) {
    let (result, args) = match &caller.blocks[bi].insts[ii] {
        Inst {
            result,
            op: Op::Call { args, .. },
        } => (*result, args.clone()),
        _ => unreachable!("call site must be a Call"),
    };

    // Fresh caller value id for every callee value (params included — they receive the call's
    // arguments through the entry block's parameters).
    let mut vmap: FxHashMap<u32, ValueId> = FxHashMap::default();
    for (vid, ty) in callee.value_types.iter().enumerate() {
        let nv = ValueId(caller.value_types.len() as u32);
        caller.value_types.push(ty.clone());
        vmap.insert(vid as u32, nv);
    }
    let mapv = |v: ValueId| -> ValueId { vmap[&v.0] };

    // Callee block b -> caller block (base + b); the continuation block follows them.
    let base = caller.blocks.len() as u32;
    let bmap = |b: BlockId| -> BlockId { BlockId(base + b.0) };
    let cont_id = BlockId(base + callee.blocks.len() as u32);

    // Split the call's block: instructions after the call (plus its terminator) become the
    // continuation, which takes the call's result as a parameter.
    let tail = caller.blocks[bi].insts.split_off(ii + 1);
    caller.blocks[bi].insts.truncate(ii); // drop the call instruction itself
    let orig_term = std::mem::replace(
        &mut caller.blocks[bi].term,
        Terminator::Br {
            target: bmap(callee.entry),
            args, // the call's arguments flow into the callee entry's parameters
        },
    );
    let cont = BasicBlock {
        id: cont_id,
        params: match result {
            Some(r) => vec![r],
            None => vec![],
        },
        insts: tail,
        term: orig_term,
    };

    // `Op::VecKernelCall.kernel` is an index into the *owning* function's `vec_kernels` table, not a
    // `ValueId`, so `map_op_uses` does not (and must not) touch it. Re-parenting the callee's body
    // therefore has to carry its recipes into the caller's table and shift every copied index by
    // where they landed — otherwise the spliced call names one of the caller's own recipes (silent
    // wrong kernel) or an index past the end (dangling). Empty for every non-vectorized callee, so
    // this is a no-op on the common path.
    let kbase = caller.vec_kernels.len() as u32;
    caller
        .vec_kernels
        .extend(callee.vec_kernels.iter().cloned());

    // Copy the callee's blocks with everything remapped; `ret` becomes a branch to the continuation.
    for cb in &callee.blocks {
        let params = cb.params.iter().map(|p| mapv(*p)).collect();
        let insts = cb
            .insts
            .iter()
            .map(|inst| {
                let mut op = inst.op.clone();
                map_op_uses(&mut op, mapv);
                if let Op::VecKernelCall { kernel, .. } = &mut op {
                    *kernel += kbase;
                }
                Inst {
                    result: inst.result.map(mapv),
                    op,
                }
            })
            .collect();
        let term = remap_term(&cb.term, &bmap, &mapv, cont_id);
        caller.blocks.push(BasicBlock {
            id: bmap(cb.id),
            params,
            insts,
            term,
        });
    }
    caller.blocks.push(cont);
}

fn remap_term(
    t: &Terminator,
    bmap: &impl Fn(BlockId) -> BlockId,
    mapv: &impl Fn(ValueId) -> ValueId,
    cont: BlockId,
) -> Terminator {
    let margs = |a: &[ValueId]| a.iter().map(|v| mapv(*v)).collect();
    match t {
        Terminator::Ret(Some(v)) => Terminator::Br {
            target: cont,
            args: vec![mapv(*v)],
        },
        Terminator::Ret(None) => Terminator::Br {
            target: cont,
            args: vec![],
        },
        Terminator::Br { target, args } => Terminator::Br {
            target: bmap(*target),
            args: margs(args),
        },
        Terminator::CondBr {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => Terminator::CondBr {
            cond: mapv(*cond),
            then_blk: bmap(*then_blk),
            then_args: margs(then_args),
            else_blk: bmap(*else_blk),
            else_args: margs(else_args),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scc_emits_callees_before_callers() {
        // 0 -> 1 -> 2, plus a 3 <-> 4 cycle. Reverse topological order means 2 before 1 before 0.
        let adj = vec![vec![1], vec![2], vec![], vec![4], vec![3]];
        let sccs = strongly_connected(&adj);
        let pos = |x: u32| {
            sccs.iter()
                .position(|c| c.contains(&x))
                .expect("every node is in some component")
        };
        assert!(pos(2) < pos(1), "callee 2 must precede caller 1");
        assert!(pos(1) < pos(0), "callee 1 must precede caller 0");
        // The cycle is one component holding both members.
        let cyc = sccs.iter().find(|c| c.contains(&3)).unwrap();
        assert_eq!(cyc.len(), 2, "3 and 4 form one SCC: {sccs:?}");
        assert!(cyc.contains(&4));
    }

    #[test]
    fn scc_handles_a_self_loop_and_isolated_nodes() {
        let adj = vec![vec![0], vec![], vec![]];
        let sccs = strongly_connected(&adj);
        assert_eq!(sccs.len(), 3, "three singleton components: {sccs:?}");
        // A self-edge is what marks node 0 recursive; the component itself is still a singleton.
        assert!(sccs.iter().any(|c| c == &vec![0]));
    }

    #[test]
    fn cost_model_floor_is_the_old_blanket_limit() {
        // With nothing else going for it, the model approves exactly what the old `SIZE_LIMIT` did,
        // so no site that used to be inlined can stop being inlined.
        assert!(should_inline(BASE_LIMIT, 5, false, false, false));
        assert!(!should_inline(BASE_LIMIT + 1, 5, false, false, false));
    }

    #[test]
    fn cost_model_widens_for_sole_site_loops_and_constants() {
        // A sole call site duplicates nothing, so it earns the big budget.
        assert!(should_inline(SOLE_SITE_LIMIT, 1, false, false, false));
        assert!(!should_inline(SOLE_SITE_LIMIT + 1, 1, false, false, false));
        // A multi-site callee too big for the floor still gets in when the site is hot, when it
        // brings a loop nest with it, or when its arguments are constants.
        assert!(!should_inline(BASE_LIMIT + 10, 5, false, false, false));
        assert!(should_inline(BASE_LIMIT + 10, 5, true, false, false));
        assert!(should_inline(BASE_LIMIT + 10, 5, false, true, false));
        assert!(should_inline(BASE_LIMIT + 10, 5, false, false, true));
    }
}
