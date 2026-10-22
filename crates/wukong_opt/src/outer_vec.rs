//! Outer-loop vectorization: widen a two-deep nest **across its outer counter**.
//!
//! The inner-loop path widens a loop's own iterations, `W` of them per group. That is the wrong
//! axis for a nest whose inner loop carries a value from one iteration to the next — a matrix-
//! vector product's per-row sum, a per-row Horner recurrence. A float sum may not be reassociated
//! (module docs), so widened along `c` it keeps one serial fold — a `W`-long dependent chain per
//! group, no faster than the scalar loop — and a recurrence `h = a·h + b` cannot be widened along
//! `c` at all. Across the rows, the same work is `W` **independent** chains: lane `k` runs the
//! inner loop for row `r + k`, its own accumulator, in exactly the scalar loop's order. That is
//! LLVM's VPlan outer-loop path, and the order-preservation argument is per lane:
//!
//! * the inner loop's start, bound and test do not depend on the row, so every lane runs the same
//!   iterations — the group's inner loop is one scalar loop whose counter `c` every lane shares,
//!   and lane `k`'s value at step `c` is what row `r + k`'s scalar loop computed at step `c`, by the
//!   same operations on the same operands in the same order (lane-wise ops only, nothing
//!   reassociated), so the result is bit-exact;
//! * memory: the nest stores only outside the inner loop (before or after it), each store at
//!   element `r + k` of its object (one element per lane, never shared), and every stored object
//!   is provably apart from everything else the nest touches (two stack slots, a `@noalias`
//!   promise, provenance) — except the same element of the same row (`y[r]` read back after it is
//!   written), which each lane still does in program order. So no lane reads what another lane
//!   writes, and running row `r + 1`'s inner step `c` before row `r`'s step `c + 1` is invisible.
//!
//! The rows past the last whole group run in the original nest, which stays as the epilogue — its
//! inner loop is then widened along `c` like any other.
//!
//! # The shape
//!
//! ```text
//!   H_o (test r)  ->  P (inner preheader)  ->  H_i (test c)  <->  B (inner latch)
//!                                                 |
//!                                                 v
//!                                              X (outer latch)  ->  H_o
//! ```
//!
//! Every block straight-line, both headers test-only, one exit each (`H_o` out of the nest, `H_i`
//! to `X`), the outer loop carrying only its counter. Values are [`Kind`]s: the same in every lane
//! ([`Kind::Uniform`] — `c`, the inner bound, a scalar read), the counter plus a uniform offset
//! ([`Kind::Ramp`] — what a row-contiguous address indexes by), anything else that varies by row
//! ([`Kind::Vary`], a vector, or per-lane scalars where no packed form exists), and an address
//! that varies by row ([`Kind::Addr`], read only by the load or store it feeds: one vector access
//! when its index is a ramp, one scalar access per lane otherwise).
//!
//! # When
//!
//! Only when it beats what the nest gets otherwise, a rule rather than a measurement (none of these
//! forms has been timed): the inner loop must carry a row-varying value the inner path cannot widen
//! in parallel. A reduction that is not reassociable (a float sum, a float min/max) qualifies —
//! the inner path folds it serially, and once it has, `unroll_jam` no longer sees the nest. A
//! recurrence qualifies only when every inner step loads whole vectors (rows of a group contiguous,
//! or one value for the group): over gathered rows, `unroll_jam`'s four scalar chains do the same
//! work without the packing. An integer sum, or an elementwise inner loop, widens well on its own
//! and is left to the inner path. `WUKONG_NO_OUTER_VEC` turns the path off.

use super::*;

/// What one value of the nest is in the vector nest.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// The same in every lane: one scalar for the group.
    Uniform,
    /// The outer counter plus a uniform offset: `r + k + u` in lane `k`.
    Ramp,
    /// Varies by row, any other way.
    Vary,
    /// A `gep` off a uniform base by a row-varying index: an address, only read as one.
    Addr,
}

/// How a row-varying instruction is reproduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum How {
    /// The same op on vector parts ([`emit_widened`]).
    Widen,
    /// A conversion, by its route.
    Cast(CastRoute),
    /// The scalar op once per lane.
    Replicate,
    /// A load or store of `W` contiguous elements (its address's index is a ramp).
    Contiguous,
    /// A load of one element per lane.
    Gather,
    /// Add or subtract a uniform value to or from a ramp: another ramp, no code.
    RampStep,
    /// A row-varying address: no code until its load or store.
    Addr,
}

/// Everything [`apply`] needs.
pub(super) struct OuterPlan {
    pub(super) header: BlockId,
    preheader: BlockId,
    pre: BlockId,
    inner_header: BlockId,
    inner_body: BlockId,
    post: BlockId,
    iv_index: usize,
    iv: ValueId,
    iv_ty: MirType,
    pub(super) w: u32,
    trip: TripPlan,
    kinds: FxHashMap<u32, Kind>,
    how: FxHashMap<u32, How>,
    /// Per inner header parameter: is it a vector (one lane per row)?
    carried_vec: Vec<bool>,
}

/// May outer-loop vectorization run? `WUKONG_NO_OUTER_VEC` is its kill switch.
pub(super) fn enabled() -> bool {
    std::env::var_os("WUKONG_NO_OUTER_VEC").is_none()
}

/// The outer-loop plan for the nest whose outer loop is `forest.loops[outer]`, or why not.
pub(super) fn plan(
    f: &Function,
    forest: &mut LoopForest,
    outer: usize,
) -> Result<OuterPlan, &'static str> {
    // ---- structure, before any value analysis -----------------------------------------------
    let (pre, inner, post) = {
        let lo = &forest.loops[outer];
        let [inner_id] = lo.children.as_slice() else {
            return Err("not a two-deep nest");
        };
        let li = forest.get(*inner_id);
        if !li.children.is_empty() {
            return Err("not a two-deep nest");
        }
        if lo.preheader.is_none()
            || lo.latches.len() != 1
            || !lo.abnormal_exits.is_empty()
            || lo.exits.len() != 1
            || lo.exits[0].0 != lo.header
            || lo.blocks.len() != 5
        {
            return Err("the outer loop is not one test, one body and one latch");
        }
        if li.latches.len() != 1
            || !li.abnormal_exits.is_empty()
            || li.blocks.len() != 2
            || li.exits.len() != 1
            || li.exits[0].0 != li.header
        {
            return Err("the inner loop is not one test and one straight-line body");
        }
        let post = lo.latches[0];
        if li.exits[0].1 != post {
            return Err("the inner loop does not leave into the outer latch");
        }
        let inside: Vec<BlockId> = succs(&f.block(lo.header).term)
            .into_iter()
            .filter(|b| lo.blocks.contains(b))
            .collect();
        let [pre] = inside.as_slice() else {
            return Err("the outer header does not enter one block");
        };
        if li.preheader != Some(*pre) {
            return Err("the outer body is not one block ahead of the inner loop");
        }
        if !header_is_test_only(f, lo) || !header_is_test_only(f, li) {
            return Err("a header computes more than its exit test");
        }
        if !matches!(&f.block(post).term, Terminator::Br { target, .. } if *target == lo.header)
            || !f.block(post).params.is_empty()
        {
            return Err("the outer latch is not a plain jump back");
        }
        if !matches!(&f.block(li.latches[0]).term, Terminator::Br { target, .. } if *target == li.header)
        {
            return Err("the inner latch is not a plain jump back");
        }
        match &f.block(li.header).term {
            Terminator::CondBr {
                then_args,
                else_args,
                ..
            } if then_args.is_empty() && else_args.is_empty() => {}
            _ => return Err("the inner exit passes values"),
        }
        (*pre, inner_id.0 as usize, post)
    };

    // ---- induction variables, trip counts, memory ---------------------------------------------
    let facts = loop_info::DepFacts::default();
    loop_info::analyze_one_with(f, forest, outer, loop_info::Stages::Memory, facts);
    loop_info::analyze_one_with(f, forest, inner, loop_info::Stages::Memory, facts);
    let lo = &forest.loops[outer];
    let li = &forest.loops[inner];
    let oiv = lo.primary().ok_or("no outer induction variable")?;
    if oiv.step != IvStep::Const(1) || !oiv.ty.is_int() || oiv.ty == MirType::I1 {
        return Err("the outer counter does not step by one");
    }
    let iiv = li.primary().ok_or("no inner induction variable")?;
    if iiv.step != IvStep::Const(1)
        || !matches!(li.trip, TripCount::Const(_) | TripCount::Affine { .. })
    {
        return Err("the inner loop is not a counted loop");
    }
    if lo
        .carried
        .iter()
        .enumerate()
        .any(|(k, c)| k != oiv.param_index && !matches!(c.kind, Carried::Invariant))
    {
        return Err("the outer loop carries a value besides its counter");
    }
    let header_ps = &f.block(lo.header).params;
    if header_ps.get(oiv.param_index) != Some(&oiv.value) {
        return Err("the outer counter is not a header parameter");
    }
    memory_verdict(f, lo, pre, post)?;

    // ---- the row-varying values -----------------------------------------------------------------
    let inner_params = f.block(li.header).params.clone();
    let entry_args = match &f.block(pre).term {
        Terminator::Br { args, .. } => args.clone(),
        _ => return Err("the inner preheader does not jump"),
    };
    let latch_args = match &f.block(li.latches[0]).term {
        Terminator::Br { args, .. } => args.clone(),
        _ => return Err("the inner latch does not jump"),
    };
    let mut carried_vec = vec![false; inner_params.len()];
    let mut walk = None;
    for _ in 0..=inner_params.len() {
        let w = classify(
            f,
            oiv.value,
            &oiv.ty,
            pre,
            li.header,
            li.latches[0],
            post,
            &inner_params,
            &carried_vec,
            iiv.param_index,
        )?;
        let mut grew = false;
        for (k, _) in inner_params.iter().enumerate() {
            let varies = |v: &ValueId| matches!(w.kind(*v), Kind::Ramp | Kind::Vary);
            if !carried_vec[k] && (varies(&entry_args[k]) || varies(&latch_args[k])) {
                carried_vec[k] = true;
                grew = true;
            }
        }
        let done = !grew;
        walk = Some(w);
        if done {
            break;
        }
    }
    let walk = walk.ok_or("no classification")?;
    if carried_vec[iiv.param_index] {
        return Err("the inner counter varies by row");
    }
    // The inner loop's test is the group's: it must not vary by row.
    if let Terminator::CondBr { cond, .. } = &f.block(li.header).term {
        if walk.kind(*cond) != Kind::Uniform {
            return Err("the inner trip varies by row");
        }
    }
    for (k, p) in inner_params.iter().enumerate() {
        if carried_vec[k] && (lane_bytes(f.value_type(*p)).is_none() || is_half(f.value_type(*p))) {
            return Err("a row-varying inner value has no vector lane type");
        }
    }

    // ---- profitability: only what the other paths do worse ----------------------------------------
    // A strict-order reduction (a float sum, a float min/max) the inner path widens with one serial
    // fold — a `W`-long chain per group — and once it has, `unroll_jam` no longer sees the nest:
    // across the rows it is `W` independent chains. A recurrence the inner path declines, and
    // `unroll_jam` then runs four rows as four scalar chains; across the rows is only better when
    // each inner step loads one vector rather than gathering a row's element per lane, so a
    // recurrence over gathered rows stays with `unroll_jam`.
    let mut strict = false;
    let mut recurrence = false;
    for (k, c) in li.carried.iter().enumerate() {
        if !carried_vec[k] {
            continue;
        }
        match c.kind {
            Carried::Recurrence => recurrence = true,
            Carried::Reduction(r) if !li.reductions[r].reassociable => strict = true,
            _ => {}
        }
    }
    if !strict && !recurrence {
        return Err("the inner loop widens on its own");
    }
    let inner_gathers = f.block(li.latches[0]).insts.iter().any(|i| {
        i.result
            .is_some_and(|r| walk.how.get(&r.0) == Some(&How::Gather))
    });
    if !strict && inner_gathers {
        return Err("a recurrence over gathered rows: unroll-and-jam runs it as scalar chains");
    }

    // ---- width ---------------------------------------------------------------------------------
    let mut narrowest = u32::MAX;
    let mut widest = 0;
    for t in walk.vec_tys.iter().chain(
        inner_params
            .iter()
            .zip(&carried_vec)
            .filter(|(_, v)| **v)
            .map(|(p, _)| f.value_type(*p)),
    ) {
        let b = lane_bytes(t).ok_or("a row-varying value has no vector lane type")?;
        narrowest = narrowest.min(b);
        widest = widest.max(b);
    }
    if widest == 0 {
        return Err("nothing in the nest becomes a vector");
    }
    if widest / narrowest > MAX_PARTS {
        return Err("the nest's lane widths would split one value across more than four vectors");
    }
    let w = VEC_BYTES / narrowest;
    let trip = match &lo.trip {
        TripCount::Const(n) => {
            let groups = n / u64::from(w);
            if groups < MIN_TRIP_GROUPS {
                return Err("the outer trip is below one vector group");
            }
            TripPlan::Const(groups * u64::from(w))
        }
        TripCount::Affine {
            step: 1,
            inclusive: false,
            end,
            signed,
            ..
        } => TripPlan::Runtime {
            end: *end,
            signed: *signed,
        },
        _ => return Err("the outer trip is not a unit-step half-open range"),
    };
    Ok(OuterPlan {
        header: lo.header,
        preheader: lo.preheader.expect("checked above"),
        pre,
        inner_header: li.header,
        inner_body: li.latches[0],
        post,
        iv_index: oiv.param_index,
        iv: oiv.value,
        iv_ty: oiv.ty.clone(),
        w,
        trip,
        kinds: walk.kinds,
        how: walk.how,
        carried_vec,
    })
}

/// The nest's memory, as the module docs require it: every store before or after the inner loop,
/// at element `r` of its object (unit stride in the outer counter, the element type its index
/// counts in), and every stored object apart from everything else the nest touches but the same
/// element of the same row.
fn memory_verdict(
    f: &Function,
    lo: &NaturalLoop,
    pre: BlockId,
    post: BlockId,
) -> Result<(), &'static str> {
    let inside: FxHashSet<u32> = lo
        .blocks
        .iter()
        .flat_map(|b| {
            let blk = f.block(*b);
            blk.params
                .iter()
                .map(|p| p.0)
                .chain(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)))
        })
        .collect();
    let mut promised = crate::alias::PromisedBases::new(f);
    let mut alias: Option<crate::alias::AliasInfo> = None;
    let handle = |m: MemBase| match m {
        MemBase::Outer(v) => v,
        MemBase::ReloadedOuter { value, .. } => value,
    };
    for s in lo.accesses.iter().filter(|a| a.kind == AccessKind::Store) {
        if s.block != pre && s.block != post {
            return Err("the nest stores inside its inner loop");
        }
        let AddrForm::Affine { base, index, elem } = &s.addr else {
            return Err("a store of the nest is not at the row's own element");
        };
        if index.coeff != Coeff::Const(1) || *elem != s.value_ty {
            return Err("a store of the nest is not at the row's own element");
        }
        for a in &lo.accesses {
            if (a.block, a.inst) == (s.block, s.inst) {
                continue;
            }
            // A slice re-reading its data pointer out of its fat pointer: a typed element store
            // never lands on a slice header (no type punning — `bce`'s rule).
            if a.kind == AccessKind::Load
                && a.value_ty == MirType::Ptr
                && !matches!(s.value_ty, MirType::Ptr | MirType::I8 | MirType::I1)
            {
                continue;
            }
            let Some(ab) = a
                .addr
                .base()
                .or(a.gather_base)
                .or_else(|| guarded_base(f, &inside, a.ptr))
            else {
                return Err("an access of the nest has no known object");
            };
            if base.same_object(ab) {
                let same_element = matches!(&a.addr, AddrForm::Affine { index: ai, elem: ae, .. }
                    if ai == index && ae == elem && a.value_ty == s.value_ty);
                if !same_element {
                    return Err("a stored object is also touched at another element");
                }
                continue;
            }
            let apart = distinct_alloca_pair(f, *base, ab)
                || promised.disjoint(*base, ab)
                || !alias
                    .get_or_insert_with(|| crate::alias::AliasInfo::analyze(f))
                    .may_alias(handle(*base), handle(ab));
            if !apart {
                return Err("a stored object may be what the nest reads");
            }
        }
    }
    Ok(())
}

/// The object an element access `gep b, idx` reads when its base `b` is a slice's data pointer that
/// `licm` hoisted to the inner preheader under a first-iteration guard: `b = load ptr (select(c,
/// hdr, dummy))`, `hdr` and the private `dummy` slot both defined outside the nest. The select
/// picks `dummy` only on an entry where the inner loop runs no iteration, and then no element
/// access runs either, so every access that does run reads the slice `hdr` holds.
fn guarded_base(f: &Function, inside: &FxHashSet<u32>, ptr: ValueId) -> Option<MemBase> {
    let Some(Op::Gep { ptr: b, .. }) = op_of(f, ptr) else {
        return None;
    };
    let Some(Op::Load(sel, MirType::Ptr)) = op_of(f, *b) else {
        return None;
    };
    let Some(Op::Select(_, hdr, dummy)) = op_of(f, *sel) else {
        return None;
    };
    let outside = |v: &ValueId| !inside.contains(&v.0);
    if !outside(hdr) || !outside(dummy) || !matches!(op_of(f, *dummy), Some(Op::Alloca { .. })) {
        return None;
    }
    Some(MemBase::ReloadedOuter {
        addr: *hdr,
        value: *b,
    })
}

/// One classification walk over the nest's blocks, in order: the inner preheader, the inner
/// header's test, the inner body, the outer latch.
struct Walk {
    kinds: FxHashMap<u32, Kind>,
    how: FxHashMap<u32, How>,
    /// Lane types of the values the vector nest holds as vectors.
    vec_tys: Vec<MirType>,
}

impl Walk {
    fn kind(&self, v: ValueId) -> Kind {
        self.kinds.get(&v.0).copied().unwrap_or(Kind::Uniform)
    }
}

#[allow(clippy::too_many_arguments)]
fn classify(
    f: &Function,
    iv: ValueId,
    iv_ty: &MirType,
    pre: BlockId,
    inner_header: BlockId,
    inner_body: BlockId,
    post: BlockId,
    inner_params: &[ValueId],
    carried_vec: &[bool],
    inner_iv_index: usize,
) -> Result<Walk, &'static str> {
    let mut w = Walk {
        kinds: FxHashMap::default(),
        how: FxHashMap::default(),
        vec_tys: Vec::new(),
    };
    w.kinds.insert(iv.0, Kind::Ramp);
    for (k, p) in inner_params.iter().enumerate() {
        let kind = if k != inner_iv_index && carried_vec[k] {
            Kind::Vary
        } else {
            Kind::Uniform
        };
        w.kinds.insert(p.0, kind);
    }
    for b in [pre, inner_header, inner_body, post] {
        let stores_here = b == pre || b == post;
        for inst in &f.block(b).insts {
            classify_inst(f, &mut w, inst, iv_ty, stores_here)?;
        }
        // A terminator reads its values as they are: a branch argument of the inner loop is a
        // carried value (vector or uniform, settled by the caller), and the outer latch passes
        // back only the counter and invariants.
        let mut addr_use = false;
        crate::each_term_use(&f.block(b).term, &mut |u| {
            addr_use |= w.kind(u) == Kind::Addr;
        });
        if addr_use {
            return Err("a row-varying address is used as a value");
        }
    }
    Ok(w)
}

fn classify_inst(
    f: &Function,
    w: &mut Walk,
    inst: &Inst,
    iv_ty: &MirType,
    stores_here: bool,
) -> Result<(), &'static str> {
    // Every instruction of the vector nest runs once per group, so nothing with an effect a row
    // could observe may be in it: no call (one call per group would be fewer calls), no slot, no
    // kernel, barrier or atomic.
    if matches!(
        inst.op,
        Op::Call { .. }
            | Op::VecKernelCall { .. }
            | Op::AtomicAdd { .. }
            | Op::Barrier
            | Op::Alloca { .. }
    ) {
        return Err("an instruction the nest cannot widen");
    }
    // An address is read only as the address of the load or store it feeds.
    let address = match &inst.op {
        Op::Load(p, _) => Some(*p),
        Op::Store { ptr, .. } => Some(*ptr),
        _ => None,
    };
    let mut bad_addr = false;
    each_op_use(&inst.op, &mut |u| {
        bad_addr |= w.kind(u) == Kind::Addr && Some(u) != address;
    });
    if bad_addr {
        return Err("a row-varying address is used as a value");
    }
    let mut any = false;
    each_op_use(&inst.op, &mut |u| any |= w.kind(u) != Kind::Uniform);
    let r = inst.result;
    let set = |w: &mut Walk, kind: Kind, how: Option<How>| {
        if let Some(r) = r {
            w.kinds.insert(r.0, kind);
            if let Some(h) = how {
                w.how.insert(r.0, h);
            }
        }
    };
    let vary_ty = |w: &mut Walk, t: &MirType| -> Result<(), &'static str> {
        lane_bytes(t).ok_or("a row-varying value has no vector lane type")?;
        // A bf16 or f16 value rounds to its grid at every cast, inside a row's recurrence too;
        // the inner path keeps such an accumulate scalar, and so does this one.
        if is_half(t) {
            return Err("a row-varying half-precision value");
        }
        w.vec_tys.push(t.clone());
        Ok(())
    };
    match &inst.op {
        Op::Store { ptr, value } => {
            if !stores_here {
                return Err("the nest stores inside its inner loop");
            }
            let Some(Op::Gep { index, elem, .. }) = op_of(f, *ptr) else {
                return Err("a store of the nest is not at the row's own element");
            };
            if w.kind(*ptr) != Kind::Addr || w.kind(*index) != Kind::Ramp {
                return Err("a store of the nest is not at the row's own element");
            }
            if elem != f.value_type(*value) {
                return Err("a store moves another type than its index counts in");
            }
            vary_ty(w, elem)?;
            // A store has no result; its plan is read off its address.
        }
        _ if !any => set(w, Kind::Uniform, None),
        Op::Load(p, ty) => {
            if w.kind(*p) != Kind::Addr {
                return Err("a load through a row-varying pointer");
            }
            let Some(Op::Gep { index, elem, .. }) = op_of(f, *p) else {
                return Err("a load through a row-varying pointer");
            };
            vary_ty(w, ty)?;
            let how = if w.kind(*index) == Kind::Ramp && elem == ty {
                How::Contiguous
            } else {
                How::Gather
            };
            set(w, Kind::Vary, Some(how));
        }
        Op::Gep { ptr, index, .. } => {
            if w.kind(*ptr) != Kind::Uniform || w.kind(*index) == Kind::Addr {
                return Err("an address off a row-varying base");
            }
            set(w, Kind::Addr, Some(How::Addr));
        }
        Op::Bin(op, a, b) => {
            let ty = f
                .value_type(r.ok_or("a binary op without a result")?)
                .clone();
            let (ka, kb) = (w.kind(*a), w.kind(*b));
            let ramp_step = ty == *iv_ty
                && ((*op == BinOp::Add
                    && ((ka == Kind::Ramp && kb == Kind::Uniform)
                        || (kb == Kind::Ramp && ka == Kind::Uniform)))
                    || (*op == BinOp::Sub && ka == Kind::Ramp && kb == Kind::Uniform));
            if ramp_step {
                set(w, Kind::Ramp, Some(How::RampStep));
                return Ok(());
            }
            for (k, v) in [(ka, a), (kb, b)] {
                if k == Kind::Ramp {
                    vary_ty(w, f.value_type(*v))?;
                }
            }
            vary_ty(w, &ty)?;
            let widen = match op {
                BinOp::Shl | BinOp::LShr | BinOp::AShr => kb == Kind::Uniform,
                BinOp::Mul if ty == MirType::I8 => true,
                _ => bin_widenable(*op, &ty),
            };
            set(
                w,
                Kind::Vary,
                Some(if widen { How::Widen } else { How::Replicate }),
            );
        }
        Op::Cmp(_, a, _) => {
            let operand = f.value_type(*a).clone();
            if operand == MirType::I1 {
                return Err("a compare of two masks");
            }
            vary_ty(w, &operand)?;
            set(w, Kind::Vary, Some(How::Widen));
        }
        Op::Select(c, ..) => {
            if w.kind(*c) != Kind::Vary {
                return Err("a select of row-varying values on one condition");
            }
            vary_ty(w, f.value_type(r.ok_or("a select without a result")?))?;
            set(w, Kind::Vary, Some(How::Widen));
        }
        Op::Neg(_) | Op::Not(_) | Op::Fma(..) | Op::Sqrt(_) | Op::Round(..) => {
            let ty = f.value_type(r.ok_or("an op without a result")?).clone();
            vary_ty(w, &ty)?;
            let mut ok = true;
            each_op_use(&inst.op, &mut |u| {
                if w.kind(u) == Kind::Ramp && f.value_type(u) != &ty {
                    ok = false;
                }
            });
            if !ok {
                return Err("a ramp of another type than its op");
            }
            set(w, Kind::Vary, Some(How::Widen));
        }
        Op::Cast(kind, v, to) => {
            let from = f.value_type(*v).clone();
            if from == MirType::I1 || matches!(kind, CastKind::IntToPtr | CastKind::PtrToInt) {
                return Err("a cast with no lane form");
            }
            vary_ty(w, &from)?;
            vary_ty(w, to)?;
            let how = match cast_route(*kind, &from, to) {
                Some(route) => How::Cast(route),
                None => How::Replicate,
            };
            set(w, Kind::Vary, Some(how));
        }
        _ => return Err("an instruction the nest cannot widen"),
    }
    Ok(())
}

/// Splice the vector nest in front of the original, which stays as the epilogue for the rows past
/// the last whole group:
///
/// ```text
///   preheader -> guard -> vh --(rv != vec_end)--> vp -> vih <-> vb
///                          |                            |
///                          v                            v
///                         vx -> H_o (the original)     vpost -> vh (rv + W)
/// ```
pub(super) fn apply(f: &mut Function, p: &OuterPlan) {
    let header_params: Vec<ValueId> = f.block(p.header).params.clone();
    let header_tys: Vec<MirType> = header_params
        .iter()
        .map(|v| f.value_type(*v).clone())
        .collect();
    let inner_params: Vec<ValueId> = f.block(p.inner_header).params.clone();
    // The nest's own span ([`nest_loc`]) for what no single instruction owns: the guard, the trip,
    // the counter's step and the back edges. [`Nest::inst`] moves the cursor to each reproduced
    // instruction's own span, as the inner path's `emit_body` does.
    let nest_at = nest_loc(f, p);
    let mut e = Emit::new(f, nest_at);

    let guard_params: Vec<ValueId> = header_tys.iter().map(|t| e.value(t.clone())).collect();
    let guard = e.block(guard_params.clone(), Terminator::Unreachable);
    let vh_params: Vec<ValueId> = header_tys.iter().map(|t| e.value(t.clone())).collect();
    let vh = e.block(vh_params.clone(), Terminator::Unreachable);
    let vp = e.block(Vec::new(), Terminator::Unreachable);
    // The inner header's parameters: a row-varying one as its parts, the rest as they are.
    let mut vih_params: Vec<ValueId> = Vec::new();
    let mut vih_of: Vec<Vec<ValueId>> = Vec::new();
    for (k, ip) in inner_params.iter().enumerate() {
        let ty = e.f.value_type(*ip).clone();
        // A row's accumulator is the scalar one, a lane per row: it keeps that one's line.
        e.loc = e.f.value_loc(*ip).unwrap_or(nest_at);
        let vals: Vec<ValueId> = if p.carried_vec[k] {
            (0..parts_of(&ty, p.w))
                .map(|_| e.value(part_ty(&ty)))
                .collect()
        } else {
            vec![e.value(ty)]
        };
        vih_params.extend(&vals);
        vih_of.push(vals);
    }
    e.loc = nest_at;
    let vih = e.block(vih_params, Terminator::Unreachable);
    let vb = e.block(Vec::new(), Terminator::Unreachable);
    let vpost = e.block(Vec::new(), Terminator::Unreachable);
    let vx = e.block(Vec::new(), Terminator::Unreachable);

    // ---- guard and vector header --------------------------------------------------------------
    let start = guard_params[p.iv_index];
    let n_vec = trip_count(&mut e, &p.trip, &p.iv_ty, p.w, guard, start);
    let vec_end = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, n_vec));
    e.f.blocks[guard.0 as usize].term = Terminator::Br {
        target: vh,
        args: guard_params.clone(),
    };
    let more = e.push(
        vh,
        MirType::I1,
        Op::Cmp(CmpOp::Ne, vh_params[p.iv_index], vec_end),
    );
    e.f.blocks[vh.0 as usize].term = Terminator::CondBr {
        cond: more,
        then_blk: vp,
        then_args: Vec::new(),
        else_blk: vx,
        else_args: Vec::new(),
    };
    e.f.blocks[vx.0 as usize].term = Terminator::Br {
        target: p.header,
        args: vh_params.clone(),
    };

    // ---- the body -----------------------------------------------------------------------------
    let mut s = Nest {
        p,
        map: FxHashMap::default(),
        ramps: FxHashMap::default(),
        rv: vh_params[p.iv_index],
        at: nest_at,
    };
    for (k, hp) in header_params.iter().enumerate() {
        if k != p.iv_index {
            s.map.insert(hp.0, Wide::Uniform(vh_params[k]));
        }
    }
    s.ramps.insert(p.iv.0, None);

    s.block(&mut e, p.pre, vp);
    let entry_args = term_args(e.f, p.pre);
    let entry = s.args(&mut e, vp, &entry_args, &inner_params);
    e.f.blocks[vp.0 as usize].term = Terminator::Br {
        target: vih,
        args: entry,
    };
    for (k, ip) in inner_params.iter().enumerate() {
        let wide = if p.carried_vec[k] {
            Wide::Vector(vih_of[k].clone())
        } else {
            Wide::Uniform(vih_of[k][0])
        };
        s.map.insert(ip.0, wide);
    }
    s.block(&mut e, p.inner_header, vih);
    let Terminator::CondBr { cond, then_blk, .. } = e.f.block(p.inner_header).term.clone() else {
        unreachable!("the inner header tests its trip")
    };
    let (t, el) = if then_blk == p.inner_body {
        (vb, vpost)
    } else {
        (vpost, vb)
    };
    e.f.blocks[vih.0 as usize].term = Terminator::CondBr {
        cond: resolve_uniform(&s.map, cond),
        then_blk: t,
        then_args: Vec::new(),
        else_blk: el,
        else_args: Vec::new(),
    };
    s.block(&mut e, p.inner_body, vb);
    let back_args = term_args(e.f, p.inner_body);
    let back = s.args(&mut e, vb, &back_args, &inner_params);
    e.f.blocks[vb.0 as usize].term = Terminator::Br {
        target: vih,
        args: back,
    };
    s.block(&mut e, p.post, vpost);
    e.loc = nest_at;
    let latch = term_args(e.f, p.post);
    let mut next = Vec::with_capacity(latch.len());
    for (k, a) in latch.iter().enumerate() {
        if k == p.iv_index {
            let wv = e.int(vpost, &p.iv_ty, i64::from(p.w));
            next.push(e.push(vpost, p.iv_ty.clone(), Op::Bin(BinOp::Add, s.rv, wv)));
        } else {
            next.push(resolve_uniform(&s.map, *a));
        }
    }
    e.f.blocks[vpost.0 as usize].term = Terminator::Br {
        target: vh,
        args: next,
    };

    retarget(
        &mut e.f.blocks[p.preheader.0 as usize].term,
        p.header,
        guard,
    );
}

/// The span of the source nest `p` widens: its outer header's exit test, else its counter, else
/// the first located value of the block before the inner loop. Dummy when none is located (a nest
/// the optimizer itself built), so its new values stay unknown rather than borrow another line.
fn nest_loc(f: &Function, p: &OuterPlan) -> Span {
    let header = f.block(p.header);
    let exit_test = match &header.term {
        Terminator::CondBr { cond, .. } => f.value_loc(*cond),
        _ => None,
    };
    exit_test
        .or_else(|| f.value_loc(header.params[p.iv_index]))
        .or_else(|| {
            f.block(p.pre)
                .insts
                .iter()
                .find_map(|i| i.result.and_then(|r| f.value_loc(r)))
        })
        .unwrap_or_else(Span::dummy)
}

/// The arguments a block's plain jump passes.
fn term_args(f: &Function, b: BlockId) -> Vec<ValueId> {
    match &f.block(b).term {
        Terminator::Br { args, .. } => args.clone(),
        _ => unreachable!("planned as a plain jump"),
    }
}

/// The vector nest being built.
struct Nest<'p> {
    p: &'p OuterPlan,
    /// Original value -> its counterpart.
    map: FxHashMap<u32, Wide>,
    /// Original ramp value -> its uniform offset from the counter (`None`: none).
    ramps: FxHashMap<u32, Option<ValueId>>,
    /// The group's first row.
    rv: ValueId,
    /// The nest's span ([`nest_loc`]): where an instruction with no span of its own goes.
    at: Span,
}

impl Nest<'_> {
    /// The scalar `rv + offset` of a ramp: its lane 0.
    fn ramp_base(&self, e: &mut Emit<'_>, b: BlockId, v: ValueId) -> ValueId {
        match self.ramps[&v.0] {
            None => self.rv,
            Some(off) => e.push(b, self.p.iv_ty.clone(), Op::Bin(BinOp::Add, self.rv, off)),
        }
    }

    /// A ramp as vector parts of its own type: `splat(base + q·L) + iota` for part `q`.
    fn ramp_vector(&self, e: &mut Emit<'_>, b: BlockId, v: ValueId) -> Vec<ValueId> {
        let ty = self.p.iv_ty.clone();
        let vt = part_ty(&ty);
        let per = lanes_of(&ty).expect("an integer counter has a vector width");
        let base = self.ramp_base(e, b, v);
        let iota = e.push(b, vt.clone(), Op::Iota(vt.clone()));
        (0..parts_of(&ty, self.p.w))
            .map(|q| {
                let first = if q == 0 {
                    base
                } else {
                    let off = e.int(b, &ty, (q as u32 * per) as i64);
                    e.push(b, ty.clone(), Op::Bin(BinOp::Add, base, off))
                };
                let splat = e.push(b, vt.clone(), Op::Splat(first));
                e.push(b, vt.clone(), Op::Bin(BinOp::Add, splat, iota))
            })
            .collect()
    }

    /// The ramps among `op`'s operands, put in `map` as vectors (or per-lane scalars when
    /// `lanes`) for the helpers that read `map`; returns them, to be taken out again.
    fn lend_ramps(&mut self, e: &mut Emit<'_>, b: BlockId, op: &Op, lanes: bool) -> Vec<u32> {
        let mut uses = Vec::new();
        each_op_use(op, &mut |u| uses.push(u));
        let mut lent = Vec::new();
        for u in uses {
            if self.ramps.contains_key(&u.0) && !self.map.contains_key(&u.0) {
                let wide = if lanes {
                    let base = self.ramp_base(e, b, u);
                    Wide::Lanes(iv_lane_values(e, b, base, &self.p.iv_ty, self.p.w, 1))
                } else {
                    Wide::Vector(self.ramp_vector(e, b, u))
                };
                self.map.insert(u.0, wide);
                lent.push(u.0);
            }
        }
        lent
    }

    fn give_back(&mut self, lent: Vec<u32>) {
        for v in lent {
            self.map.remove(&v);
        }
    }

    /// The lane-0 address of a contiguous access through `ptr` (`gep base, ramp`).
    fn contiguous_ptr(&self, e: &mut Emit<'_>, b: BlockId, ptr: ValueId) -> (ValueId, MirType) {
        let Some(Op::Gep {
            ptr: base,
            index,
            elem,
        }) = op_of(e.f, ptr).cloned()
        else {
            unreachable!("planned as a gep")
        };
        let base = resolve_uniform(&self.map, base);
        let at = self.ramp_base(e, b, index);
        let p = e.push(
            b,
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: at,
                elem: elem.clone(),
            },
        );
        (p, elem)
    }

    /// One address per lane through `ptr` (`gep base, index`).
    fn lane_ptrs(&mut self, e: &mut Emit<'_>, b: BlockId, ptr: ValueId) -> Vec<ValueId> {
        let Some(Op::Gep {
            ptr: base,
            index,
            elem,
        }) = op_of(e.f, ptr).cloned()
        else {
            unreachable!("planned as a gep")
        };
        let base = resolve_uniform(&self.map, base);
        let idx = if self.ramps.contains_key(&index.0) {
            let first = self.ramp_base(e, b, index);
            iv_lane_values(e, b, first, &self.p.iv_ty, self.p.w, 1)
        } else {
            lane_values(e, b, &self.map, index, self.p.w)
        };
        idx.into_iter()
            .map(|i| {
                e.push(
                    b,
                    MirType::Ptr,
                    Op::Gep {
                        ptr: base,
                        index: i,
                        elem: elem.clone(),
                    },
                )
            })
            .collect()
    }

    /// Reproduce every instruction of `from` in `to`.
    fn block(&mut self, e: &mut Emit<'_>, from: BlockId, to: BlockId) {
        let insts = e.f.block(from).insts.clone();
        for inst in &insts {
            self.inst(e, to, inst);
        }
    }

    fn inst(&mut self, e: &mut Emit<'_>, b: BlockId, inst: &Inst) {
        let w = self.p.w;
        e.loc = inst_loc(e.f, inst).unwrap_or(self.at);
        if let Op::Store { ptr, value } = &inst.op {
            let (at, elem) = self.contiguous_ptr(e, b, *ptr);
            let lent = self.lend_ramps(e, b, &inst.op, false);
            let parts = to_vector(e, b, &self.map, *value, &elem, w);
            self.give_back(lent);
            for (k, v) in parts.into_iter().enumerate() {
                store_part(e, b, at, &elem, k, v, 1);
            }
            return;
        }
        let Some(r) = inst.result else {
            unreachable!("the plan admits no other effect")
        };
        let ty = e.f.value_type(r).clone();
        match self.p.kinds.get(&r.0).copied().unwrap_or(Kind::Uniform) {
            Kind::Uniform => {
                let mut op = inst.op.clone();
                crate::map_op_uses(&mut op, |u| resolve_uniform(&self.map, u));
                let nv = e.push(b, ty, op);
                self.map.insert(r.0, Wide::Uniform(nv));
            }
            Kind::Addr => {}
            Kind::Ramp => {
                let Op::Bin(op, x, y) = inst.op else {
                    unreachable!("a ramp step is an add or a subtract")
                };
                let (ramp, other) = if self.ramps.contains_key(&x.0) {
                    (x, y)
                } else {
                    (y, x)
                };
                let other = resolve_uniform(&self.map, other);
                let off = match (self.ramps[&ramp.0], op) {
                    (None, BinOp::Add) => other,
                    (None, _) => {
                        let zero = e.int(b, &self.p.iv_ty, 0);
                        e.push(b, ty, Op::Bin(BinOp::Sub, zero, other))
                    }
                    (Some(o), op) => e.push(b, ty, Op::Bin(op, o, other)),
                };
                self.ramps.insert(r.0, Some(off));
            }
            Kind::Vary => {
                let how = self.p.how[&r.0];
                let wide = match (how, &inst.op) {
                    (How::Contiguous, Op::Load(ptr, lty)) => {
                        let (at, _) = self.contiguous_ptr(e, b, *ptr);
                        Wide::Vector(
                            (0..parts_of(lty, w))
                                .map(|k| load_part(e, b, at, lty, k, 1))
                                .collect(),
                        )
                    }
                    (How::Gather, Op::Load(ptr, lty)) => {
                        let ptrs = self.lane_ptrs(e, b, *ptr);
                        let lanes: Vec<ValueId> = ptrs
                            .into_iter()
                            .map(|q| e.push(b, lty.clone(), Op::Load(q, lty.clone())))
                            .collect();
                        Wide::Vector(pack_lanes(e, b, &lanes, lty))
                    }
                    (How::Widen, op) => {
                        let lane = match op {
                            Op::Cmp(_, x, _) => mask_lane(e.f.value_type(*x)),
                            _ => ty.clone(),
                        };
                        let lent = self.lend_ramps(e, b, op, false);
                        let parts = emit_widened(e, b, &self.map, op, w, &lane);
                        self.give_back(lent);
                        Wide::Vector(parts)
                    }
                    (How::Cast(route), Op::Cast(kind, x, to)) => {
                        let from = e.f.value_type(*x).clone();
                        let lent = self.lend_ramps(e, b, &inst.op, false);
                        let xs = to_vector(e, b, &self.map, *x, &from, w);
                        self.give_back(lent);
                        Wide::Vector(emit_cast(e, b, *kind, xs, &from, to, route))
                    }
                    (How::Replicate, op) => {
                        let lent = self.lend_ramps(e, b, op, true);
                        let mut uses = Vec::new();
                        each_op_use(op, &mut |u| uses.push(u));
                        let mut table: FxHashMap<u32, Vec<ValueId>> = FxHashMap::default();
                        for u in uses {
                            if let std::collections::hash_map::Entry::Vacant(slot) =
                                table.entry(u.0)
                            {
                                slot.insert(lane_values(e, b, &self.map, u, w));
                            }
                        }
                        self.give_back(lent);
                        let lanes = (0..w as usize)
                            .map(|k| {
                                let mut one = op.clone();
                                crate::map_op_uses(&mut one, |u| table[&u.0][k]);
                                e.push(b, ty.clone(), one)
                            })
                            .collect();
                        Wide::Lanes(lanes)
                    }
                    _ => unreachable!("planned: {how:?} for {:?}", inst.op),
                };
                self.map.insert(r.0, wide);
            }
        }
    }

    /// The arguments of a jump into the inner header: a row-varying parameter takes its value's
    /// parts, any other its scalar.
    fn args(
        &mut self,
        e: &mut Emit<'_>,
        b: BlockId,
        args: &[ValueId],
        params: &[ValueId],
    ) -> Vec<ValueId> {
        let mut out = Vec::new();
        for (k, a) in args.iter().enumerate() {
            if self.p.carried_vec[k] {
                // A row's entry splat or back-edge value is the scalar argument it stands for.
                e.loc =
                    e.f.value_loc(*a)
                        .or_else(|| e.f.value_loc(params[k]))
                        .unwrap_or(self.at);
                let ty = e.f.value_type(params[k]).clone();
                let op = Op::Splat(*a);
                let lent = self.lend_ramps(e, b, &op, false);
                out.extend(to_vector(e, b, &self.map, *a, &ty, self.p.w));
                self.give_back(lent);
            } else {
                out.push(resolve_uniform(&self.map, *a));
            }
        }
        out
    }
}
