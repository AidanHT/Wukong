//! The mid-function `@parallel` region outliner's reach past fixed arrays: **`[]T` slices**,
//! **run-time trip counts** and **kernel builtin calls** inside a region (N85). The region scan,
//! the mixed-radix proof and the outlining itself live in `lib.rs` (`try_emit_parallel_region` and
//! its section comment); this module holds what those three additions need beside them.
//!
//! **Slices.** A `[]T` capture is `(data, len)` behind a fat pointer that no region statement can
//! re-point — a capture is never assigned whole inside a region, and no admitted call receives one
//! by name — so every iteration indexes the same data, and an access `s[i]` takes the per-buffer
//! proof exactly like an array's. What the proof cannot settle is where two NAMES point. Two fixed
//! arrays the caller may bind to one buffer are one buffer at one offset (the union proof); a
//! slice may view any part of any buffer of its element scalar (`f(x[0..8], x[4..12])`), so a pair
//! with a slice in it ([`RegionRelation::Runtime`]) is tested at run time instead
//! ([`FnLowerer::emit_region_guard`]): `ptrtoint` range ends, `Gep`-measured so each backend counts
//! in its own address unit (the `kernel_guard.rs` argument), and the region is threaded only when
//! every such pair is apart. A pair whose accesses pass the union proof is also accepted when both
//! start at ONE address (`f(x, x)`, in place). Otherwise the outlined body is called directly over
//! every chunk on the calling thread, which is the serial loop — and is exactly what the
//! interpreter does for `wukong_parallel_for`, so the interpreter runs the same program on either
//! arm.
//!
//! **Run-time trip counts.** With `N` unknown there is no modulus, so no index term can be "outer":
//! every term must be bounded and the whole index of iteration `hh` must lie in `[hh·C, hh·C + C)`
//! (`region_extent` with `m = None`). [`RegionTrip`] carries the distinction to the emitter.
//!
//! **Kernel builtins.** `softmax`, `matmul`, `rmsnorm`, `gelu(x, out)`, `sum`, … are admitted in a
//! region as exactly the accesses of their operands ([`FnLowerer::scan_region_builtin`]): sema's
//! shape record and the lowering's run-time extent checks (`builtins.rs`) pin each buffer operand to
//! the elements its shape names, the kernel reads its inputs and writes only its output, and the
//! outlined body calls the SERIAL kernel (the region supplies the threading). A range view
//! `s[lo..hi]` with a constant `hi - lo = L` is the range access `s[lo + j]`, `j ∈ [0, L)` — a
//! bounded term like an inner `for` variable — so a row kernel on row `r` (`out[r*C..r*C + C]`)
//! passes the proof as the iteration's own partition. A whole captured buffer is an opaque access:
//! fine to read, never a write the proof can place. Any other operand form declines.

use std::collections::BTreeMap;

use wukong_ast::{Expr, ExprKind};
use wukong_mir::{BinOp, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_sema::{builtin_operand_dims, builtin_output, BuiltinCall, BuiltinDim, KernelBuiltin};
use wukong_span::{FxHashMap as HashMap, Symbol};
use wukong_types::Ty;

use crate::{noalias, single_path, FnLowerer, RegionRest, RegionScan};

/// A region loop's trip count: a compile-time constant, or a pure bound of a signed counter that
/// the call site evaluates once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RegionTrip {
    Const(i64),
    Runtime,
}

impl RegionTrip {
    pub(crate) fn constant(self) -> Option<i64> {
        match self {
            RegionTrip::Const(n) => Some(n),
            RegionTrip::Runtime => None,
        }
    }
}

/// What `match_parallel_region` proved: the captures in first-appearance order (the env layout),
/// and the pairs of captured buffers only a run-time range test can separate.
pub(crate) struct RegionPlan {
    pub(crate) captures: Vec<Symbol>,
    pub(crate) guards: Vec<RegionGuard>,
    /// The sema type of every guarded capture (a slice's element type sizes its range).
    pub(crate) tys: HashMap<Symbol, Ty>,
}

/// Two captures that may share storage at an offset no name can tell (at least one is a slice),
/// at least one of them written or atomically updated in the region. Threading needs their element
/// ranges disjoint — or, with `same_base_ok`, starting at one address, which is what the union of
/// their accesses was proved for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct RegionGuard {
    pub(crate) a: Symbol,
    pub(crate) b: Symbol,
    pub(crate) same_base_ok: bool,
}

/// Add the guard `{a, b}` once; when the pair is already there the stricter verdict wins.
pub(crate) fn push_guard(guards: &mut Vec<RegionGuard>, a: Symbol, b: Symbol, same_base_ok: bool) {
    if let Some(g) = guards
        .iter_mut()
        .find(|g| (g.a, g.b) == (a, b) || (g.a, g.b) == (b, a))
    {
        g.same_base_ok &= same_base_ok;
        return;
    }
    guards.push(RegionGuard { a, b, same_base_ok });
}

/// How two distinct captured buffers' storage relates, as far as their names and types tell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RegionRelation {
    /// Never the same storage: different element scalars, two different allocas, a fresh alloca
    /// and a parameter, or parameters a `@noalias` promise separates.
    Distinct,
    /// One buffer at one offset, possibly: the same binding under two names, or fixed-array
    /// parameters the caller may bind to one array. The union of their accesses must pass.
    Union,
    /// May overlap at an offset only the run time knows: a slice is involved.
    Runtime,
}

/// The constant length `hi - lo` of a range access whose two ends decomposed to `lo_rest` and
/// `hi_rest` under one digit signature: their variable terms must agree exactly and their constants
/// differ by a positive amount. `None` otherwise (a run-time length the proof cannot bound).
pub(crate) fn region_span(lo_rest: &[RegionRest], hi_rest: &[RegionRest]) -> Option<i128> {
    fn fold(rest: &[RegionRest]) -> Option<(i128, Vec<(Symbol, i128)>)> {
        let mut lit: i128 = 0;
        let mut vars: BTreeMap<Symbol, i128> = BTreeMap::new();
        for term in rest {
            match *term {
                RegionRest::Lit(k) => lit = lit.checked_add(k)?,
                RegionRest::Var(v, s) => {
                    let e = vars.entry(v).or_insert(0);
                    *e = e.checked_add(s)?;
                }
                RegionRest::Span(_) => return None,
            }
        }
        Some((lit, vars.into_iter().filter(|(_, s)| *s != 0).collect()))
    }
    let (lo, lo_vars) = fold(lo_rest)?;
    let (hi, hi_vars) = fold(hi_rest)?;
    if lo_vars != hi_vars {
        return None;
    }
    let len = hi.checked_sub(lo)?;
    (len >= 1).then_some(len)
}

impl FnLowerer<'_> {
    /// The sema type of a captured buffer: a parameter's declared type, else the type an access in
    /// the region spelled it with.
    pub(crate) fn region_capture_ty<'s>(
        &'s self,
        sym: Symbol,
        sc: &'s RegionScan<'_>,
    ) -> Option<&'s Ty> {
        self.buffer_param_ty(sym).or_else(|| sc.cap_tys.get(&sym))
    }

    /// The storage relation of two distinct captured buffers `a` and `b` (see [`RegionRelation`]).
    pub(crate) fn region_relation(
        &self,
        a: Symbol,
        b: Symbol,
        sc: &RegionScan<'_>,
    ) -> RegionRelation {
        let (Some((va, _)), Some((vb, _))) = (self.lookup(a), self.lookup(b)) else {
            return RegionRelation::Union;
        };
        if va == vb {
            return RegionRelation::Union;
        }
        let (ta, tb) = (self.region_capture_ty(a, sc), self.region_capture_ty(b, sc));
        let is_slice = |v: ValueId, t: Option<&Ty>| {
            self.slice_slots.contains(&v) || matches!(t, Some(Ty::Slice(_)))
        };
        if is_slice(va, ta) || is_slice(vb, tb) {
            // The element scalar is the one fact that rules sharing out (`noalias.rs`): Wukong has
            // no reinterpreting view. A slice NAME is never trusted further, `@noalias` included
            // (`promised_disjoint` refuses slices: a callee can re-point a fat pointer).
            return match (ta, tb) {
                (Some(x), Some(y)) if !noalias::could_share_storage(x, y) => {
                    RegionRelation::Distinct
                }
                _ => RegionRelation::Runtime,
            };
        }
        // Two fixed arrays: parameters over one element scalar may be one caller buffer (P1.20)
        // unless promised apart; a fresh alloca is nobody else's storage.
        match (self.buffer_param_ty(a), self.buffer_param_ty(b)) {
            (Some(x), Some(y))
                if noalias::could_share_storage(x, y) && !self.promised_disjoint(a, b) =>
            {
                RegionRelation::Union
            }
            _ => RegionRelation::Distinct,
        }
    }

    /// Do `a` and `b` index the same element type, a scalar — the condition under which one start
    /// address makes their indices the same coordinates, so the union proof speaks about them?
    pub(crate) fn region_same_elements(&self, a: Symbol, b: Symbol, sc: &RegionScan<'_>) -> bool {
        let scalar = |t: Option<&Ty>| match t {
            Some(Ty::Slice(e)) | Some(Ty::Array { elem: e, .. }) => match e.as_ref() {
                Ty::Scalar(s) => Some(*s),
                _ => None,
            },
            _ => None,
        };
        matches!(
            (scalar(self.region_capture_ty(a, sc)), scalar(self.region_capture_ty(b, sc))),
            (Some(x), Some(y)) if x == y
        )
    }

    /// A kernel builtin call in a region body: each buffer operand is an access of its own
    /// ([`Self::scan_region_operand`]) — the output a write, the others reads — and each scalar
    /// operand an ordinary read. A dimension held by a binding (a turbofish variable, a generic
    /// dimension's hidden parameter) is read by the lowering through that binding, so it must be
    /// captured like any other read; a dimension derived from a tensor shape declines (tensor
    /// operands do anyway).
    pub(crate) fn scan_region_builtin<'e>(
        &self,
        call: &BuiltinCall,
        args: &'e [Expr],
        sc: &mut RegionScan<'e>,
    ) {
        for dim in &call.dims {
            match dim {
                BuiltinDim::Const(_) | BuiltinDim::SliceLen { .. } => {}
                &BuiltinDim::Named { name, .. } => {
                    if name == sc.hh || sc.is_local(name) {
                        continue;
                    }
                    if self.lookup(name).is_none() || !self.region_capture(name, sc) {
                        sc.decline("a kernel-builtin dimension with no capturable binding");
                        return;
                    }
                }
                BuiltinDim::Derived(_) => {
                    sc.decline("a kernel-builtin dimension derived from a tensor shape");
                    return;
                }
            }
        }
        let kind = call.kind;
        let out = builtin_output(kind);
        let dims = builtin_operand_dims(kind);
        for (i, a) in args.iter().enumerate() {
            // `sdpa(q, k, v, out, s, d, scale, causal)` carries its extents as values.
            let buffer = if kind == KernelBuiltin::Sdpa {
                i < 4
            } else {
                dims.get(i).is_some_and(Option::is_some)
            };
            if buffer {
                self.scan_region_operand(a, out == Some(i), sc);
            } else {
                self.scan_region_expr(a, sc);
            }
            if !sc.ok {
                return;
            }
        }
    }

    /// One buffer operand of a kernel builtin. The lowering pins it to exactly the elements its
    /// shape names (a slice of any other length traps), so the kernel touches the operand's whole
    /// extent and nothing else:
    /// * a range view `s[lo..hi]` of a captured slice is the range access `[lo, hi)` (its length
    ///   must be constant for the proof to place it; `s[lo..]` runs to the run-time length and is
    ///   opaque);
    /// * a captured array or slice named whole is an opaque access;
    /// * a body-local array is the iteration's own storage.
    fn scan_region_operand<'e>(&self, a: &'e Expr, write: bool, sc: &mut RegionScan<'e>) {
        if let ExprKind::Call { callee, args, .. } = &a.kind {
            if self.sema.is_range_slice_call(callee) {
                for x in args.iter().skip(1) {
                    self.scan_region_expr(x, sc);
                }
                if !sc.ok {
                    return;
                }
                let Some((base, bsym)) = args.first().and_then(|b| Some((b, single_path(b)?)))
                else {
                    sc.decline("a range view of a computed base");
                    return;
                };
                if bsym == sc.hh || sc.is_local(bsym) || self.lookup(bsym).is_none() {
                    sc.decline("a range view of a name that is not a captured slice");
                    return;
                }
                if !self.region_index_capture(bsym, base, sc) {
                    return;
                }
                let (lo, hi) = match args.len() {
                    3 => (Some(&args[1]), Some(&args[2])),
                    _ => (None, None),
                };
                sc.record(bsym, lo, hi, write);
                return;
            }
        }
        let Some(bsym) = single_path(a) else {
            sc.decline("a kernel operand that is not a name or a range view");
            return;
        };
        let ty = self.expr_ty(a);
        if bsym == sc.hh || sc.is_local(bsym) {
            if !(matches!(ty, Ty::Array { .. }) && self.region_value_is_owned(&ty)) {
                sc.decline("a kernel operand body-local that views shared storage");
            }
            return;
        }
        if self.lookup(bsym).is_none() {
            sc.decline("a kernel operand that is not a runtime binding");
            return;
        }
        if !self.region_index_capture(bsym, a, sc) {
            return;
        }
        sc.record(bsym, None, None, write);
    }

    /// `s.len()` on a captured slice: a read of the fat pointer, which no region statement
    /// writes. Registers the capture (the outlined body reads the length through it) and returns
    /// `true`; `false` for any other call, which the caller then classifies.
    pub(crate) fn region_slice_len<'e>(
        &self,
        callee: &'e Expr,
        args: &'e [Expr],
        sc: &mut RegionScan<'e>,
    ) -> bool {
        let ExprKind::Field { base, name } = &callee.kind else {
            return false;
        };
        if !args.is_empty() || self.interner.resolve(name.sym) != "len" {
            return false;
        }
        let Some(bsym) = single_path(base) else {
            return false;
        };
        let ty = self.expr_ty(base);
        if !matches!(ty, Ty::Slice(_))
            || bsym == sc.hh
            || sc.is_local(bsym)
            || self.lookup(bsym).is_none()
        {
            return false;
        }
        sc.note_ty(bsym, ty);
        if !self.region_capture(bsym, sc) {
            sc.decline("the length of a slice whose storage cannot be identified");
        }
        true
    }

    /// The run-time test of a region's guards: `true` when every guarded pair's element ranges are
    /// disjoint (or start at one address, where that was proved enough). Each range is
    /// `[ptrtoint(base), ptrtoint(gep(base, len)))` — a slice's `(data, len)` from its fat pointer,
    /// an array's base and declared length — so each backend measures it in its own address unit
    /// and both reach the same verdict (`kernel_guard.rs`).
    pub(crate) fn emit_region_guard(&mut self, plan: &RegionPlan) -> ValueId {
        let mut ranges: HashMap<Symbol, (ValueId, ValueId)> = HashMap::default();
        let mut ok: Option<ValueId> = None;
        for g in &plan.guards {
            let (alo, ahi) = self.region_range(g.a, plan, &mut ranges);
            let (blo, bhi) = self.region_range(g.b, plan, &mut ranges);
            let a_below = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ule, ahi, blo));
            let b_below = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ule, bhi, alo));
            let mut apart = self
                .builder
                .build(MirType::I1, Op::Bin(BinOp::Or, a_below, b_below));
            if g.same_base_ok {
                let same = self
                    .builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Eq, alo, blo));
                apart = self
                    .builder
                    .build(MirType::I1, Op::Bin(BinOp::Or, apart, same));
            }
            ok = Some(match ok {
                None => apart,
                Some(prev) => self
                    .builder
                    .build(MirType::I1, Op::Bin(BinOp::And, prev, apart)),
            });
        }
        ok.unwrap_or_else(|| {
            self.builder
                .build(MirType::I1, Op::ConstInt(1, MirType::I1))
        })
    }

    /// The `[lo, hi)` address range (as `i64`) of captured buffer `sym`, computed once per guard.
    fn region_range(
        &mut self,
        sym: Symbol,
        plan: &RegionPlan,
        ranges: &mut HashMap<Symbol, (ValueId, ValueId)>,
    ) -> (ValueId, ValueId) {
        if let Some(r) = ranges.get(&sym) {
            return *r;
        }
        let (v, mty) = self.lookup(sym).expect("a guarded capture is in scope");
        let span = if self.slice_slots.contains(&v) {
            match plan.tys.get(&sym) {
                Some(Ty::Slice(elem)) => {
                    let elem = self.mir_ty_of(elem);
                    let (data, len) = self.text_slice_parts(v);
                    Some((data, len, elem))
                }
                _ => None,
            }
        } else {
            match mty {
                MirType::Array(elem, n) => {
                    let len = self
                        .builder
                        .build(MirType::I64, Op::ConstInt(i128::from(n), MirType::I64));
                    Some((v, len, *elem))
                }
                _ => None,
            }
        };
        let range = match span {
            Some((base, len, elem)) => {
                let lo = self.builder.build(
                    MirType::I64,
                    Op::Cast(CastKind::PtrToInt, base, MirType::I64),
                );
                let end = self.builder.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: base,
                        index: len,
                        elem,
                    },
                );
                let hi = self.builder.build(
                    MirType::I64,
                    Op::Cast(CastKind::PtrToInt, end, MirType::I64),
                );
                (lo, hi)
            }
            // No measurable extent (never produced by the scan, which admits only arrays and
            // scalar slices): the whole address space, so the pair always runs serially.
            None => {
                let lo = self
                    .builder
                    .build(MirType::I64, Op::ConstInt(0, MirType::I64));
                let hi = self
                    .builder
                    .build(MirType::I64, Op::ConstInt(-1, MirType::I64));
                (lo, hi)
            }
        };
        ranges.insert(sym, range);
        range
    }
}

#[cfg(test)]
thread_local! {
    /// Every region verdict `try_emit_parallel_region` reached on this thread, in order: `Ok` for
    /// a loop the proof admitted, `Err(reason)` for one that stays serial. Read by the decline
    /// census (`par_region_tests`).
    static TRACE: std::cell::RefCell<Vec<Result<(), &'static str>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn trace(verdict: Result<(), &'static str>) {
    TRACE.with(|t| t.borrow_mut().push(verdict));
}

/// Drain this thread's region-verdict trace.
#[cfg(test)]
pub(crate) fn take_trace() -> Vec<Result<(), &'static str>> {
    TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}
