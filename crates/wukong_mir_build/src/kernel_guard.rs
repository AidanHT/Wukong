//! **Run-time alias versioning of recognized kernel dispatches.**
//!
//! A recognized kernel equals the loop nest it replaces only under a disjointness precondition the
//! recognizer cannot always establish from inside one function: `fn mm(a, b, mut c)` called as
//! `mm(x, y, x)` turns the tiled `wukong_sgemm` into a different program. `alias.rs` answers the
//! compile-time half ([`crate::alias::plan_kernel_alias`]: `Disjoint`, `MustAlias`, `MayAlias` per
//! operand pair). This module is the run-time half, for the `MayAlias` pairs:
//!
//! ```text
//!   lo_i = ptrtoint(base_i)          hi_i = ptrtoint(gep(base_i, elems_i))
//!   ok   = AND over checked pairs of  (hi_a <= lo_b || hi_b <= lo_a)          -- Disjoint
//!                                  or (... || (lo_a == lo_b && hi_a == hi_b))  -- ExactInPlace
//!   cond_br ok, kernel, fallback
//!   kernel:   call wukong_*(...)                     ; br join
//!   fallback: <the ORIGINAL source loop, recognition and @parallel off> ; br join
//! ```
//!
//! That is LLVM's `LoopVersioning` / Polly run-time alias check, and it replaces two weaker forms
//! this compiler used before:
//!
//! * a **compile-time decline** whenever disjointness was not provable, which took a library
//!   routine off its kernel for every call once ANY call site aliased it; and
//! * `emit_nonoverlapping_call`'s **serial-kernel fallback** for the embedding gather and the
//!   scatter-add, whose may-overlap arm called a SERIAL kernel whose own `# Safety` clause also says
//!   "non-overlapping" and which never range-tested the third operand (`ids`). The fallback here is
//!   always the source loop, which is the program's meaning by definition, and every operand the
//!   contract constrains is range-tested.
//!
//! **Why the test is exact on both backends.** The kernel and the source loop are not always
//! bit-identical (GEMM reassociates), so the interpreter and the native backend must take the SAME
//! arm for the same program, or the differential gate would compare two different programs. The
//! extent end is therefore a `Gep`, not a hand-scaled byte count: the interpreter strides a `Gep`
//! by its slot footprint and Cranelift by the element's byte size, so each backend measures each
//! range in its own address unit. Two operands whose extents stay inside their own buffers then
//! overlap in slots exactly when they overlap in bytes, and both backends agree on the verdict. (A
//! hand-scaled `* 4` — `emit_nonoverlapping_call`'s form — over-reports overlap on the interpreter's
//! slot memory, which was harmless only because both of its arms ran the same interpreter code.)
//!
//! **The protocol.** An emitter computes its base pointers and extents (pure MIR), then calls
//! [`FnLowerer::open_kernel_guard`] immediately before its kernel call(s) and after every decline
//! it can still take. Under a versioning context ([`FnLowerer::emit_alias_versioned`] — reached
//! through `emit_versioned_loop` for the per-loop cascade and the elementwise kernel probes of
//! `try_vectorize_for`, `emit_versioned_stmts` for the statement windows, and
//! `emit_nonempty_reduction` for the reductions) a `MayAlias` plan opens the guard and leaves the
//! builder in the kernel arm with the fallback pending; with no context — a probe, an outlined
//! `@parallel` chunk (whose body starts with `KernelGuard::Off`) — a `MayAlias` plan declines exactly
//! as every emitter did before, so no caller is ever handed a half-built branch it does not know how
//! to finish.

use wukong_ast::{Block, Expr, ExprKind, ForIter, Pattern, Stmt, StmtKind, UnOp};
use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_span::Symbol;

use crate::alias::{plan_kernel_alias, KernelAliasPlan, OperandPair, PairContract};
use crate::{buffer_base, single_path, Dim, FnLowerer, LowpMatmulNest, Pool2dNest};

/// The alias-versioning state of the dispatch currently being emitted. Rides on `GemmSyms` beside
/// the `recognize` switch, for the reason that struct's doc gives: it is already threaded to every
/// emitter, and it is toggled around a single emitter call exactly the way `recognize` is toggled
/// around a fallback lowering.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelGuard {
    /// No versioning context: a `MayAlias` dispatch declines (the pre-versioning behavior).
    Off,
    /// [`FnLowerer::emit_alias_versioned`] is around this emitter and can lower its source on the
    /// may-overlap arm.
    Ready,
    /// A guard is open: the builder is in the kernel arm, and this block still needs the source
    /// loop lowered into it.
    Pending(BlockId),
}

/// One buffer operand of a guarded kernel dispatch.
pub(crate) struct GuardOperand {
    /// The operand's symbol, as the recognizer named it (what the compile-time classifier compares).
    pub(crate) sym: Symbol,
    /// The pointer the kernel is handed (after any batch/head offset).
    pub(crate) base: ValueId,
    /// How many `elem`s from `base` the kernel touches — exactly, never a superset: an extent that
    /// ran past its own buffer could make the two backends disagree about an overlap.
    pub(crate) elems: ValueId,
    /// The element type, so the extent's end is a `Gep` each backend scales in its own address unit.
    pub(crate) elem: MirType,
}

impl GuardOperand {
    pub(crate) fn new(sym: Symbol, base: ValueId, elems: ValueId, elem: MirType) -> Self {
        GuardOperand {
            sym,
            base,
            elems,
            elem,
        }
    }
}

impl FnLowerer<'_> {
    /// `a * b` as an `i64` extent.
    pub(super) fn guard_extent(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.builder.build(MirType::I64, Op::Bin(BinOp::Mul, a, b))
    }

    /// Open the alias guard for the kernel call(s) about to be emitted, or report that the kernel
    /// may not be emitted at all.
    ///
    /// * `true`, no MIR — every constrained pair is compile-time safe; emit the kernel as usual.
    /// * `true`, guard opened — some pair may alias and a versioning context is active: the
    ///   overlap test is built, the builder now sits in the kernel arm, and the source loop will be
    ///   lowered on the other arm by the context. The caller MUST emit its kernel call(s) and
    ///   return `true`; it must not decline after this point.
    /// * `false` — some pair is the same buffer under a `Disjoint` contract, or may alias with no
    ///   versioning context to lower the source. The caller declines and the source loop lowers.
    pub(super) fn open_kernel_guard(
        &mut self,
        ops: &[GuardOperand],
        pairs: &[OperandPair],
    ) -> bool {
        let syms: Vec<Symbol> = ops.iter().map(|o| o.sym).collect();
        let checks = match plan_kernel_alias(self.gemm.alias_free, &syms, pairs) {
            KernelAliasPlan::Dispatch => return true,
            KernelAliasPlan::Decline => return false,
            KernelAliasPlan::Version(checks) => checks,
        };
        // A `@noalias` promise settles a pair the call-site bit cannot (`promised_disjoint`): the
        // source states it and sema rejects the call sites that provably break it, so that pair
        // needs no run-time test. Only when every remaining pair is promised does the kernel
        // dispatch unguarded, as `all_provably_disjoint` answers the compile-time question.
        let checks: Vec<OperandPair> = checks
            .into_iter()
            .filter(|p| !self.promised_disjoint(syms[p.a], syms[p.b]))
            .collect();
        if checks.is_empty() {
            return true;
        }
        if self.gemm.guard != KernelGuard::Ready {
            return false;
        }
        let mut ranges: Vec<Option<(ValueId, ValueId)>> = vec![None; ops.len()];
        let mut ok: Option<ValueId> = None;
        for pair in checks {
            let (alo, ahi) = self.guard_range(ops, &mut ranges, pair.a);
            let (blo, bhi) = self.guard_range(ops, &mut ranges, pair.b);
            let a_below = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ule, ahi, blo));
            let b_below = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ule, bhi, alo));
            let mut pair_ok = self
                .builder
                .build(MirType::I1, Op::Bin(BinOp::Or, a_below, b_below));
            if pair.contract == PairContract::ExactInPlace {
                let same_lo = self
                    .builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Eq, alo, blo));
                let same_hi = self
                    .builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Eq, ahi, bhi));
                let exact = self
                    .builder
                    .build(MirType::I1, Op::Bin(BinOp::And, same_lo, same_hi));
                pair_ok = self
                    .builder
                    .build(MirType::I1, Op::Bin(BinOp::Or, pair_ok, exact));
            }
            ok = Some(match ok {
                None => pair_ok,
                Some(prev) => self
                    .builder
                    .build(MirType::I1, Op::Bin(BinOp::And, prev, pair_ok)),
            });
        }
        // `Version` is never empty (`plan_kernel_alias` answers `Dispatch` for that), so `ok` is set.
        let Some(ok) = ok else {
            return true;
        };
        let kernel = self.builder.new_block();
        let fallback = self.builder.new_block();
        self.builder.cond_br(ok, kernel, vec![], fallback, vec![]);
        self.builder.switch_to(kernel);
        self.gemm.guard = KernelGuard::Pending(fallback);
        true
    }

    /// The guard of a bf16/f16 GEMM (`wukong_sgemm_{bf16,f16}_{nt,tn}[_epi]`): half-width `A`
    /// (`m*k`) and `B` (`k*n`), an f32 `C` (`m*n`) that must be disjoint from both, and the f32
    /// epilogue bias (`n`) when there is one.
    pub(super) fn open_lowp_gemm_guard(
        &mut self,
        nest: &LowpMatmulNest,
        [a, b, c]: [ValueId; 3],
        [m, k, n]: [ValueId; 3],
        bias: Option<(Symbol, ValueId)>,
    ) -> bool {
        let half = if nest.f16 {
            MirType::F16
        } else {
            MirType::BF16
        };
        let a_elems = self.guard_extent(m, k);
        let b_elems = self.guard_extent(k, n);
        let c_elems = self.guard_extent(m, n);
        let mut ops = vec![
            GuardOperand::new(nest.a, a, a_elems, half.clone()),
            GuardOperand::new(nest.b, b, b_elems, half),
            GuardOperand::new(nest.c, c, c_elems, MirType::F32),
        ];
        let mut pairs = vec![OperandPair::disjoint(0, 2), OperandPair::disjoint(1, 2)];
        if let Some((sym, ptr)) = bias {
            ops.push(GuardOperand::new(sym, ptr, n, MirType::F32));
            pairs.push(OperandPair::disjoint(3, 2));
        }
        self.open_kernel_guard(&ops, &pairs)
    }

    /// The guard of a two-operand reduction: an f32 input `x` of `rows * cols` and an output of
    /// `out_elems` `out_elem`s (a column or row result, f32 or an `i32` index) that must be disjoint
    /// from it. The column reductions, the arg-reductions, log-sum-exp and entropy.
    pub(super) fn open_reduce_guard(
        &mut self,
        (x_sym, x): (Symbol, ValueId),
        (out_sym, out): (Symbol, ValueId),
        rows: ValueId,
        cols: ValueId,
        out_elems: ValueId,
        out_elem: MirType,
    ) -> bool {
        let x_elems = self.guard_extent(rows, cols);
        let ops = [
            GuardOperand::new(x_sym, x, x_elems, MirType::F32),
            GuardOperand::new(out_sym, out, out_elems, out_elem),
        ];
        self.open_kernel_guard(&ops, &[OperandPair::disjoint(0, 1)])
    }

    /// The guard of a two-input per-row loss (`wukong_kldiv_f32`, `wukong_kd_loss_f32`): two f32
    /// `rows * cols` inputs and an f32 `rows` output disjoint from both.
    pub(super) fn open_loss_guard(
        &mut self,
        [(p_sym, p), (q_sym, q)]: [(Symbol, ValueId); 2],
        (out_sym, out): (Symbol, ValueId),
        rows: ValueId,
        cols: ValueId,
    ) -> bool {
        let n = self.guard_extent(rows, cols);
        let ops = [
            GuardOperand::new(p_sym, p, n, MirType::F32),
            GuardOperand::new(q_sym, q, n, MirType::F32),
            GuardOperand::new(out_sym, out, rows, MirType::F32),
        ];
        self.open_kernel_guard(
            &ops,
            &[OperandPair::disjoint(0, 2), OperandPair::disjoint(1, 2)],
        )
    }

    /// The guard of a row-local scan (cumsum, cumprod, cummax/cummin, lrscan): every input and the
    /// output are f32 `rows * cols`, and the output may be EXACTLY any input (the runtime's
    /// documented in-place spelling) but must not otherwise overlap it.
    pub(super) fn open_scan_guard(
        &mut self,
        inputs: &[(Symbol, ValueId)],
        (out_sym, out): (Symbol, ValueId),
        rows: ValueId,
        cols: ValueId,
    ) -> bool {
        let n = self.guard_extent(rows, cols);
        let mut ops: Vec<GuardOperand> = inputs
            .iter()
            .map(|&(sym, ptr)| GuardOperand::new(sym, ptr, n, MirType::F32))
            .collect();
        let out_index = ops.len();
        ops.push(GuardOperand::new(out_sym, out, n, MirType::F32));
        let pairs: Vec<OperandPair> = (0..out_index)
            .map(|i| OperandPair::exact_in_place(i, out_index))
            .collect();
        self.open_kernel_guard(&ops, &pairs)
    }

    /// The guard of an ELEMENTWISE dispatch (`vmath`, `vmath2`, `velem`, `vhorner`, the int dequant,
    /// the half-input axpby): `ops` are every operand of every kernel call the dispatch emits — each
    /// already offset to the loop start, with the trip count as its extent in its own element — and
    /// `outs` the indices of the outputs among them.
    ///
    /// Each output may be EXACTLY any operand of its own element type: that is the in-place map, which
    /// an 8-lane body computes exactly as the source loop does. It must otherwise be disjoint from
    /// every operand of every call. A SHIFTED overlap makes the vector body read lanes the source
    /// loop has already overwritten; and where a multi-statement body becomes one full-range pass per
    /// statement, a later pass would read (or overwrite) what an earlier pass wrote ahead of the
    /// loop's own interleaved order. A mixed-width pair is always `Disjoint`: "exactly in place" is not
    /// something two element sizes can share, and the two backends measure such a pair's ranges in
    /// different units. Pairs between two inputs are nobody's business.
    pub(super) fn open_elementwise_guard(&mut self, ops: &[GuardOperand], outs: &[usize]) -> bool {
        let mut pairs = Vec::new();
        for &o in outs {
            for (p, op) in ops.iter().enumerate() {
                // An output-output pair is listed once, while the earlier output is `o`.
                if p == o || (p < o && outs.contains(&p)) {
                    continue;
                }
                pairs.push(if op.elem == ops[o].elem {
                    OperandPair::exact_in_place(p, o)
                } else {
                    OperandPair::disjoint(p, o)
                });
            }
        }
        self.open_kernel_guard(ops, &pairs)
    }

    /// The guard of a normalization backward (`wukong_{rms,layer}norm_bwd_f32`): `x`, `dy` and the
    /// output `dx` are f32 `rows * cols`, `gamma` is f32 `cols`. `dx` must be disjoint from `x` and
    /// `gamma`; over `dy` it may be EXACTLY in place when `dy_in_place` (RMSNorm) and must be
    /// disjoint otherwise (LayerNorm).
    pub(super) fn open_norm_bwd_guard(
        &mut self,
        [(x_sym, x), (dy_sym, dy), (g_sym, gamma)]: [(Symbol, ValueId); 3],
        (dx_sym, dx): (Symbol, ValueId),
        rows: ValueId,
        cols: ValueId,
        dy_in_place: bool,
    ) -> bool {
        let n = self.guard_extent(rows, cols);
        let ops = [
            GuardOperand::new(x_sym, x, n, MirType::F32),
            GuardOperand::new(dy_sym, dy, n, MirType::F32),
            GuardOperand::new(g_sym, gamma, cols, MirType::F32),
            GuardOperand::new(dx_sym, dx, n, MirType::F32),
        ];
        let dy_pair = if dy_in_place {
            OperandPair::exact_in_place(1, 3)
        } else {
            OperandPair::disjoint(1, 3)
        };
        self.open_kernel_guard(
            &ops,
            &[
                OperandPair::disjoint(0, 3),
                dy_pair,
                OperandPair::disjoint(2, 3),
            ],
        )
    }

    /// Are the memory reads an elementwise recognizer HOISTS out of its loop — a coefficient
    /// `s[0]` or `*p`, evaluated ONCE before the kernel call where the source loop re-reads it on
    /// every iteration — from storage provably disjoint from every buffer the kernel writes
    /// (`outs`)?
    ///
    /// The run-time guard ranges only the kernel's BUFFER operands. A hoisted coefficient reaches
    /// the kernel as a scalar, so it is in no range: an `out` that overlaps the element it was read
    /// from changes what the source loop reads after the first store, while the kernel keeps the
    /// pre-loop value. `out[j] = out[0] * x[j]` is the same-symbol spelling (the source chains each
    /// store into `out[0]`'s later reads; the kernel scales every lane by the old `out[0]`), and
    /// `fn f(s: []f32, x: []f32, mut out: []f32)` called `f(buf, x, buf)` is the two-name one.
    /// Neither is visible to the differential gates: the interpreter marshals the same hoisted
    /// scalar. A read whose base this cannot name answers `false`. Pure; the caller declines
    /// BEFORE opening any guard, and the loop then lowers with the generic vectorizer's own
    /// dependence test, or scalar.
    pub(super) fn hoisted_reads_disjoint(&self, exprs: &[&Expr], outs: &[Symbol]) -> bool {
        exprs.iter().all(|e| self.hoisted_read_disjoint(e, outs))
    }

    fn hoisted_read_disjoint(&self, e: &Expr, outs: &[Symbol]) -> bool {
        let base_disjoint = |base: Option<Symbol>| {
            base.is_some_and(|b| outs.iter().all(|&o| self.provably_disjoint(b, o)))
        };
        match &e.kind {
            ExprKind::Index { base, indices } => {
                base_disjoint(buffer_base(base, self.interner))
                    && indices.iter().all(|i| self.hoisted_read_disjoint(i, outs))
            }
            ExprKind::Unary {
                op: UnOp::Deref,
                expr,
            } => base_disjoint(single_path(expr)),
            ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => {
                self.hoisted_read_disjoint(expr, outs)
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.hoisted_read_disjoint(lhs, outs) && self.hoisted_read_disjoint(rhs, outs)
            }
            ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
                self.hoisted_read_disjoint(base, outs)
            }
            ExprKind::ArrayLit(items) => items.iter().all(|i| self.hoisted_read_disjoint(i, outs)),
            ExprKind::ArrayRepeat { value, count } => {
                self.hoisted_read_disjoint(value, outs) && self.hoisted_read_disjoint(count, outs)
            }
            ExprKind::Call { args, .. } => args.iter().all(|a| self.hoisted_read_disjoint(a, outs)),
            // A path or a literal reads no buffer element. Every other form is refused by
            // `coeff_is_hoistable` before a coefficient gets here; answering `false` keeps a future
            // widening of that list from becoming a silent hole.
            ExprKind::Path(_)
            | ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_) => true,
            _ => false,
        }
    }

    /// `[ptrtoint(base), ptrtoint(gep(base, elems)))` for operand `i`, built once per operand.
    fn guard_range(
        &mut self,
        ops: &[GuardOperand],
        ranges: &mut [Option<(ValueId, ValueId)>],
        i: usize,
    ) -> (ValueId, ValueId) {
        if let Some(r) = ranges[i] {
            return r;
        }
        let op = &ops[i];
        let lo = self.builder.build(
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, op.base, MirType::I64),
        );
        let end = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: op.base,
                index: op.elems,
                elem: op.elem.clone(),
            },
        );
        let hi = self.builder.build(
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, end, MirType::I64),
        );
        ranges[i] = Some((lo, hi));
        (lo, hi)
    }

    /// [`Self::emit_alias_versioned`] for a per-loop recognizer of the `lower_for` cascade whose
    /// emitter is not already behind `emit_nonempty_reduction_loop`: the source is the loop itself.
    pub(super) fn emit_versioned_loop(
        &mut self,
        (label, pat, iter, body): (Option<Symbol>, &Pattern, &ForIter, &Block),
        emit: impl FnOnce(&mut Self) -> bool,
    ) -> bool {
        if !self.loop_bounds_reevaluable(iter) {
            return emit(self);
        }
        self.emit_alias_versioned(emit, |me| me.lower_for(label, pat, iter, body, &[]))
    }

    /// May this loop's range bounds be evaluated a SECOND time? The kernel arm's setup evaluates
    /// them once (the extent), before the guard; the may-overlap arm then lowers the whole loop,
    /// bounds included. For a pure bound that is only a recomputation, but a bound with an effect
    /// (`0..next_len()` where `next_len` prints) would run twice on that arm. Such a loop gets no
    /// versioning context — `emit` runs as before versioning existed, and a `MayAlias` pair declines.
    fn loop_bounds_reevaluable(&self, iter: &ForIter) -> bool {
        match iter {
            ForIter::Range {
                start, end, step, ..
            } => {
                self.coeff_is_hoistable(start)
                    && [end, step].into_iter().all(|bound| match bound {
                        Some(e) => self.coeff_is_hoistable(e),
                        None => true,
                    })
            }
            ForIter::Expr(e) => self.coeff_is_hoistable(e),
        }
    }

    /// [`Self::emit_alias_versioned`] for a statement-WINDOW recognizer (a fused norm, a flat 1-D
    /// scan, a copy-then-norm pair): the source is the run of statements the kernel call replaces.
    /// Every window matcher has already proved that the window's own `let` bindings are unmentioned
    /// after it, so lowering them on the may-overlap arm only leaves dead slots behind the join —
    /// the same shape `emit_nonempty_reduction_stmts` lowers on its empty-extent arm.
    pub(super) fn emit_versioned_stmts(
        &mut self,
        source: &[Stmt],
        emit: impl FnOnce(&mut Self) -> bool,
    ) -> bool {
        // The window's loops re-evaluate their bounds on the may-overlap arm (see
        // `loop_bounds_reevaluable`), and the kernel's extent was taken from one of them.
        let reevaluable = source.iter().all(|stmt| match &stmt.kind {
            StmtKind::For { iter, .. } => self.loop_bounds_reevaluable(iter),
            _ => true,
        });
        if !reevaluable {
            return emit(self);
        }
        self.emit_alias_versioned(emit, |me| {
            for stmt in source {
                if me.terminated {
                    break;
                }
                me.lower_stmt(stmt);
            }
        })
    }

    /// Run `emit` (ONE recognizer emitter for ONE loop or window) under a versioning context, and
    /// finish whatever guard it opened by lowering `lower_source` — the loop or window the emitter
    /// replaces — on the may-overlap arm. `emit` may lower straight-line statements (the generic
    /// vectorizer's scalar tail inside `try_vectorize_for`), but never a nested loop or window: a
    /// guard opened there would be finished with THIS source.
    ///
    /// The fallback lowers with recognition AND `@parallel` off: it is the path taken precisely
    /// when the operands may overlap, so it must be the source loop in source order. A kernel there
    /// would be the bug this module exists to remove, and an outlined parallel region over
    /// overlapping buffers is the same bug with threads.
    ///
    /// Returns what `emit` returned. If `emit` opened a guard and then declined anyway (an emitter
    /// bug the protocol forbids, handled rather than trusted), the kernel arm is routed into the
    /// fallback block and `false` is returned with the builder positioned there, so the caller's
    /// ordinary decline path lowers the source exactly once, reached from both arms.
    pub(super) fn emit_alias_versioned(
        &mut self,
        emit: impl FnOnce(&mut Self) -> bool,
        lower_source: impl FnOnce(&mut Self),
    ) -> bool {
        let outer = std::mem::replace(&mut self.gemm.guard, KernelGuard::Ready);
        let fired = emit(self);
        let state = std::mem::replace(&mut self.gemm.guard, outer);
        let KernelGuard::Pending(fallback) = state else {
            return fired;
        };
        if !fired {
            if !self.terminated {
                self.builder.br(fallback, vec![]);
            }
            self.builder.switch_to(fallback);
            self.terminated = false;
            return false;
        }
        let join = self.builder.new_block();
        if !self.terminated {
            self.builder.br(join, vec![]);
        }
        self.builder.switch_to(fallback);
        self.terminated = false;
        let (recognize, parallel) = (self.gemm.recognize, self.parallel_fn);
        self.gemm.recognize = false;
        self.parallel_fn = false;
        lower_source(self);
        self.gemm.recognize = recognize;
        self.parallel_fn = parallel;
        if !self.terminated {
            self.builder.br(join, vec![]);
        }
        self.builder.switch_to(join);
        self.terminated = false;
        true
    }
}

/// The per-channel plane of a recognized pooling nest: the input's `H * W` when `output` is false,
/// the output's `OH * OW` when it is true, with `OH = (H - KH) / SH + 1` and `OW = (W - KW) / SW + 1`
/// — the no-padding extent the kernel recomputes, which `match_pool2d` pins the source's own loop
/// bounds to. `None` unless every dim is the literal `match_pool2d` requires, or on overflow; the
/// caller then declines.
pub(crate) fn pool2d_plane(nest: &Pool2dNest, output: bool) -> Option<i64> {
    let (Dim::Lit(h), Dim::Lit(w), Dim::Lit(kh), Dim::Lit(kw), Dim::Lit(sh), Dim::Lit(sw)) =
        (nest.h, nest.w, nest.kh, nest.kw, nest.sh, nest.sw)
    else {
        return None;
    };
    if !output {
        return h.checked_mul(w);
    }
    if sh <= 0 || sw <= 0 {
        return None;
    }
    let oh = h.checked_sub(kh)?.checked_div(sh)?.checked_add(1)?;
    let ow = w.checked_sub(kw)?.checked_div(sw)?.checked_add(1)?;
    if oh <= 0 || ow <= 0 {
        return None;
    }
    oh.checked_mul(ow)
}
