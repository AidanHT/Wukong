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
//! it can still take. Under a versioning context ([`FnLowerer::emit_alias_versioned`], or
//! `emit_nonempty_reduction`, which wraps one) a `MayAlias` plan opens the guard and leaves the
//! builder in the kernel arm with the fallback pending; with no context — a whole-function
//! interception, a probe, a window this module has not been threaded through — a `MayAlias` plan
//! declines exactly as every emitter did before, so no caller is ever handed a half-built branch it
//! does not know how to finish.

use wukong_ast::{Block, ForIter, Pattern};
use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_span::Symbol;

use crate::alias::{plan_kernel_alias, KernelAliasPlan, OperandPair, PairContract};
use crate::{Dim, FnLowerer, LowpMatmulNest, Pool2dNest};

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
        self.emit_alias_versioned(emit, |me| me.lower_for(label, pat, iter, body, &[]))
    }

    /// Run `emit` (ONE recognizer emitter, which must not itself lower statements) under a
    /// versioning context, and finish whatever guard it opened by lowering `lower_source` — the
    /// loop or window the emitter replaces — on the may-overlap arm.
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
