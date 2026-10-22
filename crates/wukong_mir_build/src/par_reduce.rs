//! **The `@parallel` reduction tree** (P13.7): in a fast-mode `@parallel` function, the value of an
//! `f32` reduction loop is a function of the program alone — never of whether a multicore kernel
//! fired.
//!
//! `@parallel` licenses one reassociation of a float fold, and only one: the tree of the runtime's
//! reduction kernel (`wukong_runtime::reduce::tree` — fixed 8,192-element chunks, a 64-element
//! eight-accumulator partition inside each, a fixed halving combine, the chunk partials folded in
//! ascending order), applied to the loop's addends in iteration order and folded into the seed once.
//! Every way the compiler can lower such a loop computes exactly that tree:
//!
//! | how the loop is lowered | what runs |
//! |---|---|
//! | the kernel shape `for k in 0..n { s += f(x[k], y[k]) }` in the function body | `wukong_sreduce_f32_parallel` (native) / `wukong_sreduce_f32` (interpreter) — bit-identical twins |
//! | the same shape inside an outlined region body or the whole-function outliner's chunk body, or in a versioned kernel's fallback | `wukong_sreduce_f32`, the serial twin (the region already supplies the threads) |
//! | a `bf16`/`f16` array read through `as f32` | `wukong_{dot,sum,reduce}_{bf16,f16}_parallel` — the same tree on the exactly widened values (`lowp.rs`) |
//! | **anything else** — an `..=` range, a `step`, a non-zero start, an index that is not the bare loop variable, an addend no kernel has, `@generic`, `WUKONG_NO_RECOGNIZE`, an unsigned or narrow counter | **this module**: the loop as written fills a chunk buffer with its addends, and each full (or final) chunk is handed to `wukong_sreduce_f32` |
//!
//! The fallback is exact by construction rather than by imitation: it never re-implements the
//! tree, it hands the runtime's own serial twin the very elements the kernel would have read, one
//! chunk at a time. Calling the twin on one chunk returns `ident ⊕ partial`, and folding those into
//! a running total from `ident` is bit-for-bit the kernel's `ident ⊕ p0 ⊕ p1 ⊕ …`: for `+` the
//! running total can never be `-0.0` (it starts at `+0.0`, and a round-to-nearest sum is `-0.0`
//! only when both operands are), so `t + (0.0 + p)` equals `t + p` even for `p = -0.0`; for the
//! max/min folds `ident ⊕ p` is `p` exactly.
//!
//! An element contributes as the kernel's ops do, and keeps the rounding the source gives it before
//! the fold. A product addend `a * b` (the `RED_DOT` form — `x[k] * y[k]`, `(x[k] - y[k]) *
//! (x[k] - y[k])`, `a * b * c` as `(a * b) * c`) is one fused multiply-add of its two factors, each
//! converted to `f32` exactly as the source converts it: always in `s = s + a * b` / `s = a * b + s`,
//! where the source itself contracts the product at the `f32` of the `+` whatever the product's own
//! type (`try_contract_fma`), so a `bf16 * bf16` or `f32 * i16` product is never rounded on its own;
//! and in the compound `s += a * b` when the product is an `f32`, as the kernel and the vectorizer's
//! tree fuse it. A compound product of another type (`bf16 * bf16`, `i16 * i16`) is rounded to that
//! type by the source before the `+` sees it, so it is a plain addend here too. Any other addend is
//! evaluated as written — an `f32`, a half widened exactly, or one of the narrow integers sema
//! promotes into the `f32` fold (`i8`/`u8`/`i16`/`u16`, converted with its own signedness) — and
//! added. A running `fmax`/`fmin` folds with the `maxps`/`minps` rule, and `fmax(m, abs(e))` is the
//! absmax.
//!
//! What it does **not** cover, and keeps in the order it had: a strict function (`@strict_fp`, the
//! source-order fold), a non-`f32` accumulator (an `f64` or integer reduction keeps the vectorizer's
//! tree; integer folds are exact in any order), a body that is not a single fold statement, an
//! addend that calls anything but a math intrinsic or reads the accumulator, a `while` loop and a
//! `for x in array` loop (both keep the source fold, as they do without `@parallel`).
//!
//! The GPU backends are outside the contract, like every recognized kernel there: `--backend=gpu`
//! offloads the ops of `wukong_sreduce_f32` it supports to a device reduction of its own order, and
//! `--backend=gpu-native` lowers the call to a device helper (`mrt_sreduce`) that folds
//! sequentially; both are gated by the CPU<->GPU tolerance, not by bits.
//! The helper reads its operands with generic `ld`, so a chunk buffer in the `.local` frame is legal.

use std::borrow::Cow;

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Pattern, StmtKind};
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};
use wukong_span::Symbol;

use crate::{
    expr_mentions, mask_ty, single_path, FnLowerer, MathIntrinsic, RED_DOT, RED_MAX, RED_MIN,
    RED_SUM,
};

/// The runtime's fixed reduction chunk (`wukong_runtime::reduce::RCHUNK`): the buffer the fallback
/// fills holds at most one chunk, and a full buffer is one chunk partial.
pub(crate) const TREE_CHUNK: u64 = 8192;

/// How one `@parallel` reduction loop was lowered — recorded for the corpus census in tests.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum TreePath {
    /// `wukong_sreduce_f32[_parallel]` on the loop's own arrays.
    Kernel,
    /// A `bf16`/`f16` `_parallel` kernel.
    Lowp,
    /// This module's chunk-buffer fallback.
    Fallback,
}

#[cfg(test)]
thread_local! {
    static TRACE: std::cell::RefCell<Vec<TreePath>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Record how a `@parallel` reduction was lowered (tests only).
#[cfg(test)]
pub(crate) fn trace(path: TreePath) {
    TRACE.with(|t| t.borrow_mut().push(path));
}

/// Drain this thread's reduction-path trace.
#[cfg(test)]
pub(crate) fn take_trace() -> Vec<TreePath> {
    TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

/// A reduction loop the fallback can lower: the accumulator, the runtime op, and the element(s)
/// each iteration contributes.
struct TreePlan {
    /// The accumulator's slot (an in-scope `f32` scalar).
    s_slot: ValueId,
    /// `RED_SUM`, `RED_DOT`, `RED_MAX` or `RED_MIN`.
    op: i64,
    /// The addend (or, for `RED_DOT`, the product's left factor).
    p: Expr,
    /// The product's right factor (`RED_DOT` only).
    q: Option<Expr>,
}

/// The fold of one reduction body, as the vectorizer reads it (`reduction_of`): `s += e`,
/// `s = s + e` / `s = e + s`, a longer `+` chain holding `s` once (its addend the chain with `s`
/// removed), or `s = fmax(s, e)` / `fmin` in either operand order.
enum Fold {
    Add,
    Max,
    Min,
}

/// `e` with the single `+`-chain occurrence of the accumulator `s` removed — the vectorizer's
/// `reduction_of` rule, mirrored so the two reassociating lowerings agree on what the addend is. A
/// node on the path to `s` is rebuilt under its own id (so its sema type — the accumulator's — still
/// resolves); `s` may not be the left operand of a subtraction nor its right one.
fn strip_accumulator(e: &Expr, s: Symbol) -> Option<Expr> {
    let ExprKind::Binary { op, lhs, rhs } = &e.kind else {
        return None;
    };
    let is_s = |x: &Expr| single_path(x) == Some(s);
    let rebuilt = |l: Expr, r: Expr| Expr {
        id: e.id,
        kind: ExprKind::Binary {
            op: *op,
            lhs: Box::new(l),
            rhs: Box::new(r),
        },
        span: e.span,
    };
    match op {
        ast::BinOp::Add => {
            if is_s(lhs) {
                Some((**rhs).clone())
            } else if is_s(rhs) {
                Some((**lhs).clone())
            } else if let Some(l) = strip_accumulator(lhs, s) {
                Some(rebuilt(l, (**rhs).clone()))
            } else {
                strip_accumulator(rhs, s).map(|r| rebuilt((**lhs).clone(), r))
            }
        }
        ast::BinOp::Sub if !is_s(lhs) => {
            strip_accumulator(lhs, s).map(|l| rebuilt(l, (**rhs).clone()))
        }
        _ => None,
    }
}

/// Does `e` — an expression [`FnLowerer::tree_pure`] admitted — name `s`? Exact over that subset
/// (an `if` and a block included, which the general `expr_mentions` treats as unknown); any other
/// kind answers `true`.
fn tree_mentions(e: &Expr, s: Symbol) -> bool {
    let block =
        |b: &Block| !b.stmts.is_empty() || b.tail.as_ref().is_none_or(|t| tree_mentions(t, s));
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Char(_) | ExprKind::Bool(_) => false,
        ExprKind::Path(_) => expr_mentions(e, s),
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => tree_mentions(expr, s),
        ExprKind::Binary { lhs, rhs, .. } => tree_mentions(lhs, s) || tree_mentions(rhs, s),
        ExprKind::Index { base, indices } => {
            tree_mentions(base, s) || indices.iter().any(|i| tree_mentions(i, s))
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => tree_mentions(base, s),
        ExprKind::Call {
            callee,
            generic_args,
            args,
        } => {
            !generic_args.is_empty()
                || tree_mentions(callee, s)
                || args.iter().any(|a| tree_mentions(a, s))
        }
        ExprKind::Block(b) => block(b),
        ExprKind::If {
            cond,
            then_branch,
            else_branch: Some(other),
        } => tree_mentions(cond, s) || block(then_branch) || tree_mentions(other, s),
        _ => true,
    }
}

impl FnLowerer<'_> {
    /// Is `e` an addend the fallback may evaluate in a buffer-filling loop instead of the fold?
    /// Literals, names, arithmetic, casts, indexing, field reads, math intrinsics and `if`/blocks of
    /// those. A user call is refused (it could observe the accumulator, or print in an order the
    /// fold no longer has), and so is every expression kind not modelled here.
    fn tree_pure(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Char(_) | ExprKind::Bool(_) => true,
            ExprKind::Path(p) => p.is_single(),
            ExprKind::Unary { op, expr } => {
                matches!(op, ast::UnOp::Neg | ast::UnOp::Not) && self.tree_pure(expr)
            }
            ExprKind::Binary { lhs, rhs, .. } => self.tree_pure(lhs) && self.tree_pure(rhs),
            ExprKind::Cast { expr, .. } => self.tree_pure(expr),
            ExprKind::Index { base, indices } => {
                self.tree_pure(base) && indices.iter().all(|i| self.tree_pure(i))
            }
            ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
                self.tree_pure(base)
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                generic_args.is_empty()
                    && self.vectorizable_intrinsic(callee).is_some()
                    && args.iter().all(|a| self.tree_pure(a))
            }
            ExprKind::Block(b) => self.tree_pure_block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch: Some(other),
            } => self.tree_pure(cond) && self.tree_pure_block(then_branch) && self.tree_pure(other),
            _ => false,
        }
    }

    fn tree_pure_block(&self, b: &Block) -> bool {
        b.stmts.is_empty() && b.tail.as_ref().is_some_and(|t| self.tree_pure(t))
    }

    /// The plan for a `@parallel` reduction loop the fallback lowers, or `None` when the loop is not
    /// one (it then lowers exactly as before). Pure: emits no MIR.
    fn tree_plan(&self, pat: &Pattern, iter: &ForIter, body: &Block) -> Option<TreePlan> {
        let ForIter::Range {
            start,
            end: Some(end),
            step,
            ..
        } = iter
        else {
            return None;
        };
        if !(self.expr_mir(start).is_int() && self.expr_mir(end).is_int()) {
            return None;
        }
        let k = match &pat.kind {
            ast::PatKind::Ident(k) => Some(*k),
            ast::PatKind::Wildcard => None,
            _ => return None,
        };
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        let StmtKind::Assign { target, op, value } = &body.stmts[0].kind else {
            return None;
        };
        let s = single_path(target)?;
        if Some(s) == k {
            return None;
        }
        let Some((s_slot, MirType::F32)) = self.lookup(s) else {
            return None;
        };
        let is_s = |e: &Expr| single_path(e) == Some(s);
        // `contracts`: the addend is a direct operand of the accumulator's own `+` in an assignment
        // (`s = s + e` / `s = e + s`) — the one place `try_contract_fma` fuses a product addend into
        // the fold, at that `+` node's type. A compound `s += a * b` lowers the product on its own.
        let (addend, fold, contracts): (Cow<'_, Expr>, Fold, bool) = match op {
            ast::AssignOp::Add => (Cow::Borrowed(value), Fold::Add, false),
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if is_s(lhs) => (Cow::Borrowed(&**rhs), Fold::Add, true),
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if is_s(rhs) => (Cow::Borrowed(&**lhs), Fold::Add, true),
                ExprKind::Binary { .. } => {
                    (Cow::Owned(strip_accumulator(value, s)?), Fold::Add, false)
                }
                ExprKind::Call { callee, args, .. } if args.len() == 2 => {
                    let fold = match self.vectorizable_intrinsic(callee) {
                        Some(MathIntrinsic::Fmax) => Fold::Max,
                        Some(MathIntrinsic::Fmin) => Fold::Min,
                        _ => return None,
                    };
                    if is_s(&args[0]) {
                        (Cow::Borrowed(&args[1]), fold, false)
                    } else if is_s(&args[1]) {
                        (Cow::Borrowed(&args[0]), fold, false)
                    } else {
                        return None;
                    }
                }
                _ => return None,
            },
            _ => return None,
        };
        // Nothing the loop evaluates may read the accumulator: the fallback updates it once, after
        // the loop. (`expr_mentions` is conservative: an unmodelled kind counts as a read.)
        if !self.tree_pure(&addend)
            || tree_mentions(&addend, s)
            || expr_mentions(start, s)
            || expr_mentions(end, s)
            || step.as_ref().is_some_and(|st| expr_mentions(st, s))
        {
            return None;
        }
        // A value the source converts into the `f32` fold: an `f32`, a half (exactly), or — for a
        // sum — one of the narrow integers sema promotes into `f32` (`i8`/`u8`/`i16`/`u16`, also
        // exactly; `fmax`/`fmin` refuse an integer operand, E0445). The fallback converts it with
        // the source's own cast, signedness included (`tree_store_element`).
        let widens = |e: &Expr| {
            let t = self.expr_mir(e);
            matches!(t, MirType::F32 | MirType::F16 | MirType::BF16)
                || (t.is_int() && t != MirType::I1)
        };
        // The fused-multiply-add form, `RED_DOT` over the product's two factors widened to `f32`. A
        // product addend contributes `fma(a, b, acc)` when the source fuses it — `s = s + a * b`,
        // whatever the product's own type (`try_contract_fma` widens each factor to the `f32` of the
        // `+` and rounds once) — and, as every reduction kernel and the vectorizer's tree do, in the
        // compound `s += a * b` when the product is itself an `f32`. Only a compound product of
        // another type (`bf16 * bf16`, `i16 * i16`) is the source's rounded product, a plain addend
        // below: the source rounds it to its own type before the `+` ever sees it.
        if let (
            Fold::Add,
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            },
        ) = (&fold, &addend.kind)
        {
            let fused_at = if contracts {
                self.expr_mir(value)
            } else {
                self.expr_mir(&addend)
            };
            if fused_at == MirType::F32 && widens(lhs) && widens(rhs) {
                return Some(TreePlan {
                    s_slot,
                    op: RED_DOT,
                    p: (**lhs).clone(),
                    q: Some((**rhs).clone()),
                });
            }
        }
        let lane_ok = match fold {
            Fold::Add => widens(&addend),
            Fold::Max | Fold::Min => matches!(
                self.expr_mir(&addend),
                MirType::F32 | MirType::F16 | MirType::BF16
            ),
        };
        if !lane_ok {
            return None;
        }
        let op = match fold {
            Fold::Add => RED_SUM,
            Fold::Max => RED_MAX,
            Fold::Min => RED_MIN,
        };
        Some(TreePlan {
            s_slot,
            op,
            p: addend.into_owned(),
            q: None,
        })
    }

    /// Lower a `@parallel` `f32` reduction loop that no kernel took through the chunk-buffer
    /// fallback (see the module docs). Returns false, having emitted nothing, when this is not a
    /// fast-mode `@parallel` function or the loop is not such a reduction.
    pub(super) fn try_emit_tree_reduction(
        &mut self,
        label: Option<Symbol>,
        pat: &Pattern,
        iter: &ForIter,
        body: &Block,
    ) -> bool {
        if !self.gemm.par_reduce {
            return false;
        }
        let Some(plan) = self.tree_plan(pat, iter, body) else {
            return false;
        };
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive,
            step,
        } = iter
        else {
            return false;
        };
        let _ = label; // the body is one fold statement: nothing can `break`/`continue` to it
        #[cfg(test)]
        trace(TreePath::Fallback);
        let cap = self.tree_buffer_len(start, end, *inclusive, step.as_ref());
        let buf_ty = MirType::Array(Box::new(MirType::F32), cap as u32);
        let pbuf = self.builder.alloca(buf_ty.clone());
        let qbuf = if plan.q.is_some() {
            self.builder.alloca(buf_ty)
        } else {
            pbuf
        };
        let fill = self.builder.alloca(MirType::I64);
        let total = self.builder.alloca(MirType::F32);
        let ran = self.builder.alloca(MirType::I64);
        let zero = self.tree_i64(0);
        self.builder.build_void(Op::Store {
            ptr: fill,
            value: zero,
        });
        self.builder.build_void(Op::Store {
            ptr: ran,
            value: zero,
        });
        let ident = self.tree_ident(plan.op);
        self.builder.build_void(Op::Store {
            ptr: total,
            value: ident,
        });

        // The loop control, exactly as the scalar `for` lowering spells it: the start evaluated once,
        // the end re-evaluated at every header, `i += step` in the latch, the same counter type and
        // comparison. Only the body differs — it stores the addend instead of folding it.
        let (ity, signed) = self.range_counter_type(start, end);
        let slot = self.builder.alloca(ity.clone());
        let s0 = self.lower_range_bound(start, &ity, signed);
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: s0,
        });
        self.push_scope();
        if let ast::PatKind::Ident(name) = &pat.kind {
            self.bind(*name, slot, ity.clone());
        }
        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let flush_bb = self.builder.new_block();
        let latch = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        self.terminated = false;
        let i_val = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let end_val = self.lower_range_bound(end, &ity, signed);
        let pred = match (*inclusive, signed) {
            (false, true) => CmpOp::Slt,
            (true, true) => CmpOp::Sle,
            (false, false) => CmpOp::Ult,
            (true, false) => CmpOp::Ule,
        };
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(pred, i_val, end_val));
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        // body: buf[fill] = p (and qbuf[fill] = q); fill += 1; a full buffer is one chunk.
        self.builder.switch_to(body_bb);
        self.terminated = false;
        let at = self
            .builder
            .build(MirType::I64, Op::Load(fill, MirType::I64));
        self.tree_store_element(&plan.p, pbuf, at);
        if let Some(q) = &plan.q {
            self.tree_store_element(q, qbuf, at);
        }
        let one = self.tree_i64(1);
        let next_fill = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, at, one));
        self.builder.build_void(Op::Store {
            ptr: fill,
            value: next_fill,
        });
        let capv = self.tree_i64(cap as i128);
        let full = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Eq, next_fill, capv));
        self.builder.cond_br(full, flush_bb, vec![], latch, vec![]);

        self.builder.switch_to(flush_bb);
        self.terminated = false;
        self.tree_flush_chunk(&plan, (pbuf, qbuf), (total, ran), capv);
        let zero = self.tree_i64(0);
        self.builder.build_void(Op::Store {
            ptr: fill,
            value: zero,
        });
        self.builder.br(latch, vec![]);

        // latch: i += step, exactly as the scalar loop.
        self.builder.switch_to(latch);
        self.terminated = false;
        let cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let step_val = match step {
            Some(st) if self.expr_mir(st).is_int() => self.lower_range_bound(st, &ity, signed),
            Some(st) => self.lower_expr(st),
            None => self
                .builder
                .build(ity.clone(), Op::ConstInt(1, ity.clone())),
        };
        let next = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, cur, step_val));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: next,
        });
        self.builder.br(header, vec![]);
        self.pop_scope();

        // exit: the final, partial chunk.
        self.builder.switch_to(exit);
        self.terminated = false;
        let left = self
            .builder
            .build(MirType::I64, Op::Load(fill, MirType::I64));
        let zero = self.tree_i64(0);
        let pending = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Sgt, left, zero));
        let tail_bb = self.builder.new_block();
        let done = self.builder.new_block();
        self.builder.cond_br(pending, tail_bb, vec![], done, vec![]);
        self.builder.switch_to(tail_bb);
        self.tree_flush_chunk(&plan, (pbuf, qbuf), (total, ran), left);
        self.builder.br(done, vec![]);

        // done: the seed folds in the total once — or stays bit for bit when no iteration ran.
        self.builder.switch_to(done);
        self.terminated = false;
        let seed = self
            .builder
            .build(MirType::F32, Op::Load(plan.s_slot, MirType::F32));
        let tot = self
            .builder
            .build(MirType::F32, Op::Load(total, MirType::F32));
        let combined = self.tree_fold2(plan.op, seed, tot);
        let any = self
            .builder
            .build(MirType::I64, Op::Load(ran, MirType::I64));
        let zero = self.tree_i64(0);
        let nonempty = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Ne, any, zero));
        let new_s = self
            .builder
            .build(MirType::F32, Op::Select(nonempty, combined, seed));
        self.builder.build_void(Op::Store {
            ptr: plan.s_slot,
            value: new_s,
        });
        true
    }

    /// The buffer length: one chunk, or the whole trip when it is a smaller compile-time constant
    /// (an inner `for d in 0..64` reduction needs 64 elements of frame, not 8,192).
    fn tree_buffer_len(
        &self,
        start: &Expr,
        end: &Expr,
        inclusive: bool,
        step: Option<&Expr>,
    ) -> u64 {
        let trip = match (self.const_usize_expr(start), self.const_usize_expr(end)) {
            (Some(a), Some(b)) => {
                let (a, b) = (u64::from(a), u64::from(b));
                let span = if inclusive {
                    (b + 1).saturating_sub(a)
                } else {
                    b.saturating_sub(a)
                };
                match step {
                    None => Some(span),
                    Some(st) => match self.const_usize_expr(st) {
                        Some(k) if k > 0 => Some(span.div_ceil(u64::from(k))),
                        _ => None,
                    },
                }
            }
            _ => None,
        };
        trip.map_or(TREE_CHUNK, |t| t.clamp(1, TREE_CHUNK))
    }

    /// Evaluate one addend (or product factor) in the current iteration and store it, as `f32`, at
    /// `buf[at]` — converted exactly as the source converts it into the fold (`coerce_to` with the
    /// operand's own signedness, so a `u16` is not read as an `i16`).
    fn tree_store_element(&mut self, e: &Expr, buf: ValueId, at: ValueId) {
        let v = self.lower_expr(e);
        let from = self.operand_mir(v, e);
        let signed = self.signed(e);
        let v = self.coerce_to(v, &from, &MirType::F32, signed);
        let ptr = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: buf,
                index: at,
                elem: MirType::F32,
            },
        );
        self.builder.build_void(Op::Store { ptr, value: v });
    }

    /// `total = total ⊕ wukong_sreduce_f32(pbuf, qbuf, len, op); ran = 1` — one chunk partial.
    fn tree_flush_chunk(
        &mut self,
        plan: &TreePlan,
        (pbuf, qbuf): (ValueId, ValueId),
        (total, ran): (ValueId, ValueId),
        len: ValueId,
    ) {
        let opv = self.tree_i64(plan.op as i128);
        let part = self.builder.build(
            MirType::F32,
            Op::Call {
                func: self.gemm.sred,
                args: vec![pbuf, qbuf, len, opv],
            },
        );
        let cur = self
            .builder
            .build(MirType::F32, Op::Load(total, MirType::F32));
        let next = self.tree_fold2(plan.op, cur, part);
        self.builder.build_void(Op::Store {
            ptr: total,
            value: next,
        });
        let one = self.tree_i64(1);
        self.builder.build_void(Op::Store {
            ptr: ran,
            value: one,
        });
    }

    /// The kernel's `fold2(a, b)`: `a + b`, or `a > b ? a : b` / `a < b ? a : b` as `Cmp + Select`
    /// (the `maxps`/`minps` rule, and the very combine `try_emit_parallel_reduction` emits).
    fn tree_fold2(&mut self, op: i64, a: ValueId, b: ValueId) -> ValueId {
        match op {
            RED_MAX | RED_MIN => {
                let pred = if op == RED_MIN {
                    CmpOp::Folt
                } else {
                    CmpOp::Fogt
                };
                let mask = self
                    .builder
                    .build(mask_ty(&MirType::F32), Op::Cmp(pred, a, b));
                self.builder.build(MirType::F32, Op::Select(mask, a, b))
            }
            _ => self.builder.build(MirType::F32, Op::Bin(BinOp::FAdd, a, b)),
        }
    }

    /// An `i64` constant.
    fn tree_i64(&mut self, v: i128) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(v, MirType::I64))
    }

    /// The fold identity: `+0.0`, or `-∞` / `+∞` for max / min.
    fn tree_ident(&mut self, op: i64) -> ValueId {
        let v = match op {
            RED_MAX => f64::NEG_INFINITY,
            RED_MIN => f64::INFINITY,
            _ => 0.0,
        };
        self.builder
            .build(MirType::F32, Op::ConstFloat(v, MirType::F32))
    }
}
