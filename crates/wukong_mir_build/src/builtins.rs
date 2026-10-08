//! Lowering of the **named kernel builtins** (`matmul`, `matmul_nt`, `matmul_tn`, `linear`,
//! `transpose`, `softmax`, `log_softmax`, `layernorm`, `rmsnorm`, the two-argument activations,
//! `sum`/`amax`/`amin`/`dot`, and `sdpa`) — the explicit door to the `wukong_runtime` kernels that
//! until now was reachable only through a recognizer. `wukong_sema::builtins` shape-checks every call
//! and records how each of the builtin's dimensions resolved ([`wukong_sema::BuiltinCall`]); this
//! module lowers the call from that record, so the two crates cannot disagree on an extent.
//!
//! A builtin is not an optimization of some loop the program also means — the call *is* the
//! program — so it lowers regardless of `@generic` / recognition being off, exactly like `sdpa`
//! always did. Both backends call the identical runtime symbol (the interpreter marshals its slot
//! memory through it), so the result is bit-exact across interp and native by construction; the
//! fixtures `tests/run/builtin_*.wk` print every lane against expectations derived from the scalar
//! definition of each operation.
//!
//! What the lowering adds around the kernel call, so that a builtin means the same thing on every
//! backend for every input the type checker accepts:
//!
//! * **Runtime extent checks.** A dimension known only at run time (a turbofish variable, a generic
//!   dimension, a slice length) must lie in `0..=i32::MAX`, and an operand whose element count is
//!   only known at run time — a `[]f32` slice, or a fixed array against a runtime dimension — must
//!   hold exactly the elements its shape names (`sdpa`: at least `s·d`). A violation traps through
//!   the `assert` intrinsic (exit 1 on every backend) instead of letting the kernel read or write
//!   past a buffer. The `i32::MAX` bound is what keeps every element-count product below `2^63`, so
//!   the checks themselves cannot be fooled by a wrapped multiplication.
//! * **Value semantics under aliasing.** Each builtin reads its inputs as they were before the call
//!   and then writes its output. The runtime kernels are only defined for disjoint operands (or, for
//!   the elementwise and row-wise kernels, an output that *exactly* aliases the first input, which
//!   they document as in-place safe), and the interpreter refuses a kernel call outside that
//!   contract. So the lowering compares the output's address range against every input's at run
//!   time — `Gep` + `PtrToInt`, which both backends measure in their own unit (interpreter slots,
//!   native bytes), so the test is exact on each — and on any other overlap runs the kernel into a
//!   fresh heap buffer and copies the result out. The only pair the test is omitted for is the
//!   documented exact alias spelled with one name (`softmax(x, x)`), where it is decidable
//!   statically. A static "distinct names are distinct buffers" shortcut is deliberately NOT taken:
//!   the whole-program alias facts (`alias.rs`) only account for names that are spelled as index
//!   bases, and a builtin operand need not be one.
//! * **Empty contractions.** The GEMM kernels return without writing anything when `K <= 0`, but
//!   `A·B` over an empty contraction is the zero matrix (and `linear` is the broadcast bias), so a
//!   `K` that is not a positive constant gets that fill on its `K == 0` path.
//! * **`@parallel`.** Inside a `@parallel` function the multicore `_parallel` twin is called; every
//!   twin used here is bit-identical to its serial kernel.

use wukong_ast::{Expr, ExprKind};
use wukong_mir::{BinOp, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_sema::{
    builtin_operand_dims, builtin_output, Activation, BuiltinCall, BuiltinDim, KernelBuiltin,
};
use wukong_span::Symbol;
use wukong_types::{Dim, Scalar, Ty};

use crate::{
    DefKind, FnLowerer, EPI_ACT_IDENTITY, NORM_LAYERNORM, NORM_LOGSOFTMAX, NORM_RMSNORM,
    NORM_SOFTMAX, RED_DOT, RED_MAX, RED_MIN, RED_SUM, SLICE_LEN_OFF, VMATH_EXP, VMATH_GELU,
    VMATH_LOG, VMATH_RELU, VMATH_SIGMOID, VMATH_SILU, VMATH_TANH,
};

/// The largest extent a builtin accepts from a value known only at run time (`i32::MAX`). Any two
/// such extents multiply to less than `2^62`, so no element-count product the runtime checks
/// compute can wrap.
const MAX_RUNTIME_EXTENT: i128 = i32::MAX as i128;

/// The largest *constant* extent that may share a call with a runtime one: `2^32 · i32::MAX` still
/// fits an `i64`, so a mixed constant·runtime element count cannot wrap either.
const MAX_MIXED_CONST: u64 = 1 << 32;

/// One lowered buffer operand.
struct BufArg {
    /// The data pointer the kernel reads/writes.
    ptr: ValueId,
    /// The runtime element count of a `[]f32` slice operand.
    slice_len: Option<ValueId>,
    /// The static `f32` element count of a fixed-array operand (or of a tensor whose extents are
    /// all constants — `sdpa` only, whose tensor operands sema cannot unify with its value-typed
    /// `s`/`d`).
    array_len: Option<u64>,
    /// The operand's name when it is spelled as a plain variable: the one case where an exact alias
    /// is decidable without a runtime test.
    sym: Option<Symbol>,
}

/// The number of `f32` scalars a (possibly nested) fixed `f32` array holds.
fn flat_f32_len(t: &Ty) -> Option<u64> {
    match t {
        Ty::Scalar(Scalar::F32) => Some(1),
        Ty::Array { elem, len, .. } => flat_f32_len(elem)?.checked_mul(*len),
        _ => None,
    }
}

/// The element count of a tensor type whose extents are all constants.
fn const_tensor_len(t: &Ty) -> Option<u64> {
    let Ty::Tensor { shape, .. } = t else {
        return None;
    };
    let mut n: u64 = 1;
    for d in &shape.0 {
        match d {
            Dim::Const(c) => n = n.checked_mul(*c)?,
            Dim::Var(_) | Dim::Dynamic | Dim::Expr(_) => return None,
        }
    }
    Some(n)
}

/// Whether the kernel is documented to accept its output *exactly* aliasing its first input
/// (`x == out`): the elementwise and row-wise kernels read each element (each row) before they write
/// it. The GEMMs and the transpose are not, and neither is any *other* input (`gamma`, `beta`).
fn in_place_ok(kind: KernelBuiltin) -> bool {
    use KernelBuiltin::*;
    matches!(kind, Softmax | LogSoftmax | LayerNorm | RmsNorm | Map(_))
}

fn vmath_op(act: Activation) -> i64 {
    match act {
        Activation::Exp => VMATH_EXP as i64,
        Activation::Log => VMATH_LOG as i64,
        Activation::Tanh => VMATH_TANH as i64,
        Activation::Sigmoid => VMATH_SIGMOID as i64,
        Activation::Relu => VMATH_RELU,
        Activation::Silu => VMATH_SILU as i64,
        Activation::Gelu => VMATH_GELU as i64,
    }
}

/// Refuse (as a codegen limit) a constant extent the checks below could not keep exact: one outside
/// the kernels' signed 64-bit ABI, one too large to share a call with a runtime extent, or a fully
/// constant operand whose element count does not fit an `i64`.
fn const_dims_fit(
    dims: &[BuiltinDim],
    operand_dims: &[Option<&'static [usize]>],
) -> Result<(), &'static str> {
    let any_runtime = dims.iter().any(|d| !matches!(d, BuiltinDim::Const(_)));
    for d in dims {
        if let BuiltinDim::Const(n) = d {
            if *n > i64::MAX as u64 {
                return Err(
                    "a kernel-builtin extent that does not fit the kernels' signed 64-bit ABI",
                );
            }
            if any_runtime && *n > MAX_MIXED_CONST {
                return Err(
                    "a kernel-builtin constant extent above 2^32 in a call that also has a \
                     runtime extent",
                );
            }
        }
    }
    for od in operand_dims.iter().flatten() {
        let mut acc: u64 = 1;
        for &i in od.iter() {
            if let Some(BuiltinDim::Const(n)) = dims.get(i) {
                acc = acc
                    .checked_mul(*n)
                    .filter(|v| *v <= i64::MAX as u64)
                    .ok_or("a kernel-builtin operand whose element count does not fit an i64")?;
            }
        }
    }
    Ok(())
}

impl FnLowerer<'_> {
    fn const_i64(&mut self, v: i128) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(v, MirType::I64))
    }

    /// `0 <= d <= i32::MAX`, as an `i1`.
    fn extent_in_range(&mut self, d: ValueId) -> ValueId {
        let zero = self.const_i64(0);
        let max = self.const_i64(MAX_RUNTIME_EXTENT);
        let lo = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Sge, d, zero));
        let hi = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Sle, d, max));
        self.builder.build(MirType::I1, Op::Bin(BinOp::And, lo, hi))
    }

    /// Trap (through the `assert` intrinsic) unless every condition holds. `false` when a user
    /// function named `assert` shadows the intrinsic — the caller then reports a codegen limit
    /// rather than calling the user's function with a check it never asked for.
    fn emit_builtin_checks(&mut self, checks: Vec<ValueId>) -> bool {
        if checks.is_empty() {
            return true;
        }
        let assert_fn = self.gemm.assert_fn;
        if matches!(
            self.sema.defs.lookup(assert_fn).map(|d| &d.kind),
            Some(DefKind::Fn(_))
        ) {
            return false;
        }
        for c in checks {
            self.builder.build_void(Op::Call {
                func: assert_fn,
                args: vec![c],
            });
        }
        true
    }

    /// Lower a call that sema resolved to a named kernel builtin, or `None` when `e` is not one
    /// (sema records only a call that shape-checked cleanly).
    pub(crate) fn lower_kernel_builtin(&mut self, e: &Expr, args: &[Expr]) -> Option<ValueId> {
        let call: BuiltinCall = self.sema.builtin_calls.get(&e.id)?.clone();
        if call.kind == KernelBuiltin::Sdpa {
            return Some(self.lower_sdpa_builtin(e, args));
        }
        let operand_dims = builtin_operand_dims(call.kind);
        if operand_dims.len() != args.len() {
            return None;
        }
        if let Err(what) = const_dims_fit(&call.dims, &operand_dims) {
            self.unsupported(e.span, what);
            return Some(self.const_zero(MirType::I32));
        }

        // Lower every argument exactly once, in source order.
        let mut bufs: Vec<Option<BufArg>> = Vec::with_capacity(args.len());
        let mut eps: Option<ValueId> = None;
        for (a, dims) in args.iter().zip(&operand_dims) {
            if dims.is_some() {
                let b = self.lower_builtin_buffer(a);
                bufs.push(Some(b));
            } else {
                // The one scalar parameter is an `eps`: the norm kernels take its `f32` bits.
                let f = self.lower_coerced(a, &MirType::F32);
                let bits = self
                    .builder
                    .build(MirType::I32, Op::Cast(CastKind::Bitcast, f, MirType::I32));
                eps = Some(
                    self.builder
                        .build(MirType::I64, Op::Cast(CastKind::ZExt, bits, MirType::I64)),
                );
                bufs.push(None);
            }
        }

        // Materialize every dimension as an i64; a runtime one must be a sane extent.
        let mut checks: Vec<ValueId> = Vec::new();
        let mut dims: Vec<ValueId> = Vec::with_capacity(call.dims.len());
        for d in &call.dims {
            let v = match d {
                BuiltinDim::Const(n) => self.const_i64(*n as i128),
                &BuiltinDim::Named { name, signed } => {
                    let Some((slot, ty)) = self.lookup(name) else {
                        self.unsupported(
                            e.span,
                            "a kernel-builtin dimension with no binding in scope",
                        );
                        return Some(self.const_zero(MirType::I32));
                    };
                    let v = self.builder.build(ty.clone(), Op::Load(slot, ty.clone()));
                    let v = self.coerce_to(v, &ty, &MirType::I64, signed);
                    let ok = self.extent_in_range(v);
                    checks.push(ok);
                    v
                }
                &BuiltinDim::SliceLen { arg, div } => {
                    let Some(len) = bufs.get(arg).and_then(|b| b.as_ref()?.slice_len) else {
                        self.unsupported(
                            e.span,
                            "a kernel-builtin dimension read from a non-slice",
                        );
                        return Some(self.const_zero(MirType::I32));
                    };
                    let v = if div == 1 {
                        len
                    } else {
                        let dv = self.const_i64(div as i128);
                        self.builder
                            .build(MirType::I64, Op::Bin(BinOp::SDiv, len, dv))
                    };
                    let ok = self.extent_in_range(v);
                    checks.push(ok);
                    v
                }
                // A derived extent from a tensor operand's shape (`3*D`): evaluated from the
                // in-scope dimension bindings exactly as a stride over that axis is.
                BuiltinDim::Derived(dim) => {
                    let v = dim
                        .to_poly()
                        .and_then(|p| self.emit_dim_poly(&p, &|me, s| me.dim_var_value(s)));
                    let Some(v) = v else {
                        self.unsupported(
                            e.span,
                            "a kernel-builtin derived dimension with no binding in scope",
                        );
                        return Some(self.const_zero(MirType::I32));
                    };
                    let ok = self.extent_in_range(v);
                    checks.push(ok);
                    v
                }
            };
            dims.push(v);
        }

        // Each buffer operand's element count (the product of its shape's dimensions), and the
        // exact-length check for every operand whose length is not a type-level fact.
        let mut counts: Vec<Option<ValueId>> = Vec::with_capacity(args.len());
        for (i, od) in operand_dims.iter().enumerate() {
            let Some(od) = od else {
                counts.push(None);
                continue;
            };
            let all_const = od
                .iter()
                .all(|&d| matches!(call.dims.get(d), Some(BuiltinDim::Const(_))));
            let mut count = self.const_i64(1);
            for &d in od.iter() {
                count = self
                    .builder
                    .build(MirType::I64, Op::Bin(BinOp::Mul, count, dims[d]));
            }
            counts.push(Some(count));
            // A rank-1 slice whose own length IS the dimension needs no check against itself.
            if let [d] = od[..] {
                if call.dims.get(d) == Some(&BuiltinDim::SliceLen { arg: i, div: 1 }) {
                    continue;
                }
            }
            let Some(b) = bufs[i].as_ref() else { continue };
            let have = match (b.slice_len, b.array_len) {
                (Some(len), _) => len,
                // A fully static array was checked by sema (E0501).
                (None, Some(_)) if all_const => continue,
                (None, Some(n)) => self.const_i64(n as i128),
                // A tensor operand: its shape was unified with the call by sema.
                (None, None) => continue,
            };
            checks.push(
                self.builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Eq, have, count)),
            );
        }
        if !self.emit_builtin_checks(checks) {
            self.unsupported(
                e.span,
                "a kernel builtin with a runtime extent in a program that defines its own `assert`",
            );
            return Some(self.const_zero(MirType::I32));
        }

        let ptrs: Vec<Option<ValueId>> = bufs.iter().map(|b| b.as_ref().map(|b| b.ptr)).collect();
        let Some(out_ix) = builtin_output(call.kind) else {
            // A reduction: no output buffer, so nothing can alias.
            return self.emit_builtin_kernel(&call, &ptrs, None, &dims, eps, None);
        };
        let out_ptr = ptrs[out_ix]?;
        let out_count = counts[out_ix]?;

        // Which inputs need a runtime overlap test against the output? All of them, except the
        // documented exact alias of the first input spelled with the output's own name.
        let out_sym = bufs[out_ix].as_ref().and_then(|b| b.sym);
        let mut tested: Vec<(ValueId, ValueId, bool)> = Vec::new();
        for (i, b) in bufs.iter().enumerate() {
            let Some(b) = b else { continue };
            if i == out_ix {
                continue;
            }
            let exact_ok = i == 0 && in_place_ok(call.kind);
            if exact_ok && b.sym.is_some() && b.sym == out_sym {
                continue;
            }
            let Some(n) = counts[i] else { continue };
            tested.push((b.ptr, n, exact_ok));
        }
        let hazard = self.overlap_hazard(out_ptr, out_count, &tested);
        Some(
            self.dispatch_with_scratch(hazard, out_ptr, out_count, |this, out| {
                this.emit_builtin_kernel(&call, &ptrs, out, &dims, eps, Some(out_count));
            }),
        )
    }

    /// `true` (as an `i1`) when the `count`-element output at `out` overlaps one of `inputs` in a
    /// way its kernel's contract forbids; each input is `(ptr, count, exact_alias_ok)`. `None` when
    /// there is nothing to test.
    fn overlap_hazard(
        &mut self,
        out: ValueId,
        out_count: ValueId,
        inputs: &[(ValueId, ValueId, bool)],
    ) -> Option<ValueId> {
        if inputs.is_empty() {
            return None;
        }
        let (out_lo, out_hi) = self.builtin_range(out, out_count);
        let mut bad: Option<ValueId> = None;
        for &(p, n, exact_ok) in inputs {
            let (lo, hi) = self.builtin_range(p, n);
            // Two half-open ranges intersect iff both are non-empty and each one ends after the
            // other begins. An empty range overlaps nothing, wherever its base points (an empty
            // input's base may well lie inside the output); the output's own emptiness is folded
            // in once, below.
            let a_above = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ugt, hi, out_lo));
            let b_above = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ugt, out_hi, lo));
            let nonempty = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Ugt, hi, lo));
            let crossing = self
                .builder
                .build(MirType::I1, Op::Bin(BinOp::And, a_above, b_above));
            let mut hazard = self
                .builder
                .build(MirType::I1, Op::Bin(BinOp::And, crossing, nonempty));
            if exact_ok {
                // An exact alias of the first input is the kernel's documented in-place form.
                let shifted = self
                    .builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Ne, lo, out_lo));
                hazard = self
                    .builder
                    .build(MirType::I1, Op::Bin(BinOp::And, hazard, shifted));
            }
            bad = Some(match bad {
                None => hazard,
                Some(prev) => self
                    .builder
                    .build(MirType::I1, Op::Bin(BinOp::Or, prev, hazard)),
            });
        }
        let bad = bad?;
        let out_nonempty = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Ugt, out_hi, out_lo));
        Some(
            self.builder
                .build(MirType::I1, Op::Bin(BinOp::And, bad, out_nonempty)),
        )
    }

    /// Run `emit(out)` directly when `hazard` is `None`; otherwise branch on it: the clear path
    /// writes the real output (`emit(None)`), the hazard path writes a fresh zeroed `f32` heap
    /// buffer of `count` elements (`emit(Some(tmp))`), copies it over the output and frees it.
    /// Returns the unit placeholder.
    fn dispatch_with_scratch(
        &mut self,
        hazard: Option<ValueId>,
        out: ValueId,
        count: ValueId,
        emit: impl Fn(&mut Self, Option<ValueId>),
    ) -> ValueId {
        let Some(hazard) = hazard else {
            emit(self, None);
            return self.const_zero(MirType::I32);
        };
        let scratch_blk = self.builder.new_block();
        let direct_blk = self.builder.new_block();
        let join = self.builder.new_block();
        self.builder
            .cond_br(hazard, scratch_blk, vec![], direct_blk, vec![]);

        self.builder.switch_to(direct_blk);
        emit(self, None);
        self.builder.br(join, vec![]);

        // A non-positive count is a kernel no-op; the allocation is clamped to zero like `alloc_f32`.
        self.builder.switch_to(scratch_blk);
        let zero = self.const_i64(0);
        let negative = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, count, zero));
        let alloc_count = self
            .builder
            .build(MirType::I64, Op::Select(negative, zero, count));
        let esize = self.const_i64(4);
        let is_float = self.const_i64(1);
        let rt_alloc = self.gemm.rt_alloc;
        let tmp = self.builder.build(
            MirType::Ptr,
            Op::Call {
                func: rt_alloc,
                args: vec![alloc_count, esize, is_float],
            },
        );
        emit(self, Some(tmp));
        self.emit_counted_loop(alloc_count, |this, i| {
            let sp = this.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: tmp,
                    index: i,
                    elem: MirType::F32,
                },
            );
            let v = this.builder.build(MirType::F32, Op::Load(sp, MirType::F32));
            let dp = this.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: out,
                    index: i,
                    elem: MirType::F32,
                },
            );
            this.builder.build_void(Op::Store { ptr: dp, value: v });
        });
        let rt_free = self.gemm.rt_free;
        self.builder.build_void(Op::Call {
            func: rt_free,
            args: vec![tmp],
        });
        self.builder.br(join, vec![]);

        self.builder.switch_to(join);
        self.const_zero(MirType::I32)
    }

    /// Lower one buffer argument of a builtin to its data pointer, plus what the extent checks
    /// need: a slice's runtime length, a fixed array's static length. Keys on the *sema* type, the
    /// same authority `kernel_operand_ptr` uses. One level of `&`/`*` (an `sdpa` operand only) is
    /// looked through: `&a` of an array is its base pointer, and a reference to a slice is the
    /// address of its fat pointer, exactly like the slice value itself.
    fn lower_builtin_buffer(&mut self, a: &Expr) -> BufArg {
        let mut ty = self.expr_ty(a);
        if let Ty::Ref { pointee, .. } | Ty::Ptr { pointee, .. } = ty {
            ty = *pointee;
        }
        let sym = match &a.kind {
            ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
            _ => None,
        };
        let v = self.lower_expr(a);
        match &ty {
            Ty::Slice(_) => {
                // The fat pointer `{ data @ 0, len @ 8 }`.
                let ptr = self.builder.build(MirType::Ptr, Op::Load(v, MirType::Ptr));
                let lp = self.field_ptr(v, SLICE_LEN_OFF);
                let len = self.builder.build(MirType::I64, Op::Load(lp, MirType::I64));
                BufArg {
                    ptr,
                    slice_len: Some(len),
                    array_len: None,
                    sym,
                }
            }
            Ty::Array { .. } => BufArg {
                ptr: v,
                slice_len: None,
                array_len: flat_f32_len(&ty),
                sym,
            },
            _ => BufArg {
                ptr: v,
                slice_len: None,
                array_len: const_tensor_len(&ty),
                sym,
            },
        }
    }

    /// `[lo, hi)` of a buffer of `count` f32 elements at `p`, as integers. The end is a `Gep`, so
    /// both backends measure in their own unit (interpreter slots, native bytes) and the overlap
    /// test is exact on each.
    fn builtin_range(&mut self, p: ValueId, count: ValueId) -> (ValueId, ValueId) {
        let end = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: p,
                index: count,
                elem: MirType::F32,
            },
        );
        let lo = self
            .builder
            .build(MirType::I64, Op::Cast(CastKind::PtrToInt, p, MirType::I64));
        let hi = self.builder.build(
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, end, MirType::I64),
        );
        (lo, hi)
    }

    /// `for i in 0..count { body(i) }` over an `i64` counter, as a block-parameter counted loop.
    fn emit_counted_loop(&mut self, count: ValueId, body: impl FnOnce(&mut Self, ValueId)) {
        let zero = self.const_i64(0);
        let header = self.builder.new_block();
        let body_blk = self.builder.new_block();
        let exit = self.builder.new_block();
        let i = self.builder.block_param(header, MirType::I64);
        self.builder.br(header, vec![zero]);
        self.builder.switch_to(header);
        let more = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, i, count));
        self.builder.cond_br(more, body_blk, vec![], exit, vec![]);
        self.builder.switch_to(body_blk);
        body(self, i);
        let one = self.const_i64(1);
        let next = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        self.builder.br(header, vec![next]);
        self.builder.switch_to(exit);
    }

    /// The `K <= 0` result of a GEMM builtin: the zero matrix, or for `linear` the bias broadcast
    /// down every row (`out[i, j] = b[j]`, written as `out[t] = b[t % N]` over the flat `M·N`).
    fn emit_empty_contraction(
        &mut self,
        out: ValueId,
        count: ValueId,
        bias: Option<(ValueId, ValueId)>,
    ) {
        self.emit_counted_loop(count, |this, t| {
            let v = match bias {
                Some((b, n)) => {
                    let j = this.builder.build(MirType::I64, Op::Bin(BinOp::SRem, t, n));
                    let bp = this.builder.build(
                        MirType::Ptr,
                        Op::Gep {
                            ptr: b,
                            index: j,
                            elem: MirType::F32,
                        },
                    );
                    this.builder.build(MirType::F32, Op::Load(bp, MirType::F32))
                }
                None => this
                    .builder
                    .build(MirType::F32, Op::ConstFloat(0.0, MirType::F32)),
            };
            let dp = this.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: out,
                    index: t,
                    elem: MirType::F32,
                },
            );
            this.builder.build_void(Op::Store { ptr: dp, value: v });
        });
    }

    /// Emit the runtime-kernel call itself. `out` overrides the output pointer (the scratch path);
    /// `out_count` is the output's element count, which the GEMM family's empty-contraction fill
    /// needs. Returns the call's value for a reduction, and a unit placeholder otherwise.
    fn emit_builtin_kernel(
        &mut self,
        call: &BuiltinCall,
        ptrs: &[Option<ValueId>],
        out: Option<ValueId>,
        dims: &[ValueId],
        eps: Option<ValueId>,
        out_count: Option<ValueId>,
    ) -> Option<ValueId> {
        use KernelBuiltin::*;
        let kind = call.kind;
        let p = |i: usize| ptrs.get(i).copied().flatten();
        let out_ptr = |i: usize| out.or_else(|| p(i));
        let par = self.parallel_fn;
        let g = self.gemm;
        let (func, args) = match kind {
            Matmul | MatmulNt | MatmulTn | Linear => {
                let (func, c_ix) = match (kind, par) {
                    (Matmul, false) => (g.mm, 2),
                    (Matmul, true) => (g.mm_par, 2),
                    (MatmulNt, false) => (g.nt, 2),
                    (MatmulNt, true) => (g.nt_par, 2),
                    (MatmulTn, false) => (g.tn, 2),
                    (MatmulTn, true) => (g.tn_par, 2),
                    (_, false) => (g.nt_epi, 3),
                    (_, true) => (g.nt_epi_par, 3),
                };
                let c = out_ptr(c_ix)?;
                let bias = if kind == Linear { Some(p(2)?) } else { None };
                let beta = self.const_i64(0);
                let mut args = vec![p(0)?, p(1)?, c, dims[0], dims[1], dims[2], beta];
                if let Some(b) = bias {
                    let act = self.const_i64(EPI_ACT_IDENTITY as i128);
                    args.push(b);
                    args.push(act);
                }
                // The kernels return without writing for `K <= 0`; the builtin means the
                // empty-contraction result there instead.
                let empty_fill = bias.map(|b| (b, dims[2]));
                match call.dims.get(1) {
                    Some(BuiltinDim::Const(0)) => {
                        self.emit_empty_contraction(c, out_count?, empty_fill);
                    }
                    Some(BuiltinDim::Const(_)) => {
                        self.builder.build_void(Op::Call { func, args });
                    }
                    _ => {
                        let zero = self.const_i64(0);
                        let nonempty = self
                            .builder
                            .build(MirType::I1, Op::Cmp(CmpOp::Sgt, dims[1], zero));
                        let kernel_blk = self.builder.new_block();
                        let fill_blk = self.builder.new_block();
                        let join = self.builder.new_block();
                        self.builder
                            .cond_br(nonempty, kernel_blk, vec![], fill_blk, vec![]);
                        self.builder.switch_to(kernel_blk);
                        self.builder.build_void(Op::Call { func, args });
                        self.builder.br(join, vec![]);
                        self.builder.switch_to(fill_blk);
                        self.emit_empty_contraction(c, out_count?, empty_fill);
                        self.builder.br(join, vec![]);
                        self.builder.switch_to(join);
                    }
                }
                return Some(self.const_zero(MirType::I32));
            }
            Transpose => {
                let func = if par { g.transpose_par } else { g.transpose };
                (func, vec![p(0)?, out_ptr(1)?, dims[0], dims[1]])
            }
            Softmax | LogSoftmax => {
                let func = if par { g.norm_par } else { g.norm };
                let op = if kind == Softmax {
                    NORM_SOFTMAX
                } else {
                    NORM_LOGSOFTMAX
                };
                let eps0 = self.const_i64(0);
                let opv = self.const_i64(op as i128);
                (func, vec![p(0)?, out_ptr(1)?, dims[0], dims[1], eps0, opv])
            }
            LayerNorm | RmsNorm => {
                let func = if par {
                    g.norm_affine_par
                } else {
                    g.norm_affine
                };
                let (beta, out_ix, op) = if kind == LayerNorm {
                    (p(2)?, 3, NORM_LAYERNORM)
                } else {
                    // An absent beta is the null pointer, built as an integer 0 (a `Ptr`-typed
                    // `ConstInt` is invalid MIR) — the affine-norm convention `emit_norm` uses.
                    (self.const_i64(0), 2, NORM_RMSNORM)
                };
                let opv = self.const_i64(op as i128);
                (
                    func,
                    vec![
                        p(0)?,
                        out_ptr(out_ix)?,
                        p(1)?,
                        beta,
                        dims[0],
                        dims[1],
                        eps?,
                        opv,
                    ],
                )
            }
            Map(act) => {
                let func = if par { g.vmath_par } else { g.vmath };
                let opv = self.const_i64(vmath_op(act) as i128);
                (func, vec![p(0)?, out_ptr(1)?, dims[0], opv])
            }
            Sum | Amax | Amin | Dot => {
                let func = if par { g.sred_par } else { g.sred };
                let op = match kind {
                    Sum => RED_SUM,
                    Amax => RED_MAX,
                    Amin => RED_MIN,
                    _ => RED_DOT,
                };
                let x = p(0)?;
                let y = if kind == Dot { p(1)? } else { x };
                let opv = self.const_i64(op as i128);
                let v = self.builder.build(
                    MirType::F32,
                    Op::Call {
                        func,
                        args: vec![x, y, dims[0], opv],
                    },
                );
                return Some(v);
            }
            Sdpa => return None,
        };
        self.builder.build_void(Op::Call { func, args });
        Some(self.const_zero(MirType::I32))
    }

    /// `sdpa(q, k, v, out, s, d, scale, causal)`. Sema has checked the operand types and, for
    /// constant `s`/`d`, every statically sized extent. Here: a constant dimension outside the
    /// kernel's signed 64-bit ABI is a codegen limit (`C0001`); a runtime `s`/`d` must lie in
    /// `0..=i32::MAX` and every operand whose length is known holds at least `s·d` elements (else
    /// the call traps like a failed `assert`); and an `out` overlapping `q`, `k` or `v` — which the
    /// kernel reads while it writes `out` — computes into scratch and copies out.
    fn lower_sdpa_builtin(&mut self, e: &Expr, args: &[Expr]) -> ValueId {
        if args.len() != 8 {
            self.unsupported(
                e.span,
                "an `sdpa` call that is not `sdpa(q, k, v, out, s, d, scale, causal)`",
            );
            return self.const_zero(MirType::I32);
        }
        for a in &args[4..6] {
            if self
                .const_runtime_u64_expr(a)
                .is_some_and(|v| v > i64::MAX as u64)
            {
                self.unsupported(
                    a.span,
                    "an `sdpa` dimension that does not fit the attention kernel's signed 64-bit ABI",
                );
                return self.const_zero(MirType::I32);
            }
        }
        let bufs: Vec<BufArg> = args[..4]
            .iter()
            .map(|a| self.lower_builtin_buffer(a))
            .collect();
        let s = self.lower_coerced(&args[4], &MirType::I64);
        let d = self.lower_coerced(&args[5], &MirType::I64);
        let scale = self.lower_coerced(&args[6], &MirType::F32);
        let causal = self.lower_coerced(&args[7], &MirType::I64);

        let s_ok = self.extent_in_range(s);
        let d_ok = self.extent_in_range(d);
        let mut checks = vec![s_ok, d_ok];
        let need = self.builder.build(MirType::I64, Op::Bin(BinOp::Mul, s, d));
        for b in &bufs {
            let have = match (b.slice_len, b.array_len) {
                (Some(len), _) => len,
                (None, Some(n)) => self.const_i64(n as i128),
                // A symbolic tensor: its extent is the caller's contract.
                (None, None) => continue,
            };
            checks.push(
                self.builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Sge, have, need)),
            );
        }
        if !self.emit_builtin_checks(checks) {
            self.unsupported(
                e.span,
                "an `sdpa` call in a program that defines its own `assert`",
            );
            return self.const_zero(MirType::I32);
        }

        let (q, k, v, out) = (bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr);
        let inputs = [(q, need, false), (k, need, false), (v, need, false)];
        let hazard = self.overlap_hazard(out, need, &inputs);
        let attn = self.gemm.attn;
        self.dispatch_with_scratch(hazard, out, need, |this, tmp| {
            let o = tmp.unwrap_or(out);
            this.builder.build_void(Op::Call {
                func: attn,
                args: vec![q, k, v, o, s, d, scale, causal],
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::lower_program;
    use wukong_span::{Interner, SourceId};

    /// Lower `src` (which must type-check) and return its verified MIR text and the diagnostics.
    fn mir(src: &str) -> (String, Vec<wukong_diag::Diagnostic>) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = lower_program(&module, &sema, &mut interner);
        let errors = wukong_mir::verify::verify_program(&prog);
        assert!(errors.is_empty(), "invalid MIR: {errors:?}");
        (wukong_mir::print::print_program(&prog, &interner), diags)
    }

    #[test]
    fn a_static_matmul_is_one_kernel_call_behind_an_overlap_test() {
        let (text, diags) = mir(
            "fn main() -> i32 { let a: [f32; 6] = [1.0; 6]; let b: [f32; 12] = [1.0; 12]; \
             let mut c: [f32; 8] = [0.0; 8]; matmul::<2, 3, 4>(a, b, c); return 0; }",
        );
        assert!(diags.is_empty(), "{diags:?}");
        // The direct call and the scratch-path call, and the scratch round trip.
        assert_eq!(text.matches("call wukong_sgemm(").count(), 2, "{text}");
        assert!(text.contains("call wukong_rt_alloc("), "{text}");
        assert!(text.contains("call wukong_rt_free("), "{text}");
        // Every extent is a type-level fact: no runtime extent check.
        assert!(!text.contains("call assert("), "{text}");
    }

    #[test]
    fn an_in_place_activation_spelled_with_one_name_needs_no_overlap_test() {
        let (text, diags) = mir(
            "fn main() -> i32 { let mut x: [f32; 4] = [1.0; 4]; gelu(x, x); \
             return x[0] as i32; }",
        );
        assert!(diags.is_empty(), "{diags:?}");
        assert_eq!(text.matches("call wukong_vmath_f32(").count(), 1, "{text}");
        assert!(!text.contains("wukong_rt_alloc"), "{text}");
    }

    #[test]
    fn slice_operands_are_checked_at_run_time() {
        let (text, diags) = mir("fn f(x: []f32, y: []f32) -> f32 { return dot(x, y); }");
        assert!(diags.is_empty(), "{diags:?}");
        assert!(text.contains("call wukong_sreduce_f32("), "{text}");
        // The extent of `x` must be a sane size, and `y` must hold exactly that many elements.
        assert_eq!(text.matches("call assert(").count(), 2, "{text}");
    }

    #[test]
    fn a_runtime_contraction_guards_the_empty_case() {
        let (text, diags) = mir(
            "fn f(a: []f32, b: []f32, mut c: []f32, m: i64, k: i64, n: i64) { \
             matmul_nt::<m, k, n>(a, b, c); }",
        );
        assert!(diags.is_empty(), "{diags:?}");
        assert!(text.contains("call wukong_sgemm_nt("), "{text}");
        // Three dimension range checks and three exact-length checks.
        assert_eq!(text.matches("call assert(").count(), 6, "{text}");
        assert!(text.contains("sgt"), "the K > 0 guard: {text}");
    }

    #[test]
    fn sdpa_is_guarded_and_lowered_to_the_attention_kernel() {
        let (text, diags) =
            mir("fn f(q: []f32, mut o: []f32, s: i64, d: i64) { sdpa(q, q, q, o, s, d, 0.5, 1); }");
        assert!(diags.is_empty(), "{diags:?}");
        assert_eq!(
            text.matches("call wukong_attention_f32(").count(),
            2,
            "{text}"
        );
        // s and d range checks, then `q` (three times) and `o` long enough for s*d.
        assert_eq!(text.matches("call assert(").count(), 6, "{text}");
    }

    #[test]
    fn an_sdpa_dimension_outside_the_abi_is_a_codegen_limit() {
        let mut interner = Interner::new();
        let src = "fn main() -> i32 { let q: [f32; 4] = [0.0; 4]; let mut o: [f32; 4] = [0.0; 4]; \
                   sdpa(q, q, q, o, 9223372036854775808u64, 2, 1.0, 0); return 0; }";
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (_, diags) = lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().any(|d| d.code == Some("C0001")), "{diags:?}");
    }
}
