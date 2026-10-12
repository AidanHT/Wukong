//! Lowering of the ML I/O and randomness builtins that `wukong_sema::ml_builtins` types: typed file
//! I/O over every element type, the safetensors / `.npy` / GGUF container readers and the GGUF
//! tensor-table queries, the Philox4x32-10 counter-based RNG, and the top-k / argsort /
//! categorical-sampling kernels.
//!
//! **Calls.** Every builtin but the RNG scalars lowers to one opaque `Op::Call` of a
//! `wukong_runtime` entry. Integer operands are widened to `i64` (by their own signedness), float
//! operands to `f32`, and a buffer operand — a `[]T` slice, a `[T; N]` array or a contiguous
//! `Tensor[T, …]` — becomes its data pointer plus its `i64` element count (the slice's fat-pointer
//! fields, the array's length, the product of the tensor's dims). A container read into a tensor
//! also passes the tensor's dims (an `[i64; rank]` stack buffer and the rank) so the runtime can
//! check the file's shape against them; a flat destination passes rank `-1`. Every call site of one
//! entry therefore has the same MIR signature, which the textual-LLVM backend relies on.
//!
//! **Elements.** A buffer operand holds exactly the element the builtin moves
//! (`wukong_sema::MlBuiltin::buffer_elem`). Sema checks it, deferring a dtype-generic
//! `Tensor[T, ..]` operand to each instantiation (`E0416`); an instance its bounded re-check never
//! reached is refused here as a codegen limit (`C0001`) rather than handing the runtime a buffer of
//! another element.
//!
//! The six original `read_<T>` / `write_<T>` element types keep their frozen per-type symbols
//! (`wukong_rt_read_f32`, …), so their MIR is unchanged; every other element type and every
//! offset read goes through the dtype-coded `wukong_rt_read_dt` / `wukong_rt_write_dt`.
//!
//! **Maps.** `map_<T>` / `st_map_<T>` / `npy_map_<T>` / `gguf_map_<T>` lower to one
//! `wukong_rt_map(path, name, container, dtype, &len) -> data` call (a whole-file map passes the
//! path again for the unused name) and a fresh `{ data, len }` slice — the shape `alloc_<T>` builds,
//! so the result is freed by `free`.
//!
//! **RNG.** `rand_u32` / `rand_f32` / `randn_f32` lower to plain integer MIR — the Philox round is a
//! 32×32→64 multiply (`zext`, `mul`, `lshr`, `trunc`), xors and adds — so they need no runtime call,
//! are bit-identical on every backend by construction, and are safe in a `@parallel` region (no
//! state). `randn_f32` adds a Box–Muller step on the compiler's own `log` polynomial and scalar
//! `cos`, both of which already agree bit-for-bit across backends.

use wukong_ast::Expr;
use wukong_mir::{BinOp, CastKind, MirType, Op, ValueId};
use wukong_sema::{io_dtype_code, ml_builtin, ml_builtin_is_pure, DefKind, GgufQuery, MlBuiltin};
use wukong_span::{Interner, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{FnLowerer, SLICE_LEN_OFF, SLICE_PTR_OFF, SLICE_SIZE};

/// Philox4x32 round multipliers and key increments — `wukong_runtime::random`'s constants (this
/// crate does not depend on the runtime; the interpreter's differential test pins the lowering to
/// the runtime reference).
const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// `2⁻²⁴`: `rand_f32` scales the top 24 bits of a Philox word by it.
const INV_2_24: f64 = 1.0 / 16_777_216.0;

/// The runtime entry points this module lowers to, interned once at the top of `lower_program`
/// (`FnLowerer` holds only a shared `&Interner`) and carried on `GemmSyms`.
#[derive(Clone, Copy)]
pub(crate) struct MlSyms {
    read_dt: Symbol,
    write_dt: Symbol,
    file_size: Symbol,
    st_read: Symbol,
    npy_read: Symbol,
    gguf_read: Symbol,
    gguf_query: Symbol,
    map: Symbol,
    rand_fill: Symbol,
    topk: Symbol,
    argsort: Symbol,
    sample_categorical: Symbol,
}

impl MlSyms {
    pub(crate) fn new(interner: &mut Interner) -> MlSyms {
        MlSyms {
            read_dt: interner.intern("wukong_rt_read_dt"),
            write_dt: interner.intern("wukong_rt_write_dt"),
            file_size: interner.intern("wukong_rt_file_size"),
            st_read: interner.intern("wukong_rt_st_read"),
            npy_read: interner.intern("wukong_rt_npy_read"),
            gguf_read: interner.intern("wukong_rt_gguf_read"),
            gguf_query: interner.intern("wukong_rt_gguf_query"),
            map: interner.intern("wukong_rt_map"),
            rand_fill: interner.intern("wukong_rand_fill_f32"),
            topk: interner.intern("wukong_topk_f32"),
            argsort: interner.intern("wukong_argsort_f32"),
            sample_categorical: interner.intern("wukong_sample_categorical_f32"),
        }
    }
}

/// A lowered buffer operand: its data pointer, its `i64` element count, and — for a tensor — the
/// `i64` value of each dimension.
pub(crate) struct MlBuf {
    pub(crate) data: ValueId,
    pub(crate) len: ValueId,
    dims: Option<Vec<ValueId>>,
}

impl FnLowerer<'_> {
    /// Whether the call `name(…)` is a pure RNG builtin (not shadowed by a user function), which a
    /// `@parallel` region may evaluate per iteration in any order.
    pub(crate) fn is_pure_ml_builtin(&self, name: Symbol) -> bool {
        ml_builtin_is_pure(self.interner.resolve(name))
            && !matches!(
                self.sema.defs.lookup(name).map(|d| &d.kind),
                Some(DefKind::Fn(_))
            )
    }

    /// Lower a call to a builtin of this family, or return `None` for any other callee (or an
    /// operand shape sema should have rejected), so `lower_call` falls through to its other paths.
    /// A user function of the same name never reaches here: `lower_call` resolves it first.
    pub(crate) fn lower_ml_builtin(&mut self, name: Symbol, args: &[Expr]) -> Option<ValueId> {
        let interner = self.interner;
        let b = ml_builtin(interner.resolve(name))?;
        let syms = self.gemm.ml;
        if let Some(bad) = (0..args.len()).find(|&i| {
            b.buffer_elem(i)
                .is_some_and(|t| !self.ml_operand_holds(&args[i], t))
        }) {
            self.unsupported(
                args[bad].span,
                "an ML-builtin buffer operand whose element is not the one the builtin moves (an \
                 instantiation the type checker did not re-check)",
            );
            // The placeholder has the builtin's result type.
            return Some(match b {
                MlBuiltin::RandFill => self.const_zero(MirType::I32),
                _ => self.ml_i64_const(0),
            });
        }
        match b {
            MlBuiltin::Read(t) | MlBuiltin::Write(t) => {
                if args.len() != 2 {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                let buf = self.ml_buffer(&args[1])?;
                let read = matches!(b, MlBuiltin::Read(_));
                if let Some(func) = self.ml_frozen_symbol(t, read) {
                    return Some(self.ml_call(func, vec![path, buf.data, buf.len]));
                }
                let dt = self.ml_i64_const(io_dtype_code(t));
                Some(if read {
                    let zero = self.ml_i64_const(0);
                    self.ml_call(syms.read_dt, vec![path, zero, buf.data, buf.len, dt])
                } else {
                    self.ml_call(syms.write_dt, vec![path, buf.data, buf.len, dt])
                })
            }
            MlBuiltin::ReadAt(t) => {
                if args.len() != 3 {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                let off = self.ml_i64_arg(&args[1]);
                let buf = self.ml_buffer(&args[2])?;
                let dt = self.ml_i64_const(io_dtype_code(t));
                Some(self.ml_call(syms.read_dt, vec![path, off, buf.data, buf.len, dt]))
            }
            MlBuiltin::FileSize => {
                if args.len() != 1 {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                Some(self.ml_call(syms.file_size, vec![path]))
            }
            MlBuiltin::StRead(t) | MlBuiltin::GgufRead(t) => {
                if args.len() != 3 {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                let tensor = self.lower_expr(&args[1]);
                let buf = self.ml_buffer(&args[2])?;
                let dt = self.ml_i64_const(io_dtype_code(t));
                let (dims, rank) = self.ml_dims_arg(&buf);
                let func = match b {
                    MlBuiltin::StRead(_) => syms.st_read,
                    _ => syms.gguf_read,
                };
                Some(self.ml_call(func, vec![path, tensor, buf.data, buf.len, dt, dims, rank]))
            }
            MlBuiltin::GgufQuery(q) => {
                let dim = q == GgufQuery::Dim;
                if args.len() != 2 + usize::from(dim) {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                let tensor = self.lower_expr(&args[1]);
                let what = self.ml_i64_const(q.code());
                let k = if dim {
                    self.ml_i64_arg(&args[2])
                } else {
                    self.ml_i64_const(0)
                };
                Some(self.ml_call(syms.gguf_query, vec![path, tensor, what, k]))
            }
            MlBuiltin::Map(src, t) => {
                if args.len() != 1 + usize::from(src.named()) {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                // The whole-file maps have no name; the runtime reads the operand only for the
                // named containers, so the path stands in for it.
                let name = if src.named() {
                    self.lower_expr(&args[1])
                } else {
                    path
                };
                let kind = self.ml_i64_const(src.code());
                let dt = self.ml_i64_const(io_dtype_code(t));
                let len_slot = self.builder.alloca(MirType::I64);
                let data = self.builder.build(
                    MirType::Ptr,
                    Op::Call {
                        func: syms.map,
                        args: vec![path, name, kind, dt, len_slot],
                    },
                );
                let len = self
                    .builder
                    .build(MirType::I64, Op::Load(len_slot, MirType::I64));
                Some(self.ml_slice(data, len))
            }
            MlBuiltin::NpyRead(t) => {
                if args.len() != 2 {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                let buf = self.ml_buffer(&args[1])?;
                let dt = self.ml_i64_const(io_dtype_code(t));
                let (dims, rank) = self.ml_dims_arg(&buf);
                Some(self.ml_call(syms.npy_read, vec![path, buf.data, buf.len, dt, dims, rank]))
            }
            MlBuiltin::RandU32 | MlBuiltin::RandF32 | MlBuiltin::RandnF32 => {
                if args.len() != 2 {
                    return None;
                }
                let seed = self.ml_i64_arg(&args[0]);
                let ctr = self.ml_i64_arg(&args[1]);
                let words = self.ml_philox(seed, ctr);
                Some(match b {
                    MlBuiltin::RandU32 => words[0],
                    MlBuiltin::RandF32 => self.ml_unit_f32(words[0], 0),
                    _ => self.ml_box_muller(words[0], words[1]),
                })
            }
            MlBuiltin::RandFill => {
                if args.len() != 3 {
                    return None;
                }
                let buf = self.ml_buffer(&args[0])?;
                let seed = self.ml_i64_arg(&args[1]);
                let ctr = self.ml_i64_arg(&args[2]);
                self.builder.build_void(Op::Call {
                    func: syms.rand_fill,
                    args: vec![buf.data, buf.len, seed, ctr],
                });
                Some(self.const_zero(MirType::I32))
            }
            MlBuiltin::TopK => {
                if args.len() != 3 {
                    return None;
                }
                let x = self.ml_buffer(&args[0])?;
                let vals = self.ml_buffer(&args[1])?;
                let idx = self.ml_buffer(&args[2])?;
                Some(self.ml_call(
                    syms.topk,
                    vec![x.data, x.len, vals.data, vals.len, idx.data, idx.len],
                ))
            }
            MlBuiltin::Argsort { descending } => {
                if args.len() != 2 {
                    return None;
                }
                let x = self.ml_buffer(&args[0])?;
                let idx = self.ml_buffer(&args[1])?;
                let desc = self.ml_i64_const(descending as i64);
                Some(self.ml_call(syms.argsort, vec![x.data, x.len, idx.data, idx.len, desc]))
            }
            MlBuiltin::SampleCategorical => {
                if args.len() != 2 {
                    return None;
                }
                let p = self.ml_buffer(&args[0])?;
                let u = self.ml_f32_arg(&args[1]);
                Some(self.ml_call(syms.sample_categorical, vec![p.data, p.len, u]))
            }
        }
    }

    /// The frozen per-type symbol of an original `read_<T>` / `write_<T>` element type, if `t` is
    /// one of the six.
    fn ml_frozen_symbol(&self, t: Scalar, read: bool) -> Option<Symbol> {
        let g = &self.gemm;
        Some(match (t, read) {
            (Scalar::F32, true) => g.rt_read_f32,
            (Scalar::F64, true) => g.rt_read_f64,
            (Scalar::I32, true) => g.rt_read_i32,
            (Scalar::I64, true) => g.rt_read_i64,
            (Scalar::I8, true) => g.rt_read_i8,
            (Scalar::U8, true) => g.rt_read_u8,
            (Scalar::F32, false) => g.rt_write_f32,
            (Scalar::F64, false) => g.rt_write_f64,
            (Scalar::I32, false) => g.rt_write_i32,
            (Scalar::I64, false) => g.rt_write_i64,
            (Scalar::I8, false) => g.rt_write_i8,
            (Scalar::U8, false) => g.rt_write_u8,
            _ => return None,
        })
    }

    /// A fresh 16-byte slice fat pointer `{ data, len }` — the value-position slice convention
    /// `alloc_<T>` builds, so the result binds, passes, indexes and `free`s like any `[]T`.
    fn ml_slice(&mut self, data: ValueId, len: ValueId) -> ValueId {
        let sbuf = self
            .builder
            .alloca(MirType::Array(Box::new(MirType::I8), SLICE_SIZE as u32));
        let dp = self.field_ptr(sbuf, SLICE_PTR_OFF);
        self.builder.build_void(Op::Store {
            ptr: dp,
            value: data,
        });
        let lp = self.field_ptr(sbuf, SLICE_LEN_OFF);
        self.builder.build_void(Op::Store {
            ptr: lp,
            value: len,
        });
        sbuf
    }

    /// An `i64`-returning runtime call.
    pub(crate) fn ml_call(&mut self, func: Symbol, args: Vec<ValueId>) -> ValueId {
        self.builder.build(MirType::I64, Op::Call { func, args })
    }

    pub(crate) fn ml_i64_const(&mut self, v: i64) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(v as i128, MirType::I64))
    }

    /// An integer operand widened to `i64` by its own signedness — the bits `x as u64` has.
    pub(crate) fn ml_i64_arg(&mut self, e: &Expr) -> ValueId {
        let from = self.expr_mir(e);
        let signed = self.signed(e);
        let v = self.lower_expr(e);
        self.coerce_to(v, &from, &MirType::I64, signed)
    }

    /// A float operand converted to `f32`.
    pub(crate) fn ml_f32_arg(&mut self, e: &Expr) -> ValueId {
        let from = self.expr_mir(e);
        let signed = self.signed(e);
        let v = self.lower_expr(e);
        self.coerce_to(v, &from, &MirType::F32, signed)
    }

    /// Does buffer operand `e` hold `want` elements once this instance's types are substituted?
    /// Sema guarantees it — a dtype-generic `Tensor[T, ..]` reaches one of these builtins only from
    /// an instantiation binding `T = want` (`E0416` otherwise) — so this is the backstop for an
    /// instance sema did not re-check (`instantiate.rs` re-checks a bounded number), never a second
    /// rule: the runtime reads the buffer's bytes as `want`. The kernel builtins' twin is
    /// `builtins.rs`'s `builtin_operands_are_f32`.
    fn ml_operand_holds(&self, e: &Expr, want: Scalar) -> bool {
        match self.expr_ty(e) {
            Ty::Tensor { elem, elem_var, .. } => elem == want && elem_var.is_none(),
            Ty::Array { elem, .. } | Ty::Slice(elem) => *elem == Ty::Scalar(want),
            // Not a buffer sema would have accepted; `ml_buffer` declines it as before.
            _ => true,
        }
    }

    /// Lower a buffer operand (a slice, an array or a contiguous tensor) to its data pointer and
    /// element count. `None` for any other type, or a tensor dimension with no value in scope.
    pub(crate) fn ml_buffer(&mut self, e: &Expr) -> Option<MlBuf> {
        match self.expr_ty(e) {
            Ty::Slice(_) => {
                let sbuf = self.lower_expr(e);
                let dp = self.field_ptr(sbuf, SLICE_PTR_OFF);
                let data = self.builder.build(MirType::Ptr, Op::Load(dp, MirType::Ptr));
                let lp = self.field_ptr(sbuf, SLICE_LEN_OFF);
                let len = self.builder.build(MirType::I64, Op::Load(lp, MirType::I64));
                Some(MlBuf {
                    data,
                    len,
                    dims: None,
                })
            }
            Ty::Array { len, .. } => {
                let n = i64::try_from(len).ok()?;
                // An array operand is its base pointer (the by-pointer aggregate convention).
                let data = self.lower_expr(e);
                let len = self.ml_i64_const(n);
                Some(MlBuf {
                    data,
                    len,
                    dims: None,
                })
            }
            Ty::Tensor { shape, .. } => {
                let data = self.lower_expr(e);
                let mut dims = Vec::with_capacity(shape.0.len());
                for d in &shape.0 {
                    dims.push(self.tensor_dim_value(d)?);
                }
                let mut len = self.ml_i64_const(1);
                for &d in &dims {
                    len = self
                        .builder
                        .build(MirType::I64, Op::Bin(BinOp::Mul, len, d));
                }
                Some(MlBuf {
                    data,
                    len,
                    dims: Some(dims),
                })
            }
            _ => None,
        }
    }

    /// The `(dims, rank)` operands of a container read: a tensor destination's dims stored into a
    /// fresh `[i64; rank]` stack buffer and its rank; a flat destination's rank `-1` (the one-slot
    /// buffer is never read).
    fn ml_dims_arg(&mut self, buf: &MlBuf) -> (ValueId, ValueId) {
        let dims: &[ValueId] = buf.dims.as_deref().unwrap_or(&[]);
        let slots = dims.len().max(1) as u32;
        let base = self
            .builder
            .alloca(MirType::Array(Box::new(MirType::I64), slots));
        for (k, &d) in dims.iter().enumerate() {
            let index = self.ml_i64_const(k as i64);
            let ptr = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base,
                    index,
                    elem: MirType::I64,
                },
            );
            self.builder.build_void(Op::Store { ptr, value: d });
        }
        let rank = match &buf.dims {
            Some(d) => d.len() as i64,
            None => -1,
        };
        let rank = self.ml_i64_const(rank);
        (base, rank)
    }

    fn ml_i32_const(&mut self, v: u32) -> ValueId {
        // The canonical (sign-extended) image of the 32-bit pattern, as the interpreter masks it.
        self.builder
            .build(MirType::I32, Op::ConstInt(v as i32 as i128, MirType::I32))
    }

    /// The low 32 bits of an `i64`.
    fn ml_lo32(&mut self, v: ValueId) -> ValueId {
        self.builder
            .build(MirType::I32, Op::Cast(CastKind::Trunc, v, MirType::I32))
    }

    /// The high 32 bits of an `i64` (a logical shift, so the sign is not smeared in).
    fn ml_hi32(&mut self, v: ValueId) -> ValueId {
        let k = self.ml_i64_const(32);
        let s = self.builder.build(MirType::I64, Op::Bin(BinOp::LShr, v, k));
        self.ml_lo32(s)
    }

    /// `(hi, lo)` of the 64-bit product `m · x` of two 32-bit unsigned values: both factors are
    /// zero-extended, so the `i64` product is exact (it is below 2⁶⁴) and wrapping only reinterprets
    /// its sign.
    fn ml_mulhilo(&mut self, m: u32, x: ValueId) -> (ValueId, ValueId) {
        let xz = self
            .builder
            .build(MirType::I64, Op::Cast(CastKind::ZExt, x, MirType::I64));
        let mc = self.ml_i64_const(i64::from(m));
        let p = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Mul, xz, mc));
        let hi = self.ml_hi32(p);
        let lo = self.ml_lo32(p);
        (hi, lo)
    }

    fn ml_xor32(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.builder.build(MirType::I32, Op::Bin(BinOp::Xor, a, b))
    }

    /// The Philox4x32-10 block for `(seed, ctr)` (both `i64`): key `(seed lo, seed hi)`, counter
    /// `(ctr lo, ctr hi, 0, 0)`, ten rounds, the key bumped before every round after the first —
    /// `wukong_runtime::random::philox_block`, op for op.
    fn ml_philox(&mut self, seed: ValueId, ctr: ValueId) -> [ValueId; 4] {
        let zero = self.ml_i32_const(0);
        let c0 = self.ml_lo32(ctr);
        let c1 = self.ml_hi32(ctr);
        let mut c = [c0, c1, zero, zero];
        let k0 = self.ml_lo32(seed);
        let k1 = self.ml_hi32(seed);
        let mut k = [k0, k1];
        for round in 0..10 {
            if round > 0 {
                let w0 = self.ml_i32_const(PHILOX_W0);
                let w1 = self.ml_i32_const(PHILOX_W1);
                let n0 = self
                    .builder
                    .build(MirType::I32, Op::Bin(BinOp::Add, k[0], w0));
                let n1 = self
                    .builder
                    .build(MirType::I32, Op::Bin(BinOp::Add, k[1], w1));
                k = [n0, n1];
            }
            let (hi0, lo0) = self.ml_mulhilo(PHILOX_M0, c[0]);
            let (hi1, lo1) = self.ml_mulhilo(PHILOX_M1, c[2]);
            let t0 = self.ml_xor32(hi1, c[1]);
            let n0 = self.ml_xor32(t0, k[0]);
            let t2 = self.ml_xor32(hi0, c[3]);
            let n2 = self.ml_xor32(t2, k[1]);
            c = [n0, lo1, n2, lo0];
        }
        c
    }

    /// `((w >> 8) + bias) · 2⁻²⁴` as an `f32` — exact, since the integer is at most 2²⁴. Bias 0 is
    /// `rand_f32`'s `[0, 1)`; bias 1 gives the `(0, 1]` Box–Muller needs under its `log`.
    fn ml_unit_f32(&mut self, w: ValueId, bias: u32) -> ValueId {
        let eight = self.ml_i32_const(8);
        let mut top = self
            .builder
            .build(MirType::I32, Op::Bin(BinOp::LShr, w, eight));
        if bias != 0 {
            let b = self.ml_i32_const(bias);
            top = self
                .builder
                .build(MirType::I32, Op::Bin(BinOp::Add, top, b));
        }
        let f = self
            .builder
            .build(MirType::F32, Op::Cast(CastKind::SiToFp, top, MirType::F32));
        let scale = self
            .builder
            .build(MirType::F32, Op::ConstFloat(INV_2_24, MirType::F32));
        self.builder
            .build(MirType::F32, Op::Bin(BinOp::FMul, f, scale))
    }

    /// `randn_f32`: one Box–Muller sample `sqrt(-2·ln u1) · cos(2π·u2)` from words 0 and 1 of the
    /// block, with `u1 ∈ (0, 1]` (so the log is finite) and `u2 ∈ [0, 1)`. `-2·ln u1` is spelled
    /// `0 − 2·ln u1` so `u1 = 1` gives `+0`, never a `-0.0` sample.
    fn ml_box_muller(&mut self, w0: ValueId, w1: ValueId) -> ValueId {
        let f32t = MirType::F32;
        let u1 = self.ml_unit_f32(w0, 1);
        let u2 = self.ml_unit_f32(w1, 0);
        let ln = self.emit_log(u1, &f32t);
        let two = self
            .builder
            .build(f32t.clone(), Op::ConstFloat(2.0, f32t.clone()));
        let two_ln = self
            .builder
            .build(f32t.clone(), Op::Bin(BinOp::FMul, two, ln));
        let zero = self
            .builder
            .build(f32t.clone(), Op::ConstFloat(0.0, f32t.clone()));
        let radicand = self
            .builder
            .build(f32t.clone(), Op::Bin(BinOp::FSub, zero, two_ln));
        let r = self.builder.build(f32t.clone(), Op::Sqrt(radicand));
        let tau = self.builder.build(
            f32t.clone(),
            Op::ConstFloat(std::f32::consts::TAU as f64, f32t.clone()),
        );
        let angle = self
            .builder
            .build(f32t.clone(), Op::Bin(BinOp::FMul, u2, tau));
        let cos = self.emit_trig(angle, &f32t, true);
        self.builder.build(f32t, Op::Bin(BinOp::FMul, r, cos))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::lower_program;
    use wukong_mir::{Op, Program};
    use wukong_span::{Interner, SourceId};

    /// A module whose `main` first requires 512 distinct instantiations of a trivial generic — the
    /// whole budget `wukong_sema::instantiate` re-checks (`MAX_INSTANCES`) — and then runs `tail`.
    /// Every instantiation `tail` requires is lowered (mir_build's monomorphizer has no budget) but
    /// never re-checked by sema, which is exactly the instance a lowering backstop exists for.
    pub(crate) fn past_the_recheck_budget(decls: &str, tail: &str) -> String {
        let mut src =
            format!("fn id<T>(x: T) -> i32 {{ return 0; }}\n{decls}\nfn main() -> i32 {{\n");
        for n in 1..=512 {
            src.push_str(&format!(
                "    let a{n}: [f32; {n}] = [0.0; {n}];\n    id(a{n});\n"
            ));
        }
        src.push_str(tail);
        src.push_str("\n}\n");
        src
    }

    /// Check and lower `src`: sema's error codes, lowering's error codes, and the program.
    pub(crate) fn check_and_lower(
        src: &str,
    ) -> (Vec<&'static str>, Vec<&'static str>, Program, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        let codes = |d: &[wukong_diag::Diagnostic]| -> Vec<&'static str> {
            d.iter()
                .filter(|d| d.is_error())
                .filter_map(|d| d.code)
                .collect()
        };
        let sema_codes = codes(&sdiags);
        let (prog, ldiags) = lower_program(&module, &sema, &mut interner);
        (sema_codes, codes(&ldiags), prog, interner)
    }

    pub(crate) fn calls(prog: &Program, interner: &Interner, callee: &str) -> usize {
        prog.funcs
            .iter()
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(&i.op, Op::Call { func, .. } if interner.resolve(*func) == callee))
            .count()
    }

    /// N18's typed-I/O half defers a `Tensor[T, ..]` operand's element to sema's per-instance
    /// re-check (`E0416`), and that re-check stops after a budget of instances. Past it, an
    /// instance binding `T = f64` reached `write_f32` unchecked and lowered to
    /// `wukong_rt_write_f32(path, <f64 buffer>, 4)` — the f64 bytes written as f32 natively, four
    /// f64 slots read by the interpreter. The lowering must refuse it as a codegen limit (`C0001`)
    /// rather than emit the call, as the kernel builtins already do.
    #[test]
    fn an_unchecked_instance_never_reaches_a_typed_io_builtin_at_the_wrong_element() {
        let decls = "fn save<T>(x: Tensor[T, 4]) -> i64 { return write_f32(\"o.bin\", x); }\n\
                     fn load<T>(mut y: Tensor[T, 4]) -> i64 { return read_f32(\"o.bin\", y); }";
        let tail = "    let x: [f64; 4] = [1.0; 4];\n    let mut y: [f64; 4] = [0.0; 4];\n    \
                    return (save(x) + load(y)) as i32;";
        let (sema, lower, prog, it) = check_and_lower(&past_the_recheck_budget(decls, tail));
        assert!(
            sema.is_empty(),
            "the premise: sema's re-check budget is spent before `save`/`load` ({sema:?})"
        );
        assert_eq!(
            lower,
            ["C0001", "C0001"],
            "one refusal per wrong-element operand"
        );
        assert_eq!(calls(&prog, &it, "wukong_rt_write_f32"), 0);
        assert_eq!(calls(&prog, &it, "wukong_rt_read_f32"), 0);
    }

    /// Control: the same budget-spent module at `T = f32` lowers cleanly to the runtime calls.
    #[test]
    fn an_unchecked_instance_at_the_builtins_element_still_lowers() {
        let decls = "fn save<T>(x: Tensor[T, 4]) -> i64 { return write_f32(\"o.bin\", x); }";
        let tail = "    let x: [f32; 4] = [1.0; 4];\n    return save(x) as i32;";
        let (sema, lower, prog, it) = check_and_lower(&past_the_recheck_budget(decls, tail));
        assert!(sema.is_empty(), "{sema:?}");
        assert!(lower.is_empty(), "{lower:?}");
        assert_eq!(calls(&prog, &it, "wukong_rt_write_f32"), 1);
    }
}
