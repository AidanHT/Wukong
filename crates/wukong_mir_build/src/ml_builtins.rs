//! Lowering of the ML I/O and randomness builtins that `wukong_sema::ml_builtins` types: typed file
//! I/O over every element type, the safetensors / `.npy` container readers, and the Philox4x32-10
//! counter-based RNG.
//!
//! **Calls.** Every builtin but the RNG scalars lowers to one opaque `Op::Call` of a
//! `wukong_runtime` entry. Integer operands are widened to `i64` (by their own signedness), and a
//! buffer operand — a `[]T` slice, a `[T; N]` array or a contiguous
//! `Tensor[T, …]` — becomes its data pointer plus its `i64` element count (the slice's fat-pointer
//! fields, the array's length, the product of the tensor's dims). A container read into a tensor
//! also passes the tensor's dims (an `[i64; rank]` stack buffer and the rank) so the runtime can
//! check the file's shape against them; a flat destination passes rank `-1`. Every call site of one
//! entry therefore has the same MIR signature, which the textual-LLVM backend relies on.
//!
//! The six original `read_<T>` / `write_<T>` element types keep their frozen per-type symbols
//! (`wukong_rt_read_f32`, …), so their MIR is unchanged; every other element type and every
//! offset read goes through the dtype-coded `wukong_rt_read_dt` / `wukong_rt_write_dt`.
//!
//! **RNG.** `rand_u32` / `rand_f32` / `randn_f32` lower to plain integer MIR — the Philox round is a
//! 32×32→64 multiply (`zext`, `mul`, `lshr`, `trunc`), xors and adds — so they need no runtime call,
//! are bit-identical on every backend by construction, and are safe in a `@parallel` region (no
//! state). `randn_f32` adds a Box–Muller step on the compiler's own `log` polynomial and scalar
//! `cos`, both of which already agree bit-for-bit across backends.

use wukong_ast::Expr;
use wukong_mir::{BinOp, CastKind, MirType, Op, ValueId};
use wukong_sema::{io_dtype_code, ml_builtin, ml_builtin_is_pure, DefKind, MlBuiltin};
use wukong_span::{Interner, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{FnLowerer, SLICE_LEN_OFF, SLICE_PTR_OFF};

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
    rand_fill: Symbol,
}

impl MlSyms {
    pub(crate) fn new(interner: &mut Interner) -> MlSyms {
        MlSyms {
            read_dt: interner.intern("wukong_rt_read_dt"),
            write_dt: interner.intern("wukong_rt_write_dt"),
            file_size: interner.intern("wukong_rt_file_size"),
            st_read: interner.intern("wukong_rt_st_read"),
            npy_read: interner.intern("wukong_rt_npy_read"),
            rand_fill: interner.intern("wukong_rand_fill_f32"),
        }
    }
}

/// A lowered buffer operand: its data pointer, its `i64` element count, and — for a tensor — the
/// `i64` value of each dimension.
struct MlBuf {
    data: ValueId,
    len: ValueId,
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
            MlBuiltin::StRead(t) => {
                if args.len() != 3 {
                    return None;
                }
                let path = self.lower_expr(&args[0]);
                let tensor = self.lower_expr(&args[1]);
                let buf = self.ml_buffer(&args[2])?;
                let dt = self.ml_i64_const(io_dtype_code(t));
                let (dims, rank) = self.ml_dims_arg(&buf);
                Some(self.ml_call(
                    syms.st_read,
                    vec![path, tensor, buf.data, buf.len, dt, dims, rank],
                ))
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

    /// An `i64`-returning runtime call.
    fn ml_call(&mut self, func: Symbol, args: Vec<ValueId>) -> ValueId {
        self.builder.build(MirType::I64, Op::Call { func, args })
    }

    fn ml_i64_const(&mut self, v: i64) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(v as i128, MirType::I64))
    }

    /// An integer operand widened to `i64` by its own signedness — the bits `x as u64` has.
    fn ml_i64_arg(&mut self, e: &Expr) -> ValueId {
        let from = self.expr_mir(e);
        let signed = self.signed(e);
        let v = self.lower_expr(e);
        self.coerce_to(v, &from, &MirType::I64, signed)
    }

    /// Lower a buffer operand (a slice, an array or a contiguous tensor) to its data pointer and
    /// element count. `None` for any other type, or a tensor dimension with no value in scope.
    fn ml_buffer(&mut self, e: &Expr) -> Option<MlBuf> {
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
                for d in shape.0 {
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
