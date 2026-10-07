//! Lowering of the ML I/O builtins that `wukong_sema::ml_builtins` types: typed file I/O over every
//! element type and the safetensors / `.npy` container readers.
//!
//! **Calls.** Every builtin lowers to one opaque `Op::Call` of a
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

use wukong_ast::Expr;
use wukong_mir::{BinOp, MirType, Op, ValueId};
use wukong_sema::{io_dtype_code, ml_builtin, MlBuiltin};
use wukong_span::{Interner, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{FnLowerer, SLICE_LEN_OFF, SLICE_PTR_OFF};

/// The runtime entry points this module lowers to, interned once at the top of `lower_program`
/// (`FnLowerer` holds only a shared `&Interner`) and carried on `GemmSyms`.
#[derive(Clone, Copy)]
pub(crate) struct MlSyms {
    read_dt: Symbol,
    write_dt: Symbol,
    file_size: Symbol,
    st_read: Symbol,
    npy_read: Symbol,
}

impl MlSyms {
    pub(crate) fn new(interner: &mut Interner) -> MlSyms {
        MlSyms {
            read_dt: interner.intern("wukong_rt_read_dt"),
            write_dt: interner.intern("wukong_rt_write_dt"),
            file_size: interner.intern("wukong_rt_file_size"),
            st_read: interner.intern("wukong_rt_st_read"),
            npy_read: interner.intern("wukong_rt_npy_read"),
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
}
