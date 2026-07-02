//! `VecKernel` — a backend-agnostic recipe for a vectorizable straight-line loop body, produced by
//! the general vectorizer (`mercury_mir_build`) and consumed by two backends that must agree
//! bit-for-bit:
//!
//! * the **interpreter** marshals it lane-by-lane through its abstract slot memory (the oracle), and
//! * the **Cranelift** backend assembles it into true 256-bit AVX2 machine code — the width Cranelift's
//!   128-bit CLIF vector ISA cannot legalize (see `mercury_codegen_cranelift::avx2`).
//!
//! Both derive their per-lane semantics from [`VecKernel::eval_lane`] below, so the recipe is the
//! single source of truth; the differential gate polices that the two executors match on the
//! non-multiple-of-8 tails the caller runs scalar-side.
//!
//! A kernel has the fixed ABI `fn(ptrs: *const *mut u8, scalars: *const f32, n: u64)`: `ptrs[k]` is
//! the (already `start`-offset) base of unit-stride stream `k`, `scalars[k]` the k-th loop-invariant
//! f32 (a param, an outer local, or a stride-0 array read the caller pre-loaded), and `n` the
//! multiple-of-8 element count assigned to the vector part. The caller runs the scalar remainder.

use mercury_span::Symbol;

/// A lane-wise binary arithmetic op. Division is included; the assembler emits `vdivps`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecBin {
    Add,
    Sub,
    Mul,
    Div,
}

/// A lane-wise ordered float comparison producing a boolean mask (1.0 / 0.0 in the reference
/// evaluator; all-ones / all-zero lanes under `vcmpps`). Ordered ⇒ false on NaN, matching Rust's
/// `<`,`<=`,`>`,`>=`,`==`,`!=` for every finite/inf case, which is what the interpreter oracle uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecCmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// One straight-line SSA operation. Operands are indices into the body prefix (op `i` may only
/// reference ops `< i`); the result of op `i` is "value `i`". `Store` produces no value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VecOp {
    /// Unit-stride vector load of stream `s` at the current lane group.
    Load { stream: u32 },
    /// Broadcast the k-th loop-invariant scalar.
    Splat { scalar: u32 },
    /// Broadcast an immediate f32 constant (its raw bits).
    Const { bits: u32 },
    /// Lane-wise `a ⊕ b`.
    Bin { op: VecBin, a: u32, b: u32 },
    /// Fused `a * b + c`, single-rounded (`vfmadd` / Rust `mul_add`).
    Fma { a: u32, b: u32, c: u32 },
    /// Lane-wise square root.
    Sqrt { a: u32 },
    /// Lane-wise negation (sign-bit flip; bit-exact for ±0 and NaN).
    Neg { a: u32 },
    /// Lane-wise ordered compare → mask.
    Cmp { pred: VecCmp, a: u32, b: u32 },
    /// Blend: `mask ? a : b` (`vblendvps`; the reference picks `a` when the mask lane is non-zero).
    Select { mask: u32, a: u32, b: u32 },
    /// Unit-stride vector store of value `val` to stream `s`.
    Store { stream: u32, val: u32 },
}

/// A synthesized 256-bit vector kernel: a symbol the caller `Call`s, the stream/scalar counts, the
/// straight-line body, and the desired unroll (the assembler may reduce it to fit 16 YMM registers).
#[derive(Clone, Debug)]
pub struct VecKernel {
    pub name: Symbol,
    pub streams: u32,
    pub scalars: u32,
    pub ops: Vec<VecOp>,
    pub unroll: u32,
}

impl VecKernel {
    /// Evaluate the body for one lane and write its stores. `load(stream)` yields that stream's
    /// current element, `scalar(k)` the k-th invariant, `store(stream, v)` writes an output. This is
    /// the reference semantics the AVX2 assembler reproduces per lane and the interpreter marshals
    /// element-by-element — the two must agree bit-for-bit (all arithmetic is f32; `Fma` is
    /// single-rounded via `mul_add`; `Neg` flips the sign bit; ordered compares are false on NaN).
    pub fn eval_lane(
        &self,
        mut load: impl FnMut(u32) -> f32,
        scalar: impl Fn(u32) -> f32,
        mut store: impl FnMut(u32, f32),
    ) {
        let mut vals: Vec<f32> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let v = match *op {
                VecOp::Load { stream } => load(stream),
                VecOp::Splat { scalar: k } => scalar(k),
                VecOp::Const { bits } => f32::from_bits(bits),
                VecOp::Bin { op, a, b } => {
                    let (x, y) = (vals[a as usize], vals[b as usize]);
                    match op {
                        VecBin::Add => x + y,
                        VecBin::Sub => x - y,
                        VecBin::Mul => x * y,
                        VecBin::Div => x / y,
                    }
                }
                VecOp::Fma { a, b, c } => vals[a as usize].mul_add(vals[b as usize], vals[c as usize]),
                VecOp::Sqrt { a } => vals[a as usize].sqrt(),
                VecOp::Neg { a } => f32::from_bits(vals[a as usize].to_bits() ^ 0x8000_0000),
                VecOp::Cmp { pred, a, b } => {
                    let (x, y) = (vals[a as usize], vals[b as usize]);
                    let t = match pred {
                        VecCmp::Lt => x < y,
                        VecCmp::Le => x <= y,
                        VecCmp::Gt => x > y,
                        VecCmp::Ge => x >= y,
                        VecCmp::Eq => x == y,
                        VecCmp::Ne => x != y,
                    };
                    if t {
                        1.0
                    } else {
                        0.0
                    }
                }
                VecOp::Select { mask, a, b } => {
                    if vals[mask as usize] != 0.0 {
                        vals[a as usize]
                    } else {
                        vals[b as usize]
                    }
                }
                VecOp::Store { stream, val } => {
                    store(stream, vals[val as usize]);
                    f32::NAN // stores carry no value; never referenced
                }
            };
            vals.push(v);
        }
    }
}
