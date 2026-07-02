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
        // Store-to-load forwarding within the lane: the assembler stores each stream to memory and a
        // later `Load` of that stream re-reads the same address, so a body that writes then reads a
        // stream — e.g. a fused `t[k] = …; o[k] = f(t[k])` — observes the just-written value. Mirror
        // that here so the interpreter oracle matches the machine code bit-for-bit. `self.streams`
        // bounds every stream index.
        let mut stored: Vec<Option<f32>> = vec![None; self.streams as usize];
        for op in &self.ops {
            let v = match *op {
                VecOp::Load { stream } => {
                    stored[stream as usize].unwrap_or_else(|| load(stream))
                }
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
                    let x = vals[val as usize];
                    stored[stream as usize] = Some(x); // forward to later loads of this stream
                    store(stream, x);
                    f32::NAN // stores carry no value; never referenced
                }
            };
            vals.push(v);
        }
    }

    /// The register pressure of this body, or `None` if it is outside the AVX2 emitter's coverage
    /// (0 or more than [`VEC_MAX_STREAMS`] streams, or a `Const` the vectorizer should have folded
    /// into `scalars`). Single-sourced here so the Cranelift assembler's `plan_registers` and the
    /// vectorizer's feasibility gate compute the *same* P/H and can never disagree about whether a
    /// body fits — a mismatch would turn a fallback-eligible body into a hard `assemble_kernel`
    /// error. Pure over the op list.
    pub fn pressure(&self) -> Option<VecPressure> {
        if self.streams == 0 || self.streams > VEC_MAX_STREAMS {
            return None;
        }
        if self.ops.iter().any(|o| matches!(o, VecOp::Const { .. })) {
            return None;
        }
        // Distinct invariant scalars referenced by `Splat`, first-seen order (each hoisted to a reg).
        let mut hoist_scalars: Vec<u32> = Vec::new();
        for o in &self.ops {
            if let VecOp::Splat { scalar } = *o {
                if !hoist_scalars.contains(&scalar) {
                    hoist_scalars.push(scalar);
                }
            }
        }
        let needs_signmask = self.ops.iter().any(|o| matches!(o, VecOp::Neg { .. }));

        // last_use[i]: last op index that references value i.
        let mut last_use = vec![0usize; self.ops.len()];
        for (i, o) in self.ops.iter().enumerate() {
            for v in op_operands(o) {
                last_use[v as usize] = i;
            }
        }
        // P = peak simultaneously-live body registers: allocate a result before freeing this op's
        // dead operands (mirrors the emitter's ordering, so P is an exact upper bound).
        let mut live: Vec<usize> = Vec::new();
        let mut per_group: usize = 0;
        for (i, o) in self.ops.iter().enumerate() {
            if op_produces_value(o) {
                live.push(i);
                per_group = per_group.max(live.len());
            }
            for v in op_operands(o) {
                // Hoisted (Splat/Const) values never occupy a body register.
                let hoisted =
                    matches!(self.ops[v as usize], VecOp::Splat { .. } | VecOp::Const { .. });
                if !hoisted && last_use[v as usize] == i {
                    live.retain(|&x| x != v as usize);
                }
            }
        }
        Some(VecPressure {
            hoist_scalars,
            needs_signmask,
            last_use,
            per_group: per_group.max(1) as u32,
        })
    }
}

/// Architectural YMM registers a kernel may use.
pub const VEC_NREG: u32 = 16;
/// The maximum number of distinct unit-stride streams the AVX2 emitter keeps in base-pointer GPRs.
pub const VEC_MAX_STREAMS: u32 = 4;

/// A backend-agnostic register-pressure summary of a [`VecKernel`] body, consumed by the Cranelift
/// assembler (to build a concrete register plan) and by the vectorizer (to check the body fits before
/// committing to the 256-bit path).
pub struct VecPressure {
    /// Distinct invariant scalars referenced by `Splat`, first-seen order (each hoisted to a reg).
    pub hoist_scalars: Vec<u32>,
    /// The body negates, so a sign-mask constant occupies one more fixed register.
    pub needs_signmask: bool,
    /// `last_use[i]` = the last op index that references value `i`.
    pub last_use: Vec<usize>,
    /// P: peak simultaneously-live body registers (results that are neither hoisted nor stores).
    pub per_group: u32,
}

impl VecPressure {
    /// H: fixed registers taken by hoisted broadcasts (invariant scalars + an optional sign-mask).
    pub fn hoist(&self) -> u32 {
        self.hoist_scalars.len() as u32 + self.needs_signmask as u32
    }
    /// Whether one group plus the hoisted broadcasts fit the register file (unroll ≥ 1 feasible).
    pub fn fits(&self) -> bool {
        self.per_group + self.hoist() <= VEC_NREG
    }
}

/// The operand value-indices an op reads (for liveness). `Load`/`Splat`/`Const` read nothing.
pub fn op_operands(op: &VecOp) -> Vec<u32> {
    match *op {
        VecOp::Load { .. } | VecOp::Splat { .. } | VecOp::Const { .. } => vec![],
        VecOp::Bin { a, b, .. } | VecOp::Cmp { a, b, .. } => vec![a, b],
        VecOp::Fma { a, b, c } => vec![a, b, c],
        VecOp::Select { mask, a, b } => vec![mask, a, b],
        VecOp::Sqrt { a } | VecOp::Neg { a } => vec![a],
        VecOp::Store { val, .. } => vec![val],
    }
}

/// Whether an op produces a body value that needs a (non-hoisted) register.
pub fn op_produces_value(op: &VecOp) -> bool {
    !matches!(
        op,
        VecOp::Store { .. } | VecOp::Splat { .. } | VecOp::Const { .. }
    )
}
