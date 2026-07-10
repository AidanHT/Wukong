//! `VecKernel` — a backend-agnostic recipe for a vectorizable straight-line loop body, produced by
//! the general vectorizer (`wukong_mir_build`) and consumed by two backends that must agree
//! bit-for-bit:
//!
//! * the **interpreter** marshals it lane-by-lane through its abstract slot memory (the oracle), and
//! * the **Cranelift** backend assembles it into true 256-bit AVX2 machine code — the width Cranelift's
//!   128-bit CLIF vector ISA cannot legalize (see `wukong_codegen_cranelift::avx2`).
//!
//! Both derive their per-lane semantics from [`VecKernel::eval_lane`] below, so the recipe is the
//! single source of truth; the differential gate polices that the two executors match on the
//! non-multiple-of-8 tails the caller runs scalar-side.
//!
//! A kernel has the fixed ABI `fn(ptrs: *const *mut u8, scalars: *const f32, n: u64)`: `ptrs[k]` is
//! the (already `start`-offset) base of unit-stride stream `k`, `scalars[k]` the k-th loop-invariant
//! f32 (a param, an outer local, or a stride-0 array read the caller pre-loaded), and `n` the
//! multiple-of-8 element count assigned to the vector part. The caller runs the scalar remainder.

use wukong_span::Symbol;

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

/// The fold of a reduction kernel. `Fmax`/`Fmin` use the x86 `vmaxps`/`vminps` operand order
/// (`src1` returned only when strictly greater/less), so the reference below and the machine code
/// agree on ties and NaN.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecRedOp {
    Add,
    Fmax,
    Fmin,
}

impl VecRedOp {
    /// The fold identity (initial accumulator): 0 for a sum, ∓∞ for max/min so empty lanes lose.
    pub fn identity(self) -> f32 {
        match self {
            VecRedOp::Add => 0.0,
            VecRedOp::Fmax => f32::NEG_INFINITY,
            VecRedOp::Fmin => f32::INFINITY,
        }
    }
    /// `fold(acc, x)` with the assembler's operand order (`acc` = src1, `x` = src2). `vmaxps`/`vminps`
    /// return src2 unless src1 is strictly greater/less, so `if acc > x { acc } else { x }` matches
    /// them exactly on equal values (incl. ±0) and NaN.
    pub fn fold(self, acc: f32, x: f32) -> f32 {
        match self {
            VecRedOp::Add => acc + x,
            VecRedOp::Fmax => {
                if acc > x {
                    acc
                } else {
                    x
                }
            }
            VecRedOp::Fmin => {
                if acc < x {
                    acc
                } else {
                    x
                }
            }
        }
    }
}

/// The reduction a reduction-kernel performs: fold every lane's value of op `value` (the addend) into
/// an accumulator and return the horizontal fold. Present ⇒ the kernel has no `Store` and returns an
/// f32 (in xmm0) instead of writing memory.
///
/// `fma` fuses an `Add` reduction of a product: when the addend is `X*Y`, `fma = Some((X, Y))` and the
/// per-lane fold is `acc = fma(X, Y, acc)` in *one* rounding (`vfmadd231ps` / `f32::mul_add`) — the
/// product op never enters the recipe, so the fold matches the 128-bit path's fused accumulate, costs
/// one fewer op, and frees a register for more accumulators. `None` ⇒ the plain fold of `value`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VecReduce {
    pub op: VecRedOp,
    pub value: u32,
    pub fma: Option<(u32, u32)>,
}

/// A synthesized 256-bit vector kernel: a symbol the caller `Call`s, the stream/scalar counts, the
/// straight-line body, and the desired unroll (the assembler may reduce it to fit 16 YMM registers).
/// `reduce` distinguishes an *elementwise* kernel (`None` — stores its results) from a *reduction*
/// kernel (`Some` — folds one value across all lanes and returns the horizontal result).
#[derive(Clone, Debug)]
pub struct VecKernel {
    pub name: Symbol,
    pub streams: u32,
    pub scalars: u32,
    pub ops: Vec<VecOp>,
    pub unroll: u32,
    pub reduce: Option<VecReduce>,
}

impl VecKernel {
    /// Evaluate the body for one lane and write its stores. `load(stream)` yields that stream's
    /// current element, `scalar(k)` the k-th invariant, `store(stream, v)` writes an output. This is
    /// the reference semantics the AVX2 assembler reproduces per lane and the interpreter marshals
    /// element-by-element — the two must agree bit-for-bit (all arithmetic is f32; `Fma` is
    /// single-rounded via `mul_add`; `Neg` flips the sign bit; ordered compares are false on NaN).
    pub fn eval_lane(
        &self,
        load: impl FnMut(u32) -> f32,
        scalar: impl Fn(u32) -> f32,
        store: impl FnMut(u32, f32),
    ) {
        self.eval_all(load, scalar, store);
    }

    /// Evaluate every op for one lane, returning all their values (the reference used by both
    /// [`Self::eval_lane`] for elementwise kernels and [`Self::eval_reduction`] for reductions).
    fn eval_all(
        &self,
        mut load: impl FnMut(u32) -> f32,
        scalar: impl Fn(u32) -> f32,
        mut store: impl FnMut(u32, f32),
    ) -> Vec<f32> {
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
        vals
    }

    /// The unroll a reduction kernel uses: as many accumulator copies as the requested unroll, capped
    /// so the accumulators plus one addend's live registers plus the hoisted broadcasts fit the 16
    /// YMM registers. Single-sourced (via [`Self::pressure`]) so the assembler and the interpreter's
    /// [`Self::eval_reduction`] pick the *same* number of accumulators — the reassociation grouping,
    /// hence the exact float result, depends on it. `1` if the body doesn't fit at all (the caller
    /// gates on [`Self::reduction_fits`] first).
    pub fn reduction_unroll(&self) -> u32 {
        let Some(p) = self.pressure() else { return 1 };
        // Each unrolled copy needs its own `P` addend registers *and* its own accumulator, so the
        // register cost per copy is `P + 1`; the hoisted broadcasts are shared.
        let denom = p.per_group + 1;
        let avail = VEC_NREG.saturating_sub(p.hoist());
        self.unroll.max(1).min((avail / denom).max(1))
    }

    /// Whether this reduction body fits: one copy's addend registers + one accumulator + the hoisted
    /// broadcasts fit the register file (`P + 1 + H ≤ 16`). The vectorizer checks this before
    /// committing to the 256-bit reduction path.
    pub fn reduction_fits(&self) -> bool {
        matches!(self.pressure(), Some(p) if p.per_group + 1 + p.hoist() <= VEC_NREG)
    }

    /// Evaluate a reduction kernel over `n` elements (a multiple of 8) exactly as the AVX2 assembler
    /// does, returning the horizontal fold. `load(stream, element)` reads a stream's element; the
    /// caller folds the `[n, end)` scalar tail into this result afterward. The reassociation is fixed
    /// and identical on both backends: `U` lane-accumulators filled by the main U-unrolled loop and a
    /// single-group cleanup, combined lane-wise into accumulator 0, then a *sequential* lane-0..7
    /// horizontal fold (the assembler spills the accumulator and folds the eight lanes in order, so no
    /// tree-reduction order to match). The reassociated form is the oracle (both backends run it).
    pub fn eval_reduction(
        &self,
        n: usize,
        mut load: impl FnMut(u32, usize) -> f32,
        scalar: impl Fn(u32) -> f32,
    ) -> f32 {
        let red = self.reduce.expect("eval_reduction on a non-reduction kernel");
        let lanes = LANES_USIZE;
        let u = self.reduction_unroll() as usize;
        // Fold one lane's addend into its accumulator: a fused `acc = fma(X, Y, acc)` in a single
        // rounding (bit-identical to the assembler's `vfmadd231ps`) when `fma` is set, else the plain
        // fold of the addend value.
        let addend = |acc: f32, vals: &[f32]| -> f32 {
            match red.fma {
                Some((x, y)) => vals[x as usize].mul_add(vals[y as usize], acc),
                None => red.op.fold(acc, vals[red.value as usize]),
            }
        };
        // acc[copy * lanes + lane]. The addend of element `e` is op `red.value` evaluated with the
        // load closure reading element `e`.
        let mut acc = vec![red.op.identity(); u * lanes];
        let mut i = 0usize;
        // Main loop: U independent accumulators, computed and folded one copy at a time.
        let ustep = u * lanes;
        while i + ustep <= n {
            for cu in 0..u {
                for lane in 0..lanes {
                    let e = i + cu * lanes + lane;
                    let vals = self.eval_all(|s| load(s, e), &scalar, |_, _| {});
                    let idx = cu * lanes + lane;
                    acc[idx] = addend(acc[idx], &vals);
                }
            }
            i += ustep;
        }
        // Cleanup: remaining full 8-groups fold into accumulator 0.
        while i + lanes <= n {
            for lane in 0..lanes {
                let e = i + lane;
                let vals = self.eval_all(|s| load(s, e), &scalar, |_, _| {});
                acc[lane] = addend(acc[lane], &vals);
            }
            i += lanes;
        }
        // Combine the U accumulators into copy 0, lane-wise.
        for cu in 1..u {
            for lane in 0..lanes {
                acc[lane] = red.op.fold(acc[lane], acc[cu * lanes + lane]);
            }
        }
        // Horizontal fold of the eight lanes, in order 0..8.
        let mut r = acc[0];
        for lane in 1..lanes {
            r = red.op.fold(r, acc[lane]);
        }
        r
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
        // A reduction's fold reads its addend operand(s) *after* the last op (the fold is not itself an
        // op), so those registers must survive the whole body. Mark their last use one past the end so
        // neither the peak-liveness scan below nor the emitter's `free_after` reclaims them early —
        // essential for the fused `fma` operands, which have no consuming op at all.
        if let Some(red) = self.reduce {
            let end = self.ops.len();
            match red.fma {
                Some((x, y)) => {
                    last_use[x as usize] = end;
                    last_use[y as usize] = end;
                }
                None => last_use[red.value as usize] = end,
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
/// f32 lanes per 256-bit YMM register.
pub const VEC_LANES: u32 = 8;
const LANES_USIZE: usize = VEC_LANES as usize;
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
