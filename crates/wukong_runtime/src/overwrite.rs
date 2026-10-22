//! Which kernel entry points **overwrite an output buffer completely** — the contract the native
//! backend's stack coloring (`wukong_opt::slot_lifetime`) needs before it may end a scratch array's
//! lifetime at a kernel call.
//!
//! A program that declares `let mut q: [f32; 512] = [0.0; 512];` at the top of a function and fills
//! `q` with a recognized GEMM much later keeps `q`'s zeros live from the top to the GEMM, unless the
//! compiler knows the GEMM never reads `q` before writing every element of it. Then the zeros are
//! dead the moment the call starts, and `q`'s frame range can hold another array until then. Which
//! calls qualify is a property of the kernel's implementation, so it is stated here, next to the
//! kernels, as data: an exact symbol, the operand it writes, how many elements of it, and the
//! operand values under which that is true. A name pattern ("anything called `*_f32` with an `out`
//! operand") is never a contract.
//!
//! # What an entry promises
//!
//! For a call with exactly [`Overwrite::arity`] operands, every operand listed in
//! [`Overwrite::extent`] or [`Overwrite::positive`] at least 1, and every operand of
//! [`Overwrite::fixed`] one of its listed values, the kernel **writes every byte of**
//! `[out, out + elem_bytes · Π extent)` and its result there does not depend on what those bytes held
//! before the call — no element is read before the kernel itself has written it. That holds only
//! when the output overlaps no input operand, which the consumer establishes separately (it checks
//! that the output's storage reaches the call through the output operand alone).
//!
//! The conditions are the kernels' own early exits: a GEMM returns without writing anything when
//! `K <= 0` (an empty contraction is the zero matrix, which the lowering fills itself), so `K` is
//! `positive` although it is no extent of `C`; `beta != 0` accumulates into `C`, so `beta` is
//! `fixed` at 0; a norm with an op code it does not know writes nothing, so `op` is `fixed` to the
//! codes it implements.
//!
//! # How the table is held to the code
//!
//! `every_contract_holds_on_the_kernel` runs each entry on several shapes and every allowed `fixed`
//! value, once per output pre-fill — a quiet NaN, `-3.0e30`, `+3.0e30`, and a pattern that varies
//! per element over both signs and tiny and huge magnitudes — and requires every run to give the
//! same bits over the whole stated extent. An element the kernel leaves alone keeps different
//! fills. An element computed from its old value differs unless every fill is an identity of that
//! computation, and the fills are chosen so that none is an identity of a fold a kernel would
//! plausibly seed from its output: the NaN breaks a sum or a product, `-3.0e30` a running min,
//! `+3.0e30` a running max or max of magnitudes (for which the NaN and `-3.0e30` are both
//! identities under the x86 max rule), and the per-element pattern a min of magnitudes
//! (`the_fills_catch_a_fold_seeded_from_the_output` holds each of those claims). It is a test, not
//! a proof: a read of the old value through a computation for which every fill happens to be an
//! identity still passes, so a new entry's kernel is read, not only run. Every integer operand an
//! entry leaves unconstrained (an activation or op code, an `eps`, a `causal` flag) is also probed
//! at 0 and -1, so an early exit on a value the entry admits fails the test as well. A `_parallel`
//! entry with a size gate of its own that `cfg(test)` leaves in force — the GEMMs' 2^23 MACs, the
//! `vmath`, `velem` and transpose element counts — also runs one shape past it
//! (`every_gated_parallel_entry_runs_past_its_gate`), so the test holds its parallel path and not
//! only the serial kernel it falls back to below the gate.

/// One kernel entry point's output contract (see the module note).
#[derive(Clone, Copy, Debug)]
pub struct Overwrite {
    /// The exported symbol.
    pub symbol: &'static str,
    /// The C-ABI operand count the contract is stated for.
    pub arity: usize,
    /// The operand the kernel writes.
    pub out: usize,
    /// Bytes per output element.
    pub elem_bytes: u32,
    /// Integer operands whose product is the output's element count.
    pub extent: &'static [usize],
    /// Further integer operands that must be at least 1 for the kernel to write anything.
    pub positive: &'static [usize],
    /// Integer operands that must be one of the listed values.
    pub fixed: &'static [(usize, &'static [i64])],
}

/// `beta == 0`: the GEMM overwrites `C` instead of accumulating into it.
const BETA_ZERO: &[(usize, &[i64])] = &[(6, &[0])];
/// The row-norm codes `norm_row` implements (`NORM_SOFTMAX` ..= `NORM_L2NORM`); any other writes
/// nothing.
const NORM_OPS: &[(usize, &[i64])] = &[(5, &[0, 1, 2, 3, 4])];
/// The affine-norm codes `norm_affine_row` implements (`NORM_LAYERNORM`, `NORM_RMSNORM`).
const NORM_AFFINE_OPS: &[(usize, &[i64])] = &[(7, &[1, 2])];

/// `C[m, n]` of a `(a, b, c, m, k, n, beta, ..)` GEMM: written when `m, n, k >= 1` and `beta == 0`.
const fn gemm(symbol: &'static str, arity: usize) -> Overwrite {
    Overwrite {
        symbol,
        arity,
        out: 2,
        elem_bytes: 4,
        extent: &[3, 5],
        positive: &[4],
        fixed: BETA_ZERO,
    }
}

/// `y[m]` of a `(a, x, y, m, n, ..)` GEMV: written when `m, n >= 1`.
const fn gemv(symbol: &'static str, arity: usize) -> Overwrite {
    Overwrite {
        symbol,
        arity,
        out: 2,
        elem_bytes: 4,
        extent: &[3],
        positive: &[4],
        fixed: &[],
    }
}

/// A `(x, out, rows, cols)` kernel writing `out[rows * cols]`: transpose and the row scans.
const fn rows_cols(symbol: &'static str) -> Overwrite {
    Overwrite {
        symbol,
        arity: 4,
        out: 1,
        elem_bytes: 4,
        extent: &[2, 3],
        positive: &[],
        fixed: &[],
    }
}

/// A `(x, out, rows, cols)` column reduction writing `out[cols]` when `rows >= 1`.
const fn column(symbol: &'static str) -> Overwrite {
    Overwrite {
        symbol,
        arity: 4,
        out: 1,
        elem_bytes: 4,
        extent: &[3],
        positive: &[2],
        fixed: &[],
    }
}

/// Every kernel whose output overwrite the compiler may rely on. A symbol absent from this table is
/// assumed to read (or partly keep) its output.
pub const OVERWRITES: &[Overwrite] = &[
    gemm("wukong_sgemm", 7),
    gemm("wukong_sgemm_parallel", 7),
    gemm("wukong_sgemm_nt", 7),
    gemm("wukong_sgemm_nt_parallel", 7),
    gemm("wukong_sgemm_tn", 7),
    gemm("wukong_sgemm_tn_parallel", 7),
    gemm("wukong_sgemm_nt_alpha", 8),
    gemm("wukong_sgemm_nt_alpha_parallel", 8),
    gemm("wukong_sgemm_nt_epi", 9),
    gemm("wukong_sgemm_nt_epi_parallel", 9),
    gemv("wukong_sgemv", 5),
    gemv("wukong_sgemv_parallel", 5),
    gemv("wukong_sgemv_alpha", 6),
    gemv("wukong_sgemv_alpha_parallel", 6),
    // (x, out, n, op): every op code writes `out` — an unknown one is the identity.
    Overwrite {
        symbol: "wukong_vmath_f32",
        arity: 4,
        out: 1,
        elem_bytes: 4,
        extent: &[2],
        positive: &[],
        fixed: &[],
    },
    Overwrite {
        symbol: "wukong_vmath_f32_parallel",
        arity: 4,
        out: 1,
        elem_bytes: 4,
        extent: &[2],
        positive: &[],
        fixed: &[],
    },
    // (x, y, out, n, op): an unknown op is the identity in `x`, never a no-write.
    Overwrite {
        symbol: "wukong_vmath2_f32",
        arity: 5,
        out: 2,
        elem_bytes: 4,
        extent: &[3],
        positive: &[],
        fixed: &[],
    },
    // (x, y, out, n, a, b, c, op)
    Overwrite {
        symbol: "wukong_velem_f32",
        arity: 8,
        out: 2,
        elem_bytes: 4,
        extent: &[3],
        positive: &[],
        fixed: &[],
    },
    Overwrite {
        symbol: "wukong_velem_f32_parallel",
        arity: 8,
        out: 2,
        elem_bytes: 4,
        extent: &[3],
        positive: &[],
        fixed: &[],
    },
    // (x, out, rows, cols, eps_bits, op)
    Overwrite {
        symbol: "wukong_norm_f32",
        arity: 6,
        out: 1,
        elem_bytes: 4,
        extent: &[2, 3],
        positive: &[],
        fixed: NORM_OPS,
    },
    Overwrite {
        symbol: "wukong_norm_f32_parallel",
        arity: 6,
        out: 1,
        elem_bytes: 4,
        extent: &[2, 3],
        positive: &[],
        fixed: NORM_OPS,
    },
    // (x, out, gamma, beta, rows, cols, eps_bits, op)
    Overwrite {
        symbol: "wukong_norm_affine_f32",
        arity: 8,
        out: 1,
        elem_bytes: 4,
        extent: &[4, 5],
        positive: &[],
        fixed: NORM_AFFINE_OPS,
    },
    Overwrite {
        symbol: "wukong_norm_affine_f32_parallel",
        arity: 8,
        out: 1,
        elem_bytes: 4,
        extent: &[4, 5],
        positive: &[],
        fixed: NORM_AFFINE_OPS,
    },
    // (x, b, out, rows, cols, op)
    Overwrite {
        symbol: "wukong_bias_bcast_f32",
        arity: 6,
        out: 2,
        elem_bytes: 4,
        extent: &[3, 4],
        positive: &[],
        fixed: &[],
    },
    Overwrite {
        symbol: "wukong_bias_bcast_f32_parallel",
        arity: 6,
        out: 2,
        elem_bytes: 4,
        extent: &[3, 4],
        positive: &[],
        fixed: &[],
    },
    rows_cols("wukong_transpose_f32"),
    rows_cols("wukong_transpose_f32_parallel"),
    rows_cols("wukong_cumsum_f32"),
    rows_cols("wukong_cumsum_f32_parallel"),
    column("wukong_colsum_f32"),
    column("wukong_colsum_f32_parallel"),
    column("wukong_colmean_f32"),
    column("wukong_colmean_f32_parallel"),
    column("wukong_colsumsq_f32"),
    column("wukong_colsumsq_f32_parallel"),
    column("wukong_colmax_f32"),
    column("wukong_colmax_f32_parallel"),
    column("wukong_colmin_f32"),
    column("wukong_colmin_f32_parallel"),
    column("wukong_colmaxabs_f32"),
    column("wukong_colmaxabs_f32_parallel"),
    // (q, k, v, out, s, d, scale, causal): "`O` is fully overwritten".
    Overwrite {
        symbol: "wukong_attention_f32",
        arity: 8,
        out: 3,
        elem_bytes: 4,
        extent: &[4, 5],
        positive: &[],
        fixed: &[],
    },
];

/// The contract of `symbol`, if the table states one.
pub fn overwrite_of(symbol: &str) -> Option<&'static Overwrite> {
    OVERWRITES.iter().find(|o| o.symbol == symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    /// One operand of a test call.
    #[derive(Clone, Copy, Debug)]
    enum Arg {
        /// An input buffer of this many `f32`.
        In(usize),
        /// The output buffer.
        Out,
        Int(i64),
        Float(f32),
    }

    /// An operand bound to the call: a pointer, an integer or a float.
    #[derive(Clone, Copy)]
    enum Val {
        Ptr(*mut f32),
        Int(i64),
        Float(f32),
    }

    impl Val {
        fn p(self) -> *mut f32 {
            match self {
                Val::Ptr(p) => p,
                _ => panic!("operand is not a buffer"),
            }
        }
        fn i(self) -> i64 {
            match self {
                Val::Int(v) => v,
                _ => panic!("operand is not an integer"),
            }
        }
        fn f(self) -> f32 {
            match self {
                Val::Float(v) => v,
                _ => panic!("operand is not a float"),
            }
        }
    }

    /// Call `symbol` with `a`: one typed call per entry of [`OVERWRITES`].
    ///
    /// # Safety
    /// Every buffer in `a` is valid for the extent the kernel reads or writes through it.
    unsafe fn invoke(symbol: &str, a: &[Val]) {
        let p = |k: usize| a[k].p();
        let i = |k: usize| a[k].i();
        let f = |k: usize| a[k].f();
        // SAFETY: the caller's contract.
        unsafe {
            match symbol {
                "wukong_sgemm" => wukong_sgemm(p(0), p(1), p(2), i(3), i(4), i(5), i(6)),
                "wukong_sgemm_parallel" => {
                    wukong_sgemm_parallel(p(0), p(1), p(2), i(3), i(4), i(5), i(6))
                }
                "wukong_sgemm_nt" => wukong_sgemm_nt(p(0), p(1), p(2), i(3), i(4), i(5), i(6)),
                "wukong_sgemm_nt_parallel" => {
                    wukong_sgemm_nt_parallel(p(0), p(1), p(2), i(3), i(4), i(5), i(6))
                }
                "wukong_sgemm_tn" => wukong_sgemm_tn(p(0), p(1), p(2), i(3), i(4), i(5), i(6)),
                "wukong_sgemm_tn_parallel" => {
                    wukong_sgemm_tn_parallel(p(0), p(1), p(2), i(3), i(4), i(5), i(6))
                }
                "wukong_sgemm_nt_alpha" => {
                    wukong_sgemm_nt_alpha(p(0), p(1), p(2), i(3), i(4), i(5), i(6), f(7))
                }
                "wukong_sgemm_nt_alpha_parallel" => {
                    wukong_sgemm_nt_alpha_parallel(p(0), p(1), p(2), i(3), i(4), i(5), i(6), f(7))
                }
                "wukong_sgemm_nt_epi" => {
                    wukong_sgemm_nt_epi(p(0), p(1), p(2), i(3), i(4), i(5), i(6), p(7), i(8))
                }
                "wukong_sgemm_nt_epi_parallel" => wukong_sgemm_nt_epi_parallel(
                    p(0),
                    p(1),
                    p(2),
                    i(3),
                    i(4),
                    i(5),
                    i(6),
                    p(7),
                    i(8),
                ),
                "wukong_sgemv" => wukong_sgemv(p(0), p(1), p(2), i(3), i(4)),
                "wukong_sgemv_parallel" => wukong_sgemv_parallel(p(0), p(1), p(2), i(3), i(4)),
                "wukong_sgemv_alpha" => wukong_sgemv_alpha(p(0), p(1), p(2), i(3), i(4), f(5)),
                "wukong_sgemv_alpha_parallel" => {
                    wukong_sgemv_alpha_parallel(p(0), p(1), p(2), i(3), i(4), f(5))
                }
                "wukong_vmath_f32" => wukong_vmath_f32(p(0), p(1), i(2), i(3)),
                "wukong_vmath_f32_parallel" => wukong_vmath_f32_parallel(p(0), p(1), i(2), i(3)),
                "wukong_vmath2_f32" => wukong_vmath2_f32(p(0), p(1), p(2), i(3), i(4)),
                "wukong_velem_f32" => {
                    wukong_velem_f32(p(0), p(1), p(2), i(3), f(4), f(5), f(6), i(7))
                }
                "wukong_velem_f32_parallel" => {
                    wukong_velem_f32_parallel(p(0), p(1), p(2), i(3), f(4), f(5), f(6), i(7))
                }
                "wukong_norm_f32" => wukong_norm_f32(p(0), p(1), i(2), i(3), i(4), i(5)),
                "wukong_norm_f32_parallel" => {
                    wukong_norm_f32_parallel(p(0), p(1), i(2), i(3), i(4), i(5))
                }
                "wukong_norm_affine_f32" => {
                    wukong_norm_affine_f32(p(0), p(1), p(2), p(3), i(4), i(5), i(6), i(7))
                }
                "wukong_norm_affine_f32_parallel" => {
                    wukong_norm_affine_f32_parallel(p(0), p(1), p(2), p(3), i(4), i(5), i(6), i(7))
                }
                "wukong_bias_bcast_f32" => {
                    wukong_bias_bcast_f32(p(0), p(1), p(2), i(3), i(4), i(5))
                }
                "wukong_bias_bcast_f32_parallel" => {
                    wukong_bias_bcast_f32_parallel(p(0), p(1), p(2), i(3), i(4), i(5))
                }
                "wukong_transpose_f32" => wukong_transpose_f32(p(0), p(1), i(2), i(3)),
                "wukong_transpose_f32_parallel" => {
                    wukong_transpose_f32_parallel(p(0), p(1), i(2), i(3))
                }
                "wukong_cumsum_f32" => wukong_cumsum_f32(p(0), p(1), i(2), i(3)),
                "wukong_cumsum_f32_parallel" => wukong_cumsum_f32_parallel(p(0), p(1), i(2), i(3)),
                "wukong_colsum_f32" => wukong_colsum_f32(p(0), p(1), i(2), i(3)),
                "wukong_colsum_f32_parallel" => wukong_colsum_f32_parallel(p(0), p(1), i(2), i(3)),
                "wukong_colmean_f32" => wukong_colmean_f32(p(0), p(1), i(2), i(3)),
                "wukong_colmean_f32_parallel" => {
                    wukong_colmean_f32_parallel(p(0), p(1), i(2), i(3))
                }
                "wukong_colsumsq_f32" => wukong_colsumsq_f32(p(0), p(1), i(2), i(3)),
                "wukong_colsumsq_f32_parallel" => {
                    wukong_colsumsq_f32_parallel(p(0), p(1), i(2), i(3))
                }
                "wukong_colmax_f32" => wukong_colmax_f32(p(0), p(1), i(2), i(3)),
                "wukong_colmax_f32_parallel" => wukong_colmax_f32_parallel(p(0), p(1), i(2), i(3)),
                "wukong_colmin_f32" => wukong_colmin_f32(p(0), p(1), i(2), i(3)),
                "wukong_colmin_f32_parallel" => wukong_colmin_f32_parallel(p(0), p(1), i(2), i(3)),
                "wukong_colmaxabs_f32" => wukong_colmaxabs_f32(p(0), p(1), i(2), i(3)),
                "wukong_colmaxabs_f32_parallel" => {
                    wukong_colmaxabs_f32_parallel(p(0), p(1), i(2), i(3))
                }
                "wukong_attention_f32" => {
                    wukong_attention_f32(p(0), p(1), p(2), p(3), i(4), i(5), f(6), i(7))
                }
                other => panic!("no test call for `{other}`: add one to `invoke` and `calls`"),
            }
        }
    }

    /// `(m, k, n)` of a GEMM past the parallel entries' 2^23-MAC gate (8.73M MACs), with three K
    /// blocks — the first overwrites `C`, the later ones accumulate into what it wrote — and MR and
    /// NR remainders.
    const GEMM_PAST_GATE: (i64, i64, i64) = (97, 900, 100);

    /// Every operand list the test runs `symbol` with: each shape crossed with each value its
    /// op-code operands take. The shapes include 1-element, odd and multi-tile sizes, and, for a
    /// `_parallel` entry with a size gate of its own ([`past_size_gate`]), one shape past it: the
    /// small shapes reach only the serial kernel such an entry falls back to.
    fn calls(symbol: &str) -> Vec<Vec<Arg>> {
        use Arg::*;
        let par = symbol.ends_with("_parallel");
        let shapes3: [(i64, i64, i64); 5] =
            [(1, 1, 1), (3, 5, 7), (7, 13, 2), (17, 33, 9), (64, 70, 48)];
        let shapes3 = shapes3.into_iter().chain(par.then_some(GEMM_PAST_GATE));
        let shapes2: [(i64, i64); 6] = [(1, 1), (1, 9), (5, 3), (9, 17), (33, 40), (70, 130)];
        let lens: [i64; 6] = [1, 7, 8, 13, 64, 1000];
        let u = |v: i64| v as usize;
        let mut out = Vec::new();
        match symbol {
            "wukong_sgemm"
            | "wukong_sgemm_parallel"
            | "wukong_sgemm_nt"
            | "wukong_sgemm_nt_parallel"
            | "wukong_sgemm_tn"
            | "wukong_sgemm_tn_parallel" => {
                for (m, k, n) in shapes3.clone() {
                    out.push(vec![
                        In(u(m * k)),
                        In(u(k * n)),
                        Out,
                        Int(m),
                        Int(k),
                        Int(n),
                        Int(0),
                    ]);
                }
            }
            "wukong_sgemm_nt_alpha" | "wukong_sgemm_nt_alpha_parallel" => {
                for (m, k, n) in shapes3.clone() {
                    out.push(vec![
                        In(u(m * k)),
                        In(u(k * n)),
                        Out,
                        Int(m),
                        Int(k),
                        Int(n),
                        Int(0),
                        Float(0.375),
                    ]);
                }
            }
            "wukong_sgemm_nt_epi" | "wukong_sgemm_nt_epi_parallel" => {
                for (m, k, n) in shapes3.clone() {
                    for act in [0, 1, 2, 3, 7] {
                        out.push(vec![
                            In(u(m * k)),
                            In(u(k * n)),
                            Out,
                            Int(m),
                            Int(k),
                            Int(n),
                            Int(0),
                            In(u(n)),
                            Int(act),
                        ]);
                    }
                }
            }
            "wukong_sgemv" | "wukong_sgemv_parallel" => {
                for (m, n) in shapes2 {
                    out.push(vec![In(u(m * n)), In(u(n)), Out, Int(m), Int(n)]);
                }
            }
            "wukong_sgemv_alpha" | "wukong_sgemv_alpha_parallel" => {
                for (m, n) in shapes2 {
                    out.push(vec![
                        In(u(m * n)),
                        In(u(n)),
                        Out,
                        Int(m),
                        Int(n),
                        Float(-1.5),
                    ]);
                }
            }
            "wukong_vmath_f32" | "wukong_vmath_f32_parallel" => {
                // 40,000: two whole chunks of the parallel entry and a ragged third.
                for n in lens.into_iter().chain(par.then_some(40_000)) {
                    for op in (0..=40).chain([-1, 1000]) {
                        out.push(vec![In(u(n)), Out, Int(n), Int(op)]);
                    }
                }
            }
            "wukong_vmath2_f32" => {
                for n in lens {
                    for op in (0..=15).chain([-1, 1000]) {
                        out.push(vec![In(u(n)), In(u(n)), Out, Int(n), Int(op)]);
                    }
                }
            }
            "wukong_velem_f32" | "wukong_velem_f32_parallel" => {
                // 270,007: 32 whole chunks of the parallel entry and a ragged 33rd.
                for n in lens.into_iter().chain(par.then_some(270_007)) {
                    for op in [0, 1, 2, 3, 256, 257, 258, 512, 513, 514, 1024, 1025, 1026] {
                        for (a, b, c) in [(1.0, 0.0, 0.0), (0.5, -2.0, 0.25)] {
                            out.push(vec![
                                In(u(n)),
                                In(u(n)),
                                Out,
                                Int(n),
                                Float(a),
                                Float(b),
                                Float(c),
                                Int(op),
                            ]);
                        }
                    }
                }
            }
            "wukong_norm_f32" | "wukong_norm_f32_parallel" => {
                let eps = i64::from(1.0e-5f32.to_bits());
                for (r, c) in shapes2 {
                    for op in [0, 1, 2, 3, 4] {
                        out.push(vec![In(u(r * c)), Out, Int(r), Int(c), Int(eps), Int(op)]);
                    }
                }
            }
            "wukong_norm_affine_f32" | "wukong_norm_affine_f32_parallel" => {
                let eps = i64::from(1.0e-5f32.to_bits());
                for (r, c) in shapes2 {
                    for op in [1, 2] {
                        out.push(vec![
                            In(u(r * c)),
                            Out,
                            In(u(c)),
                            In(u(c)),
                            Int(r),
                            Int(c),
                            Int(eps),
                            Int(op),
                        ]);
                    }
                }
            }
            "wukong_bias_bcast_f32" | "wukong_bias_bcast_f32_parallel" => {
                for (r, c) in shapes2 {
                    for op in (-1..=40).chain([1000]) {
                        out.push(vec![In(u(r * c)), In(u(c)), Out, Int(r), Int(c), Int(op)]);
                    }
                }
            }
            "wukong_transpose_f32" | "wukong_transpose_f32_parallel" => {
                // 300x400: ragged 32-row blocks on both axes.
                for (r, c) in shapes2.into_iter().chain(par.then_some((300, 400))) {
                    out.push(vec![In(u(r * c)), Out, Int(r), Int(c)]);
                }
            }
            "wukong_cumsum_f32"
            | "wukong_cumsum_f32_parallel"
            | "wukong_colsum_f32"
            | "wukong_colsum_f32_parallel"
            | "wukong_colmean_f32"
            | "wukong_colmean_f32_parallel"
            | "wukong_colsumsq_f32"
            | "wukong_colsumsq_f32_parallel"
            | "wukong_colmax_f32"
            | "wukong_colmax_f32_parallel"
            | "wukong_colmin_f32"
            | "wukong_colmin_f32_parallel"
            | "wukong_colmaxabs_f32"
            | "wukong_colmaxabs_f32_parallel" => {
                for (r, c) in shapes2 {
                    out.push(vec![In(u(r * c)), Out, Int(r), Int(c)]);
                }
            }
            "wukong_attention_f32" => {
                for (s, d) in shapes2 {
                    for causal in [0, 1] {
                        out.push(vec![
                            In(u(s * d)),
                            In(u(s * d)),
                            In(u(s * d)),
                            Out,
                            Int(s),
                            Int(d),
                            Float(0.25),
                            Int(causal),
                        ]);
                    }
                }
            }
            other => panic!("no test operands for `{other}`: add them to `calls`"),
        }
        out
    }

    /// Deterministic, finite, mixed-sign inputs, seeded per operand so no two buffers are equal.
    fn input(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let h = (i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 2003;
                (h as f32 - 1001.0) / 997.0
            })
            .collect()
    }

    /// Room for the output of a test call none of whose inputs is longer, whatever extent the table
    /// claims: an entry that misstates the output must make the test fail, not write past the
    /// buffer. A call with a longer input gets that input's length ([`room_for`]): every test call's
    /// true output is no longer than one of the two.
    const OUT_ROOM: usize = 1 << 16;

    /// The most room any test call gets: the pre-fills are built this long.
    const MAX_ROOM: usize = 1 << 19;

    /// The output room for a test call with operands `args` (see [`OUT_ROOM`]).
    fn room_for(args: &[Arg]) -> usize {
        args.iter().fold(OUT_ROOM, |room, a| match a {
            Arg::In(n) => room.max(*n),
            _ => room,
        })
    }

    /// Does a call of the `_parallel` entry `symbol` with operands `args` pass the entry's own size
    /// gate, one that `cfg(test)` leaves in force? `None` for an entry without one: the others fork
    /// through `par::fork_units`/`fork_rows`, whose work threshold `cfg(test)` lowers to zero, at
    /// any test shape with two or more rows, row pairs or column stripes (the broadcast bias at
    /// more than one 8-row band).
    fn past_size_gate(symbol: &str, args: &[Arg]) -> Option<bool> {
        let int = |k: usize| match args[k] {
            Arg::Int(v) => u64::try_from(v).unwrap_or(0),
            other => panic!("{symbol}: operand {k} is {other:?}, not an integer"),
        };
        let at_least = |work: u64, gate: usize| Some(work >= gate as u64);
        match symbol {
            "wukong_sgemm_parallel"
            | "wukong_sgemm_nt_parallel"
            | "wukong_sgemm_tn_parallel"
            | "wukong_sgemm_nt_alpha_parallel"
            | "wukong_sgemm_nt_epi_parallel" => {
                Some(int(3) * int(4) * int(5) >= crate::gemm::par_min_macs())
            }
            "wukong_vmath_f32_parallel" => at_least(int(2), crate::vmath::VMATH_PAR_MIN),
            "wukong_velem_f32_parallel" => at_least(int(3), crate::velem::velem_par_min()),
            "wukong_transpose_f32_parallel" => {
                at_least(int(2) * int(3), crate::transpose::TRANSPOSE_PAR_MIN)
            }
            _ => None,
        }
    }

    /// Element `i` of output pre-fill number `fill`. A kernel that writes an element before it reads
    /// it leaves the same bits there under every fill, so the test compares them all. No finite set
    /// of fills sees every read of the old value — a fold whose identity each fill happens to be
    /// passes them all — so each fill here breaks a fold the others leave alone. The test inputs lie
    /// in about [-1, 1], and the runtime's max and min follow x86 `maxps`/`minps`, which return the
    /// second operand when the first is a NaN:
    ///
    /// 0. a quiet NaN poisons a sum or a product, but `max(old, x)` and `min(old, x)` return `x`;
    /// 1. `-3.0e30` survives a sum and a running min, but is the identity of a running max;
    /// 2. `+3.0e30` survives a running max and a max of magnitudes `max(old, |x|)`;
    /// 3. a pattern that varies per element — both signs, tiny and huge magnitudes, `-0.0` and a
    ///    second NaN payload, with period 7 so neighbouring lanes of a vector differ — survives a
    ///    min of magnitudes `min(|old|, |x|)`, for which every uniform fill above is the identity.
    ///
    /// `the_fills_catch_a_fold_seeded_from_the_output` pins each of those claims.
    fn prefill(fill: usize, i: usize) -> f32 {
        const MIXED: [f32; 7] = [
            7.0e-31,
            -2.5e30,
            f32::from_bits(0x7fc0_5678),
            2.5e30,
            -0.0,
            -7.0e-31,
            0.5,
        ];
        match fill {
            0 => f32::from_bits(0x7fc0_1234),
            1 => -3.0e30,
            2 => 3.0e30,
            _ => MIXED[i % MIXED.len()],
        }
    }

    /// How many output pre-fills the contract is checked under (see [`prefill`]).
    const FILLS: usize = 4;

    /// Every pre-fill, `room` elements each.
    fn fill_buffers(room: usize) -> Vec<Vec<f32>> {
        (0..FILLS)
            .map(|f| (0..room).map(|i| prefill(f, i)).collect())
            .collect()
    }

    /// Run `kernel` once per buffer of `fills`, each time over a fresh copy of that buffer's first
    /// `room` elements, and return the first element of `[0, len)` whose bits differ between two
    /// runs, with its bits in each. `None`: the result there did not depend on the fill.
    fn fill_dependence(
        fills: &[Vec<f32>],
        room: usize,
        len: usize,
        mut kernel: impl FnMut(&mut [f32]),
    ) -> Option<(usize, Vec<u32>)> {
        let results: Vec<Vec<u32>> = fills
            .iter()
            .map(|fill| {
                let mut out = fill[..room].to_vec();
                kernel(&mut out);
                out[..len].iter().map(|v| v.to_bits()).collect()
            })
            .collect();
        (0..len)
            .find(|&i| results.iter().any(|r| r[i] != results[0][i]))
            .map(|i| (i, results.iter().map(|r| r[i]).collect()))
    }

    /// Every entry's contract holds on the kernel itself: for each test call that meets the entry's
    /// conditions, the stated output extent comes out bit-identical under every [`prefill`] — every
    /// element is written, and none is computed from its old value by a fold the fills tell apart
    /// (the module note says what that leaves). Each call is also run with every integer operand
    /// the entry leaves unconstrained set to 0 and to -1: a value the entry admits must not reach an
    /// early exit.
    #[test]
    fn every_contract_holds_on_the_kernel() {
        let fills = fill_buffers(MAX_ROOM);
        let mut checked = 0usize;
        for c in OVERWRITES {
            let cases = calls(c.symbol);
            assert!(!cases.is_empty(), "{}", c.symbol);
            let constrained = |k: usize| {
                c.extent.contains(&k) || c.positive.contains(&k) || c.fixed.iter().any(|f| f.0 == k)
            };
            let mut variants: Vec<Vec<Arg>> = Vec::new();
            for args in cases {
                for (k, a) in args.iter().enumerate() {
                    if matches!(a, Arg::Int(_)) && !constrained(k) {
                        for v in [0, -1] {
                            let mut probe = args.clone();
                            probe[k] = Arg::Int(v);
                            variants.push(probe);
                        }
                    }
                }
                variants.push(args);
            }
            for args in variants {
                assert_eq!(args.len(), c.arity, "{}: arity", c.symbol);
                assert!(
                    matches!(args[c.out], Arg::Out),
                    "{}: the output operand",
                    c.symbol
                );
                let int = |k: usize| match args[k] {
                    Arg::Int(v) => v,
                    other => panic!("{}: operand {k} is {other:?}, not an integer", c.symbol),
                };
                // Every test call is one the contract speaks for.
                let applies = c.extent.iter().chain(c.positive).all(|&k| int(k) >= 1)
                    && c.fixed.iter().all(|(k, vals)| vals.contains(&int(*k)));
                assert!(
                    applies,
                    "{}: a test call outside the contract: {args:?}",
                    c.symbol
                );
                let len: usize = c.extent.iter().map(|&k| int(k) as usize).product();
                let room = room_for(&args);
                assert!(len <= room && room <= MAX_ROOM, "{}", c.symbol);
                let inputs: Vec<Vec<f32>> = args
                    .iter()
                    .enumerate()
                    .filter_map(|(k, a)| match a {
                        Arg::In(n) => Some(input(*n, k)),
                        _ => None,
                    })
                    .collect();
                let found = fill_dependence(&fills, room, len, |outbuf| {
                    // A fresh copy of the inputs per run, so a kernel that wrongly wrote one cannot
                    // feed the next run different operands.
                    let mut owned = inputs.clone();
                    let mut ins = owned.iter_mut();
                    let vals: Vec<Val> = args
                        .iter()
                        .map(|a| match a {
                            Arg::In(_) => Val::Ptr(ins.next().expect("an input").as_mut_ptr()),
                            Arg::Out => Val::Ptr(outbuf.as_mut_ptr()),
                            Arg::Int(v) => Val::Int(*v),
                            Arg::Float(v) => Val::Float(*v),
                        })
                        .collect();
                    // SAFETY: every input is sized by `calls` for the shape it is passed with, and
                    // the output buffer is larger than any output a test shape produces.
                    unsafe { invoke(c.symbol, &vals) };
                });
                if let Some((i, bits)) = found {
                    let bits: Vec<String> = bits.iter().map(|b| format!("{b:#010x}")).collect();
                    panic!(
                        "{}: element {i} of {len} depends on the output's old contents \
                         ({}) for operands {args:?}",
                        c.symbol,
                        bits.join(" vs ")
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 3000, "only {checked} calls checked");
    }

    /// The fills catch a kernel that folds into its output instead of seeding the fold from its input.
    /// Each mutant is a column reduction `out[j] = fold(out[j], x[i, j])` over every row `i` — what
    /// `wukong_colmax_f32` would be without its row-0 seed — with the runtime's scalar max and min
    /// rule (`(a > b) ? a : b`, the twin of `_mm256_max_ps(a, b)`, which returns `b` when `a` is a
    /// NaN). Every one reads its output, so the fills must tell each apart; the real kernel, which
    /// seeds from row 0, must pass. The pins at the end hold [`prefill`]'s note to the fills: each
    /// extra fill is there because the fills before it pass a fold.
    #[test]
    fn the_fills_catch_a_fold_seeded_from_the_output() {
        fn max(a: f32, b: f32) -> f32 {
            if a > b {
                a
            } else {
                b
            }
        }
        fn min(a: f32, b: f32) -> f32 {
            if a < b {
                a
            } else {
                b
            }
        }
        fn running_max(o: f32, v: f32) -> f32 {
            max(o, v)
        }
        fn running_max_abs(o: f32, v: f32) -> f32 {
            max(o, v.abs())
        }
        fn running_min(o: f32, v: f32) -> f32 {
            min(o, v)
        }
        fn running_min_of_magnitudes(o: f32, v: f32) -> f32 {
            min(o.abs(), v.abs())
        }
        fn running_sum(o: f32, v: f32) -> f32 {
            o + v
        }
        fn running_product(o: f32, v: f32) -> f32 {
            o * v
        }
        type Fold = fn(f32, f32) -> f32;
        /// `out[j] = fold(out[j], x[i, j])` for every row `i` of a `[rows, cols]` matrix.
        fn fold_columns(out: &mut [f32], x: &[f32], rows: usize, cols: usize, fold: Fold) {
            for i in 0..rows {
                for j in 0..cols {
                    out[j] = fold(out[j], x[i * cols + j]);
                }
            }
        }
        let folds: [(&str, Fold); 6] = [
            ("max", running_max),
            ("max of magnitudes", running_max_abs),
            ("min", running_min),
            ("min of magnitudes", running_min_of_magnitudes),
            ("sum", running_sum),
            ("product", running_product),
        ];
        let fills = fill_buffers(OUT_ROOM);
        for (rows, cols) in [(1usize, 1usize), (5, 3), (33, 40)] {
            let x = input(rows * cols, 0);
            for (name, fold) in folds {
                assert!(
                    fill_dependence(&fills, OUT_ROOM, cols, |out| {
                        fold_columns(out, &x, rows, cols, fold)
                    })
                    .is_some(),
                    "a running {name} seeded from the output passes every fill ({rows}x{cols})"
                );
            }
            let colmax = |out: &mut [f32]| {
                // SAFETY: `x` holds rows*cols elements and `out` at least `cols`.
                unsafe { wukong_colmax_f32(x.as_ptr(), out.as_mut_ptr(), rows as i64, cols as i64) }
            };
            assert_eq!(fill_dependence(&fills, OUT_ROOM, cols, colmax), None);
        }
        // The pins: which prefix of the fills first catches the folds the earlier fills pass.
        let (rows, cols) = (33, 40);
        let x = input(rows * cols, 0);
        let caught = |n: usize, fold: Fold| {
            fill_dependence(&fills[..n], OUT_ROOM, cols, |out| {
                fold_columns(out, &x, rows, cols, fold)
            })
            .is_some()
        };
        // The NaN and -3e30 both pass a running max and a max of magnitudes; +3e30 catches them.
        for fold in [running_max, running_max_abs] {
            assert!(!caught(2, fold) && caught(3, fold));
        }
        // Every uniform fill passes a min of magnitudes; the per-element pattern catches it.
        assert!(!caught(3, running_min_of_magnitudes) && caught(4, running_min_of_magnitudes));
    }

    /// Every `_parallel` entry with a size gate of its own runs past it on at least one test call, so
    /// `every_contract_holds_on_the_kernel` holds its parallel path to the contract, not only the
    /// serial kernel it falls back to below the gate.
    #[test]
    fn every_gated_parallel_entry_runs_past_its_gate() {
        let mut gated = 0;
        for c in OVERWRITES
            .iter()
            .filter(|c| c.symbol.ends_with("_parallel"))
        {
            let past: Vec<Option<bool>> = calls(c.symbol)
                .iter()
                .map(|args| past_size_gate(c.symbol, args))
                .collect();
            if past.iter().all(Option::is_none) {
                continue;
            }
            gated += 1;
            assert!(
                past.contains(&Some(true)),
                "{}: no test call reaches its parallel path",
                c.symbol
            );
        }
        // The five GEMMs, vmath, velem and transpose.
        assert_eq!(gated, 8);
    }

    /// The table is well-formed: one entry per symbol, and every operand index it names is inside
    /// the call and is not the output.
    #[test]
    fn the_table_is_well_formed() {
        for (i, c) in OVERWRITES.iter().enumerate() {
            assert!(
                OVERWRITES[..i].iter().all(|o| o.symbol != c.symbol),
                "{} listed twice",
                c.symbol
            );
            assert!(c.out < c.arity && c.elem_bytes > 0 && !c.extent.is_empty());
            let named = c
                .extent
                .iter()
                .chain(c.positive)
                .chain(c.fixed.iter().map(|f| &f.0));
            for &k in named {
                assert!(k < c.arity && k != c.out, "{}: operand {k}", c.symbol);
            }
            assert_eq!(overwrite_of(c.symbol).map(|o| o.symbol), Some(c.symbol));
        }
        assert!(overwrite_of("wukong_sreduce_f32").is_none());
    }
}
