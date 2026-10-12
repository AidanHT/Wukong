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
//! value, twice — once over an output pre-filled with a NaN pattern, once over a large finite
//! pattern — and requires the two results to be bit-identical over the whole stated extent. An
//! element the kernel leaves alone keeps two different fills; an element computed from its old value
//! almost always differs too. Every integer operand an entry leaves unconstrained (an activation or
//! op code, an `eps`, a `causal` flag) is also probed at 0 and -1, so an early exit on a value the
//! entry admits fails the test as well.

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

    /// Every operand list the test runs `symbol` with: each shape crossed with each value its
    /// op-code operands take. The shapes include 1-element, odd and multi-tile sizes.
    fn calls(symbol: &str) -> Vec<Vec<Arg>> {
        use Arg::*;
        let shapes3: [(i64, i64, i64); 5] =
            [(1, 1, 1), (3, 5, 7), (7, 13, 2), (17, 33, 9), (64, 70, 48)];
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
                for (m, k, n) in shapes3 {
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
                for (m, k, n) in shapes3 {
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
                for (m, k, n) in shapes3 {
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
                for n in lens {
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
                for n in lens {
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
            "wukong_transpose_f32"
            | "wukong_transpose_f32_parallel"
            | "wukong_cumsum_f32"
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

    /// Room for any output a test shape can produce, whatever extent the table claims: an entry that
    /// misstates the output must make the test fail, not write past the buffer.
    const OUT_ROOM: usize = 1 << 16;

    /// Every entry's contract holds on the kernel itself: for each test call that meets the entry's
    /// conditions, the stated output extent comes out bit-identical whether the buffer held a NaN
    /// pattern or a large finite one before the call — every element is written, and none is
    /// computed from its old value. Each call is also run with every integer operand the entry
    /// leaves unconstrained set to 0 and to -1: a value the entry admits must not reach an early
    /// exit. See the module note.
    #[test]
    fn every_contract_holds_on_the_kernel() {
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
                assert!(len <= OUT_ROOM, "{}", c.symbol);
                let mut results: Vec<Vec<u32>> = Vec::new();
                for fill in [f32::from_bits(0x7fc0_1234), -3.0e30f32] {
                    let mut inputs: Vec<Vec<f32>> = args
                        .iter()
                        .enumerate()
                        .filter_map(|(k, a)| match a {
                            Arg::In(n) => Some(input(*n, k)),
                            _ => None,
                        })
                        .collect();
                    let mut outbuf = vec![fill; OUT_ROOM];
                    let mut ins = inputs.iter_mut();
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
                    results.push(outbuf[..len].iter().map(|v| v.to_bits()).collect());
                }
                if let Some(i) = (0..len).find(|&i| results[0][i] != results[1][i]) {
                    panic!(
                        "{}: element {i} of {len} depends on the output's old contents \
                         ({:#010x} vs {:#010x}) for operands {args:?}",
                        c.symbol, results[0][i], results[1][i]
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 3000, "only {checked} calls checked");
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
