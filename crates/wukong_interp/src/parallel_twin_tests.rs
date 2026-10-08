//! The interpreter calls the SERIAL runtime kernel for a symbol and for its `_parallel` twin alike
//! (only the bf16/f16 reductions call their parallel kernel, because their serial form folds the
//! array flat). That substitution is what keeps the oracle single-threaded, and it is exact only if
//! every such parallel kernel is bit-identical to its serial form ON INPUTS LARGE ENOUGH TO TAKE ITS
//! PARALLEL PATH: below the runtime's own gates (`PAR_MIN_MACS = 1 << 23` for GEMM,
//! `TRANSPOSE_PAR_MIN = 1 << 16`, 8..256 rows or columns elsewhere) a `_parallel` entry simply runs
//! the serial loop, and the corpus's largest program never reaches the GEMM or transpose gate. So
//! `native_matches_interpreter` cannot prove the substitution for those families from a compiled
//! program; this does, directly, for the families whose interpreter arm makes it.
//!
//! On a single-worker pool a parallel entry degenerates to serial and these pass trivially, which
//! is the correct outcome: the claim is "whatever native runs equals what the oracle runs".

type Gemm = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64, i64, i64, i64);
type Unary = unsafe extern "C" fn(*const f32, *mut f32, i64, i64);
type UnaryIndex = unsafe extern "C" fn(*const f32, *mut i32, i64, i64);
type Binary = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64, i64);

/// Deterministic values in `[-1, 1)` — no NaN, no infinity, few exact ties.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let top = (s >> 40) as i64 - (1 << 23);
            top as f32 / (1 << 23) as f32
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn gemm_parallel_twins_equal_the_serial_kernel_the_interpreter_calls() {
    // 208^3 = 8,998,912 multiply-adds, above `PAR_MIN_MACS` (8,388,608).
    let (m, k, n) = (208usize, 208usize, 208usize);
    let a = values(1, m * k);
    let b = values(2, k * n);
    let twins: [(&str, Gemm, Gemm); 3] = [
        (
            "nn",
            wukong_runtime::wukong_sgemm,
            wukong_runtime::wukong_sgemm_parallel,
        ),
        (
            "nt",
            wukong_runtime::wukong_sgemm_nt,
            wukong_runtime::wukong_sgemm_nt_parallel,
        ),
        (
            "tn",
            wukong_runtime::wukong_sgemm_tn,
            wukong_runtime::wukong_sgemm_tn_parallel,
        ),
    ];
    for (layout, serial, parallel) in twins {
        for beta in [0i64, 1] {
            let seed = values(3, m * n);
            let mut cs = seed.clone();
            let mut cp = seed;
            // SAFETY: a is m*k, b is k*n (n*k for NT, k*m/k*n for TN: the same counts), c is m*n.
            unsafe {
                serial(
                    a.as_ptr(),
                    b.as_ptr(),
                    cs.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    beta,
                );
                parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    cp.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    beta,
                );
            }
            assert_eq!(bits(&cs), bits(&cp), "sgemm {layout} beta {beta}");
        }
    }
}

#[test]
fn matrix_vector_and_transpose_parallel_twins_equal_their_serial_kernels() {
    let (rows, cols) = (300usize, 300usize);
    let a = values(4, rows * cols);
    let v = values(5, rows.max(cols));

    let mut ys = vec![0.0f32; rows];
    let mut yp = vec![0.0f32; rows];
    let mut gs = vec![0.0f32; cols];
    let mut gp = vec![0.0f32; cols];
    // 300 * 300 = 90,000 elements, above `TRANSPOSE_PAR_MIN` (65,536).
    let mut ts = vec![0.0f32; rows * cols];
    let mut tp = vec![0.0f32; rows * cols];
    // SAFETY: every buffer has exactly the extent its kernel reads or writes.
    unsafe {
        wukong_runtime::wukong_sgemv(
            a.as_ptr(),
            v.as_ptr(),
            ys.as_mut_ptr(),
            rows as i64,
            cols as i64,
        );
        wukong_runtime::wukong_sgemv_parallel(
            a.as_ptr(),
            v.as_ptr(),
            yp.as_mut_ptr(),
            rows as i64,
            cols as i64,
        );
        wukong_runtime::wukong_sgevm_f32(
            v.as_ptr(),
            a.as_ptr(),
            gs.as_mut_ptr(),
            rows as i64,
            cols as i64,
            0.5,
        );
        wukong_runtime::wukong_sgevm_f32_parallel(
            v.as_ptr(),
            a.as_ptr(),
            gp.as_mut_ptr(),
            rows as i64,
            cols as i64,
            0.5,
        );
        wukong_runtime::wukong_transpose_f32(a.as_ptr(), ts.as_mut_ptr(), rows as i64, cols as i64);
        wukong_runtime::wukong_transpose_f32_parallel(
            a.as_ptr(),
            tp.as_mut_ptr(),
            rows as i64,
            cols as i64,
        );
    }
    assert_eq!(bits(&ys), bits(&yp), "sgemv");
    assert_eq!(bits(&gs), bits(&gp), "sgevm");
    assert_eq!(bits(&ts), bits(&tp), "transpose");
}

#[test]
fn row_and_column_parallel_twins_equal_their_serial_kernels() {
    // 300 rows and 300 columns clear every row/column gate in the runtime (the largest is 256).
    let (rows, cols) = (300usize, 300usize);
    let x = values(6, rows * cols);
    let y = values(7, rows * cols);
    // Strictly positive values for the unary group, so entropy's `p * log p` is finite: the claim
    // is bit-identity, and a NaN on both sides would make it weaker than it looks.
    let pos: Vec<f32> = x.iter().map(|v| v.abs() + 1e-3).collect();

    let unary: [(&str, Unary, Unary, usize); 6] = [
        (
            "colsum",
            wukong_runtime::wukong_colsum_f32,
            wukong_runtime::wukong_colsum_f32_parallel,
            cols,
        ),
        (
            "colmax",
            wukong_runtime::wukong_colmax_f32,
            wukong_runtime::wukong_colmax_f32_parallel,
            cols,
        ),
        (
            "cumsum",
            wukong_runtime::wukong_cumsum_f32,
            wukong_runtime::wukong_cumsum_f32_parallel,
            rows * cols,
        ),
        (
            "cumprod",
            wukong_runtime::wukong_cumprod_f32,
            wukong_runtime::wukong_cumprod_f32_parallel,
            rows * cols,
        ),
        (
            "logsumexp",
            wukong_runtime::wukong_logsumexp_f32,
            wukong_runtime::wukong_logsumexp_f32_parallel,
            rows,
        ),
        (
            "entropy",
            wukong_runtime::wukong_entropy_f32,
            wukong_runtime::wukong_entropy_f32_parallel,
            rows,
        ),
    ];
    for (name, serial, parallel, out_len) in unary {
        let mut os = vec![0.0f32; out_len];
        let mut op = vec![0.0f32; out_len];
        // SAFETY: pos is rows*cols; the output has the extent the kernel writes.
        unsafe {
            serial(pos.as_ptr(), os.as_mut_ptr(), rows as i64, cols as i64);
            parallel(pos.as_ptr(), op.as_mut_ptr(), rows as i64, cols as i64);
        }
        assert_eq!(bits(&os), bits(&op), "{name}");
    }

    let index: [(&str, UnaryIndex, UnaryIndex, usize); 2] = [
        (
            "rowargmax",
            wukong_runtime::wukong_rowargmax_i32,
            wukong_runtime::wukong_rowargmax_i32_parallel,
            rows,
        ),
        (
            "colargmin",
            wukong_runtime::wukong_colargmin_i32,
            wukong_runtime::wukong_colargmin_i32_parallel,
            cols,
        ),
    ];
    for (name, serial, parallel, out_len) in index {
        let mut os = vec![0i32; out_len];
        let mut op = vec![0i32; out_len];
        // SAFETY: x is rows*cols; the index output has the extent the kernel writes.
        unsafe {
            serial(x.as_ptr(), os.as_mut_ptr(), rows as i64, cols as i64);
            parallel(x.as_ptr(), op.as_mut_ptr(), rows as i64, cols as i64);
        }
        assert_eq!(os, op, "{name}");
    }

    let binary: [(&str, Binary, Binary); 2] = [
        (
            "softmax_bwd",
            wukong_runtime::wukong_softmax_bwd_f32,
            wukong_runtime::wukong_softmax_bwd_f32_parallel,
        ),
        (
            "lrscan",
            wukong_runtime::wukong_lrscan_f32,
            wukong_runtime::wukong_lrscan_f32_parallel,
        ),
    ];
    for (name, serial, parallel) in binary {
        let mut os = vec![0.0f32; rows * cols];
        let mut op = vec![0.0f32; rows * cols];
        // SAFETY: both inputs and the output are rows*cols.
        unsafe {
            serial(
                x.as_ptr(),
                y.as_ptr(),
                os.as_mut_ptr(),
                rows as i64,
                cols as i64,
            );
            parallel(
                x.as_ptr(),
                y.as_ptr(),
                op.as_mut_ptr(),
                rows as i64,
                cols as i64,
            );
        }
        assert_eq!(bits(&os), bits(&op), "{name}");
    }

    let eps = i64::from(1e-5f32.to_bits());
    let mut ns = vec![0.0f32; rows * cols];
    let mut np = vec![0.0f32; rows * cols];
    // SAFETY: x and both outputs are rows*cols.
    unsafe {
        wukong_runtime::wukong_norm_f32(
            x.as_ptr(),
            ns.as_mut_ptr(),
            rows as i64,
            cols as i64,
            eps,
            wukong_runtime::NORM_SOFTMAX,
        );
        wukong_runtime::wukong_norm_f32_parallel(
            x.as_ptr(),
            np.as_mut_ptr(),
            rows as i64,
            cols as i64,
            eps,
            wukong_runtime::NORM_SOFTMAX,
        );
    }
    assert_eq!(bits(&ns), bits(&np), "norm softmax");
}

/// The fused, scaled, quantized and half-precision GEMM entries: every one of their `_parallel`
/// names is answered by the interpreter's SERIAL arm, so each pair must be bit-identical above the
/// runtime's parallel gate (`PAR_MIN_MACS` for the float forms; the int8 forms fork per row with no
/// gate at all).
#[test]
fn fused_scaled_quantized_and_half_gemm_twins_equal_their_serial_kernels() {
    // 208^3 = 8,998,912 multiply-adds, above `PAR_MIN_MACS` (8,388,608).
    let (m, k, n) = (208usize, 208usize, 208usize);
    let (mi, ki, ni) = (m as i64, k as i64, n as i64);
    let a = values(11, m * k);
    let b = values(12, n * k);
    let bias = values(13, n);
    let seed = values(14, m * n);
    let ah: Vec<u16> = a
        .iter()
        .map(|&v| wukong_runtime::f32_to_bf16_bits(v))
        .collect();
    let bh: Vec<u16> = b
        .iter()
        .map(|&v| wukong_runtime::f32_to_bf16_bits(v))
        .collect();
    for beta in [0i64, 1] {
        // `C = silu(A·Bᵀ + bias)` (act 3), the fused nn.Linear epilogue.
        let (mut cs, mut cp) = (seed.clone(), seed.clone());
        // SAFETY: a is m*k, b is n*k, c is m*n, bias is n — the kernels' contract.
        unsafe {
            wukong_runtime::wukong_sgemm_nt_epi(
                a.as_ptr(),
                b.as_ptr(),
                cs.as_mut_ptr(),
                mi,
                ki,
                ni,
                beta,
                bias.as_ptr(),
                3,
            );
            wukong_runtime::wukong_sgemm_nt_epi_parallel(
                a.as_ptr(),
                b.as_ptr(),
                cp.as_mut_ptr(),
                mi,
                ki,
                ni,
                beta,
                bias.as_ptr(),
                3,
            );
        }
        assert_eq!(bits(&cs), bits(&cp), "sgemm_nt_epi beta {beta}");
        // `C = alpha·(A·Bᵀ)`, the attention-score form.
        let (mut cs, mut cp) = (seed.clone(), seed.clone());
        // SAFETY: as above.
        unsafe {
            wukong_runtime::wukong_sgemm_nt_alpha(
                a.as_ptr(),
                b.as_ptr(),
                cs.as_mut_ptr(),
                mi,
                ki,
                ni,
                beta,
                0.125,
            );
            wukong_runtime::wukong_sgemm_nt_alpha_parallel(
                a.as_ptr(),
                b.as_ptr(),
                cp.as_mut_ptr(),
                mi,
                ki,
                ni,
                beta,
                0.125,
            );
        }
        assert_eq!(bits(&cs), bits(&cp), "sgemm_nt_alpha beta {beta}");
        // bf16 operands, f32 accumulate and output.
        let (mut cs, mut cp) = (seed.clone(), seed.clone());
        // SAFETY: ah is m*k and bh is n*k bf16 bits; c is m*n.
        unsafe {
            wukong_runtime::wukong_sgemm_bf16_nt(
                ah.as_ptr(),
                bh.as_ptr(),
                cs.as_mut_ptr(),
                mi,
                ki,
                ni,
                beta,
            );
            wukong_runtime::wukong_sgemm_bf16_nt_parallel(
                ah.as_ptr(),
                bh.as_ptr(),
                cp.as_mut_ptr(),
                mi,
                ki,
                ni,
                beta,
            );
        }
        assert_eq!(bits(&cs), bits(&cp), "sgemm_bf16_nt beta {beta}");
    }

    // The scaled matrix-vector product: 208 rows clears the GEMV row gate.
    let x = values(15, n);
    let (mut ys, mut yp) = (vec![0.0f32; m], vec![0.0f32; m]);
    // SAFETY: a is m*n (m == n here), x is n, y is m.
    unsafe {
        wukong_runtime::wukong_sgemv_alpha(a.as_ptr(), x.as_ptr(), ys.as_mut_ptr(), mi, ni, 0.5);
        wukong_runtime::wukong_sgemv_alpha_parallel(
            a.as_ptr(),
            x.as_ptr(),
            yp.as_mut_ptr(),
            mi,
            ni,
            0.5,
        );
    }
    assert_eq!(bits(&ys), bits(&yp), "sgemv_alpha");

    // u8 x i8 -> i32, and the fused per-channel dequant (with and without a bias).
    let (qm, qk, qn) = (64usize, 96usize, 48usize);
    let qa: Vec<u8> = values(16, qm * qk)
        .iter()
        .map(|&v| ((v + 1.0) * 127.5) as u8)
        .collect();
    let qb: Vec<i8> = values(17, qn * qk)
        .iter()
        .map(|&v| (v * 127.0) as i8)
        .collect();
    let (mut is, mut ip) = (vec![0i32; qm * qn], vec![0i32; qm * qn]);
    // SAFETY: qa is m*k u8, qb is n*k i8, the accumulator is m*n i32.
    unsafe {
        wukong_runtime::wukong_i8gemm_nt(
            qa.as_ptr(),
            qb.as_ptr(),
            is.as_mut_ptr(),
            qm as i64,
            qk as i64,
            qn as i64,
        );
        wukong_runtime::wukong_i8gemm_nt_parallel(
            qa.as_ptr(),
            qb.as_ptr(),
            ip.as_mut_ptr(),
            qm as i64,
            qk as i64,
            qn as i64,
        );
    }
    assert_eq!(is, ip, "i8gemm_nt");
    let scale_b: Vec<f32> = values(18, qn).iter().map(|v| v.abs() + 0.5).collect();
    let qbias = values(19, qn);
    for with_bias in [false, true] {
        let bias_ptr = if with_bias {
            qbias.as_ptr()
        } else {
            std::ptr::null()
        };
        let (mut ds, mut dp) = (vec![0.0f32; qm * qn], vec![0.0f32; qm * qn]);
        // SAFETY: as above; scale_b is n; the bias is null or n — the kernels' contract.
        unsafe {
            wukong_runtime::wukong_i8gemm_nt_deq(
                qa.as_ptr(),
                qb.as_ptr(),
                ds.as_mut_ptr(),
                qm as i64,
                qk as i64,
                qn as i64,
                0.25,
                scale_b.as_ptr(),
                bias_ptr,
                3,
            );
            wukong_runtime::wukong_i8gemm_nt_deq_parallel(
                qa.as_ptr(),
                qb.as_ptr(),
                dp.as_mut_ptr(),
                qm as i64,
                qk as i64,
                qn as i64,
                0.25,
                scale_b.as_ptr(),
                bias_ptr,
                3,
            );
        }
        assert_eq!(bits(&ds), bits(&dp), "i8gemm_nt_deq bias {with_bias}");
    }
}

type NormBwd = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, i64, i64, i64);
type Pool = unsafe extern "C" fn(*const f32, *mut f32, i64, i64, i64, i64, i64, i64, i64);

/// The row-, column- and element-parallel families the interpreter also answers with the serial
/// kernel: norm backwards, the losses, RoPE, the extreme scans, column/row arg-reductions, the
/// affine norm, broadcast bias, pooling, the 16-bit transpose, per-channel dequant, and the
/// streaming elementwise maps. Every shape clears its family's parallel gate (300 rows, 256
/// columns; a 300x300 transpose > `TRANSPOSE_PAR_MIN`; 16 pooling channels > `POOL_PAR_MIN`; 2^19
/// elements > the 256 Ki `velem` gate; 2^17 > `VMATH_PAR_MIN`).
#[test]
fn row_loss_norm_pool_and_elementwise_twins_equal_their_serial_kernels() {
    let (rows, cols) = (300usize, 256usize);
    let (ri, ci) = (rows as i64, cols as i64);
    let x = values(21, rows * cols);
    let dy = values(22, rows * cols);
    let pos: Vec<f32> = x.iter().map(|v| v.abs() + 1e-3).collect();
    let qpos: Vec<f32> = dy.iter().map(|v| v.abs() + 1e-3).collect();
    let gamma = values(23, cols);
    let beta = values(24, cols);
    let eps = i64::from(1e-5f32.to_bits());
    let full = || vec![0.0f32; rows * cols];

    // Norm backwards: dx from (x, dy, gamma).
    let norm_bwd: [(&str, NormBwd, NormBwd); 2] = [
        (
            "rmsnorm_bwd",
            wukong_runtime::wukong_rmsnorm_bwd_f32,
            wukong_runtime::wukong_rmsnorm_bwd_f32_parallel,
        ),
        (
            "layernorm_bwd",
            wukong_runtime::wukong_layernorm_bwd_f32,
            wukong_runtime::wukong_layernorm_bwd_f32_parallel,
        ),
    ];
    for (name, serial, parallel) in norm_bwd {
        let (mut ds, mut dp) = (full(), full());
        // SAFETY: x, dy and dx are rows*cols; gamma is cols.
        unsafe {
            serial(
                x.as_ptr(),
                dy.as_ptr(),
                gamma.as_ptr(),
                ds.as_mut_ptr(),
                ri,
                ci,
                eps,
            );
            parallel(
                x.as_ptr(),
                dy.as_ptr(),
                gamma.as_ptr(),
                dp.as_mut_ptr(),
                ri,
                ci,
                eps,
            );
        }
        assert_eq!(bits(&ds), bits(&dp), "{name}");
    }

    // Hard-label cross-entropy, forward and backward; the targets spread over the columns.
    let target: Vec<i32> = (0..rows).map(|r| ((r * 7) % cols) as i32).collect();
    let (mut ls, mut lp) = (vec![0.0f32; rows], vec![0.0f32; rows]);
    let (mut gs, mut gp) = (full(), full());
    // SAFETY: x is rows*cols; target and loss are rows; the gradient is rows*cols.
    unsafe {
        wukong_runtime::wukong_xent_fwd_f32(x.as_ptr(), target.as_ptr(), ls.as_mut_ptr(), ri, ci);
        wukong_runtime::wukong_xent_fwd_f32_parallel(
            x.as_ptr(),
            target.as_ptr(),
            lp.as_mut_ptr(),
            ri,
            ci,
        );
        wukong_runtime::wukong_xent_bwd_f32(x.as_ptr(), target.as_ptr(), gs.as_mut_ptr(), ri, ci);
        wukong_runtime::wukong_xent_bwd_f32_parallel(
            x.as_ptr(),
            target.as_ptr(),
            gp.as_mut_ptr(),
            ri,
            ci,
        );
    }
    assert_eq!(bits(&ls), bits(&lp), "xent_fwd");
    assert_eq!(bits(&gs), bits(&gp), "xent_bwd");

    // KL divergence over two positive distributions, and soft-label cross-entropy.
    let losses: [(&str, Binary, Binary, &[f32], &[f32]); 2] = [
        (
            "kldiv",
            wukong_runtime::wukong_kldiv_f32,
            wukong_runtime::wukong_kldiv_f32_parallel,
            &pos,
            &qpos,
        ),
        (
            "kd_loss",
            wukong_runtime::wukong_kd_loss_f32,
            wukong_runtime::wukong_kd_loss_f32_parallel,
            &x,
            &qpos,
        ),
    ];
    for (name, serial, parallel, p, q) in losses {
        let (mut os, mut op) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        // SAFETY: p and q are rows*cols; the output is rows.
        unsafe {
            serial(p.as_ptr(), q.as_ptr(), os.as_mut_ptr(), ri, ci);
            parallel(p.as_ptr(), q.as_ptr(), op.as_mut_ptr(), ri, ci);
        }
        assert_eq!(bits(&os), bits(&op), "{name}");
    }

    // RoPE forward and backward: each row rotates `half = cols / 2` pairs.
    let half = cols / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| 1.0 / (1.0 + i as f32)).collect();
    let rope: [(&str, Binary, Binary); 2] = [
        (
            "rope",
            wukong_runtime::wukong_rope_f32,
            wukong_runtime::wukong_rope_f32_parallel,
        ),
        (
            "rope_bwd",
            wukong_runtime::wukong_rope_bwd_f32,
            wukong_runtime::wukong_rope_bwd_f32_parallel,
        ),
    ];
    for (name, serial, parallel) in rope {
        let (mut os, mut op) = (full(), full());
        // SAFETY: x and the output are rows * 2*half; inv_freq is half.
        unsafe {
            serial(
                x.as_ptr(),
                inv_freq.as_ptr(),
                os.as_mut_ptr(),
                ri,
                half as i64,
            );
            parallel(
                x.as_ptr(),
                inv_freq.as_ptr(),
                op.as_mut_ptr(),
                ri,
                half as i64,
            );
        }
        assert_eq!(bits(&os), bits(&op), "{name}");
    }

    // Extreme scans and the column mean (output length per family).
    let unary: [(&str, Unary, Unary, usize); 3] = [
        (
            "cummax",
            wukong_runtime::wukong_cummax_f32,
            wukong_runtime::wukong_cummax_f32_parallel,
            rows * cols,
        ),
        (
            "cummin",
            wukong_runtime::wukong_cummin_f32,
            wukong_runtime::wukong_cummin_f32_parallel,
            rows * cols,
        ),
        (
            "colmean",
            wukong_runtime::wukong_colmean_f32,
            wukong_runtime::wukong_colmean_f32_parallel,
            cols,
        ),
    ];
    for (name, serial, parallel, out_len) in unary {
        let (mut os, mut op) = (vec![0.0f32; out_len], vec![0.0f32; out_len]);
        // SAFETY: x is rows*cols; the output has the extent the kernel writes.
        unsafe {
            serial(x.as_ptr(), os.as_mut_ptr(), ri, ci);
            parallel(x.as_ptr(), op.as_mut_ptr(), ri, ci);
        }
        assert_eq!(bits(&os), bits(&op), "{name}");
    }
    let index: [(&str, UnaryIndex, UnaryIndex, usize); 2] = [
        (
            "colargmax",
            wukong_runtime::wukong_colargmax_i32,
            wukong_runtime::wukong_colargmax_i32_parallel,
            cols,
        ),
        (
            "rowargmin",
            wukong_runtime::wukong_rowargmin_i32,
            wukong_runtime::wukong_rowargmin_i32_parallel,
            rows,
        ),
    ];
    for (name, serial, parallel, out_len) in index {
        let (mut os, mut op) = (vec![0i32; out_len], vec![0i32; out_len]);
        // SAFETY: x is rows*cols; the index output has the extent the kernel writes.
        unsafe {
            serial(x.as_ptr(), os.as_mut_ptr(), ri, ci);
            parallel(x.as_ptr(), op.as_mut_ptr(), ri, ci);
        }
        assert_eq!(os, op, "{name}");
    }

    // Affine LayerNorm and the broadcast bias add.
    let (mut ns, mut np) = (full(), full());
    let (mut bs, mut bp) = (full(), full());
    // SAFETY: x and the outputs are rows*cols; gamma, beta and the bias row are cols.
    unsafe {
        wukong_runtime::wukong_norm_affine_f32(
            x.as_ptr(),
            ns.as_mut_ptr(),
            gamma.as_ptr(),
            beta.as_ptr(),
            ri,
            ci,
            eps,
            wukong_runtime::NORM_LAYERNORM,
        );
        wukong_runtime::wukong_norm_affine_f32_parallel(
            x.as_ptr(),
            np.as_mut_ptr(),
            gamma.as_ptr(),
            beta.as_ptr(),
            ri,
            ci,
            eps,
            wukong_runtime::NORM_LAYERNORM,
        );
        wukong_runtime::wukong_bias_bcast_f32(
            x.as_ptr(),
            beta.as_ptr(),
            bs.as_mut_ptr(),
            ri,
            ci,
            wukong_runtime::BIAS_ACT_NONE,
        );
        wukong_runtime::wukong_bias_bcast_f32_parallel(
            x.as_ptr(),
            beta.as_ptr(),
            bp.as_mut_ptr(),
            ri,
            ci,
            wukong_runtime::BIAS_ACT_NONE,
        );
    }
    assert_eq!(bits(&ns), bits(&np), "norm_affine layernorm");
    assert_eq!(bits(&bs), bits(&bp), "bias_bcast");

    // Pooling: 16 channels of 32x32, a 2x2 window at stride 2 -> 16 x 16 x 16.
    let (ch, h, w) = (16usize, 32usize, 32usize);
    let plane = values(25, ch * h * w);
    let pooled = ch * 16 * 16;
    let pool: [(&str, Pool, Pool); 2] = [
        (
            "maxpool2d",
            wukong_runtime::wukong_maxpool2d_f32,
            wukong_runtime::wukong_maxpool2d_f32_parallel,
        ),
        (
            "avgpool2d",
            wukong_runtime::wukong_avgpool2d_f32,
            wukong_runtime::wukong_avgpool2d_f32_parallel,
        ),
    ];
    for (name, serial, parallel) in pool {
        let (mut os, mut op) = (vec![0.0f32; pooled], vec![0.0f32; pooled]);
        // SAFETY: the plane is ch*h*w; the output is ch*oh*ow for this no-padding window.
        unsafe {
            serial(
                plane.as_ptr(),
                os.as_mut_ptr(),
                ch as i64,
                h as i64,
                w as i64,
                2,
                2,
                2,
                2,
            );
            parallel(
                plane.as_ptr(),
                op.as_mut_ptr(),
                ch as i64,
                h as i64,
                w as i64,
                2,
                2,
                2,
                2,
            );
        }
        assert_eq!(bits(&os), bits(&op), "{name}");
    }

    // The 16-bit transpose: 300 x 300 = 90,000 elements, above `TRANSPOSE_PAR_MIN` (65,536).
    let tn = 300usize;
    let src: Vec<u16> = values(26, tn * tn)
        .iter()
        .map(|&v| wukong_runtime::f32_to_bf16_bits(v))
        .collect();
    let (mut ts, mut tp) = (vec![0u16; tn * tn], vec![0u16; tn * tn]);
    // SAFETY: src and both outputs are tn*tn u16.
    unsafe {
        wukong_runtime::wukong_transpose_u16(src.as_ptr(), ts.as_mut_ptr(), tn as i64, tn as i64);
        wukong_runtime::wukong_transpose_u16_parallel(
            src.as_ptr(),
            tp.as_mut_ptr(),
            tn as i64,
            tn as i64,
        );
    }
    assert_eq!(ts, tp, "transpose_u16");

    // Per-channel int8 dequant over the [rows, cols] matrix.
    let q: Vec<u8> = x.iter().map(|&v| (v * 127.0) as i8 as u8).collect();
    let scale: Vec<f32> = gamma.iter().map(|v| v.abs() + 0.5).collect();
    let dq_op = wukong_runtime::DQ_ID | wukong_runtime::DQ_I8;
    let (mut qs, mut qp) = (full(), full());
    // SAFETY: q is rows*cols bytes; the output is rows*cols f32; scale is cols.
    unsafe {
        wukong_runtime::wukong_dequant_perchan_f32(
            q.as_ptr(),
            qs.as_mut_ptr(),
            ri,
            ci,
            scale.as_ptr(),
            dq_op,
        );
        wukong_runtime::wukong_dequant_perchan_f32_parallel(
            q.as_ptr(),
            qp.as_mut_ptr(),
            ri,
            ci,
            scale.as_ptr(),
            dq_op,
        );
    }
    assert_eq!(bits(&qs), bits(&qp), "dequant_perchan");

    // Streaming elementwise maps above their element gates.
    let big = 1usize << 19;
    let ex = values(27, big);
    let ey = values(28, big);
    let ve_op = wukong_runtime::VE_USE_Y | wukong_runtime::VE_RELU;
    let (mut es, mut ep) = (vec![0.0f32; big], vec![0.0f32; big]);
    // SAFETY: x, y and both outputs are `big` f32.
    unsafe {
        wukong_runtime::wukong_velem_f32(
            ex.as_ptr(),
            ey.as_ptr(),
            es.as_mut_ptr(),
            big as i64,
            0.5,
            -1.5,
            0.25,
            ve_op,
        );
        wukong_runtime::wukong_velem_f32_parallel(
            ex.as_ptr(),
            ey.as_ptr(),
            ep.as_mut_ptr(),
            big as i64,
            0.5,
            -1.5,
            0.25,
            ve_op,
        );
    }
    assert_eq!(bits(&es), bits(&ep), "velem");
    let vn = 1usize << 17;
    let (mut vs, mut vp) = (vec![0.0f32; vn], vec![0.0f32; vn]);
    // SAFETY: ex holds at least vn f32; both outputs are vn.
    unsafe {
        wukong_runtime::wukong_vmath_f32(
            ex.as_ptr(),
            vs.as_mut_ptr(),
            vn as i64,
            wukong_runtime::VM_TANH,
        );
        wukong_runtime::wukong_vmath_f32_parallel(
            ex.as_ptr(),
            vp.as_mut_ptr(),
            vn as i64,
            wukong_runtime::VM_TANH,
        );
    }
    assert_eq!(bits(&vs), bits(&vp), "vmath tanh");
}
