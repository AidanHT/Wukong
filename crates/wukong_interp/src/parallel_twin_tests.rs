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
