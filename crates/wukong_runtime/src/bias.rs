//! Broadcast-bias add — `out[i*C + j] = act(x[i*C + j] + b[j])` — the pre-activation bias a
//! `[rows, cols]` tensor adds across **every row** (the `+ bias` / `+ position` add every FFN /
//! attention projection ends with). The bias vector `b` is `cols` long and **broadcast down the
//! rows**: row `i` reads `x[i*C .. i*C+C]` and adds the same `b[0..C]`.
//!
//! Why it beats gcc/g++/rustc `-O3 -march=native` on this memory-bound shape:
//!  1. **256-bit width.** The bias operand `b[j]` is indexed by the *inner* loop var alone (a
//!     row-broadcast stride-0-in-`i` read), so Wukong's affine `velem` recognizer — which requires
//!     every operand to be a unit-stride read of the *flat* index — declines it, and the nest would
//!     otherwise fall to Cranelift's 128-bit generic vectorizer or a scalar loop. This kernel folds the
//!     whole row-broadcast into one 256-bit AVX2 streaming pass (8 columns/step).
//!  2. **Fused activation.** The common form is `act(x + bias)` (a Linear's bias-add immediately
//!     followed by ReLU/GELU/SiLU). The activation folds an `exp`/`erf` C/Rust keep scalar, so the
//!     fused 256-bit pass wins like the `vmath` activation dispatch — and reuses the *exact* `vmath`
//!     activation ([`crate::vmath::apply1`] scalar / the 8-lane kernel [`crate::vmath::with_act8`]
//!     inlines into this kernel's loop), so the fused result is bit-for-bit `wukong_vmath_f32(x + b)`.
//!
//! Determinism / twin contract: the AVX2 lanes and the scalar tail/fallback share the exact same
//! per-element sequence (`x + b[j]`, one IEEE add, then the shared activation lane), so every lane of
//! every path agrees bit-for-bit (`bias_scalar_matches_avx` pins it across activations and non-mult-of-8
//! tails). The interpreter marshals its abstract memory through this **identical** kernel, so the
//! differential oracle stays exact. Rows are independent, so `_parallel` maps disjoint row bands across
//! cores with no cross-row combine → serial == parallel bit-for-bit.

use rayon::prelude::*;

/// Sentinel `op` meaning **no activation** (identity — a pure broadcast add). Any other value is a
/// `crate::vmath::VM_*` activation code; an unrecognized one also degenerates to identity (matching
/// [`crate::vmath::apply1`], which returns its input for an unknown op).
pub const BIAS_ACT_NONE: i64 = -1;

/// Software-prefetch distance (elements ahead of the streamed `x` row). A prefetch past the buffer end
/// is a hint the hardware silently drops, so the final rows need no guard — but the *address* is still
/// formed in Rust, and `<*const T>::add` requires the result to stay inside the allocation, so the
/// prefetch site computes it with `wrapping_add` (no such requirement, and it lowers to the same
/// `lea`+`prefetcht0` — verified byte-identical in the release asm of `bias_avx2`).
const PF_AHEAD: usize = 64;

/// One element `act(x + b)`, sharing the *identical* scalar activation the `vmath` kernel and the
/// inlined-poly MIR use — so a fused-activation bias is bit-for-bit consistent with `wukong_vmath_f32`.
#[inline]
fn bias1(op: i64, x: f32, b: f32) -> f32 {
    crate::vmath::apply1(op, x + b)
}

/// Scalar reference / no-AVX2 fallback: `out[i*C+j] = act(x[i*C+j] + b[j])`. The row loop reloads the
/// cache-resident `b`; the per-element op is exactly [`bias1`], so it matches the AVX2 lanes.
unsafe fn bias_scalar(
    x: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    op: i64,
) {
    for i in 0..rows {
        let row = i * cols;
        for j in 0..cols {
            *out.add(row + j) = bias1(op, *x.add(row + j), *b.add(j));
        }
    }
}

/// AVX2 dispatcher: the op is resolved **once**, here, and the row loop [`bias_avx2_loop`] is
/// instantiated with the activation as a generic callable — through [`crate::vmath::with_act8`] for a
/// `VM_*` code (one instantiation per kernel, polynomial inlined), or with the identity for
/// [`BIAS_ACT_NONE`] / an unrecognized op (the sum unmodified, matching `apply1`'s identity). The
/// identity instantiation is gcc's own tight load→add→store loop.
///
/// It used to hand the loop a `vmath8_for` function pointer, and the shipped binary called it once
/// per vector — `vmovaps %ymm0,0x80(%rsp)` / `vzeroupper` / `call *%r15` / reload, on each of the four
/// unrolled chains, with the polynomial's constants rematerialized inside the callee every time
/// (P9.3, disassembled from the release build). The ×4 unroll could not overlap four opaque calls.
///
/// # Safety
/// `x`/`out` valid for `rows*cols` f32, `b` for `cols`; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bias_avx2(
    x: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    op: i64,
) {
    struct Body {
        x: *const f32,
        b: *const f32,
        out: *mut f32,
        rows: usize,
        cols: usize,
        op: i64,
    }
    impl crate::vmath::Act8Body for Body {
        unsafe fn run<F: Fn(std::arch::x86_64::__m256) -> std::arch::x86_64::__m256>(self, f: F) {
            // SAFETY: the dispatcher's contract — buffers valid, AVX2+FMA checked.
            unsafe { bias_avx2_loop(self.x, self.b, self.out, self.rows, self.cols, self.op, f) }
        }
    }
    let body = Body {
        x,
        b,
        out,
        rows,
        cols,
        op,
    };
    if !crate::vmath::with_act8(op, body) {
        bias_avx2_loop(x, b, out, rows, cols, op, |v| v);
    }
}

/// The broadcast-bias row loop for one activation `act` (the identity for a pure add): `out[i, j] =
/// act(x[i, j] + b[j])`, ×4-unrolled — four independent load→add→act chains with no branch, hiding
/// the load/add latency the way gcc's unrolled auto-vectorization of this nest does, and now also
/// four independent *inlined* polynomial chains — with the non-temporal store regime and a scalar
/// tail on the identical per-element op ([`bias1`]).
///
/// # Safety
/// `x`/`out` valid for `rows*cols` f32, `b` for `cols`; AVX2+FMA available; `act` an avx2+fma lane
/// op equal lane-for-lane to `apply1(op, ·)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bias_avx2_loop<F: Fn(std::arch::x86_64::__m256) -> std::arch::x86_64::__m256>(
    x: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    op: i64,
    act: F,
) {
    use std::arch::x86_64::*;
    let nt = use_nt(rows.saturating_mul(cols));
    let addv = |xr: *const f32, j: usize| {
        _mm256_add_ps(_mm256_loadu_ps(xr.add(j)), _mm256_loadu_ps(b.add(j)))
    };
    macro_rules! store {
        ($p:expr, $v:expr) => {
            if nt {
                crate::velem::assert_nt_aligned($p);
                _mm256_stream_ps($p, $v);
            } else {
                _mm256_storeu_ps($p, $v);
            }
        };
    }
    for i in 0..rows {
        let xr = x.add(i * cols);
        let outr = out.add(i * cols);
        let mut j = 0usize;
        // Peel a scalar prologue until this row's `out` is 32-byte aligned (vmovntps faults on a
        // misaligned address); regular stores don't need it.
        if nt {
            while j < cols && (outr.add(j) as usize) & 31 != 0 {
                *outr.add(j) = bias1(op, *xr.add(j), *b.add(j));
                j += 1;
            }
        }
        while j + 32 <= cols {
            if nt {
                // `wrapping_add`, not `add`: the target is deliberately past the end of `x` on the
                // final rows (see `PF_AHEAD`), which `add`'s in-bounds precondition forbids. A
                // prefetch of an unmapped address is architecturally a no-op.
                _mm_prefetch(xr.wrapping_add(j + PF_AHEAD) as *const i8, _MM_HINT_T0);
            }
            let r0 = act(addv(xr, j));
            let r1 = act(addv(xr, j + 8));
            let r2 = act(addv(xr, j + 16));
            let r3 = act(addv(xr, j + 24));
            store!(outr.add(j), r0);
            store!(outr.add(j + 8), r1);
            store!(outr.add(j + 16), r2);
            store!(outr.add(j + 24), r3);
            j += 32;
        }
        while j + 8 <= cols {
            store!(outr.add(j), act(addv(xr, j)));
            j += 8;
        }
        // Scalar tail (`cols` not a multiple of 8) — the identical per-element op as the lanes.
        while j < cols {
            *outr.add(j) = bias1(op, *xr.add(j), *b.add(j));
            j += 1;
        }
    }
    if nt {
        _mm_sfence(); // non-temporal stores are weakly ordered; fence before the buffer is read back.
    }
}

/// The x read and output write total eight bytes per element; bias is cache-resident.
#[inline]
fn use_nt(n: usize) -> bool {
    n.saturating_mul(2).saturating_mul(4) >= crate::nt_policy::nt_min_bytes()
}

/// `out[i*cols + j] = act(x[i*cols + j] + b[j])` for `i in 0..rows`, `j in 0..cols` — the broadcast-bias
/// add (`b` a `cols`-long vector added across every row), with an optional fused activation `act`
/// selected by `op` (a `crate::vmath::VM_*` code, or [`BIAS_ACT_NONE`] for a pure add). Uses the 256-bit
/// AVX2/FMA kernel when available (8 columns/step + a scalar tail), else a scalar fallback. `x` and
/// `out` may be identical (in-place bias-add).
///
/// # Safety
/// `x` and `out` must each be valid for `rows*cols` `f32`; `b` for `cols` `f32`.
/// The `x` and `out` ranges must be identical or disjoint; shifted overlap is unsupported.
/// The `b` range must be disjoint from the written `out` range.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x`; the bias row `b`
/// must not overlap `out`.
#[no_mangle]
pub unsafe extern "C" fn wukong_bias_bcast_f32(
    x: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    op: i64,
) {
    // P14.5: x/out valid for rows*cols, b for cols; `b` must not overlap `out` (out may be x in place).
    crate::safety::debug_extent_2d(x, rows, cols);
    crate::safety::debug_extent_2d(out, rows, cols);
    crate::safety::debug_extent(b, cols);
    crate::safety::debug_disjoint(out, rows.saturating_mul(cols), b, cols);
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows, cols) = (rows as usize, cols as usize);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for rows*cols / cols by the caller contract.
            unsafe { bias_avx2(x, b, out, rows, cols, op) };
            return;
        }
    }
    // Keep raw pointers through the fallback: x == out is an explicitly supported
    // in-place operation and cannot be represented by overlapping Rust slices.
    bias_scalar(x, b, out, rows, cols, op);
}

/// Multicore broadcast-bias — the `@parallel` twin of [`wukong_bias_bcast_f32`]. Rows are independent,
/// so disjoint row bands run on separate cores with no cross-row combine. The bias vector is shared
/// read-only and must not overlap any output band; exact in-place input remains local to each band.
/// The per-row routine is the serial kernel, so the result is **bit-identical** to serial on any core
/// count (interp == native).
///
/// # Safety
/// `x` and `out` must each be valid for `rows*cols` `f32`; `b` for `cols` `f32`.
/// The `x` and `out` ranges must be identical or disjoint; shifted overlap is unsupported.
/// The `b` range must be disjoint from the entire written `out` range, including other row bands.
/// **Aliasing: exactly in place, otherwise disjoint** — `out` may be EXACTLY `x`; the bias row `b`
/// must not overlap `out`.
#[no_mangle]
pub unsafe extern "C" fn wukong_bias_bcast_f32_parallel(
    x: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: i64,
    cols: i64,
    op: i64,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    let (rows_u, cols_u) = (rows as usize, cols as usize);
    // One task per contiguous row band. `RBAND` rows/task keeps the per-task work meaty enough to
    // amortize the dispatch while still filling cores for a tall matrix.
    const RBAND: usize = 8;
    if rows_u <= RBAND {
        return wukong_bias_bcast_f32(x, b, out, rows, cols, op);
    }
    // In a transformer forward the broadcast-bias is one of the first parallel ops, so this can be
    // the process's FIRST rayon touch — configure the global pool before forking, the invariant
    // `ensure_global_pool` states. Forking bare here builds rayon's default 2 MiB-stack registry and
    // makes the runtime's later 16 MiB `build_global` silently lose the race, so outlined
    // `@parallel` region bodies (~1.5 MiB of privatized scratch) end up on 2 MiB stacks. Pool
    // configuration only — the row-band split below is worker-count independent, so bits are
    // unchanged (`bias_parallel_matches_serial` stays exact).
    crate::ensure_global_pool();
    let (xa, ba, oa) = (x as usize, b as usize, out as usize);
    let nbands = rows_u.div_ceil(RBAND);
    (0..nbands).into_par_iter().for_each(|band| {
        let r0 = band * RBAND;
        let r1 = ((band + 1) * RBAND).min(rows_u);
        let off = r0 * cols_u;
        // SAFETY: disjoint row band [r0, r1); each pointer valid for the band's rows by contract, and
        // `b` (read-only, shared) valid for `cols` and disjoint from every output band. The input
        // is identical to the output or disjoint from it, so writes cannot race another band's reads.
        unsafe {
            wukong_bias_bcast_f32(
                (xa as *const f32).add(off),
                ba as *const f32,
                (oa as *mut f32).add(off),
                (r1 - r0) as i64,
                cols,
                op,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn nt_alignment_row_peels() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let cols = 101usize;
        let rows = (crate::nt_policy::test_min_bytes() / (8 * cols) + 1).max(8);
        let n = rows * cols;
        assert!(use_nt(n));
        let x = vec![2.0; n];
        let b = vec![1.0; cols];
        crate::velem::check_nt_output(n, -99.0, 3.0, |out| unsafe {
            wukong_bias_bcast_f32(
                x.as_ptr(),
                b.as_ptr(),
                out,
                rows as i64,
                cols as i64,
                BIAS_ACT_NONE,
            );
        });
    }

    use crate::vmath::{VM_GELU, VM_RELU, VM_SILU};

    #[test]
    fn bias_exact_in_place_matches_scalar_and_parallel() {
        for (rows, cols) in [(1usize, 1usize), (3, 7), (9, 8), (17, 33)] {
            let b: Vec<f32> = (0..cols).map(|j| (j % 5) as f32 / 4.0).collect();
            let x: Vec<f32> = (0..rows * cols)
                .map(|i| (i % 17) as f32 / 4.0 - 2.0)
                .collect();
            for op in [BIAS_ACT_NONE, VM_RELU, VM_GELU, VM_SILU] {
                let want: Vec<u32> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| bias1(op, v, b[i % cols]).to_bits())
                    .collect();
                for path in 0..3 {
                    let mut got = x.clone();
                    let p = got.as_mut_ptr();
                    unsafe {
                        match path {
                            0 => bias_scalar(p, b.as_ptr(), p, rows, cols, op),
                            1 => wukong_bias_bcast_f32(
                                p,
                                b.as_ptr(),
                                p,
                                rows as i64,
                                cols as i64,
                                op,
                            ),
                            _ => wukong_bias_bcast_f32_parallel(
                                p,
                                b.as_ptr(),
                                p,
                                rows as i64,
                                cols as i64,
                                op,
                            ),
                        }
                    }
                    assert_eq!(
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        want,
                        "path={path}, shape={rows}x{cols}, op={op}"
                    );
                }
            }
        }
    }

    /// The AVX2 lanes and the scalar tail/fallback must agree element-for-element — across several
    /// activations, a `cols` that is not a multiple of 8 (forcing a tail), and a size large enough to
    /// cross the non-temporal-store threshold (so the `vmovntps` alignment prologue, the streaming
    /// stores and the trailing `sfence` are really executed — narrowing the 32-byte peel to 16 makes
    /// this test fault, which it did not before the last shape was added).
    #[test]
    fn bias_scalar_matches_avx() {
        // The final shape preserves its original minimum and grows with the shared NT
        // policy. Smaller fixed shapes retain their numerical coverage under any override.
        // 1501 is not a multiple of 8 (forcing the scalar tail) and the row stride 1501*4 = 6004
        // bytes is not a multiple of 32, so successive rows start at different mod-32 offsets and
        // the `vmovntps` alignment prologue peels a different amount on each row.
        for &(rows, cols) in &[
            (3usize, 5usize),
            (7, 8),
            (4, 13),
            (2, 64),
            (700, 1500),
            (
                crate::nt_policy::test_len(8, 1400 * 1501).div_ceil(1501),
                1501,
            ),
        ] {
            let n = rows * cols;
            let x: Vec<f32> = (0..n).map(|i| (i as f32 % 37.0) * 0.1 - 1.8).collect();
            let b: Vec<f32> = (0..cols).map(|j| (j as f32 % 11.0) * 0.25 - 1.3).collect();
            for &op in &[BIAS_ACT_NONE, VM_RELU, VM_GELU, VM_SILU] {
                let mut got = vec![0.0f32; n];
                unsafe {
                    wukong_bias_bcast_f32(
                        x.as_ptr(),
                        b.as_ptr(),
                        got.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        op,
                    );
                }
                let mut want = vec![0.0f32; n];
                unsafe {
                    bias_scalar(x.as_ptr(), b.as_ptr(), want.as_mut_ptr(), rows, cols, op);
                }
                for k in 0..n {
                    assert_eq!(
                        got[k].to_bits(),
                        want[k].to_bits(),
                        "op {op} rows {rows} cols {cols} k {k}"
                    );
                }
            }
        }
    }

    /// Every activation code now runs its OWN monomorphized instantiation of the row loop (the
    /// kernel is inlined through `vmath::with_act8` instead of called through a pointer), so each
    /// instantiation has to be exercised on its own: all `VM_*` codes, the `BIAS_ACT_NONE` identity
    /// and an off-table code, AVX2 lanes vs the scalar twin bit-for-bit, on a row that hits the ×4
    /// body, the 8-wide remainder and the scalar tail (45 = 32 + 8 + 5). Mixed-sign sums put the
    /// domain-restricted ops (log, acosh, atanh, asin, ...) on both sides of their domain edge, where
    /// the lanes must still reproduce the scalar NaN bits.
    #[test]
    fn every_activation_instantiation_matches_the_scalar_twin() {
        use crate::vmath::VM_CBRT;
        let (rows, cols) = (5usize, 45usize);
        let n = rows * cols;
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 - (n / 2) as f32) * 0.027)
            .collect();
        let b: Vec<f32> = (0..cols).map(|j| (j % 9) as f32 * 0.11 - 0.4).collect();
        let ops: Vec<i64> = (0..=VM_CBRT).chain([BIAS_ACT_NONE, VM_CBRT + 1]).collect();
        for &op in &ops {
            let mut got = vec![0.0f32; n];
            let mut want = vec![0.0f32; n];
            unsafe {
                wukong_bias_bcast_f32(
                    x.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    op,
                );
                bias_scalar(x.as_ptr(), b.as_ptr(), want.as_mut_ptr(), rows, cols, op);
            }
            for k in 0..n {
                assert_eq!(
                    got[k].to_bits(),
                    want[k].to_bits(),
                    "op {op} k {k}: {} vs {}",
                    got[k],
                    want[k]
                );
            }
        }
    }

    /// The `_parallel` kernel must be bit-identical to serial on any core count (rows independent, no
    /// cross-row combine) — the thread-count-independence contract the differential gate rests on.
    #[test]
    fn bias_parallel_matches_serial() {
        let (rows, cols) = (523usize, 97usize); // > RBAND, non-mult-of-8 cols → tail + several bands
        let n = rows * cols;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 53.0) * 0.05 - 1.3).collect();
        let b: Vec<f32> = (0..cols).map(|j| (j as f32 % 7.0) * 0.4 - 1.1).collect();
        for &op in &[BIAS_ACT_NONE, VM_RELU, VM_GELU] {
            let mut ser = vec![0.0f32; n];
            let mut par = vec![0.0f32; n];
            unsafe {
                wukong_bias_bcast_f32(
                    x.as_ptr(),
                    b.as_ptr(),
                    ser.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    op,
                );
                wukong_bias_bcast_f32_parallel(
                    x.as_ptr(),
                    b.as_ptr(),
                    par.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    op,
                );
            }
            for k in 0..n {
                assert_eq!(ser[k].to_bits(), par[k].to_bits(), "op {op} k {k}");
            }
        }
    }

    /// The pure broadcast add (`op = BIAS_ACT_NONE`) is one IEEE `f32` add per element, so it must equal
    /// an independent f64 reference **exactly** (a single f32 rounding, no reassociation) — the kernel
    /// is gate-blind, so this pins it against a reference outside the interp==native pair.
    #[test]
    fn bias_add_is_exact_f64_reference() {
        let (rows, cols) = (129usize, 40usize);
        let n = rows * cols;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 91.0) * 0.031 - 1.4).collect();
        let b: Vec<f32> = (0..cols).map(|j| (j as f32 % 17.0) * 0.13 - 1.05).collect();
        let mut got = vec![0.0f32; n];
        unsafe {
            wukong_bias_bcast_f32(
                x.as_ptr(),
                b.as_ptr(),
                got.as_mut_ptr(),
                rows as i64,
                cols as i64,
                BIAS_ACT_NONE,
            );
        }
        for i in 0..rows {
            for j in 0..cols {
                let want = (x[i * cols + j] as f64 + b[j] as f64) as f32; // single f32 round
                assert_eq!(got[i * cols + j].to_bits(), want.to_bits(), "i {i} j {j}");
            }
        }
    }
}

#[cfg(test)]
mod relu_zero_contract_tests {
    use super::*;
    use crate::vmath::VM_RELU;
    // Input bits and independently specified MAXPS(x, +0) result bits.
    const CASES: [(u32, u32); 12] = [
        (0x0000_0000, 0),
        (0x8000_0000, 0),
        (0x7fc1_2345, 0),
        (0xffc1_2345, 0),
        (0x7f80_0001, 0),
        (0xff80_0001, 0),
        (0x7f80_0000, 0x7f80_0000),
        (0xff80_0000, 0),
        (0x0000_0001, 0x0000_0001),
        (0x8000_0001, 0),
        (0x3fc0_0000, 0x3fc0_0000),
        (0xbfc0_0000, 0),
    ];

    #[test]
    fn relu_zero_contract_bias_scalar_simd_and_public() {
        // Seventeen rows force multiple public parallel bands; seventeen columns exercise
        // two SIMD vectors and a scalar tail. -0 + -0 preserves -0 before activation.
        let (rows, cols) = (17usize, 17usize);
        let bias = vec![-0.0f32; cols];
        for offset in 0..CASES.len() {
            let x: Vec<_> = (0..rows * cols)
                .map(|i| f32::from_bits(CASES[(i + offset) % CASES.len()].0))
                .collect();
            let expected: Vec<_> = (0..rows * cols)
                .map(|i| CASES[(i + offset) % CASES.len()].1)
                .collect();
            let mut out = vec![-99.0; rows * cols];
            unsafe {
                bias_scalar(
                    x.as_ptr(),
                    bias.as_ptr(),
                    out.as_mut_ptr(),
                    rows,
                    cols,
                    VM_RELU,
                );
            }
            assert_eq!(
                out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
            );
            for parallel in [false, true] {
                out.fill(-99.0);
                let kernel = if parallel {
                    wukong_bias_bcast_f32_parallel
                } else {
                    wukong_bias_bcast_f32
                };
                unsafe {
                    kernel(
                        x.as_ptr(),
                        bias.as_ptr(),
                        out.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        VM_RELU,
                    );
                }
                assert_eq!(
                    out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                );
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                out.fill(-99.0);
                unsafe {
                    bias_avx2(
                        x.as_ptr(),
                        bias.as_ptr(),
                        out.as_mut_ptr(),
                        rows,
                        cols,
                        VM_RELU,
                    );
                }
                assert_eq!(
                    out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
}
