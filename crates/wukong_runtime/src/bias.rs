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
//!     activation ([`crate::vmath::apply1`] scalar / [`crate::vmath::vmath8_for`] vector), so the fused
//!     result is bit-for-bit `wukong_vmath_f32(x + b)`.
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
fn bias_scalar(x: &[f32], b: &[f32], out: &mut [f32], rows: usize, cols: usize, op: i64) {
    for i in 0..rows {
        let row = i * cols;
        for j in 0..cols {
            out[row + j] = bias1(op, x[row + j], b[j]);
        }
    }
}

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
    use std::arch::x86_64::*;
    // The 8-lane activation (or `None` for identity / an unrecognized op → the sum unmodified). Resolved
    // once here and dispatched on **once** (the `match` below), so the hot loop carries no per-block
    // branch — the identity bias-add compiles to gcc's own tight load→add→store loop, and the activation
    // path folds `f` inline. `vmath8_for` is the vector twin of `apply1`, so lanes == the scalar tail.
    let act8 = crate::vmath::vmath8_for(op);
    let nt = use_nt(rows.saturating_mul(cols));
    let addv = |xr: *const f32, j: usize| {
        _mm256_add_ps(_mm256_loadu_ps(xr.add(j)), _mm256_loadu_ps(b.add(j)))
    };
    macro_rules! store {
        ($p:expr, $v:expr) => {
            if nt {
                _mm256_stream_ps($p, $v);
            } else {
                _mm256_storeu_ps($p, $v);
            }
        };
    }
    // The per-row loop, parameterized by `$mk(xr, j)` → the 8-lane result at column `j`. Instantiated
    // once per activation branch (below), so `$mk` is a fixed inlined expression — the ×4 unroll then
    // runs four independent load→add(→act) chains with no branch, hiding the load/add latency the way
    // gcc's unrolled auto-vectorization of this (fully inner-unit-stride) nest does.
    macro_rules! run {
        ($mk:expr) => {
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
                        // `wrapping_add`, not `add`: the target is deliberately past the end of `x`
                        // on the final rows (see `PF_AHEAD`), which `add`'s in-bounds precondition
                        // forbids. A prefetch of an unmapped address is architecturally a no-op.
                        _mm_prefetch(xr.wrapping_add(j + PF_AHEAD) as *const i8, _MM_HINT_T0);
                    }
                    let r0 = $mk(xr, j);
                    let r1 = $mk(xr, j + 8);
                    let r2 = $mk(xr, j + 16);
                    let r3 = $mk(xr, j + 24);
                    store!(outr.add(j), r0);
                    store!(outr.add(j + 8), r1);
                    store!(outr.add(j + 16), r2);
                    store!(outr.add(j + 24), r3);
                    j += 32;
                }
                while j + 8 <= cols {
                    store!(outr.add(j), $mk(xr, j));
                    j += 8;
                }
                // Scalar tail (`cols` not a multiple of 8) — the identical per-element op as the lanes.
                while j < cols {
                    *outr.add(j) = bias1(op, *xr.add(j), *b.add(j));
                    j += 1;
                }
            }
        };
    }
    match act8 {
        None => run!(&addv),
        Some(f) => run!(|xr, j| f(addv(xr, j))),
    }
    if nt {
        _mm_sfence(); // non-temporal stores are weakly ordered; fence before the buffer is read back.
    }
}

/// Total streamed bytes (the `x` read + the `out` write; `b` is cache-resident and negligible) at/above
/// which the store goes non-temporal — once the working set spills L3, `vmovntps` skips the
/// read-for-ownership traffic a cacheable store pays. Keys on the **total** bytes touched, not the
/// element count (the streaming-kernel rule). ~10 MiB ≈ this machine's L3; 2 streams (x, out) of f32.
const NT_MIN_BYTES: usize = 10 * 1024 * 1024;

#[inline]
fn use_nt(n: usize) -> bool {
    n.saturating_mul(2).saturating_mul(4) >= NT_MIN_BYTES
}

/// `out[i*cols + j] = act(x[i*cols + j] + b[j])` for `i in 0..rows`, `j in 0..cols` — the broadcast-bias
/// add (`b` a `cols`-long vector added across every row), with an optional fused activation `act`
/// selected by `op` (a `crate::vmath::VM_*` code, or [`BIAS_ACT_NONE`] for a pure add). Uses the 256-bit
/// AVX2/FMA kernel when available (8 columns/step + a scalar tail), else a scalar fallback. `x` and
/// `out` may alias (in-place bias-add).
///
/// # Safety
/// `x` and `out` must each be valid for `rows*cols` `f32`; `b` for `cols` `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_bias_bcast_f32(
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
    let (rows, cols) = (rows as usize, cols as usize);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for rows*cols / cols by the caller contract.
            unsafe { bias_avx2(x, b, out, rows, cols, op) };
            return;
        }
    }
    let n = rows * cols;
    let xs = std::slice::from_raw_parts(x, n);
    let bs = std::slice::from_raw_parts(b, cols);
    let os = std::slice::from_raw_parts_mut(out, n);
    bias_scalar(xs, bs, os, rows, cols, op);
}

/// Multicore broadcast-bias — the `@parallel` twin of [`wukong_bias_bcast_f32`]. Rows are independent,
/// so disjoint row bands run on separate cores with no cross-row combine; the per-row routine is the
/// serial kernel, so the result is **bit-identical** to serial on any core count (interp == native).
///
/// # Safety
/// `x` and `out` must each be valid for `rows*cols` `f32`; `b` for `cols` `f32`.
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
        // `b` (read-only, shared) valid for `cols`. Bands never overlap, so the writes don't race.
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
    use crate::vmath::{VM_GELU, VM_RELU, VM_SILU};

    /// The AVX2 lanes and the scalar tail/fallback must agree element-for-element — across several
    /// activations, a `cols` that is not a multiple of 8 (forcing a tail), and a size large enough to
    /// cross the non-temporal-store threshold (so the `vmovntps` alignment prologue, the streaming
    /// stores and the trailing `sfence` are really executed — narrowing the 32-byte peel to 16 makes
    /// this test fault, which it did not before the last shape was added).
    #[test]
    fn bias_scalar_matches_avx() {
        // The last shape is the only one that crosses `NT_MIN_BYTES`: `use_nt` needs
        // rows*cols >= 1_310_720 elements, so (700, 1500) = 1_050_000 (8.0 MiB of traffic) still
        // takes the cacheable-store path and (1400, 1501) = 2_101_400 (16.0 MiB) takes the NT one.
        // 1501 is not a multiple of 8 (forcing the scalar tail) and the row stride 1501*4 = 6004
        // bytes is not a multiple of 32, so successive rows start at different mod-32 offsets and
        // the `vmovntps` alignment prologue peels a different amount on each row.
        for &(rows, cols) in &[
            (3usize, 5usize),
            (7, 8),
            (4, 13),
            (2, 64),
            (700, 1500),
            (1400, 1501),
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
                bias_scalar(&x, &b, &mut want, rows, cols, op);
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
