//! **fp8 *training* kernels** (Phase 6) — the E5M2 backward GEMM, per-tensor `amax` reduction, and
//! delayed-scaling quantization that an fp8 training step (Session I's optimizer/AD loop) composes.
//!
//! fp8 training uses **two** fp8 formats (the NVIDIA Transformer-Engine recipe): **E4M3** (4 exp / 3
//! mantissa, more precision) for the *forward* activations and weights, and **E5M2** (5 exp / 2
//! mantissa, more range) for the *backward* gradients (which span a wider dynamic range). The backward
//! GEMMs therefore multiply an **E5M2** gradient by an **E4M3** weight/activation and accumulate in f32
//! — `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32`. E5M2 and E4M3 share the *identical* 8-bit
//! `m16n8k32` per-lane fragment layout (both are 1-byte operands), so the kernel here is the proven
//! [`crate::ptx_fp8`] fragment-reuse GEMM with only the `mma` operand-type tokens swapped — kept in its
//! own file so the load-bearing forward E4M3 path is never touched.
//!
//! **Correctness (first law).** The kernel reads the e5m2/e4m3 *bits* the host uploads and the tensor
//! core decodes them in hardware; the f64 reference decodes the same bits via [`e5m2_to_f32`] /
//! [`crate::ptx_fp8::e4m3_to_f32`] and multiplies in f64. The quantization rounding is thus *baked into
//! both sides identically* — the only residual error is the kernel's f32 accumulation order, so the
//! tolerance is the ordinary `c·√K·ε_f32` GEMM bound (tight, **not** an fp8-slack fudge): E5M2's coarse
//! 2-bit mantissa shows up when the caller quantizes a real f32 gradient, not in this GEMM gate.

use crate::ptx_fp8::{FP8_TM, FP8_TN};

/// OCP **E5M2** (1 sign, 5 exp bias 15, 2 mantissa; max normal 57344) round-to-nearest-even from `f32`,
/// returning the 8 stored bits — the wider-range backward-gradient format. Saturates `|x|` to the max
/// normal (training operands don't carry Inf into the GEMM). The E5M2 twin of
/// [`crate::ptx_fp8::f32_to_e4m3`].
///
/// **Subnormals are encoded, not flushed** (values `m·2⁻¹⁶`, `m ∈ 1..=3`, covering `[2⁻¹⁶, 2⁻¹⁴)`),
/// and `-0.0` keeps its sign bit — because this function is the *host half* of a codec whose device
/// half is Ada's `cvt.rn.satfinite.e5m2x2.f32` ([`quantize_scaled_e5m2_ptx`]), which does both. The
/// old flush-to-zero made the two halves disagree: `f32_to_e5m2(1.5e-5)` returned `0x00` while the
/// device returned `0x01`, so a tile quantized on the host and a tile quantized on the device fed the
/// tensor cores different bits. It also made the fp8 GEMM gates a *circular* oracle w.r.t. this
/// encoder (both sides call it), so nothing could observe the disagreement — see
/// `e5m2_host_matches_device_converter`.
pub fn f32_to_e5m2(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x == 0.0 {
        return sign; // ±0 — the device converter preserves the sign of -0.0
    }
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7f; // exp=31, mant!=0 => NaN
    }
    let a = a.min(57344.0); // saturate to max normal (1.75 * 2^15)
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127; // unbiased f32 exponent
    let mant = bits & 0x7f_ffff;
    let exp = e + 15; // e5m2 biased exponent
    if exp <= 0 {
        // Subnormal range: the value is `m · 2⁻¹⁶` with `m = round_ties_even(|x| · 2¹⁶)`. Round the
        // *explicit* 24-bit significand `1.mant` right by `sh = 22 - exp >= 22` bits, ties to even.
        let sh = (22 - exp) as u32;
        if sh >= 32 {
            return sign; // |x| < 2⁻¹⁷ — rounds to ±0 (also catches f32 subnormal inputs)
        }
        let full = (1u32 << 23) | mant; // < 2^24, so `full + round_bias` cannot overflow u32
        let round_bias = (1u32 << (sh - 1)) - 1 + ((full >> sh) & 1);
        let m = (full + round_bias) >> sh;
        // m == 4 carries out of the subnormal range into the smallest normal (exp field 1, mant 0).
        return if m >= 4 { sign | 0x04 } else { sign | (m as u8) };
    }
    // round the 23-bit mantissa to 2 bits, ties-to-even
    let shift = 23 - 2;
    let round_bias = (1u32 << (shift - 1)) - 1 + ((mant >> shift) & 1);
    let m2 = (mant + round_bias) >> shift;
    let (exp, m2) = if m2 == 4 { (exp + 1, 0) } else { (exp, m2) }; // mantissa carry
    if exp >= 31 {
        return sign | 0x7b; // saturate to max normal (exp=30, mant=3)
    }
    sign | ((exp as u8) << 2) | (m2 as u8)
}

/// Widen E5M2 stored bits back to `f32` (the value the tensor core multiplies) — the reference twin of
/// [`f32_to_e5m2`]. Handles subnormals (exp==0) and Inf/NaN (exp==31) per the OCP spec; the GEMM gates
/// keep data in the normal range so the decode is exact.
pub fn e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 2) & 0x1f) as i32;
    let m = (b & 0x03) as f32;
    if exp == 0 {
        sign * (m / 4.0) * 2f32.powi(-14) // subnormal: 2^(1-15)
    } else if exp == 31 {
        if m == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + m / 4.0) * 2f32.powi(exp - 15)
    }
}

/// Generate a fragment-reuse fp8 GEMM `C = A·Bᵀ` (f32 accumulate) with the `mma` operand types set to
/// `atype`/`btype` (`"e4m3"` or `"e5m2"`). Byte-for-byte the structure of
/// [`crate::ptx_fp8::fp8_gemm_mt_ptx`] — 8-bit `m16n8k32` fragments are layout-identical across the two
/// fp8 formats, so only the type tokens and the entry name differ. A `[M,K]` row-major, B `[N,K]`
/// row-major (= `K×N` col-major for the `.col` operand). Requires M%(16·TM)==0, N%(8·TN)==0, K%32==0.
fn gen_fp8_mt_typed(entry: &str, atype: &str, btype: &str) -> String {
    let (tm, tn) = (FP8_TM, FP8_TN);
    let mut s = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
    s += &format!(".visible .entry {entry}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{{\n");
    s += "    .reg .pred %p;\n";
    s += "    .reg .b32 %M,%N,%K,%lane,%grp,%tg4,%tg2,%row0,%col0,%k,%tmp;\n";
    let mut f32regs = String::from("%z");
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                f32regs += &format!(",%d{mi}_{ni}_{r}");
            }
        }
    }
    s += &format!("    .reg .f32 {f32regs};\n");
    let mut b32regs = String::new();
    for mi in 0..tm {
        for r in 0..4 {
            b32regs += &format!("%a{mi}_{r},");
        }
    }
    for ni in 0..tn {
        for r in 0..2 {
            b32regs += &format!("%b{ni}_{r},");
        }
    }
    s += &format!("    .reg .b32 {}; \n", b32regs.trim_end_matches(','));
    let mut b64regs = String::from("%A,%B,%C,%t,%kk,%cp");
    for mi in 0..tm {
        b64regs += &format!(",%a0p{mi},%a8p{mi}");
    }
    for ni in 0..tn {
        b64regs += &format!(",%bp{ni}");
    }
    s += &format!("    .reg .b64 {b64regs};\n");

    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    s += "    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg4,%lane,3;\n";
    s += "    shl.b32 %tg2,%tg4,1;\n    shl.b32 %tg4,%tg4,2;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %row0,%tmp,{};\n", 16 * tm);
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %col0,%tmp,{};\n", 8 * tn);

    for mi in 0..tm {
        s += &format!("    add.s32 %tmp,%row0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.s32 %tmp,%tmp,%tg4;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %a0p{mi},%A,%t;\n", mi * 16);
        s += &format!("    add.s32 %tmp,%row0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.s32 %tmp,%tmp,%tg4;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %a8p{mi},%A,%t;\n", mi * 16 + 8);
    }
    for ni in 0..tn {
        s += &format!("    add.s32 %tmp,%col0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.s32 %tmp,%tmp,%tg4;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %bp{ni},%B,%t;\n", ni * 8);
    }
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.f32 %d{mi}_{ni}_{r},0f00000000;\n");
            }
        }
    }
    s += "    mov.u32 %k,0;\nKLOOP:\n    setp.ge.u32 %p,%k,%K;\n    @%p bra KEND;\n    cvt.u64.u32 %kk,%k;\n";
    for mi in 0..tm {
        s += &format!("    add.s64 %t,%a0p{mi},%kk;\n    ld.global.b32 %a{mi}_0,[%t];\n    ld.global.b32 %a{mi}_2,[%t+16];\n");
        s += &format!("    add.s64 %t,%a8p{mi},%kk;\n    ld.global.b32 %a{mi}_1,[%t];\n    ld.global.b32 %a{mi}_3,[%t+16];\n");
    }
    for ni in 0..tn {
        s += &format!("    add.s64 %t,%bp{ni},%kk;\n    ld.global.b32 %b{ni}_0,[%t];\n    ld.global.b32 %b{ni}_1,[%t+16];\n");
    }
    for mi in 0..tm {
        for ni in 0..tn {
            s += &format!("    mma.sync.aligned.m16n8k32.row.col.f32.{atype}.{btype}.f32\n        {{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}}, {{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}}, {{%b{ni}_0,%b{ni}_1}}, {{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}};\n");
        }
    }
    s += "    add.u32 %k,%k,32;\n    bra KLOOP;\nKEND:\n";
    for mi in 0..tm {
        for ni in 0..tn {
            s += &format!("    add.s32 %tmp,%row0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%N;\n    add.s32 %tmp,%tmp,%col0;\n    add.s32 %tmp,%tmp,{};\n    add.s32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,2;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %cp,%C,%t;\n", mi * 16, ni * 8);
            s += &format!("    st.global.f32 [%cp],%d{mi}_{ni}_0;\n    st.global.f32 [%cp+4],%d{mi}_{ni}_1;\n");
            s += &format!("    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %cp,%cp,%t;\n    st.global.f32 [%cp],%d{mi}_{ni}_2;\n    st.global.f32 [%cp+4],%d{mi}_{ni}_3;\n");
        }
    }
    s += "    ret;\n}\n";
    s
}

/// **fp8 backward GEMM** `C = A·Bᵀ`, A **E5M2** (gradient), B **E4M3** (weight/activation), f32 out —
/// the `dX = dY·W` / `dW = dYᵀ·X` building block of an fp8 training step (entry `fp8_bwd_gemm_nt`).
/// Requires M%(16·TM)==0, N%(8·TN)==0, K%32==0. See [`gen_fp8_mt_typed`].
pub fn fp8_bwd_gemm_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_fp8_mt_typed("fp8_bwd_gemm_nt", "e5m2", "e4m3")).as_str()
}

/// **fp8 E5M2×E5M2 GEMM** `C = A·Bᵀ`, both operands E5M2, f32 out (entry `fp8_e5m2_gemm_nt`) — the
/// grad·grad-shaped backward contraction (e.g. when both operands are wide-range). See [`gen_fp8_mt_typed`].
pub fn fp8_e5m2_gemm_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_fp8_mt_typed("fp8_e5m2_gemm_nt", "e5m2", "e5m2")).as_str()
}

/// Max representable magnitude of each fp8 format (the delayed-scaling denominator).
pub const E4M3_MAX: f32 = 448.0;
pub const E5M2_MAX: f32 = 57344.0;

/// **Per-tensor `amax` reduction** `out[gid] = max_i |x[i]|` (entry `amax_f32`) — each thread grid-strides
/// its share of `x[N]` tracking a running max-abs and writes one partial to `out[globalThreadId]`; the
/// host takes the final max over the `gridDim·blockDim` partials. **Deterministic** (fixed grid, no
/// atomics — M12): the per-thread partition and the associative-and-commutative `max` give a
/// run-to-run-identical result. `amax` is the calibration statistic for delayed scaling: the scale that
/// maps a tensor's largest magnitude onto the fp8 max so the fp8 range is fully used.
pub const AMAX_PTX: &str = r#".version 8.4
.target sm_89
.address_size 64

.visible .entry amax_f32(
    .param .u32 pN,
    .param .u64 pX,
    .param .u64 pPartials
)
{
    .reg .pred %p;
    .reg .b32 %n,%tix,%bid,%ntx,%nb,%gid,%stride,%i;
    .reg .f32 %m,%v;
    .reg .b64 %X,%P,%off,%ptr;

    ld.param.u32 %n,[pN];
    ld.param.u64 %X,[pX];
    ld.param.u64 %P,[pPartials];
    cvta.to.global.u64 %X,%X;
    cvta.to.global.u64 %P,%P;
    mov.u32 %tix,%tid.x;
    mov.u32 %bid,%ctaid.x;
    mov.u32 %ntx,%ntid.x;
    mov.u32 %nb,%nctaid.x;
    mul.lo.u32 %gid,%bid,%ntx;
    add.u32 %gid,%gid,%tix;            // global thread id
    mul.lo.u32 %stride,%nb,%ntx;       // total thread count
    mov.f32 %m,0f00000000;
    mov.u32 %i,%gid;
AMAX_LOOP:
    setp.ge.u32 %p,%i,%n;
    @%p bra AMAX_END;
    mul.wide.u32 %off,%i,4;
    add.s64 %ptr,%X,%off;
    ld.global.f32 %v,[%ptr];
    abs.f32 %v,%v;
    max.f32 %m,%m,%v;
    add.u32 %i,%i,%stride;
    bra AMAX_LOOP;
AMAX_END:
    mul.wide.u32 %off,%gid,4;
    add.s64 %ptr,%P,%off;
    st.global.f32 [%ptr],%m;
    ret;
}
"#;

/// Delayed-scaling factor `recip = fp8_max / amax`: multiply a tensor by this *before* quantizing so its
/// largest magnitude maps onto the fp8 max (using the full dynamic range), then divide the dequantized
/// result by `recip` downstream. Uses the *previous* step's `amax` (hence "delayed"). Guards a zero /
/// non-finite `amax` (→ `1.0`, an identity scale).
pub fn delayed_scale_recip(amax: f32, fp8_max: f32) -> f32 {
    if amax <= 0.0 || !amax.is_finite() {
        1.0
    } else {
        fp8_max / amax
    }
}

/// Apply delayed scaling and quantize to **E5M2** (the backward-gradient format): `e5m2(x · recip)`.
pub fn quantize_e5m2_scaled(x: f32, recip: f32) -> u8 {
    f32_to_e5m2(x * recip)
}

/// Apply delayed scaling and quantize to **E4M3** (the forward activation/weight format): `e4m3(x · recip)`.
pub fn quantize_e4m3_scaled(x: f32, recip: f32) -> u8 {
    crate::ptx_fp8::f32_to_e4m3(x * recip)
}

/// Generate a **device-resident delayed-scaling quantize** kernel `out[i] = fp8(x[i]·recip)` using Ada's
/// hardware packed-fp8 converter `cvt.rn.satfinite.{fmt}.f32` (`fmt` = `e5m2x2` or `e4m3x2`) — two f32
/// per `cvt` (a→high byte, b→low byte), so each thread handles an even pair and stores a `.b16` (the
/// pair of fp8 bytes). Keeps the whole fp8 quantization on-GPU (no host round-trip) — the residency the
/// fp8 training step needs. Grid-strided over `N/2` pairs; requires N even. Entry `{entry}`.
fn gen_quantize_scaled(entry: &str, fmt: &str) -> String {
    format!(
        r#".version 8.4
.target sm_89
.address_size 64

.visible .entry {entry}(
    .param .u32 pN,
    .param .u64 pX,
    .param .u64 pOut,
    .param .f32 pRecip
)
{{
    .reg .pred %p;
    .reg .b32 %n,%nh,%gid,%stride,%i2,%tix,%bid,%ntx,%nb;
    .reg .f32 %recip,%va,%vb;
    .reg .b16 %packed;
    .reg .b64 %X,%O,%off,%ptr;
    ld.param.u32 %n,[pN];
    ld.param.u64 %X,[pX];
    ld.param.u64 %O,[pOut];
    ld.param.f32 %recip,[pRecip];
    cvta.to.global.u64 %X,%X;
    cvta.to.global.u64 %O,%O;
    mov.u32 %tix,%tid.x;
    mov.u32 %bid,%ctaid.x;
    mov.u32 %ntx,%ntid.x;
    mov.u32 %nb,%nctaid.x;
    mul.lo.u32 %gid,%bid,%ntx;
    add.u32 %gid,%gid,%tix;
    mul.lo.u32 %stride,%nb,%ntx;
    shr.u32 %nh,%n,1;
Q_{entry}:
    setp.ge.u32 %p,%gid,%nh;
    @%p bra QE_{entry};
    mul.lo.u32 %i2,%gid,2;
    mul.wide.u32 %off,%i2,4;
    add.s64 %ptr,%X,%off;
    ld.global.f32 %vb,[%ptr];        // element 2*gid -> low byte of the pair
    ld.global.f32 %va,[%ptr+4];      // element 2*gid+1 -> high byte
    mul.f32 %vb,%vb,%recip;
    mul.f32 %va,%va,%recip;
    cvt.rn.satfinite.{fmt}.f32 %packed,%va,%vb;   // d[15:8]=fp8(va), d[7:0]=fp8(vb)
    cvt.u64.u32 %off,%i2;            // 1 byte/elem => byte offset == element index
    add.s64 %ptr,%O,%off;
    st.global.b16 [%ptr],%packed;    // little-endian: out[2*gid]=fp8(vb), out[2*gid+1]=fp8(va)
    add.u32 %gid,%gid,%stride;
    bra Q_{entry};
QE_{entry}:
    ret;
}}
"#
    )
}

/// Device delayed-scaling quantize to **E5M2** (`out[i] = e5m2(x[i]·recip)`, entry `quantize_scaled_e5m2`).
pub fn quantize_scaled_e5m2_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_quantize_scaled("quantize_scaled_e5m2", "e5m2x2")).as_str()
}

/// Device delayed-scaling quantize to **E4M3** (`out[i] = e4m3(x[i]·recip)`, entry `quantize_scaled_e4m3`).
pub fn quantize_scaled_e4m3_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_quantize_scaled("quantize_scaled_e4m3", "e4m3x2")).as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert one generated module is pure ASCII, naming the offending line *and the offending
    /// character* if not. A single non-ASCII byte anywhere in a PTX string is a `ptxas fatal`; on the
    /// device path it surfaces only as an opaque `DriverError`/`CUDA_ERROR_INVALID_PTX` out of
    /// `cuModuleLoadData`, with nothing pointing at the one character at fault.
    fn assert_ptx_ascii(what: &str, ptx: &str) {
        assert!(!ptx.is_empty(), "{what}: generated an empty module");
        if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
            let bad: Vec<char> = line.chars().filter(|c| !c.is_ascii()).collect();
            panic!(
                "{what}: PTX line {} is not ASCII (ptxas fatal at cuModuleLoadData) -- \
                 offending char(s) {bad:?} in: {line}",
                i + 1
            );
        }
    }

    /// **The crate's PTX-is-ASCII gate for this file** — every other PTX generator family carries one
    /// (`ptx::every_dispatched_ptx_family_is_pure_ascii`, `ptx_conv::every_conv_generator_emits_ascii_ptx`,
    /// `ptx_flash::flash_ptx_is_pure_ascii`, ...); this one had none, so the fp8 *training* kernels were
    /// the single unfenced family in the crate.
    ///
    /// The risk here is not theoretical: this file's prose is written with `×`, `→`, `·`, `⁻¹⁶`, `√`
    /// and `≥` **immediately adjacent** to the `format!`/`writeln!` lines that build the kernels, and
    /// the emitted PTX itself already carries inline `//` comments (`// global thread id`,
    /// `// d[15:8]=fp8(va)`), so a one-keystroke copy of a doc line into an emitted line is the live
    /// failure mode. A GPU-less `cargo test` never loads a module and every fp8 device test *skips*, so
    /// such a regression lands green here and detonates only on a machine with a device — as every fp8
    /// backward GEMM, `amax` and delayed-scaling quantize dying at once.
    ///
    /// Pure-CPU: these functions only build text. Both private generators are swept over their whole
    /// parameter space (all four `mma` operand-type pairings, both packed-converter formats), not just
    /// the configs the public wrappers happen to pin today, so a new wrapper is fenced before it exists.
    #[test]
    fn every_fp8_train_generator_emits_ascii_ptx() {
        // The five modules `gpu.rs` hands to the driver JIT (`Gpu::function`).
        assert_ptx_ascii("fp8_bwd_gemm_ptx", fp8_bwd_gemm_ptx());
        assert_ptx_ascii("fp8_e5m2_gemm_ptx", fp8_e5m2_gemm_ptx());
        assert_ptx_ascii("AMAX_PTX", AMAX_PTX);
        assert_ptx_ascii("quantize_scaled_e5m2_ptx", quantize_scaled_e5m2_ptx());
        assert_ptx_ascii("quantize_scaled_e4m3_ptx", quantize_scaled_e4m3_ptx());

        // `gen_fp8_mt_typed` over every operand-type pairing it can be asked for -- the two shipped
        // wrappers pin (e5m2,e4m3) and (e5m2,e5m2), but the type tokens are interpolated into the
        // `mma` line, so the forward (e4m3,e4m3) and mixed (e4m3,e5m2) spellings must be gated too.
        for atype in ["e4m3", "e5m2"] {
            for btype in ["e4m3", "e5m2"] {
                assert_ptx_ascii(
                    &format!("gen_fp8_mt_typed(a={atype},b={btype})"),
                    &gen_fp8_mt_typed(&format!("fp8_mt_{atype}_{btype}"), atype, btype),
                );
            }
        }

        // `gen_quantize_scaled` over both Ada packed-converter formats. The entry name is interpolated
        // into the branch labels as well as the `.entry`, so it is swept alongside the format.
        for (entry, fmt) in
            [("quantize_scaled_e5m2", "e5m2x2"), ("quantize_scaled_e4m3", "e4m3x2")]
        {
            assert_ptx_ascii(
                &format!("gen_quantize_scaled({entry},{fmt})"),
                &gen_quantize_scaled(entry, fmt),
            );
        }
    }

    /// E5M2 round-trip exactness on representable values + rounding spot-checks (no GPU needed).
    #[test]
    fn e5m2_host_roundtrip() {
        // Exactly representable: 1.0, 1.25, 1.5, 1.75 (mantissa 0..3 at exp 15), powers of two, signs.
        for &v in &[0.0f32, 1.0, -1.0, 1.25, 1.5, 1.75, 2.0, 0.5, 4.0, -3.5, 256.0, -49152.0] {
            let q = e5m2_to_f32(f32_to_e5m2(v));
            assert_eq!(q, v, "E5M2 should represent {v} exactly (got {q})");
        }
        // Max normal saturation.
        assert_eq!(e5m2_to_f32(f32_to_e5m2(1e30)), 57344.0);
        assert_eq!(e5m2_to_f32(f32_to_e5m2(-1e30)), -57344.0);
        // Round-to-nearest-even: 1.3 -> 1.25 (0.05 below) vs 1.375 tie -> even (1.5, mantissa 2).
        assert_eq!(e5m2_to_f32(f32_to_e5m2(1.3)), 1.25);
        assert_eq!(e5m2_to_f32(f32_to_e5m2(1.375)), 1.5); // tie to even mantissa (10b)
        // Wider range than E4M3 (whose max is 448): 1024 is representable in E5M2.
        assert_eq!(e5m2_to_f32(f32_to_e5m2(1024.0)), 1024.0);
    }

    /// **Subnormals and signed zero round-trip** (no GPU). E5M2's subnormals are `m·2⁻¹⁶` for
    /// `m ∈ 1..=3`, i.e. exactly `[2⁻¹⁶, 2⁻¹⁴)`; they used to be flushed to zero, which silently
    /// dropped every gradient element below 2⁻¹⁴ *and* disagreed with the device converter.
    #[test]
    fn e5m2_subnormals_and_signed_zero() {
        let d = 2f32.powi(-16); // the smallest positive E5M2 subnormal
        for (v, bits) in [
            (d, 0x01u8),
            (2.0 * d, 0x02),
            (3.0 * d, 0x03),
            (-d, 0x81),
            (-3.0 * d, 0x83),
            (4.0 * d, 0x04),  // 2^-14: the smallest NORMAL (carry out of the subnormal range)
            (-0.0, 0x80),     // the sign bit survives ±0
            (0.0, 0x00),
        ] {
            assert_eq!(f32_to_e5m2(v), bits, "f32_to_e5m2({v:e}) should be {bits:#04x}");
            // Exactly representable => the decode returns the same magnitude.
            assert_eq!(e5m2_to_f32(bits).abs(), v.abs(), "e5m2_to_f32({bits:#04x})");
        }
        // Round-to-nearest-even inside the subnormal range, including the two ties.
        assert_eq!(f32_to_e5m2(1.4 * d), 0x01, "1.4d rounds down to 1d");
        assert_eq!(f32_to_e5m2(1.5 * d), 0x02, "tie 1.5d -> even (2d)");
        assert_eq!(f32_to_e5m2(2.5 * d), 0x02, "tie 2.5d -> even (2d)");
        assert_eq!(f32_to_e5m2(2.6 * d), 0x03, "2.6d rounds up to 3d");
        // Below half the smallest subnormal everything still flushes (with its sign).
        assert_eq!(f32_to_e5m2(0.4 * d), 0x00);
        assert_eq!(f32_to_e5m2(-0.4 * d), 0x80);
        assert_eq!(f32_to_e5m2(0.5 * d), 0x00, "tie 0.5d -> even (zero)");
        assert_eq!(f32_to_e5m2(f32::MIN_POSITIVE / 4.0), 0x00, "an f32 subnormal input flushes");
    }

    /// **The host encoder and the device converter must produce the SAME BITS** — the two halves of
    /// one codec. `quantize_scaled_e5m2` is Ada's hardware `cvt.rn.satfinite.e5m2x2.f32`; this host
    /// encoder is the f64 oracle every fp8 gate decodes with. Because the GEMM gates call the host
    /// encoder on *both* sides they are circular w.r.t. it, so this byte-for-byte comparison against an
    /// independent implementation (the silicon) is the only thing that can catch an encoder bug — and
    /// it did: the flush-to-zero path disagreed on the whole subnormal range and on `-0.0`.
    #[test]
    fn e5m2_host_matches_device_converter() {
        let d = 2f32.powi(-16);
        let mut xs: Vec<f32> = vec![
            0.0, -0.0, d, -d, 1.4 * d, 1.5 * d, 2.5 * d, 3.0 * d, 4.0 * d, 0.5 * d, -0.4 * d,
            1.0, -1.0, 1.3, 1.375, 0.5, -3.5, 256.0, -49152.0, 57344.0, -57344.0, 1e-5, -1e-5,
            f32::MIN_POSITIVE, 6.1e-5, -6.1e-5,
        ];
        let mut rng = crate::diff::Rng::new(0xE5E5);
        xs.extend(rng.vec(256, -2.0, 2.0));
        xs.extend(rng.vec(256, -1e-4, 1e-4)); // straddles the subnormal boundary
        if xs.len() % 2 == 1 {
            xs.push(0.0); // the packed device converter needs an even count
        }
        let want: Vec<u8> = xs.iter().map(|&v| f32_to_e5m2(v)).collect();
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            crate::diff::skip_or_fail(
                "e5m2_host_matches_device_converter",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        };
        let got = crate::gpu::quantize_scaled_fp8(g, &xs, 1.0, true).unwrap();
        let bad: Vec<String> = got
            .iter()
            .zip(&want)
            .zip(&xs)
            .filter(|((a, b), _)| a != b)
            .map(|((a, b), x)| format!("x={x:e}: device {a:#04x} != host {b:#04x}"))
            .collect();
        assert!(bad.is_empty(), "host/device E5M2 encoders disagree:\n{}", bad.join("\n"));
        eprintln!(
            "[gate] E5M2 host encoder == device cvt.rn.satfinite.e5m2x2.f32 on {} values \
             (subnormals and -0.0 included) ✓",
            xs.len()
        );
    }

    /// Delayed scaling maps a tensor whose max magnitude **exceeds E4M3's range** (1000 > 448) into the
    /// fp8 range, and the dequantize-then-unscale recovers each element within E5M2's 2-bit-mantissa
    /// resolution (≤ ~1/8 relative) — the wide-range gradient path E5M2 exists for.
    #[test]
    fn delayed_scaling_recovers_tensor() {
        let x = [3.0f32, -1000.0, 12.5, 0.0, 700.0, -250.0];
        let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs())); // 1000
        let recip = delayed_scale_recip(amax, E5M2_MAX); // 57344/1000
        for &v in &x {
            let recovered = e5m2_to_f32(quantize_e5m2_scaled(v, recip)) / recip;
            let tol = v.abs() * 0.13 + 1e-6; // E5M2 half-ULP is <= 1/8 relative for normals
            assert!((recovered - v).abs() <= tol, "delayed-scale {v} -> {recovered} (tol {tol})");
        }
        // The scale actually used the upper fp8 range (the largest element maps near E5M2_MAX).
        let big = e5m2_to_f32(quantize_e5m2_scaled(-1000.0, recip)).abs();
        assert!(big > 0.5 * E5M2_MAX, "largest element should use the upper fp8 range, got {big}");
    }
}
