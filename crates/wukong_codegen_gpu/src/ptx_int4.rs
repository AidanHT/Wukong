//! **W4A16 weight-only int4 decode** — the LLM-inference workhorse, and the one GPU metric where the
//! library field is *immature* (there is no robust general cuBLAS/cuBLASLt int4-decode GEMM the way
//! there is for fp16/int8), so Wukong can post a *documented lead* rather than chase a gold standard.
//!
//! The weights are 4-bit (group-wise quantized along K, with a per-group fp16 scale and an optional
//! integer zero-point — GPTQ/AWQ style); the activations stay fp16. The kernel reads the **packed
//! int4 weight from global** (8 weights per 32-bit word — a **4× smaller weight footprint** than fp16,
//! the bandwidth win that makes decode memory-bound-friendly), **unpacks int4 → fp16 on the fly inside
//! the K-loop** (one `lop3` per interleaved nibble *pair* → `sub.rn.f16x2` the zero-offset →
//! `mul.rn.f16x2` the group scale, two weights per op), stages the dequantized fp16 tile
//! into shared memory, and then runs the **identical fp16 tensor-core MMA** as the dense `wmma`
//! path (`wmma.mma.sync.m16n16k16`, f32 accumulate). So the only deviation from an *exact* dequant of
//! the weight is the same f32-accumulation-order tolerance the fp16 GEMM already carries (~2e-3) — a
//! kernel that dequantizes wrong is a miscompile, gated against an exact f64 dequant reference.
//!
//! Layout (mirrors the `nn.Linear` contract `C = A·Wᵀ`):
//! * **A** activations `[M,K]` row-major fp16.
//! * **W** weights `[N,K]` row-major, quantized group-wise along K (group `G`, default 128). The packed
//!   form is `[N, K/8]` `u32` (8 unsigned 4-bit nibbles per word, **Marlin/AWQ-interleaved** —
//!   [`nibble_pos`] — so the kernel extracts a *pair* with one `lop3` straight into an `f16x2`), plus
//!   per-group fp16 **scales** `[N, K/G]` (and, for the asymmetric path, integer **zero-points**
//!   `[N, K/G]`). Symmetric weights are stored **offset-binary** (`u = q+8`) so both paths dequant
//!   uniformly as `(u - Z)·scale` with `Z = 8` (symmetric) or `Z = zero[group]` (asymmetric).
//! * **C** output `[M,N]` row-major f32.
//!
//! The dequant a single weight goes through — `w = (u - Z)·scale` in fp16 — is reproduced *bit-for-bit*
//! on the host by [`dequant_weight`] (so the gate's reference weight equals the kernel's reconstructed
//! weight exactly; every f16x2 intermediate `1024+u`, `1024+Z`, and their difference is exact in fp16);
//! the host [`quantize_weight_symmetric`] / [`quantize_weight_asymmetric`] produce the packed layout the
//! kernel consumes.
//!
//! **Target floor: `sm_80`** ([`crate::ptx_target::HDR_SM80`], which *is* this file's existing
//! `.version 7.8` at the Ampere floor). Nothing here is Ada-only: the unpack is `lop3.b32` (`sm_50`+)
//! plus `sub/mul.rn.f16x2` (`sm_53`+), and the math is `wmma.{load,mma,store}...m16n16k16.f32`
//! (`sm_70`+) with plain `ld/st`-staged SMEM (this family uses no `cp.async` at all). The former
//! `sm_89` tag was the development card's own arch, and it made the module fail to load on every
//! Ampere part for no gain.

use std::sync::OnceLock;

use half::f16;

use crate::ptx_target::HDR_SM80;

/// Default group size along K (per-group scale/zero-point granularity). 128 is the GPTQ/AWQ default;
/// it is a multiple of the 16-wide WMMA K-step, so a `wmma` K-tile never straddles two groups (one
/// scale load serves the whole 16-strip).
pub const GROUP_SIZE: usize = 128;

/// A group-wise int4-quantized weight matrix `[N,K]` in the exact layout the W4A16 kernel consumes:
/// packed 4-bit values (8 per `u32` word, `[N, K/8]`), per-group fp16 `scales` (`[N, K/G]`), and an
/// optional per-group integer `zeros` (`[N, K/G]`, the asymmetric/AWQ path). `signed` records whether
/// the nibbles are two's-complement `[-7,7]` (symmetric) or unsigned `[0,15]` offset by `zeros`.
#[derive(Clone, Debug)]
pub struct QuantWeight {
    pub n: usize,
    pub k: usize,
    pub group: usize,
    /// Packed unsigned nibbles, `[N, K/8]` row-major, in [`nibble_pos`] interleaved order (weight
    /// `(r, kk)` is at bit `nibble_pos(kk%8)` of word `[r, kk/8]`); read with [`unpack_nibble`].
    pub packed: Vec<u32>,
    /// Per-group fp16 scale, `[N, K/G]` row-major.
    pub scales: Vec<f16>,
    /// Per-group integer zero-point, `[N, K/G]` — `Some` for the asymmetric path (`Z = zero[group]`).
    /// **This is the discriminator**: every launcher (`crate::gpu::gemm_nt_w4a16` and friends) and the
    /// host oracle [`dequant_weight`] pick symmetric vs asymmetric by matching on this field alone.
    pub zeros: Option<Vec<u8>>,
    /// `true` ⇒ symmetric, offset-binary nibbles (`u = q+8`, dequant `Z = 8`); `false` ⇒ asymmetric,
    /// unsigned nibbles with per-group `zeros` (`Z = zero[group]`). Both quantizers keep this equal to
    /// `zeros.is_none()`; it is a **label, not a discriminator** — neither the kernel selection nor the
    /// host oracle reads it (both match on `zeros`, see [`dequant_weight`]), so an inconsistent pair
    /// cannot make the two compute different shapes.
    pub signed: bool,
}

impl QuantWeight {
    /// Groups per row, `K/G`.
    pub fn groups(&self) -> usize {
        self.k / self.group
    }
    /// `u32` words per row, `K/8`.
    pub fn words(&self) -> usize {
        self.k / 8
    }
    /// Packed-weight bytes actually moved from HBM per full pass (`N·K/2`) — the figure whose 4×
    /// shrink vs the fp16 weight (`N·K·2`) is the decode bandwidth win.
    pub fn packed_bytes(&self) -> usize {
        self.packed.len() * 4
    }
}

/// Bit position of weight `j ∈ [0,8)` within its packed `u32` word, in the **Marlin/AWQ interleaved**
/// order: even weights `2jj` go to bits `[4·jj, 4·jj+4)` (the low 16 bits) and odd weights `2jj+1` to
/// bits `[16+4·jj, 16+4·jj+4)` (the high 16 bits). This is what lets the kernel extract a *pair* with a
/// single `lop3` (`(word>>4jj) & 0x000F000F | 0x64006400`) straight into an `f16x2` — no per-element
/// shift/convert. `nibble_pos(2jj)=4jj`, `nibble_pos(2jj+1)=16+4jj`.
#[inline]
pub fn nibble_pos(j: usize) -> u32 {
    ((j / 2) * 4 + (j % 2) * 16) as u32
}

/// Read the raw unsigned nibble (`[0,15]`) of weight `kk` from its packed row `word_row` (`[K/8]` u32),
/// honoring the interleaved [`nibble_pos`] order — the host counterpart to the kernel's `lop3` extract.
#[inline]
pub fn unpack_nibble(word_row: &[u32], kk: usize) -> u32 {
    (word_row[kk / 8] >> nibble_pos(kk % 8)) & 0xF
}

/// Pack an **unsigned** nibble `u ∈ [0,15]` for weight `kk` into `out` at its interleaved [`nibble_pos`].
/// Symmetric weights are stored offset-binary (`u = q+8`); asymmetric ones store `q` directly — both
/// dequant uniformly as `(u - Z)·scale` in the kernel (`Z = 8` symmetric, `Z = zero[group]` asymmetric).
#[inline]
fn pack_nibble(out: &mut u32, kk: usize, u: i32) {
    *out |= ((u & 0xF) as u32) << nibble_pos(kk % 8);
}

/// **Symmetric** group-wise int4 quantization of an `[N,K]` fp16-range weight (no zero-point): per
/// group, `scale = max|w| / 7` and `q = round(w/scale)` clamped to `[-7,7]`, stored **offset-binary**
/// as the unsigned nibble `u = q + 8 ∈ [1,15]` (so the kernel's fast `lop3` unpack — which yields an
/// unsigned `[0,15]` — dequants uniformly as `(u - 8)·scale`). The scale is stored in fp16 and the
/// quantization is done against that *stored* fp16 scale, so the dequant the kernel reconstructs is the
/// best fp16 approximation of `w`. A zero-amax group gets `scale = 1`. `K` must be a multiple of
/// `group`, and `group` a multiple of 8.
pub fn quantize_weight_symmetric(w: &[f32], n: usize, k: usize, group: usize) -> QuantWeight {
    assert_eq!(w.len(), n * k, "weight must be N*K");
    assert!(k % group == 0, "K={k} must be a multiple of group={group}");
    assert!(
        group % 8 == 0,
        "group={group} must be a multiple of 8 (nibble packing)"
    );
    let kg = k / group;
    let kw = k / 8;
    let mut packed = vec![0u32; n * kw];
    let mut scales = vec![f16::ZERO; n * kg];
    for r in 0..n {
        for g in 0..kg {
            let base = r * k + g * group;
            let mut amax = 0.0f32;
            for i in 0..group {
                amax = amax.max(w[base + i].abs());
            }
            let scale16 = if amax > 0.0 {
                f16::from_f32(amax / 7.0)
            } else {
                f16::ONE
            };
            scales[r * kg + g] = scale16;
            let inv = 1.0f32 / scale16.to_f32();
            for i in 0..group {
                let kk = g * group + i;
                let q = (w[base + i] * inv).round().clamp(-7.0, 7.0) as i32;
                pack_nibble(&mut packed[r * kw + kk / 8], kk, q + 8); // offset-binary u = q+8 ∈ [1,15]
            }
        }
    }
    QuantWeight {
        n,
        k,
        group,
        packed,
        scales,
        zeros: None,
        signed: true,
    }
}

/// **Asymmetric** (AWQ/GPTQ-style) group-wise int4 quantization: unsigned nibbles `q ∈ [0,15]` with a
/// per-group fp16 `scale` *and* an integer `zero ∈ [0,15]` such that `w ≈ (q - zero)·scale`. Per group
/// the value range is **extended to include 0** (`qlo = min(min,0)`, `qhi = max(max,0)`) so the
/// zero-point — the `q` that maps to `w=0`, `zero = round(-qlo/scale)` — is always representable in
/// `[0,15]` (the standard affine-quant convention; otherwise an all-positive group would push the grid
/// off one end). `scale = (qhi-qlo)/15`, `q = round(w/scale)+zero` clamped to `[0,15]`. The kernel's
/// `(q - zero)` subtract is one extra instruction on the unpack path.
pub fn quantize_weight_asymmetric(w: &[f32], n: usize, k: usize, group: usize) -> QuantWeight {
    assert_eq!(w.len(), n * k, "weight must be N*K");
    assert!(k % group == 0, "K={k} must be a multiple of group={group}");
    assert!(
        group % 8 == 0,
        "group={group} must be a multiple of 8 (nibble packing)"
    );
    let kg = k / group;
    let kw = k / 8;
    let mut packed = vec![0u32; n * kw];
    let mut scales = vec![f16::ZERO; n * kg];
    let mut zeros = vec![0u8; n * kg];
    for r in 0..n {
        for g in 0..kg {
            let base = r * k + g * group;
            let (mut lo, mut hi) = (0.0f32, 0.0f32); // seed with 0 so the grid spans 0 (zero-point in range)
            for i in 0..group {
                lo = lo.min(w[base + i]);
                hi = hi.max(w[base + i]);
            }
            let scale16 = if hi > lo {
                f16::from_f32((hi - lo) / 15.0)
            } else {
                f16::ONE
            };
            let s = scale16.to_f32();
            let zero = (-lo / s).round().clamp(0.0, 15.0) as i32;
            scales[r * kg + g] = scale16;
            zeros[r * kg + g] = zero as u8;
            let inv = 1.0f32 / s;
            for i in 0..group {
                let kk = g * group + i;
                let q = ((w[base + i] * inv).round() as i32 + zero).clamp(0, 15);
                pack_nibble(&mut packed[r * kw + kk / 8], kk, q);
            }
        }
    }
    QuantWeight {
        n,
        k,
        group,
        packed,
        scales,
        zeros: Some(zeros),
        signed: false,
    }
}

/// Reconstruct the fp16 weight `[N,K]` **exactly as the kernel does** — read the unsigned interleaved
/// nibble `u`, subtract the unified zero-offset `Z` (`8` for the symmetric/offset-binary path,
/// `zero[group]` for the asymmetric path), and scale: `w = (u - Z)·scale` in fp16. `f16::from_f32(v as
/// f32)` is exact for the small integer `v = u-Z`, and the half-crate `f16*f16` is the correctly-rounded
/// product (matching PTX `mul.rn.f16x2`), so the gate's reference weight equals the kernel's bit-for-bit
/// and the only kernel error is the f32 accumulation order.
///
/// **Which path is taken is decided by [`QuantWeight::zeros`], never by `signed`** — that is the field
/// every launcher dispatches on (`crate::gpu::gemm_nt_w4a16` matches on `qw.zeros` to pick
/// `gemm_nt_w4a16` vs `gemm_nt_w4a16_z`; the split-K and autotune paths assert `zeros.is_none()`).
/// Since this function *is* the oracle the W4A16 first-law gate compares the kernel against, reading a
/// second, independent discriminator here would let the oracle and the kernel compute different shapes
/// for any `QuantWeight` built by hand rather than by the two quantizers.
pub fn dequant_weight(qw: &QuantWeight) -> Vec<f16> {
    let (n, k, group) = (qw.n, qw.k, qw.group);
    let (kg, kw) = (qw.groups(), qw.words());
    let zeros = qw.zeros.as_deref();
    let mut w = vec![f16::ZERO; n * k];
    for r in 0..n {
        let row = &qw.packed[r * kw..r * kw + kw];
        for kk in 0..k {
            let u = unpack_nibble(row, kk) as i32;
            let z = match zeros {
                None => 8, // symmetric / offset-binary: u = q+8
                Some(z) => z[r * kg + kk / group] as i32,
            };
            let scale = qw.scales[r * kg + kk / group];
            w[r * k + kk] = f16::from_f32((u - z) as f32) * scale;
        }
    }
    w
}

/// f64 reference `C = A·dequant(W)ᵀ` with `A` pre-rounded to fp16 — the *exact* arithmetic the W4A16
/// kernel performs (fp16 activations × fp16-dequantized weights, f32 accumulate), so the only deviation
/// is the tensor-core f32 accumulation order. `A` is `[M,K]` f32 (rounded to f16 here, the price the
/// fp16 path pays); the gate asserts the kernel matches this within the fp16-accumulate tolerance.
pub fn reference_w4a16(a: &[f32], qw: &QuantWeight, m: usize) -> Vec<f32> {
    let (k, n) = (qw.k, qw.n);
    assert_eq!(a.len(), m * k, "A must be M*K");
    let w = dequant_weight(qw);
    let af: Vec<f64> = a
        .iter()
        .map(|&x| f16::from_f32(x).to_f32() as f64)
        .collect();
    let wf: Vec<f64> = w.iter().map(|x| x.to_f32() as f64).collect();
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += af[i * k + kk] * wf[j * k + kk];
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

// ---- W4A16 PTX generator -------------------------------------------------------------------------
// Mirrors `ptx_wmma::entry_smem` (the proven SMEM-staged fp16 tensor-core tile): a CTA of
// warps_m×warps_n warps cooperatively stages an SM_BM×16 A tile and an SM_BN×16 *dequantized* B tile
// into shared memory each K-step, then every warp computes its tm×tn grid of 16×16 WMMA tiles out of
// shared memory. The ONE departure from the fp16 path is the B staging: instead of a 128-bit fp16
// copy, each thread loads one packed int4 `u32` (8 weights) from global, unpacks it as four f16x2
// pairs (`lop3` → `sub.rn.f16x2` the zero-offset → `mul.rn.f16x2` the group scale), and stores 8 fp16
// into the SMEM B tile with one `st.shared.v4` in the identical row-major
// `[bn,16]` layout `wmma.load.b.col` expects — so the global B traffic is 4-bit while the MMA is the
// byte-identical fp16 tile. Requires M%bm==0, N%bn==0, K%group==0 (group a multiple of 16).

/// SMEM B-tile K-width (the WMMA K-step). One scale serves the whole strip because group ≥ this.
const BK: usize = 16;

/// Generate a W4A16 SMEM-staged WMMA GEMM entry `C = A·dequant(W)ᵀ`. `group` is baked in (static-shape
/// specialization: the scale/word strides become constant shifts). `zero_point` selects the asymmetric
/// `(q-z)` unpack (extra `pZeros` param) over the symmetric signed path. `bm`,`bn` are 16-multiples;
/// the A staging requires `bm·16` to be a whole multiple of `threads·8` (128-bit f16 loads) and the B
/// staging requires `bn·16/8 = bn·2` to be a whole multiple of `threads` (one packed word per thread).
#[allow(clippy::too_many_arguments)]
fn entry_w4a16(
    name: &str,
    bm: usize,
    bn: usize,
    warps_m: usize,
    warps_n: usize,
    group: usize,
    zero_point: bool,
    static_dims: Option<(usize, usize, usize)>,
    splitk: bool,
) -> String {
    assert!(
        group.is_power_of_two() && group % BK == 0,
        "group must be a power of two ≥ {BK}"
    );
    assert!(
        !(splitk && static_dims.is_some()),
        "{name}: split-K uses runtime M·N for the plane offset (dynamic dims)"
    );
    if let Some((m, n, k)) = static_dims {
        assert!(
            m % bm == 0 && n % bn == 0 && k % group == 0,
            "static dims must tile the kernel"
        );
    }
    let nab = 8; // f16 WMMA a/b fragment is 8×.b32
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m);
    let tn = bn / (16 * warps_n);
    let smem_a = bm * BK * 2; // bytes
    let smem_b = bn * BK * 2;
    let a_chunks = bm * BK / (threads * 8); // 128-bit (8×f16) A chunks per thread
    let b_words = bn * BK / 8; // packed int4 words in the B tile
    let b_chunks = b_words / threads; // one packed word per thread per chunk
    assert!(
        a_chunks * threads * 8 == bm * BK,
        "A staging must tile evenly"
    );
    assert!(b_chunks * threads == b_words, "B staging must tile evenly");
    let wn_shift = warps_n.trailing_zeros();
    let wm = (16 * tm) as i64;
    let wn = (16 * tn) as i64;
    let kw_shift = 3u32; // K/8, kt/8 (8 weights per word)
    let kg_shift = group.trailing_zeros(); // K/group, kt/group

    let veclist = |prefix: &str, n: usize| -> String {
        let regs: Vec<String> = (0..n).map(|i| format!("%{prefix}{i}")).collect();
        format!("{{{}}}", regs.join(","))
    };

    let zeros_param = if zero_point {
        ",\n    .param .u64 pZeros"
    } else {
        ""
    };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    \
         .param .u64 pA,\n    .param .u64 pBq,\n    .param .u64 pScales,\n    .param .u64 pC{zeros_param}\n)\n{{\n"
    );
    s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
    s += &format!("    .shared .align 16 .b8 smemB_{name}[{smem_b}];\n");
    s += "    .reg .pred %p0;\n";
    // %tix is the linear thread id (NOT %tid — that is the threadIdx special register; a user reg
    // named %tid makes the assembler read %tid.x as a video selector and reject it).
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%ldm,%v0,%v1,%v2,%v3;\n";
    // int4 B-staging scratch. %pk0..%pk3 hold the 4 dequantized f16x2 pairs (8 weights) for one
    // st.shared.v4; %scx2 / %zsub are the broadcast scale and zero-offset subtrahend for the f16x2
    // dequant; %wsh is the shifted word feeding each pair's lop3. NB: not %p* — %p0 is the predicate
    // register, and reusing the name makes ptxas read it as a pred.
    s += "    .reg .b32 %nrow,%half,%nn,%sidx,%widx,%word,%wsh,%sbase,%kw,%ktw,%kgr,%ktg,%pk0,%pk1,%pk2,%pk3,%scx2,%zsub;\n";
    s += "    .reg .b16 %sc;\n";
    if splitk {
        s += "    .reg .b32 %kbeg,%kend,%kslice;\n";
    }
    if zero_point {
        s += "    .reg .b32 %zv,%ztmp;\n    .reg .b64 %Zeros,%zptr;\n";
    }
    // accumulator + a/b fragments.
    let mut decl_c = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                decl_c += &format!("%c{ti}_{tj}_{r},");
            }
        }
    }
    s += &format!("    .reg .f32 {};\n", decl_c.trim_end_matches(','));
    let mut decl_ab = String::new();
    for ti in 0..tm {
        for r in 0..nab {
            decl_ab += &format!("%a{ti}_{r},");
        }
    }
    for tj in 0..tn {
        for r in 0..nab {
            decl_ab += &format!("%b{tj}_{r},");
        }
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    s += "    .reg .b64 %A,%Bq,%Scl,%C,%off,%gp,%gptr,%cptr,%sptr,%wptr;\n";

    // Static-shape specialization (Wukong's compile-time-shapes lever): bake M/N/K as constants so
    // ptxas constant-folds and strength-reduces the hot-loop strides — every `mul.lo.s32 ...,%K` /
    // `...,%N` and the K/8, K/group shifts become shifts/constants (e.g. ×4096 → <<12). Dynamic loads
    // the dims from params. The signature is identical either way (the launcher is unchanged).
    match static_dims {
        Some((m, n, k)) => {
            s += &format!("    mov.u32 %M,{m};\n    mov.u32 %N,{n};\n    mov.u32 %K,{k};\n");
        }
        None => {
            s +=
                "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
        }
    }
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %Bq,[pBq];\n    ld.param.u64 %Scl,[pScales];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %Bq,%Bq;\n    cvta.to.global.u64 %Scl,%Scl;\n    cvta.to.global.u64 %C,%C;\n";
    if zero_point {
        s += "    ld.param.u64 %Zeros,[pZeros];\n    cvta.to.global.u64 %Zeros,%Zeros;\n";
    }
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    if splitk {
        // K-split across gridDim.z: this CTA owns K-range [kbeg,kend) and writes its partial f32 tile to
        // its OWN M×N plane of the partials buffer (C rebased by ctaid.z·M·N) — no overlap, no atomics ⇒
        // deterministic. kslice = K/gridDim.z (host guarantees K%(sk·group)==0 ⇒ each split is whole
        // groups, so the per-group scale indexing kt/group stays correct). A separate fixed-order
        // reduction kernel (`w4a16_splitk_reduce`) then sums the gridDim.z planes into the final C.
        s += "    mov.u32 %tmp,%nctaid.z;\n    div.u32 %kslice,%K,%tmp;\n";
        s += "    mov.u32 %tmp,%ctaid.z;\n    mul.lo.s32 %kbeg,%tmp,%kslice;\n    add.u32 %kend,%kbeg,%kslice;\n";
        s += "    mul.lo.s32 %tmp2,%M,%N;\n    mul.lo.s32 %tmp2,%tmp2,%tmp;\n    mul.wide.u32 %off,%tmp2,4;\n    add.s64 %C,%C,%off;\n";
    }
    s += "    mov.u32 %ldm,16;\n";
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n";
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n");
    s += &format!("    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    // Constant-stride helpers (static-shape specialization: word/group strides are shifts of K).
    s += &format!("    shr.u32 %kw,%K,{kw_shift};\n    shr.u32 %kgr,%K,{kg_shift};\n");
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                s += &format!("    mov.f32 %c{ti}_{tj}_{r},0f00000000;\n");
            }
        }
    }

    let (kstart, kstop) = if splitk {
        ("%kbeg", "%kend")
    } else {
        ("0", "%K")
    };
    s += &format!("    mov.u32 %kt,{kstart};\n");
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,{kstop};\n    @%p0 bra KEND_{name};\n");

    // --- Stage A: bm×16 f16 from global A[M,K], 128-bit (8×f16) chunks (identical to the fp16 path). ---
    for li in 0..a_chunks {
        if li == 0 {
            s += "    mov.u32 %e,%tix;\n";
        } else {
            s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
        }
        s += "    shr.u32 %r,%e,1;\n    and.b32 %c,%e,1;\n    shl.b32 %c,%c,3;\n";
        s += "    add.u32 %tmp,%baseRow,%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kt;\n    add.u32 %tmp,%tmp,%c;\n";
        s += "    mul.wide.u32 %off,%tmp,2;\n    add.s64 %gptr,%A,%off;\n";
        s += "    ld.global.v4.u32 {%v0,%v1,%v2,%v3},[%gptr];\n";
        s += &format!("    mov.u32 %tmp,smemA_{name};\n    shl.b32 %tmp2,%e,4;\n    add.u32 %tmp,%tmp,%tmp2;\n");
        s += "    st.shared.v4.u32 [%tmp],{%v0,%v1,%v2,%v3};\n";
    }

    // --- Stage B: bn×16 dequantized f16, unpacked from packed int4 Bq[N,K/8] + scales[N,K/G]. ---
    s += &format!("    shr.u32 %ktw,%kt,{kw_shift};\n    shr.u32 %ktg,%kt,{kg_shift};\n");
    for li in 0..b_chunks {
        if li == 0 {
            s += "    mov.u32 %e,%tix;\n";
        } else {
            s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
        }
        // n = e/2 (2 words per 16-wide row), half = e&1 (which 8-wide K sub-strip); nn = baseCol+n.
        s += "    shr.u32 %nrow,%e,1;\n    and.b32 %half,%e,1;\n    add.u32 %nn,%baseCol,%nrow;\n";
        // scale S[nn*(K/G) + kt/G]  (one fp16 per (row,group); the 16-strip is within one group).
        s += "    mul.lo.s32 %sidx,%nn,%kgr;\n    add.u32 %sidx,%sidx,%ktg;\n";
        s += "    mul.wide.u32 %off,%sidx,2;\n    add.s64 %sptr,%Scl,%off;\n    ld.global.b16 %sc,[%sptr];\n";
        if zero_point {
            // zero-point Z[nn*(K/G) + kt/G] (u8, 1 byte/elem ⇒ byte offset == sidx; widen to 64-bit
            // for the pointer add), loaded zero-extended into %zv for the integer subtract.
            s += "    cvt.u64.u32 %off,%sidx;\n    add.s64 %zptr,%Zeros,%off;\n    ld.global.u8 %zv,[%zptr];\n";
        }
        // packed word Bq[nn*(K/8) + kt/8 + half]  (8 interleaved nibbles).
        s += "    mul.lo.s32 %widx,%nn,%kw;\n    add.u32 %widx,%widx,%ktw;\n    add.u32 %widx,%widx,%half;\n";
        s += "    mul.wide.u32 %off,%widx,4;\n    add.s64 %wptr,%Bq,%off;\n    ld.global.u32 %word,[%wptr];\n";
        // Broadcast the f16 scale to both f16x2 lanes; form the unified zero-offset subtrahend (0x6400|Z
        // duplicated): symmetric Z=8 ⇒ const 0x6408 = fp16(1032); asymmetric Z=zero[group] ⇒ 0x6400|zero.
        s += "    mov.b32 %scx2,{%sc,%sc};\n";
        if zero_point {
            s += "    or.b32 %zv,%zv,0x6400;\n    shl.b32 %ztmp,%zv,16;\n    or.b32 %zsub,%zv,%ztmp;\n";
        } else {
            s += "    mov.b32 %zsub,0x64086408;\n"; // fp16(1032) in both lanes
        }
        // SMEM dest base byte = e*16 (chunk base). Fast Marlin/AWQ unpack: each `lop3` extracts an
        // interleaved nibble PAIR as `(word>>4jj)&0x000F000F | 0x64006400` — an f16x2 of (1024+u) per
        // lane — then `sub.f16x2` the zero-offset (→ u-Z) and `mul.f16x2` the scale dequant TWO weights
        // per op, no per-element shift/convert. The 4 pairs write the 16-byte row with one st.shared.v4.
        s += &format!("    mov.u32 %sbase,smemB_{name};\n    shl.b32 %tmp,%e,4;\n    add.u32 %sbase,%sbase,%tmp;\n");
        for jj in 0..4 {
            if jj == 0 {
                s += "    lop3.b32 %pk0,%word,0x000f000f,0x64006400,0xea;\n";
            } else {
                s += &format!("    shr.b32 %wsh,%word,{};\n    lop3.b32 %pk{jj},%wsh,0x000f000f,0x64006400,0xea;\n", 4 * jj);
            }
            s += &format!("    sub.rn.f16x2 %pk{jj},%pk{jj},%zsub;\n    mul.rn.f16x2 %pk{jj},%pk{jj},%scx2;\n");
        }
        s += "    st.shared.v4.u32 [%sbase],{%pk0,%pk1,%pk2,%pk3};\n";
    }
    let _ = veclist; // (declared above for the fragment lists below)
    s += "    bar.sync 0;\n";

    // --- Compute: each warp loads its fragments from SMEM and accumulates (identical to fp16 path). ---
    for ti in 0..tm {
        s += &format!("    mov.u32 %tmp,smemA_{name};\n");
        s += &format!(
            "    mul.lo.s32 %tmp2,%warpRow,{wm};\n    add.u32 %tmp2,%tmp2,{};\n",
            ti * 16
        );
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let ra = veclist(&format!("a{ti}_"), nab);
        s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.f16 {ra}, [%gp], %ldm;\n");
    }
    for tj in 0..tn {
        s += &format!("    mov.u32 %tmp,smemB_{name};\n");
        s += &format!(
            "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
            tj * 16
        );
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let rb = veclist(&format!("b{tj}_"), nab);
        s += &format!("    wmma.load.b.sync.aligned.m16n16k16.col.f16 {rb}, [%gp], %ldm;\n");
    }
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"), nab);
        for tj in 0..tn {
            let rb = veclist(&format!("b{tj}_"), nab);
            let cc = veclist(&format!("c{ti}_{tj}_"), 8);
            s += &format!(
                "    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {cc}, {ra}, {rb}, {cc};\n"
            );
        }
    }
    s += "    bar.sync 0;\n";
    s += &format!("    add.u32 %kt,%kt,16;\n    bra KLOOP_{name};\n");

    s += &format!("KEND_{name}:\n");
    for ti in 0..tm {
        for tj in 0..tn {
            s += &format!(
                "    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n",
                ti * 16
            );
            s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
            s += &format!(
                "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
                tj * 16
            );
            s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
            s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
            let cc = veclist(&format!("c{ti}_{tj}_"), 8);
            s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;\n");
        }
    }
    s += "    ret;\n}\n";
    s
}

/// 64×64 CTA tile (2×2 warps, 128 threads) — the default W4A16 tile, matching the fp16 `_sm` config.
pub const W4_BM: usize = 64;
pub const W4_BN: usize = 64;
pub const W4_WARPS_M: usize = 2;
pub const W4_WARPS_N: usize = 2;
pub const W4_THREADS: usize = W4_WARPS_M * W4_WARPS_N * 32;

/// W4A16 module — entry `gemm_nt_w4a16` (symmetric signed int4) and `gemm_nt_w4a16_z` (asymmetric,
/// zero-point), both the 64×64 SMEM-staged tensor-core tile with `GROUP_SIZE` baked in (the dims M/N/K
/// stay runtime params).
pub fn w4a16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(HDR_SM80);
        m += &entry_w4a16(
            "gemm_nt_w4a16",
            W4_BM,
            W4_BN,
            W4_WARPS_M,
            W4_WARPS_N,
            GROUP_SIZE,
            false,
            None,
            false,
        );
        m += &entry_w4a16(
            "gemm_nt_w4a16_z",
            W4_BM,
            W4_BN,
            W4_WARPS_M,
            W4_WARPS_N,
            GROUP_SIZE,
            true,
            None,
            false,
        );
        m
    })
    .as_str()
}

/// **Deterministic fixed-order reduction kernel** for W4A16 split-K (`w4a16_splitk_reduce`): sums the
/// `sk` partial f32 planes (`Part[z·MN + i]`, written disjointly by the split-K GEMM CTAs) into the
/// final `C[i]`. One thread per output element, grid-stride; the inner `z = 0,1,…,sk-1` loop fixes the
/// summation order, so the result is bit-identical run-to-run (M12) — the property a float `atomicAdd`
/// reduction can't promise. Params: `pMN` (= M·N), `pSK` (= number of splits), `pPart`, `pC`.
const W4A16_SPLITK_REDUCE: &str = r#"
.visible .entry w4a16_splitk_reduce(
    .param .u32 pMN,
    .param .u32 pSK,
    .param .u64 pPart,
    .param .u64 pC
)
{
    .reg .pred %p;
    .reg .b32 %mn,%sk,%i,%stride,%z,%zi,%t,%b,%nt,%ng;
    .reg .f32 %acc,%v;
    .reg .b64 %Part,%C,%off,%pp,%cc;
    ld.param.u32 %mn,[pMN];
    ld.param.u32 %sk,[pSK];
    ld.param.u64 %Part,[pPart];
    ld.param.u64 %C,[pC];
    cvta.to.global.u64 %Part,%Part;
    cvta.to.global.u64 %C,%C;
    mov.u32 %t,%tid.x;
    mov.u32 %b,%ctaid.x;
    mov.u32 %nt,%ntid.x;
    mad.lo.s32 %i,%b,%nt,%t;
    mov.u32 %ng,%nctaid.x;
    mul.lo.s32 %stride,%ng,%nt;
RLOOP:
    setp.ge.u32 %p,%i,%mn;
    @%p bra REND;
    mov.f32 %acc,0f00000000;
    mov.u32 %z,0;
    mov.u32 %zi,%i;
RZ:
    setp.ge.u32 %p,%z,%sk;
    @%p bra RZEND;
    mul.wide.u32 %off,%zi,4;
    add.s64 %pp,%Part,%off;
    ld.global.f32 %v,[%pp];
    add.f32 %acc,%acc,%v;
    add.u32 %zi,%zi,%mn;
    add.u32 %z,%z,1;
    bra RZ;
RZEND:
    mul.wide.u32 %off,%i,4;
    add.s64 %cc,%C,%off;
    st.global.f32 [%cc],%acc;
    add.u32 %i,%i,%stride;
    bra RLOOP;
REND:
    ret;
}
"#;

/// **W4A16 split-K module** for the thin-M / small-N decode regime (the dominant LLM-inference shape:
/// tiny M, large K, the M·N grid leaves SMs idle). Launched with `gridDim.z = sk`, each CTA computes a
/// partial over its K-range and writes it to its own plane of an `sk·M·N` f32 buffer; the bundled
/// `w4a16_splitk_reduce` then sums the planes in fixed order → **deterministic** final C (a float
/// `atomicAdd` split-K could not be). Symmetric (no zero-point) path; entry `gemm_nt_w4a16_sk`. Same
/// fp16-accumulate tolerance as [`w4a16_ptx`] (the partials are the same products, only the cross-CTA
/// K-partition differs). Requires K % (`sk`·`GROUP_SIZE`) == 0 so every split is whole quant groups.
pub fn w4a16_splitk_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(HDR_SM80);
        m += &entry_w4a16(
            "gemm_nt_w4a16_sk",
            W4_BM,
            W4_BN,
            W4_WARPS_M,
            W4_WARPS_N,
            GROUP_SIZE,
            false,
            None,
            true,
        );
        m += W4A16_SPLITK_REDUCE;
        m
    })
    .as_str()
}

/// **Static-shape-specialized** W4A16 module for the exact compile-time dims `(m,n,k)` — Wukong's
/// no-library lever (§1A.2): the dims are baked as constants, so ptxas strength-reduces every hot-loop
/// stride (`×K`, `×N`, `K/8`, `K/group`) to shifts/immediates (for power-of-two dims like 4096→`<<12`),
/// where the dynamic kernel must keep them as register multiplies. One entry `gemm_nt_w4a16_static`
/// (`_z` for the zero-point path). Returns an owned module string (per-shape ⇒ not interned); the
/// driver JIT + the persistent cubin cache (M10) make the per-shape compile a one-time, cached cost.
pub fn w4a16_static_ptx(m: usize, n: usize, k: usize, zero_point: bool) -> String {
    let name = if zero_point {
        "gemm_nt_w4a16_static_z"
    } else {
        "gemm_nt_w4a16_static"
    };
    let mut s = String::from(HDR_SM80);
    s += &entry_w4a16(
        name,
        W4_BM,
        W4_BN,
        W4_WARPS_M,
        W4_WARPS_N,
        GROUP_SIZE,
        zero_point,
        Some((m, n, k)),
        false,
    );
    s
}

/// Entry name for the static-shape kernel ([`w4a16_static_ptx`]); `_z` suffix for the zero-point path.
pub fn w4a16_static_entry(zero_point: bool) -> &'static str {
    if zero_point {
        "gemm_nt_w4a16_static_z"
    } else {
        "gemm_nt_w4a16_static"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx_target::TARGET_SM80;

    /// Host quant→dequant must be self-consistent and faithful: the symmetric reconstruction stays
    /// within one quantization step of the original, and the packed/scale shapes are exactly the
    /// kernel's expected layout. (Pure-CPU — runs under plain `cargo test --features gpu`.)
    #[test]
    fn symmetric_quant_roundtrip_is_faithful() {
        let (n, k, group) = (8usize, 256usize, GROUP_SIZE);
        let mut rng = crate::diff::Rng::new(0xA11CE);
        let w = rng.vec(n * k, -2.0, 2.0);
        let qw = quantize_weight_symmetric(&w, n, k, group);
        assert_eq!(qw.packed.len(), n * k / 8);
        assert_eq!(qw.scales.len(), n * k / group);
        assert!(qw.zeros.is_none() && qw.signed);
        let deq = dequant_weight(&qw);
        // Each weight is within ~one step (scale) of the original — a faithful symmetric quantizer.
        for r in 0..n {
            for g in 0..(k / group) {
                let scale = qw.scales[r * (k / group) + g].to_f32();
                for i in 0..group {
                    let kk = g * group + i;
                    let err = (deq[r * k + kk].to_f32() - w[r * k + kk]).abs();
                    assert!(err <= scale + 1e-3, "deq err {err} > step {scale}");
                }
            }
        }
    }

    /// Asymmetric (zero-point) quantization must reconstruct within one step too, and carry a zeros
    /// table of the right shape with unsigned nibbles.
    #[test]
    fn asymmetric_quant_roundtrip_is_faithful() {
        let (n, k, group) = (4usize, 128usize, GROUP_SIZE);
        let mut rng = crate::diff::Rng::new(0xB0B);
        // A deliberately asymmetric range (all-positive) — where a zero-point earns its keep.
        let w: Vec<f32> = (0..n * k).map(|_| rng.f32_range(0.5, 3.0)).collect();
        let qw = quantize_weight_asymmetric(&w, n, k, group);
        assert_eq!(qw.zeros.as_ref().unwrap().len(), n * k / group);
        assert!(!qw.signed);
        let deq = dequant_weight(&qw);
        for r in 0..n {
            let scale = qw.scales[r * (k / group)].to_f32();
            for kk in 0..k {
                let err = (deq[r * k + kk].to_f32() - w[r * k + kk]).abs();
                assert!(err <= scale + 1e-3, "asym deq err {err} > step {scale}");
            }
        }
    }

    /// **The oracle and the launchers must dispatch symmetric-vs-asymmetric off the SAME field.**
    /// `QuantWeight` is `pub` with `pub` fields, so an external caller loading GPTQ/AWQ weights builds
    /// one by hand; `gpu::gemm_nt_w4a16` (and every other W4A16 launcher, plus `autotune`) selects the
    /// kernel by matching on `zeros`, so `dequant_weight` — the exact reference the W4A16 first-law
    /// gate compares against — must read `zeros` too. Reading `signed` instead silently redefined the
    /// oracle for an inconsistent pair, and `zeros.unwrap()` panicked outright on the other one.
    #[test]
    fn dequant_follows_zeros_not_the_signed_flag() {
        let (n, k, group) = (2usize, 256usize, GROUP_SIZE);
        let mut rng = crate::diff::Rng::new(0x4E12);
        let w = rng.vec(n * k, -2.0, 2.0);

        // A hand-built asymmetric weight that mislabels itself `signed: true`. The launcher takes the
        // `Some(zeros)` arm and dequants (u - z)*scale, so the reference must do the same.
        let asym = quantize_weight_asymmetric(&w, n, k, group);
        let want = dequant_weight(&asym);
        let mislabelled = QuantWeight {
            signed: true,
            ..asym.clone()
        };
        assert_eq!(
            dequant_weight(&mislabelled),
            want,
            "oracle must follow `zeros`, not `signed`"
        );

        // The mirror case: a symmetric weight mislabelled `signed: false`. The launcher takes the
        // `None` arm (Z = 8); the reference must not panic on `zeros.unwrap()`.
        let sym = quantize_weight_symmetric(&w, n, k, group);
        let want = dequant_weight(&sym);
        let mislabelled = QuantWeight {
            signed: false,
            ..sym.clone()
        };
        assert_eq!(
            dequant_weight(&mislabelled),
            want,
            "oracle must not read `signed` for Z"
        );
    }

    /// The generated PTX must be **pure ASCII** (a single non-ASCII byte is a `ptxas fatal` on this
    /// box), contain both entry points, and carry the family's **`sm_80` floor** — the `lop3` unpack
    /// and the `m16n16k16` `wmma` are Ampere-legal, so tagging Ada would lock the module out of every
    /// A100 for nothing. PTX is forward-compatible only, so the floor must be the lowest legal arch.
    #[test]
    fn w4a16_ptx_is_ascii_and_complete() {
        for (label, ptx) in [
            ("w4a16", w4a16_ptx().to_string()),
            ("w4a16_splitk", w4a16_splitk_ptx().to_string()),
        ] {
            assert!(
                ptx.starts_with(HDR_SM80),
                "{label}: must open with the routed HDR_SM80 header"
            );
            assert!(
                ptx.contains(TARGET_SM80),
                "{label}: must carry the int4 floor {TARGET_SM80}"
            );
            assert!(!ptx.contains("sm_89"), "{label}: nothing here is Ada-only");
        }
        let ptx = w4a16_ptx();
        assert!(ptx.is_ascii(), "PTX must be ASCII");
        assert!(ptx.contains(".visible .entry gemm_nt_w4a16("));
        assert!(ptx.contains(".visible .entry gemm_nt_w4a16_z("));
        assert!(ptx.contains("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32"));
        assert!(
            ptx.contains("lop3.b32"),
            "fast Marlin/AWQ unpack must use lop3"
        );
        // Static-shape module: ASCII, the right entry, and the dims baked as `mov` constants (not loaded
        // from params) so ptxas can strength-reduce — e.g. K=4096 appears as an immediate.
        let st = w4a16_static_ptx(64, 4096, 4096, false);
        assert!(st.is_ascii(), "static PTX must be ASCII");
        assert!(
            st.starts_with(HDR_SM80),
            "static PTX must open with the routed HDR_SM80 header"
        );
        assert!(st.contains(".visible .entry gemm_nt_w4a16_static("));
        assert!(
            st.contains("mov.u32 %K,4096;"),
            "static kernel must bake K as a constant"
        );
    }
}
