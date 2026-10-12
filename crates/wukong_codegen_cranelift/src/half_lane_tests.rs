//! The vector f16 conversions of [`crate::half_lanes`], JIT-compiled and compared bit for bit with
//! the scalar path they replace.
//!
//! The scalar path is the runtime shims (`wukong_f32_to_f16_bits` / `wukong_f16_bits_to_f32`), the
//! `half` crate — F16C's `vcvtps2ph`/`vcvtph2ps` on this host when it has F16C, and the crate's
//! integer software routine on one that does not. So every result is checked against **both**: the
//! shim as it runs here, and [`soft_f32_to_f16`] / [`soft_f16_to_f32`], a transcription of the
//! `half` crate's software fallback (`binary16/arch.rs`, `f32_to_f16_fallback` and
//! `f16_to_f32_fallback`, half 2.7.1). A conversion that matched only the instruction, or only the
//! fallback, would let native code differ between two hosts.
//!
//! f16 to f32 is checked exhaustively (all 65536 encodings). f32 to f16 is checked densely: every
//! sign and exponent against a set of mantissas that put a one, a run of ones or a tie at every
//! rounding position; every f16 encoding's value, its midpoint with the next encoding (the exact
//! tie) and the f32 neighbours of both; every NaN payload's top ten bits with SNaN and QNaN low
//! bits; and a million pseudo-random bit patterns. Each runs under the host's ISA and under the
//! x86-64 baseline (SSE2 only), whose lowering of the integer and pack operations differs.

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Signature};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use wukong_mir::MirType;

use crate::half_lanes;
use crate::target::TargetCpu;

/// `half`'s `f32_to_f16_fallback`, transcribed.
pub(crate) fn soft_f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = x & 0x8000_0000;
    let exp = x & 0x7f80_0000;
    let man = x & 0x007f_ffff;
    if exp == 0x7f80_0000 {
        let nan_bit = if man == 0 { 0 } else { 0x0200 };
        return ((sign >> 16) | 0x7c00 | nan_bit | (man >> 13)) as u16;
    }
    let half_sign = sign >> 16;
    let unbiased_exp = ((exp >> 23) as i32) - 127;
    let half_exp = unbiased_exp + 15;
    if half_exp >= 0x1f {
        return (half_sign | 0x7c00) as u16;
    }
    if half_exp <= 0 {
        if 14 - half_exp > 24 {
            return half_sign as u16;
        }
        let man = man | 0x0080_0000;
        let mut half_man = man >> (14 - half_exp);
        let round_bit = 1 << (13 - half_exp);
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            half_man += 1;
        }
        return (half_sign | half_man) as u16;
    }
    let half_exp = (half_exp as u32) << 10;
    let half_man = man >> 13;
    let round_bit = 0x0000_1000;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        ((half_sign | half_exp | half_man) + 1) as u16
    } else {
        (half_sign | half_exp | half_man) as u16
    }
}

/// `half`'s `f16_to_f32_fallback`, transcribed.
pub(crate) fn soft_f16_to_f32(i: u16) -> f32 {
    if i & 0x7fff == 0 {
        return f32::from_bits((i as u32) << 16);
    }
    let half_sign = (i & 0x8000) as u32;
    let half_exp = (i & 0x7c00) as u32;
    let half_man = (i & 0x03ff) as u32;
    if half_exp == 0x7c00 {
        if half_man == 0 {
            return f32::from_bits((half_sign << 16) | 0x7f80_0000);
        }
        return f32::from_bits((half_sign << 16) | 0x7fc0_0000 | (half_man << 13));
    }
    let sign = half_sign << 16;
    let unbiased_exp = ((half_exp as i32) >> 10) - 15;
    if half_exp == 0 {
        let e = (half_man as u16).leading_zeros() - 6;
        let exp = (127 - 15 - e) << 23;
        let man = (half_man << (14 + e)) & 0x7f_ffff;
        return f32::from_bits(sign | exp | man);
    }
    let exp = ((unbiased_exp + 127) as u32) << 23;
    let man = (half_man & 0x03ff) << 13;
    f32::from_bits(sign | exp | man)
}

/// Which conversion a JIT-compiled probe performs on four lanes, `fn(src, dst)`.
#[derive(Clone, Copy)]
enum Probe {
    /// Four f16 encodings at `src` → [`half_lanes::load_x4`] → the four `f32` bit patterns at
    /// `dst`.
    Widen,
    /// Four `f32`s at `src` → [`half_lanes::store_x4`] → four f16 encodings at `dst`.
    Narrow,
    /// Four `f32`s at `src` → [`half_lanes::round_x4`] → four `f32`s at `dst`.
    Round,
}

/// The JIT-compiled `probe`, under `target`'s instruction set. The module is returned so it stays
/// alive while the function runs.
fn compile(
    target: TargetCpu,
    probe: Probe,
) -> (JITModule, unsafe extern "C" fn(*const u8, *mut u8)) {
    let isa = crate::make_isa_for(false, true, target).expect("isa");
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);
    let ptr = module.target_config().pointer_type();
    let mut sig = Signature::new(module.target_config().default_call_conv);
    sig.params.push(AbiParam::new(ptr));
    sig.params.push(AbiParam::new(ptr));
    let fid = module
        .declare_function("half_probe", Linkage::Export, &sig)
        .expect("declare");
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let blk = b.create_block();
        b.append_block_params_for_function_params(blk);
        b.switch_to_block(blk);
        let (src, dst) = (b.block_params(blk)[0], b.block_params(blk)[1]);
        // Not `aligned`: the probe must not depend on how the caller's buffers are placed.
        let flags = MemFlags::new().with_notrap();
        match probe {
            Probe::Widen => {
                let v = half_lanes::load_x4(&mut b, &MirType::F16, flags, src);
                b.ins().store(flags, v, dst, 0);
            }
            Probe::Narrow => {
                let v = b.ins().load(types::F32X4, flags, src, 0);
                half_lanes::store_x4(&mut b, &MirType::F16, flags, v, dst);
            }
            Probe::Round => {
                let v = b.ins().load(types::F32X4, flags, src, 0);
                let r = half_lanes::round_x4(&mut b, &MirType::F16, v);
                b.ins().store(flags, r, dst, 0);
            }
        }
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize();
    }
    module.define_function(fid, &mut ctx).expect("define");
    module.finalize_definitions().expect("finalize");
    let code = module.get_finalized_function(fid);
    // SAFETY: the function was declared with exactly this two-pointer signature.
    let f =
        unsafe { std::mem::transmute::<*const u8, unsafe extern "C" fn(*const u8, *mut u8)>(code) };
    (module, f)
}

/// The two instruction sets every probe runs under: the host's, and the SSE2 baseline.
fn targets() -> [TargetCpu; 2] {
    [TargetCpu::Native, TargetCpu::X86_64]
}

/// A 16-byte-aligned four-lane buffer.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
struct Lanes([u8; 16]);

/// Run `probe` over `inputs` four at a time (the last group padded with its own first element) and
/// return one output per input. Inputs and outputs are `in_bytes`/`out_bytes` wide (2 for an f16
/// encoding, 4 for an `f32`), little-endian, carried in `u32`s.
fn run(
    target: TargetCpu,
    probe: Probe,
    inputs: &[u32],
    in_bytes: usize,
    out_bytes: usize,
) -> Vec<u32> {
    let (module, f) = compile(target, probe);
    let mut out = Vec::with_capacity(inputs.len());
    for group in inputs.chunks(4) {
        let mut src = Lanes::default();
        for k in 0..4 {
            let x = group[k.min(group.len() - 1)].to_le_bytes();
            src.0[k * in_bytes..(k + 1) * in_bytes].copy_from_slice(&x[..in_bytes]);
        }
        let mut dst = Lanes::default();
        // SAFETY: `src` and `dst` each hold 16 bytes, the most the probe reads or writes.
        unsafe { f(src.0.as_ptr(), dst.0.as_mut_ptr()) };
        for k in 0..group.len() {
            let mut b = [0u8; 4];
            b[..out_bytes].copy_from_slice(&dst.0[k * out_bytes..(k + 1) * out_bytes]);
            out.push(u32::from_le_bytes(b));
        }
    }
    // SAFETY: nothing compiled in this module is called after this point.
    unsafe { module.free_memory() };
    out
}

/// Every f16 encoding widens to the shim's `f32` and the software routine's, bit for bit — and so
/// every NaN keeps its payload and gains the quiet bit, and every subnormal is exact.
#[test]
fn every_f16_encoding_widens_like_the_shim() {
    let all: Vec<u32> = (0..=u32::from(u16::MAX)).collect();
    for target in targets() {
        let got = run(target, Probe::Widen, &all, 2, 4);
        for (&h, &g) in all.iter().zip(&got) {
            let h = h as u16;
            let shim = wukong_runtime::f16_bits_to_f32(h).to_bits();
            let soft = soft_f16_to_f32(h).to_bits();
            assert_eq!(
                shim, soft,
                "the shim and the software routine disagree on {h:#06x}"
            );
            assert_eq!(
                g, shim,
                "{target:?}: f16 {h:#06x} widened to {g:#010x}, shim {shim:#010x}"
            );
        }
    }
}

/// Every f16 encoding's value narrows back to the encoding the shim produces: itself, or for a
/// signalling NaN the same payload with the quiet bit set.
#[test]
fn every_f16_value_narrows_back_like_the_shim() {
    let values: Vec<u32> = (0..=u16::MAX)
        .map(|h| wukong_runtime::f16_bits_to_f32(h).to_bits())
        .collect();
    for target in targets() {
        let got = run(target, Probe::Narrow, &values, 4, 2);
        for (h, (&v, &g)) in values.iter().zip(&got).enumerate() {
            let g = g as u16;
            let shim = wukong_runtime::f32_to_f16_bits(f32::from_bits(v));
            assert_eq!(
                g, shim,
                "{target:?}: f16 {h:#06x} (f32 {v:#010x}) narrowed to {g:#06x}"
            );
            let expect = if (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0 {
                h as u16 | 0x200
            } else {
                h as u16
            };
            assert_eq!(g, expect, "{target:?}: f16 {h:#06x} did not round-trip");
        }
    }
}

/// The dense f32 inputs: see the module docs.
pub(crate) fn dense_f32_inputs() -> Vec<u32> {
    let mut v: Vec<u32> = Vec::new();
    // Every sign and exponent, with mantissas that put a one, a run of ones and a tie at every
    // rounding position an f16 result can have (bit 13 for a normal, up to bit 23 below it).
    let mut mants: Vec<u32> = vec![
        0, 1, 2, 3, 0x7f_ffff, 0x7f_fffe, 0x40_0000, 0x3f_ffff, 0x55_5555,
    ];
    for k in 0..23 {
        let b = 1u32 << k;
        mants.extend([
            b,
            b - 1,
            b + 1,
            b | (b << 1),
            (b << 1).wrapping_sub(1),
            b | 1,
        ]);
    }
    mants.iter_mut().for_each(|m| *m &= 0x7f_ffff);
    for sign in [0u32, 0x8000_0000] {
        for exp in 0u32..256 {
            for &m in &mants {
                v.push(sign | (exp << 23) | m);
            }
        }
    }
    // Every f16 value, the exact midpoint to the next encoding, and the f32 neighbours of both
    // (the midpoint above the largest finite f16, 65520, rounds to infinity).
    for h in 0u16..0x7c00 {
        let lo = f64::from(soft_f16_to_f32(h));
        let hi = if h == 0x7bff {
            65536.0
        } else {
            f64::from(soft_f16_to_f32(h + 1))
        };
        let mid = ((lo + hi) / 2.0) as f32;
        assert_eq!(
            f64::from(mid),
            (lo + hi) / 2.0,
            "the midpoint above {h:#06x} is not an f32"
        );
        for x in [lo as f32, mid] {
            let b = x.to_bits();
            for d in [b, b.wrapping_add(1), b.wrapping_sub(1)] {
                v.push(d);
                v.push(d | 0x8000_0000);
            }
        }
    }
    // Every NaN payload's top ten bits, with the low bits empty, set, or alternating, and the
    // quiet bit either way (bit 22 is among the ten).
    for sign in [0u32, 0x8000_0000] {
        for top in 0u32..1024 {
            for low in [0u32, 1, 0x1fff, 0x0aaa] {
                let m = (top << 13) | low;
                if m != 0 {
                    v.push(sign | 0x7f80_0000 | m);
                }
            }
        }
    }
    // Pseudo-random bit patterns (xorshift32).
    let mut s: u32 = 0x9e37_79b9;
    for _ in 0..1_000_000 {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        v.push(s);
    }
    v
}

/// f32 to f16 on the dense inputs matches the shim and the software routine bit for bit: every
/// rounding tie goes to even, every overflow is infinity, every subnormal boundary rounds the same
/// way, and every NaN keeps its payload's top ten bits with the quiet bit set.
#[test]
fn f32_narrows_like_the_shim_on_dense_inputs() {
    let inputs = dense_f32_inputs();
    for target in targets() {
        let got = run(target, Probe::Narrow, &inputs, 4, 2);
        for (&x, &g) in inputs.iter().zip(&got) {
            let g = g as u16;
            let shim = wukong_runtime::f32_to_f16_bits(f32::from_bits(x));
            let soft = soft_f32_to_f16(f32::from_bits(x));
            assert_eq!(
                shim, soft,
                "the shim and the software routine disagree on {x:#010x}"
            );
            assert_eq!(
                g, shim,
                "{target:?}: f32 {x:#010x} narrowed to {g:#06x}, shim {shim:#06x}"
            );
        }
    }
}

/// The lane-wise round to the f16 grid (a vector `fptrunc .. to f16`) is the scalar
/// `round_f16` — the shims' round trip — on the dense inputs.
#[test]
fn f16_grid_rounding_matches_the_shims() {
    let inputs = dense_f32_inputs();
    for target in targets() {
        let got = run(target, Probe::Round, &inputs, 4, 4);
        for (&x, &g) in inputs.iter().zip(&got) {
            let shim = wukong_runtime::round_f16(f32::from_bits(x)).to_bits();
            assert_eq!(
                g, shim,
                "{target:?}: f32 {x:#010x} rounded to {g:#010x}, shim {shim:#010x}"
            );
        }
    }
}
