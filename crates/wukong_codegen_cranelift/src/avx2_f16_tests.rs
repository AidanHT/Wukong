//! The raw-AVX2 recipe's f16 operations (`LoadF16`, `StoreF16`, `RoundF16` — F16C's
//! `vcvtph2ps`/`vcvtps2ph`), assembled, installed and run, against the scalar path's runtime shims
//! and against the recipe's reference evaluator (what the interpreter runs), bit for bit.
//!
//! F16C is used only because it is bit-identical to the shims on every input class: the shims are
//! the `half` crate, which on an F16C host *is* `vcvtps2ph` with imm8 0 and `vcvtph2ps`, and on any
//! other host an integer routine `half_lane_tests` checks against both on every f16 encoding and a
//! dense cover of `f32` inputs. Here the machine code is checked the same way: widening on all
//! 65536 encodings, narrowing and grid rounding on the dense `f32` cover (every exponent, a tie and
//! a run at every rounding position, every f16 midpoint and its neighbours, every NaN payload, a
//! million random patterns), and a mixed body — two f16 inputs, arithmetic, a rounding, an f16 and
//! an `f32` output — at unrolled and single-group trips on unaligned buffers, with the elements
//! past the vector part left untouched.

use cranelift_codegen::ir::{types, AbiParam, Signature};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use wukong_mir::{VecBin, VecKernel, VecLane, VecOp};

use crate::avx2::{assemble_kernel, host_supports_kernels};
use crate::half_lane_tests::{dense_f32_inputs, soft_f16_to_f32, soft_f32_to_f16};

/// Elements of padding past every buffer's vector part, filled with a canary the kernel must not
/// touch.
const PAD: usize = 24;
const CANARY16: u16 = 0xa5c3;
const CANARY32: u32 = 0xdead_beef;

fn kernel(streams: u32, scalars: u32, ops: Vec<VecOp>) -> VecKernel {
    VecKernel {
        name: wukong_span::Interner::new().intern("k"),
        lane: VecLane::F32,
        streams,
        scalars,
        ops,
        unroll: 4,
        reduce: None,
    }
}

/// Can this host run an f16 recipe? Where AVX2 is present but F16C is not, the assembler must
/// refuse rather than emit bytes that `#UD`.
fn f16_host(k: &VecKernel) -> bool {
    if wukong_mir::host_supports_f16c_kernels() {
        return true;
    }
    if host_supports_kernels() {
        assert!(
            assemble_kernel(k).is_err(),
            "an f16 recipe assembled on a host without F16C"
        );
    }
    false
}

/// One stream's buffer: `offset` elements of lead-in (so the stream is not 32-byte aligned), the
/// data, and [`PAD`] canary elements.
struct Buf<T: Copy> {
    mem: Vec<T>,
    offset: usize,
}

impl<T: Copy> Buf<T> {
    fn new(data: &[T], offset: usize, canary: T) -> Buf<T> {
        let mut mem = vec![canary; offset];
        mem.extend_from_slice(data);
        mem.extend(std::iter::repeat_n(canary, PAD));
        Buf { mem, offset }
    }
    fn ptr(&mut self) -> *mut u8 {
        // SAFETY: `offset` is within `mem`.
        unsafe { self.mem.as_mut_ptr().add(self.offset) as *mut u8 }
    }
    fn data(&self, n: usize) -> &[T] {
        &self.mem[self.offset..self.offset + n]
    }
}

/// A JIT module holding `bytes` as the kernel `fn(ptrs, scalars, n)`, and its address.
fn install(bytes: &[u8]) -> (JITModule, *const u8) {
    let isa = crate::make_isa(false).expect("isa");
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);
    let ptr = module.target_config().pointer_type();
    let mut sig = Signature::new(module.target_config().default_call_conv);
    sig.params.push(AbiParam::new(ptr));
    sig.params.push(AbiParam::new(ptr));
    sig.params.push(AbiParam::new(types::I64));
    let fid = module
        .declare_function("f16_kernel", Linkage::Export, &sig)
        .expect("declare");
    module
        .define_function_bytes(fid, 16, bytes, &[])
        .expect("define");
    module.finalize_definitions().expect("finalize");
    let code = module.get_finalized_function(fid);
    (module, code)
}

/// Install `bytes` and call the kernel once over `n` elements.
fn run(bytes: &[u8], ptrs: &[*mut u8], scalars: &[f32], n: usize) {
    let (module, code) = install(bytes);
    // SAFETY: the bytes are an assembled kernel with this ABI (`avx2` module docs), and every
    // stream pointer addresses at least `n` elements of its width.
    let call: extern "C" fn(*const *mut u8, *const f32, u64) = unsafe { std::mem::transmute(code) };
    call(ptrs.as_ptr(), scalars.as_ptr(), n as u64);
    // SAFETY: nothing installed in the module runs after this point.
    unsafe { module.free_memory() };
}

/// `LoadF16` of every f16 encoding, stored as `f32`: the shim's widening, and the reference's.
#[test]
fn f16c_widening_matches_the_shim_on_every_encoding() {
    let k = kernel(
        2,
        0,
        vec![
            VecOp::LoadF16 { stream: 0 },
            VecOp::Store { stream: 1, val: 0 },
        ],
    );
    assert!(k.validate().is_empty(), "{:?}", k.validate());
    if !f16_host(&k) {
        return;
    }
    let bytes = assemble_kernel(&k).expect("assemble");
    let input: Vec<u16> = (0..=u16::MAX).collect();
    let n = input.len();
    for offset in [0usize, 1] {
        let mut src = Buf::new(&input, offset, CANARY16);
        let mut dst = Buf::new(&vec![0u32; n], offset, CANARY32);
        run(&bytes, &[src.ptr(), dst.ptr()], &[], n);
        for (&h, &got) in input.iter().zip(dst.data(n)) {
            let shim = wukong_runtime::f16_bits_to_f32(h).to_bits();
            assert_eq!(
                shim,
                soft_f16_to_f32(h).to_bits(),
                "shim vs software {h:#06x}"
            );
            assert_eq!(
                got, shim,
                "f16 {h:#06x} widened to {got:#010x}, shim {shim:#010x}"
            );
            let mut want = 0f32;
            k.eval_lane(
                |_| wukong_runtime::f16_bits_to_f32(h),
                |_| 0.0,
                |_, v| want = v,
            );
            assert_eq!(got, want.to_bits(), "reference disagrees on {h:#06x}");
        }
        assert!(
            dst.mem[offset + n..].iter().all(|&c| c == CANARY32),
            "store past n"
        );
    }
}

/// `StoreF16` of the dense `f32` cover: the shim's rounding, ties to even, overflow to infinity,
/// subnormal boundaries and NaN payloads, and the reference's.
#[test]
fn f16c_narrowing_matches_the_shim_on_dense_inputs() {
    let k = kernel(
        2,
        0,
        vec![
            VecOp::Load { stream: 0 },
            VecOp::StoreF16 { stream: 1, val: 0 },
        ],
    );
    assert!(k.validate().is_empty(), "{:?}", k.validate());
    if !f16_host(&k) {
        return;
    }
    let bytes = assemble_kernel(&k).expect("assemble");
    let mut input = dense_f32_inputs();
    input.truncate(input.len() / 8 * 8);
    let n = input.len();
    let mut src = Buf::new(&input, 1, CANARY32);
    let mut dst = Buf::new(&vec![0u16; n], 1, CANARY16);
    run(&bytes, &[src.ptr(), dst.ptr()], &[], n);
    for (&x, &got) in input.iter().zip(dst.data(n)) {
        let shim = wukong_runtime::f32_to_f16_bits(f32::from_bits(x));
        assert_eq!(
            shim,
            soft_f32_to_f16(f32::from_bits(x)),
            "shim vs software {x:#010x}"
        );
        assert_eq!(
            got, shim,
            "f32 {x:#010x} narrowed to {got:#06x}, shim {shim:#06x}"
        );
        let mut want = 0f32;
        k.eval_lane(|_| f32::from_bits(x), |_| 0.0, |_, v| want = v);
        assert_eq!(
            got,
            wukong_runtime::f32_to_f16_bits(want),
            "reference disagrees on {x:#010x}"
        );
    }
    assert!(
        dst.mem[1 + n..].iter().all(|&c| c == CANARY16),
        "store past n"
    );
}

/// `RoundF16` (a vector `fptrunc .. to f16`) on the dense cover is the shims' round trip.
#[test]
fn f16c_rounding_matches_the_shims_on_dense_inputs() {
    let k = kernel(
        2,
        0,
        vec![
            VecOp::Load { stream: 0 },
            VecOp::RoundF16 { a: 0 },
            VecOp::Store { stream: 1, val: 1 },
        ],
    );
    assert!(k.validate().is_empty(), "{:?}", k.validate());
    if !f16_host(&k) {
        return;
    }
    let bytes = assemble_kernel(&k).expect("assemble");
    let mut input = dense_f32_inputs();
    input.truncate(input.len() / 8 * 8);
    let n = input.len();
    let mut src = Buf::new(&input, 0, CANARY32);
    let mut dst = Buf::new(&vec![0u32; n], 0, CANARY32);
    run(&bytes, &[src.ptr(), dst.ptr()], &[], n);
    for (&x, &got) in input.iter().zip(dst.data(n)) {
        let shim = wukong_runtime::round_f16(f32::from_bits(x)).to_bits();
        assert_eq!(
            got, shim,
            "f32 {x:#010x} rounded to {got:#010x}, shim {shim:#010x}"
        );
    }
}

/// A mixed body, element for element against the reference evaluator: `x`, `y` f16 streams,
/// `r = round_f16(x * y)`, `o = fma(r, s, x)` stored as f16 and `r` stored as `f32`, at trips that
/// take the unrolled main loop and the single-group cleanup (and both), on unaligned buffers. The
/// inputs cycle through every class an f16 holds, both NaN kinds included.
#[test]
fn a_mixed_f16_body_matches_the_reference() {
    let k = kernel(
        4,
        1,
        vec![
            VecOp::LoadF16 { stream: 0 }, // 0: x
            VecOp::LoadF16 { stream: 1 }, // 1: y
            VecOp::Bin {
                op: VecBin::Mul,
                a: 0,
                b: 1,
            }, // 2: x * y
            VecOp::RoundF16 { a: 2 },     // 3: r = f16(x * y)
            VecOp::Splat { scalar: 0 },   // 4: s
            VecOp::Fma { a: 3, b: 4, c: 0 }, // 5: r * s + x
            VecOp::StoreF16 { stream: 2, val: 5 },
            VecOp::Store { stream: 3, val: 3 },
        ],
    );
    assert!(k.validate().is_empty(), "{:?}", k.validate());
    if !f16_host(&k) {
        return;
    }
    let bytes = assemble_kernel(&k).expect("assemble");
    const XS: [u16; 13] = [
        0x7e01, 0x7d00, 0x8000, 0x0000, 0x7c00, 0xfc00, 0x3e00, 0xc080, 0x0001, 0x7bff, 0x3c01,
        0x03ff, 0xbc00,
    ];
    const YS: [u16; 11] = [
        0x3c00, 0x4000, 0x7c00, 0x8000, 0x0000, 0x3bff, 0xc200, 0x0200, 0x6c00, 0x3800, 0x7e02,
    ];
    let s = 1.5f32;
    for n in [8usize, 32, 40, 64, 1000, 1024 + 24] {
        let xs: Vec<u16> = (0..n).map(|i| XS[i % XS.len()]).collect();
        let ys: Vec<u16> = (0..n).map(|i| YS[(7 * i + 3) % YS.len()]).collect();
        let mut x = Buf::new(&xs, 3, CANARY16);
        let mut y = Buf::new(&ys, 5, CANARY16);
        let mut o = Buf::new(&vec![0u16; n], 1, CANARY16);
        let mut r = Buf::new(&vec![0u32; n], 1, CANARY32);
        run(&bytes, &[x.ptr(), y.ptr(), o.ptr(), r.ptr()], &[s], n);
        for i in 0..n {
            let (mut want_o, mut want_r) = (0f32, 0f32);
            k.eval_lane(
                |stream| wukong_runtime::f16_bits_to_f32([xs[i], ys[i]][stream as usize]),
                |_| s,
                |stream, v| {
                    if stream == 2 {
                        want_o = v;
                    } else {
                        want_r = v;
                    }
                },
            );
            assert_eq!(
                o.data(n)[i],
                wukong_runtime::f32_to_f16_bits(want_o),
                "n={n} lane {i}: o, x={:#06x} y={:#06x}",
                xs[i],
                ys[i]
            );
            assert_eq!(
                r.data(n)[i],
                want_r.to_bits(),
                "n={n} lane {i}: r, x={:#06x} y={:#06x}",
                xs[i],
                ys[i]
            );
        }
        assert!(
            o.mem[1 + n..].iter().all(|&c| c == CANARY16),
            "o past n={n}"
        );
        assert!(
            r.mem[1 + n..].iter().all(|&c| c == CANARY32),
            "r past n={n}"
        );
    }
}

/// The recipe's own rules: an f16 operation belongs to an `f32` lane, and a stream has one
/// element width.
#[test]
fn f16_operations_validate_only_on_an_f32_lane_with_one_width_per_stream() {
    let mut k = kernel(
        2,
        0,
        vec![
            VecOp::LoadF16 { stream: 0 },
            VecOp::StoreF16 { stream: 1, val: 0 },
        ],
    );
    assert!(k.validate().is_empty());
    assert!(k.uses_f16());
    k.lane = VecLane::F64;
    assert!(
        k.validate().iter().any(|e| e.contains("non-f32 lane")),
        "{:?}",
        k.validate()
    );
    let mixed = kernel(
        1,
        0,
        vec![
            VecOp::LoadF16 { stream: 0 },
            VecOp::Store { stream: 0, val: 0 },
        ],
    );
    assert!(
        mixed
            .validate()
            .iter()
            .any(|e| e.contains("both as f16 and as f32")),
        "{:?}",
        mixed.validate()
    );
    assert!(
        mixed.pressure().is_none(),
        "an invalid recipe has no register plan"
    );
}
