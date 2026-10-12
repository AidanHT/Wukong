use super::*;
use wukong_mir::Builder;

fn emit(f: Function, interner: &Interner) -> String {
    let mut program = Program::new();
    program.funcs.push(f);
    let text = try_emit_llvm_ir(&program, interner).unwrap();
    if let Some(dir) = std::env::var_os("WUKONG_LLVM_SEMANTICS_OUTPUT") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            std::path::Path::new(&dir)
                .join(format!("{}.ll", interner.resolve(program.funcs[0].name))),
            &text,
        )
        .unwrap();
    }
    text
}

#[test]
fn saturating_conversion_modules() {
    let mut i = Interner::new();
    for (from_name, from) in [("f32", MirType::F32), ("f64", MirType::F64)] {
        for (bits, to) in [
            (1, MirType::I1),
            (8, MirType::I8),
            (16, MirType::I16),
            (32, MirType::I32),
            (64, MirType::I64),
        ] {
            for (signed, kind) in [("i", CastKind::FpToSi), ("u", CastKind::FpToUi)] {
                let mut b = Builder::new(
                    i.intern(&format!("sat_{from_name}_{signed}{bits}")),
                    if bits == 1 { MirType::I8 } else { to.clone() },
                );
                let a = b.add_param(from.clone());
                let r = b.build(to.clone(), Op::Cast(kind, a, to.clone()));
                let returned = if bits == 1 {
                    b.build(MirType::I8, Op::Cast(CastKind::ZExt, r, MirType::I8))
                } else {
                    r
                };
                b.ret(Some(returned));
                let text = emit(b.finish(), &i);
                assert!(
                    text.contains(&format!("@llvm.{}.sat.", cast_name(kind))),
                    "{text}"
                );
                assert!(
                    !text.contains(&format!("= {} ", cast_name(kind))),
                    "unsaturated conversion: {text}"
                );
                if bits == 1 {
                    assert!(text.contains("trunc i32 %v1.sat to i1"), "{text}");
                }

                // Pointer wrappers avoid a platform-specific vector calling convention in JIT tests.
                let vf = MirType::Vec(Box::new(from.clone()), 4);
                let vt = MirType::Vec(Box::new(to.clone()), 4);
                let mut b = Builder::new(
                    i.intern(&format!("sat_vec_{from_name}_{signed}{bits}")),
                    MirType::Void,
                );
                let input = b.add_param(MirType::Ptr);
                let output = b.add_param(MirType::Ptr);
                let x = b.build(vf.clone(), Op::Load(input, vf));
                let r = b.build(vt.clone(), Op::Cast(kind, x, vt));
                // Store boolean lanes as bytes so the C ABI oracle does not interpret LLVM's
                // packed <4 x i1> memory representation as four independent bytes.
                let stored = if bits == 1 {
                    let bytes = MirType::Vec(Box::new(MirType::I8), 4);
                    b.build(bytes.clone(), Op::Cast(CastKind::ZExt, r, bytes))
                } else {
                    r
                };
                b.build_void(Op::Store {
                    ptr: output,
                    value: stored,
                });
                b.ret(None);
                let text = emit(b.finish(), &i);
                assert!(
                    text.contains(&format!(
                        "@llvm.{}.sat.v4i{}.v4{from_name}",
                        cast_name(kind),
                        if bits == 1 { 32 } else { bits }
                    )),
                    "{text}"
                );
                if bits == 1 {
                    assert!(
                        text.contains("trunc <4 x i32> %v3.sat to <4 x i1>"),
                        "{text}"
                    );
                    assert!(text.contains("zext <4 x i1> %v3 to <4 x i8>"), "{text}");
                }
            }
        }
    }
}

#[test]
fn exact_fma_modules() {
    let mut i = Interner::new();
    for (name, ty) in [("f32", MirType::F32), ("f64", MirType::F64)] {
        let mut b = Builder::new(i.intern(&format!("fma_{name}")), ty.clone());
        let a = b.add_param(ty.clone());
        let x = b.add_param(ty.clone());
        let c = b.add_param(ty.clone());
        let r = b.build(ty.clone(), Op::Fma(a, x, c));
        b.ret(Some(r));
        let text = emit(b.finish(), &i);
        assert!(text.contains(&format!("@llvm.fma.{name}")), "{text}");
        assert!(
            !text.contains("fmul") && !text.contains("fadd") && !text.contains("contract"),
            "{text}"
        );

        let vt = MirType::Vec(Box::new(ty), 4);
        let mut b = Builder::new(i.intern(&format!("fma_vec_{name}")), MirType::Void);
        let ap = b.add_param(MirType::Ptr);
        let bp = b.add_param(MirType::Ptr);
        let cp = b.add_param(MirType::Ptr);
        let output = b.add_param(MirType::Ptr);
        let a = b.build(vt.clone(), Op::Load(ap, vt.clone()));
        let x = b.build(vt.clone(), Op::Load(bp, vt.clone()));
        let c = b.build(vt.clone(), Op::Load(cp, vt.clone()));
        let r = b.build(vt, Op::Fma(a, x, c));
        b.build_void(Op::Store {
            ptr: output,
            value: r,
        });
        b.ret(None);
        let text = emit(b.finish(), &i);
        assert!(text.contains(&format!("@llvm.fma.v4{name}")), "{text}");
    }
}

#[test]
fn lowp_fma_computes_on_f32_registers() {
    // f16/bf16 values live in f32 registers (the interpreter's model), so a low-precision FMA is
    // the single-rounded f32 FMA the interpreter and Cranelift both evaluate.
    let mut i = Interner::new();
    for (ty, intrinsic) in [
        (MirType::F16, "@llvm.fma.f32(float"),
        (MirType::BF16, "@llvm.fma.f32(float"),
        (
            MirType::Vec(Box::new(MirType::F16), 4),
            "@llvm.fma.v4f32(<4 x float>",
        ),
    ] {
        let mut b = Builder::new(i.intern("lowp_fma"), ty.clone());
        let a = b.add_param(ty.clone());
        let r = b.build(ty, Op::Fma(a, a, a));
        b.ret(Some(r));
        let text = emit(b.finish(), &i);
        assert!(text.contains(intrinsic), "{text}");
        assert!(!text.contains("half") && !text.contains("bfloat"), "{text}");
    }
}

/// bf16 narrowing is the runtime's integer rounding, spelled in IR (`narrow_half`), so a module
/// needs no `__truncsfbf2`. `verify_semantics.py` runs both modules on every f32 bit pattern it
/// samples (ties, carries into the exponent, infinities, signalling and quiet NaN payloads) and
/// compares the stored bits with `wukong_runtime::f32_to_bf16_bits` transcribed. Inputs are read
/// from memory so a NaN payload reaches the module bit-exact.
#[test]
fn bf16_narrowing_modules() {
    let mut i = Interner::new();
    // bf16_bits(src: *f32) -> i16: the stored bits of `(*src) as bf16`.
    let mut b = Builder::new(i.intern("bf16_bits"), MirType::I16);
    let src = b.add_param(MirType::Ptr);
    let x = b.build(MirType::F32, Op::Load(src, MirType::F32));
    let h = b.build(MirType::BF16, Op::Cast(CastKind::FpTrunc, x, MirType::BF16));
    let slot = b.alloca(MirType::BF16);
    b.build_void(Op::Store {
        ptr: slot,
        value: h,
    });
    let bits = b.build(MirType::I16, Op::Load(slot, MirType::I16));
    b.ret(Some(bits));
    let text = emit(b.finish(), &i);
    assert!(text.contains("bitcast float %v1 to i32"), "{text}");
    assert!(
        !text
            .lines()
            .any(|l| l.contains("fptrunc") && l.contains("bfloat")),
        "{text}"
    );

    // bf16_vec_bits(src: *[f32; 4], dst: *[i16; 4]): the same, four lanes at once.
    let vf = MirType::Vec(Box::new(MirType::F32), 4);
    let vh = MirType::Vec(Box::new(MirType::BF16), 4);
    let vi = MirType::Vec(Box::new(MirType::I16), 4);
    let mut b = Builder::new(i.intern("bf16_vec_bits"), MirType::Void);
    let src = b.add_param(MirType::Ptr);
    let dst = b.add_param(MirType::Ptr);
    let x = b.build(vf.clone(), Op::Load(src, vf));
    let h = b.build(vh.clone(), Op::Cast(CastKind::FpTrunc, x, vh.clone()));
    let slot = b.alloca(vh);
    b.build_void(Op::Store {
        ptr: slot,
        value: h,
    });
    let bits = b.build(vi.clone(), Op::Load(slot, vi));
    b.build_void(Op::Store {
        ptr: dst,
        value: bits,
    });
    b.ret(None);
    let text = emit(b.finish(), &i);
    assert!(
        text.contains("bitcast <4 x float> %v2 to <4 x i32>"),
        "{text}"
    );
    assert!(text.contains("select <4 x i1>"), "{text}");
}

/// An explicit `alloca` alignment (`@align(N)`) only ever raises the native slot alignment the
/// emitter already promises.
#[test]
fn explicit_alloca_alignment_raises_the_slot_alignment() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("aligned_slots"), MirType::Void);
    b.alloca_aligned(
        MirType::Array(Box::new(MirType::I8), 24),
        wukong_mir::AddrSpace::Local,
        64,
    );
    // A 64-byte slot is naturally 32-aligned; `align 8` must not lower that.
    b.alloca_aligned(
        MirType::Array(Box::new(MirType::I8), 64),
        wukong_mir::AddrSpace::Local,
        8,
    );
    b.alloca(MirType::Array(Box::new(MirType::I8), 24));
    b.ret(None);
    let text = emit(b.finish(), &i);
    assert!(text.contains("alloca [24 x i8], align 64"), "{text}");
    assert!(text.contains("alloca [64 x i8], align 32"), "{text}");
    assert!(text.contains("alloca [24 x i8], align 16"), "{text}");
}

/// `atomic.add` is one `seq_cst` `atomicrmw add` of the value's own width, its old value named and
/// never read, at every integer width the verifier admits.
#[test]
fn atomic_add_is_one_seq_cst_atomicrmw() {
    let mut i = Interner::new();
    for (name, ty, llvm) in [
        ("atomic_i8", MirType::I8, "i8"),
        ("atomic_i16", MirType::I16, "i16"),
        ("atomic_i32", MirType::I32, "i32"),
        ("atomic_i64", MirType::I64, "i64"),
    ] {
        let mut b = Builder::new(i.intern(name), MirType::Void);
        let p = b.add_param(MirType::Ptr);
        let v = b.add_param(ty);
        b.atomic_add(p, v);
        b.ret(None);
        let text = emit(b.finish(), &i);
        let rmw: Vec<&str> = text.lines().filter(|l| l.contains("atomicrmw")).collect();
        assert_eq!(
            rmw,
            [format!("  %t0 = atomicrmw add ptr %v0, {llvm} %v1 seq_cst").as_str()],
            "{text}"
        );
    }
}

/// The 8-bit floats' codec, spelled in IR (`fp8.rs`). `verify_semantics.py` runs these modules on
/// every code and on thousands of `f32`/`f64` bit patterns (every midpoint and its neighbours,
/// ties, saturation, overflow, subnormals, NaNs) against an exact transcription of the runtime's
/// reference encoder, so the module rounds exactly as the interpreter does:
///
/// * `f8_{e4m3,e5m2}_code_{f32,f64}(src: *f32|*f64) -> i8`: the stored code of `(*src) as f8`
///   (an `fptrunc` to the type, a store, the byte read back);
/// * `f8_{e4m3,e5m2}_value(src: *i8) -> i32`: the `f32` bits of a stored code widened to `f32`
///   (a load decoding the byte, an `fpext` rounding it on the way out).
#[test]
fn f8_codec_modules() {
    let mut i = Interner::new();
    for (fname, f8) in [("e4m3", MirType::F8E4M3), ("e5m2", MirType::F8E5M2)] {
        for (sname, src_ty) in [("f32", MirType::F32), ("f64", MirType::F64)] {
            let mut b = Builder::new(i.intern(&format!("f8_{fname}_code_{sname}")), MirType::I8);
            let src = b.add_param(MirType::Ptr);
            let x = b.build(src_ty.clone(), Op::Load(src, src_ty.clone()));
            let h = b.build(f8.clone(), Op::Cast(CastKind::FpTrunc, x, f8.clone()));
            let slot = b.alloca(f8.clone());
            b.build_void(Op::Store {
                ptr: slot,
                value: h,
            });
            let code = b.build(MirType::I8, Op::Load(slot, MirType::I8));
            b.ret(Some(code));
            let text = emit(b.finish(), &i);
            assert!(text.contains(&format!("@llvm.roundeven.{sname}")), "{text}");
            assert!(text.contains("store i8"), "{text}");
            assert!(!text.contains("fptrunc"), "{text}");
        }
        let mut b = Builder::new(i.intern(&format!("f8_{fname}_value")), MirType::I32);
        let src = b.add_param(MirType::Ptr);
        let v = b.build(f8.clone(), Op::Load(src, f8.clone()));
        let w = b.build(MirType::F32, Op::Cast(CastKind::FpExt, v, MirType::F32));
        let bits = b.build(MirType::I32, Op::Cast(CastKind::Bitcast, w, MirType::I32));
        b.ret(Some(bits));
        let text = emit(b.finish(), &i);
        assert!(text.contains("load i8, ptr %v0"), "{text}");
        assert!(!text.contains("fpext"), "{text}");
    }
}
