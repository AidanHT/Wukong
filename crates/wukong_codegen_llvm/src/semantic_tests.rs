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
                    continue;
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
                b.build_void(Op::Store {
                    ptr: output,
                    value: r,
                });
                b.ret(None);
                let text = emit(b.finish(), &i);
                assert!(
                    text.contains(&format!(
                        "@llvm.{}.sat.v4i{bits}.v4{from_name}",
                        cast_name(kind)
                    )),
                    "{text}"
                );
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
fn lowp_fma_declines_until_promoted_ssa_is_available() {
    let mut i = Interner::new();
    for ty in [
        MirType::F16,
        MirType::BF16,
        MirType::Vec(Box::new(MirType::F16), 4),
    ] {
        let mut b = Builder::new(i.intern("lowp_fma"), ty.clone());
        let a = b.add_param(ty.clone());
        let r = b.build(ty, Op::Fma(a, a, a));
        b.ret(Some(r));
        let mut p = Program::new();
        p.funcs.push(b.finish());
        assert!(try_emit_llvm_ir(&p, &i)
            .unwrap_err()
            .contains("promoted f32 SSA"));
    }
}
