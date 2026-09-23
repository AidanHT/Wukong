use wukong_mir::{Builder, CastKind, MirType, Op, Program};
use wukong_span::Interner;

/// Runtime calls use signed i64 dimensions even though native pointers and loop indices are
/// unsigned. Exercise the actual JIT ABI, including the parallel entry, without valid input
/// buffers: a non-positive dimension must return before any memory access or scheduling.
#[test]
fn scatter_nonpositive_extents_match_interpreter() {
    let mut interner = Interner::new();
    let main = interner.intern("main");
    let mut b = Builder::new(main, MirType::I64);
    let output = b.alloca(MirType::I64);
    let sentinel = b.build(MirType::I64, Op::ConstInt(42, MirType::I64));
    b.build_void(Op::Store {
        ptr: output,
        value: sentinel,
    });
    let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
    let null = b.build(
        MirType::Ptr,
        Op::Cast(CastKind::IntToPtr, zero, MirType::Ptr),
    );
    for name in ["wukong_scatter_add_f32", "wukong_scatter_add_f32_parallel"] {
        let func = interner.intern(name);
        for axis in 0..3 {
            for extent in [0, -1, i64::MIN] {
                let mut dims = [i64::MAX; 3];
                dims[axis] = extent;
                let mut args = vec![output, null, null];
                for dim in dims {
                    args.push(b.build(MirType::I64, Op::ConstInt(dim as i128, MirType::I64)));
                }
                b.build_void(Op::Call { func, args });
            }
        }
    }
    let result = b.build(MirType::I64, Op::Load(output, MirType::I64));
    b.ret(Some(result));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    for level in [0, 2] {
        let mut optimized = program.clone();
        wukong_opt::optimize(&mut optimized, level);
        let errors = wukong_mir::verify::verify_program(&optimized);
        assert!(errors.is_empty(), "MIR at O{level}: {errors:?}");
        assert_eq!(
            wukong_interp::run_with_output(&optimized, main, &interner).unwrap(),
            (42, vec![]),
            "interpreter at O{level}"
        );
        assert_eq!(
            wukong_codegen_cranelift::jit_run(&optimized, main, &interner).unwrap(),
            (42, vec![]),
            "native at O{level}"
        );
    }
}
