use super::*;
use wukong_mir::{BinOp, Builder, CmpOp};

fn dispatcher(it: &mut Interner) -> (Program, Symbol) {
    let mut leaf = Builder::new(it.intern("leaf"), MirType::Void);
    let data = leaf.add_param(MirType::Ptr);
    let selector = leaf.add_param(MirType::I64);
    leaf.build_void(Op::Call {
        func: it.intern("print"),
        args: vec![selector],
    });
    let zero = leaf.build(MirType::I64, Op::ConstInt(0, MirType::I64));
    let valid = leaf.build(MirType::I1, Op::Cmp(CmpOp::Ne, selector, zero));
    leaf.build_void(Op::Call {
        func: it.intern("assert"),
        args: vec![valid],
    });
    let old = leaf.build(MirType::F32, Op::Load(data, MirType::F32));
    let one = leaf.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
    let next = leaf.build(MirType::F32, Op::Bin(BinOp::FAdd, old, one));
    leaf.build_void(Op::Store {
        ptr: data,
        value: next,
    });
    leaf.ret(None);
    let leaf = leaf.finish();
    let mut b = Builder::new(it.intern("dispatch"), MirType::Void);
    let table = b.add_param(MirType::Ptr);
    let selector = b.add_param(MirType::I64);
    let data = b.build(MirType::Ptr, Op::Load(table, MirType::Ptr));
    b.build_void(Op::Call {
        func: leaf.name,
        args: vec![data, selector],
    });
    // A failed nested assertion must stop this store too.
    let value = b.build(MirType::F32, Op::ConstFloat(42.0, MirType::F32));
    b.build_void(Op::Store { ptr: data, value });
    b.ret(None);
    let b = b.finish();
    let entry = b.name;
    let mut program = Program::new();
    program.funcs = vec![leaf, b];
    (program, entry)
}

#[test]
fn fixed_abi_rejects_wrong_parameter_and_return_signatures() {
    let mut it = Interner::new();
    for (params, ret) in [
        (vec![], MirType::Void),
        (vec![MirType::Ptr], MirType::Void),
        (vec![MirType::I64, MirType::Ptr], MirType::Void),
        (vec![MirType::Ptr, MirType::I64], MirType::I64),
    ] {
        let name = it.intern("wrong");
        let mut b = Builder::new(name, ret.clone());
        for ty in params {
            b.add_param(ty);
        }
        let result = (ret != MirType::Void).then(|| b.build(ret.clone(), Op::ConstInt(0, ret)));
        b.ret(result);
        let mut program = Program::new();
        program.funcs.push(b.finish());
        let error = jit_compile_buffer_kernel(&program, name, &it)
            .err()
            .unwrap();
        assert!(error.contains("signature (Ptr, I64) -> Void"), "{error}");
    }
}

#[test]
fn guarded_calls_own_code_reset_state_and_propagate_nested_assertions() {
    let kernel = {
        let mut it = Interner::new();
        let (program, entry) = dispatcher(&mut it);
        assert!(jit_compile(&program, entry, &it)
            .err()
            .unwrap()
            .contains("no parameters"));
        // The raw API compiles it as well, with raw-mode failure semantics (a failed assertion
        // aborts the process), so only the guarded handle below is run.
        assert!(jit_module(&program, &it).is_ok());
        jit_compile_buffer_kernel(&program, entry, &it).unwrap()
    }; // MIR, interner and compilation temporaries gone; the handle alone retains the code.
    let mut data = [5.0f32];
    for selector in [1, 0, 2, 0, 3] {
        let before = data[0].to_bits();
        let pointers = [data.as_mut_ptr()];
        // SAFETY: dispatch reads one table slot, whose single writable f32 remains live.
        let result = unsafe { kernel.run(&pointers, selector) };
        if selector == 0 {
            assert_eq!(result, Err("assertion failed".to_string()));
            assert_eq!(
                data[0].to_bits(),
                before,
                "writes must stop after the assertion"
            );
        } else {
            assert_eq!(result.unwrap(), format!("{selector}\n").as_bytes());
            assert_eq!(data[0].to_bits(), 42.0f32.to_bits());
        }
    }
}
