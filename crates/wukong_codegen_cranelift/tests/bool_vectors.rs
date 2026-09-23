use std::fmt::Write;
use std::process::Command;
use wukong_mir::{Builder, CastKind, MirType, Op, Program, ValueId};
use wukong_span::{Interner, Symbol};

fn gep(b: &mut Builder, ptr: ValueId, index: usize, elem: MirType) -> ValueId {
    let index = b.build(MirType::I64, Op::ConstInt(index as i128, MirType::I64));
    b.build(MirType::Ptr, Op::Gep { ptr, index, elem })
}

fn add_converter(
    p: &mut Program,
    it: &mut Interner,
    from: MirType,
    n: u32,
    signed: bool,
) -> Symbol {
    let name = it.intern(&format!("cast_{}_{n}_{signed}", from.display()));
    let mut b = Builder::new(name, MirType::Void);
    let input = b.add_param(MirType::Ptr);
    let output = b.add_param(MirType::Ptr);
    let choose = b.add_param(MirType::I1);
    let src = MirType::Vec(Box::new(from), n);
    let dst = MirType::Vec(Box::new(MirType::I1), n);
    let values = b.build(src.clone(), Op::Load(input, src));
    let kind = if signed {
        CastKind::FpToSi
    } else {
        CastKind::FpToUi
    };
    let cast = b.build(dst.clone(), Op::Cast(kind, values, dst.clone()));
    let inverse = b.build(dst.clone(), Op::Not(cast));
    let join = b.new_block();
    let chosen = b.block_param(join, dst.clone());
    // Both edges target the same block with different vector arguments.
    b.cond_br(choose, join, vec![cast], join, vec![inverse]);
    b.switch_to(join);
    let slot = b.alloca(MirType::Array(Box::new(MirType::I1), n));
    b.build_void(Op::Store {
        ptr: slot,
        value: chosen,
    });
    let loaded = b.build(dst.clone(), Op::Load(slot, dst));
    b.build_void(Op::Store {
        ptr: output,
        value: loaded,
    });
    for k in 0..n {
        let lane = b.build(MirType::I1, Op::ExtractLane(loaded, k));
        let byte = b.build(MirType::I8, Op::Cast(CastKind::ZExt, lane, MirType::I8));
        let ptr = gep(&mut b, output, (n + k) as usize, MirType::I8);
        b.build_void(Op::Store { ptr, value: byte });
    }
    b.ret(None);
    p.funcs.push(b.finish());
    name
}

// Expected bits are explicit i32/u32 saturation followed by parity. No interpreter/runtime
// result is used to derive them. Additional boundaries differ with source precision.
fn cases(f64_source: bool) -> Vec<(f64, u8, u8)> {
    let mut values = vec![
        (0.0, 0, 0),
        (-0.0, 0, 0),
        (1.9, 1, 1),
        (-1.9, 1, 0),
        (2.0, 0, 0),
        (127.0, 1, 1),
        (128.0, 0, 0),
        (129.0, 1, 1),
        (255.0, 1, 1),
        (256.0, 0, 0),
        (257.0, 1, 1),
        (-128.0, 0, 0),
        (-129.0, 1, 0),
        (65536.0, 0, 0),
        (2147483648.0, 1, 0),
        (4294967296.0, 1, 1),
        (-2147483648.0, 0, 0),
        (f64::NAN, 0, 0),
        (-f64::NAN, 0, 0),
        (f64::INFINITY, 1, 1),
        (f64::NEG_INFINITY, 0, 0),
    ];
    if f64_source {
        values.extend([
            (2147483647.0, 1, 1),
            (2147483649.0, 1, 1),
            (4294967294.0, 1, 0),
            (4294967295.0, 1, 1),
            (-2147483647.0, 1, 0),
            (-2147483649.0, 0, 0),
            (f64::from_bits(2147483648.0f64.to_bits() - 1), 1, 1),
            (f64::from_bits(2147483648.0f64.to_bits() + 1), 1, 0),
            (f64::from_bits(4294967296.0f64.to_bits() - 1), 1, 1),
            (f64::from_bits(4294967296.0f64.to_bits() + 1), 1, 1),
        ]);
    } else {
        values.extend([
            (f32::from_bits(2147483648.0f32.to_bits() - 1) as f64, 0, 0),
            (f32::from_bits(2147483648.0f32.to_bits() + 1) as f64, 1, 0),
            (f32::from_bits(4294967296.0f32.to_bits() - 1) as f64, 1, 0),
            (f32::from_bits(4294967296.0f32.to_bits() + 1) as f64, 1, 1),
        ]);
    }
    values
}

#[test]
fn compact_cast_lanes_and_storage_execute_in_jit_and_object() {
    let mut p = Program::new();
    let mut it = Interner::new();
    let mut converters = Vec::new();
    for from in [MirType::F32, MirType::F64] {
        for n in [2, 4, 16] {
            for signed in [true, false] {
                let name = add_converter(&mut p, &mut it, from.clone(), n, signed);
                converters.push((name, from.clone(), n as usize, signed));
            }
        }
    }
    // A Rust harness links and executes the emitted object as well as the in-process JIT.
    // Its signature only contains pointers and a scalar boolean; no vector ABI is assumed.
    let dir = std::env::temp_dir().join(format!("wukong-bool-vectors-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut harness = String::from("extern crate wukong_runtime;\n");
    // MSVC requires even unused object imports to resolve. Real kernel symbols come from
    // this test build's runtime archive; these host shims must never execute in a cast fixture.
    for symbol in [
        "wukong_rt_print_i64",
        "wukong_rt_print_u64",
        "wukong_rt_print_f64",
        "wukong_rt_print_str",
        "wukong_rt_assert",
        "wukong_rt_assert_failed",
        "wukong_rt_fmod_f32",
        "wukong_rt_fmod_f64",
    ] {
        writeln!(
            harness,
            "#[no_mangle] extern \"C\" fn {symbol}() -> ! {{ std::process::abort() }}"
        )
        .unwrap();
    }
    harness.push_str(
        "fn main() { std::hint::black_box(wukong_runtime::wukong_rt_alloc as usize); unsafe {\n",
    );
    for (name, from, n, signed) in &converters {
        let name = it.resolve(*name);
        let ty = from.display();
        writeln!(
            harness,
            "{{ extern \"C\" {{ fn {name}(x: *const {ty}, out: *mut u8, choose: u8); }}"
        )
        .unwrap();
        let data = cases(*from == MirType::F64);
        let mut inputs = String::new();
        let mut expected = Vec::new();
        for &(x, s, u) in &data {
            let repr = if *from == MirType::F64 {
                x.to_bits()
            } else {
                (x as f32).to_bits() as u64
            };
            write!(inputs, "{ty}::from_bits({repr}),").unwrap();
            expected.push(if *signed { s } else { u });
        }
        writeln!(
            harness,
            "let data = [{inputs}]; let expected: [u8; {}] = {expected:?};",
            data.len()
        )
        .unwrap();
        writeln!(
            harness,
            "for start in 0..data.len() {{ for choose in [0u8, 1] {{
            let input: [{ty}; {n}] = std::array::from_fn(|k| data[(start+k)%data.len()]);
            let mut out = [0xa5u8; {}];
            {name}(input.as_ptr(), out.as_mut_ptr().add(1), choose);
            assert_eq!(out[0], 0xa5); assert_eq!(out[{}], 0xa5);
            for k in 0..{n} {{ let want = expected[(start+k)%data.len()] ^ (1-choose);
                assert_eq!(out[k+1], want, \"{name}: stored lane {{k}}\");
                assert_eq!(out[k+1+{n}], want, \"{name}: extracted lane {{k}}\");
            }}
        }} }} }}",
            2 * n + 2,
            2 * n + 1
        )
        .unwrap();
    }
    harness.push_str("} }\n");
    let harness_path = dir.join("harness.rs");
    std::fs::write(&harness_path, harness).unwrap();
    let deps = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    let runtime = std::fs::read_dir(&deps)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("libwukong_runtime-")
                && path.extension().is_some_and(|ext| ext == "rlib")
        })
        .max_by_key(|path| path.metadata().unwrap().modified().unwrap())
        .expect("the current test build must provide a runtime archive");
    for opt in [0, 2] {
        let mut optimized = p.clone();
        wukong_opt::optimize(&mut optimized, opt);
        let errors = wukong_mir::verify::verify_program(&optimized);
        assert!(errors.is_empty(), "O{opt}: {errors:?}");
        let module = wukong_codegen_cranelift::jit_module(&optimized, &it).unwrap();
        for (name, from, n, signed) in &converters {
            let data = cases(*from == MirType::F64);
            for start in 0..data.len() {
                for choose in [0u8, 1] {
                    let mut out = vec![0xa5u8; 2 * n + 2];
                    unsafe {
                        if *from == MirType::F32 {
                            let input: Vec<f32> = (0..*n)
                                .map(|k| data[(start + k) % data.len()].0 as f32)
                                .collect();
                            let call: extern "C" fn(*const f32, *mut u8, u8) =
                                std::mem::transmute(module.func_ptr(*name).unwrap());
                            call(input.as_ptr(), out.as_mut_ptr().add(1), choose);
                        } else {
                            let input: Vec<f64> =
                                (0..*n).map(|k| data[(start + k) % data.len()].0).collect();
                            let call: extern "C" fn(*const f64, *mut u8, u8) =
                                std::mem::transmute(module.func_ptr(*name).unwrap());
                            call(input.as_ptr(), out.as_mut_ptr().add(1), choose);
                        }
                    }
                    assert_eq!(out[0], 0xa5);
                    assert_eq!(
                        out[2 * n + 1],
                        0xa5,
                        "compact storage must not overwrite the guard"
                    );
                    for k in 0..*n {
                        let (_, s, u) = data[(start + k) % data.len()];
                        let want = (if *signed { s } else { u }) ^ (1 - choose);
                        assert_eq!(
                            out[k + 1],
                            want,
                            "{:?}/{n}/{signed}/O{opt}/start{start}/lane{k}",
                            from
                        );
                        assert_eq!(
                            out[k + 1 + n],
                            want,
                            "extracted lane must match stored byte"
                        );
                    }
                }
            }
        }
        let object = dir.join(format!("casts_O{opt}.o"));
        std::fs::write(
            &object,
            wukong_codegen_cranelift::emit_object(&optimized, &it).unwrap(),
        )
        .unwrap();
        let executable = dir.join(format!("casts_O{opt}{}", std::env::consts::EXE_SUFFIX));
        let linked = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
            .arg(&harness_path)
            .args(["--edition=2021", "-C", "panic=abort", "-C"])
            .arg(format!("link-arg={}", object.display()))
            .arg("--extern")
            .arg(format!("wukong_runtime={}", runtime.display()))
            .arg("-L")
            .arg(format!("dependency={}", deps.display()))
            .arg("-o")
            .arg(&executable)
            .output()
            .expect("Rust toolchain is required to execute the object regression");
        assert!(
            linked.status.success(),
            "{}",
            String::from_utf8_lossy(&linked.stderr)
        );
        let ran = Command::new(&executable).output().unwrap();
        assert!(
            ran.status.success(),
            "O{opt}: {}",
            String::from_utf8_lossy(&ran.stderr)
        );
    }
}

#[test]
fn compact_cast_connected_vector_abi_declines_before_codegen() {
    for vector_return in [false, true] {
        let mut it = Interner::new();
        let source = MirType::Vec(Box::new(MirType::F32), 4);
        let target = MirType::Vec(Box::new(MirType::I1), 4);
        let mut b = Builder::new(
            it.intern("unsupported_abi"),
            if vector_return {
                target.clone()
            } else {
                MirType::Void
            },
        );
        let input = b.add_param(source.clone());
        let cast = b.build(target.clone(), Op::Cast(CastKind::FpToSi, input, target));
        b.ret(vector_return.then_some(cast));
        let mut p = Program::new();
        p.funcs.push(b.finish());
        assert!(wukong_mir::verify::verify_program(&p).is_empty());
        for error in [
            wukong_codegen_cranelift::jit_module(&p, &it).err().unwrap(),
            wukong_codegen_cranelift::emit_object(&p, &it).unwrap_err(),
        ] {
            assert!(
                error.contains("vector function parameters or returns require a vector ABI"),
                "{error}"
            );
        }
    }
}

#[test]
fn compact_cast_rejects_unbounded_shapes_and_lane_repacking() {
    for (n, repack, message) in [
        (u32::MAX, false, "supports 1 through 16 lanes"),
        (4, true, "change vector lane count require repacking"),
    ] {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("boundary"), MirType::I1);
        let input = b.add_param(MirType::Ptr);
        let source = MirType::Vec(Box::new(MirType::F32), n);
        let target = MirType::Vec(Box::new(MirType::I1), n);
        let values = b.build(source.clone(), Op::Load(input, source));
        let cast = b.build(target.clone(), Op::Cast(CastKind::FpToSi, values, target));
        if repack {
            let wide = MirType::Vec(Box::new(MirType::I64), 2);
            let _ = b.build(wide.clone(), Op::Cast(CastKind::Bitcast, values, wide));
        }
        let bit = b.build(MirType::I1, Op::ExtractLane(cast, 0));
        b.ret(Some(bit));
        let mut p = Program::new();
        p.funcs.push(b.finish());
        let errors = wukong_mir::verify::verify_program(&p);
        assert!(errors.is_empty(), "valid MIR boundary fixture: {errors:?}");
        for error in [
            wukong_codegen_cranelift::jit_module(&p, &it).err().unwrap(),
            wukong_codegen_cranelift::emit_object(&p, &it).unwrap_err(),
        ] {
            assert!(error.contains(message), "{error}");
        }
    }
}

#[test]
fn compact_cast_preserves_fused_math_and_boolean_select_masks() {
    for (from, input, cancel) in [
        // (2^12+1)^2 - (2^24+2^13) = 1; separately rounded f32 multiply gives 0.
        (MirType::F32, 4097.0, -16785408.0),
        // (2^27+1)^2 - (2^54+2^28) = 1; separately rounded f64 multiply gives 0.
        (MirType::F64, 134217729.0, -18014398777917440.0),
    ] {
        let mut it = Interner::new();
        let name = it.intern("fused_masks");
        let mut b = Builder::new(name, MirType::Void);
        let x = b.add_param(from.clone());
        let output = b.add_param(MirType::Ptr);
        let vector = MirType::Vec(Box::new(from.clone()), 4);
        let x = b.build(vector.clone(), Op::Splat(x));
        let c = b.build(from.clone(), Op::ConstFloat(cancel, from.clone()));
        let c = b.build(vector.clone(), Op::Splat(c));
        let fma = b.build(vector.clone(), Op::Fma(x, x, c));
        let bools = MirType::Vec(Box::new(MirType::I1), 4);
        let mask = b.build(bools.clone(), Op::Cast(CastKind::FpToSi, fma, bools));
        let one = b.build(from.clone(), Op::ConstFloat(1.0, from.clone()));
        let one = b.build(vector.clone(), Op::Splat(one));
        let zero = b.build(from.clone(), Op::ConstFloat(0.0, from.clone()));
        let zero = b.build(vector.clone(), Op::Splat(zero));
        let selected = b.build(vector.clone(), Op::Select(mask, one, zero));
        let mask_type = MirType::Vec(Box::new(MirType::I32), 4);
        let compared = b.build(mask_type, Op::Cmp(wukong_mir::CmpOp::Foeq, selected, one));
        b.build_void(Op::Store {
            ptr: output,
            value: compared,
        });
        b.ret(None);
        let mut p = Program::new();
        p.funcs.push(b.finish());
        for opt in [0, 2] {
            let mut optimized = p.clone();
            wukong_opt::optimize(&mut optimized, opt);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            let module = wukong_codegen_cranelift::jit_module(&optimized, &it).unwrap();
            let mut actual = [i32::MIN; 4];
            unsafe {
                if from == MirType::F32 {
                    let call: extern "C" fn(f32, *mut i32) =
                        std::mem::transmute(module.func_ptr(name).unwrap());
                    call(input as f32, actual.as_mut_ptr());
                } else {
                    let call: extern "C" fn(f64, *mut i32) =
                        std::mem::transmute(module.func_ptr(name).unwrap());
                    call(input, actual.as_mut_ptr());
                }
            }
            assert_eq!(
                actual, [1; 4],
                "single-round FMA and 0/1 comparison masks: {from:?}/O{opt}"
            );
            assert!(!wukong_codegen_cranelift::emit_object(&optimized, &it)
                .unwrap()
                .is_empty());
        }
    }
}

#[test]
fn compact_cast_boolean_arithmetic_uses_one_bit_width() {
    use wukong_mir::BinOp::*;
    let cases = [
        (Add, [0, 1, 1, 0]),
        (Sub, [0, 1, 1, 0]),
        (Mul, [0, 0, 0, 1]),
        (SDiv, [0, 0, 0, 1]),
        (UDiv, [0, 0, 0, 1]),
        (SRem, [0, 0, 0, 0]),
        (URem, [0, 0, 0, 0]),
        (And, [0, 0, 0, 1]),
        (Or, [0, 1, 1, 1]),
        (Xor, [0, 1, 1, 0]),
        (Shl, [0, 1, 0, 1]),
        (LShr, [0, 1, 0, 1]),
        (AShr, [0, 1, 0, 1]),
    ];
    let mut it = Interner::new();
    let name = it.intern("boolean_ops");
    let mut b = Builder::new(name, MirType::Void);
    let xptr = b.add_param(MirType::Ptr);
    let yptr = b.add_param(MirType::Ptr);
    let output = b.add_param(MirType::Ptr);
    let vector = MirType::Vec(Box::new(MirType::F32), 4);
    let bools = MirType::Vec(Box::new(MirType::I1), 4);
    let x = b.build(vector.clone(), Op::Load(xptr, vector.clone()));
    let y = b.build(vector.clone(), Op::Load(yptr, vector));
    let x = b.build(bools.clone(), Op::Cast(CastKind::FpToSi, x, bools.clone()));
    let y = b.build(bools.clone(), Op::Cast(CastKind::FpToSi, y, bools.clone()));
    for (k, (op, _)) in cases.iter().enumerate() {
        let result = b.build(bools.clone(), Op::Bin(*op, x, y));
        let ptr = gep(&mut b, output, k * 4, MirType::I1);
        b.build_void(Op::Store { ptr, value: result });
    }
    let negated = b.build(bools, Op::Neg(x));
    let ptr = gep(&mut b, output, cases.len() * 4, MirType::I1);
    b.build_void(Op::Store {
        ptr,
        value: negated,
    });
    b.ret(None);
    let mut p = Program::new();
    p.funcs.push(b.finish());
    let expected: Vec<u8> = cases
        .iter()
        .flat_map(|(_, lanes)| *lanes)
        .chain([0, 1, 0, 1])
        .collect();
    for opt in [0, 2] {
        let mut optimized = p.clone();
        wukong_opt::optimize(&mut optimized, opt);
        assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
        let module = wukong_codegen_cranelift::jit_module(&optimized, &it).unwrap();
        let mut actual = vec![0xa5u8; expected.len()];
        unsafe {
            let call: extern "C" fn(*const f32, *const f32, *mut u8) =
                std::mem::transmute(module.func_ptr(name).unwrap());
            call(
                [0.0, 129.0, 0.0, 129.0].as_ptr(),
                [0.0, 0.0, 129.0, 129.0].as_ptr(),
                actual.as_mut_ptr(),
            );
        }
        assert_eq!(actual, expected, "boolean lane truth tables at O{opt}");
        assert!(!wukong_codegen_cranelift::emit_object(&optimized, &it)
            .unwrap()
            .is_empty());
    }
}

#[test]
fn scalar_boolean_shifts_mask_counts_at_declared_width() {
    let mut p = Program::new();
    let mut it = Interner::new();
    let main = it.intern("main");
    let print = it.intern("print");
    let mut entry = Builder::new(main, MirType::I32);
    let mut names = Vec::new();
    for op in [
        wukong_mir::BinOp::Shl,
        wukong_mir::BinOp::LShr,
        wukong_mir::BinOp::AShr,
    ] {
        let name = it.intern(op.name());
        names.push(name);
        let mut b = Builder::new(name, MirType::I1);
        let x = b.add_param(MirType::I1);
        let y = b.add_param(MirType::I1);
        let shifted = b.build(MirType::I1, Op::Bin(op, x, y));
        b.ret(Some(shifted));
        p.funcs.push(b.finish());
        for x in [0, 1] {
            for y in [0, 1] {
                let x = entry.build(MirType::I1, Op::ConstInt(x, MirType::I1));
                let y = entry.build(MirType::I1, Op::ConstInt(y, MirType::I1));
                let value = entry.build(
                    MirType::I1,
                    Op::Call {
                        func: name,
                        args: vec![x, y],
                    },
                );
                let value =
                    entry.build(MirType::I32, Op::Cast(CastKind::ZExt, value, MirType::I32));
                entry.build_void(Op::Call {
                    func: print,
                    args: vec![value],
                });
            }
        }
    }
    let zero = entry.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    entry.ret(Some(zero));
    p.funcs.push(entry.finish());
    let expected = (0, b"0\n0\n1\n1\n0\n0\n1\n1\n0\n0\n1\n1\n".to_vec());
    for opt in [0, 2] {
        let mut optimized = p.clone();
        wukong_opt::optimize(&mut optimized, opt);
        assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
        assert_eq!(
            wukong_interp::run_with_output(&optimized, main, &it).unwrap(),
            expected
        );
        assert_eq!(
            wukong_codegen_cranelift::jit_run(&optimized, main, &it).unwrap(),
            expected
        );
        // Keep the raw entrypoints in a separate library program: optimizing the executable
        // can inline and remove them after the constant calls in main have been folded.
        let mut library = p.clone();
        library.funcs.retain(|f| f.name != main);
        wukong_opt::optimize(&mut library, opt);
        let module = wukong_codegen_cranelift::jit_module(&library, &it).unwrap();
        for name in &names {
            let call: unsafe extern "C" fn(u8, u8) -> u8 =
                unsafe { std::mem::transmute(module.func_ptr(*name).unwrap()) };
            for x in [0, 1] {
                for y in [0, 1] {
                    assert_eq!(unsafe { call(x, y) }, x, "{name:?}({x}, {y}) at O{opt}");
                }
            }
        }
        assert!(!wukong_codegen_cranelift::emit_object(&optimized, &it)
            .unwrap()
            .is_empty());
    }
}
