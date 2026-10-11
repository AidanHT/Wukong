use super::*;
use std::collections::BTreeSet;
use wukong_mir::{Builder, RoundMode};

fn module(f: Function, interner: &Interner) -> String {
    let mut p = Program::new();
    p.funcs.push(f);
    let text = try_emit_llvm_ir(&p, interner).expect("representable MIR");
    // The separate LLVM verifier command consumes these exact regression modules. No external
    // executable is required to run the portable Rust assertions.
    if let Some(dir) = std::env::var_os("WUKONG_LLVM_TEST_OUTPUT") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            std::path::Path::new(&dir).join(format!("{}.ll", interner.resolve(p.funcs[0].name))),
            &text,
        )
        .unwrap();
    }
    text
}

#[test]
fn exact_float_literals_all_widths() {
    assert_eq!(fmt_float(0.1, &MirType::F32), "0x3FB99999A0000000");
    assert_eq!(fmt_float(-0.0, &MirType::F64), "0x8000000000000000");
    assert_eq!(
        fmt_float(f64::INFINITY, &MirType::F32),
        "0x7FF0000000000000"
    );
    assert_eq!(
        fmt_float(f64::NEG_INFINITY, &MirType::F32),
        "0xFFF0000000000000"
    );
    // A half/bfloat constant is its grid-rounded value in a `float` register (the interpreter's
    // `ConstFloat` rounding): 0.1 rounds to 0.0999755859375 on the f16 grid and to 0.10009765625
    // on the bf16 grid (hand-derived: f32 0x3DCCCCCD rounds to f16 0x2E66 and to bf16 0x3DCD).
    assert_eq!(fmt_float(1.0, &MirType::F16), "0x3FF0000000000000");
    assert_eq!(fmt_float(0.1, &MirType::F16), "0x3FB9980000000000");
    assert_eq!(fmt_float(0.1, &MirType::BF16), "0x3FB9A00000000000");
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("float_literals"), MirType::Void);
    for ty in [MirType::F16, MirType::BF16, MirType::F32, MirType::F64] {
        let ptr = b.alloca(ty.clone());
        for v in [
            0.1,
            1e-5,
            1e-40,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let value = b.build(ty.clone(), Op::ConstFloat(v, ty.clone()));
            b.build_void(Op::Store { ptr, value });
        }
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(
        !text.contains("NaN") && !text.contains(" inf") && !text.contains(" -inf"),
        "{text}"
    );
}

#[test]
fn vector_neg_not_and_identity_casts() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("vector_ops"), MirType::Void);
    let vi = MirType::Vec(Box::new(MirType::I32), 4);
    let vf = MirType::Vec(Box::new(MirType::F32), 4);
    let a = b.add_param(vi.clone());
    let x = b.add_param(vf.clone());
    b.build(vi.clone(), Op::Neg(a));
    b.build(vi, Op::Not(a));
    b.build(vf, Op::Neg(x));
    for ty in [MirType::F16, MirType::BF16, MirType::Ptr, MirType::I32] {
        let p = b.add_param(ty.clone());
        b.build(ty.clone(), Op::Cast(CastKind::FpTrunc, p, ty));
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(text.contains("sub <4 x i32> zeroinitializer"), "{text}");
    assert!(text.contains("<i32 -1, i32 -1, i32 -1, i32 -1>"), "{text}");
    assert!(text.contains("fneg <4 x float>"), "{text}");
    // Pointer and integer same-type casts are identities. A same-type f16/bf16 `fptrunc` is not:
    // the interpreter rounds the f32 register onto the grid, so it is a narrowing round trip.
    // Values: a=%v0, x=%v1, the three ops %v2..%v4, then each param/cast pair from %v5.
    assert!(text.contains("fptrunc float %v5 to half"), "{text}");
    // bf16 rounds with the runtime's own integer arithmetic (no `__truncsfbf2` libcall).
    assert!(
        text.contains("%t2.bits = bitcast float %v7 to i32"),
        "{text}"
    );
    assert!(text.contains("%t3 = fpext bfloat %t2 to float"), "{text}");
    assert!(
        !text.contains("fptrunc half") && !text.contains("fptrunc bfloat"),
        "{text}"
    );
    assert_eq!(text.matches("select i1 true").count(), 4);
}

#[test]
fn math_calls_have_declarations_and_result_bindings() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("math_calls"), MirType::F32);
    let p = b.add_param(MirType::F32);
    let sqrt = b.build(MirType::F32, Op::Sqrt(p));
    let round = b.build(MirType::F32, Op::Round(RoundMode::Nearest, sqrt));
    b.ret(Some(round));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("declare float @llvm.sqrt.f32(float)"),
        "{text}"
    );
    assert!(
        text.contains("declare float @llvm.roundeven.f32(float)"),
        "{text}"
    );
    assert!(text.contains("%v1 = call float @llvm.sqrt.f32"), "{text}");
    assert!(
        text.contains("%v2 = call float @llvm.roundeven.f32(float %v1)"),
        "{text}"
    );
}

#[test]
fn integer_bit_ops_call_the_llvm_bit_intrinsics() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("bit_ops"), MirType::I32);
    let p = b.add_param(MirType::I32);
    let pop = b.build(MirType::I32, Op::IntUnary(IntUnOp::Popcnt, p));
    let lz = b.build(MirType::I32, Op::IntUnary(IntUnOp::Clz, pop));
    let tz = b.build(MirType::I32, Op::IntUnary(IntUnOp::Ctz, lz));
    let sw = b.build(MirType::I32, Op::IntUnary(IntUnOp::Bswap, tz));
    b.ret(Some(sw));
    let text = module(b.finish(), &i);
    for decl in [
        "declare i32 @llvm.ctpop.i32(i32)",
        "declare i32 @llvm.ctlz.i32(i32, i1)",
        "declare i32 @llvm.cttz.i32(i32, i1)",
        "declare i32 @llvm.bswap.i32(i32)",
    ] {
        assert!(text.contains(decl), "missing `{decl}`:\n{text}");
    }
    // `is_zero_poison` is false: a zero operand counts the full width, as the MIR op defines.
    for call in [
        "%v1 = call i32 @llvm.ctpop.i32(i32 %v0)",
        "%v2 = call i32 @llvm.ctlz.i32(i32 %v1, i1 false)",
        "%v3 = call i32 @llvm.cttz.i32(i32 %v2, i1 false)",
        "%v4 = call i32 @llvm.bswap.i32(i32 %v3)",
    ] {
        assert!(text.contains(call), "missing `{call}`:\n{text}");
    }
}

#[test]
fn external_declarations_and_conflicting_abis() {
    let mut i = Interner::new();
    let print = i.intern("foreign_sink");
    let mut b = Builder::new(i.intern("external_call"), MirType::Void);
    let a = b.add_param(MirType::I32);
    b.build_void(Op::Call {
        func: print,
        args: vec![a],
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(text.contains("declare void @foreign_sink(i32)"), "{text}");
    let mut p = Program::new();
    let mut b = Builder::new(i.intern("conflicting_calls"), MirType::Void);
    let a = b.add_param(MirType::I32);
    let x = b.add_param(MirType::F32);
    b.build_void(Op::Call {
        func: print,
        args: vec![a],
    });
    b.build_void(Op::Call {
        func: print,
        args: vec![x],
    });
    b.ret(None);
    p.funcs.push(b.finish());
    let err = try_emit_llvm_ir(&p, &i).unwrap_err();
    assert!(
        err.contains("incompatible call signatures") && err.contains("foreign_sink"),
        "{err}"
    );
    assert!(LlvmBackend.compile(&p, p.funcs[0].name, &i).is_err());
}

#[test]
fn duplicate_successor_edges_keep_distinct_phi_values() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("duplicate_edges"), MirType::I32);
    let cond = b.add_param(MirType::I1);
    let a = b.add_param(MirType::I32);
    let z = b.add_param(MirType::I32);
    let target = b.new_block();
    let phi = b.block_param(target, MirType::I32);
    b.set_term(Terminator::CondBr {
        cond,
        then_blk: target,
        then_args: vec![a],
        else_blk: target,
        else_args: vec![z],
    });
    b.switch_to(target);
    b.ret(Some(phi));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("phi i32 [ %v1, %bb0 ], [ %v2, %bb0.else ]"),
        "{text}"
    );
    assert!(text.contains("bb0.else:\n  br label %bb1"), "{text}");
}

#[test]
fn integer_guards_and_unordered_not_equal() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("integer_guards"), MirType::Void);
    for ty in [
        MirType::I8,
        MirType::I16,
        MirType::I32,
        MirType::I64,
        MirType::Vec(Box::new(MirType::I32), 4),
    ] {
        let a = b.add_param(ty.clone());
        let d = b.add_param(ty.clone());
        for op in [
            BinOp::SDiv,
            BinOp::UDiv,
            BinOp::SRem,
            BinOp::URem,
            BinOp::Shl,
            BinOp::LShr,
            BinOp::AShr,
        ] {
            b.build(ty.clone(), Op::Bin(op, a, d));
        }
    }
    let x = b.add_param(MirType::F32);
    b.build(MirType::I1, Op::Cmp(CmpOp::Fone, x, x));
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(text.contains("fcmp une float"), "{text}");
    assert_eq!(text.matches(".safe = select").count(), 20);
    assert_eq!(text.matches(".shift = and").count(), 15);
    assert_eq!(text.matches(".wrapped = select").count(), 5);
}

#[test]
fn scalar_guard_execution_modules() {
    let mut i = Interner::new();
    for (name, op) in [
        ("sdiv", BinOp::SDiv),
        ("udiv", BinOp::UDiv),
        ("srem", BinOp::SRem),
        ("urem", BinOp::URem),
        ("shl", BinOp::Shl),
        ("lshr", BinOp::LShr),
        ("ashr", BinOp::AShr),
    ] {
        let mut b = Builder::new(i.intern(name), MirType::I32);
        let a = b.add_param(MirType::I32);
        let d = b.add_param(MirType::I32);
        let result = b.build(MirType::I32, Op::Bin(op, a, d));
        b.ret(Some(result));
        let text = module(b.finish(), &i);
        assert!(text.contains("ret i32 %v2"), "{text}");
    }
    let mut b = Builder::new(i.intern("unordered_ne"), MirType::I1);
    let a = b.add_param(MirType::F64);
    let d = b.add_param(MirType::F64);
    let result = b.build(MirType::I1, Op::Cmp(CmpOp::Fone, a, d));
    b.ret(Some(result));
    let text = module(b.finish(), &i);
    assert!(text.contains("fcmp une double"), "{text}");
}

#[test]
fn prints_and_asserts_use_the_aot_runtime_abi() {
    let mut i = Interner::new();
    let print = i.intern("print");
    let mut b = Builder::new(i.intern("runtime_prints"), MirType::Void);
    for ty in [
        MirType::I8,
        MirType::I32,
        MirType::F32,
        MirType::F64,
        MirType::F16,
        MirType::BF16,
        MirType::I1,
        MirType::I64,
        MirType::Ptr,
    ] {
        let a = b.add_param(ty);
        b.build_void(Op::Call {
            func: print,
            args: vec![a],
        });
    }
    let unsigned = b.add_param(MirType::I64);
    b.build_void(Op::Call {
        func: i.intern("println_u"),
        args: vec![unsigned],
    });
    let text_ptr = b.add_param(MirType::Ptr);
    b.build_void(Op::Call {
        func: i.intern("print_str"),
        args: vec![text_ptr],
    });
    let flag = b.add_param(MirType::F32);
    b.build_void(Op::Call {
        func: i.intern("assert"),
        args: vec![flag],
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    for decl in [
        "declare void @wukong_rt_print_i64(i64)",
        "declare void @wukong_rt_print_f64(double)",
        "declare void @wukong_rt_print_f32(float)",
        "declare void @wukong_rt_print_u64(i64)",
        "declare void @wukong_rt_print_str(ptr)",
        "declare void @wukong_rt_assert(i64)",
    ] {
        assert!(text.contains(decl), "{decl}:\n{text}");
    }
    // Exactly Cranelift's routing: ints sign-extend, booleans zero-extend, an f64 prints as a
    // double, an f32 (and an f16/bf16, which live in f32 registers) prints at f32 precision through
    // `wukong_rt_print_f32` with no widening, pointers print their address.
    for line in [
        "sext i8 %v0 to i64",
        "sext i32 %v1 to i64",
        "call void @wukong_rt_print_f32(float %v2)",
        "call void @wukong_rt_print_f64(double %v3)",
        "call void @wukong_rt_print_f32(float %v4)",
        "call void @wukong_rt_print_f32(float %v5)",
        "zext i1 %v6 to i64",
        "call void @wukong_rt_print_i64(i64 %v7)",
        "ptrtoint ptr %v8 to i64",
        "call void @wukong_rt_print_u64(i64 %v9)",
        "call void @wukong_rt_print_str(ptr %v10)",
        "fcmp une float %v11, 0.0",
    ] {
        assert!(text.contains(line), "{line}:\n{text}");
    }
    assert!(
        !text.contains("@print(") && !text.contains("__wukong_print"),
        "{text}"
    );
}

#[test]
fn runtime_kernel_integers_widen_to_the_c_abi() {
    let mut i = Interner::new();
    let kernel = i.intern("wukong_probe_kernel");
    let mut b = Builder::new(i.intern("kernel_abi"), MirType::Void);
    let p = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I32);
    let alpha = b.add_param(MirType::F32);
    let flag = b.add_param(MirType::I1);
    b.build_void(Op::Call {
        func: kernel,
        args: vec![p, n, alpha, flag],
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(
        text.contains("declare void @wukong_probe_kernel(ptr, i64, float, i64)"),
        "{text}"
    );
    assert!(text.contains("%t0 = sext i32 %v1 to i64"), "{text}");
    assert!(text.contains("%t1 = zext i1 %v3 to i64"), "{text}");
    assert!(
        text.contains("call void @wukong_probe_kernel(ptr %v0, i64 %t0, float %v2, i64 %t1)"),
        "{text}"
    );
}

#[test]
fn half_values_compute_in_f32_registers_and_round_at_memory() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("half_registers"), MirType::BF16);
    let x = b.add_param(MirType::BF16);
    let y = b.add_param(MirType::BF16);
    let sum = b.build(MirType::BF16, Op::Bin(BinOp::FAdd, x, y));
    let slot = b.alloca(MirType::BF16);
    b.build_void(Op::Store {
        ptr: slot,
        value: sum,
    });
    let back = b.build(MirType::BF16, Op::Load(slot, MirType::BF16));
    let widened = b.build(MirType::F32, Op::Cast(CastKind::FpExt, back, MirType::F32));
    let narrowed = b.build(
        MirType::BF16,
        Op::Cast(CastKind::FpTrunc, widened, MirType::BF16),
    );
    b.ret(Some(narrowed));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("define float @half_registers(float %v0, float %v1)"),
        "{text}"
    );
    // The add is f32 arithmetic on f32 registers: no intermediate rounding to the bf16 grid.
    assert!(text.contains("%v2 = fadd float %v0, %v1"), "{text}");
    assert!(!text.contains("fadd bfloat"), "{text}");
    assert!(text.contains("alloca bfloat"), "{text}");
    for line in [
        "%t0.bits = bitcast float %v2 to i32",
        "%t0.nan = fcmp uno float %v2, %v2",
        "%t0.top = lshr i32 %t0.bits, 16",
        "%t0.lsb = and i32 %t0.top, 1",
        "%t0.bias = add i32 %t0.lsb, 32767",
        "%t0.sum = add i32 %t0.bits, %t0.bias",
        "%t0.rne = lshr i32 %t0.sum, 16",
        "%t0.qnan = or i32 %t0.top, 64",
        "%t0.pick = select i1 %t0.nan, i32 %t0.qnan, i32 %t0.rne",
        "%t0.half = trunc i32 %t0.pick to i16",
        "%t0 = bitcast i16 %t0.half to bfloat",
    ] {
        assert!(text.contains(line), "{line}:\n{text}");
    }
    assert!(
        !text
            .lines()
            .any(|l| l.contains("fptrunc") && l.contains("bfloat")),
        "{text}"
    );
    assert!(text.contains("store bfloat %t0"), "{text}");
    assert!(text.contains("%v4.mem = load bfloat"), "{text}");
    assert!(
        text.contains("%v4 = fpext bfloat %v4.mem to float"),
        "{text}"
    );
    // Widening a bf16 value snaps the register onto the grid first; narrowing rounds again.
    assert!(text.contains("bitcast float %v4 to i32"), "{text}");
    assert!(text.contains("bitcast float %v5 to i32"), "{text}");
    assert!(text.contains("ret float %v6"), "{text}");
}

#[test]
fn booleans_are_bytes_in_memory() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("bool_memory"), MirType::I1);
    let flag = b.add_param(MirType::I1);
    let mask = b.add_param(MirType::Vec(Box::new(MirType::I1), 4));
    let slot = b.alloca(MirType::I1);
    b.build_void(Op::Store {
        ptr: slot,
        value: flag,
    });
    let vslot = b.alloca(MirType::Vec(Box::new(MirType::I1), 4));
    b.build_void(Op::Store {
        ptr: vslot,
        value: mask,
    });
    let back = b.build(MirType::I1, Op::Load(slot, MirType::I1));
    b.ret(Some(back));
    let text = module(b.finish(), &i);
    assert!(text.contains("alloca i8"), "{text}");
    assert!(text.contains("alloca <4 x i8>"), "{text}");
    assert!(text.contains("zext i1 %v0 to i8"), "{text}");
    assert!(text.contains("zext <4 x i1> %v1 to <4 x i8>"), "{text}");
    assert!(text.contains("trunc i8 %v"), "{text}");
    assert!(
        !text.contains("store i1") && !text.contains("load i1"),
        "{text}"
    );
}

/// The module names the host's triple and data layout, layout first (LLVM's own print order), and
/// both are what this host's x86-64 C ABI is: COFF mangling on Windows, ELF on Linux.
#[test]
fn module_names_the_host_target() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("triple"), MirType::Void);
    b.ret(None);
    let text = module(b.finish(), &i);
    let Some((triple, layout)) = host_target() else {
        assert!(!text.contains("target "), "{text}");
        return;
    };
    let mut header = text.lines().skip(1);
    if let Some(layout) = layout {
        assert_eq!(
            header.next(),
            Some(format!("target datalayout = \"{layout}\"").as_str())
        );
    }
    assert_eq!(
        header.next(),
        Some(format!("target triple = \"{triple}\"").as_str())
    );
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "msvc"
    )) {
        assert_eq!(triple, "x86_64-pc-windows-msvc");
        assert_eq!(layout, Some(DATALAYOUT_X86_64_COFF));
        assert!(DATALAYOUT_X86_64_COFF.starts_with("e-m:w-"));
    }
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        assert_eq!(triple, "x86_64-unknown-linux-gnu");
        assert_eq!(layout, Some(DATALAYOUT_X86_64_ELF));
        assert!(DATALAYOUT_X86_64_ELF.starts_with("e-m:e-"));
    }
}

// --- The link-ABI law ------------------------------------------------------------------------
//
// Every `wukong_*` symbol a module declares must be declared with the exact C signature it is
// defined with — `wukong_runtime`'s `#[no_mangle] extern "C"` functions, or the `--emit=exe` link
// shim (`wukong_driver`'s `WUKONG_RT_SHIM`) for the print/assert entries — and within one module
// every symbol has one signature across its `define`/`declare` and every `call`. LLVM's verifier
// checks neither: with opaque pointers a call through a mismatched function type is valid IR (and
// undefined behaviour), which is how an f64 vector reduction was once defined `float` and called
// `double`. The definitions are parsed from the Rust sources themselves.

/// `(symbol) -> (result, parameters)` in LLVM types.
type CAbi = BTreeMap<String, (String, Vec<String>)>;

fn repo_file(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(rel)
}

/// The LLVM type of a Rust type in a C signature: every pointer, reference and function pointer is
/// `ptr`; integers keep their width (the runtime widens its entries to `i64` by convention, which is
/// exactly what the law checks); `()`/`!` is `void`.
fn c_type(rust: &str) -> Result<String, String> {
    let t = rust.split_whitespace().collect::<Vec<_>>().join(" ");
    let fn_ptr = t.contains("fn(") || t.contains("fn (");
    if t.starts_with('*') || t.starts_with('&') || fn_ptr {
        return Ok("ptr".into());
    }
    Ok(match t.as_str() {
        "i64" | "u64" | "isize" | "usize" => "i64",
        "i32" | "u32" => "i32",
        "i16" | "u16" => "i16",
        "i8" | "u8" => "i8",
        "bool" => "i1",
        "f32" => "float",
        "f64" => "double",
        "()" | "!" => "void",
        other => return Err(format!("no C-ABI rule for the Rust type `{other}`")),
    }
    .into())
}

/// Split `s` at its top-level commas (outside `()`, `<>`, `[]`; an `->` arrow is not a bracket).
fn split_top(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '-' if chars.peek() == Some(&'>') => {
                cur.push(c);
                cur.push(chars.next().unwrap());
                continue;
            }
            '(' | '<' | '[' => depth += 1,
            ')' | '>' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    parts
}

/// The byte index just past the bracket matching the one at `open`.
fn matching(text: &str, open: usize) -> usize {
    let bytes = text.as_bytes();
    let (o, c) = (
        bytes[open],
        match bytes[open] {
            b'(' => b')',
            b'[' => b']',
            _ => b'}',
        },
    );
    let mut depth = 0;
    for (k, &b) in bytes.iter().enumerate().skip(open) {
        if b == o {
            depth += 1;
        } else if b == c {
            depth -= 1;
            if depth == 0 {
                return k + 1;
            }
        }
    }
    panic!("unbalanced bracket at byte {open}");
}

/// Every `#[no_mangle]` (or `#[unsafe(no_mangle)]`) `extern "C" fn` in `text`, after expanding the
/// file's own `macro_rules!` templates (the runtime stamps its typed file I/O entries from one).
fn parse_c_definitions(text: &str, into: &mut CAbi) -> Result<(), String> {
    let mut sources = vec![text.to_string()];
    // A single-arm `macro_rules! m { ($a:ident, …) => { … } }` whose body defines a C entry: expand
    // each `m!(…)` invocation in this file by substituting its arguments.
    let mut at = 0;
    while let Some(k) = text[at..].find("macro_rules!") {
        let start = at + k;
        at = start + 1;
        let name: String = text[start + "macro_rules!".len()..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let brace = start + text[start..].find('{').unwrap();
        let def_end = matching(text, brace);
        let def = &text[brace + 1..def_end - 1];
        if !def.contains("extern \"C\" fn $") {
            continue;
        }
        let paren = def.find('(').unwrap();
        let matcher = &def[paren + 1..matching(def, paren) - 1];
        let vars: Vec<String> = split_top(matcher)
            .iter()
            .map(|v| {
                v.trim()
                    .trim_start_matches('$')
                    .split(':')
                    .next()
                    .unwrap()
                    .to_string()
            })
            .collect();
        let body_open = def.find("=>").unwrap() + def[def.find("=>").unwrap()..].find('{').unwrap();
        let body = &def[body_open + 1..matching(def, body_open) - 1];
        let call = format!("{name}!(");
        let mut from = def_end;
        while let Some(c) = text[from..].find(&call) {
            let open = from + c + call.len() - 1;
            let args = split_top(&text[open + 1..matching(text, open) - 1]);
            assert_eq!(args.len(), vars.len(), "`{name}!` invocation arity");
            let mut expanded = body.to_string();
            // Longest variable first, so `$T` cannot clobber a `$Tx`.
            let mut order: Vec<usize> = (0..vars.len()).collect();
            order.sort_by_key(|&v| std::cmp::Reverse(vars[v].len()));
            for v in order {
                expanded = expanded.replace(&format!("${}", vars[v]), args[v].trim());
            }
            sources.push(expanded);
            from = open;
        }
    }
    for src in &sources {
        for (pos, _) in src.match_indices("no_mangle") {
            // Only a real attribute at the start of a line, never prose that mentions one.
            let line_start = src[..pos].rfind('\n').map_or(0, |n| n + 1);
            let lead = src[line_start..pos].trim();
            if lead != "#[" && lead != "#[unsafe(" {
                continue;
            }
            let mut rest = &src[pos + src[pos..].find(']').unwrap() + 1..];
            // Skip trivia between the attribute and the item: whitespace, comments, attributes.
            loop {
                rest = rest.trim_start();
                if rest.starts_with("//") {
                    rest = &rest[rest.find('\n').unwrap_or(rest.len())..];
                } else if rest.starts_with("#[") {
                    rest = &rest[matching(rest, 1)..];
                } else {
                    break;
                }
            }
            let header: String = rest
                .split('(')
                .next()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let Some(name) = header
                .strip_prefix("pub ")
                .unwrap_or(&header)
                .trim_start_matches("unsafe ")
                .strip_prefix("extern \"C\" fn ")
            else {
                continue; // a `#[no_mangle]` static, or not a C function
            };
            if name.starts_with('$') {
                continue; // a macro template, counted at each expansion instead
            }
            let open = rest.find('(').unwrap();
            let close = matching(rest, open);
            let params = rest[open + 1..close - 1]
                .lines()
                .map(|l| l.split("//").next().unwrap())
                .collect::<Vec<_>>()
                .join(" ");
            let mut sig = Vec::new();
            for p in split_top(&params) {
                let (_, ty) = p
                    .split_once(':')
                    .ok_or_else(|| format!("`{name}`: unparsed parameter `{p}`"))?;
                sig.push(c_type(ty).map_err(|e| format!("`{name}`: {e}"))?);
            }
            let after = rest[close..].trim_start();
            let ret = match after.strip_prefix("->") {
                Some(r) => c_type(r[..r.find('{').unwrap()].trim().trim_end_matches("where"))
                    .map_err(|e| format!("`{name}` result: {e}"))?,
                None => "void".into(),
            };
            let entry = (ret, sig);
            if let Some(previous) = into.get(name) {
                if previous != &entry {
                    return Err(format!(
                        "`{name}` is defined twice: {previous:?} and {entry:?}"
                    ));
                }
            }
            into.insert(name.to_string(), entry);
        }
    }
    Ok(())
}

/// Every C entry `wukong_runtime` defines (the whole `src` tree).
fn runtime_definitions() -> CAbi {
    fn walk(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, files);
            } else if p.extension().is_some_and(|x| x == "rs") {
                files.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&repo_file("crates/wukong_runtime/src"), &mut files);
    files.sort();
    let mut abi = CAbi::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        parse_c_definitions(&text, &mut abi).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
    }
    abi
}

/// The print/assert entries of the `--emit=exe` link shim (the C fallback mirrors them).
fn link_shim_definitions() -> CAbi {
    let text = std::fs::read_to_string(repo_file("crates/wukong_driver/src/lib.rs")).unwrap();
    let shim_start = text
        .find("const WUKONG_RT_SHIM")
        .expect("the link shim source");
    let shim = &text[shim_start..shim_start + text[shim_start..].find("\"##;").unwrap()];
    let mut abi = CAbi::new();
    parse_c_definitions(shim, &mut abi).unwrap();
    abi
}

fn render_runtime_abi(abi: &CAbi) -> String {
    let mut s = String::from(
        "//! GENERATED by `closeout_tests::runtime_abi_table_is_the_runtime_source`; do not edit by hand.\n\
         //! Regenerate: `WUKONG_UPDATE_RUNTIME_ABI=1 cargo test -p wukong_codegen_llvm --lib runtime_abi_table`.\n\
         \n\
         /// `(symbol, result, parameters)` of every `#[no_mangle] extern \"C\"` function `wukong_runtime`\n\
         /// defines, in LLVM types, sorted by symbol.\n\
         #[rustfmt::skip]\n\
         pub(crate) const RUNTIME_ABI: &[(&str, &str, &[&str])] = &[\n",
    );
    for (name, (ret, params)) in abi {
        let params: Vec<String> = params.iter().map(|p| format!("\"{p}\"")).collect();
        let _ = writeln!(s, "    (\"{name}\", \"{ret}\", &[{}]),", params.join(", "));
    }
    s.push_str("];\n");
    s
}

/// The generated table the emitter declares runtime calls from is exactly the runtime's sources:
/// a new, removed or retyped `#[no_mangle] extern "C"` entry fails here until the table is
/// regenerated (`WUKONG_UPDATE_RUNTIME_ABI=1`).
#[test]
fn runtime_abi_table_is_the_runtime_source() {
    let abi = runtime_definitions();
    assert!(abi.len() > 150, "parsed only {} runtime entries", abi.len());
    // Spot checks the parser must get right: plain, `unsafe`, macro-stamped and float entries.
    let sig = |n: &str| abi.get(n).cloned().unwrap_or_else(|| panic!("no `{n}`"));
    let v = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(
        sig("wukong_sgemm"),
        (
            "void".into(),
            v(&["ptr", "ptr", "ptr", "i64", "i64", "i64", "i64"])
        )
    );
    assert_eq!(
        sig("wukong_rt_read_f32"),
        ("i64".into(), v(&["ptr", "ptr", "i64"]))
    );
    assert_eq!(
        sig("wukong_rt_math_f64"),
        ("double".into(), v(&["double", "i64"]))
    );
    assert_eq!(sig("wukong_now_ns"), ("i64".into(), v(&[])));
    let rendered = render_runtime_abi(&abi);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runtime_abi.rs");
    if std::env::var_os("WUKONG_UPDATE_RUNTIME_ABI").is_some() {
        std::fs::write(&path, &rendered).unwrap();
    }
    let table: CAbi = runtime_abi::RUNTIME_ABI
        .iter()
        .map(|(n, r, p)| (n.to_string(), (r.to_string(), v(p))))
        .collect();
    let stale: Vec<String> = abi
        .iter()
        .filter(|(n, s)| table.get(*n) != Some(s))
        .map(|(n, s)| format!("{n}: source {s:?}, table {:?}", table.get(n)))
        .chain(
            table
                .keys()
                .filter(|n| !abi.contains_key(*n))
                .map(|n| format!("{n}: in the table, not in wukong_runtime")),
        )
        .collect();
    assert!(
        stale.is_empty(),
        "src/runtime_abi.rs is stale (regenerate with WUKONG_UPDATE_RUNTIME_ABI=1):\n{}",
        stale.join("\n")
    );
    // Sorted, so `runtime_abi` can binary-search it.
    assert!(runtime_abi::RUNTIME_ABI.windows(2).all(|w| w[0].0 < w[1].0));
    assert_eq!(runtime_abi("wukong_sgemm").unwrap().1.len(), 7);
}

/// `print`/`assert` lower to the link shim's entries with the shim's own parameter types.
#[test]
fn print_and_assert_lower_to_the_link_shim_abi() {
    let shim = link_shim_definitions();
    for intrinsic in [
        Intrinsic::PrintInt,
        Intrinsic::PrintUint,
        Intrinsic::PrintFloat,
        Intrinsic::PrintF32,
        Intrinsic::PrintStr,
        Intrinsic::PrintBool,
        Intrinsic::PrintChar,
        Intrinsic::Assert,
    ] {
        assert_eq!(
            shim.get(intrinsic.symbol()),
            Some(&("void".to_string(), vec![intrinsic.param().to_string()])),
            "{intrinsic:?}"
        );
    }
}

/// One parsed signature: `(result, parameter types)`.
type IrSig = (String, Vec<String>);

/// The type that leads an operand or parameter: a bracketed `<…>`/`[…]` aggregate, else the first
/// word, with `signext`/`zeroext`/`noalias` dropped.
fn leading_type(item: &str) -> String {
    let item = item.trim();
    if item.starts_with('<') || item.starts_with('[') {
        let close = if item.starts_with('<') { '>' } else { ']' };
        return item[..=item.find(close).unwrap()].to_string();
    }
    item.split_whitespace()
        .find(|t| !matches!(*t, "signext" | "zeroext" | "noalias"))
        .unwrap_or("")
        .to_string()
}

/// For every `@name` a module defines, declares or calls, each distinct signature it is used at.
fn signatures_by_symbol(ir: &str) -> BTreeMap<String, BTreeSet<IrSig>> {
    let mut sigs: BTreeMap<String, BTreeSet<IrSig>> = BTreeMap::new();
    for line in ir.lines() {
        let line = line.trim();
        let head = if let Some(rest) = line.strip_prefix("define ") {
            rest.trim_start_matches("internal ")
        } else if let Some(rest) = line.strip_prefix("declare ") {
            rest
        } else if let Some(k) = line.find("call ") {
            &line[k + "call ".len()..]
        } else {
            continue;
        };
        let Some(at) = head.find(" @") else { continue };
        let ret = leading_type(&head[..at]);
        let name_end = at + 2 + head[at + 2..].find('(').unwrap();
        let name = head[at + 2..name_end].to_string();
        let args = &head[name_end + 1..matching(head, name_end) - 1];
        // A definition's parameter ends in its `%name`, a call's argument in its operand; the type
        // leads the item either way.
        let params = split_top(args).iter().map(|a| leading_type(a)).collect();
        sigs.entry(name).or_default().insert((ret, params));
    }
    sigs
}

/// The symbols of `ir` used at more than one signature.
fn signature_conflicts(ir: &str) -> Vec<String> {
    signatures_by_symbol(ir)
        .into_iter()
        .filter(|(_, s)| s.len() > 1)
        .map(|(n, s)| format!("@{n}: {s:?}"))
        .collect()
}

/// The `wukong_*` declarations of `ir` that differ from the symbol's C definition (or have none).
fn link_abi_violations(ir: &str, runtime: &CAbi, shim: &CAbi) -> Vec<String> {
    let mut bad = Vec::new();
    for line in ir.lines().filter(|l| l.starts_with("declare ")) {
        let sigs = signatures_by_symbol(line);
        let (name, set) = sigs.iter().next().unwrap();
        if !name.starts_with("wukong_") {
            continue;
        }
        let declared = set.iter().next().unwrap();
        match runtime.get(name).or_else(|| shim.get(name)) {
            Some(def) if def == declared => {}
            Some(def) => bad.push(format!("@{name}: declared {declared:?}, defined {def:?}")),
            None => bad.push(format!("@{name}: declared {declared:?}, defined nowhere")),
        }
    }
    bad
}

/// The checker itself flags a call through a type other than the definition's, and a declaration
/// that is not the runtime's.
#[test]
fn signature_checker_flags_mismatches() {
    let ir = "define internal float @k(ptr %ptrs, ptr %scalars, i64 %n) {\n\
              entry:\n  %v3 = call double @k(ptr %v0, ptr %v1, i64 4096)\n}\n\
              declare void @wukong_sgemm_nt_epi(ptr, ptr, ptr, i64, i64, i64, i64, i64, i64)\n";
    let conflicts = signature_conflicts(ir);
    assert_eq!(conflicts.len(), 1, "{conflicts:?}");
    assert!(conflicts[0].starts_with("@k:"), "{conflicts:?}");
    let violations = link_abi_violations(ir, &runtime_definitions(), &CAbi::new());
    assert_eq!(violations.len(), 1, "{violations:?}");
    assert!(
        violations[0].contains("wukong_sgemm_nt_epi"),
        "{violations:?}"
    );
    // Vector operands and literal operands parse to their types.
    let ir = "  call void @f(<4 x float> <float 1.0, float 2.0>, ptr null, i64 -1, ptr inttoptr (i64 5 to ptr))";
    let sigs = signatures_by_symbol(ir);
    assert_eq!(
        sigs["f"].iter().next().unwrap().1,
        vec!["<4 x float>", "ptr", "i64", "ptr"]
    );
}

fn corpus_module(path: &std::path::Path, opt: u8) -> Option<String> {
    let src = std::fs::read_to_string(path).unwrap();
    let mut interner = Interner::new();
    let (ast, parsed) = wukong_parser::parse_module(&src, wukong_span::SourceId(0), &mut interner);
    if parsed.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, checked) = wukong_sema::check(&ast, &interner);
    if checked.iter().any(|d| d.is_error()) {
        return None; // a fixture that needs the driver's import loader
    }
    let (mut program, lowered) = wukong_mir_build::lower_program(&ast, &sema, &mut interner);
    if lowered.iter().any(|d| d.is_error()) {
        return None;
    }
    wukong_opt::optimize(&mut program, opt);
    Some(
        try_emit_llvm_ir(&program, &interner)
            .unwrap_or_else(|e| panic!("{} -O{opt}: declined: {e}", path.display())),
    )
}

/// The law over the whole run corpus, at -O0 and -O2: every `wukong_*` declaration is the symbol's
/// C definition, and every symbol has one signature across its definition, declaration and calls.
#[test]
fn link_abi_law_holds_on_the_run_corpus() {
    let runtime = runtime_definitions();
    let shim = link_shim_definitions();
    let mut fixtures: Vec<_> = std::fs::read_dir(repo_file("tests/run"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "wk"))
        .collect();
    fixtures.sort();
    let (mut checked, mut skipped, mut declared) = (0, Vec::new(), BTreeSet::new());
    let mut problems = Vec::new();
    for path in &fixtures {
        for opt in [0u8, 2] {
            let Some(ir) = corpus_module(path, opt) else {
                skipped.push(path.file_name().unwrap().to_string_lossy().into_owned());
                continue;
            };
            checked += 1;
            let name = path.file_name().unwrap().to_string_lossy();
            for c in signature_conflicts(&ir) {
                problems.push(format!("{name} -O{opt}: {c}"));
            }
            for v in link_abi_violations(&ir, &runtime, &shim) {
                problems.push(format!("{name} -O{opt}: {v}"));
            }
            for line in ir
                .lines()
                .filter(|l| l.starts_with("declare ") && l.contains(" @wukong_"))
            {
                declared.insert(signatures_by_symbol(line).into_keys().next().unwrap());
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
    // Only the multi-file fixtures (which need the driver's import loader) are skipped.
    assert!(skipped.len() <= 2 * 4, "skipped {skipped:?}");
    assert!(checked >= 2 * 480, "checked only {checked} modules");
    // The law is not vacuous: the corpus reaches well over a hundred runtime entries.
    assert!(
        declared.len() > 100,
        "{} runtime symbols declared",
        declared.len()
    );
}

#[test]
fn runtime_nullable_pointer_calls_share_one_abi() {
    let mut i = Interner::new();
    let kernel = i.intern("wukong_optional_bias");
    for null_first in [false, true] {
        let mut b = Builder::new(
            i.intern(if null_first {
                "nullable_first"
            } else {
                "pointer_first"
            }),
            MirType::Void,
        );
        let ptr = b.add_param(MirType::Ptr);
        let null = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        for a in if null_first { [null, ptr] } else { [ptr, null] } {
            b.build_void(Op::Call {
                func: kernel,
                args: vec![a],
            });
        }
        b.ret(None);
        let text = module(b.finish(), &i);
        assert!(
            text.contains("declare void @wukong_optional_bias(ptr)"),
            "{text}"
        );
        assert!(
            text.contains("call void @wukong_optional_bias(ptr null)"),
            "{text}"
        );
    }
    let mut b = Builder::new(i.intern("nonzero_pointer"), MirType::Void);
    let ptr = b.add_param(MirType::Ptr);
    let integer = b.build(MirType::I64, Op::ConstInt(7, MirType::I64));
    for a in [ptr, integer] {
        b.build_void(Op::Call {
            func: kernel,
            args: vec![a],
        });
    }
    b.ret(None);
    let mut p = Program::new();
    p.funcs.push(b.finish());
    assert!(try_emit_llvm_ir(&p, &i)
        .unwrap_err()
        .contains("incompatible call signatures"));
}

#[test]
fn discarded_defined_return_and_vector_kernel_reduction() {
    let mut i = Interner::new();
    let name = i.intern("discarded_return");
    let mut b = Builder::new(name, MirType::I32);
    let a = b.add_param(MirType::I32);
    b.build_void(Op::Call {
        func: name,
        args: vec![a],
    });
    b.ret(Some(a));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("call i32 @discarded_return(i32 %v0)"),
        "{text}"
    );
    assert!(!text.contains("declare i32 @discarded_return"), "{text}");
    let mut b = Builder::new(i.intern("kernel_reduction"), MirType::F32);
    let ptrs = b.add_param(MirType::Ptr);
    let scalars = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I32);
    // dot(x, y): two unit-stride loads, folded with a fused `acc = fma(x, y, acc)`.
    let kernel = b.add_vec_kernel(wukong_mir::VecKernel {
        name: i.intern("dot"),
        lane: wukong_mir::VecLane::F32,
        streams: 2,
        scalars: 0,
        ops: vec![
            wukong_mir::VecOp::Load { stream: 0 },
            wukong_mir::VecOp::Load { stream: 1 },
        ],
        unroll: 2,
        reduce: Some(wukong_mir::VecReduce {
            op: wukong_mir::VecRedOp::Add,
            value: 0,
            fma: Some((0, 1)),
        }),
    });
    let sum = b.build(
        MirType::F32,
        Op::VecKernelCall {
            kernel,
            ptrs,
            scalars,
            n,
        },
    );
    b.ret(Some(sum));
    let f = b.finish();
    let copies = f.vec_kernels[0].reduction_unroll();
    let text = module(f, &i);
    // The kernel is DEFINED in the module (no external symbol to link) ...
    assert!(
        text.contains(
            "define internal float @__wukong_veckernel_kernel_reduction_0(ptr %ptrs, ptr %scalars, i64 %n)"
        ),
        "{text}"
    );
    assert!(
        !text.contains("declare float @__wukong_veckernel"),
        "{text}"
    );
    assert!(
        text.contains("declare float @llvm.fma.f32(float, float, float)"),
        "{text}"
    );
    // ... with the evaluator's accumulator layout (U copies of 8 lanes) and fused fold ...
    assert!(
        text.contains(&format!("%acc = alloca [{} x float]", copies * 8)),
        "{text}"
    );
    assert!(
        text.contains("%m.acc = call float @llvm.fma.f32(float %m0, float %m1, float %j.old)"),
        "{text}"
    );
    assert!(
        text.contains("%c.acc = call float @llvm.fma.f32(float %c0, float %c1, float %l.old)"),
        "{text}"
    );
    // ... and the in-order horizontal fold ends at lane 7.
    assert!(text.contains("ret float %h7"), "{text}");
    // The call zero-extends the element count to the kernel's i64, as Cranelift does.
    assert!(text.contains("%t0 = zext i32 %v2 to i64"), "{text}");
    assert!(
        text.contains(
            "%v3 = call float @__wukong_veckernel_kernel_reduction_0(ptr %v0, ptr %v1, i64 %t0)"
        ),
        "{text}"
    );

    let mut b = Builder::new(i.intern("kernel_map"), MirType::Void);
    let ptrs = b.add_param(MirType::Ptr);
    let scalars = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    // out[i] = x[i] != x[i] ? x[i] : -(x[i] * s)
    let kernel = b.add_vec_kernel(wukong_mir::VecKernel {
        name: i.intern("map"),
        lane: wukong_mir::VecLane::F32,
        streams: 2,
        scalars: 1,
        ops: vec![
            wukong_mir::VecOp::Load { stream: 0 },
            wukong_mir::VecOp::Splat { scalar: 0 },
            wukong_mir::VecOp::Bin {
                op: wukong_mir::VecBin::Mul,
                a: 0,
                b: 1,
            },
            wukong_mir::VecOp::Neg { a: 2 },
            wukong_mir::VecOp::Cmp {
                pred: wukong_mir::VecCmp::Ne,
                a: 0,
                b: 0,
            },
            wukong_mir::VecOp::Select {
                mask: 4,
                a: 0,
                b: 3,
            },
            wukong_mir::VecOp::Store { stream: 1, val: 5 },
        ],
        unroll: 1,
        reduce: None,
    });
    b.build_void(Op::VecKernelCall {
        kernel,
        ptrs,
        scalars,
        n,
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    for line in [
        "define internal void @__wukong_veckernel_kernel_map_0(ptr %ptrs, ptr %scalars, i64 %n)",
        "%scalar0 = load float, ptr %scalar0.p",
        "%x2 = fmul float %x0, %scalar0",
        "%x3 = fneg float %x2",
        "%x4.c = fcmp une float %x0, %x0",
        "%x5 = select i1 %x5.c, float %x0, float %x3",
        "store float %x5, ptr %x6.a",
        "call void @__wukong_veckernel_kernel_map_0(ptr %v0, ptr %v1, i64 %v2)",
    ] {
        assert!(text.contains(line), "{line}:\n{text}");
    }

    let mut b = Builder::new(i.intern("kernel_missing"), MirType::Void);
    let ptrs = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    b.build_void(Op::VecKernelCall {
        kernel: 3,
        ptrs,
        scalars: ptrs,
        n,
    });
    b.ret(None);
    let mut p = Program::new();
    p.funcs.push(b.finish());
    assert!(try_emit_llvm_ir(&p, &i)
        .unwrap_err()
        .contains("vector kernel #3"));
}

#[test]
fn vector_integer_masks_and_lowp_casts() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("vector_masks_lowp_casts"), MirType::Void);
    let vf = MirType::Vec(Box::new(MirType::F32), 4);
    let vi = MirType::Vec(Box::new(MirType::I32), 4);
    let x = b.add_param(vf.clone());
    let y = b.add_param(vf.clone());
    let mask = b.build(vi, Op::Cmp(CmpOp::Folt, x, y));
    b.build(vf.clone(), Op::Select(mask, x, y));
    b.build(vf, Op::Select(x, x, y));
    for (from, to) in [
        (MirType::F16, MirType::BF16),
        (MirType::BF16, MirType::F16),
        (MirType::F64, MirType::F16),
        (MirType::F64, MirType::BF16),
    ] {
        let a = b.add_param(from);
        b.build(to.clone(), Op::Cast(CastKind::FpTrunc, a, to));
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(
        text.contains("sext <4 x i1> %v2.mask to <4 x i32>"),
        "{text}"
    );
    assert!(text.contains("select <4 x i1> %v3.cond"), "{text}");
    // f16 <-> bf16 rounds the f32 register straight onto the target grid; a double source first
    // rounds to f32 (`round_xx(v as f32)`, the interpreter's double rounding).
    assert!(text.contains("bitcast float %v5 to i32"), "{text}");
    assert!(text.contains("fptrunc float %v7 to half"), "{text}");
    assert!(
        !text
            .lines()
            .any(|l| l.contains("fptrunc") && l.contains("bfloat")),
        "{text}"
    );
    assert_eq!(text.matches("fptrunc double").count(), 2, "{text}");
    assert!(
        text.contains("fcmp une <4 x float> %v0, zeroinitializer"),
        "{text}"
    );

    // The vector form takes the same f32 lane adapter instead of declining.
    let mut b = Builder::new(i.intern("vector_lowp"), MirType::Void);
    let from = MirType::Vec(Box::new(MirType::F64), 4);
    let to = MirType::Vec(Box::new(MirType::F16), 4);
    let a = b.add_param(from);
    b.build(to.clone(), Op::Cast(CastKind::FpTrunc, a, to));
    b.ret(None);
    let text = module(b.finish(), &i);
    for line in [
        "%t0 = fptrunc <4 x double> %v0 to <4 x float>",
        "%t1 = fptrunc <4 x float> %t0 to <4 x half>",
        "%t2 = fpext <4 x half> %t1 to <4 x float>",
        "%v1 = select i1 true, <4 x float> %t2, <4 x float> %t2",
    ] {
        assert!(text.contains(line), "{line}:\n{text}");
    }
}

/// A runtime call is declared with the definition's signature whatever the call site passes: a
/// literal-zero bias is `ptr null` in a `ptr` slot (it used to make the slot `i64` in a module
/// whose every call passed zero), any other integer there is its `inttoptr`, narrow integers widen,
/// and a narrower MIR result is taken from the definition's `i64` and truncated.
#[test]
fn runtime_calls_take_the_definitions_signature() {
    let mut i = Interner::new();
    let epi = i.intern("wukong_sgemm_nt_epi");
    let size = i.intern("wukong_rt_file_size");
    let mut b = Builder::new(i.intern("runtime_adapt"), MirType::I32);
    let p = b.add_param(MirType::Ptr);
    let m = b.add_param(MirType::I32);
    let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
    let seven = b.build(MirType::I64, Op::ConstInt(7, MirType::I64));
    for bias in [zero, seven] {
        b.build_void(Op::Call {
            func: epi,
            args: vec![p, p, p, m, m, m, zero, bias, zero],
        });
    }
    let n = b.build(
        MirType::I32,
        Op::Call {
            func: size,
            args: vec![p],
        },
    );
    b.build_void(Op::Call {
        func: size,
        args: vec![p],
    });
    b.ret(Some(n));
    let text = module(b.finish(), &i);
    for line in [
        "declare void @wukong_sgemm_nt_epi(ptr, ptr, ptr, i64, i64, i64, i64, ptr, i64)",
        "declare i64 @wukong_rt_file_size(ptr)",
        "%t0 = sext i32 %v1 to i64",
        "call void @wukong_sgemm_nt_epi(ptr %v0, ptr %v0, ptr %v0, i64 %t0, i64 %t1, i64 %t2, i64 0, ptr null, i64 0)",
        "inttoptr i64 7 to ptr",
        "%v4.ret = call i64 @wukong_rt_file_size(ptr %v0)",
        "%v4 = trunc i64 %v4.ret to i32",
        "  call i64 @wukong_rt_file_size(ptr %v0)",
    ] {
        assert!(text.contains(line), "missing `{line}`:\n{text}");
    }
    assert!(signature_conflicts(&text).is_empty(), "{text}");
    assert!(
        link_abi_violations(&text, &runtime_definitions(), &CAbi::new()).is_empty(),
        "{text}"
    );
}

fn kernel_module(name: &str, kernel: wukong_mir::VecKernel, i: &mut Interner) -> String {
    let result = kernel.reduce.map(|_| kernel.lane.mir_type());
    let mut b = Builder::new(i.intern(name), MirType::Void);
    let ptrs = b.add_param(MirType::Ptr);
    let scalars = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    let k = b.add_vec_kernel(kernel);
    let call = Op::VecKernelCall {
        kernel: k,
        ptrs,
        scalars,
        n,
    };
    match result {
        Some(t) => {
            b.build(t, call);
        }
        None => b.build_void(call),
    }
    b.ret(None);
    let text = module(b.finish(), i);
    assert!(signature_conflicts(&text).is_empty(), "{text}");
    text
}

/// The vectorizer's f64/i32/i64 recipes are defined in their own lane type, with the evaluator's
/// arithmetic: double precision and 4-lane accumulators for f64, wrapping integer ops and signed
/// compares for the integer lanes. They were all spelled `float` before.
#[test]
fn vector_kernels_are_defined_in_their_lane_type() {
    use wukong_mir::{VecBin, VecCmp, VecKernel, VecLane, VecOp, VecRedOp, VecReduce};
    let mut i = Interner::new();
    // f64: sum of x * (y + s), fused into the accumulator.
    let text = kernel_module(
        "dot_f64",
        VecKernel {
            name: i.intern("k"),
            lane: VecLane::F64,
            streams: 2,
            scalars: 1,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Splat { scalar: 0 },
                VecOp::Bin {
                    op: VecBin::Add,
                    a: 1,
                    b: 2,
                },
            ],
            unroll: 2,
            reduce: Some(VecReduce {
                op: VecRedOp::Add,
                value: 0,
                fma: Some((0, 3)),
            }),
        },
        &mut i,
    );
    for line in [
        "define internal double @__wukong_veckernel_dot_f64_0(ptr %ptrs, ptr %scalars, i64 %n)",
        "%scalar0.p = getelementptr double, ptr %scalars, i64 0",
        "%scalar0 = load double, ptr %scalar0.p",
        "%acc = alloca [8 x double]",
        "store double 0x0000000000000000, ptr %acc7.p",
        "%m0 = load double, ptr %m0.a",
        "%m3 = fadd double %m1, %scalar0",
        "%m.acc = call double @llvm.fma.f64(double %m0, double %m3, double %j.old)",
        "%t.end = add i64 %t, 4",
        "%g1.3 = fadd double %f3, %f7",
        "ret double %h3",
        "declare double @llvm.fma.f64(double, double, double)",
        "%v3 = call double @__wukong_veckernel_dot_f64_0(ptr %v0, ptr %v1, i64 %v2)",
    ] {
        assert!(text.contains(line), "missing `{line}`:\n{text}");
    }
    assert!(!text.contains("float"), "{text}");

    // i32: out = (x < y) ? x : -y, elementwise.
    let text = kernel_module(
        "select_i32",
        VecKernel {
            name: i.intern("k"),
            lane: VecLane::I32,
            streams: 3,
            scalars: 0,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Cmp {
                    pred: VecCmp::Lt,
                    a: 0,
                    b: 1,
                },
                VecOp::Neg { a: 1 },
                VecOp::Select {
                    mask: 2,
                    a: 0,
                    b: 3,
                },
                VecOp::Store { stream: 2, val: 4 },
            ],
            unroll: 1,
            reduce: None,
        },
        &mut i,
    );
    for line in [
        "define internal void @__wukong_veckernel_select_i32_0(ptr %ptrs, ptr %scalars, i64 %n)",
        "%x0.a = getelementptr i32, ptr %base0, i64 %i",
        "%x2.c = icmp slt i32 %x0, %x1",
        "%x2 = select i1 %x2.c, i32 1, i32 0",
        "%x3 = sub i32 0, %x1",
        "%x4.c = icmp ne i32 %x2, 0",
        "%x4 = select i1 %x4.c, i32 %x0, i32 %x3",
        "store i32 %x4, ptr %x5.a",
        "call void @__wukong_veckernel_select_i32_0(ptr %v0, ptr %v1, i64 %v2)",
    ] {
        assert!(text.contains(line), "missing `{line}`:\n{text}");
    }

    // i64: a wrapping sum of x * y.
    let text = kernel_module(
        "sum_i64",
        VecKernel {
            name: i.intern("k"),
            lane: VecLane::I64,
            streams: 2,
            scalars: 0,
            ops: vec![
                VecOp::Load { stream: 0 },
                VecOp::Load { stream: 1 },
                VecOp::Bin {
                    op: VecBin::Mul,
                    a: 0,
                    b: 1,
                },
            ],
            unroll: 1,
            reduce: Some(VecReduce {
                op: VecRedOp::Add,
                value: 2,
                fma: None,
            }),
        },
        &mut i,
    );
    for line in [
        "define internal i64 @__wukong_veckernel_sum_i64_0(ptr %ptrs, ptr %scalars, i64 %n)",
        "%acc = alloca [4 x i64]",
        "store i64 0, ptr %acc0.p",
        "%m2 = mul i64 %m0, %m1",
        "%m.acc = add i64 %j.old, %m2",
        "%h3 = add i64 %h2, %f3",
        "ret i64 %h3",
        "%v3 = call i64 @__wukong_veckernel_sum_i64_0(ptr %v0, ptr %v1, i64 %v2)",
    ] {
        assert!(text.contains(line), "missing `{line}`:\n{text}");
    }
}

/// Every refusal carries a [`DECLINED`] tag, and every tag is reachable: each row is triggered
/// here once, by the smallest MIR that needs it.
#[test]
fn every_decline_is_catalogued() {
    let mut i = Interner::new();
    let mut tags = BTreeSet::new();
    let mut decline = |f: Function, i: &Interner| {
        let mut p = Program::new();
        p.funcs.push(f);
        let e = try_emit_llvm_ir(&p, i).expect_err("a decline");
        let tag = e
            .strip_prefix("LLVM emission [")
            .and_then(|r| r.split_once("]: "))
            .map(|(t, _)| t.to_string())
            .unwrap_or_else(|| panic!("untagged decline: {e}"));
        assert!(
            DECLINED.iter().any(|(t, _)| *t == tag),
            "`{tag}` is not in DECLINED: {e}"
        );
        tags.insert(tag);
    };
    let vf = MirType::Vec(Box::new(MirType::F32), 4);

    let mut b = Builder::new(i.intern("d_vector_print"), MirType::Void);
    let v = b.add_param(vf);
    b.build_void(Op::Call {
        func: i.intern("print"),
        args: vec![v],
    });
    b.ret(None);
    decline(b.finish(), &i);

    let mut b = Builder::new(i.intern("d_half_bitcast"), MirType::I16);
    let h = b.add_param(MirType::F16);
    let bits = b.build(MirType::I16, Op::Cast(CastKind::Bitcast, h, MirType::I16));
    b.ret(Some(bits));
    decline(b.finish(), &i);

    let mut b = Builder::new(i.intern("d_signatures"), MirType::Void);
    let a = b.add_param(MirType::I32);
    let x = b.add_param(MirType::F32);
    for arg in [a, x] {
        b.build_void(Op::Call {
            func: i.intern("foreign_sink"),
            args: vec![arg],
        });
    }
    b.ret(None);
    decline(b.finish(), &i);

    let mut b = Builder::new(i.intern("d_func_addr"), MirType::Ptr);
    let f = b.build(MirType::Ptr, Op::FuncAddr(i.intern("nowhere_typed")));
    b.ret(Some(f));
    decline(b.finish(), &i);

    let mut b = Builder::new(i.intern("d_runtime_abi"), MirType::Void);
    let narrow = b.add_param(MirType::F32);
    let op = b.add_param(MirType::I64);
    b.build_void(Op::Call {
        func: i.intern("wukong_rt_math_f64"),
        args: vec![narrow, op],
    });
    b.ret(None);
    decline(b.finish(), &i);

    let mut b = Builder::new(i.intern("d_vector_kernel"), MirType::Void);
    let ptrs = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    b.build_void(Op::VecKernelCall {
        kernel: 0,
        ptrs,
        scalars: ptrs,
        n,
    });
    b.ret(None);
    decline(b.finish(), &i);

    let mut b = Builder::new(i.intern("d_math"), MirType::I32);
    let k = b.add_param(MirType::I32);
    let r = b.build(MirType::I32, Op::Sqrt(k));
    b.ret(Some(r));
    decline(b.finish(), &i);

    let catalogued: BTreeSet<String> = DECLINED.iter().map(|(t, _)| t.to_string()).collect();
    assert_eq!(tags, catalogued);
    for (tag, why) in DECLINED {
        assert!(why.len() > 40, "`{tag}` needs a real reason");
    }
}

/// A vector load or store promises one lane's alignment, never the whole register's: the
/// vectorizer accesses `<4 x float>` at any element offset, and LLVM's implicit `align 16` turned
/// that into a `movaps` that faulted when the module was executed (the verifier cannot see it).
/// Stack slots carry the native backend's alignment, so a byte-buffer aggregate's fields are
/// aligned as their natural-alignment loads promise.
#[test]
fn vector_accesses_promise_only_lane_alignment() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("alignment"), MirType::Void);
    let p = b.add_param(MirType::Ptr);
    let vf = MirType::Vec(Box::new(MirType::F32), 4);
    let vb = MirType::Vec(Box::new(MirType::I1), 8);
    let vh = MirType::Vec(Box::new(MirType::BF16), 8);
    let v = b.build(vf.clone(), Op::Load(p, vf));
    b.build_void(Op::Store { ptr: p, value: v });
    let m = b.build(vb.clone(), Op::Load(p, vb));
    b.build_void(Op::Store { ptr: p, value: m });
    let h = b.build(vh.clone(), Op::Load(p, vh));
    b.build_void(Op::Store { ptr: p, value: h });
    let s = b.build(MirType::F64, Op::Load(p, MirType::F64));
    b.build_void(Op::Store { ptr: p, value: s });
    for (ty, _) in [
        (MirType::I8, 1),
        (MirType::Array(Box::new(MirType::I8), 3), 4),
        (MirType::Array(Box::new(MirType::I8), 24), 16),
        (MirType::Array(Box::new(MirType::F32), 8), 32),
        (MirType::Array(Box::new(MirType::F64), 512), 32),
    ] {
        b.alloca(ty);
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    for line in [
        "%v1 = load <4 x float>, ptr %v0, align 4",
        "store <4 x float> %v1, ptr %v0, align 4",
        "%v2.mem = load <8 x i8>, ptr %v0, align 1",
        "store <8 x i8> %t0, ptr %v0, align 1",
        "%v3.mem = load <8 x bfloat>, ptr %v0, align 2",
        "store <8 x bfloat> %t1, ptr %v0, align 2",
        "%v4 = load double, ptr %v0\n",
        "store double %v4, ptr %v0\n",
        "%v5 = alloca i8, align 1",
        "%v6 = alloca [3 x i8], align 4",
        "%v7 = alloca [24 x i8], align 16",
        "%v8 = alloca [8 x float], align 32",
        "%v9 = alloca [512 x double], align 32",
    ] {
        assert!(
            text.contains(line),
            "missing `{}`:\n{text}",
            line.trim_end()
        );
    }
}
