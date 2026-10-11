//! `Op::Alloca`'s explicit `align` (`@align(N)` on a `let`) is honoured natively at every call
//! depth: the base address of an `align 64` and an `align 32` slot is a multiple of 64 and 32 in
//! every frame, whatever the stack pointer's parity there.
//!
//! The x64 ABI keeps the stack pointer 16-byte aligned only and Cranelift never realigns a frame,
//! so a slot asking for more than 16 has to be rounded up at run time; a lowering that merely
//! requested the alignment of Cranelift would be right only in the frames that happened to land
//! on a 64-byte boundary. Addresses are observable only in MIR (the language has no
//! pointer-to-integer cast), so the probe is built by hand, and it is native-only: the
//! interpreter's addresses are slot indices, not bytes.

use wukong_mir::{AddrSpace, BinOp, Builder, CastKind, CmpOp, MirType, Op, Program, ValueId};
use wukong_span::{Interner, Symbol};

fn bytes(n: u32) -> MirType {
    MirType::Array(Box::new(MirType::I8), n)
}

fn konst(b: &mut Builder, v: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(v, MirType::I64))
}

fn addr(b: &mut Builder, slot: ValueId) -> ValueId {
    b.build(
        MirType::I64,
        Op::Cast(CastKind::PtrToInt, slot, MirType::I64),
    )
}

/// `probe(d) -> i64` allocates an `align 64` `[24 x i8]` slot, an `align 32` `[40 x i8]` slot and a
/// natural 16-byte one, recurses `d` deep, and returns one packed word OR-ed over every frame:
/// bits 0..6 the aligned bases' misalignment (`a64 & 63 | a32 & 31`, which must be 0), and bit
/// `8 + k` set when the natural slot sat at `k * 16 mod 64` (it is never realigned, so it reports
/// the stack pointer's parity — the evidence the frames really landed at different offsets).
/// `main` runs it directly and under four callers whose frames differ by 16 bytes each, prints the
/// parity bits and returns the misalignment as its exit code.
fn probe_program() -> (Program, Interner, Symbol) {
    let mut it = Interner::new();
    let probe = it.intern("probe");
    let print = it.intern("print");
    let main = it.intern("main");
    let mut program = Program::new();

    let mut b = Builder::new(probe, MirType::I64);
    let d = b.add_param(MirType::I64);
    let a64 = b.alloca_aligned(bytes(24), AddrSpace::Local, 64);
    let a32 = b.alloca_aligned(bytes(40), AddrSpace::Local, 32);
    let natural = b.alloca(MirType::Array(Box::new(MirType::I64), 2));
    let zero = konst(&mut b, 0);
    let zero8 = b.build(MirType::I8, Op::ConstInt(0, MirType::I8));
    for slot in [a64, a32] {
        b.build_void(Op::Store {
            ptr: slot,
            value: zero8,
        });
    }
    b.build_void(Op::Store {
        ptr: natural,
        value: zero,
    });
    let (p64, p32, pn) = (addr(&mut b, a64), addr(&mut b, a32), addr(&mut b, natural));
    let m63 = konst(&mut b, 63);
    let m31 = konst(&mut b, 31);
    let m48 = konst(&mut b, 48);
    let mis64 = b.build(MirType::I64, Op::Bin(BinOp::And, p64, m63));
    let mis32 = b.build(MirType::I64, Op::Bin(BinOp::And, p32, m31));
    let mis = b.build(MirType::I64, Op::Bin(BinOp::Or, mis64, mis32));
    // 1 << (8 + (pn & 48) / 16)
    let par = b.build(MirType::I64, Op::Bin(BinOp::And, pn, m48));
    let four = konst(&mut b, 4);
    let k = b.build(MirType::I64, Op::Bin(BinOp::LShr, par, four));
    let eight = konst(&mut b, 8);
    let sh = b.build(MirType::I64, Op::Bin(BinOp::Add, k, eight));
    let one = konst(&mut b, 1);
    let bit = b.build(MirType::I64, Op::Bin(BinOp::Shl, one, sh));
    let word = b.build(MirType::I64, Op::Bin(BinOp::Or, mis, bit));
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, d, zero));
    let deeper = b.new_block();
    let leaf = b.new_block();
    b.cond_br(more, deeper, vec![], leaf, vec![]);
    b.switch_to(deeper);
    let d1 = b.build(MirType::I64, Op::Bin(BinOp::Sub, d, one));
    let rest = b.build(
        MirType::I64,
        Op::Call {
            func: probe,
            args: vec![d1],
        },
    );
    let acc = b.build(MirType::I64, Op::Bin(BinOp::Or, rest, word));
    b.ret(Some(acc));
    b.switch_to(leaf);
    b.ret(Some(word));
    program.funcs.push(b.finish());

    // Callers whose own frames differ by 16 bytes: 1, 3, 5 and 7 `i64` slots.
    let mut pads = Vec::new();
    for (name, n) in [("pad_a", 1u32), ("pad_b", 3), ("pad_c", 5), ("pad_d", 7)] {
        let sym = it.intern(name);
        let mut b = Builder::new(sym, MirType::I64);
        let d = b.add_param(MirType::I64);
        let slot = b.alloca(MirType::Array(Box::new(MirType::I64), n));
        b.build_void(Op::Store {
            ptr: slot,
            value: d,
        });
        let r = b.build(
            MirType::I64,
            Op::Call {
                func: probe,
                args: vec![d],
            },
        );
        b.ret(Some(r));
        program.funcs.push(b.finish());
        pads.push(sym);
    }

    let mut b = Builder::new(main, MirType::I32);
    let depth = konst(&mut b, 8);
    let mut all = b.build(
        MirType::I64,
        Op::Call {
            func: probe,
            args: vec![depth],
        },
    );
    for pad in pads {
        let r = b.build(
            MirType::I64,
            Op::Call {
                func: pad,
                args: vec![depth],
            },
        );
        all = b.build(MirType::I64, Op::Bin(BinOp::Or, all, r));
    }
    let eight = konst(&mut b, 8);
    let parities = b.build(MirType::I64, Op::Bin(BinOp::LShr, all, eight));
    b.build_void(Op::Call {
        func: print,
        args: vec![parities],
    });
    let m255 = konst(&mut b, 255);
    let mis = b.build(MirType::I64, Op::Bin(BinOp::And, all, m255));
    let code = b.build(MirType::I32, Op::Cast(CastKind::Trunc, mis, MirType::I32));
    b.ret(Some(code));
    program.funcs.push(b.finish());
    let errors = wukong_mir::verify::verify_program(&program);
    assert!(errors.is_empty(), "{errors:?}");
    (program, it, main)
}

#[test]
fn explicitly_aligned_slots_are_aligned_in_every_frame() {
    let (program, interner, main) = probe_program();
    for level in [0u8, 2] {
        let mut optimized = program.clone();
        wukong_opt::optimize(&mut optimized, level);
        let (code, out) =
            wukong_codegen_cranelift::jit_run(&optimized, main, &interner).expect("jit");
        let parities: u32 = String::from_utf8(out).unwrap().trim().parse().unwrap();
        // Not vacuous: the natural slot sat at more than one offset mod 64, so a base that was
        // only 16- or 32-aligned would have been misaligned in some frame.
        assert!(
            parities.count_ones() >= 2,
            "-O{level}: every frame had one 64-byte parity ({parities:#b}), so the probe proves nothing"
        );
        assert_eq!(code, 0, "-O{level}: an aligned slot base was misaligned");
    }
}

/// The explicit alignment is printed in the MIR dump only when it is set, so every slot the
/// compiler makes (`align 0`) prints exactly as it always did.
#[test]
fn the_dump_names_only_an_explicit_alignment() {
    let (program, interner, _) = probe_program();
    let text = wukong_mir::print::print_program(&program, &interner);
    assert!(text.contains("alloca [24 x i8] align 64"), "{text}");
    assert!(text.contains("alloca [40 x i8] align 32"), "{text}");
    assert!(text.contains("alloca [2 x i64]\n"), "{text}");
}
