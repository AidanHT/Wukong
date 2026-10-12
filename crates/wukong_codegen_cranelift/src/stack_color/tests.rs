//! Stack-slot coloring: the plan, the frame it produces, and that sharing never changes a value.

use super::*;
use crate::{jit_compile_with, BackendOpt};
use wukong_mir::{BinOp, Builder, CastKind, CmpOp, Program};
use wukong_span::{Interner, SourceId, Symbol};

fn arr(n: u32) -> MirType {
    MirType::Array(Box::new(MirType::F32), n)
}

/// Parse, check, lower and optimize `src` at `opt`.
fn lower(src: &str, opt: u8) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
    wukong_opt::optimize(&mut program, opt);
    (program, interner)
}

/// Three scratch arrays, each filled, folded into a scalar that seeds the next and printed, never
/// live at the same time: `a` is dead once `sa` is summed, `b` once `sb` is. `keep` is read at the
/// very end, so it stays live across all three and must never share with any of them.
const SEQUENTIAL: &str = "
fn main() -> i32 {
    let mut keep: [f32; 8] = [0.0; 8];
    for i in 0..8 { keep[i] = (i as f32) * 10.0; }
    let mut a: [f32; 256] = [0.0; 256];
    for i in 0..256 { a[i] = (i as f32) + 0.5; }
    let mut sa: f32 = 0.0;
    for i in 0..256 { sa = sa + a[i]; }
    print(a[0]);
    print(a[255]);
    let mut b: [f32; 256] = [0.0; 256];
    for i in 0..256 { b[i] = (i as f32) * 2.0 + sa / 32768.0; }
    let mut sb: f32 = 0.0;
    for i in 0..256 { sb = sb + b[i]; }
    print(b[0]);
    print(b[255]);
    let mut c: [f32; 256] = [0.0; 256];
    for i in 0..256 { c[i] = sb - (i as f32); }
    print(c[0]);
    print(c[255]);
    print(sa);
    print(sb);
    for i in 0..8 { print(keep[i]); }
    return 0;
}
";

/// The values `SEQUENTIAL` prints, derived from its scalar definition (every partial sum is exact
/// in `f32`): `a[i] = i + 0.5`, so `sa = 32640 + 128 = 32768`; `b[i] = 2i + 1`, so
/// `sb = 2 * 32640 + 256 = 65536`; `c[i] = 65536 - i`; `keep[i] = 10i`.
const SEQUENTIAL_OUT: &[&str] = &[
    "0.5", "255.5", "1", "511", "65536", "65281", "32768", "65536", "0", "10", "20", "30", "40",
    "50", "60", "70",
];

fn lines(out: &[u8]) -> Vec<String> {
    String::from_utf8(out.to_vec())
        .unwrap()
        .lines()
        .map(|l| l.trim().to_string())
        .collect()
}

/// The JIT at source level `opt` (the code generator's level follows it, as in the driver).
fn jit_at(program: &Program, interner: &Interner, opt: u8, coloring: bool) -> (i64, Vec<u8>) {
    let main = interner.get("main").expect("main");
    with_coloring(coloring, || {
        jit_compile_with(program, main, interner, BackendOpt::for_level(opt))
            .expect("jit")
            .run()
            .expect("run")
    })
}

/// At every `-O` level the scratch arrays share frame space (fewer static slot bytes than one slot
/// each), and the program prints exactly its scalar definition natively and in the interpreter,
/// with coloring on and off.
#[test]
fn sequential_scratch_shares_a_slot_and_computes_the_same_values() {
    for opt in [0u8, 2] {
        let (program, interner) = lower(SEQUENTIAL, opt);
        let main = interner.get("main").expect("main");
        let bytes = frame_bytes(&program, opt);
        assert!(
            bytes.colored + 2 * 1024 <= bytes.separate,
            "-O{opt}: two of the three 1 KiB scratch arrays should fold into the third: {bytes:?}"
        );
        let (code, interp) = wukong_interp::run_with_output(&program, main, &interner).unwrap();
        assert_eq!(code, 0);
        assert_eq!(lines(&interp), SEQUENTIAL_OUT, "-O{opt} interpreter");
        for on in [true, false] {
            let (code, native) = jit_at(&program, &interner, opt, on);
            assert_eq!(code, 0);
            assert_eq!(native, interp, "-O{opt}, coloring {on}");
        }
    }
}

/// `tests/run/stack_color_phases.wk`: of its five 256-byte arrays `a`, `b`, `c`, `d`, `e`, the pairs
/// that are never live together (`a`/`b`, `d`/`e`) share, so the frame loses at least two arrays,
/// while `c` overlaps `b`, `d` and `e` and never shares with any of them. (Which pairs end up
/// together also depends on the alias regions — `d` and `e` are handed to a recognized runtime
/// kernel — so the test pins the savings and the separations, not one exact partition.) The
/// fixture's printed lanes, checked by the run gates on both backends, prove the shared slots hold
/// the right values.
#[test]
fn the_phases_fixture_shares_disjoint_arrays_and_separates_overlapping_ones() {
    let src = include_str!("../../../../tests/run/stack_color_phases.wk");
    let array = padded(alloca_slot(256, 0).0);
    for opt in [0u8, 2] {
        let (program, interner) = lower(src, opt);
        let bytes = frame_bytes(&program, opt);
        assert!(
            bytes.separate - bytes.colored >= 2 * array,
            "-O{opt}: {bytes:?}"
        );
        let main = interner.get("main").expect("main");
        let f = program.function(main).unwrap();
        let calls = CaptureSummary::analyze(&program);
        let alias = AliasInfo::analyze(f);
        let cfg = ColorConfig {
            calls: &calls,
            mix_regions: opt == 0,
        };
        let plan = SlotPlan::new(f, cfg, &alias);
        // The five arrays, in declaration order (mir_build appends each to the entry block).
        let colors: Vec<Option<u32>> = f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|inst| match (&inst.op, inst.result) {
                (Op::Alloca { ty, .. }, Some(v)) if *ty == arr(64) => {
                    Some(plan.color_of(v).map(|(c, _)| c))
                }
                _ => None,
            })
            .collect();
        assert_eq!(colors.len(), 5, "-O{opt}: every array stays in memory");
        let [_a, b, c, d, e] = colors[..] else {
            unreachable!()
        };
        for (name, other) in [("b", b), ("d", d), ("e", e)] {
            assert!(
                c.is_none() || c != other,
                "-O{opt}: c overlaps {name} and must not share with it: {colors:?}"
            );
        }
    }
}

/// `probe(d)`: three 4 KiB scratch arrays used one after another, and an `i64` mark whose address
/// is taken (so it escapes and keeps a slot of its own). `probe(0)` returns its mark's address;
/// `probe(1)` returns its own mark's address minus its callee's — one frame's size. The scratch
/// values flow into a checksum so none of the arrays is dead.
fn probe_program() -> (Program, Interner, Symbol) {
    let mut i = Interner::new();
    let probe = i.intern("probe");
    let print = i.intern("print");
    let main = i.intern("main");
    let mut program = Program::new();

    let mut b = Builder::new(probe, MirType::I64);
    let d = b.add_param(MirType::I64);
    let scratch: Vec<ValueId> = (0..3).map(|_| b.alloca(arr(1024))).collect();
    let mark = b.alloca(MirType::I64);
    let c = |b: &mut Builder, v: i128| b.build(MirType::I64, Op::ConstInt(v, MirType::I64));
    let df = b.build(MirType::F32, Op::Cast(CastKind::SiToFp, d, MirType::F32));
    let mut carry = df;
    for (k, &s) in scratch.iter().enumerate() {
        let lane = c(&mut b, 100 * (k as i128 + 1));
        let at = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: s,
                index: lane,
                elem: MirType::F32,
            },
        );
        let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let v = b.build(MirType::F32, Op::Bin(BinOp::FAdd, carry, one));
        b.build_void(Op::Store { ptr: at, value: v });
        carry = b.build(MirType::F32, Op::Load(at, MirType::F32));
    }
    // The checksum `d + 3` goes into the mark, and is read back into the result's low bits below.
    let check = b.build(
        MirType::I64,
        Op::Cast(CastKind::FpToSi, carry, MirType::I64),
    );
    b.build_void(Op::Store {
        ptr: mark,
        value: check,
    });
    let addr = b.build(
        MirType::I64,
        Op::Cast(CastKind::PtrToInt, mark, MirType::I64),
    );
    let zero = c(&mut b, 0);
    let deeper = b.new_block();
    let leaf = b.new_block();
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, d, zero));
    b.cond_br(more, deeper, vec![], leaf, vec![]);
    b.switch_to(deeper);
    let one = c(&mut b, 1);
    let d1 = b.build(MirType::I64, Op::Bin(BinOp::Sub, d, one));
    let child = b.build(
        MirType::I64,
        Op::Call {
            func: probe,
            args: vec![d1],
        },
    );
    let frame = b.build(MirType::I64, Op::Bin(BinOp::Sub, addr, child));
    // frame * 16 + checksum (the checksum is 4 at d = 1).
    let sixteen = c(&mut b, 16);
    let scaled = b.build(MirType::I64, Op::Bin(BinOp::Mul, frame, sixteen));
    let kept = b.build(MirType::I64, Op::Load(mark, MirType::I64));
    let word = b.build(MirType::I64, Op::Bin(BinOp::Add, scaled, kept));
    b.ret(Some(word));
    b.switch_to(leaf);
    b.ret(Some(addr));
    program.funcs.push(b.finish());

    let mut b = Builder::new(main, MirType::I32);
    let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
    let w = b.build(
        MirType::I64,
        Op::Call {
            func: probe,
            args: vec![one],
        },
    );
    b.build_void(Op::Call {
        func: print,
        args: vec![w],
    });
    let ok = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(ok));
    program.funcs.push(b.finish());
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    (program, i, main)
}

/// Run the probe natively, with the code generator at source level `opt`; returns (frame size in
/// bytes, checksum).
fn probe_frame(opt: u8, on: bool) -> (i64, i64) {
    let (program, interner, _) = probe_program();
    let (code, out) = jit_at(&program, &interner, opt, on);
    assert_eq!(code, 0);
    let w: i64 = String::from_utf8(out).unwrap().trim().parse().unwrap();
    (w >> 4, w & 15)
}

/// The real frame, measured on the machine: coloring folds the three 4 KiB scratch arrays into one,
/// so a frame shrinks by at least the two arrays it no longer reserves.
#[test]
fn coloring_shrinks_the_machine_frame() {
    for opt in [0u8, 2] {
        let (separate, sum_separate) = probe_frame(opt, false);
        let (colored, sum_colored) = probe_frame(opt, true);
        assert_eq!(sum_separate, 4, "-O{opt}: 1 + three increments");
        assert_eq!(sum_colored, 4, "-O{opt}");
        assert!(
            separate >= 3 * 4096,
            "-O{opt}: without coloring every scratch array has its own slot: {separate}"
        );
        assert!(
            colored + 2 * 4096 <= separate,
            "-O{opt}: coloring should save two 4 KiB slots: {colored} vs {separate}"
        );
        let (program, _, _) = probe_program();
        let bytes = frame_bytes(&program, opt);
        assert!(bytes.separate - bytes.colored >= 2 * 4096, "{bytes:?}");
    }
}

/// Plan-level rules: a slot lowered through a block parameter is not private in Cranelift's alias
/// regions (the provenance analysis loses it there) although the lifetime analysis tracks it, so
/// when Cranelift optimizes it must not share with a private slot even when their lifetimes are
/// disjoint, while two private slots with disjoint lifetimes do share, under the larger one's slot.
/// At `-O0` (no alias analysis in Cranelift) all three share, and the private ones are demoted to
/// the default region.
#[test]
fn colors_mix_alias_regions_only_without_alias_analysis() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let mut b = Builder::new(main, MirType::Void);
    let routed = b.alloca(arr(64));
    let small = b.alloca(arr(16));
    let big = b.alloca(arr(32));
    // `routed` reaches its use through a block parameter.
    let next = b.new_block();
    let p = b.block_param(next, MirType::Ptr);
    b.br(next, vec![routed]);
    b.switch_to(next);
    let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
    b.build_void(Op::Store { ptr: p, value: one });
    // Then `small`, then `big`, one after the other.
    b.build_void(Op::Store {
        ptr: small,
        value: one,
    });
    b.build_void(Op::Store {
        ptr: big,
        value: one,
    });
    b.ret(None);
    let mut program = Program::new();
    program.funcs.push(b.finish());
    let f = &program.funcs[0];
    let calls = CaptureSummary::analyze(&program);
    let alias = AliasInfo::analyze(f);
    assert!(!alias.is_private_stack(routed));
    assert!(alias.is_private_stack(small) && alias.is_private_stack(big));
    let lifetimes = SlotLifetimes::analyze(f, &calls);
    assert!(
        !lifetimes.escapes(0),
        "the lifetime analysis follows the block parameter"
    );
    assert!(!lifetimes.interfere(0, 1) && !lifetimes.interfere(0, 2));
    let strict = ColorConfig {
        calls: &calls,
        mix_regions: false,
    };
    let plan = SlotPlan::new(f, strict, &alias);
    assert!(plan.color_of(routed).is_none(), "different alias region");
    let (cs, spec_s) = plan.color_of(small).expect("small shares");
    let (cb, spec_b) = plan.color_of(big).expect("big shares");
    assert_eq!(cs, cb);
    assert_eq!(spec_s, spec_b);
    assert_eq!(spec_b, alloca_slot(128, 0).0, "the larger member's slot");
    assert_eq!(plan.color_count(), 1);
    assert!(!plan.demotes(small) && !plan.demotes(big));

    let mixed = SlotPlan::new(
        f,
        ColorConfig {
            calls: &calls,
            mix_regions: true,
        },
        &alias,
    );
    let colors: Vec<u32> = [routed, small, big]
        .iter()
        .map(|&v| mixed.color_of(v).expect("all three share").0)
        .collect();
    assert!(colors.iter().all(|&c| c == colors[0]), "{colors:?}");
    assert_eq!(mixed.color_of(routed).unwrap().1, alloca_slot(256, 0).0);
    assert!(
        mixed.demotes(small) && mixed.demotes(big),
        "private members give up the region"
    );
    assert!(!mixed.demotes(routed));
}

/// Over-aligned members keep their own rounding inside a shared slot: a 64-byte `@align(64)` slot
/// and a plain one share, the shared slot covers the over-aligned member's slack, and it is at
/// least 16-aligned so the rounding stays inside it.
#[test]
fn an_over_aligned_member_keeps_its_slack() {
    let (plain, want_plain) = alloca_slot(256, 0);
    assert_eq!(want_plain, 32);
    let (wide, want_wide) = alloca_slot(64, 64);
    assert_eq!(want_wide, 64);
    assert_eq!(wide.bytes, 64 + 48);
    let cover = plain.cover(wide);
    assert!(cover.bytes >= plain.bytes && cover.bytes >= wide.bytes);
    assert!(cover.align_shift >= 4);
    // Small slots stay exactly as before coloring existed.
    assert_eq!(
        alloca_slot(4, 0).0,
        SlotSpec {
            bytes: 4,
            align_shift: 2
        }
    );
    assert_eq!(
        alloca_slot(24, 0).0,
        SlotSpec {
            bytes: 24,
            align_shift: 4
        }
    );
}

/// A function in which nothing shares lowers to the same object bytes with coloring on and off:
/// `main`'s two arrays are live together (both are filled, then both are read), so the plan is
/// empty and every slot is created exactly as without coloring.
#[test]
fn nothing_shared_means_identical_code() {
    let src = "
fn main() -> i32 {
    let mut a: [f32; 64] = [0.0; 64];
    let mut b: [f32; 64] = [0.0; 64];
    for i in 0..64 { a[i] = i as f32; }
    for i in 0..64 { b[i] = (i as f32) * 3.0; }
    let mut s: f32 = 0.0;
    for i in 0..64 { s = s + a[i] - b[63 - i]; }
    print(s);
    return 0;
}
";
    let (program, interner) = lower(src, 1);
    let bytes = frame_bytes(&program, 2);
    assert!(bytes.separate >= 512, "{bytes:?}");
    assert_eq!(bytes.colored, bytes.separate, "{bytes:?}");
    let on = with_coloring(true, || crate::emit_object(&program, &interner)).unwrap();
    let off = with_coloring(false, || crate::emit_object(&program, &interner)).unwrap();
    assert!(on == off, "object bytes differ with nothing to share");
}

/// Corpus-wide static frame bytes — `tests/run` and `examples`, every program that needs no import —
/// with and without coloring, at -O0 and -O2. A report, not a gate: run with
/// `cargo test -p wukong_codegen_cranelift --lib corpus_frame_bytes -- --ignored --nocapture`.
#[test]
#[ignore]
fn corpus_frame_bytes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for dir in ["tests/run", "examples"] {
        let mut files: Vec<_> = std::fs::read_dir(root.join(dir))
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wk"))
            .collect();
        files.sort();
        for opt in [0u8, 2] {
            let mut total = FrameBytes::default();
            let (mut measured, mut skipped, mut shrunk) = (0, 0, 0);
            let mut top: Vec<(u64, u64, String)> = Vec::new();
            for path in &files {
                let src = std::fs::read_to_string(path).unwrap();
                let mut interner = Interner::new();
                let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
                if pd.iter().any(|d| d.is_error()) || src.lines().any(|l| l.starts_with("import "))
                {
                    skipped += 1;
                    continue;
                }
                let (sema, sd) = wukong_sema::check(&module, &interner);
                if sd.iter().any(|d| d.is_error()) {
                    skipped += 1;
                    continue;
                }
                let (mut program, ld) =
                    wukong_mir_build::lower_program(&module, &sema, &mut interner);
                if ld.iter().any(|d| d.is_error()) {
                    skipped += 1;
                    continue;
                }
                wukong_opt::optimize(&mut program, opt);
                let b = frame_bytes(&program, opt);
                if b.colored < b.separate {
                    shrunk += 1;
                    top.push((
                        b.separate - b.colored,
                        b.separate,
                        path.file_name().unwrap().to_string_lossy().into_owned(),
                    ));
                }
                total += b;
                measured += 1;
            }
            top.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)));
            eprintln!(
                "{dir} -O{opt}: {measured} programs ({skipped} skipped), {shrunk} with a smaller \
                 frame: {} -> {} slot bytes ({:.1}% saved)",
                total.separate,
                total.colored,
                100.0 * (total.separate - total.colored) as f64 / total.separate.max(1) as f64
            );
            for (saved, of, name) in top.iter().take(8) {
                eprintln!("    {saved:>8} of {of:>8} bytes  {name}");
            }
            assert!(total.colored <= total.separate);
        }
    }
}
