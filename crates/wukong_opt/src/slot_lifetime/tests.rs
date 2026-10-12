//! The lifetime analysis on hand-built MIR: what each slot's lifetime is, when two slots
//! interfere, and every way an address can escape.

use super::*;
use wukong_mir::{BinOp, Builder, CmpOp, VecBin, VecKernel, VecLane, VecOp};
use wukong_span::Interner;

fn arr(n: u32) -> MirType {
    MirType::Array(Box::new(MirType::F32), n)
}

fn i64c(b: &mut Builder, v: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(v, MirType::I64))
}

fn f32c(b: &mut Builder, v: f64) -> ValueId {
    b.build(MirType::F32, Op::ConstFloat(v, MirType::F32))
}

/// `store 1.0 -> p[k]`.
fn write(b: &mut Builder, p: ValueId, k: i128) {
    let i = i64c(b, k);
    let at = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: p,
            index: i,
            elem: MirType::F32,
        },
    );
    let one = f32c(b, 1.0);
    b.build_void(Op::Store {
        ptr: at,
        value: one,
    });
}

/// `load p[k]`.
fn read(b: &mut Builder, p: ValueId, k: i128) -> ValueId {
    let i = i64c(b, k);
    let at = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: p,
            index: i,
            elem: MirType::F32,
        },
    );
    b.build(MirType::F32, Op::Load(at, MirType::F32))
}

fn call(b: &mut Builder, func: Symbol, args: Vec<ValueId>) {
    b.build_void(Op::Call { func, args });
}

/// Analyze function `name` of `program` with the program's capture summary.
fn lifetimes(program: &Program, name: Symbol) -> SlotLifetimes {
    assert!(
        wukong_mir::verify::verify_program(program).is_empty(),
        "{:?}",
        wukong_mir::verify::verify_program(program)
    );
    let calls = CaptureSummary::analyze(program);
    SlotLifetimes::analyze(program.function(name).unwrap(), &calls)
}

fn program_of(funcs: Vec<Function>) -> Program {
    let mut p = Program::new();
    p.funcs = funcs;
    p
}

#[test]
fn sequential_scratch_does_not_interfere() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let a = b.alloca(arr(64));
    let c = b.alloca(arr(64));
    write(&mut b, a, 0);
    let x = read(&mut b, a, 0);
    call(&mut b, sink, vec![x]);
    write(&mut b, c, 1);
    let y = read(&mut b, c, 1);
    call(&mut b, sink, vec![y]);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    assert_eq!(lt.len(), 2);
    assert!(!lt.escapes(0) && !lt.escapes(1));
    assert!(
        !lt.interfere(0, 1),
        "{:?} vs {:?}",
        lt.live_ranges(0),
        lt.live_ranges(1)
    );
    assert!(lt.interfere(0, 0));
    // Each lifetime is exactly first use to last use (the `gep` of the write to the load).
    assert_eq!(lt.live_ranges(0).len(), 1);
    assert!(!lt.live_at(0, 0, 0), "an alloca instruction is not a use");
}

#[test]
fn interleaved_uses_interfere() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let a = b.alloca(arr(8));
    let c = b.alloca(arr(8));
    write(&mut b, a, 0);
    write(&mut b, c, 0);
    let x = read(&mut b, a, 0);
    call(&mut b, sink, vec![x]);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    assert!(lt.interfere(0, 1));
}

/// `a` is written before a loop and read after it; `c` is used only inside the loop. `a` must stay
/// live through every block of the loop, so the two interfere — and a slot used only inside the
/// loop (`c`) does not interfere with one used only after it (`d`).
#[test]
fn a_use_across_a_loop_keeps_the_slot_live_around_it() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let a = b.alloca(arr(8));
    let c = b.alloca(arr(8));
    let d = b.alloca(arr(8));
    write(&mut b, a, 0);
    let zero = i64c(&mut b, 0);
    let head = b.new_block();
    let iv = b.block_param(head, MirType::I64);
    let body = b.new_block();
    let exit = b.new_block();
    b.br(head, vec![zero]);
    b.switch_to(head);
    let ten = i64c(&mut b, 10);
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, iv, ten));
    b.cond_br(more, body, vec![], exit, vec![]);
    b.switch_to(body);
    write(&mut b, c, 2);
    let y = read(&mut b, c, 2);
    call(&mut b, sink, vec![y]);
    let one = i64c(&mut b, 1);
    let next = b.build(MirType::I64, Op::Bin(BinOp::Add, iv, one));
    b.br(head, vec![next]);
    b.switch_to(exit);
    let x = read(&mut b, a, 0);
    call(&mut b, sink, vec![x]);
    write(&mut b, d, 3);
    let z = read(&mut b, d, 3);
    call(&mut b, sink, vec![z]);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    assert!(lt.interfere(0, 1), "a is live through the loop that uses c");
    assert!(!lt.interfere(1, 2), "c is dead once the loop exits");
    assert!(
        !lt.interfere(0, 2),
        "a's last read comes before d's first write in the exit block"
    );
    // c is live at the loop header (between iterations), not only inside the body.
    assert!(lt.live_at(1, head.0 as usize, 0));
    // ... and before the loop it is not.
    assert!(!lt.live_at(1, 0, 0));
}

/// The address flows through a block parameter into a later block and is read there: the slot
/// stays live across the branch, so a slot used in between interferes.
#[test]
fn a_derived_pointer_through_a_block_parameter_extends_the_lifetime() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let a = b.alloca(arr(8));
    let c = b.alloca(arr(8));
    write(&mut b, a, 1);
    let four = i64c(&mut b, 4);
    let p = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: a,
            index: four,
            elem: MirType::F32,
        },
    );
    let next = b.new_block();
    let q = b.block_param(next, MirType::Ptr);
    b.br(next, vec![p]);
    b.switch_to(next);
    write(&mut b, c, 0);
    let y = read(&mut b, c, 0);
    call(&mut b, sink, vec![y]);
    let x = b.build(MirType::F32, Op::Load(q, MirType::F32));
    call(&mut b, sink, vec![x]);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    assert!(
        !lt.escapes(0),
        "a block parameter does not lose the address"
    );
    assert!(lt.interfere(0, 1));
}

/// A `select` between two slots is derived from both.
#[test]
fn a_select_is_derived_from_both_arms() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let flag = b.add_param(MirType::I1);
    let a = b.alloca(arr(8));
    let c = b.alloca(arr(8));
    let d = b.alloca(arr(8));
    write(&mut b, a, 0);
    write(&mut b, c, 0);
    let s = b.build(MirType::Ptr, Op::Select(flag, a, c));
    write(&mut b, d, 0);
    let x = b.build(MirType::F32, Op::Load(s, MirType::F32));
    call(&mut b, sink, vec![x]);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    assert!(lt.interfere(0, 2) && lt.interfere(1, 2));
    assert!(!lt.escapes(0) && !lt.escapes(1));
}

/// Builds, into `main`, one way for the slot (first argument) to escape; the second argument is a
/// pointer parameter.
type Escape = dyn Fn(&mut Builder, ValueId, ValueId);

/// Every way an address can leave the analysis makes the slot live for the whole function.
#[test]
fn escapes_make_a_slot_live_everywhere() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let ext = i.intern("c_keep");
    let cases: Vec<(&str, Box<Escape>)> = vec![
        (
            "ptrtoint",
            Box::new(|b: &mut Builder, a, _| {
                let n = b.build(MirType::I64, Op::Cast(CastKind::PtrToInt, a, MirType::I64));
                let sink = b.alloca(MirType::I64);
                b.build_void(Op::Store {
                    ptr: sink,
                    value: n,
                });
            }),
        ),
        (
            "stored into a parameter's storage",
            Box::new(|b: &mut Builder, a, out| {
                b.build_void(Op::Store { ptr: out, value: a });
            }),
        ),
        (
            "passed to an extern function",
            Box::new(move |b: &mut Builder, a, _| {
                b.build_void(Op::Call {
                    func: ext,
                    args: vec![a],
                });
            }),
        ),
        (
            "held by a slot that escapes",
            Box::new(|b: &mut Builder, a, _| {
                let holder = b.alloca(MirType::Ptr);
                b.build_void(Op::Store {
                    ptr: holder,
                    value: a,
                });
                let n = b.build(
                    MirType::I64,
                    Op::Cast(CastKind::PtrToInt, holder, MirType::I64),
                );
                let sink = b.alloca(MirType::I64);
                b.build_void(Op::Store {
                    ptr: sink,
                    value: n,
                });
            }),
        ),
        (
            "fed to arithmetic after a round trip through memory",
            Box::new(|b: &mut Builder, a, out| {
                let holder = b.alloca(MirType::Ptr);
                b.build_void(Op::Store {
                    ptr: holder,
                    value: a,
                });
                let bits = b.build(MirType::I64, Op::Load(holder, MirType::I64));
                let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
                let moved = b.build(MirType::I64, Op::Bin(BinOp::Add, bits, one));
                b.build_void(Op::Store {
                    ptr: out,
                    value: moved,
                });
            }),
        ),
    ];
    for (what, make) in cases {
        let mut b = Builder::new(main, MirType::Void);
        let out = b.add_param(MirType::Ptr);
        let a = b.alloca(arr(8));
        let c = b.alloca(arr(8));
        write(&mut b, c, 0);
        make(&mut b, a, out);
        b.ret(None);
        let mut p = program_of(vec![b.finish()]);
        p.externs.push(wukong_mir::ExternFn {
            name: ext,
            params: vec![wukong_mir::ExternParam {
                ty: MirType::Ptr,
                signed: false,
            }],
            ret: wukong_mir::ExternParam {
                ty: MirType::Void,
                signed: false,
            },
        });
        let lt = lifetimes(&p, main);
        assert!(lt.escapes(0), "{what}: the slot should escape");
        assert!(!lt.escapes(1), "{what}: the other slot is untouched");
        assert!(lt.interfere(0, 1), "{what}");
        assert!(lt.live_at(0, 0, 0), "{what}: live everywhere");
    }
}

/// Returning a derived pointer escapes it.
#[test]
fn a_returned_address_escapes() {
    let mut i = Interner::new();
    let f = i.intern("f");
    let mut b = Builder::new(f, MirType::Ptr);
    let a = b.alloca(arr(8));
    let k = i64c(&mut b, 2);
    let p = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: a,
            index: k,
            elem: MirType::F32,
        },
    );
    b.ret(Some(p));
    let prog = program_of(vec![b.finish()]);
    assert!(lifetimes(&prog, f).escapes(0));
}

/// A user callee that only reads its parameter does not capture it, so the caller's slot is used
/// at the call and nowhere else; one that stores its parameter into another parameter's storage
/// captures it, and so does a caller that forwards the pointer to such a callee (transitively,
/// through recursion too).
#[test]
fn the_capture_summary_follows_the_call_graph() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let reader = i.intern("reader");
    let keeper = i.intern("keeper");
    let forward = i.intern("forward");
    let rec = i.intern("rec");

    // reader(p) -> f32 { return p[0] }
    let mut b = Builder::new(reader, MirType::F32);
    let p = b.add_param(MirType::Ptr);
    let x = read(&mut b, p, 0);
    b.ret(Some(x));
    let reader_fn = b.finish();

    // keeper(p, out) { *out = p }
    let mut b = Builder::new(keeper, MirType::Void);
    let p = b.add_param(MirType::Ptr);
    let out = b.add_param(MirType::Ptr);
    b.build_void(Op::Store { ptr: out, value: p });
    b.ret(None);
    let keeper_fn = b.finish();

    // forward(p, out) { keeper(p, out) }
    let mut b = Builder::new(forward, MirType::Void);
    let p = b.add_param(MirType::Ptr);
    let out = b.add_param(MirType::Ptr);
    call(&mut b, keeper, vec![p, out]);
    b.ret(None);
    let forward_fn = b.finish();

    // rec(p, n) { if n > 0 { rec(p, n - 1) } }   — recursion alone captures nothing
    let mut b = Builder::new(rec, MirType::Void);
    let p = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    let zero = i64c(&mut b, 0);
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, n, zero));
    let again = b.new_block();
    let done = b.new_block();
    b.cond_br(more, again, vec![], done, vec![]);
    b.switch_to(again);
    let one = i64c(&mut b, 1);
    let m = b.build(MirType::I64, Op::Bin(BinOp::Sub, n, one));
    call(&mut b, rec, vec![p, m]);
    b.ret(None);
    b.switch_to(done);
    b.ret(None);
    let rec_fn = b.finish();

    // main(out) { a: reader(a); c: forward(c, out); d: rec(d, 3); e: written after everything }
    let mut b = Builder::new(main, MirType::Void);
    let out = b.add_param(MirType::Ptr);
    let a = b.alloca(arr(8));
    let c = b.alloca(arr(8));
    let d = b.alloca(arr(8));
    let e = b.alloca(arr(8));
    write(&mut b, a, 0);
    let _ = b.build(
        MirType::F32,
        Op::Call {
            func: reader,
            args: vec![a],
        },
    );
    call(&mut b, forward, vec![c, out]);
    let three = i64c(&mut b, 3);
    call(&mut b, rec, vec![d, three]);
    write(&mut b, e, 0);
    b.ret(None);
    let p = program_of(vec![b.finish(), reader_fn, keeper_fn, forward_fn, rec_fn]);

    let calls = CaptureSummary::analyze(&p);
    assert!(!calls.captures(reader, 0));
    assert!(calls.captures(keeper, 0));
    assert!(
        !calls.captures(keeper, 1),
        "the destination is written, not retained"
    );
    assert!(
        calls.captures(forward, 0),
        "a capture reaches the caller that forwards"
    );
    assert!(
        !calls.captures(rec, 0),
        "self-recursion alone captures nothing"
    );

    let lt = lifetimes(&p, main);
    assert!(!lt.escapes(0), "reader does not capture");
    assert!(lt.escapes(1), "forward captures");
    assert!(!lt.escapes(2), "rec does not capture");
    assert!(!lt.interfere(0, 3), "a is dead after the call to reader");
    assert!(!lt.interfere(2, 3), "d is dead after the call to rec");
    assert!(
        lt.interfere(1, 3),
        "an escaped slot interferes with everything"
    );
}

/// A runtime entry point does not retain its arguments: the slot is used at the call only. The
/// environment of a `@parallel` region is different — the runtime calls user code with it — so
/// an address captured in it (stored into the environment slot) escapes, and so does the
/// environment itself.
#[test]
fn a_parallel_region_capture_escapes_but_a_runtime_kernel_argument_does_not() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let body = i.intern("body");
    let sgemv = i.intern("wukong_sgemv");
    let par = i.intern("wukong_parallel_for");

    let mut b = Builder::new(body, MirType::Void);
    let _ = b.add_param(MirType::I64);
    let _ = b.add_param(MirType::Ptr);
    b.ret(None);
    let body_fn = b.finish();

    let mut b = Builder::new(main, MirType::Void);
    let a = b.alloca(arr(16));
    let x = b.alloca(arr(4));
    let y = b.alloca(arr(4));
    let cap = b.alloca(arr(8));
    let env = b.alloca(MirType::Array(Box::new(MirType::Ptr), 1));
    let later = b.alloca(arr(16));
    write(&mut b, a, 0);
    write(&mut b, x, 0);
    let four = i64c(&mut b, 4);
    call(&mut b, sgemv, vec![a, x, y, four, four]);
    let v = read(&mut b, y, 0);
    let _ = v;
    b.build_void(Op::Store {
        ptr: env,
        value: cap,
    });
    let g = b.build(MirType::Ptr, Op::FuncAddr(body));
    let n = i64c(&mut b, 8);
    call(&mut b, par, vec![n, g, env]);
    write(&mut b, later, 0);
    b.ret(None);
    let p = program_of(vec![b.finish(), body_fn]);
    let lt = lifetimes(&p, main);
    let idx = |v: ValueId| lt.index_of(v).unwrap();
    for s in [a, x, y] {
        assert!(!lt.escapes(idx(s)), "a runtime kernel retains nothing");
        assert!(!lt.interfere(idx(s), idx(later)));
    }
    assert!(
        lt.interfere(idx(a), idx(y)),
        "both are arguments of one call"
    );
    assert!(lt.escapes(idx(cap)), "captured by the parallel region");
    assert!(lt.escapes(idx(env)), "handed to user code by the runtime");
}

/// A vector kernel's stream table holds the stream addresses: the streams are used where the
/// kernel is called, not only where their addresses were stored into the table, and they do not
/// escape.
#[test]
fn a_vector_kernel_uses_the_streams_its_table_holds() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let src = b.alloca(arr(64));
    let table = b.alloca(MirType::Array(Box::new(MirType::Ptr), 1));
    let scalars = b.alloca(arr(1));
    let mid = b.alloca(arr(64));
    let after = b.alloca(arr(64));
    write(&mut b, src, 0);
    b.build_void(Op::Store {
        ptr: table,
        value: src,
    });
    // `mid` is used between the table store and the kernel call: `src` must still be live there.
    write(&mut b, mid, 0);
    let y = read(&mut b, mid, 0);
    call(&mut b, sink, vec![y]);
    // out[i] = src[i] + scalar: one stream is read and written in place.
    let k = b.add_vec_kernel(VecKernel {
        name: i.intern("k"),
        lane: VecLane::F32,
        streams: 1,
        scalars: 1,
        unroll: 1,
        ops: vec![
            VecOp::Load { stream: 0 },
            VecOp::Splat { scalar: 0 },
            VecOp::Bin {
                op: VecBin::Add,
                a: 0,
                b: 1,
            },
            VecOp::Store { stream: 0, val: 2 },
        ],
        reduce: None,
    });
    let n = i64c(&mut b, 64);
    b.build_void(Op::VecKernelCall {
        kernel: k,
        ptrs: table,
        scalars,
        n,
    });
    write(&mut b, after, 0);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    let idx = |v: ValueId| lt.index_of(v).unwrap();
    assert!(
        !lt.escapes(idx(src)),
        "a stream table does not lose the address"
    );
    assert!(
        lt.interfere(idx(src), idx(mid)),
        "the kernel reads src after mid's uses"
    );
    assert!(!lt.interfere(idx(src), idx(after)));
    assert!(!lt.interfere(idx(table), idx(after)));
}

/// A slot that holds an address passes it on to whatever is loaded from it: a slice's fat pointer
/// keeps its data array live wherever the data pointer read back out of it is used.
#[test]
fn an_address_stored_in_a_slot_follows_the_loads_from_it() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let data = b.alloca(arr(16));
    let header = b.alloca(MirType::Array(Box::new(MirType::I8), 16));
    let other = b.alloca(arr(16));
    write(&mut b, data, 3);
    b.build_void(Op::Store {
        ptr: header,
        value: data,
    });
    write(&mut b, other, 0);
    let o = read(&mut b, other, 0);
    call(&mut b, sink, vec![o]);
    let dp = b.build(MirType::Ptr, Op::Load(header, MirType::Ptr));
    let x = read(&mut b, dp, 3);
    call(&mut b, sink, vec![x]);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    let idx = |v: ValueId| lt.index_of(v).unwrap();
    assert!(!lt.escapes(idx(data)));
    assert!(lt.interfere(idx(data), idx(other)));
}

/// Passing a slot that holds an address to a call hands the held address over too: it escapes.
#[test]
fn a_held_address_passed_to_a_call_escapes() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let reader = i.intern("reader");
    let mut b = Builder::new(reader, MirType::Void);
    let _ = b.add_param(MirType::Ptr);
    b.ret(None);
    let reader_fn = b.finish();

    let mut b = Builder::new(main, MirType::Void);
    let data = b.alloca(arr(16));
    let header = b.alloca(MirType::Array(Box::new(MirType::I8), 16));
    b.build_void(Op::Store {
        ptr: header,
        value: data,
    });
    call(&mut b, reader, vec![header]);
    b.ret(None);
    let p = program_of(vec![b.finish(), reader_fn]);
    let lt = lifetimes(&p, main);
    assert!(lt.escapes(0), "the callee could keep what the header holds");
    assert!(
        !lt.escapes(1),
        "the header itself is only read by a non-capturing callee"
    );
}

#[test]
fn range_helpers() {
    assert!(ranges_intersect(&[(0, 3), (10, 12)], &[(4, 9), (12, 20)]));
    assert!(!ranges_intersect(&[(0, 3), (10, 11)], &[(4, 9), (12, 20)]));
    assert!(!ranges_intersect(&[], &[(0, 9)]));
    assert_eq!(
        ranges_union(&[(0, 3), (10, 11)], &[(4, 9), (13, 20)]),
        vec![(0, 11), (13, 20)]
    );
}

/// How a test's fill loop over `s` deviates from a full overwrite.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Fill {
    /// `for i in 0..64 { s[i] = 1.0 }`: every byte, nothing read.
    Full,
    /// `for i in 0..32`: half the array.
    Half,
    /// The body also reads `s[i]` before writing it.
    ReadsInside,
    /// The store runs only on even iterations.
    Conditional,
    /// The body can leave the loop early (a `break` to the code after it).
    EarlyExit,
    /// The store is `s[i + 1]`: shifted, so `s[0]` is never written.
    Shifted,
    /// Unrolled by two: `i` steps by 2 and the body stores `s[i]` and `s[i + 1]`.
    Unrolled,
    /// `i` steps by 2 and the body stores only `s[i]`: every odd element is left alone.
    Strided,
}

/// `for i in 0..end { <store 1.0 -> slot[i + shift]> }`, shaped by `how`. Leaves the builder in the
/// block after the loop.
fn fill_loop(b: &mut Builder, slot: ValueId, how: Fill) {
    let (end, shift) = match how {
        Fill::Half => (32, 0),
        Fill::Shifted => (63, 1),
        _ => (64, 0),
    };
    let stride = if matches!(how, Fill::Unrolled | Fill::Strided) {
        2
    } else {
        1
    };
    let zero = i64c(b, 0);
    let head = b.new_block();
    let iv = b.block_param(head, MirType::I64);
    let body = b.new_block();
    let exit = b.new_block();
    b.br(head, vec![zero]);
    b.switch_to(head);
    let n = i64c(b, end);
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, iv, n));
    b.cond_br(more, body, vec![], exit, vec![]);
    b.switch_to(body);
    let k = i64c(b, shift);
    let idx = b.build(MirType::I64, Op::Bin(BinOp::Add, iv, k));
    let at = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: slot,
            index: idx,
            elem: MirType::F32,
        },
    );
    let one = f32c(b, 1.0);
    let latch = b.new_block();
    match how {
        Fill::ReadsInside => {
            let old = b.build(MirType::F32, Op::Load(at, MirType::F32));
            let v = b.build(MirType::F32, Op::Bin(BinOp::FAdd, old, one));
            b.build_void(Op::Store { ptr: at, value: v });
            b.br(latch, vec![]);
        }
        Fill::Conditional => {
            let two = i64c(b, 2);
            let r = b.build(MirType::I64, Op::Bin(BinOp::SRem, iv, two));
            let z = i64c(b, 0);
            let even = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, r, z));
            let write = b.new_block();
            b.cond_br(even, write, vec![], latch, vec![]);
            b.switch_to(write);
            b.build_void(Op::Store {
                ptr: at,
                value: one,
            });
            b.br(latch, vec![]);
        }
        Fill::EarlyExit => {
            b.build_void(Op::Store {
                ptr: at,
                value: one,
            });
            let ten = i64c(b, 10);
            let stop = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, iv, ten));
            b.cond_br(stop, exit, vec![], latch, vec![]);
        }
        Fill::Unrolled => {
            b.build_void(Op::Store {
                ptr: at,
                value: one,
            });
            let k1 = i64c(b, 1);
            let idx1 = b.build(MirType::I64, Op::Bin(BinOp::Add, iv, k1));
            let at1 = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: slot,
                    index: idx1,
                    elem: MirType::F32,
                },
            );
            b.build_void(Op::Store {
                ptr: at1,
                value: one,
            });
            b.br(latch, vec![]);
        }
        _ => {
            b.build_void(Op::Store {
                ptr: at,
                value: one,
            });
            b.br(latch, vec![]);
        }
    }
    b.switch_to(latch);
    let step = i64c(b, stride);
    let next = b.build(MirType::I64, Op::Bin(BinOp::Add, iv, step));
    b.br(head, vec![next]);
    b.switch_to(exit);
}

/// Per-iteration scratch: inside an outer loop, `s` is filled (as `how` says) and read, then `t`
/// is fully filled and read. Returns whether the two interfere.
fn per_iteration_scratch(how: Fill) -> bool {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let s = b.alloca(arr(64));
    let t = b.alloca(arr(64));
    let zero = i64c(&mut b, 0);
    let outer = b.new_block();
    let h = b.block_param(outer, MirType::I64);
    let obody = b.new_block();
    let done = b.new_block();
    b.br(outer, vec![zero]);
    b.switch_to(outer);
    let four = i64c(&mut b, 4);
    let more = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, h, four));
    b.cond_br(more, obody, vec![], done, vec![]);
    b.switch_to(obody);
    fill_loop(&mut b, s, how);
    let x = read(&mut b, s, 3);
    call(&mut b, sink, vec![x]);
    fill_loop(&mut b, t, Fill::Full);
    let y = read(&mut b, t, 5);
    call(&mut b, sink, vec![y]);
    let one = i64c(&mut b, 1);
    let h1 = b.build(MirType::I64, Op::Bin(BinOp::Add, h, one));
    b.br(outer, vec![h1]);
    b.switch_to(done);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes(&p, main);
    assert!(!lt.escapes(0) && !lt.escapes(1));
    lt.interfere(0, 1)
}

/// A full overwrite at the top of every iteration kills the previous iteration's contents, so two
/// per-iteration scratch arrays used one after the other share; every way the overwrite can fail
/// to be full keeps the slot live around the outer loop, and the two interfere again.
#[test]
fn a_full_overwrite_separates_per_iteration_scratch() {
    for how in [Fill::Full, Fill::Unrolled] {
        assert!(
            !per_iteration_scratch(how),
            "{how:?}: a full re-initialization kills the slot on entry to its fill loop"
        );
    }
    for how in [
        Fill::Half,
        Fill::ReadsInside,
        Fill::Conditional,
        Fill::EarlyExit,
        Fill::Shifted,
        Fill::Strided,
    ] {
        assert!(
            per_iteration_scratch(how),
            "{how:?} does not overwrite every byte before a read, so it kills nothing"
        );
    }
}

/// The same separation from source, through the real front end at -O0: scratch arrays declared
/// inside a loop body are re-initialized by their `[v; N]` initializers on every iteration.
#[test]
fn scratch_declared_in_a_loop_body_shares_at_o0() {
    let src = "
fn main() -> i32 {
    let mut total: f32 = 0.0;
    for h in 0..4 {
        let mut s: [f32; 64] = [0.0; 64];
        for i in 0..64 { s[i] = (i + h) as f32; }
        let mut m: f32 = 0.0;
        for i in 0..64 { m = m + s[i]; }
        let mut t: [f32; 64] = [1.0; 64];
        for i in 0..64 { t[i] = t[i] * m; }
        for i in 0..64 { total = total + t[i]; }
    }
    print(total);
    return 0;
}
";
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "{pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
    let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(ld.iter().all(|d| !d.is_error()), "{ld:?}");
    let main = interner.intern("main");
    let lt = lifetimes(&program, main);
    let f = program.function(main).unwrap();
    let arrays: Vec<usize> = lt
        .allocas()
        .enumerate()
        .filter(|&(_, v)| {
            f.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                i.result == Some(v) && matches!(&i.op, Op::Alloca { ty, .. } if *ty == arr(64))
            })
        })
        .map(|(k, _)| k)
        .collect();
    assert_eq!(arrays.len(), 2, "s and t");
    assert!(!lt.interfere(arrays[0], arrays[1]));
}

/// A runtime entry point that calls user code back is recognized even where its function argument
/// is no longer a visible `func_addr`: one routed through a block parameter, and — once any call
/// site in the program hands that entry point a function address — one that arrives as a plain
/// parameter.
#[test]
fn a_callback_runtime_captures_through_block_parameters_and_parameters() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let body = i.intern("body");
    let other = i.intern("other");
    let par = i.intern("wukong_parallel_for");

    let mut b = Builder::new(body, MirType::Void);
    let _ = b.add_param(MirType::I64);
    let _ = b.add_param(MirType::Ptr);
    b.ret(None);
    let body_fn = b.finish();

    // main: the body address reaches the call through a block parameter.
    let mut b = Builder::new(main, MirType::Void);
    let cap = b.alloca(arr(8));
    let env = b.alloca(MirType::Array(Box::new(MirType::Ptr), 1));
    b.build_void(Op::Store {
        ptr: env,
        value: cap,
    });
    let g = b.build(MirType::Ptr, Op::FuncAddr(body));
    let next = b.new_block();
    let p = b.block_param(next, MirType::Ptr);
    b.br(next, vec![g]);
    b.switch_to(next);
    let n = i64c(&mut b, 8);
    call(&mut b, par, vec![n, p, env]);
    b.ret(None);
    let main_fn = b.finish();

    // other(fp): the body address is a parameter.
    let mut b = Builder::new(other, MirType::Void);
    let fp = b.add_param(MirType::Ptr);
    let cap2 = b.alloca(arr(8));
    let env2 = b.alloca(MirType::Array(Box::new(MirType::Ptr), 1));
    b.build_void(Op::Store {
        ptr: env2,
        value: cap2,
    });
    let n = i64c(&mut b, 8);
    call(&mut b, par, vec![n, fp, env2]);
    b.ret(None);
    let other_fn = b.finish();

    let p = program_of(vec![main_fn, other_fn, body_fn]);
    assert!(CaptureSummary::analyze(&p).captures(par, 2));
    let lt = lifetimes(&p, main);
    assert!(lt.escapes(0) && lt.escapes(1), "main: cap and env");
    let lt = lifetimes(&p, other);
    assert!(lt.escapes(0) && lt.escapes(1), "other: cap2 and env2");
}
