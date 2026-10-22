//! The full-overwrite kills on hand-built MIR and from source: loop nests, counter slots, the `-O2`
//! loop shapes, runtime kernels and vector kernels — each shape that is recognized, and for each
//! the ways it can fail to overwrite every byte before a read (which must keep the slot live).
//!
//! Every test asks the same question: do two scratch arrays, each fully rewritten before it is
//! read, interfere? With the rewrite recognized they do not; with it not recognized (or not a full
//! overwrite) the first array's old contents stay live across the second's use, and they do.

use super::super::{CaptureSummary, KernelOverwrite, SlotLifetimes};
use wukong_mir::{
    BinOp, BlockId, Builder, CastKind, CmpOp, Function, MirType, Op, Program, ValueId, VecBin,
    VecKernel, VecLane, VecOp,
};
use wukong_span::{Interner, Symbol};

fn arr(n: u32) -> MirType {
    MirType::Array(Box::new(MirType::F32), n)
}

fn int(b: &mut Builder, ty: &MirType, v: i128) -> ValueId {
    b.build(ty.clone(), Op::ConstInt(v, ty.clone()))
}

fn i64c(b: &mut Builder, v: i128) -> ValueId {
    int(b, &MirType::I64, v)
}

fn f32c(b: &mut Builder, v: f64) -> ValueId {
    b.build(MirType::F32, Op::ConstFloat(v, MirType::F32))
}

/// `&p[idx]` over `f32` elements (`idx` of any integer type; a narrow one is sign-extended).
fn elem(b: &mut Builder, p: ValueId, idx: ValueId) -> ValueId {
    let idx = if b.value_type(idx) == &MirType::I64 {
        idx
    } else {
        b.build(MirType::I64, Op::Cast(CastKind::SExt, idx, MirType::I64))
    };
    b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: p,
            index: idx,
            elem: MirType::F32,
        },
    )
}

/// `p[k] = 1.0` at a constant index.
fn write_at(b: &mut Builder, p: ValueId, k: i128) {
    let i = i64c(b, k);
    let at = elem(b, p, i);
    let one = f32c(b, 1.0);
    b.build_void(Op::Store {
        ptr: at,
        value: one,
    });
}

/// `print(p[k])`.
fn read_at(b: &mut Builder, p: ValueId, k: i128, sink: Symbol) {
    let i = i64c(b, k);
    let at = elem(b, p, i);
    let v = b.build(MirType::F32, Op::Load(at, MirType::F32));
    b.build_void(Op::Call {
        func: sink,
        args: vec![v],
    });
}

fn bin(b: &mut Builder, op: BinOp, x: ValueId, y: ValueId) -> ValueId {
    let ty = b.value_type(x).clone();
    b.build(ty, Op::Bin(op, x, y))
}

/// An open counted loop: its header, induction variable and exit block. The builder sits in the
/// body.
struct Lp {
    head: BlockId,
    iv: ValueId,
    exit: BlockId,
}

/// Open `for iv = start; iv <pred> end; ..` (block-parameter induction variable).
fn open(b: &mut Builder, start: ValueId, pred: CmpOp, end: i128) -> Lp {
    let ty = b.value_type(start).clone();
    let head = b.new_block();
    let iv = b.block_param(head, ty.clone());
    let body = b.new_block();
    let exit = b.new_block();
    b.br(head, vec![start]);
    b.switch_to(head);
    let bound = int(b, &ty, end);
    let more = b.build(MirType::I1, Op::Cmp(pred, iv, bound));
    b.cond_br(more, body, vec![], exit, vec![]);
    b.switch_to(body);
    Lp { head, iv, exit }
}

/// Close loop `l` from the current block: `iv + stride` back to the header; the builder moves to
/// the exit block.
fn close(b: &mut Builder, l: &Lp, stride: i128) {
    let ty = b.value_type(l.iv).clone();
    let k = int(b, &ty, stride);
    let next = bin(b, BinOp::Add, l.iv, k);
    b.br(l.head, vec![next]);
    b.switch_to(l.exit);
}

/// Analyze `main` of `p`, with the program's capture summary extended by `overwrites`.
fn lifetimes_with(
    p: &Program,
    main: Symbol,
    overwrites: Vec<(Symbol, KernelOverwrite)>,
) -> SlotLifetimes {
    let errors = wukong_mir::verify::verify_program(p);
    assert!(errors.is_empty(), "{errors:?}");
    let calls = CaptureSummary::analyze(p).with_overwrites(overwrites);
    SlotLifetimes::analyze(p.function(main).unwrap(), &calls)
}

fn program_of(funcs: Vec<Function>) -> Program {
    let mut p = Program::new();
    p.funcs = funcs;
    p
}

/// Inside an outer loop, `s` is filled by `fill_s` and read, then `t` is filled by `fill_t` and
/// read; both are `len` floats. Do `s` and `t` interfere?
fn per_iteration(
    len: u32,
    fill_s: &dyn Fn(&mut Builder, ValueId),
    fill_t: &dyn Fn(&mut Builder, ValueId),
) -> bool {
    per_iteration_in(&mut Interner::new(), len, fill_s, fill_t, Vec::new())
}

/// [`per_iteration`] with `overwrites` registered, interning through `i`.
fn per_iteration_in(
    i: &mut Interner,
    len: u32,
    fill_s: &dyn Fn(&mut Builder, ValueId),
    fill_t: &dyn Fn(&mut Builder, ValueId),
    overwrites: Vec<(Symbol, KernelOverwrite)>,
) -> bool {
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let s = b.alloca(arr(len));
    let t = b.alloca(arr(len));
    let zero = i64c(&mut b, 0);
    let outer = open(&mut b, zero, CmpOp::Slt, 4);
    fill_s(&mut b, s);
    read_at(&mut b, s, 3, sink);
    fill_t(&mut b, t);
    read_at(&mut b, t, 5, sink);
    close(&mut b, &outer, 1);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes_with(&p, main, overwrites);
    let (si, ti) = (lt.index_of(s).unwrap(), lt.index_of(t).unwrap());
    assert!(!lt.escapes(si) && !lt.escapes(ti));
    lt.interfere(si, ti)
}

/// `for i in 0..len { p[i] = 1.0 }`.
fn fill_1d(b: &mut Builder, p: ValueId, len: i128) {
    let zero = i64c(b, 0);
    let l = open(b, zero, CmpOp::Slt, len);
    let at = elem(b, p, l.iv);
    let one = f32c(b, 1.0);
    b.build_void(Op::Store {
        ptr: at,
        value: one,
    });
    close(b, &l, 1);
}

/// How a two-level nest over a 512-float slot deviates from a full overwrite.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Nest {
    /// `for i in 0..8 { for j in 0..64 { s[i*64 + j] = 1.0 } }`.
    RowMajor,
    /// `for i in 0..8 { for j in 0..64 { s[j*8 + i] = 1.0 } }`: the inner loop takes the wide
    /// stride, which only the outer one closes.
    ColumnMajor,
    /// `for i in 0..8 { for j in 0..64 { s[i*64 + j] = 1.0 } }` in `i32` arithmetic, sign-extended.
    Narrow,
    /// The inner loop stops at 63: the last element of every row is never written.
    ShortRows,
    /// `s[i*128 + j]` for `i in 0..4`: rows of 64 with gaps of 64 between them.
    Gaps,
    /// The inner body reads `s[i*64 + j]` before writing it.
    ReadsInside,
    /// `for j in i..64`: the inner loop's start depends on the outer one.
    Triangular,
    /// The store runs only when `j` is even.
    Conditional,
}

fn nest_fill(b: &mut Builder, s: ValueId, how: Nest) {
    let ty = if how == Nest::Narrow {
        MirType::I32
    } else {
        MirType::I64
    };
    let rows = if how == Nest::Gaps { 4 } else { 8 };
    let zero = int(b, &ty, 0);
    let outer = open(b, zero, CmpOp::Slt, rows);
    let jstart = if how == Nest::Triangular {
        outer.iv
    } else {
        int(b, &ty, 0)
    };
    let cols = if how == Nest::ShortRows { 63 } else { 64 };
    let inner = open(b, jstart, CmpOp::Slt, cols);
    let (i, j) = (outer.iv, inner.iv);
    let idx = match how {
        Nest::ColumnMajor => {
            let eight = int(b, &ty, 8);
            let row = bin(b, BinOp::Mul, j, eight);
            bin(b, BinOp::Add, row, i)
        }
        Nest::Gaps => {
            let w = int(b, &ty, 128);
            let row = bin(b, BinOp::Mul, i, w);
            bin(b, BinOp::Add, row, j)
        }
        _ => {
            let w = int(b, &ty, 64);
            let row = bin(b, BinOp::Mul, i, w);
            bin(b, BinOp::Add, row, j)
        }
    };
    let at = elem(b, s, idx);
    let one = f32c(b, 1.0);
    match how {
        Nest::ReadsInside => {
            let old = b.build(MirType::F32, Op::Load(at, MirType::F32));
            let v = b.build(MirType::F32, Op::Bin(BinOp::FAdd, old, one));
            b.build_void(Op::Store { ptr: at, value: v });
        }
        Nest::Conditional => {
            let two = int(b, &ty, 2);
            let r = bin(b, BinOp::SRem, j, two);
            let z = int(b, &ty, 0);
            let even = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, r, z));
            let write = b.new_block();
            let latch = b.new_block();
            b.cond_br(even, write, vec![], latch, vec![]);
            b.switch_to(write);
            b.build_void(Op::Store {
                ptr: at,
                value: one,
            });
            b.br(latch, vec![]);
            b.switch_to(latch);
        }
        _ => b.build_void(Op::Store {
            ptr: at,
            value: one,
        }),
    }
    close(b, &inner, 1);
    close(b, &outer, 1);
}

/// A nest whose iteration space tiles the slot is a full overwrite in either loop order and in
/// narrow (sign-extended) index arithmetic; every way of missing a byte, reading first, or
/// depending on the outer variable keeps the slot live around the outer loop.
#[test]
fn a_nest_that_tiles_the_slot_is_a_full_overwrite() {
    let full = |b: &mut Builder, t: ValueId| fill_1d(b, t, 512);
    for how in [Nest::RowMajor, Nest::ColumnMajor, Nest::Narrow] {
        assert!(
            !per_iteration(512, &|b, s| nest_fill(b, s, how), &full),
            "{how:?}: the nest rewrites the whole slot every outer iteration"
        );
    }
    for how in [
        Nest::ShortRows,
        Nest::Gaps,
        Nest::ReadsInside,
        Nest::Triangular,
        Nest::Conditional,
    ] {
        assert!(
            per_iteration(512, &|b, s| nest_fill(b, s, how), &full),
            "{how:?} is not a full overwrite"
        );
    }
}

/// How an `-O2` fill loop over a slot is shaped.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape {
    /// The vectorizer's main loop: `iv != 64`, step 4, one `<4 x f32>` store per iteration.
    Vector,
    /// The unroller's body: `iv < 61`, four stores at `iv`, `iv+1`, `iv+2`, `iv+3`, and the
    /// variable advanced by a chain of four adds.
    ChainedAdds,
    /// A vector loop over the first 8 floats of a 10-float slot, then the scalar remainder that
    /// starts where it stopped (`8..10`).
    VectorThenRemainder,
    /// The same with a remainder that stops at 9.
    VectorThenShortRemainder,
    /// A vector loop whose `!=` test the variable can step over (`iv != 62`, step 4).
    SkippedBound,
}

fn shaped_fill(b: &mut Builder, s: ValueId, shape: Shape) {
    let one = f32c(b, 1.0);
    let zero = i64c(b, 0);
    match shape {
        Shape::Vector | Shape::SkippedBound => {
            let end = if shape == Shape::Vector { 64 } else { 62 };
            let lanes = b.build(MirType::Vec(Box::new(MirType::F32), 4), Op::Splat(one));
            let l = open(b, zero, CmpOp::Ne, end);
            let at = elem(b, s, l.iv);
            b.build_void(Op::Store {
                ptr: at,
                value: lanes,
            });
            close(b, &l, 4);
        }
        Shape::ChainedAdds => {
            let l = open(b, zero, CmpOp::Slt, 61);
            let mut i = l.iv;
            for _ in 0..4 {
                let at = elem(b, s, i);
                b.build_void(Op::Store {
                    ptr: at,
                    value: one,
                });
                let k = i64c(b, 1);
                i = bin(b, BinOp::Add, i, k);
            }
            b.br(l.head, vec![i]);
            b.switch_to(l.exit);
        }
        Shape::VectorThenRemainder | Shape::VectorThenShortRemainder => {
            let lanes = b.build(MirType::Vec(Box::new(MirType::F32), 4), Op::Splat(one));
            // Vector part: `iv != 8`, step 4; its exit passes the variable on to the remainder.
            let head = b.new_block();
            let iv = b.block_param(head, MirType::I64);
            let body = b.new_block();
            let rem = b.new_block();
            let riv = b.block_param(rem, MirType::I64);
            let rbody = b.new_block();
            let done = b.new_block();
            b.br(head, vec![zero]);
            b.switch_to(head);
            let eight = i64c(b, 8);
            let more = b.build(MirType::I1, Op::Cmp(CmpOp::Ne, iv, eight));
            b.cond_br(more, body, vec![], rem, vec![iv]);
            b.switch_to(body);
            let at = elem(b, s, iv);
            b.build_void(Op::Store {
                ptr: at,
                value: lanes,
            });
            let four = i64c(b, 4);
            let next = bin(b, BinOp::Add, iv, four);
            b.br(head, vec![next]);
            // Remainder: `riv < end`, step 1.
            b.switch_to(rem);
            let end = if shape == Shape::VectorThenRemainder {
                10
            } else {
                9
            };
            let bound = i64c(b, end);
            let more = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, riv, bound));
            b.cond_br(more, rbody, vec![], done, vec![]);
            b.switch_to(rbody);
            let at = elem(b, s, riv);
            b.build_void(Op::Store {
                ptr: at,
                value: one,
            });
            let k = i64c(b, 1);
            let rnext = bin(b, BinOp::Add, riv, k);
            b.br(rem, vec![rnext]);
            b.switch_to(done);
        }
    }
}

/// The loop shapes `-O2` leaves behind are full overwrites when they cover the slot: a vector body
/// tested with `!=`, an unrolled body advancing its variable through a chain of adds, and a vector
/// body followed by the remainder loop that finishes the slot. A remainder that stops short, or a
/// `!=` test the variable steps over, is not.
#[test]
fn the_o2_loop_shapes_are_full_overwrites() {
    for (shape, len) in [
        (Shape::Vector, 64),
        (Shape::ChainedAdds, 64),
        (Shape::VectorThenRemainder, 10),
    ] {
        let full = move |b: &mut Builder, t: ValueId| fill_1d(b, t, i128::from(len));
        assert!(
            !per_iteration(len, &|b, s| shaped_fill(b, s, shape), &full),
            "{shape:?}: the loops rewrite the whole slot every outer iteration"
        );
    }
    for (shape, len) in [
        (Shape::VectorThenShortRemainder, 10),
        (Shape::SkippedBound, 64),
    ] {
        let full = move |b: &mut Builder, t: ValueId| fill_1d(b, t, i128::from(len));
        assert!(
            per_iteration(len, &|b, s| shaped_fill(b, s, shape), &full),
            "{shape:?} is not a full overwrite"
        );
    }
}

/// How a test call to the kernel `k(x, out, n, op)` — promised to overwrite `out[0..n)` when `op`
/// is 0 — is made or declared.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kern {
    /// `k(src, s, 64, 0)` with the promise registered.
    Promised,
    /// `n = 32`: half the slot.
    Short,
    /// `op = 1`, a value the promise does not cover.
    OtherOp,
    /// `k(s, s, 64, 0)`: the kernel reads its output in place.
    InPlace,
    /// The output is also reachable through another operand: a slot holding `s`'s address.
    HeldElsewhere,
    /// No promise registered.
    Unpromised,
    /// `k` is a user function of the program, so a promise under its name is never consulted.
    UserFunction,
    /// `n` is a parameter, not a constant.
    UnknownExtent,
}

/// `s` and `t` are written at the top, then each is filled by a call to `k` and read. Do they
/// interfere?
fn kernel_scratch(how: Kern) -> bool {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let k = i.intern("k");
    let mut b = Builder::new(main, MirType::Void);
    let n_param = b.add_param(MirType::I64);
    let src = b.alloca(arr(64));
    let s = b.alloca(arr(64));
    let t = b.alloca(arr(64));
    let holder = b.alloca(MirType::Array(Box::new(MirType::Ptr), 1));
    write_at(&mut b, src, 0);
    write_at(&mut b, s, 0);
    write_at(&mut b, t, 0);
    let n = match how {
        Kern::Short => i64c(&mut b, 32),
        Kern::UnknownExtent => n_param,
        _ => i64c(&mut b, 64),
    };
    let op = i64c(&mut b, if how == Kern::OtherOp { 1 } else { 0 });
    let input = match how {
        Kern::InPlace => s,
        Kern::HeldElsewhere => {
            b.build_void(Op::Store {
                ptr: holder,
                value: s,
            });
            holder
        }
        _ => src,
    };
    b.build_void(Op::Call {
        func: k,
        args: vec![input, s, n, op],
    });
    read_at(&mut b, s, 1, sink);
    let n64 = i64c(&mut b, 64);
    let op0 = i64c(&mut b, 0);
    b.build_void(Op::Call {
        func: k,
        args: vec![src, t, n64, op0],
    });
    read_at(&mut b, t, 2, sink);
    read_at(&mut b, src, 0, sink);
    b.ret(None);
    let mut funcs = vec![b.finish()];
    if how == Kern::UserFunction {
        let mut u = Builder::new(k, MirType::Void);
        for ty in [MirType::Ptr, MirType::Ptr, MirType::I64, MirType::I64] {
            u.add_param(ty);
        }
        u.ret(None);
        funcs.push(u.finish());
    }
    let promise = KernelOverwrite {
        arity: 4,
        out: 1,
        elem_bytes: 4,
        extent: &[2],
        positive: &[],
        fixed: &[(3, &[0])],
    };
    let overwrites = if how == Kern::Unpromised {
        Vec::new()
    } else {
        vec![(k, promise)]
    };
    let p = program_of(funcs);
    let lt = lifetimes_with(&p, main, overwrites);
    let (si, ti) = (lt.index_of(s).unwrap(), lt.index_of(t).unwrap());
    assert!(!lt.escapes(ti), "{how:?}");
    lt.escapes(si) || lt.interfere(si, ti)
}

/// A runtime kernel promised to overwrite its output ends the output's old lifetime at the call:
/// two arrays zeroed at the top and filled by kernels later share. Without the promise, outside its
/// conditions, with the output also an input, or with the output reachable another way, the zeros
/// stay live from the top and the arrays interfere.
#[test]
fn a_promised_kernel_output_is_a_full_overwrite() {
    assert!(!kernel_scratch(Kern::Promised));
    for how in [
        Kern::Short,
        Kern::OtherOp,
        Kern::InPlace,
        Kern::HeldElsewhere,
        Kern::Unpromised,
        Kern::UserFunction,
        Kern::UnknownExtent,
    ] {
        assert!(kernel_scratch(how), "{how:?} must not end the lifetime");
    }
}

/// How a vector-kernel call writes its output stream.
#[derive(Clone, Copy, PartialEq, Debug)]
enum VecCall {
    /// `out[i] = src[i] * 2` over 64 floats: stream 1 is only stored.
    Overwrites,
    /// `out[i] = out[i] + src[i]`: stream 1 is loaded too.
    Accumulates,
    /// The vector part covers 32 of the 64 floats.
    Short,
    /// Both streams are `s`, and stream 0 is loaded.
    Aliased,
    /// Both streams name the output and both are only stored: `out[i] = 2`, twice.
    TwoOutputs,
    /// `out[i] = src[i] * 2` stored as f16 (`StoreF16`) over 64 elements: the vector part writes
    /// 128 of the slot's 256 bytes, though 64 lanes of the `f32` lane would be 256.
    HalfStores,
}

/// `s` and `t` are written at the top, then each is filled through a vector kernel and read. Do they
/// interfere?
fn vec_scratch(how: VecCall) -> bool {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let src = b.alloca(arr(64));
    let s = b.alloca(arr(64));
    let t = b.alloca(arr(64));
    // One stream table per call site, as the vectorizer emits them.
    let tables = [
        b.alloca(MirType::Array(Box::new(MirType::Ptr), 2)),
        b.alloca(MirType::Array(Box::new(MirType::Ptr), 2)),
    ];
    let scalars = b.alloca(arr(1));
    write_at(&mut b, src, 0);
    write_at(&mut b, s, 0);
    write_at(&mut b, t, 0);
    let store1 = if how == VecCall::TwoOutputs {
        vec![
            VecOp::Splat { scalar: 0 },
            VecOp::Store { stream: 0, val: 0 },
            VecOp::Store { stream: 1, val: 0 },
        ]
    } else if how == VecCall::Accumulates {
        vec![
            VecOp::Load { stream: 0 },
            VecOp::Load { stream: 1 },
            VecOp::Bin {
                op: VecBin::Add,
                a: 0,
                b: 1,
            },
            VecOp::Store { stream: 1, val: 2 },
        ]
    } else {
        let store = if how == VecCall::HalfStores {
            VecOp::StoreF16 { stream: 1, val: 2 }
        } else {
            VecOp::Store { stream: 1, val: 2 }
        };
        vec![
            VecOp::Load { stream: 0 },
            VecOp::Splat { scalar: 0 },
            VecOp::Bin {
                op: VecBin::Mul,
                a: 0,
                b: 1,
            },
            store,
        ]
    };
    let kernel = b.add_vec_kernel(VecKernel {
        name: i.intern("vk"),
        lane: VecLane::F32,
        streams: 2,
        scalars: 1,
        unroll: 1,
        ops: store1,
        reduce: None,
    });
    let two = f32c(&mut b, 2.0);
    b.build_void(Op::Store {
        ptr: scalars,
        value: two,
    });
    for (table, out, n) in [
        (tables[0], s, if how == VecCall::Short { 32 } else { 64 }),
        (tables[1], t, 64),
    ] {
        let first = match how {
            VecCall::Aliased if out == s => s,
            VecCall::TwoOutputs => out,
            _ => src,
        };
        b.build_void(Op::Store {
            ptr: table,
            value: first,
        });
        let one = i64c(&mut b, 1);
        let entry = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: table,
                index: one,
                elem: MirType::Ptr,
            },
        );
        b.build_void(Op::Store {
            ptr: entry,
            value: out,
        });
        let n = i64c(&mut b, n);
        b.build_void(Op::VecKernelCall {
            kernel,
            ptrs: table,
            scalars,
            n,
        });
        read_at(&mut b, out, 1, sink);
    }
    read_at(&mut b, src, 0, sink);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes_with(&p, main, Vec::new());
    let (si, ti) = (lt.index_of(s).unwrap(), lt.index_of(t).unwrap());
    assert!(!lt.escapes(si) && !lt.escapes(ti), "{how:?}");
    lt.interfere(si, ti)
}

/// A vector kernel that stores a stream it never loads, over a vector part as long as the slot the
/// stream's table entry names, overwrites that slot, even when a second write-only stream names it
/// too; one that accumulates into the stream, covers part of it, or also reaches the slot through
/// a stream it loads does not. Nor does an f16 output stream measured at the `f32` lane's width:
/// its elements are two bytes, so 64 of them leave half of a 256-byte slot unwritten.
#[test]
fn a_vector_kernel_output_stream_is_a_full_overwrite() {
    for how in [VecCall::Overwrites, VecCall::TwoOutputs] {
        assert!(!vec_scratch(how), "{how:?} must end the lifetime");
    }
    for how in [
        VecCall::Accumulates,
        VecCall::Short,
        VecCall::Aliased,
        VecCall::HalfStores,
    ] {
        assert!(vec_scratch(how), "{how:?} must not end the lifetime");
    }
}

/// Parse, check, lower and optimize `src` at `opt`; the lifetimes of `main`'s `[f32; 512]` arrays.
fn arrays_of_main(src: &str, opt: u8) -> (SlotLifetimes, Vec<usize>) {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "{pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(ld.iter().all(|d| !d.is_error()), "{ld:?}");
    crate::optimize(&mut program, opt);
    let main = interner.intern("main");
    let calls = CaptureSummary::analyze(&program);
    let f = program.function(main).unwrap();
    let lt = SlotLifetimes::analyze(f, &calls);
    let arrays: Vec<usize> = lt
        .allocas()
        .enumerate()
        .filter(|&(_, v)| {
            f.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                i.result == Some(v) && matches!(&i.op, Op::Alloca { ty, .. } if *ty == arr(512))
            })
        })
        .map(|(k, _)| k)
        .collect();
    (lt, arrays)
}

/// Two arrays declared at the top and rewritten by a nest on every iteration of an outer loop,
/// read in between: `inner` is the extent of the nests' inner loops.
fn nest_program(inner: u32) -> String {
    format!(
        "
fn main() -> i32 {{
    let mut s: [f32; 512] = [0.0; 512];
    let mut t: [f32; 512] = [0.0; 512];
    let mut total: f32 = 0.0;
    for h in 0..3 {{
        for i in 0..8 {{ for j in 0..{inner} {{ s[i * 64 + j] = (i + j + h) as f32; }} }}
        for k in 0..512 {{ total = total + s[k] * 0.5; }}
        for i in 0..8 {{ for j in 0..{inner} {{ t[j * 8 + i] = (i * j + h) as f32; }} }}
        for k in 0..512 {{ total = total + t[k] * 0.25; }}
    }}
    print(total);
    return 0;
}}
"
    )
}

/// From source: at `-O0` a `for` loop keeps its variable in a counter slot, and at `-O2` in a block
/// parameter; either way a nest that rewrites the whole array on every iteration separates the two
/// arrays. A nest whose rows stop one short does not.
#[test]
fn source_nests_separate_per_iteration_arrays() {
    for opt in [0u8, 2] {
        let (lt, arrays) = arrays_of_main(&nest_program(64), opt);
        assert_eq!(arrays.len(), 2, "-O{opt}: s and t");
        assert!(
            !lt.interfere(arrays[0], arrays[1]),
            "-O{opt}: each nest rewrites its array before the read"
        );
        let (lt, arrays) = arrays_of_main(&nest_program(63), opt);
        assert_eq!(arrays.len(), 2, "-O{opt}: s and t");
        assert!(
            lt.interfere(arrays[0], arrays[1]),
            "-O{opt}: rows that stop at 63 keep the old contents live"
        );
    }
}

/// `s` and `t` are zeroed in one fused loop, then `s` is (when `rewrite`) fully refilled and read,
/// then `t` is fully refilled and read. Do they interfere?
fn fused_zeroing(rewrite: bool) -> bool {
    let mut i = Interner::new();
    let main = i.intern("main");
    let sink = i.intern("print");
    let mut b = Builder::new(main, MirType::Void);
    let s = b.alloca(arr(64));
    let t = b.alloca(arr(64));
    let zero = i64c(&mut b, 0);
    let l = open(&mut b, zero, CmpOp::Slt, 64);
    let z = f32c(&mut b, 0.0);
    for slot in [s, t] {
        let at = elem(&mut b, slot, l.iv);
        b.build_void(Op::Store { ptr: at, value: z });
    }
    close(&mut b, &l, 1);
    if rewrite {
        fill_1d(&mut b, s, 64);
    }
    read_at(&mut b, s, 3, sink);
    fill_1d(&mut b, t, 64);
    read_at(&mut b, t, 5, sink);
    b.ret(None);
    let p = program_of(vec![b.finish()]);
    let lt = lifetimes_with(&p, main, Vec::new());
    lt.interfere(lt.index_of(s).unwrap(), lt.index_of(t).unwrap())
}

/// A store is a use but not a read: a slot whose stored values are all overwritten before any read
/// occupies its storage only at the stores themselves, so two arrays zeroed in one fused loop and
/// each fully rewritten before it is read do not interfere. When `s` is read straight after the
/// fused loop, its zeros are live from their stores to the read — across `t`'s stores in the same
/// loop — and the two interfere.
#[test]
fn stores_with_no_later_read_occupy_only_their_own_points() {
    assert!(!fused_zeroing(true));
    assert!(fused_zeroing(false));
}

/// `s` is filled by four calls `k(x, &s[16*i], row, 0)`, one per row (in place: `x` is the row
/// itself), under a promise that `k` overwrites `out[0..n)` when `op` is 0. Do `s` and a fully
/// refilled `t` interfere (see [`per_iteration`])?
fn kernel_rows(row: i128, in_place: bool) -> bool {
    let promise = KernelOverwrite {
        arity: 4,
        out: 1,
        elem_bytes: 4,
        extent: &[2],
        positive: &[],
        fixed: &[(3, &[0])],
    };
    let mut i = Interner::new();
    let k = i.intern("k");
    let fill_rows = move |b: &mut Builder, s: ValueId| {
        let src = b.alloca(arr(16));
        write_at(b, src, 0);
        let zero = i64c(b, 0);
        let l = open(b, zero, CmpOp::Slt, 4);
        let w = i64c(b, 16);
        let base = bin(b, BinOp::Mul, l.iv, w);
        let out = elem(b, s, base);
        let n = i64c(b, row);
        let op = i64c(b, 0);
        let x = if in_place { out } else { src };
        b.build_void(Op::Call {
            func: k,
            args: vec![x, out, n, op],
        });
        close(b, &l, 1);
    };
    let full = |b: &mut Builder, t: ValueId| fill_1d(b, t, 64);
    per_iteration_in(&mut i, 64, &fill_rows, &full, vec![(k, promise)])
}

/// A runtime kernel promised to overwrite its output, called once per row of a slot inside a
/// counted loop, overwrites the whole slot when the rows tile it: the calls are pieces of the loop's
/// coverage like stores. Rows one element short leave the slot live, and so does a call that reads
/// the row it writes (in place).
#[test]
fn promised_kernel_calls_over_the_rows_of_a_slot_are_a_full_overwrite() {
    assert!(!kernel_rows(16, false));
    assert!(kernel_rows(15, false));
    assert!(kernel_rows(16, true));
}

/// `examples/llama_block.wk` declares its ten scratch arrays at the top of `llama_block`, zeroed,
/// and fills each later. At `-O2` the zeroing is fused into a few loops — a vector body tested with
/// `!=`, an unrolled body advancing its variable by chained adds, a remainder — and a vector kernel;
/// at `-O0` it is one `[v; N]` fill loop per array. Every array's zeroing (or its later fill) is a
/// recognized full overwrite at both levels.
#[test]
fn llama_block_scratch_zeroing_is_a_full_overwrite() {
    let src = include_str!("../../../../../examples/llama_block.wk");
    for opt in [0u8, 2] {
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "{pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
        let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(ld.iter().all(|d| !d.is_error()), "{ld:?}");
        crate::optimize(&mut program, opt);
        let f = program.function(interner.intern("llama_block")).unwrap();
        let calls = CaptureSummary::analyze(&program);
        let flow = super::super::Flow::run(f, &calls);
        let kills = super::find(f, &flow, &calls);
        let scratch: Vec<u32> = flow
            .allocas
            .iter()
            .enumerate()
            .filter(|&(_, &v)| {
                f.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                    i.result == Some(v)
                        && matches!(&i.op, Op::Alloca { ty: MirType::Array(e, n), .. }
                            if **e == MirType::F32 && *n >= 64)
                })
            })
            .map(|(k, _)| k as u32)
            .collect();
        assert_eq!(scratch.len(), 10, "-O{opt}");
        for x in scratch {
            assert!(
                kills.edges.iter().any(|e| e.2 == x) || kills.points.iter().any(|p| p.2 == x),
                "-O{opt}: scratch slot {x} ({:?}) has no full overwrite",
                flow.allocas[x as usize]
            );
        }
    }
}
