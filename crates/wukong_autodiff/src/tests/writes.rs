//! `written_params`: which buffer parameters a function may write. `--train` runs the gradient on
//! the live buffers and restores only the ones flagged here, so a missed write would leak a step's
//! forward writes into the next step's data or weights — every write channel is pinned below.

use super::*;
use crate::written_params;
use wukong_mir::{Builder, MirType, Op};
use wukong_span::Interner;

/// `p = x·wᵀ` (a recognized kernel: `x` and `w` read, the local `p` written), `loss = Σ p`,
/// `out[0] = loss`: only the loss output is written.
#[test]
fn kernel_inputs_are_read_only_and_the_loss_sink_is_written() {
    let mut it = Interner::default();
    let fwd = build_linear(&mut it, 3, 4, 2, None, false);
    assert_eq!(written_params(&fwd.func, &it), vec![false, false, true]);
}

/// A kernel whose OUTPUT operand is a parameter writes it — including an in-place activation, where
/// the parameter is both input and output.
#[test]
fn a_kernel_output_parameter_is_written() {
    let mut it = Interner::default();
    let vmath = sym(&mut it, "wukong_vmath_f32");
    let sreduce = sym(&mut it, "wukong_sreduce_f32");
    let mut b = Builder::new(sym(&mut it, "inplace"), MirType::F32);
    let x = b.add_param(PTR);
    let y = b.add_param(PTR);
    let n = ci(&mut b, 4);
    let op = ci(&mut b, VM_RELU);
    b.build_void(Op::Call {
        func: vmath,
        args: vec![x, x, n, op],
    });
    let sum = ci(&mut b, RED_SUM);
    let l = b.build(
        MirType::F32,
        Op::Call {
            func: sreduce,
            args: vec![y, y, n, sum],
        },
    );
    b.ret(Some(l));
    assert_eq!(written_params(&b.finish(), &it), vec![true, false]);
}

/// A store through a `gep` chain off a parameter writes it; a load through one does not.
#[test]
fn a_store_through_a_gep_chain_is_a_write() {
    let mut it = Interner::default();
    let mut b = Builder::new(sym(&mut it, "st"), MirType::F32);
    let src = b.add_param(PTR);
    let dst = b.add_param(PTR);
    let one = ci(&mut b, 1);
    let ps = b.build(
        PTR,
        Op::Gep {
            ptr: src,
            index: one,
            elem: MirType::F32,
        },
    );
    let v = b.build(MirType::F32, Op::Load(ps, MirType::F32));
    let pd = b.build(
        PTR,
        Op::Gep {
            ptr: dst,
            index: one,
            elem: MirType::F32,
        },
    );
    let pd2 = b.build(
        PTR,
        Op::Gep {
            ptr: pd,
            index: one,
            elem: MirType::F32,
        },
    );
    b.build_void(Op::Store { ptr: pd2, value: v });
    b.ret(Some(v));
    assert_eq!(written_params(&b.finish(), &it), vec![false, true]);
}

/// Every channel the tape cannot see through is conservatively a write: a call it does not
/// recognize, a pointer stored as a VALUE (it may be written through later), and a recognized
/// kernel name declared with a foreign signature.
#[test]
fn unknown_calls_and_escapes_are_conservatively_writes() {
    let mut it = Interner::default();
    let opaque = sym(&mut it, "wukong_rand_fill_f32");
    let sgemm = sym(&mut it, "wukong_sgemm_nt");
    let mut b = Builder::new(sym(&mut it, "esc"), MirType::F32);
    let a = b.add_param(PTR);
    let c = b.add_param(PTR);
    let d = b.add_param(PTR);
    let table = b.add_param(PTR);
    let e = b.add_param(PTR);
    let n = ci(&mut b, 4);
    b.build_void(Op::Call {
        func: opaque,
        args: vec![a, n],
    });
    b.build_void(Op::Store {
        ptr: table,
        value: c,
    });
    // Two operands instead of `wukong_sgemm_nt`'s seven: not the recognized ABI.
    b.build_void(Op::Call {
        func: sgemm,
        args: vec![d, n],
    });
    let v = b.build(MirType::F32, Op::Load(e, MirType::F32));
    b.ret(Some(v));
    assert_eq!(
        written_params(&b.finish(), &it),
        vec![true, true, true, true, false]
    );
}

/// An effect-only call (`print`, a file write) reads its buffers and writes none of them.
#[test]
fn a_file_write_reads_its_buffer() {
    let mut it = Interner::default();
    let write = sym(&mut it, "wukong_rt_write_f32");
    let mut b = Builder::new(sym(&mut it, "save"), MirType::F32);
    let path = b.add_param(PTR);
    let buf = b.add_param(PTR);
    let n = ci(&mut b, 4);
    b.build(
        MirType::I64,
        Op::Call {
            func: write,
            args: vec![path, buf, n],
        },
    );
    let v = b.build(MirType::F32, Op::Load(buf, MirType::F32));
    b.ret(Some(v));
    assert_eq!(written_params(&b.finish(), &it), vec![false, false]);
}
