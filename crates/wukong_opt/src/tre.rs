//! Tail-recursion elimination: a function's calls to itself in tail position become a loop.
//!
//! A *tail site* is a block whose last instruction calls the function itself and whose terminator
//! returns that call's result — `ret r` directly, or `br M(.., r, ..)` into an empty block `M` that
//! returns the parameter `r` lands in (the merge block of a value `if`). A void call followed by a
//! void return is one too. The rewrite is LLVM's TailCallElim:
//!
//! ```text
//!   entry(p..):  <body>                 entry(p..):  br H(p..)
//!                                  =>   H(q..):      <body, with every p read as q>
//!   S:  r = call f(a..); ret r          S:           br H(a..)
//! ```
//!
//! Each tail call starts the next iteration with its arguments as the new parameters; nothing after
//! the call ran in the original either, so every value is computed exactly as before, in the same
//! order. The function's own frame is never re-entered, so it no longer pays a call per level and
//! cannot overflow the native stack where the loop would not.
//!
//! # Accumulator recursion
//!
//! `ret op(call f(a..), x)` (either operand order) is also eliminated when `op` is an integer
//! `add`, `mul`, `and`, `or` or `xor`: the header gains an accumulator parameter starting at the
//! operation's identity, a tail site passes `op(acc, x)`, and every remaining `ret v` becomes
//! `ret op(acc, v)`. That reorders the operation, which is exact only because these are
//! associative and commutative in wrapping integer arithmetic — the property LLVM relies on for the
//! same rewrite. A float operation is **never** accumulated (reassociating it changes the rounding,
//! and the interpreter is the specification). This is what turns `n * fact(n - 1)` into a loop and
//! one of `fib(n - 1) + fib(n - 2)`'s two calls into an iteration. All tail sites of one function
//! must agree on `op`; with mixed operations only the plain sites are rewritten.
//!
//! # Preconditions
//!
//! The function may hold **no `alloca`**: a frame slot would be shared by every iteration where the
//! recursion gave each level a fresh one (and a pointer to it passed down would alias the next
//! level's). After `mem2reg` that excludes only functions with local arrays or address-taken
//! slots. The pass runs in the standard pipeline from `-O1` ([`Tre`]), and at `-O2` the inliner first
//! calls [`pre_inline`], which promotes a self-recursive function's slots on a copy and keeps the
//! rewrite when it removes **every** self-call — the call graph then no longer marks the function
//! recursive, so it becomes an ordinary inlining candidate.

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CastKind, Function, Inst, MirType, Op, Program, Terminator, ValueId,
};

use crate::fxhash::FxHashMap;
use crate::{each_op_use, map_op_uses, map_term_uses, CfgAnalyses, Mem2Reg, Pass};

/// Tail-recursion elimination as a pipeline pass (see the module docs).
pub struct Tre;

impl Pass for Tre {
    fn name(&self) -> &'static str {
        "tre"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        if eliminate(f) {
            cache.invalidate();
            true
        } else {
            false
        }
    }
}

/// Before inlining: rewrite each function whose self-calls are all in tail position into a loop, on
/// a copy whose slots `mem2reg` has promoted, and keep the copy only when no self-call is left.
/// Returns whether any function changed.
pub(crate) fn pre_inline(program: &mut Program) -> bool {
    let mut changed = false;
    for f in &mut program.funcs {
        if !calls_self(f) {
            continue;
        }
        let mut g = f.clone();
        Mem2Reg.run_function(&mut g, &mut CfgAnalyses::default());
        if eliminate(&mut g) && !calls_self(&g) {
            *f = g;
            changed = true;
        }
    }
    changed
}

/// How a tail site produces the function's result.
#[derive(Clone, Copy)]
enum SiteKind {
    /// `ret call(..)`, or a void call followed by a void return.
    Plain,
    /// `ret op(call(..), other)`.
    Acc { op: BinOp, other: ValueId },
}

struct Site {
    block: usize,
    /// Index of the self-call.
    call_at: usize,
    /// Index of the combining operation (the block's last instruction) of an accumulator site.
    op_at: Option<usize>,
    kind: SiteKind,
}

fn calls_self(f: &Function) -> bool {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(&i.op, Op::Call { func, .. } if *func == f.name))
}

/// The identity of an accumulable operation, or `None` if it is not one.
fn acc_identity(op: BinOp) -> Option<i128> {
    match op {
        BinOp::Add | BinOp::Or | BinOp::Xor => Some(0),
        BinOp::Mul => Some(1),
        BinOp::And => Some(-1),
        _ => None,
    }
}

/// What block `b` returns, when its terminator returns: `Some(Some(v))` for a value — directly, or
/// through a `br` into an empty block that returns one of its own parameters — and `Some(None)` for
/// a void return.
fn returned(f: &Function, b: usize) -> Option<Option<ValueId>> {
    match &f.blocks[b].term {
        Terminator::Ret(v) => Some(*v),
        Terminator::Br { target, args } => {
            let m = target.0 as usize;
            if m == b {
                return None;
            }
            let mb = &f.blocks[m];
            if !mb.insts.is_empty() {
                return None;
            }
            match &mb.term {
                Terminator::Ret(None) => Some(None),
                Terminator::Ret(Some(p)) => {
                    let j = mb.params.iter().position(|q| q == p)?;
                    Some(Some(*args.get(j)?))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn find_sites(f: &Function) -> Vec<Site> {
    let accumulable = f.ret.is_int() && f.ret != MirType::I1;
    let mut sites = Vec::new();
    for (b, blk) in f.blocks.iter().enumerate() {
        let Some(ret) = returned(f, b) else {
            continue;
        };
        let n = blk.insts.len();
        let Some(last) = blk.insts.last() else {
            continue;
        };
        if matches!(&last.op, Op::Call { func, .. } if *func == f.name) && last.result == ret {
            sites.push(Site {
                block: b,
                call_at: n - 1,
                op_at: None,
                kind: SiteKind::Plain,
            });
            continue;
        }
        if !accumulable || n < 2 {
            continue;
        }
        let (Some(t), Op::Bin(op, x, y)) = (last.result, &last.op) else {
            continue;
        };
        if ret != Some(t) || acc_identity(*op).is_none() {
            continue;
        }
        // The self-call whose result is one operand. Between it and the operation only pure,
        // call-independent instructions may sit (`m(n - 1) + 2` materializes its `2` there): they
        // stay where they are, and with the call gone nothing they compute changes.
        let Some(call_at) = (0..n - 1)
            .rev()
            .find(|&k| matches!(&blk.insts[k].op, Op::Call { func, .. } if *func == f.name))
        else {
            continue;
        };
        let Some(r) = blk.insts[call_at].result else {
            continue;
        };
        let independent = blk.insts[call_at + 1..n - 1].iter().all(|i| {
            let mut uses_r = false;
            each_op_use(&i.op, &mut |v| uses_r |= v == r);
            !uses_r && movable(&i.op)
        });
        if !independent {
            continue;
        }
        let other = if *x == r && *y != r {
            *y
        } else if *y == r && *x != r {
            *x
        } else {
            continue;
        };
        sites.push(Site {
            block: b,
            call_at,
            op_at: Some(n - 1),
            kind: SiteKind::Acc { op: *op, other },
        });
    }
    sites
}

/// May this instruction run without the self-call before it: no effect, no fault, no memory read
/// (the call could have written what a load would see)?
fn movable(op: &Op) -> bool {
    match op {
        Op::ConstInt(..) | Op::ConstFloat(..) | Op::Cmp(..) | Op::Not(_) | Op::Neg(_) => true,
        Op::Select(..) => true,
        Op::Bin(b, ..) => !matches!(b, BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem),
        Op::Cast(k, ..) => !matches!(k, CastKind::FpToSi | CastKind::FpToUi),
        _ => false,
    }
}

fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

/// Rewrite `f`'s tail self-calls into a loop (see the module docs). Returns whether it did.
pub(crate) fn eliminate(f: &mut Function) -> bool {
    if !calls_self(f)
        || f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i.op, Op::Alloca { .. }))
    {
        return false;
    }
    let mut sites = find_sites(f);
    let mut acc_op: Option<BinOp> = None;
    let mut mixed = false;
    for s in &sites {
        if let SiteKind::Acc { op, .. } = s.kind {
            match acc_op {
                None => acc_op = Some(op),
                Some(o) if o != op => mixed = true,
                Some(_) => {}
            }
        }
    }
    if mixed {
        sites.retain(|s| matches!(s.kind, SiteKind::Plain));
        acc_op = None;
    }
    if sites.is_empty() {
        return false;
    }

    let entry = f.entry.0 as usize;
    let params = f.params.clone();
    let ret_ty = f.ret.clone();
    // The loop header's parameters: one per function parameter, then the accumulator.
    let mut header_params: Vec<ValueId> = Vec::with_capacity(params.len() + 1);
    for p in &params {
        let ty = f.value_type(*p).clone();
        header_params.push(fresh(f, ty));
    }
    let acc = acc_op.map(|_| fresh(f, ret_ty.clone()));
    header_params.extend(acc);

    // Every read of a parameter becomes a read of the header's copy. The entry's own instructions
    // and terminator move into the header below, so they are renamed too; the entry then only
    // passes the real parameters in.
    let subst: FxHashMap<u32, ValueId> = params
        .iter()
        .map(|p| p.0)
        .zip(header_params.iter().copied())
        .collect();
    let rename = |v: ValueId| *subst.get(&v.0).unwrap_or(&v);
    for b in &mut f.blocks {
        for inst in &mut b.insts {
            map_op_uses(&mut inst.op, rename);
        }
        map_term_uses(&mut b.term, rename);
    }

    let header = BlockId(f.blocks.len() as u32);
    let body_insts = std::mem::take(&mut f.blocks[entry].insts);
    let body_term = std::mem::replace(&mut f.blocks[entry].term, Terminator::Unreachable);
    f.blocks.push(BasicBlock {
        id: header,
        params: header_params,
        insts: body_insts,
        term: body_term,
    });
    let mut entry_args = params;
    if let Some(op) = acc_op {
        let ident = fresh(f, ret_ty.clone());
        let value = acc_identity(op).expect("only accumulable operations are recorded");
        f.blocks[entry].insts.push(Inst {
            result: Some(ident),
            op: Op::ConstInt(value, ret_ty.clone()),
        });
        entry_args.push(ident);
    }
    f.blocks[entry].term = Terminator::Br {
        target: header,
        args: entry_args,
    };

    // Each tail site starts the next iteration.
    for s in &sites {
        let b = if s.block == entry {
            header.0 as usize
        } else {
            s.block
        };
        let Op::Call { args, .. } = &f.blocks[b].insts[s.call_at].op else {
            unreachable!("a tail site's call is recorded at its index");
        };
        let mut next = args.clone();
        let insts = &mut f.blocks[b].insts;
        if let Some(op_at) = s.op_at {
            insts.remove(op_at);
        }
        insts.remove(s.call_at);
        match (s.kind, acc) {
            (SiteKind::Plain, Some(a)) => next.push(a),
            (SiteKind::Plain, None) => {}
            (SiteKind::Acc { op, other }, Some(a)) => {
                let other = rename(other);
                let folded = fresh(f, ret_ty.clone());
                f.blocks[b].insts.push(Inst {
                    result: Some(folded),
                    op: Op::Bin(op, a, other),
                });
                next.push(folded);
            }
            (SiteKind::Acc { .. }, None) => {
                unreachable!("an accumulator site implies an accumulator")
            }
        }
        f.blocks[b].term = Terminator::Br {
            target: header,
            args: next,
        };
    }

    // Every other return folds in what the eliminated levels accumulated.
    if let (Some(op), Some(a)) = (acc_op, acc) {
        for b in 0..f.blocks.len() {
            if b == entry {
                continue;
            }
            let Terminator::Ret(Some(v)) = f.blocks[b].term else {
                continue;
            };
            let t = fresh(f, ret_ty.clone());
            f.blocks[b].insts.push(Inst {
                result: Some(t),
                op: Op::Bin(op, a, v),
            });
            f.blocks[b].term = Terminator::Ret(Some(t));
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    fn compile(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        (program, interner)
    }

    /// Optimize `program` at `level`, verify it, and run `main`, returning stdout.
    fn run_at(program: &Program, interner: &Interner, level: u8) -> (Program, Vec<u8>) {
        let mut optimized = program.clone();
        crate::optimize(&mut optimized, level);
        let errs = wukong_mir::verify::verify_program(&optimized);
        assert!(errs.is_empty(), "O{level}: {errs:?}");
        let main = interner.get("main").expect("main is interned");
        let (exit, stdout) = wukong_interp::run_with_output(&optimized, main, interner).unwrap();
        assert_eq!(exit, 0, "O{level}");
        (optimized, stdout)
    }

    /// How many calls `name` makes to itself, or `None` if `name` is gone (inlined).
    fn self_calls(p: &Program, interner: &Interner, name: &str) -> Option<usize> {
        let sym = interner.get(name)?;
        let f = p.function(sym)?;
        Some(
            f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| matches!(&i.op, Op::Call { func, .. } if *func == sym))
                .count(),
        )
    }

    #[test]
    fn a_self_tail_call_becomes_a_loop_and_then_inlines() {
        let src =
            "fn gcd(a: i64, b: i64) -> i64 { if b == 0 { return a; } return gcd(b, a % b); }\n\
                   fn main() -> i32 { print(gcd(1071, 462)); print(gcd(17, 5)); print(gcd(0, 9)); \
                   return 0; }";
        let (program, interner) = compile(src);
        // 1071 = 2*462 + 147, 462 = 3*147 + 21, 147 = 7*21; 17 = 3*5 + 2, 5 = 2*2 + 1; gcd(0, 9) = 9.
        let expected = b"21\n1\n9\n".to_vec();
        for level in [0u8, 1, 2] {
            let (opt, out) = run_at(&program, &interner, level);
            assert_eq!(out, expected, "O{level}");
            match level {
                0 => assert_eq!(self_calls(&opt, &interner, "gcd"), Some(1)),
                1 => assert_eq!(self_calls(&opt, &interner, "gcd"), Some(0), "a loop at O1"),
                // No longer recursive, so the inliner takes all three small call sites.
                _ => assert_eq!(self_calls(&opt, &interner, "gcd"), None, "inlined at O2"),
            }
        }
    }

    #[test]
    fn integer_accumulator_recursion_becomes_a_loop() {
        let src = "fn fact(n: i64) -> i64 { if n < 2 { return 1; } return n * fact(n - 1); }\n\
                   fn sum(n: i64) -> i64 { if n == 0 { return 0; } return sum(n - 1) + n; }\n\
                   fn xors(n: i64) -> i64 { if n == 0 { return 0; } return n ^ xors(n - 1); }\n\
                   fn main() -> i32 { print(fact(10)); print(fact(25)); print(sum(1000)); \
                   print(xors(1000)); return 0; }";
        let (program, interner) = compile(src);
        // 10! = 3628800; 25! mod 2^64 as i64 = 7034535277573963776 (the product wraps, and wrapping
        // multiplication is associative and commutative, so the reordered loop must agree);
        // 1 + .. + 1000 = 500500; 1 ^ .. ^ 1000 = 1000 (n ≡ 0 mod 4).
        let expected = b"3628800\n7034535277573963776\n500500\n1000\n".to_vec();
        for level in [0u8, 1, 2] {
            let (opt, out) = run_at(&program, &interner, level);
            assert_eq!(out, expected, "O{level}");
            if level == 1 {
                for name in ["fact", "sum", "xors"] {
                    assert_eq!(self_calls(&opt, &interner, name), Some(0), "{name} at O1");
                }
            }
        }
    }

    #[test]
    fn fib_keeps_exactly_one_call() {
        let src =
            "fn fib(n: i32) -> i32 { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }\n\
                   fn main() -> i32 { print(fib(20)); return 0; }";
        let (program, interner) = compile(src);
        for level in [0u8, 1, 2] {
            let (opt, out) = run_at(&program, &interner, level);
            assert_eq!(out, b"6765\n".to_vec(), "O{level}");
            if level >= 1 {
                // `fib(n - 2)` became the loop; `fib(n - 1)` is a real call still.
                assert_eq!(self_calls(&opt, &interner, "fib"), Some(1), "O{level}");
            }
        }
    }

    /// A float accumulation is never reassociated, so its call stays.
    #[test]
    fn float_accumulator_recursion_is_left_alone() {
        let src =
            "fn fsum(n: i64) -> f64 { if n == 0 { return 0.0; } return fsum(n - 1) + 0.1; }\n\
                   fn main() -> i32 { print((fsum(10) * 1000000.0) as i64); return 0; }";
        let (program, interner) = compile(src);
        let (_, at_o0) = run_at(&program, &interner, 0);
        for level in [1u8, 2] {
            let (opt, out) = run_at(&program, &interner, level);
            assert_eq!(out, at_o0, "O{level}");
            assert_eq!(self_calls(&opt, &interner, "fsum"), Some(1), "O{level}");
        }
    }

    /// A function with a frame slot left after `mem2reg` (a local array) keeps its call: the
    /// recursion gave every level its own slot.
    #[test]
    fn a_frame_slot_blocks_the_rewrite() {
        let src = "fn walk(n: i64, acc: i64) -> i64 { let mut t: [i64; 2] = [n, acc]; \
                   if t[0] == 0 { return t[1]; } return walk(t[0] - 1, t[1] + t[0]); }\n\
                   fn main() -> i32 { print(walk(10, 0)); return 0; }";
        let (program, interner) = compile(src);
        // 10 + 9 + .. + 1 = 55.
        let (_, at_o0) = run_at(&program, &interner, 0);
        assert_eq!(at_o0, b"55\n".to_vec());
        let (opt, out) = run_at(&program, &interner, 1);
        assert_eq!(out, at_o0);
        assert_eq!(self_calls(&opt, &interner, "walk"), Some(1));
    }

    #[test]
    fn mixed_accumulators_keep_only_plain_sites() {
        // Two accumulator sites disagree (`+` and `*`), so neither is rewritten; the plain tail
        // call is.
        let src = "fn m(n: i64) -> i64 { if n == 0 { return 1; } \
                   if n == 1 { return m(n - 1) + 2; } \
                   if n == 2 { return m(n - 1) * 3; } \
                   return m(n - 1); }\n\
                   fn main() -> i32 { print(m(5)); return 0; }";
        let (program, interner) = compile(src);
        // m(0) = 1, m(1) = 3, m(2) = 9, m(3..5) = 9.
        for level in [0u8, 1] {
            let (opt, out) = run_at(&program, &interner, level);
            assert_eq!(out, b"9\n".to_vec(), "O{level}");
            if level == 1 {
                assert_eq!(self_calls(&opt, &interner, "m"), Some(2));
            }
        }
    }
}
