//! Dropping the source-site operands of `assert` calls, for a backend that has no use for them.
//!
//! With a source locator `wukong_mir_build` lowers a failed-assertion check to
//! `assert(cond, site)` — or `assert(cond, site, message)` for `assert(cond, "message")` — whose
//! trailing operands are `GlobalAddr`s of `.rodata` strings every CPU backend reports
//! (`wukong_runtime::assert_failure_message`). The general MIR→PTX backend (`--backend=gpu-native`)
//! keeps a single assert-failed flag in its device context, reports a bare `assertion failed`, and
//! declines any `GlobalAddr` it is asked to lower, so the driver hands it a copy of the program
//! with those operands stripped: [`strip_assert_sites`] turns every `assert(cond, ..)` back into
//! the one-operand `assert(cond)` that backend has always lowered, and deletes each `GlobalAddr`
//! the stripping left without a use.

use wukong_mir::{Op, Program};
use wukong_span::Symbol;

use crate::{each_op_use, each_term_use};

/// Rewrite every call to the `assert` intrinsic (`assert`, which no function of `program` may
/// define — a user function of that name shadows the intrinsic and is left alone) to its
/// one-operand form, and remove the `GlobalAddr` instructions that only fed the dropped operands.
/// Returns the number of calls rewritten.
pub fn strip_assert_sites(program: &mut Program, assert: Symbol) -> usize {
    if program.funcs.iter().any(|f| f.name == assert) {
        return 0;
    }
    let mut rewritten = 0;
    for f in &mut program.funcs {
        let mut dropped = Vec::new();
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                if let Op::Call { func, args } = &mut inst.op {
                    if *func == assert && args.len() > 1 {
                        dropped.extend(args.drain(1..));
                        rewritten += 1;
                    }
                }
            }
        }
        if dropped.is_empty() {
            continue;
        }
        // A dropped operand may still feed something else (a message literal shared with a
        // `print` of the same string, say): only a `GlobalAddr` nothing reads any more goes.
        let mut used = vec![false; f.value_types.len()];
        for b in &f.blocks {
            for inst in &b.insts {
                each_op_use(&inst.op, &mut |v| used[v.0 as usize] = true);
            }
            each_term_use(&b.term, &mut |v| used[v.0 as usize] = true);
        }
        for b in &mut f.blocks {
            b.insts.retain(|inst| {
                !matches!(inst.op, Op::GlobalAddr(_))
                    || inst
                        .result
                        .is_none_or(|r| used[r.0 as usize] || !dropped.contains(&r))
            });
        }
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Builder;
    use wukong_mir::{MirType, StaticData};
    use wukong_span::Interner;

    /// `assert(c, site, msg)` and `assert(c, site)` become `assert(c)`; the site's `GlobalAddr`
    /// goes, while a message `GlobalAddr` that a `print_str` also reads stays, and the result
    /// verifies.
    #[test]
    fn strips_site_operands_and_their_dead_addresses() {
        let mut it = Interner::new();
        let assert = it.intern("assert");
        let site = it.intern("wukong$loc$0");
        let msg = it.intern("wukong$rodata$0");
        let mut b = Builder::new(it.intern("main"), MirType::I32);
        let c = b.build(MirType::I1, Op::ConstInt(0, MirType::I1));
        let s = b.build(MirType::Ptr, Op::GlobalAddr(site));
        let m = b.build(MirType::Ptr, Op::GlobalAddr(msg));
        b.build_void(Op::Call {
            func: it.intern("print_str"),
            args: vec![m],
        });
        b.build_void(Op::Call {
            func: assert,
            args: vec![c, s, m],
        });
        let s2 = b.build(MirType::Ptr, Op::GlobalAddr(site));
        b.build_void(Op::Call {
            func: assert,
            args: vec![c, s2],
        });
        let z = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(z));
        let mut program = Program::new();
        program.funcs.push(b.finish());
        for (name, text) in [(site, "a.wk:1:1\0"), (msg, "boom\0")] {
            program.statics.push(StaticData {
                name,
                bytes: text.as_bytes().to_vec(),
            });
        }

        assert_eq!(strip_assert_sites(&mut program, assert), 2);
        let insts: Vec<&Op> = program.funcs[0].blocks[0]
            .insts
            .iter()
            .map(|i| &i.op)
            .collect();
        let asserts: Vec<usize> = insts
            .iter()
            .filter_map(|op| match op {
                Op::Call { func, args } if *func == assert => Some(args.len()),
                _ => None,
            })
            .collect();
        assert_eq!(asserts, [1, 1]);
        let globals: Vec<Symbol> = insts
            .iter()
            .filter_map(|op| match op {
                Op::GlobalAddr(g) => Some(*g),
                _ => None,
            })
            .collect();
        assert_eq!(globals, [msg], "only the message the print reads survives");
        assert!(wukong_mir::verify::verify_program(&program).is_empty());
    }

    /// A program that defines its own `assert` is untouched: its arguments are its own values.
    #[test]
    fn a_user_function_named_assert_is_left_alone() {
        let mut it = Interner::new();
        let assert = it.intern("assert");
        let mut user = Builder::new(assert, MirType::Void);
        let _ = user.add_param(MirType::I64);
        let _ = user.add_param(MirType::I64);
        user.ret(None);
        let mut b = Builder::new(it.intern("main"), MirType::Void);
        let x = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        b.build_void(Op::Call {
            func: assert,
            args: vec![x, x],
        });
        b.ret(None);
        let mut program = Program::new();
        program.funcs.push(user.finish());
        program.funcs.push(b.finish());
        assert_eq!(strip_assert_sites(&mut program, assert), 0);
        assert!(matches!(
            &program.funcs[1].blocks[0].insts[1].op,
            Op::Call { args, .. } if args.len() == 2
        ));
    }
}
