//! A deterministic textual rendering of MIR, used by `--emit=mir` / `--emit=mir-high` and snapshot
//! tests. It is also the equality oracle for `wukong_bench`'s check that `optimize_timed` matches the
//! production `optimize` byte-for-byte, and the grep surface tests use to assert the vectorizer fired
//! (`<4 x f32>` for the 128-bit path, `veckernel` for the 256-bit one), so the output must stay a pure
//! function of the `Program`.

use crate::{Function, Inst, Op, Program, Terminator, ValueId};
use wukong_span::Interner;

/// Render a whole program: every function in `funcs` order, one blank line apart. `Program::statics`
/// is *not* rendered, so a `global_addr <sym>` line is the only trace a string literal leaves in the
/// dump. Neither are `Function::vec_kernels` — a `veckernel #k(…)` line shows the call, not the recipe.
pub fn print_program(p: &Program, interner: &Interner) -> String {
    let mut out = String::new();
    for f in &p.funcs {
        out.push_str(&print_function(f, interner));
        out.push('\n');
    }
    out
}

/// Render a single function.
pub fn print_function(f: &Function, interner: &Interner) -> String {
    let mut out = String::new();
    let params: Vec<String> = f
        .params
        .iter()
        .map(|v| format!("{} {}", val(*v), f.value_type(*v).display()))
        .collect();
    out.push_str(&format!(
        "fn {}({}) -> {} {{\n",
        interner.resolve(f.name),
        params.join(", "),
        f.ret.display()
    ));
    for b in &f.blocks {
        let bparams: Vec<String> = b
            .params
            .iter()
            .map(|v| format!("{} {}", val(*v), f.value_type(*v).display()))
            .collect();
        if bparams.is_empty() {
            out.push_str(&format!("  bb{}:\n", b.id.0));
        } else {
            out.push_str(&format!("  bb{}({}):\n", b.id.0, bparams.join(", ")));
        }
        for inst in &b.insts {
            out.push_str(&format!("    {}\n", fmt_inst(inst, interner)));
        }
        out.push_str(&format!("    {}\n", fmt_term(&b.term)));
    }
    out.push_str("}\n");
    out
}

fn val(v: ValueId) -> String {
    format!("v{}", v.0)
}

fn fmt_inst(inst: &Inst, interner: &Interner) -> String {
    let rhs = fmt_op(&inst.op, interner);
    match inst.result {
        Some(r) => format!("{} = {}", val(r), rhs),
        None => rhs,
    }
}

fn fmt_op(op: &Op, interner: &Interner) -> String {
    match op {
        Op::ConstInt(v, ty) => format!("const.{} {}", ty.display(), v),
        Op::ConstFloat(v, ty) => format!("const.{} {}", ty.display(), v),
        Op::Bin(b, l, r) => format!("{} {}, {}", b.name(), val(*l), val(*r)),
        Op::Cmp(c, l, r) => format!("cmp.{} {}, {}", c.name(), val(*l), val(*r)),
        Op::Neg(v) => format!("neg {}", val(*v)),
        Op::Not(v) => format!("not {}", val(*v)),
        Op::Cast(k, v, to) => format!("{} {} to {}", k.name(), val(*v), to.display()),
        Op::Select(c, a, b) => format!("select {}, {}, {}", val(*c), val(*a), val(*b)),
        Op::Alloca(ty) => format!("alloca {}", ty.display()),
        Op::Load(p, ty) => format!("load {} {}", ty.display(), val(*p)),
        Op::Store { ptr, value } => format!("store {}, {}", val(*value), val(*ptr)),
        Op::Gep { ptr, index, elem } => {
            format!("gep {}, {} : {}", val(*ptr), val(*index), elem.display())
        }
        Op::Call { func, args } => {
            let a: Vec<String> = args.iter().map(|v| val(*v)).collect();
            format!("call {}({})", interner.resolve(*func), a.join(", "))
        }
        Op::FuncAddr(func) => format!("func_addr {}", interner.resolve(*func)),
        Op::GlobalAddr(data) => format!("global_addr {}", interner.resolve(*data)),
        Op::Splat(v) => format!("splat {}", val(*v)),
        Op::Fma(a, b, c) => format!("fma {}, {}, {}", val(*a), val(*b), val(*c)),
        Op::Sqrt(a) => format!("sqrt {}", val(*a)),
        Op::VecKernelCall {
            kernel,
            ptrs,
            scalars,
            n,
        } => format!(
            "veckernel #{kernel}(ptrs={}, scalars={}, n={})",
            val(*ptrs),
            val(*scalars),
            val(*n)
        ),
        Op::Round(mode, a) => {
            let m = match mode {
                crate::inst::RoundMode::Nearest => "nearest",
                crate::inst::RoundMode::Floor => "floor",
                crate::inst::RoundMode::Ceil => "ceil",
                crate::inst::RoundMode::Trunc => "trunc",
            };
            format!("round.{m} {}", val(*a))
        }
    }
}

fn fmt_term(t: &Terminator) -> String {
    match t {
        Terminator::Ret(None) => "ret".to_string(),
        Terminator::Ret(Some(v)) => format!("ret {}", val(*v)),
        Terminator::Br { target, args } => format!("br bb{}{}", target.0, fmt_args(args)),
        Terminator::CondBr {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => format!(
            "cond_br {}, bb{}{}, bb{}{}",
            val(*cond),
            then_blk.0,
            fmt_args(then_args),
            else_blk.0,
            fmt_args(else_args)
        ),
        Terminator::Unreachable => "unreachable".to_string(),
    }
}

fn fmt_args(args: &[ValueId]) -> String {
    if args.is_empty() {
        String::new()
    } else {
        let a: Vec<String> = args.iter().map(|v| val(*v)).collect();
        format!("({})", a.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinOp, Builder, MirType, Op};
    use wukong_span::Interner;

    #[test]
    fn builds_and_prints_add() {
        let mut i = Interner::new();
        let name = i.intern("add");
        let mut b = Builder::new(name, MirType::I32);
        let a = b.add_param(MirType::I32);
        let c = b.add_param(MirType::I32);
        let sum = b.build(MirType::I32, Op::Bin(BinOp::Add, a, c));
        b.ret(Some(sum));
        let f = b.finish();
        let out = print_function(&f, &i);
        let expected = "\
fn add(v0 i32, v1 i32) -> i32 {
  bb0(v0 i32, v1 i32):
    v2 = add v0, v1
    ret v2
}
";
        assert_eq!(out, expected);
    }
}
