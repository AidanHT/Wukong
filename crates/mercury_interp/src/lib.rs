//! `mercury_interp` — a from-scratch MIR interpreter.
//!
//! Zero external dependencies by design: this is the always-available execution path and the
//! oracle that differential tests compare LLVM output against. It walks the CFG block by block,
//! keeping a per-call register file and a shared flat memory for `alloca`/`load`/`store`/`gep`.

use mercury_backend::{Artifact, Backend};
use mercury_mir::{
    BinOp, CastKind, CmpOp, Function, MirType, Op, Program, Terminator, ValueId,
};
use mercury_span::{Interner, Symbol};

/// A runtime value. Integers are stored width-agnostically in an `i128` and masked per result
/// type; pointers are indices into the interpreter's flat memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Int(i128),
    Float(f64),
    Ptr(usize),
    Unit,
}

impl Value {
    fn as_int(self) -> i128 {
        match self {
            Value::Int(i) => i,
            Value::Ptr(p) => p as i128,
            Value::Float(f) => f as i128,
            Value::Unit => 0,
        }
    }

    fn as_float(self) -> f64 {
        match self {
            Value::Float(f) => f,
            Value::Int(i) => i as f64,
            _ => 0.0,
        }
    }

    fn truthy(self) -> bool {
        self.as_int() != 0
    }
}

/// The interpreter backend.
pub struct Interpreter;

impl Backend for Interpreter {
    fn name(&self) -> &'static str {
        "interpreter"
    }

    fn compile(
        &self,
        program: &Program,
        entry: Symbol,
        interner: &Interner,
    ) -> Result<Artifact, String> {
        let (exit_code, stdout) = run_with_output(program, entry, interner)?;
        Ok(Artifact::Executed { exit_code, stdout })
    }
}

/// Run `entry` (typically `main`) and return its integer result as a process exit code.
pub fn run(program: &Program, entry: Symbol, interner: &Interner) -> Result<i64, String> {
    Ok(run_with_output(program, entry, interner)?.0)
}

/// Run `entry` and return both its exit code and anything it printed.
pub fn run_with_output(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let func = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    let mut interp = Interp { program, interner, memory: Vec::new(), stdout: Vec::new() };
    let result = interp.run_function(func, Vec::new())?;
    Ok((result.as_int() as i64, interp.stdout))
}

struct Interp<'a> {
    program: &'a Program,
    interner: &'a Interner,
    memory: Vec<Value>,
    stdout: Vec<u8>,
}

impl<'a> Interp<'a> {
    fn run_function(&mut self, func: &Function, args: Vec<Value>) -> Result<Value, String> {
        let mut regs: Vec<Option<Value>> = vec![None; func.value_types.len()];
        for (p, a) in func.params.iter().zip(args) {
            regs[p.0 as usize] = Some(a);
        }

        let mut cur = func.entry;
        let mut steps = 0u64;
        loop {
            steps += 1;
            if steps > 100_000_000 {
                return Err("interpreter step limit exceeded (likely an infinite loop)".into());
            }
            let block = func.block(cur);
            for inst in &block.insts {
                let v = self.eval(&inst.op, &regs)?;
                if let Some(r) = inst.result {
                    regs[r.0 as usize] = Some(v);
                }
            }
            match &block.term {
                Terminator::Ret(None) => return Ok(Value::Unit),
                Terminator::Ret(Some(v)) => return Ok(reg(&regs, *v)),
                Terminator::Br { target, args } => {
                    let vals: Vec<Value> = args.iter().map(|a| reg(&regs, *a)).collect();
                    let tb = func.block(*target);
                    for (p, val) in tb.params.iter().zip(vals) {
                        regs[p.0 as usize] = Some(val);
                    }
                    cur = *target;
                }
                Terminator::CondBr { cond, then_blk, then_args, else_blk, else_args } => {
                    let (tgt, bargs) = if reg(&regs, *cond).truthy() {
                        (*then_blk, then_args)
                    } else {
                        (*else_blk, else_args)
                    };
                    let vals: Vec<Value> = bargs.iter().map(|a| reg(&regs, *a)).collect();
                    let tb = func.block(tgt);
                    for (p, val) in tb.params.iter().zip(vals) {
                        regs[p.0 as usize] = Some(val);
                    }
                    cur = tgt;
                }
                Terminator::Unreachable => {
                    return Err("execution reached `unreachable`".into())
                }
            }
        }
    }

    fn eval(&mut self, op: &Op, regs: &[Option<Value>]) -> Result<Value, String> {
        Ok(match op {
            Op::ConstInt(v, ty) => Value::Int(mask(*v, ty)),
            Op::ConstFloat(v, _) => Value::Float(*v),
            Op::Bin(b, l, r) => apply_bin(*b, reg(regs, *l), reg(regs, *r)),
            Op::Cmp(c, l, r) => {
                Value::Int(apply_cmp(*c, reg(regs, *l), reg(regs, *r)) as i128)
            }
            Op::Neg(v) => match reg(regs, *v) {
                Value::Float(f) => Value::Float(-f),
                other => Value::Int(-other.as_int()),
            },
            Op::Not(v) => match reg(regs, *v) {
                Value::Int(i) => Value::Int(!i),
                other => Value::Int(!other.as_int()),
            },
            Op::Cast(kind, v, to) => apply_cast(*kind, reg(regs, *v), to),
            Op::Select(c, a, b) => {
                if reg(regs, *c).truthy() {
                    reg(regs, *a)
                } else {
                    reg(regs, *b)
                }
            }
            Op::Alloca(ty) => {
                let idx = self.memory.len();
                self.memory.push(default_value(ty));
                Value::Ptr(idx)
            }
            Op::Load(p, _) => {
                let idx = ptr(reg(regs, *p))?;
                *self.memory.get(idx).ok_or("load out of bounds")?
            }
            Op::Store { ptr: p, value } => {
                let idx = ptr(reg(regs, *p))?;
                let val = reg(regs, *value);
                *self.memory.get_mut(idx).ok_or("store out of bounds")? = val;
                Value::Unit
            }
            Op::Gep { ptr: p, index, .. } => {
                let base = ptr(reg(regs, *p))?;
                let off = reg(regs, *index).as_int();
                Value::Ptr((base as i128 + off) as usize)
            }
            Op::Call { func, args } => {
                let argv: Vec<Value> = args.iter().map(|a| reg(regs, *a)).collect();
                let prog = self.program;
                if let Some(callee) = prog.function(*func) {
                    self.run_function(callee, argv)?
                } else {
                    let name = self.interner.resolve(*func).to_string();
                    self.intrinsic(&name, &argv)?
                }
            }
        })
    }

    fn intrinsic(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        match name {
            "print" | "println" => {
                let text = match args.first().copied().unwrap_or(Value::Unit) {
                    Value::Int(i) => format!("{i}\n"),
                    Value::Float(f) => format!("{f}\n"),
                    Value::Ptr(p) => format!("{p}\n"),
                    Value::Unit => "\n".to_string(),
                };
                self.stdout.extend_from_slice(text.as_bytes());
                Ok(Value::Unit)
            }
            other => Err(format!("call to unknown function or intrinsic `{other}`")),
        }
    }
}

fn reg(regs: &[Option<Value>], v: ValueId) -> Value {
    regs[v.0 as usize].unwrap_or(Value::Unit)
}

fn ptr(v: Value) -> Result<usize, String> {
    match v {
        Value::Ptr(p) => Ok(p),
        _ => Err("expected a pointer".into()),
    }
}

fn default_value(ty: &MirType) -> Value {
    if ty.is_float() {
        Value::Float(0.0)
    } else if matches!(ty, MirType::Ptr) {
        Value::Ptr(0)
    } else {
        Value::Int(0)
    }
}

fn int_bits(ty: &MirType) -> u32 {
    match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => 64,
    }
}

/// Truncate an integer value to a result type's bit width (signed wrap).
fn mask(v: i128, ty: &MirType) -> i128 {
    let bits = int_bits(ty);
    if bits >= 128 {
        return v;
    }
    let shift = 128 - bits;
    (v << shift) >> shift // sign-extend from `bits`
}

fn apply_bin(op: BinOp, a: Value, b: Value) -> Value {
    use BinOp::*;
    if op.is_float() {
        let (x, y) = (a.as_float(), b.as_float());
        return Value::Float(match op {
            FAdd => x + y,
            FSub => x - y,
            FMul => x * y,
            FDiv => x / y,
            _ => unreachable!(),
        });
    }
    let (x, y) = (a.as_int(), b.as_int());
    Value::Int(match op {
        Add => x.wrapping_add(y),
        Sub => x.wrapping_sub(y),
        Mul => x.wrapping_mul(y),
        SDiv => {
            if y == 0 {
                0
            } else {
                x.wrapping_div(y)
            }
        }
        UDiv => {
            if y == 0 {
                0
            } else {
                ((x as u128) / (y as u128)) as i128
            }
        }
        SRem => {
            if y == 0 {
                0
            } else {
                x.wrapping_rem(y)
            }
        }
        URem => {
            if y == 0 {
                0
            } else {
                ((x as u128) % (y as u128)) as i128
            }
        }
        And => x & y,
        Or => x | y,
        Xor => x ^ y,
        Shl => x.wrapping_shl(y as u32),
        LShr => ((x as u128) >> (y as u32)) as i128,
        AShr => x >> (y as u32),
        FAdd | FSub | FMul | FDiv => unreachable!(),
    })
}

fn apply_cmp(op: CmpOp, a: Value, b: Value) -> bool {
    use CmpOp::*;
    if op.is_float() {
        let (x, y) = (a.as_float(), b.as_float());
        return match op {
            Foeq => x == y,
            Fone => x != y,
            Folt => x < y,
            Fole => x <= y,
            Fogt => x > y,
            Foge => x >= y,
            _ => unreachable!(),
        };
    }
    let (x, y) = (a.as_int(), b.as_int());
    let (ux, uy) = (x as u128, y as u128);
    match op {
        Eq => x == y,
        Ne => x != y,
        Slt => x < y,
        Sle => x <= y,
        Sgt => x > y,
        Sge => x >= y,
        Ult => ux < uy,
        Ule => ux <= uy,
        Ugt => ux > uy,
        Uge => ux >= uy,
        _ => unreachable!(),
    }
}

fn apply_cast(kind: CastKind, v: Value, to: &MirType) -> Value {
    use CastKind::*;
    match kind {
        SExt | ZExt | Trunc => Value::Int(mask(v.as_int(), to)),
        SiToFp | UiToFp => Value::Float(v.as_int() as f64),
        FpToSi | FpToUi => Value::Int(mask(v.as_float() as i128, to)),
        FpExt | FpTrunc => Value::Float(v.as_float()),
        Bitcast => v,
        IntToPtr => Value::Ptr(v.as_int() as usize),
        PtrToInt => Value::Int(v.as_int()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_span::SourceId;

    fn run_main(src: &str) -> i64 {
        let mut interner = Interner::new();
        let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _ld) = mercury_mir_build::lower_program(&module, &sema, &interner);
        let main = interner.intern("main");
        run(&program, main, &interner).unwrap()
    }

    #[test]
    fn runs_loop_sum() {
        let src = "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                   while i < 10 { s += i; i += 1; } return s; }";
        assert_eq!(run_main(src), 45);
    }

    #[test]
    fn runs_recursive_fib() {
        let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } \
                   return fib(n - 1) + fib(n - 2); } \
                   fn main() -> i32 { return fib(10); }";
        assert_eq!(run_main(src), 55);
    }

    #[test]
    fn print_intrinsic_captures_stdout() {
        let mut interner = Interner::new();
        let src = "fn main() -> i32 { print(42); print(7 * 6); return 0; }";
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = mercury_sema::check(&module, &interner);
        let (program, _) = mercury_mir_build::lower_program(&module, &sema, &interner);
        let main = interner.intern("main");
        let (code, out) = run_with_output(&program, main, &interner).unwrap();
        assert_eq!(code, 0);
        assert_eq!(String::from_utf8(out).unwrap(), "42\n42\n");
    }

    #[test]
    fn runs_for_range_with_step() {
        let src = "fn main() -> i32 { let mut s: i32 = 0; \
                   for i in 0..10 step 2 { s += i; } return s; }";
        // 0 + 2 + 4 + 6 + 8 = 20
        assert_eq!(run_main(src), 20);
    }
}
