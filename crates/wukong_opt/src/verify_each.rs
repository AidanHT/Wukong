//! Fallible verifier boundaries shared by release and debug optimizer callers.

use wukong_mir::{Function, Program};
use wukong_span::Symbol;

/// An internal compiler error from a specific optimization pass or whole-program boundary.
/// `function` is present for per-function passes; whole-program diagnostics carry their own
/// function identifiers in `errors`. The caller may resolve the symbol and render human or JSON
/// diagnostics without depending on a panic hook or parsing a single formatted string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationFailure {
    pub pass: &'static str,
    pub function: Option<Symbol>,
    pub errors: Vec<String>,
}

impl std::fmt::Display for VerificationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MIR verification failed after `{}`", self.pass)?;
        if let Some(function) = self.function {
            write!(f, " in {function:?}")?;
        }
        write!(f, ": {}", self.errors.join("; "))
    }
}

impl std::error::Error for VerificationFailure {}

fn checked(
    pass: &'static str,
    function: Option<Symbol>,
    errors: Vec<String>,
) -> Result<(), VerificationFailure> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(VerificationFailure {
            pass,
            function,
            errors,
        })
    }
}

/// Check a complete program, including defined callees' signatures, at a named stage. Available
/// regardless of build profile; also usable at synthesis boundaries such as autodiff output.
pub fn verify_program_stage(
    program: &Program,
    pass: &'static str,
) -> Result<(), VerificationFailure> {
    checked(pass, None, wukong_mir::verify::verify_program(program))
}

pub(crate) fn verify_function_stage(
    f: &Function,
    pass: &'static str,
) -> Result<(), VerificationFailure> {
    checked(pass, Some(f.name), wukong_mir::verify::verify_function(f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CfgAnalyses, Pass, PassManager};
    use wukong_mir::{Builder, MirType, Op, Terminator, ValueId};
    use wukong_span::{Interner, SourceId};

    fn valid_program(it: &mut Interner) -> Program {
        let mut b = Builder::new(it.intern("main"), MirType::I32);
        let value = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        b.ret(Some(value));
        let mut p = Program::new();
        p.funcs.push(b.finish());
        p
    }

    struct BreakReturn(bool);
    impl Pass for BreakReturn {
        fn name(&self) -> &'static str {
            "break-return"
        }
        fn run_function(&self, f: &mut Function, _: &mut CfgAnalyses) -> bool {
            f.blocks[0].term = Terminator::Ret(Some(ValueId(f.value_types.len() as u32 + 1)));
            self.0
        }
    }

    struct MustNotRun;
    impl Pass for MustNotRun {
        fn name(&self) -> &'static str {
            "must-not-run"
        }
        fn run_function(&self, _: &mut Function, _: &mut CfgAnalyses) -> bool {
            panic!("a later pass consumed malformed MIR");
        }
    }

    #[test]
    fn verify_each_reports_broken_pass_even_when_it_claims_no_change() {
        for changed in [false, true] {
            let mut it = Interner::new();
            let mut program = valid_program(&mut it);
            let mut pm = PassManager::new();
            pm.add(Box::new(BreakReturn(changed)));
            pm.add(Box::new(MustNotRun));
            let failure = pm.run_verified(&mut program).unwrap_err();
            assert_eq!(failure.pass, "break-return");
            assert_eq!(failure.function, Some(program.funcs[0].name));
            assert!(!failure.errors.is_empty());
        }
    }

    #[test]
    fn run_subset_verified_reports_broken_pass_and_function_without_panicking() {
        let mut it = Interner::new();
        let mut program = valid_program(&mut it);
        let mut b = Builder::new(it.intern("swept"), MirType::I32);
        let value = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        b.ret(Some(value));
        program.funcs.push(b.finish());
        let mut pm = PassManager::new();
        pm.add(Box::new(BreakReturn(true)));
        pm.add(Box::new(MustNotRun));
        let failure = pm.run_subset_verified(&mut program, &[1]).unwrap_err();
        assert_eq!(failure.pass, "break-return");
        assert_eq!(failure.function, Some(program.funcs[1].name));
        assert!(!failure.errors.is_empty());
        // Only the subset is swept.
        assert!(wukong_mir::verify::verify_function(&program.funcs[0]).is_empty());
        assert_eq!(pm.run_subset_verified(&mut program, &[]), Ok(()));
    }

    #[test]
    fn verify_each_checks_input_before_any_pass_and_at_o0() {
        let mut it = Interner::new();
        let mut program = valid_program(&mut it);
        BreakReturn(false).run_function(&mut program.funcs[0], &mut CfgAnalyses::default());
        let mut pm = PassManager::new();
        pm.add(Box::new(MustNotRun));
        assert_eq!(
            pm.run_verified(&mut program).unwrap_err().pass,
            "pipeline input"
        );
        for level in 0..=3 {
            let failure = crate::optimize_verified(&mut program.clone(), level).unwrap_err();
            assert_eq!(failure.pass, "optimization input");
            assert!(!failure.errors.is_empty());
        }
    }

    #[test]
    fn verify_each_checks_whole_program_call_signatures() {
        let mut it = Interner::new();
        let callee = it.intern("callee");
        let mut b = Builder::new(callee, MirType::Void);
        b.add_param(MirType::I32);
        b.ret(None);
        let callee = b.finish();
        let mut b = Builder::new(it.intern("main"), MirType::Void);
        let wrong = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        b.build_void(Op::Call {
            func: callee.name,
            args: vec![wrong],
        });
        b.ret(None);
        let mut p = Program::new();
        p.funcs = vec![callee, b.finish()];
        assert!(p
            .funcs
            .iter()
            .all(|f| wukong_mir::verify::verify_function(f).is_empty()));
        let failure = crate::optimize_verified(&mut p, 0).unwrap_err();
        assert_eq!(failure.pass, "optimization input");
        assert!(!failure.errors.is_empty());
    }

    #[test]
    fn verify_each_preserves_pipeline_mir_and_values_at_every_level() {
        let mut it = Interner::new();
        let src = "fn inc(x: i32) -> i32 { return x + 1; } \
                   fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                   while i < 17 { s = s + inc(i); i = i + 1; } return s; }";
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut it);
        assert!(pd.is_empty(), "{pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &it);
        assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut it);
        assert!(ld.iter().all(|d| !d.is_error()), "{ld:?}");
        let main = it.intern("main");
        for level in 0..=3 {
            let mut ordinary = program.clone();
            crate::optimize(&mut ordinary, level);
            let mut checked = program.clone();
            crate::optimize_verified(&mut checked, level).unwrap();
            assert_eq!(
                wukong_mir::print::print_program(&checked, &it),
                wukong_mir::print::print_program(&ordinary, &it),
                "level {level}"
            );
            assert_eq!(wukong_interp::run(&checked, main, &it).unwrap(), 153);
        }
    }
}
