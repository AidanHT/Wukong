//! Whole-program reachability pruning: drop every function the entry point cannot reach before the
//! optimizer and a backend spend time on it.
//!
//! `loader` splices every item of every imported file into one module and `mir_build` lowers every
//! function with a body, so without this an import-based library is optimized to fixpoint and
//! compiled by Cranelift in full even when the program calls one function of it. (The inliner's
//! cleanup removes only functions it inlined, and only at `-O2`.) This is the reachability
//! collector rustc runs before codegen, at MIR granularity.
//!
//! Type checking still covers every item — sema has already run — so pruning changes what is
//! *compiled*, never what is *accepted*. The driver applies it only where the program has a single
//! known entry point (`--run` and `--emit=exe`, rooted at `main`): an object file exports every
//! function, and the MIR dumps show the whole program.

use wukong_mir::{Op, Program};
use wukong_span::{FxHashMap, FxHashSet, Symbol};

/// Keep only the functions reachable from `roots` through direct calls and `FuncAddr` references
/// (the outlined body of a `@parallel` region is reached only by its address, exactly the rule the
/// inliner's cleanup uses), preserving their order. Returns how many functions were removed.
///
/// A root the program does not define leaves the program untouched, so the backend reports the
/// missing entry point exactly as it would have.
pub(crate) fn retain_reachable(program: &mut Program, roots: &[Symbol]) -> usize {
    let index: FxHashMap<Symbol, usize> = program
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name, i))
        .collect();
    if roots.iter().any(|root| !index.contains_key(root)) {
        return 0;
    }
    let mut reached: FxHashSet<Symbol> = FxHashSet::default();
    let mut work: Vec<Symbol> = Vec::new();
    for &root in roots {
        if reached.insert(root) {
            work.push(root);
        }
    }
    while let Some(name) = work.pop() {
        let f = &program.funcs[index[&name]];
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            let target = match &inst.op {
                Op::Call { func, .. } => *func,
                Op::FuncAddr(func) => *func,
                _ => continue,
            };
            // A callee the program does not define is a runtime kernel or an intrinsic.
            if index.contains_key(&target) && reached.insert(target) {
                work.push(target);
            }
        }
    }
    let before = program.funcs.len();
    program.funcs.retain(|f| reached.contains(&f.name));
    before - program.funcs.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{Builder, Function, MirType};
    use wukong_span::Interner;

    /// A void function that calls each of `calls` and takes the address of each of `addrs`.
    fn func(interner: &mut Interner, name: &str, calls: &[&str], addrs: &[&str]) -> Function {
        let mut b = Builder::new(interner.intern(name), MirType::Void);
        for callee in calls {
            let func = interner.intern(callee);
            b.build_void(Op::Call {
                func,
                args: Vec::new(),
            });
        }
        for target in addrs {
            let target = interner.intern(target);
            b.build(MirType::Ptr, Op::FuncAddr(target));
        }
        b.ret(None);
        b.finish()
    }

    fn names(program: &Program, interner: &Interner) -> Vec<String> {
        program
            .funcs
            .iter()
            .map(|f| interner.resolve(f.name).to_owned())
            .collect()
    }

    #[test]
    fn keeps_call_and_address_reachable_functions_in_order() {
        let mut interner = Interner::new();
        let funcs = vec![
            func(&mut interner, "unused_helper", &["leaf"], &[]),
            func(&mut interner, "leaf", &[], &[]),
            func(&mut interner, "region", &["wukong_sgemm"], &[]),
            func(&mut interner, "mid", &["leaf"], &["region"]),
            func(&mut interner, "dead_cycle_a", &["dead_cycle_b"], &[]),
            func(&mut interner, "dead_cycle_b", &["dead_cycle_a"], &[]),
            func(&mut interner, "main", &["mid", "print"], &[]),
        ];
        let mut program = Program {
            funcs,
            ..Program::new()
        };
        let main = interner.intern("main");
        assert_eq!(retain_reachable(&mut program, &[main]), 3);
        assert_eq!(
            names(&program, &interner),
            ["leaf", "region", "mid", "main"],
            "reachable functions keep their source order"
        );
        // Pruning a pruned program is a no-op.
        assert_eq!(retain_reachable(&mut program, &[main]), 0);
    }

    #[test]
    fn an_undefined_root_leaves_the_program_whole() {
        let mut interner = Interner::new();
        let funcs = vec![
            func(&mut interner, "a", &[], &[]),
            func(&mut interner, "b", &["a"], &[]),
        ];
        let mut program = Program {
            funcs,
            ..Program::new()
        };
        let main = interner.intern("main");
        assert_eq!(retain_reachable(&mut program, &[main]), 0);
        assert_eq!(names(&program, &interner), ["a", "b"]);
    }
}
