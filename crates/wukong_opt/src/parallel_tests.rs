//! The parallel schedule's laws: every thread count produces byte-identical MIR, a failure is the
//! one the sequential schedule reports, and a pass that cannot leave its thread never does.

use std::collections::BTreeSet;
use std::sync::Mutex;

use wukong_mir::{Builder, Function, MirType, Op, Program, Terminator, ValueId};
use wukong_span::{Interner, SourceId, Symbol};

use crate::effects::CallGraph;
use crate::{
    optimize_preserving_with, optimize_timed_with, optimize_verified_preserving_with, CfgAnalyses,
    OptimizeOptions, Pass, PassManager,
};

fn lower(src: &str, it: &mut Interner) -> Program {
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), it);
    assert!(pd.iter().all(|d| !d.is_error()), "{pd:?}");
    let (sema, sd) = wukong_sema::check(&module, it);
    assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
    let (program, ld) = wukong_mir_build::lower_program(&module, &sema, it);
    assert!(ld.iter().all(|d| !d.is_error()), "{ld:?}");
    program
}

fn corpus(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/run")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Twelve kernels too large to inline, each with loops the vectorizer and the unroller take, two
/// mutually recursive functions, and a call chain three deep — so the program has several levels,
/// each with several functions, and every per-function phase has work in more than one of them.
fn many_functions() -> String {
    let mut src = String::new();
    for k in 0..12 {
        src.push_str(&format!(
            "fn k{k}(x: [f32; 64], mut y: [f32; 64], s: f32) -> f32 {{\n\
             let mut acc: f32 = 0.0;\n\
             for i in 0..64 {{ y[i] = x[i] * s + {k}.0; }}\n\
             for i in 0..64 {{ if y[i] > {k}.5 {{ acc = acc + y[i]; }} else {{ acc = acc - 1.0; }} }}\n\
             for i in 0..8 {{ y[i * 8] = y[i * 8] + acc; }}\n\
             let mut j: i64 = 0;\n\
             while j < 64 {{ y[j] = y[j] * 0.5; j = j + 1; }}\n\
             return acc + y[{k}];\n}}\n"
        ));
    }
    src.push_str(
        "fn even(n: i32) -> i32 { if n == 0 { return 1; } return odd(n - 1); }\n\
         fn odd(n: i32) -> i32 { if n == 0 { return 0; } return even(n - 1); }\n\
         fn c3(x: [f32; 64], mut y: [f32; 64]) -> f32 { return k0(x, y, 1.5) + k1(x, y, 2.0); }\n\
         fn c2(x: [f32; 64], mut y: [f32; 64]) -> f32 { return c3(x, y) * k2(x, y, 0.25); }\n\
         fn main() -> i32 {\n\
         let x: [f32; 64] = [1.0; 64];\n\
         let mut y: [f32; 64] = [0.0; 64];\n\
         let mut t: f32 = c2(x, y);\n",
    );
    for k in 3..12 {
        src.push_str(&format!("t = t + k{k}(x, y, {k}.0);\n"));
    }
    src.push_str("print(t as i32); print(even(7)); return 0;\n}\n");
    src
}

fn printed(program: &Program, it: &Interner) -> String {
    wukong_mir::print::print_program(program, it)
}

/// The law the determinism gate checks over the corpus from the outside: whatever the thread
/// count, every entry point produces the same MIR, byte for byte — here at -O1 and -O2, through
/// the plain, the verified and the timed entry points, on a synthesized program built to have wide
/// call-graph levels and on corpus programs with many functions.
#[test]
fn every_thread_count_produces_the_same_mir() {
    let mut sources = vec![("many_functions".to_string(), many_functions())];
    for name in [
        "vectorized_declines.wk",
        "vec4_speculation_raw_pointer.wk",
        "opt5_fuse_multiblock.wk",
        "vec5_slp_bundles.wk",
        "vec_bf16_lanes.wk",
        "tail_recursion.wk",
        "or_pattern_bindings.wk",
    ] {
        sources.push((name.to_string(), corpus(name)));
    }
    let mut engaged = false;
    for (name, src) in &sources {
        let mut it = Interner::new();
        let program = lower(src, &mut it);
        for level in [1u8, 2] {
            let one = OptimizeOptions::FULL.with_threads(1);
            let mut want = program.clone();
            optimize_preserving_with(&mut want, level, &[], one);
            let want = printed(&want, &it);
            for threads in [2u32, 3, 8] {
                let options = OptimizeOptions::FULL.with_threads(threads);
                let mut got = program.clone();
                optimize_preserving_with(&mut got, level, &[], options);
                assert!(
                    printed(&got, &it) == want,
                    "{name} -O{level}: {threads} threads changed the MIR"
                );
                let mut verified = program.clone();
                optimize_verified_preserving_with(&mut verified, level, &[], options).unwrap();
                assert!(
                    printed(&verified, &it) == want,
                    "{name} -O{level}: the verified entry on {threads} threads changed the MIR"
                );
                let mut timed = program.clone();
                let t = optimize_timed_with(&mut timed, level, options);
                engaged |= t.threads > 1;
                assert!(
                    printed(&timed, &it) == want,
                    "{name} -O{level}: the timed entry on {threads} threads changed the MIR"
                );
            }
            let mut auto = program.clone();
            optimize_preserving_with(&mut auto, level, &[], OptimizeOptions::FULL);
            assert!(
                printed(&auto, &it) == want,
                "{name} -O{level}: the automatic thread count changed the MIR"
            );
        }
    }
    assert!(
        engaged,
        "no batch ever reached the pool: the law above compared nothing parallel"
    );
}

/// Every SCC a level's function calls is summarized before that level starts.
#[test]
fn levels_put_every_callee_in_an_earlier_group() {
    let mut it = Interner::new();
    let program = lower(&many_functions(), &mut it);
    let graph = CallGraph::of(&program);
    let levels = graph.levels();
    let mut group_of = vec![usize::MAX; graph.sccs.len()];
    for (g, sccs) in levels.iter().enumerate() {
        assert!(
            sccs.windows(2).all(|w| w[0] < w[1]),
            "SCC order within a group"
        );
        for &scc in sccs {
            group_of[scc] = g;
        }
    }
    assert!(
        group_of.iter().all(|&g| g != usize::MAX),
        "every SCC has a level"
    );
    let scc_of = |name: &str| {
        let sym = it.get(name).unwrap();
        let i = program.funcs.iter().position(|f| f.name == sym).unwrap() as u32;
        graph.sccs.iter().position(|m| m.contains(&i)).unwrap()
    };
    for f in &program.funcs {
        let caller = scc_of(it.resolve(f.name));
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            if let Op::Call { func, .. } = &inst.op {
                if program.function(*func).is_some() {
                    let callee = scc_of(it.resolve(*func));
                    assert!(
                        callee == caller || group_of[callee] < group_of[caller],
                        "{} calls {} at the same or a later level",
                        it.resolve(f.name),
                        it.resolve(*func)
                    );
                }
            }
        }
    }
    assert_eq!(scc_of("even"), scc_of("odd"), "mutual recursion is one SCC");
    assert_eq!(group_of[scc_of("k0")], 0);
    assert_eq!(group_of[scc_of("c3")], 1);
    assert_eq!(group_of[scc_of("c2")], 2);
    assert_eq!(group_of[scc_of("main")], 3);
    assert!(levels[0].len() > 2, "the leaves share a level");
}

/// Breaks every function it names: an out-of-range return value, which the verifier rejects.
struct BreakNamed {
    names: Vec<Symbol>,
}

impl Pass for BreakNamed {
    fn name(&self) -> &'static str {
        "break-named"
    }

    fn run_function(&self, f: &mut Function, _: &mut CfgAnalyses) -> bool {
        if self.names.contains(&f.name) {
            let last = f.blocks.len() - 1;
            f.blocks[last].term = Terminator::Ret(Some(ValueId(f.value_types.len() as u32 + 7)));
            return true;
        }
        false
    }
}

/// `main -> mid -> leaf1`, `main -> leaf2`: Tarjan's order is leaf1, mid, leaf2, main, but the
/// levels are {leaf1, leaf2}, {mid}, {main}. With `mid` and `leaf2` both broken the sequential
/// schedule meets `mid` first; the level schedule sees `leaf2` fail first and must still report
/// `mid`.
fn chain(it: &mut Interner) -> Program {
    let mut funcs = Vec::new();
    let names = ["main", "mid", "leaf1", "leaf2"];
    let callees: [&[&str]; 4] = [&["mid", "leaf2"], &["leaf1"], &[], &[]];
    for (name, calls) in names.iter().zip(callees) {
        let mut b = Builder::new(it.intern(name), MirType::I64);
        let mut acc = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        for callee in calls {
            let r = b.build(
                MirType::I64,
                Op::Call {
                    func: it.intern(callee),
                    args: vec![],
                },
            );
            acc = b.build(MirType::I64, Op::Bin(wukong_mir::BinOp::Add, acc, r));
        }
        b.ret(Some(acc));
        funcs.push(b.finish());
    }
    let mut p = Program::new();
    p.funcs = funcs;
    assert!(wukong_mir::verify::verify_program(&p).is_empty());
    p
}

#[test]
fn a_failure_is_the_one_the_sequential_schedule_reports() {
    let mut it = Interner::new();
    let program = chain(&mut it);
    let broken = [it.intern("mid"), it.intern("leaf2")];
    let mut reports = Vec::new();
    for threads in [1u32, 2, 4] {
        let mut pm = PassManager::new().with_threads(threads);
        pm.add_shared(Box::new(BreakNamed {
            names: broken.to_vec(),
        }));
        let failure = pm.run_verified(&mut program.clone()).unwrap_err();
        assert_eq!(failure.pass, "break-named");
        assert_eq!(
            failure.function,
            Some(it.intern("mid")),
            "{threads} threads"
        );
        reports.push(failure);
    }
    assert!(reports.windows(2).all(|w| w[0] == w[1]), "{reports:?}");
}

/// Records which thread ran each function. The functions it names meet before going on: each waits
/// (a few seconds at most) until all of them have started, so a batch holding them cannot be
/// drained by one thread while the others are still waking.
struct Recorder {
    names: Vec<Symbol>,
    started: std::sync::atomic::AtomicUsize,
    threads: Mutex<BTreeSet<String>>,
}

impl Recorder {
    fn new(names: Vec<Symbol>) -> std::sync::Arc<Recorder> {
        std::sync::Arc::new(Recorder {
            names,
            started: std::sync::atomic::AtomicUsize::new(0),
            threads: Mutex::new(BTreeSet::new()),
        })
    }
}

/// Forwards to a shared [`Recorder`], so one test can watch the same pass added both ways.
struct Watch(std::sync::Arc<Recorder>);

impl Pass for Watch {
    fn name(&self) -> &'static str {
        "watch"
    }
    fn run_function(&self, f: &mut Function, _: &mut CfgAnalyses) -> bool {
        use std::sync::atomic::Ordering;
        let r = &self.0;
        if r.names.contains(&f.name) {
            r.started.fetch_add(1, Ordering::SeqCst);
            let t0 = std::time::Instant::now();
            while r.started.load(Ordering::SeqCst) < r.names.len() && t0.elapsed().as_secs() < 5 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        let thread = std::thread::current();
        r.threads
            .lock()
            .unwrap()
            .insert(thread.name().unwrap_or("").to_string());
        false
    }
}

/// A pass added with `add` stays on the calling thread whatever the thread count; the same pass
/// added with `add_shared` shares a level's functions with the pool, and stays on the calling
/// thread again at one thread.
#[test]
fn only_shared_passes_leave_the_calling_thread() {
    let mut it = Interner::new();
    let program = chain(&mut it);
    let caller = std::thread::current().name().unwrap_or("").to_string();

    let local = Recorder::new(Vec::new());
    let mut pm = PassManager::new().with_threads(4);
    pm.add(Box::new(Watch(local.clone())));
    pm.run(&mut program.clone());
    assert_eq!(
        *local.threads.lock().unwrap(),
        BTreeSet::from([caller.clone()])
    );

    // The two leaves share level 0 and wait for each other: two threads must have run them.
    let shared = Recorder::new(vec![it.intern("leaf1"), it.intern("leaf2")]);
    let mut pm = PassManager::new().with_threads(4);
    pm.add_shared(Box::new(Watch(shared.clone())));
    pm.run(&mut program.clone());
    let threads = shared.threads.lock().unwrap().clone();
    assert!(
        threads.iter().any(|t| t.starts_with("wukong-opt-")),
        "a forced count shares level 0 with the pool: {threads:?}"
    );
    // `mid` and `main` are alone in their levels and run on the calling thread.
    assert!(threads.contains(&caller), "{threads:?}");

    let one = Recorder::new(Vec::new());
    let mut pm = PassManager::new().with_threads(1);
    pm.add_shared(Box::new(Watch(one.clone())));
    pm.run(&mut program.clone());
    assert_eq!(*one.threads.lock().unwrap(), BTreeSet::from([caller]));
}
