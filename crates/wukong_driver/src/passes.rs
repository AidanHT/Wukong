//! `--passes=<list>` and `--print-after=<pass|all>`: run a chosen optimizer pipeline, and watch it.
//!
//! The standard pipeline is `wukong_opt::optimize_preserving` — whole-program inlining, the
//! `PassManager::standard` fixpoint, unrolling and SLP with their cleanups, then switch formation.
//! This module is a **replica** of it built only from `wukong_opt`'s public API, so a user (or a
//! regression test) can run any sub-list of it, in any order, and see the MIR right after a named
//! pass. [`standard_spec`] spells each `-O` level's pipeline in the list syntax, and the law test
//! below holds the replica to it byte for byte: `run_steps(parse_spec(standard_spec(l)))` prints
//! exactly what `wukong_opt::optimize(p, l)` prints, at every level. A change to the standard
//! pipeline that is not mirrored here fails that test — and every post-pipeline stage added to
//! `optimize_preserving` must become a [`Step`] the same way `slp` did.
//!
//! The replica has no autodiff roots (`inline_program_preserving` is crate-private), so the CLI
//! refuses both flags on a gradient compile, which always runs the standard pipeline.

use std::io::Write as _;
use std::rc::Rc;

use wukong_mir::{Function, Program};
use wukong_opt::{CfgAnalyses, Pass, PassManager, VerificationFailure};
use wukong_span::{Interner, Symbol};

/// Every function pass `--passes` accepts, by its `Pass::name()`, in standard-pipeline order.
pub const FUNCTION_PASSES: &[&str] = &[
    "sroa",
    "mem2reg",
    "simplify",
    "jump-thread",
    "simplify-cfg",
    "simplify-phis",
    "dce",
    "tre",
    "cse",
    "dse",
    "loop-canon",
    "loop-delete",
    "fuse",
    "licm",
    "bce",
    "unswitch",
    "promote",
    "vectorize",
];

/// The whole-program steps `--passes` accepts. Each runs once where it is named and closes the
/// fixpoint group before it: `inline` (`wukong_opt::inline_program`), `unroll`
/// (`unroll_program` plus its cleanup sweep over the functions it changed), `slp` (`slp_program`
/// plus the same cleanup) and `form-switches`.
pub const PROGRAM_STEPS: &[&str] = &["inline", "unroll", "slp", "form-switches"];

/// The `--print-after` name that matches every pass and every program step.
pub const PRINT_ALL: &str = "all";

/// A fresh instance of the function pass named `name` (a [`FUNCTION_PASSES`] entry).
pub fn make_pass(name: &str) -> Option<Box<dyn Pass>> {
    use wukong_opt::*;
    Some(match name {
        "sroa" => Box::new(Sroa),
        "mem2reg" => Box::new(Mem2Reg),
        "simplify" => Box::new(Simplify),
        "jump-thread" => Box::new(JumpThread),
        "simplify-cfg" => Box::new(SimplifyCfg),
        "simplify-phis" => Box::new(SimplifyPhis),
        "dce" => Box::new(Dce),
        "tre" => Box::new(Tre),
        "cse" => Box::new(Cse),
        "dse" => Box::new(Dse),
        "loop-canon" => Box::new(LoopCanon),
        "loop-delete" => Box::new(LoopDelete),
        "fuse" => Box::new(LoopFuse),
        "licm" => Box::new(Licm),
        "bce" => Box::new(Bce),
        "unswitch" => Box::new(LoopUnswitch),
        "promote" => Box::new(Promote),
        "vectorize" => Box::new(Vectorize::default()),
        _ => return None,
    })
}

/// One stage of a pass list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Consecutive function passes, run together to a per-function fixpoint exactly as
    /// `PassManager::run` runs the standard pipeline (callees first, call effects summarized per
    /// call-graph SCC).
    Fixpoint(Vec<&'static str>),
    Inline,
    Unroll,
    Slp,
    FormSwitches,
}

impl Step {
    fn program_step(name: &str) -> Option<Step> {
        Some(match name {
            "inline" => Step::Inline,
            "unroll" => Step::Unroll,
            "slp" => Step::Slp,
            "form-switches" => Step::FormSwitches,
            _ => return None,
        })
    }

    fn name(&self) -> &'static str {
        match self {
            Step::Fixpoint(_) => "fixpoint",
            Step::Inline => "inline",
            Step::Unroll => "unroll",
            Step::Slp => "slp",
            Step::FormSwitches => "form-switches",
        }
    }
}

/// The valid names, for an error message.
fn valid_names() -> String {
    format!(
        "the function passes are {}; the program steps are {}",
        FUNCTION_PASSES.join(", "),
        PROGRAM_STEPS.join(", ")
    )
}

/// Parse a comma-separated pass list. Consecutive function passes form one fixpoint group; a
/// program step closes the group before it. An empty (or all-blank) list means no passes at all.
pub fn parse_spec(spec: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    if spec.trim().is_empty() {
        return Ok(steps);
    }
    let mut group: Vec<&'static str> = Vec::new();
    for raw in spec.split(',') {
        let name = raw.trim();
        if let Some(&pass) = FUNCTION_PASSES.iter().find(|&&p| p == name) {
            group.push(pass);
        } else if let Some(step) = Step::program_step(name) {
            if !group.is_empty() {
                steps.push(Step::Fixpoint(std::mem::take(&mut group)));
            }
            steps.push(step);
        } else if name.is_empty() {
            return Err(format!("empty pass name in `{spec}`"));
        } else {
            return Err(format!("unknown pass `{name}`: {}", valid_names()));
        }
    }
    if !group.is_empty() {
        steps.push(Step::Fixpoint(group));
    }
    Ok(steps)
}

/// Check a `--print-after` name: a function pass, a program step, or `all`.
pub fn check_print_after(name: &str) -> Result<(), String> {
    if name == PRINT_ALL || FUNCTION_PASSES.contains(&name) || PROGRAM_STEPS.contains(&name) {
        Ok(())
    } else {
        Err(format!(
            "unknown pass `{name}`: {}; `{PRINT_ALL}` matches every one",
            valid_names()
        ))
    }
}

/// The pass list `-O<level>` runs, in `--passes` syntax: `PassManager::standard(level)` plus the
/// whole-program stages `wukong_opt::optimize_preserving` wraps around it. `-O3` is `-O2`.
pub fn standard_spec(level: u8) -> &'static str {
    match level {
        0 => "",
        1 => "mem2reg,simplify,simplify-cfg,simplify-phis,dce,tre",
        _ => {
            "inline,sroa,mem2reg,simplify,jump-thread,simplify-cfg,simplify-phis,dce,tre,cse,dse,\
             loop-canon,loop-delete,fuse,licm,bce,unswitch,promote,vectorize,unroll,slp,\
             form-switches"
        }
    }
}

/// Where a snapshot's text goes: stderr in a compile, a buffer in a test.
type Sink = Box<dyn Fn(&str)>;

/// Write a snapshot to stderr, ignoring a closed stream (the same rule as the driver's own
/// diagnostic lines).
fn to_stderr(s: &str) {
    let _ = std::io::stderr().write_all(s.as_bytes());
}

/// Where `--print-after` snapshots go: a filter, plus a private copy of the compile's interner so
/// a pass wrapper (which `PassManager` needs to be `'static`) can render the function it just
/// changed the moment it changes. A snapshot is therefore out before the next pass runs — one
/// that panics still leaves every earlier snapshot on stderr — and none is ever buffered.
struct Printer {
    filter: String,
    interner: Interner,
    sink: Sink,
}

impl Printer {
    fn new(filter: &str, interner: &Interner, sink: Sink) -> Printer {
        // Symbols are dense indices in minting order, so re-interning every string in order gives
        // each one the same `Symbol` it has in the original.
        let mut copy = Interner::new();
        for i in 0..interner.len() {
            let sym = Symbol(i as u32);
            let again = copy.intern(interner.resolve(sym));
            assert_eq!(again, sym, "interner copy must preserve symbol numbering");
        }
        Printer {
            filter: filter.to_string(),
            interner: copy,
            sink,
        }
    }

    fn wants(&self, name: &str) -> bool {
        self.filter == PRINT_ALL || self.filter == name
    }

    /// One run of `pass` over `f`: the function as the run left it, or a one-line note when the
    /// run changed nothing (so a pass that never fires still shows that it ran, and where).
    fn function(&self, pass: &str, f: &Function, changed: bool) {
        let name = self.interner.resolve(f.name);
        if changed {
            let text = wukong_mir::print::print_function(f, &self.interner);
            (self.sink)(&format!("; *** MIR after {pass} (fn {name}) ***\n{text}"));
        } else {
            (self.sink)(&format!(
                "; *** MIR after {pass} (fn {name}): unchanged ***\n"
            ));
        }
    }

    fn program(&self, step: &str, p: &Program) {
        let text = wukong_mir::print::print_program(p, &self.interner);
        (self.sink)(&format!("; *** MIR after {step} ***\n{text}"));
    }
}

/// A pass that behaves exactly like `inner` — same name, same result, same mutation — and reports
/// every run: the function it changed, or a note that it changed nothing.
struct Observed {
    inner: Box<dyn Pass>,
    printer: Rc<Printer>,
}

impl Pass for Observed {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let changed = self.inner.run_function(f, cache);
        self.printer.function(self.inner.name(), f, changed);
        changed
    }
}

/// Run `steps` over `program`. Under `verify_each` every pass and step is followed by the MIR
/// verifier and the first failure is returned (the same boundaries
/// `wukong_opt::optimize_verified` checks); otherwise passes verify only in debug builds, as
/// `wukong_opt::optimize` does. `print_after` (a pass or step name, or `all`) prints to stderr:
/// after each run of the named function pass, the function it changed (or a one-line
/// `unchanged` note), and after the named program step, the whole program. The fixpoint driver
/// skips a pass that has nothing new to see, so a pass runs on a function only as often as the
/// rest of its group keeps changing it.
pub fn run_steps(
    program: &mut Program,
    steps: &[Step],
    verify_each: bool,
    print_after: Option<&str>,
    interner: &Interner,
) -> Result<(), VerificationFailure> {
    run_steps_to(
        program,
        steps,
        verify_each,
        print_after,
        interner,
        Box::new(to_stderr),
    )
}

/// [`run_steps`], writing the snapshots to `sink`.
fn run_steps_to(
    program: &mut Program,
    steps: &[Step],
    verify_each: bool,
    print_after: Option<&str>,
    interner: &Interner,
    sink: Sink,
) -> Result<(), VerificationFailure> {
    let printer = print_after.map(|filter| Rc::new(Printer::new(filter, interner, sink)));
    if verify_each {
        wukong_opt::verify_program_stage(program, "optimization input")?;
    }
    for step in steps {
        match step {
            Step::Fixpoint(names) => {
                let mut pm = PassManager::new();
                for name in names {
                    let inner = make_pass(name).expect("parse_spec admits only known passes");
                    match &printer {
                        Some(printer) if printer.wants(name) => pm.add(Box::new(Observed {
                            inner,
                            printer: Rc::clone(printer),
                        })),
                        _ => pm.add(inner),
                    }
                }
                if verify_each {
                    pm.run_verified(program)?;
                } else {
                    pm.run(program);
                }
                continue;
            }
            Step::Inline => {
                wukong_opt::inline_program(program);
                if verify_each {
                    wukong_opt::verify_program_stage(program, "inline")?;
                }
            }
            Step::Unroll => {
                let changed = wukong_opt::unroll_program(program);
                if verify_each {
                    wukong_opt::verify_program_stage(program, "unroll")?;
                }
                PassManager::unroll_cleanup().run_subset(program, &changed);
                if verify_each {
                    wukong_opt::verify_program_stage(program, "unroll cleanup")?;
                }
            }
            Step::Slp => {
                // `slp_program` reads the `WUKONG_NO_SLP` kill switch itself, so the replica
                // honours it exactly as `optimize` does.
                let packed = wukong_opt::slp_program(program);
                if verify_each {
                    wukong_opt::verify_program_stage(program, "slp")?;
                }
                PassManager::unroll_cleanup().run_subset(program, &packed);
                if verify_each {
                    wukong_opt::verify_program_stage(program, "slp cleanup")?;
                }
            }
            Step::FormSwitches => {
                wukong_opt::form_switches(program);
                if verify_each {
                    wukong_opt::verify_program_stage(program, "form-switches")?;
                }
            }
        }
        if let Some(printer) = &printer {
            if printer.wants(step.name()) {
                printer.program(step.name(), program);
            }
        }
    }
    if verify_each {
        wukong_opt::verify_program_stage(program, "optimization output")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::SourceMap;

    /// Every name the CLI accepts builds the pass it names.
    #[test]
    fn make_pass_builds_every_function_pass_by_its_own_name() {
        for name in FUNCTION_PASSES {
            let pass = make_pass(name).unwrap_or_else(|| panic!("no pass for `{name}`"));
            assert_eq!(pass.name(), *name);
        }
        assert!(
            make_pass("inline").is_none(),
            "a program step is not a function pass"
        );
        assert!(make_pass("bogus").is_none());
        for step in PROGRAM_STEPS {
            assert_eq!(Step::program_step(step).map(|s| s.name()), Some(*step));
        }
    }

    #[test]
    fn parse_spec_groups_function_passes_between_program_steps() {
        assert_eq!(parse_spec(""), Ok(vec![]));
        assert_eq!(parse_spec("  "), Ok(vec![]));
        assert_eq!(
            parse_spec("mem2reg, simplify,dce"),
            Ok(vec![Step::Fixpoint(vec!["mem2reg", "simplify", "dce"])])
        );
        assert_eq!(
            parse_spec("inline,mem2reg,dce,unroll,slp,cse,form-switches"),
            Ok(vec![
                Step::Inline,
                Step::Fixpoint(vec!["mem2reg", "dce"]),
                Step::Unroll,
                Step::Slp,
                Step::Fixpoint(vec!["cse"]),
                Step::FormSwitches,
            ])
        );
        let err = parse_spec("mem2reg,bogus").unwrap_err();
        assert!(err.contains("`bogus`"), "{err}");
        for name in FUNCTION_PASSES.iter().chain(PROGRAM_STEPS) {
            assert!(err.contains(name), "the error must list `{name}`: {err}");
        }
        assert!(parse_spec("mem2reg,,dce").is_err());
        assert!(check_print_after("licm").is_ok());
        assert!(check_print_after("slp").is_ok());
        assert!(check_print_after("all").is_ok());
        assert!(check_print_after("bogus")
            .unwrap_err()
            .contains("vectorize"));
    }

    /// Each level's spec parses, and names each pass of `PassManager::standard` once.
    #[test]
    fn standard_specs_parse() {
        for level in 0..=3u8 {
            parse_spec(standard_spec(level)).unwrap();
        }
        assert_eq!(standard_spec(3), standard_spec(2));
        let o2 = parse_spec(standard_spec(2)).unwrap();
        assert_eq!(o2.first(), Some(&Step::Inline));
        assert_eq!(
            o2[1..],
            [
                Step::Fixpoint(FUNCTION_PASSES.to_vec()),
                Step::Unroll,
                Step::Slp,
                Step::FormSwitches,
            ]
        );
    }

    /// Programs that between them reach every stage: inlining, the loop passes and the vectorizer,
    /// complete unrolling, SLP packing, switch formation and tail-call elimination.
    const LAW_PROGRAMS: &[&str] = &[
        "examples/fib.wk",
        "examples/saxpy_array.wk",
        "tests/run/tensor_matmul.wk",
        "tests/run/loop_unroll.wk",
        "tests/run/match_dense_enum_switch.wk",
        "tests/run/tail_recursion.wk",
        "tests/run/vec5_slp_bundles.wk",
        "tests/run/vec5_scatter.wk",
        "tests/run/scalar_gvn_unroll.wk",
    ];

    fn lower(rel: &str) -> (Program, Interner) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel);
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let mut sm = SourceMap::new();
        let id = sm.add(rel.to_string(), src);
        let (tokens, _) = wukong_lexer::tokenize(sm.source(id), id);
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "{rel}: parse errors");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "{rel}: sema errors");
        let (program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "{rel}: lowering errors");
        (program, interner)
    }

    /// The replica is the standard pipeline: at every level, running `standard_spec(level)`
    /// through `run_steps` (verified or not) leaves byte-identical MIR to `wukong_opt::optimize`.
    /// A pass added to, removed from or reordered in `PassManager::standard`, or a new stage in
    /// `optimize_preserving`, fails here until `standard_spec` and [`Step`] follow it.
    #[test]
    fn standard_spec_replica_matches_optimize_at_every_level() {
        let mut fired = std::collections::BTreeSet::new();
        for rel in LAW_PROGRAMS {
            let (program, interner) = lower(rel);
            let lowered = wukong_mir::print::print_program(&program, &interner);
            for level in 0..=3u8 {
                let mut want = program.clone();
                wukong_opt::optimize(&mut want, level);
                let want = wukong_mir::print::print_program(&want, &interner);
                let steps = parse_spec(standard_spec(level)).unwrap();
                for verify_each in [false, true] {
                    let mut got = program.clone();
                    run_steps(&mut got, &steps, verify_each, None, &interner).unwrap();
                    let got = wukong_mir::print::print_program(&got, &interner);
                    assert!(
                        got == want,
                        "{rel} -O{level} (verify_each={verify_each}): the replica diverged from \
                         wukong_opt::optimize\n--- optimize ---\n{want}\n--- replica ---\n{got}"
                    );
                }
                if level == 2 && want != lowered {
                    fired.insert(*rel);
                }
            }
        }
        // The law is only as strong as what the optimizer did to the programs.
        assert_eq!(fired.len(), LAW_PROGRAMS.len(), "{fired:?}");
    }

    /// Run the -O2 list over `rel` with `--print-after=<filter>`, returning the snapshots.
    fn snapshots(rel: &str, filter: &str) -> (String, String, String) {
        let (program, interner) = lower(rel);
        let steps = parse_spec(standard_spec(2)).unwrap();
        let mut plain = program.clone();
        run_steps_to(&mut plain, &steps, false, None, &interner, Box::new(|_| {})).unwrap();
        let log = Rc::new(std::cell::RefCell::new(String::new()));
        let sink = Rc::clone(&log);
        let mut watched = program;
        run_steps_to(
            &mut watched,
            &steps,
            false,
            Some(filter),
            &interner,
            Box::new(move |s| sink.borrow_mut().push_str(s)),
        )
        .unwrap();
        let log = log.borrow().clone();
        (
            log,
            wukong_mir::print::print_program(&plain, &interner),
            wukong_mir::print::print_program(&watched, &interner),
        )
    }

    /// Printing is observation only: a watched run leaves the same MIR. A function pass prints
    /// the function after each run that changed it; a program step prints the whole program.
    #[test]
    fn print_after_prints_the_named_pass_and_changes_nothing() {
        let (log, plain, watched) = snapshots("tests/run/loop_unroll.wk", "simplify");
        assert_eq!(watched, plain);
        assert!(log.starts_with("; *** MIR after simplify (fn "), "{log}");
        let headers: Vec<&str> = log.lines().filter(|l| l.starts_with("; ***")).collect();
        assert!(
            headers.iter().all(|h| h.contains("after simplify (fn ")),
            "{headers:?}"
        );
        // Both kinds of run are reported: the pass changes a function, then finds nothing more.
        assert!(headers.iter().any(|h| h.ends_with(") ***")), "{headers:?}");
        assert!(
            headers.iter().any(|h| h.ends_with("): unchanged ***")),
            "{headers:?}"
        );

        let (log, plain, watched) = snapshots("tests/run/loop_unroll.wk", "unroll");
        assert_eq!(watched, plain);
        let headers: Vec<&str> = log.lines().filter(|l| l.starts_with("; ***")).collect();
        assert_eq!(headers, ["; *** MIR after unroll ***"]);

        let (log, plain, watched) = snapshots("tests/run/loop_unroll.wk", "all");
        assert_eq!(watched, plain);
        for want in [
            "after inline ***",
            "after mem2reg (fn ",
            "after form-switches ***",
        ] {
            assert!(log.contains(want), "missing `{want}`");
        }
        // The last program snapshot is the result.
        let last = log
            .rsplit("; *** MIR after form-switches ***\n")
            .next()
            .unwrap();
        assert_eq!(last, plain);
    }

    #[test]
    fn printer_copies_the_interner_symbol_for_symbol() {
        let mut interner = Interner::new();
        let names = ["main", "f$[f32; 4]", "wukong$rodata$0", "main"];
        let syms: Vec<Symbol> = names.iter().map(|n| interner.intern(n)).collect();
        let printer = Printer::new("all", &interner, Box::new(|_| {}));
        for (sym, name) in syms.iter().zip(names) {
            assert_eq!(printer.interner.resolve(*sym), name);
        }
        assert!(printer.wants("dce") && printer.wants("inline"));
        assert!(!Printer::new("dce", &interner, Box::new(|_| {})).wants("cse"));
    }
}
