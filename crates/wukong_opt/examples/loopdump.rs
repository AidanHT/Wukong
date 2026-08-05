//! `loopdump` — print `wukong_opt::loop_info`'s view of a `.wk` program.
//!
//! ```text
//! cargo run -q -p wukong_opt --example loopdump -- tests/run/saxpy.wk [-O<n>] [fn-name-filter]
//! ```
//!
//! A development tool for the loop analysis, not part of the compiler. It compiles a **single**
//! source file (no `import` resolution — that lives in `wukong_driver`), optimizes it at the given
//! level (`-O2` by default, the level the vectorizer will run at), and dumps every natural loop it
//! found with its induction variables, trip count, memory accesses and dependence verdict.
//!
//! The same dump is available inside a normal `wukongc` run by setting `WUKONG_DUMP_LOOPS=1`; this
//! binary exists because it can resolve function *names*, which `wukong_opt::optimize` cannot (the
//! interner does not reach it).

use wukong_span::{Interner, SourceId};

fn main() {
    let mut path: Option<String> = None;
    let mut filter: Option<String> = None;
    let mut level: u8 = 2;
    for arg in std::env::args().skip(1) {
        if let Some(n) = arg.strip_prefix("-O") {
            match n.parse::<u8>() {
                Ok(v) => level = v,
                Err(_) => {
                    eprintln!("loopdump: bad optimization level `{arg}`");
                    std::process::exit(2);
                }
            }
        } else if path.is_none() {
            path = Some(arg);
        } else {
            filter = Some(arg);
        }
    }
    let Some(path) = path else {
        eprintln!("usage: loopdump <file.wk> [-O<n>] [fn-name-filter]");
        std::process::exit(2);
    };

    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("loopdump: {path}: {e}");
            std::process::exit(3);
        }
    };

    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        eprintln!("loopdump: parse errors: {pd:?}");
        std::process::exit(1);
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        eprintln!("loopdump: sema errors: {sd:?}");
        std::process::exit(1);
    }
    let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    wukong_opt::optimize(&mut program, level);

    for f in &program.funcs {
        let name = interner.resolve(f.name);
        if filter.as_deref().is_some_and(|want| !name.contains(want)) {
            continue;
        }
        print!("{}", wukong_opt::loop_info::dump_function_loops(f, name));
    }
}
