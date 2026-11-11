//! `wukongc --doctest <file.wk>`: compile and run the code examples in a file's doc comments.
//!
//! # What an example is
//!
//! A fenced code block (```` ``` ```` or `~~~`) inside a `///` or `//!` doc comment whose info
//! string names the language — `wk` or `wukong` — among its comma- or space-separated words. Any
//! other fence (untagged, `text`, `sh`, …) is prose and is not run. The other words are tags:
//!
//! * `ignore` — not compiled; reported as ignored.
//! * `no_run` — must compile; not run.
//! * `compile_fail` — must **fail** to compile, optionally with given codes (`compile_fail,E0503`:
//!   one of the errors must carry one of them). A run-time failure does not count: it is not run.
//! * none of these — must compile and run, and exit 0 (an `assert` that fails, a bounds check, a
//!   non-zero `return` from `main` or an `exit(n)` all fail it).
//!
//! An unknown tag fails the example, so `compile-fail` is not silently a run.
//!
//! # What an example compiles as
//!
//! A program made of **the documented file's items** (everything but its own `fn main`), the
//! example's items, and a `main`: the example's own `fn main` when it declares one (it then takes
//! no other statements), otherwise `fn main() -> i32 { <the example's statements>; return 0; }`,
//! rustdoc-style. So an example on `fn add` simply calls `add(1, 2)`, private items included, and
//! its imports resolve as the file's do (relative to the file's directory). The example's own
//! items may come before, between or after its statements, which run in order.
//!
//! The example is its own source file in the compile, but one laid out over the documented file's
//! lines: each code line sits at its own line and column (the `/// ` prefix is blanked), so a
//! diagnostic in an example points at the very line and column of the doc comment it came from.
//!
//! The front end, lowering, optimizer and backend are the ones `wukongc --run` uses, at the given
//! `-O`, `--fp`, `--target-cpu` and `--backend` (interp or native). Lints are not run on examples.
//! An example's output is captured and shown only when it fails.
//!
//! # Report
//!
//! On stdout, `cargo test`-style: one `doctest <file>:<line> (<what it documents>) ... ok` line per
//! example (`FAILED`, `ignored`), the failures' names, then
//! `doctest result: ok. P passed; F failed; I ignored`. The exit status is `0` when nothing failed
//! and `1` otherwise. A file that does not compile on its own (front end through lowering, its own
//! `fn main` included) is reported like any compile and runs nothing: a `compile_fail` example
//! must fail for its own error, not the file's.

use wukong_ast::{
    Block, Docs, Expr, ExprKind, Item, ItemKind, Module, Stmt, StmtKind, VariantData,
};
use wukong_diag::{Diagnostic, DiagnosticSink, Renderer};
use wukong_span::{Interner, SourceId, SourceMap, Span};

use crate::{
    emit_diags, emit_driver_error, emit_summary, eprint_line, exit, loader, lower_options_for,
    native_failure_diag, optimize_options_for, prune, reads_stdin, render_lints, source_name,
    verify_or_ice, BackendKind, Options,
};

/// The statement the wrapper `main` is parsed with, replaced by the example's statements.
const BODY: &str = "__wukong_doctest_body";

/// How an example is checked.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Mode {
    Run,
    NoRun,
    /// Must fail to compile; with codes, with one of them.
    CompileFail(Vec<String>),
    Ignore,
}

/// One example, as found in the doc comments.
struct Example {
    /// The 1-based line of its opening fence.
    line: usize,
    /// What the doc comment documents: `fn add`, `field Point.x`, `the file`.
    owner: String,
    /// The code lines, each with the source offset its first byte sits at.
    code: Vec<(String, u32)>,
    mode: Mode,
    /// A tag the runner does not know; the example fails with it.
    bad_tag: Option<String>,
}

/// How one example went.
enum Outcome {
    Passed,
    Ignored,
    Failed,
}

/// Run every example in `opts.input`'s doc comments. Returns the process exit code.
pub fn run(opts: &Options) -> i32 {
    let src = match &opts.source {
        Some(s) => Ok(s.clone()),
        None if reads_stdin(opts) => {
            let mut s = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut s).map(|_| s)
        }
        None => std::fs::read_to_string(&opts.input),
    };
    let src = match src {
        Ok(s) => s,
        Err(e) => {
            emit_driver_error(
                opts,
                format!("could not read `{}`: {}", source_name(opts), e),
            );
            return exit::IO_ERROR;
        }
    };
    let name = source_name(opts);
    let renderer = Renderer::new(opts.color);

    // The file itself must parse: its examples document what it says.
    let mut sm = SourceMap::new();
    let id = sm.add(name.clone(), src.clone());
    let mut interner = Interner::new();
    let (tokens, lex_diags) = wukong_lexer::tokenize(&src, id);
    let parsed = wukong_parser::parse_module_tokens_with_lints(&tokens, &src, &mut interner, 0);
    let errors: Vec<Diagnostic> = lex_diags
        .into_iter()
        .chain(parsed.diags)
        .filter(|d| d.is_error())
        .collect();
    if !errors.is_empty() {
        let n = emit_diags(&errors, opts, &renderer, &sm);
        let denied = render_lints(
            parsed.lints,
            &parsed.module,
            &interner,
            opts,
            &renderer,
            &sm,
        );
        emit_summary(n + denied, opts, &renderer);
        return exit::COMPILE_ERROR;
    }

    // ... and compile on its own, or no example is checked (see `check_file`).
    if let Err((errors, sm)) = check_file(opts, &src, &name) {
        let n = emit_diags(&errors, opts, &renderer, &sm);
        emit_summary(n, opts, &renderer);
        return exit::COMPILE_ERROR;
    }

    let examples = collect(&parsed.module, &interner, &src);
    // A program an example runs reads end of input, not the terminal or the pipe it was given.
    wukong_runtime::process::detach_stdin();
    wukong_interp::set_program_args(std::iter::once(name.clone()));

    let mut out = String::new();
    out.push_str(&format!(
        "running {} doctest{} in {name}\n",
        examples.len(),
        if examples.len() == 1 { "" } else { "s" }
    ));
    print_out(&out);
    let (mut passed, mut failed, mut ignored) = (0usize, 0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    for ex in &examples {
        let label = format!("{name}:{} ({})", ex.line, ex.owner);
        let outcome = check(opts, &src, &name, ex, &label);
        let word = match outcome {
            Outcome::Passed => {
                passed += 1;
                "ok"
            }
            Outcome::Ignored => {
                ignored += 1;
                "ignored"
            }
            Outcome::Failed => {
                failed += 1;
                failures.push(label.clone());
                "FAILED"
            }
        };
        print_out(&format!("doctest {label} ... {word}\n"));
    }
    let mut tail = String::new();
    if !failures.is_empty() {
        tail.push_str("\nfailures:\n");
        for f in &failures {
            tail.push_str(&format!("    {f}\n"));
        }
    }
    tail.push_str(&format!(
        "\ndoctest result: {}. {passed} passed; {failed} failed; {ignored} ignored\n",
        if failed == 0 { "ok" } else { "FAILED" }
    ));
    print_out(&tail);
    if failed == 0 {
        exit::OK
    } else {
        exit::COMPILE_ERROR
    }
}

/// Every example in the module's doc comments, in source order.
fn collect(m: &Module, interner: &Interner, src: &str) -> Vec<Example> {
    let name = |s: wukong_span::Symbol| interner.resolve(s).to_string();
    let mut docs: Vec<(String, &Docs)> = vec![("the file".to_string(), &m.docs)];
    for it in &m.items {
        let owner = match &it.kind {
            ItemKind::Fn(f) => format!("fn {}", name(f.name.sym)),
            ItemKind::Struct(d) => {
                for f in &d.fields {
                    let o = format!("field {}.{}", name(d.name.sym), name(f.name.sym));
                    docs.push((o, &f.docs));
                }
                format!("struct {}", name(d.name.sym))
            }
            ItemKind::Enum(d) => {
                for v in &d.variants {
                    let vn = format!("{}::{}", name(d.name.sym), name(v.name.sym));
                    docs.push((format!("variant {vn}"), &v.docs));
                    if let VariantData::Struct(fs) = &v.data {
                        for f in fs {
                            docs.push((format!("field {vn}.{}", name(f.name.sym)), &f.docs));
                        }
                    }
                }
                format!("enum {}", name(d.name.sym))
            }
            ItemKind::Const(c) => format!("const {}", name(c.name.sym)),
            ItemKind::TypeAlias(a) => format!("type {}", name(a.name.sym)),
            ItemKind::Trait(t) => {
                for (f, d) in t.methods.iter().zip(&t.method_docs) {
                    let o = format!("fn {}.{}", name(t.name.sym), name(f.name.sym));
                    docs.push((o, d));
                }
                format!("trait {}", name(t.name.sym))
            }
            ItemKind::Extern(x) => {
                for (f, d) in x.items.iter().zip(&x.item_docs) {
                    docs.push((format!("fn {}", name(f.name.sym)), d));
                }
                format!("extern {}", interner.resolve(x.abi))
            }
            ItemKind::Import(im) => {
                let segs: Vec<String> = im.path.segments.iter().map(|s| name(s.sym)).collect();
                format!("import {}", segs.join("."))
            }
            ItemKind::Impl(d) => {
                let segs: Vec<String> = d.self_ty.segments.iter().map(|s| name(s.sym)).collect();
                format!("impl {}", segs.join("."))
            }
        };
        docs.push((owner, &it.docs));
    }
    let mut out = Vec::new();
    for (owner, d) in docs {
        examples_in(d, &owner, src, &mut out);
    }
    out.sort_by_key(|e| e.line);
    out
}

/// The 1-based line `offset` is on.
fn line_of(src: &str, offset: u32) -> usize {
    src.as_bytes()[..(offset as usize).min(src.len())]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + 1
}

/// `line` without its indentation: leading ASCII spaces and tabs only, as in `Docs::unindented`
/// and CommonMark. A Unicode space (U+00A0, U+3000) is text, and stripping by byte count across
/// one would cut it in half.
fn unindent(line: &str) -> &str {
    line.trim_start_matches([' ', '\t'])
}

/// The examples in one doc comment: its fenced blocks whose info string names the language.
fn examples_in(docs: &Docs, owner: &str, src: &str, out: &mut Vec<Example>) {
    let lines = docs.unindented();
    let mut i = 0;
    while i < lines.len() {
        let (text, offset) = lines[i];
        let trimmed = unindent(text);
        let Some((fence, n)) = fence_open(trimmed) else {
            i += 1;
            continue;
        };
        let indent = text.len() - trimmed.len();
        let info = trimmed[n..].trim();
        let mut code = Vec::new();
        let mut j = i + 1;
        while j < lines.len() {
            let (t, o) = lines[j];
            let tt = unindent(t);
            if is_fence_close(tt, fence, n) {
                break;
            }
            // A fence indented by k spaces strips up to k from each line of its block. Both counts
            // are ASCII bytes, so the cut is always on a character boundary.
            let strip = indent.min(t.len() - tt.len());
            code.push((t[strip..].to_string(), o + strip as u32));
            j += 1;
        }
        if let Some((mode, bad_tag)) = classify(info) {
            out.push(Example {
                line: line_of(src, offset),
                owner: owner.to_string(),
                code,
                mode,
                bad_tag,
            });
        }
        i = j + 1;
    }
}

/// An opening fence: three or more backticks or tildes. Returns the character and the run length.
fn fence_open(line: &str) -> Option<(char, usize)> {
    let c = line.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let n = line.chars().take_while(|x| *x == c).count();
    (n >= 3).then_some((c, n))
}

/// Whether `line` closes a fence of `n` `c`s: at least that many, then nothing but whitespace.
fn is_fence_close(line: &str, c: char, n: usize) -> bool {
    let k = line.chars().take_while(|x| *x == c).count();
    k >= n && line[k * c.len_utf8()..].trim().is_empty()
}

/// The mode of a fence with this info string, or `None` when it is not a Wukong example. The
/// second value is the first tag the runner does not know.
fn classify(info: &str) -> Option<(Mode, Option<String>)> {
    let words: Vec<&str> = info
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|w| !w.is_empty())
        .collect();
    if !words.iter().any(|w| *w == "wk" || *w == "wukong") {
        return None;
    }
    let (mut ignore, mut no_run, mut fail) = (false, false, false);
    let mut codes = Vec::new();
    let mut bad = None;
    for w in words {
        match w {
            "wk" | "wukong" => {}
            "ignore" => ignore = true,
            "no_run" => no_run = true,
            "compile_fail" => fail = true,
            _ if is_code(w) => codes.push(w.to_string()),
            _ => {
                bad.get_or_insert_with(|| w.to_string());
            }
        }
    }
    if !codes.is_empty() && !fail && bad.is_none() {
        bad = Some(format!(
            "{} (an error code goes with `compile_fail`)",
            codes[0]
        ));
    }
    let mode = if ignore {
        Mode::Ignore
    } else if fail {
        Mode::CompileFail(codes)
    } else if no_run {
        Mode::NoRun
    } else {
        Mode::Run
    };
    Some((mode, bad))
}

/// A diagnostic code: `E`, `W` or `C` and four digits.
fn is_code(w: &str) -> bool {
    let b = w.as_bytes();
    b.len() == 5 && matches!(b[0], b'E' | b'W' | b'C') && b[1..].iter().all(u8::is_ascii_digit)
}

/// Check one example, printing why it failed (on stderr) when it does.
fn check(opts: &Options, src: &str, name: &str, ex: &Example, label: &str) -> Outcome {
    if let Some(tag) = &ex.bad_tag {
        emit_driver_error(
            opts,
            format!(
                "doctest {label}: unknown tag `{tag}` (the tags are `ignore`, `no_run` and \
                 `compile_fail[,<code>..]`)"
            ),
        );
        return Outcome::Failed;
    }
    if ex.mode == Mode::Ignore {
        return Outcome::Ignored;
    }
    let renderer = Renderer::new(opts.color);
    let built = match compile(opts, src, name, ex) {
        Ok(b) => b,
        Err((errors, sm)) => {
            return match &ex.mode {
                Mode::CompileFail(codes) => {
                    let hit = codes.is_empty()
                        || errors
                            .iter()
                            .any(|d| d.code.is_some_and(|c| codes.iter().any(|w| w == c)));
                    if hit {
                        Outcome::Passed
                    } else {
                        emit_driver_error(
                            opts,
                            format!(
                                "doctest {label}: failed to compile, but not with {}:",
                                codes.join(" or ")
                            ),
                        );
                        emit_diags(&errors, opts, &renderer, &sm);
                        Outcome::Failed
                    }
                }
                _ => {
                    emit_driver_error(opts, format!("doctest {label}: failed to compile:"));
                    let n = emit_diags(&errors, opts, &renderer, &sm);
                    emit_summary(n, opts, &renderer);
                    Outcome::Failed
                }
            };
        }
    };
    match &ex.mode {
        Mode::CompileFail(_) => {
            emit_driver_error(
                opts,
                format!("doctest {label}: compiled, but it is marked `compile_fail`"),
            );
            Outcome::Failed
        }
        Mode::NoRun | Mode::Ignore => Outcome::Passed,
        Mode::Run => execute(opts, built, label),
    }
}

/// A compiled example, ready to run.
struct Built {
    program: wukong_mir::Program,
    interner: Interner,
    sm: SourceMap,
}

/// Build the example's program through MIR optimization. `Err` carries the errors, with the source
/// map they point into.
fn compile(
    opts: &Options,
    src: &str,
    name: &str,
    ex: &Example,
) -> Result<Built, (Vec<Diagnostic>, SourceMap)> {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();

    // The documented file's items, its own `main` left out.
    let file = sm.add(name.to_string(), src.to_string());
    let (tokens, _) = wukong_lexer::tokenize(sm.source(file), file);
    let parsed =
        wukong_parser::parse_module_tokens_with_lints(&tokens, sm.source(file), &mut interner, 0);
    let mut next_node = parsed.next_node;
    let main_sym = interner.intern("main");
    let is_main = |it: &Item| matches!(&it.kind, ItemKind::Fn(f) if f.name.sym == main_sym && f.body.is_some());
    let mut items: Vec<Item> = parsed
        .module
        .items
        .into_iter()
        .filter(|it| !is_main(it))
        .collect();

    // The example, laid out over the file's lines and columns.
    let text = layout(src, &ex.code);
    let example = sm.add(name.to_string(), text);
    let (tokens, lex_diags) = wukong_lexer::tokenize(sm.source(example), example);
    let out =
        wukong_parser::parse_doctest_tokens(&tokens, sm.source(example), &mut interner, next_node);
    next_node = out.next_node;
    let errors: Vec<Diagnostic> = lex_diags
        .into_iter()
        .chain(out.diags)
        .filter(|d| d.is_error())
        .collect();
    if !errors.is_empty() {
        return Err((errors, sm));
    }
    let own_main = out.items.iter().find(|it| is_main(it)).map(|it| it.span);
    let Block { stmts, tail, .. } = out.block;
    let has_stmts = !stmts.is_empty() || tail.is_some();
    items.extend(out.items);
    match own_main {
        Some(span) if has_stmts => {
            let d = Diagnostic::error(
                "a doc example with its own `fn main` takes no statements outside it",
            )
            .primary(span, "this `main` is the program's entry point")
            .help("move the statements into `main`, or drop `fn main` to have them wrapped in one");
            return Err((vec![d], sm));
        }
        Some(_) => {}
        None => {
            let wrapper = sm.add(
                "<doctest>",
                format!("fn main() -> i32 {{\n    {BODY};\n    return 0;\n}}\n"),
            );
            let (tokens, _) = wukong_lexer::tokenize(sm.source(wrapper), wrapper);
            let w = wukong_parser::parse_module_tokens_with_lints(
                &tokens,
                sm.source(wrapper),
                &mut interner,
                next_node,
            );
            next_node = w.next_node;
            let mut body: Vec<Stmt> = stmts;
            if let Some(t) = tail {
                let span = t.span;
                body.push(Stmt {
                    id: wukong_ast::NodeId(next_node),
                    attrs: Vec::new(),
                    kind: StmtKind::Expr(*t),
                    span,
                });
                next_node += 1;
            }
            let body_sym = interner.intern(BODY);
            for mut item in w.module.items {
                if let ItemKind::Fn(f) = &mut item.kind {
                    if let Some(b) = &mut f.body {
                        splice(b, body_sym, &mut body);
                    }
                }
                items.push(item);
            }
        }
    }
    let span = Span::new(SourceId(example.0), 0, 0);
    let module = Module {
        name: parsed.module.name,
        docs: Docs::default(),
        items,
        span,
    };
    let mut program = match lower(opts, module, &mut sm, &mut interner, &mut next_node) {
        Ok(p) => p,
        Err(errors) => return Err((errors, sm)),
    };
    let main = interner.intern("main");
    prune::retain_reachable(&mut program, &[main]);
    wukong_opt::optimize_preserving_with(
        &mut program,
        opts.opt_level,
        &[],
        optimize_options_for(opts, &lower_options_for(opts)),
    );
    Ok(Built {
        program,
        interner,
        sm,
    })
}

/// The documented file on its own, through lowering, as `wukongc --run` compiles it (its `fn main`
/// included). Its examples are only checked when it compiles: otherwise every `compile_fail`
/// example would pass for the file's error rather than its own. (rustdoc does not run the doctests
/// of a crate that does not build either.) `Err` carries the errors, with their source map.
fn check_file(opts: &Options, src: &str, name: &str) -> Result<(), (Vec<Diagnostic>, SourceMap)> {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let file = sm.add(name.to_string(), src.to_string());
    let (tokens, _) = wukong_lexer::tokenize(sm.source(file), file);
    let parsed =
        wukong_parser::parse_module_tokens_with_lints(&tokens, sm.source(file), &mut interner, 0);
    let mut next_node = parsed.next_node;
    match lower(opts, parsed.module, &mut sm, &mut interner, &mut next_node) {
        Ok(_) => Ok(()),
        Err(errors) => Err((errors, sm)),
    }
}

/// The front end and lowering, as `compile` runs them for a `--run`: imports (relative to the
/// documented file), type aliases, sema, lowering. `Err` carries the errors.
fn lower(
    opts: &Options,
    mut module: Module,
    sm: &mut SourceMap,
    interner: &mut Interner,
    next_node: &mut u32,
) -> Result<wukong_mir::Program, Vec<Diagnostic>> {
    let mut sink = DiagnosticSink::new();
    let mut lints = Vec::new();
    let _unused_imports = loader::load_imports(
        &mut module,
        &opts.input,
        sm,
        interner,
        &mut sink,
        next_node,
        &mut lints,
    );
    if !sink.has_errors() {
        loader::expand_type_aliases(&mut module, interner, &mut sink, next_node);
    }
    if sink.has_errors() {
        return Err(sink.take().into_iter().filter(|d| d.is_error()).collect());
    }
    let (sema, sema_diags) = wukong_sema::check(&module, interner);
    let errors: Vec<Diagnostic> = sema_diags.into_iter().filter(|d| d.is_error()).collect();
    if !errors.is_empty() {
        return Err(errors);
    }

    let sm: &SourceMap = sm;
    let locate = |span: Span| -> String {
        let id = span.source;
        if span.is_dummy()
            || id.0 as usize >= sm.file_count()
            || span.lo as usize > sm.source(id).len()
        {
            return String::new();
        }
        let loc = sm.span_location(span);
        format!("{}:{}:{}", sm.name(id), loc.line, loc.col)
    };
    let (program, lower_diags) = wukong_mir_build::lower_program_located(
        &module,
        &sema,
        interner,
        lower_options_for(opts),
        &locate,
    );
    let errors: Vec<Diagnostic> = lower_diags.into_iter().filter(|d| d.is_error()).collect();
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(program)
}

/// The example's code as a source text laid over the documented file: each code line at its own
/// line and column, everything else blank. A diagnostic in the example therefore names the line and
/// column of the doc comment it is written in.
fn layout(src: &str, code: &[(String, u32)]) -> String {
    let mut out = String::new();
    let mut line = 1usize;
    for (text, offset) in code {
        let at = line_of(src, *offset);
        while line < at {
            out.push('\n');
            line += 1;
        }
        let line_start = src[..*offset as usize].rfind('\n').map_or(0, |i| i + 1);
        let col = src[line_start..*offset as usize].chars().count();
        out.extend(std::iter::repeat_n(' ', col));
        out.push_str(text);
        out.push('\n');
        line += 1;
    }
    out
}

/// Replace the wrapper's `BODY` statement with the example's statements.
fn splice(body: &mut Block, sym: wukong_span::Symbol, stmts: &mut Vec<Stmt>) {
    let is_body =
        |e: &Expr| matches!(&e.kind, ExprKind::Path(p) if p.is_single() && p.first().sym == sym);
    let mut out = Vec::with_capacity(body.stmts.len() + stmts.len());
    for stmt in std::mem::take(&mut body.stmts) {
        match &stmt.kind {
            StmtKind::Expr(e) if is_body(e) => out.append(stmts),
            _ => out.push(stmt),
        }
    }
    body.stmts = out;
}

/// Run a compiled example: it passes when it runs to its end and exits 0. On a failure its output
/// is shown, then why.
fn execute(opts: &Options, built: Built, label: &str) -> Outcome {
    let Built {
        program,
        mut interner,
        sm,
    } = built;
    if verify_or_ice(&program) {
        emit_driver_error(opts, format!("doctest {label}: internal compiler error"));
        return Outcome::Failed;
    }
    let renderer = Renderer::new(opts.color);
    let main = interner.intern("main");
    if opts.backend != BackendKind::Native {
        let refused = wukong_interp::unsupported_externs(&program, &interner);
        if !refused.is_empty() {
            for msg in refused {
                emit_driver_error(opts, format!("doctest {label}: {msg}"));
            }
            return Outcome::Failed;
        }
    }
    wukong_runtime::fmt::begin_stderr_capture();
    let (result, stdout) = match opts.backend {
        BackendKind::Native => {
            let backend = wukong_codegen_cranelift::BackendOpt::for_level(opts.opt_level);
            match wukong_codegen_cranelift::jit_compile_for_with(
                &program,
                main,
                &interner,
                opts.target_cpu,
                backend,
            ) {
                Ok(compiled) => compiled.run_capturing(),
                Err(failure) => {
                    let _ = wukong_runtime::fmt::end_stderr_capture();
                    emit_driver_error(opts, format!("doctest {label}: failed to compile:"));
                    let d = native_failure_diag(&failure, &opts.input, Some(&sm));
                    emit_diags(&[d], opts, &renderer, &sm);
                    return Outcome::Failed;
                }
            }
        }
        _ => wukong_interp::run_capturing(&program, main, &interner),
    };
    let stderr = wukong_runtime::fmt::end_stderr_capture();
    let why = match result {
        Ok(0) => return Outcome::Passed,
        Ok(code) => format!("it exited with status {code}"),
        Err(e) => e,
    };
    show_output(&stdout, &stderr);
    emit_driver_error(opts, format!("doctest {label}: {why}"));
    Outcome::Failed
}

/// A failed example's captured output, on stderr.
fn show_output(stdout: &[u8], stderr: &[u8]) {
    for (what, bytes) in [("stdout", stdout), ("stderr", stderr)] {
        if bytes.is_empty() {
            continue;
        }
        eprint_line(&format!("---- {what} ----"));
        eprint_line(String::from_utf8_lossy(bytes).trim_end());
    }
}

/// Write the report to stdout (a closed pipe is not an error; see `print_artifact`).
fn print_out(text: &str) {
    use std::io::Write;
    let mut out = std::io::stdout();
    let _ = out.write_all(text.as_bytes());
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_strings_pick_the_examples_and_their_modes() {
        assert_eq!(classify(""), None);
        assert_eq!(classify("text"), None);
        assert_eq!(classify("rust"), None);
        assert_eq!(classify("wk"), Some((Mode::Run, None)));
        assert_eq!(classify("wukong"), Some((Mode::Run, None)));
        assert_eq!(classify("wk,ignore"), Some((Mode::Ignore, None)));
        assert_eq!(classify("wk no_run"), Some((Mode::NoRun, None)));
        assert_eq!(
            classify("wk, compile_fail"),
            Some((Mode::CompileFail(vec![]), None))
        );
        assert_eq!(
            classify("wk,compile_fail,E0503,E0401"),
            Some((
                Mode::CompileFail(vec!["E0503".into(), "E0401".into()]),
                None
            ))
        );
        assert_eq!(
            classify("wk,compile-fail"),
            Some((Mode::Run, Some("compile-fail".into())))
        );
        assert!(classify("wk,E0503").unwrap().1.is_some());
    }

    #[test]
    fn fences_open_and_close_like_commonmark() {
        assert_eq!(fence_open("```wk"), Some(('`', 3)));
        assert_eq!(fence_open("~~~~"), Some(('~', 4)));
        assert_eq!(fence_open("``"), None);
        assert!(is_fence_close("```", '`', 3));
        assert!(is_fence_close("````  ", '`', 3));
        assert!(!is_fence_close("``", '`', 3));
        assert!(!is_fence_close("```wk", '`', 3));
        assert!(!is_fence_close("~~~", '`', 3));
    }

    /// The laid-out example keeps each code line on its line and column of the file.
    #[test]
    fn the_example_keeps_its_lines_and_columns() {
        let src = "/// Doc.\n/// ```wk\n/// let x = 1;\n///     f(x);\n/// ```\nfn f(x: i32) {}\n";
        let mut i = Interner::new();
        let (m, d) = wukong_parser::parse_module(src, SourceId(0), &mut i);
        assert!(d.is_empty(), "{d:?}");
        let ex = collect(&m, &i, src);
        assert_eq!(ex.len(), 1);
        assert_eq!(ex[0].line, 2);
        assert_eq!(ex[0].owner, "fn f");
        assert_eq!(
            layout(src, &ex[0].code),
            "\n\n    let x = 1;\n        f(x);\n"
        );
    }

    /// Indentation is ASCII spaces and tabs, as in `Docs::unindented` and CommonMark: a code line
    /// that starts with a multi-byte space (U+00A0) under a fence indented by one space keeps it as
    /// code — cutting one byte into it panicked the whole run — and a fence led by one is prose.
    #[test]
    fn only_ascii_whitespace_indents_an_example() {
        let src = "/// Example:\n///  ```wk\n/// \u{a0}let x = 1;\n///  ```\nfn f() {}\n";
        let mut i = Interner::new();
        let (m, d) = wukong_parser::parse_module(src, SourceId(0), &mut i);
        assert!(d.is_empty(), "{d:?}");
        let ex = collect(&m, &i, src);
        assert_eq!(ex.len(), 1);
        assert_eq!(ex[0].code.len(), 1);
        assert_eq!(ex[0].code[0].0, "\u{a0}let x = 1;");
        // The code line's offset is that of its first byte, the U+00A0 (`/// ` is 4 bytes in).
        let at = src.find('\u{a0}').unwrap() as u32;
        assert_eq!(ex[0].code[0].1, at);
        assert_eq!(layout(src, &ex[0].code), "\n\n    \u{a0}let x = 1;\n");

        let src = "/// \u{a0}```wk\n/// let x = 1;\n/// ```\nfn f() {}\n";
        let (m, d) = wukong_parser::parse_module(src, SourceId(0), &mut i);
        assert!(d.is_empty(), "{d:?}");
        assert!(collect(&m, &i, src).is_empty());
    }
}
