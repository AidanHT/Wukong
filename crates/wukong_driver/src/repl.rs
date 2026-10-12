//! `wukongc --repl`: an interactive session on standard input.
//!
//! # What a session is
//!
//! A session keeps the inputs it accepted, as source text, and compiles a whole program from them
//! for every new input — a REPL over a whole-program compiler, through the same front end,
//! lowering, optimizer and backends as `wukongc --run`. An input is one or more lines: it is
//! complete when no `(`, `[` or `{` is left open (and no block comment is), so a function body
//! can span lines. Each input is one of two kinds, decided by its first token after
//! any `@attributes`:
//!
//! * **items** — `fn`, `struct`, `enum`, `const`, `import`, `extern`, `trait`, `impl`, `type`,
//!   `pub` — are kept for every later input. An item named like an earlier one *replaces* it; an
//!   `impl` block replaces the earlier `impl` of the same type (and trait), methods and all.
//! * **statements** — anything else — run as the body of a synthesized `main`. An input that ends
//!   in an expression with no `;` prints the expression's value, `{:?}`-style (`3`, `2.0`,
//!   `"hi"`, `[1, 2]`); a `()` value prints nothing, and a value with no printed form (a pointer, a
//!   tensor, a function) prints its type in angle brackets.
//!
//! # Bindings persist by replay
//!
//! `let` bindings — and every later assignment to them — persist because each run **replays** the
//! statements of every accepted statement input, in order, ahead of the new input, in the same
//! `main`: a later `let` of the same name shadows an earlier one, as in any block. The replay's
//! standard output and standard error are captured and dropped, so only the new input's output
//! shows. What a replay cannot undo it repeats, and that is the documented cost: it re-executes the
//! earlier inputs (one that took a second costs a second on every later input), re-writes the files
//! they wrote, and reads the clock anew. A program in a session reads end of input from standard
//! input, which the session itself is reading ([`wukong_runtime::process::detach_stdin`]).
//!
//! An input is **accepted** — kept for later inputs — only when it compiles and, for statements,
//! runs to its end. One that fails to compile, fails at run time (a failed `assert`, an index out
//! of bounds) or ends the program early (`exit`, a `return` inside a block) leaves the session as it
//! was. An item input is compiled together with the replay, and the replay is run, so a
//! redefinition that breaks an earlier input is refused rather than poisoning every later one. A
//! top-level `return` or `defer` is refused outright: the synthesized `main` is not the input's to
//! end, and its end is shared by every replayed input.
//!
//! # Diagnostics
//!
//! Every input is its own source file, `<repl:N>` (`N` counts the inputs a session compiled, from
//! 1), so a diagnostic points into the input that holds the problem, with lines counted from that
//! input's first line. Errors show wherever they are (an item that breaks an earlier input shows
//! the break in that input); warnings show for the new input only. Three lints are off in a
//! session — `unused_variables`, `unused_mut` and `unused_imports` judge one input, while a binding
//! or an import is made for the inputs after it — and `dead_code` has no roots to measure against,
//! since every function is kept for later.

use std::collections::HashSet;
use std::io::{BufRead, IsTerminal, Write};

use wukong_ast::{Block, Expr, ExprKind, Item, ItemKind, Module, NodeId, Stmt, StmtKind};
use wukong_diag::{Diagnostic, DiagnosticSink, Renderer, Severity};
use wukong_lexer::TokenKind;
use wukong_sema::SemaResult;
use wukong_span::{Interner, SourceId, SourceMap, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{
    emit_diag, emit_diags, emit_driver_error, eprint_line, loader, lower_options_for,
    native_failure_diag, prune, verify_or_ice, BackendKind, ErrorFormat, Options,
};

/// Written to the run's stdout and stderr between the replay and the new input: everything before
/// it belongs to earlier inputs. ASCII record separators around a fixed word, so no program prints
/// it by accident.
const MARK: &str = "\x1ewukong-repl-input\x1e";
/// Written to the run's stdout after the new input: present only when the input ran to its end.
const END: &str = "\x1ewukong-repl-end\x1e";

/// The placeholder statements and argument the wrapper `main` is parsed with; each is replaced by
/// the parsed inputs before anything checks the program, so no later stage sees one.
const REPLAY: &str = "__wukong_repl_replay";
const INPUT: &str = "__wukong_repl_input";
const VALUE: &str = "__wukong_repl_value";

const HELP: &str = "\
Wukong REPL. Type items (fn, struct, enum, const, import, extern, trait, impl, type) to keep
them; type statements to run them. An input that ends in an expression without `;` prints its
value. An input continues on the next line while a bracket or a block comment is open.

Earlier `let` bindings persist: each input runs after a silent replay of every earlier statement
input, so their side effects (work done, files written) repeat, while their output is hidden.
An input that fails to compile or to run is not kept. A new item replaces an earlier item of
the same name.

Commands:
  :help          this text
  :quit          leave the session (also :q, :exit, or end of input)
  :reset         forget every item and binding
  :items         print the items the session keeps
  :type <expr>   print the type of an expression without running it
";

/// Run a session on standard input until it ends or `:quit`. Returns the process exit code: `0`.
pub fn run(opts: &Options) -> i32 {
    // The session reads its inputs from standard input; a program it runs must not read them.
    wukong_runtime::process::detach_stdin();
    wukong_interp::set_program_args(std::iter::once("<repl>".to_string()));
    let interactive = std::io::stdin().is_terminal();
    if interactive {
        eprint_line(&format!(
            "Wukong {} REPL ({} backend). `:help` for help, `:quit` to leave.",
            env!("CARGO_PKG_VERSION"),
            match opts.backend {
                BackendKind::Native => "native",
                _ => "interp",
            }
        ));
    }
    let mut session = Session::new(opts.clone());
    let stdin = std::io::stdin();
    let mut lines = stdin.lock();
    let mut pending = String::new();
    loop {
        if interactive {
            let prompt = if pending.is_empty() {
                format!("wk[{}]> ", session.next_no)
            } else {
                "   ...> ".to_string()
            };
            let mut err = std::io::stderr();
            let _ = err.write_all(prompt.as_bytes());
            let _ = err.flush();
        }
        let mut line = String::new();
        match lines.read_line(&mut line) {
            Ok(0) => {
                // End of input submits whatever is left, complete or not: an unclosed bracket is
                // then the parser's error to report.
                if !pending.trim().is_empty() {
                    session.submit(&pending);
                }
                break;
            }
            Ok(_) => {}
            Err(e) => {
                emit_driver_error(opts, format!("could not read standard input: {e}"));
                break;
            }
        }
        if !line.ends_with('\n') {
            line.push('\n');
        }
        if pending.is_empty() {
            let command = line.trim();
            if command.is_empty() {
                continue;
            }
            if let Some(command) = command.strip_prefix(':') {
                if !session.command(command) {
                    break;
                }
                continue;
            }
        }
        pending.push_str(&line);
        if needs_more(&pending) {
            continue;
        }
        let input = std::mem::take(&mut pending);
        session.submit(&input);
    }
    crate::exit::OK
}

/// Is `text` an unfinished input: a `(`, `[` or `{` still open, or a block comment running to the
/// end? Decided by the lexer, so a bracket inside a string or a comment is not counted. A string
/// literal cannot span lines, so an unclosed one is complete — and the lexer's error to report.
fn needs_more(text: &str) -> bool {
    let (tokens, diags) = wukong_lexer::tokenize(text, SourceId(0));
    if diags.iter().any(|d| d.code == Some("E0103")) {
        return true;
    }
    let mut depth: i64 = 0;
    for t in &tokens {
        match t.kind {
            TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => depth += 1,
            TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => depth -= 1,
            _ => {}
        }
    }
    depth > 0
}

/// What an input is: items to keep, or statements to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Items,
    Stmts,
}

/// Classify an input by its first token after any `@attr` / `@attr(..)` attributes.
fn classify(text: &str) -> Kind {
    let (tokens, _) = wukong_lexer::tokenize(text, SourceId(0));
    let kind = |i: usize| tokens.get(i).map_or(TokenKind::Eof, |t| t.kind);
    let mut i = 0;
    while kind(i) == TokenKind::At {
        // `@`, the attribute's name, then an optional balanced argument list.
        i += 2;
        if kind(i) == TokenKind::LParen {
            let mut depth = 0i64;
            loop {
                match kind(i) {
                    TokenKind::LParen => depth += 1,
                    TokenKind::RParen => depth -= 1,
                    TokenKind::Eof => break,
                    _ => {}
                }
                i += 1;
                if depth == 0 {
                    break;
                }
            }
        }
    }
    let word = |i: usize| {
        tokens
            .get(i)
            .map_or("", |t| &text[t.span.lo as usize..t.span.hi as usize])
    };
    match kind(i) {
        TokenKind::Fn
        | TokenKind::Struct
        | TokenKind::Enum
        | TokenKind::Const
        | TokenKind::Import
        | TokenKind::Extern
        | TokenKind::Trait
        | TokenKind::Impl
        | TokenKind::Pub => Kind::Items,
        // `type` is contextual: only `type Name` starts an alias.
        TokenKind::Ident if word(i) == "type" && kind(i + 1) == TokenKind::Ident => Kind::Items,
        _ => Kind::Stmts,
    }
}

/// An accepted input: its number (its file is `<repl:N>`), its text and its kind.
struct Entry {
    no: usize,
    text: String,
    kind: Kind,
}

impl Entry {
    fn file_name(&self) -> String {
        format!("<repl:{}>", self.no)
    }
}

/// What the new input's trailing expression (the block's tail) becomes in the program.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tail {
    /// An expression statement: evaluated, its value dropped.
    Stmt,
    /// The argument of `printf("{:?}\n", ..)`: evaluated and printed.
    Print,
    /// Left out: a bare name with no printed form, whose evaluation does nothing (and which the
    /// native backend could not lower as a statement: a function's name).
    Drop,
}

/// One synthesized program: the session's items, the replay, the new input and the wrapper
/// `main`, parsed but not yet checked.
struct Built {
    module: Module,
    sm: SourceMap,
    interner: Interner,
    next_node: u32,
    /// The parser's lints, of every file (only the new input's are ever shown).
    lints: Vec<Diagnostic>,
    /// The new input's lexer and parser diagnostics (accepted inputs parse clean).
    diags: Vec<Diagnostic>,
    /// The new input's file.
    input_file: Option<SourceId>,
    /// The new input's tail expression, when it ends in one.
    tail: Option<NodeId>,
    /// The tail's span, and whether it is a bare name.
    tail_span: Option<Span>,
    tail_is_name: bool,
    /// The new input's top-level `return` / `defer`, which a session refuses.
    refused: Option<(Span, &'static str)>,
    /// Item names the new input defines that the session cannot take (`main`).
    reserved: Option<Span>,
}

/// A program the front end accepted: what lowering needs, plus the new input's file and tail.
struct Front {
    module: Module,
    sema: SemaResult,
    sm: SourceMap,
    interner: Interner,
    lints: Vec<Diagnostic>,
    input_file: Option<SourceId>,
    tail: Option<NodeId>,
    tail_span: Option<Span>,
}

/// How a run went, by the markers in its output.
enum Ran {
    /// The input ran to its end.
    Completed,
    /// The input stopped: a run-time failure, or the program ended early.
    Stopped,
    /// The replay of earlier inputs did not reach the new input.
    ReplayFailed,
}

/// One item of an input, with the items that belong to it (an `impl` block's methods), and the key
/// a later definition replaces it by. `None`: never replaced (an `extern` block).
struct Group {
    key: Option<String>,
    items: Vec<Item>,
}

/// A REPL session: the accepted inputs and the options every input compiles with.
struct Session {
    opts: Options,
    entries: Vec<Entry>,
    /// The number the next compiled input gets.
    next_no: usize,
}

impl Session {
    fn new(opts: Options) -> Session {
        Session {
            opts,
            entries: Vec::new(),
            next_no: 1,
        }
    }

    fn renderer(&self) -> Renderer {
        Renderer::new(self.opts.color)
    }

    /// Run a `:command` (without its colon). Returns `false` to end the session.
    fn command(&mut self, command: &str) -> bool {
        let (name, rest) = match command.split_once(char::is_whitespace) {
            Some((name, rest)) => (name, rest.trim()),
            None => (command, ""),
        };
        match name {
            "quit" | "q" | "exit" => return false,
            "help" | "h" | "?" => print_out(HELP),
            "reset" => self.entries.clear(),
            "items" => self.print_items(),
            "type" | "t" => {
                if rest.is_empty() {
                    emit_driver_error(&self.opts, "`:type` takes one expression: `:type <expr>`");
                } else {
                    self.print_type(rest);
                }
            }
            other => emit_driver_error(
                &self.opts,
                format!("unknown command `:{other}` (`:help` lists the commands)"),
            ),
        }
        true
    }

    /// Compile and, for statements, run one complete input.
    fn submit(&mut self, text: &str) {
        // Nothing but whitespace and comments: there is nothing to compile, and no number is used.
        let (tokens, lex_diags) = wukong_lexer::tokenize(text, SourceId(0));
        if lex_diags.is_empty() && tokens.iter().all(|t| t.kind == TokenKind::Eof) {
            return;
        }
        let kind = classify(text);
        let entry = Entry {
            no: self.next_no,
            text: text.to_string(),
            kind,
        };
        self.next_no += 1;
        let accepted = match kind {
            Kind::Items => self.eval_items(&entry),
            Kind::Stmts => self.eval_stmts(&entry),
        };
        if accepted {
            self.entries.push(entry);
        }
    }

    /// An item input: compile it with every earlier input, and run the replay when there is one.
    fn eval_items(&self, entry: &Entry) -> bool {
        let built = self.build(Some(entry), Tail::Stmt);
        if let Some(span) = built.reserved {
            let d = Diagnostic::error(
                "`main` is the session's own entry point: give the function another name",
            )
            .primary(span, "defined here");
            emit_diag(&d, self.opts.error_format, &self.renderer(), &built.sm);
            return false;
        }
        let Some(front) = self.front(built, false) else {
            return false;
        };
        let has_replay = self.entries.iter().any(|e| e.kind == Kind::Stmts);
        match self.lower_and_run(front, has_replay, true) {
            Some(Ran::ReplayFailed) => false,
            Some(_) => true,
            None => false,
        }
    }

    /// A statement input: compile it after the replay and run it, printing its value when it ends
    /// in an expression.
    fn eval_stmts(&self, entry: &Entry) -> bool {
        let built = self.build(Some(entry), Tail::Stmt);
        if let Some((span, what)) = built.refused {
            let d = Diagnostic::error(format!(
                "a top-level `{what}` has no place in a REPL input: the session's `main` runs every \
                 input, so it is not this input's to end"
            ))
            .primary(span, format!("this `{what}`"));
            emit_diag(&d, self.opts.error_format, &self.renderer(), &built.sm);
            return false;
        }
        let tail = built.tail;
        let tail_is_name = built.tail_is_name;
        let Some(front) = self.front(built, false) else {
            return false;
        };
        // The tail's type decides what is printed: nothing for `()`; the value, when `printf`
        // can print it (sema decides that, by checking the printing program); else the type,
        // after evaluating the tail for its effects — unless it is a bare name, which has none.
        let value_ty = tail
            .and_then(|t| front.sema.types.get(&t))
            .filter(|ty| !matches!(ty, Ty::Unit | Ty::Unknown | Ty::Error))
            .cloned();
        let (front, type_echo) = match value_ty {
            None => (front, None),
            Some(ty) => match self.front(self.build(Some(entry), Tail::Print), true) {
                Some(printing) => (printing, None),
                None => {
                    let shown = display_ty(&ty, &front);
                    let front = if tail_is_name {
                        self.front(self.build(Some(entry), Tail::Drop), true)
                            .unwrap_or(front)
                    } else {
                        front
                    };
                    (front, Some(shown))
                }
            },
        };
        match self.lower_and_run(front, true, false) {
            Some(Ran::Completed) => {
                if let Some(ty) = type_echo {
                    print_out(&format!("<{ty}>\n"));
                }
                true
            }
            _ => false,
        }
    }

    /// `:type <expr>`: check `expr` after the replay and print its type; nothing runs and the
    /// session is unchanged.
    fn print_type(&mut self, text: &str) {
        let entry = Entry {
            no: self.next_no,
            text: text.to_string(),
            kind: Kind::Stmts,
        };
        self.next_no += 1;
        // One expression, nothing else: a statement has no type to show. A text that does not
        // parse goes on, so its errors are reported where they are.
        let (tokens, lex_diags) = wukong_lexer::tokenize(text, SourceId(0));
        let probe =
            wukong_parser::parse_stmts_tokens_with_lints(&tokens, text, &mut Interner::new(), 0);
        let parses = lex_diags.iter().chain(&probe.diags).all(|d| !d.is_error());
        if parses && !(probe.block.stmts.is_empty() && probe.block.tail.is_some()) {
            emit_driver_error(&self.opts, "`:type` takes one expression: `:type <expr>`");
            return;
        }
        let built = self.build(Some(&entry), Tail::Stmt);
        let tail = built.tail;
        let Some(front) = self.front(built, false) else {
            return;
        };
        let Some(ty) = tail.and_then(|t| front.sema.types.get(&t)).cloned() else {
            return;
        };
        // Sema publishes a `str` as the `[]u8` it is laid out as; the printing program tells the
        // two apart (a `str` argument prints as text).
        let is_str = matches!(&ty, Ty::Slice(e) if matches!(**e, Ty::Scalar(Scalar::U8)))
            && self
                .front(self.build(Some(&entry), Tail::Print), true)
                .is_some_and(|p| p.tail.is_some_and(|t| p.sema.str_print_args.contains(&t)));
        let shown = if is_str {
            "str".to_string()
        } else {
            display_ty(&ty, &front)
        };
        print_out(&format!("{shown}\n"));
    }

    /// `:items`: the source of every item the session keeps, in definition order.
    fn print_items(&self) {
        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let mut next_node = 0;
        let mut entries: Vec<(SourceId, Vec<Group>)> = Vec::new();
        for entry in self.entries.iter().filter(|e| e.kind == Kind::Items) {
            let id = sm.add(entry.file_name(), entry.text.clone());
            let (tokens, _) = wukong_lexer::tokenize(sm.source(id), id);
            let out = wukong_parser::parse_module_tokens_with_lints(
                &tokens,
                sm.source(id),
                &mut interner,
                next_node,
            );
            next_node = out.next_node;
            entries.push((id, groups(out.module.items, &interner, &sm)));
        }
        drop_replaced(&mut entries);
        let mut text = String::new();
        for (_, groups) in &entries {
            for g in groups {
                text.push_str(sm.span_text(g.items[0].span));
                text.push('\n');
            }
        }
        print_out(&text);
    }

    /// Parse the session's inputs, `input` (the new one) last, into one program around the wrapper
    /// `main`. The new input's tail becomes what `tail` says.
    fn build(&self, input: Option<&Entry>, tail: Tail) -> Built {
        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let mut next_node = 0u32;
        let mut lints = Vec::new();
        let mut diags = Vec::new();
        let mut input_file = None;
        let mut input_tail: Option<Box<Expr>> = None;
        let is_name = |e: &Expr| matches!(e.kind, ExprKind::Path(_));
        let mut refused = None;
        let mut reserved = None;

        let mut item_entries: Vec<(SourceId, Vec<Group>)> = Vec::new();
        let mut replay: Vec<Stmt> = Vec::new();
        let mut new_stmts: Vec<Stmt> = Vec::new();
        let main_sym = interner.intern("main");
        for entry in self.entries.iter().chain(input) {
            let is_new = input.is_some_and(|e| std::ptr::eq(e, entry));
            let id = sm.add(entry.file_name(), entry.text.clone());
            if is_new {
                input_file = Some(id);
            }
            let (tokens, lex_diags) = wukong_lexer::tokenize(sm.source(id), id);
            match entry.kind {
                Kind::Items => {
                    let out = wukong_parser::parse_module_tokens_with_lints(
                        &tokens,
                        sm.source(id),
                        &mut interner,
                        next_node,
                    );
                    next_node = out.next_node;
                    lints.extend(out.lints);
                    if is_new {
                        diags.extend(lex_diags);
                        diags.extend(out.diags);
                        reserved = out.module.items.iter().find_map(|item| match &item.kind {
                            ItemKind::Fn(f) if f.name.sym == main_sym => Some(f.name.span),
                            _ => None,
                        });
                    }
                    item_entries.push((id, groups(out.module.items, &interner, &sm)));
                }
                Kind::Stmts => {
                    let out = wukong_parser::parse_stmts_tokens_with_lints(
                        &tokens,
                        sm.source(id),
                        &mut interner,
                        next_node,
                    );
                    next_node = out.next_node;
                    lints.extend(out.lints);
                    let Block { stmts, tail: t, .. } = out.block;
                    if is_new {
                        diags.extend(lex_diags);
                        diags.extend(out.diags);
                        refused = stmts.iter().find_map(|s| match &s.kind {
                            StmtKind::Return(_) => Some((s.span, "return")),
                            StmtKind::Defer(_) => Some((s.span, "defer")),
                            _ => None,
                        });
                        new_stmts = stmts;
                        input_tail = t;
                    } else {
                        replay.extend(stmts);
                        // An earlier input's value was printed when it ran; replayed, it is
                        // evaluated for its effects only — a bare name has none.
                        if let Some(t) = t.filter(|t| !is_name(t)) {
                            replay.push(expr_stmt(*t, &mut next_node));
                        }
                    }
                }
            }
        }
        drop_replaced(&mut item_entries);

        // The wrapper `main`, parsed from source like any input, with its placeholders replaced.
        let tail_id = input_tail.as_ref().map(|t| t.id);
        let tail_span = input_tail.as_ref().map(|t| t.span);
        let tail_is_name = input_tail.as_deref().is_some_and(is_name);
        let print_tail = tail == Tail::Print && input_tail.is_some();
        let wrapper = sm.add("<repl>", wrapper_source(print_tail));
        let (tokens, _) = wukong_lexer::tokenize(sm.source(wrapper), wrapper);
        let out = wukong_parser::parse_module_tokens_with_lints(
            &tokens,
            sm.source(wrapper),
            &mut interner,
            next_node,
        );
        next_node = out.next_node;
        let mut wrapper_items = out.module.items;
        let syms = [REPLAY, INPUT, VALUE].map(|s| interner.intern(s));
        let mut value = None;
        match input_tail {
            Some(t) if print_tail => value = Some(t),
            Some(_) if tail == Tail::Drop => {}
            Some(t) => new_stmts.push(expr_stmt(*t, &mut next_node)),
            None => {}
        }
        for item in &mut wrapper_items {
            if let ItemKind::Fn(f) = &mut item.kind {
                if let Some(body) = &mut f.body {
                    splice(body, &syms, &mut replay, &mut new_stmts, &mut value);
                }
            }
        }

        let mut items: Vec<Item> = item_entries
            .into_iter()
            .flat_map(|(_, groups)| groups.into_iter().flat_map(|g| g.items))
            .collect();
        items.extend(wrapper_items);
        let span = Span::new(SourceId(wrapper.0), 0, 0);
        Built {
            module: Module {
                name: None,
                items,
                span,
            },
            sm,
            interner,
            next_node,
            lints,
            diags,
            input_file,
            tail: tail_id,
            tail_span,
            tail_is_name,
            refused,
            reserved,
        }
    }

    /// The front end over a built program: the import loader, type aliases, sema. Errors render
    /// (unless `quiet`) and return `None`.
    fn front(&self, built: Built, quiet: bool) -> Option<Front> {
        let Built {
            mut module,
            mut sm,
            mut interner,
            mut next_node,
            mut lints,
            diags,
            input_file,
            tail,
            tail_span,
            ..
        } = built;
        let renderer = self.renderer();
        let show = |ds: &[Diagnostic], sm: &SourceMap| {
            if quiet {
                ds.iter().filter(|d| d.is_error()).count()
            } else {
                emit_diags(ds, &self.opts, &renderer, sm)
            }
        };
        if diags.iter().any(Diagnostic::is_error) {
            let errors = show(&diags, &sm);
            self.summary(errors, quiet);
            return None;
        }
        let mut sink = DiagnosticSink::new();
        // `unused_imports` judges one input; a session's imports are for the inputs after it.
        let _unused_imports = loader::load_imports(
            &mut module,
            std::path::Path::new("<repl>"),
            &mut sm,
            &mut interner,
            &mut sink,
            &mut next_node,
            &mut lints,
        );
        if !sink.has_errors() {
            loader::expand_type_aliases(&mut module, &mut interner, &mut sink, &mut next_node);
        }
        if sink.has_errors() {
            let errors = show(sink.diagnostics(), &sm);
            self.summary(errors, quiet);
            return None;
        }
        let (sema, sema_diags) = wukong_sema::check(&module, &interner);
        if sema_diags.iter().any(Diagnostic::is_error) {
            let errors: Vec<Diagnostic> = sema_diags.into_iter().filter(|d| d.is_error()).collect();
            let errors = show(&errors, &sm);
            self.summary(errors, quiet);
            return None;
        }
        Some(Front {
            module,
            sema,
            sm,
            interner,
            lints,
            input_file,
            tail,
            tail_span,
        })
    }

    /// Lint, lower, optimize and run a checked program on the session's backend, printing the new
    /// input's output and reporting how it ended. With `run` false (an item input with no replay)
    /// the program is lowered and optimized only; `items` says the new input is items, which is
    /// then what a failing replay is blamed on. `None` after a compile error.
    fn lower_and_run(&self, front: Front, run: bool, items: bool) -> Option<Ran> {
        let Front {
            module,
            sema,
            sm,
            mut interner,
            mut lints,
            input_file,
            tail_span,
            ..
        } = front;
        let renderer = self.renderer();
        let in_input = |d: &Diagnostic| d.primary_span().map(|s| s.source) == input_file;

        // --- Lints: the new input's only, with the session's three off. ---
        let lint_options = wukong_sema::lints::LintOptions {
            roots: Vec::new(),
            library: true,
        };
        lints.extend(wukong_sema::lints::run(
            &module,
            &sema,
            &interner,
            &lint_options,
        ));
        // A value the input asked to see is not "computed and thrown away" (`W0004`), even when it
        // has no printed form and is evaluated as a statement.
        lints.retain(|d| {
            !matches!(d.code, Some("W0001") | Some("W0002") | Some("W0008"))
                && !(d.code == Some("W0004")
                    && tail_span.is_some()
                    && d.primary_span() == tail_span)
        });
        let lints: Vec<Diagnostic> =
            wukong_sema::lints::apply_levels(lints, &module, &interner, &self.opts.lint_levels)
                .into_iter()
                .filter(in_input)
                .collect();
        if lints.iter().any(Diagnostic::is_error) {
            let (denied, warned): (Vec<Diagnostic>, Vec<Diagnostic>) =
                lints.into_iter().partition(|d| d.is_error());
            let ordered: Vec<Diagnostic> = denied.into_iter().chain(warned).collect();
            let errors = emit_diags(&ordered, &self.opts, &renderer, &sm);
            self.summary(errors, false);
            return None;
        }

        // --- Lowering, as `compile` lowers a `--run`. ---
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
        let (mut program, lower_diags) = wukong_mir_build::lower_program_located(
            &module,
            &sema,
            &mut interner,
            lower_options_for(&self.opts),
            &locate,
        );
        drop(module);
        drop(sema);
        let lower_shown: Vec<Diagnostic> = lower_diags
            .into_iter()
            .filter(|d| d.is_error() || in_input(d))
            .collect();
        let lower_errors = emit_diags(&lower_shown, &self.opts, &renderer, &sm);
        emit_diags(&lints, &self.opts, &renderer, &sm);
        if lower_errors > 0 {
            self.summary(lower_errors, false);
            return None;
        }

        let main = interner.intern("main");
        prune::retain_reachable(&mut program, &[main]);
        wukong_opt::optimize_preserving_with(
            &mut program,
            self.opts.opt_level,
            &[],
            wukong_opt::OptimizeOptions::FULL,
        );
        if verify_or_ice(&program) {
            return None;
        }
        if !run {
            return Some(Ran::Completed);
        }
        if self.opts.backend == BackendKind::Interp {
            let refused = wukong_interp::unsupported_externs(&program, &interner);
            if !refused.is_empty() {
                for msg in refused {
                    emit_diag(
                        &Diagnostic::error(msg).with_code("C0001"),
                        self.opts.error_format,
                        &renderer,
                        &SourceMap::new(),
                    );
                }
                return None;
            }
        }

        // --- Run, with both streams captured: the replay's part is dropped. ---
        let (result, stdout, stderr) = match self.opts.backend {
            BackendKind::Native => {
                let backend = wukong_codegen_cranelift::BackendOpt::for_level(self.opts.opt_level);
                let compiled = match wukong_codegen_cranelift::jit_compile_for_with(
                    &program,
                    main,
                    &interner,
                    self.opts.target_cpu,
                    backend,
                ) {
                    Ok(compiled) => compiled,
                    Err(failure) => {
                        emit_diag(
                            &native_failure_diag(
                                &failure,
                                std::path::Path::new("<repl>"),
                                Some(&sm),
                            ),
                            self.opts.error_format,
                            &renderer,
                            &sm,
                        );
                        return None;
                    }
                };
                wukong_runtime::fmt::begin_stderr_capture();
                let (result, stdout) = compiled.run_capturing();
                (result, stdout, wukong_runtime::fmt::end_stderr_capture())
            }
            _ => {
                wukong_runtime::fmt::begin_stderr_capture();
                let (result, stdout) = wukong_interp::run_capturing(&program, main, &interner);
                (result, stdout, wukong_runtime::fmt::end_stderr_capture())
            }
        };
        let Some((_, input_out)) = split_once(&stdout, MARK.as_bytes()) else {
            let why = match result {
                Err(e) => e,
                Ok(code) => format!("the program ended early, with exit status {code}"),
            };
            emit_driver_error(
                &self.opts,
                format!(
                    "the session's earlier inputs no longer run{}: {why}",
                    if items { " with these items" } else { "" }
                ),
            );
            self.note("`:reset` starts the session afresh");
            return Some(Ran::ReplayFailed);
        };
        let (shown, completed) = match split_once(input_out, END.as_bytes()) {
            Some((shown, _)) => (shown, true),
            None => (input_out, false),
        };
        let shown_err = split_once(&stderr, MARK.as_bytes()).map_or(&[][..], |(_, rest)| rest);
        {
            let mut out = std::io::stdout();
            let _ = out.write_all(shown);
            let _ = out.flush();
            let mut err = std::io::stderr();
            let _ = err.write_all(shown_err);
            let _ = err.flush();
        }
        match result {
            Err(e) => {
                emit_driver_error(&self.opts, e);
                Some(Ran::Stopped)
            }
            Ok(_) if completed => Some(Ran::Completed),
            Ok(code) => {
                self.note(&format!(
                    "this input ended the program (exit status {code}) before its end, so the \
                     session does not keep it"
                ));
                Some(Ran::Stopped)
            }
        }
    }

    /// The "aborting due to" trailer after an input's errors (human format; never when quiet).
    fn summary(&self, errors: usize, quiet: bool) {
        if !quiet {
            crate::emit_summary(errors, &self.opts, &self.renderer());
        }
    }

    /// A span-less `note:` line, honouring `--error-format`.
    fn note(&self, message: &str) {
        let d = Diagnostic::new(Severity::Note, message);
        let sm = SourceMap::new();
        match self.opts.error_format {
            ErrorFormat::Human => eprint_line(&self.renderer().render(&d, &sm)),
            ErrorFormat::Json => eprint_line(&wukong_diag::to_json(&d, &sm)),
        }
    }
}

/// The wrapper `main`'s source: the replay and the input between the markers, then — when the
/// input's value is printed — the `printf` of it.
fn wrapper_source(print_value: bool) -> String {
    let lit = |s: &str| format!("\"{}\"", s.replace('\x1e', "\\x1E"));
    let mut src = format!(
        "fn main() -> i32 {{\n    {REPLAY};\n    printf({m});\n    eprintf({m});\n    {INPUT};\n",
        m = lit(MARK)
    );
    if print_value {
        src.push_str(&format!("    printf(\"{{:?}}\\n\", {VALUE});\n"));
    }
    src.push_str(&format!("    printf({});\n    return 0;\n}}\n", lit(END)));
    src
}

/// Replace the wrapper's placeholders in `body`: the `REPLAY` and `INPUT` statements by those
/// statement lists, the `VALUE` argument by the input's tail expression.
fn splice(
    body: &mut Block,
    syms: &[Symbol; 3],
    replay: &mut Vec<Stmt>,
    input: &mut Vec<Stmt>,
    value: &mut Option<Box<Expr>>,
) {
    let [replay_sym, input_sym, value_sym] = *syms;
    let is = |e: &Expr, sym: Symbol| matches!(&e.kind, ExprKind::Path(p) if p.is_single() && p.first().sym == sym);
    let mut out = Vec::with_capacity(body.stmts.len() + replay.len() + input.len());
    for mut stmt in std::mem::take(&mut body.stmts) {
        match &mut stmt.kind {
            StmtKind::Expr(e) if is(e, replay_sym) => out.append(replay),
            StmtKind::Expr(e) if is(e, input_sym) => out.append(input),
            StmtKind::Expr(Expr {
                kind: ExprKind::Call { args, .. },
                ..
            }) => {
                for arg in args.iter_mut() {
                    if is(arg, value_sym) {
                        if let Some(v) = value.take() {
                            *arg = *v;
                        }
                    }
                }
                out.push(stmt);
            }
            _ => out.push(stmt),
        }
    }
    body.stmts = out;
}

/// `expr;` as a statement with a fresh id.
fn expr_stmt(expr: Expr, next_node: &mut u32) -> Stmt {
    let id = NodeId(*next_node);
    *next_node += 1;
    Stmt {
        id,
        attrs: Vec::new(),
        span: expr.span,
        kind: StmtKind::Expr(expr),
    }
}

/// An input's items, grouped: an `impl` marker with the method functions the parser flattened out
/// of it, every other item alone. Each group's key is what a later definition replaces it by.
fn groups(items: Vec<Item>, interner: &Interner, sm: &SourceMap) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    let mut methods: HashSet<Symbol> = HashSet::new();
    for item in items {
        if let ItemKind::Fn(f) = &item.kind {
            if methods.contains(&f.name.sym) {
                out.last_mut()
                    .expect("a method follows its impl marker")
                    .items
                    .push(item);
                continue;
            }
        }
        let name = |sym: Symbol| Some(interner.resolve(sym).to_string());
        let key = match &item.kind {
            ItemKind::Fn(f) => name(f.name.sym),
            ItemKind::Struct(s) => name(s.name.sym),
            ItemKind::Enum(e) => name(e.name.sym),
            ItemKind::Const(c) => name(c.name.sym),
            ItemKind::Trait(t) => name(t.name.sym),
            ItemKind::TypeAlias(a) => name(a.name.sym),
            ItemKind::Impl(i) => {
                methods = i.methods.iter().map(|m| m.fn_name).collect();
                let ty = sm.span_text(i.self_ty.span).to_string();
                Some(match &i.trait_ref {
                    Some(t) => format!("impl {} for {ty}", sm.span_text(t.span)),
                    None => format!("impl {ty}"),
                })
            }
            // The same import, written again, is the same import.
            ItemKind::Import(_) => Some(sm.span_text(item.span).to_string()),
            ItemKind::Extern(_) => None,
        };
        if !matches!(item.kind, ItemKind::Impl(_)) {
            methods.clear();
        }
        out.push(Group {
            key,
            items: vec![item],
        });
    }
    out
}

/// Drop every group a later input redefines (by key). Within one input nothing is dropped: two
/// definitions of a name in one input are that input's error to report.
fn drop_replaced(entries: &mut [(SourceId, Vec<Group>)]) {
    let mut defined_later: HashSet<String> = HashSet::new();
    for (_, groups) in entries.iter_mut().rev() {
        groups.retain(|g| g.key.as_ref().is_none_or(|k| !defined_later.contains(k)));
        defined_later.extend(groups.iter().filter_map(|g| g.key.clone()));
    }
}

/// A type as the session shows it.
fn display_ty(ty: &Ty, front: &Front) -> String {
    ty.display(&front.interner)
}

/// Split `hay` around the first occurrence of `needle`.
fn split_once<'a>(hay: &'a [u8], needle: &[u8]) -> Option<(&'a [u8], &'a [u8])> {
    hay.windows(needle.len())
        .position(|w| w == needle)
        .map(|i| (&hay[..i], &hay[i + needle.len()..]))
}

/// Write session output (a value, `:items`, `:help`) to stdout and flush it.
fn print_out(text: &str) {
    let mut out = std::io::stdout();
    let _ = out.write_all(text.as_bytes());
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_input_continues_while_a_bracket_or_comment_is_open() {
        for done in [
            "1 + 2\n",
            "fn f() -> i32 { 1 }\n",
            "let s = \"{\";\n",
            "// (\n",
            // A string ends at its line: waiting for its quote would swallow every later line.
            "let s = \"abc (\n",
        ] {
            assert!(!needs_more(done), "{done:?}");
        }
        for more in [
            "fn f() -> i32 {\n",
            "let a = [1,\n",
            "print((1 +\n",
            "/* note\n",
            "/* note (\n*/ f(\n",
        ] {
            assert!(needs_more(more), "{more:?}");
        }
        // A closer too many is complete: the parser reports it.
        assert!(!needs_more("1 }\n"));
    }

    #[test]
    fn inputs_are_classified_by_their_first_token() {
        for items in [
            "fn f() {}",
            "pub fn f() {}",
            "@parallel fn f() {}",
            "@export(\"g\") fn g() {}",
            "struct P { x: i32 }",
            "enum E { A }",
            "const N: i32 = 4;",
            "import lib.math;",
            "impl P { fn x(self) -> i32 { self.x } }",
            "trait T { fn m(self) -> i32; }",
            "extern \"C\" { fn abs(x: i32) -> i32; }",
            "type V = [f32; 4];",
        ] {
            assert_eq!(classify(items), Kind::Items, "{items}");
        }
        for stmts in [
            "let x = 1;",
            "x + 1",
            "print(1)",
            "@unroll for i in 0..4 { print(i); }",
            "type + 1",
            "if true { 1 } else { 2 }",
        ] {
            assert_eq!(classify(stmts), Kind::Stmts, "{stmts}");
        }
    }

    /// The wrapper parses, carries both markers and the value placeholder only when asked, and
    /// its string literals spell the marker bytes the output is split at.
    #[test]
    fn the_wrapper_main_parses_with_its_placeholders() {
        for print_value in [false, true] {
            let src = wrapper_source(print_value);
            assert!(src.is_ascii());
            let (module, diags) =
                wukong_parser::parse_module(&src, SourceId(0), &mut Interner::new());
            assert!(diags.is_empty(), "{diags:?}\n{src}");
            assert_eq!(module.items.len(), 1);
            assert_eq!(src.contains(VALUE), print_value);
            assert_eq!(src.matches("x1Ewukong-repl-input").count(), 2);
        }
        assert_eq!(
            split_once(b"ab\x1ewukong-repl-input\x1ecd", MARK.as_bytes()),
            Some((&b"ab"[..], &b"cd"[..]))
        );
        assert_eq!(split_once(b"abc", MARK.as_bytes()), None);
    }

    /// A later definition replaces an earlier one of the same name, or the same `impl`; one input's
    /// own duplicates stay for sema to report.
    #[test]
    fn later_items_replace_earlier_ones_by_name() {
        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let mut next = 0;
        let mut entries = Vec::new();
        for src in [
            "struct P { x: i32 }\nimpl P { fn get(self) -> i32 { self.x } }\n\
             fn f() -> i32 { 1 }\nfn g() -> i32 { 2 }",
            "fn f() -> i32 { 3 }\nimpl P { fn other(self) -> i32 { 0 } }\nfn h() {}\nfn h() {}",
        ] {
            let id = sm.add("t", src);
            let (tokens, _) = wukong_lexer::tokenize(sm.source(id), id);
            let out = wukong_parser::parse_module_tokens_with_lints(
                &tokens,
                sm.source(id),
                &mut interner,
                next,
            );
            next = out.next_node;
            entries.push((id, groups(out.module.items, &interner, &sm)));
        }
        assert_eq!(
            entries[0].1[1].items.len(),
            2,
            "an impl's method rides in its group"
        );
        drop_replaced(&mut entries);
        let keys: Vec<Vec<String>> = entries
            .iter()
            .map(|(_, gs)| {
                gs.iter()
                    .map(|g| g.key.clone().unwrap_or_default())
                    .collect()
            })
            .collect();
        assert_eq!(keys, [vec!["P", "g"], vec!["f", "impl P", "h", "h"]]);
    }
}
