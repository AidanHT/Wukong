//! `wukongc --repl`, driven non-interactively: each test pipes a transcript into the real binary
//! and checks what it prints, on the interpreter and on the native JIT.
//!
//! Every expected line is derived from the session's documented rules (`wukong_driver::repl`) and
//! the language's printing rules (`{:?}` style: a float always shows its point, a string is quoted),
//! not recorded from a run. With standard input a pipe the session prints no banner and no prompt,
//! so stdout holds exactly the values and the program output.

use std::io::Write;
use std::process::{Command, Stdio};

/// Run `wukongc --repl args…` with `transcript` on standard input; (exit code, stdout, stderr).
fn repl(args: &[&str], transcript: &str) -> (Option<i32>, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--repl")
        .arg("--color=never")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn wukongc --repl");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(transcript.as_bytes())
        .expect("write the transcript");
    let out = child.wait_with_output().expect("wait for wukongc --repl");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Every backend a session runs on, at the default and an optimizing level.
const MODES: [[&str; 2]; 3] = [
    ["--backend=interp", "-O0"],
    ["--backend=native", "-O0"],
    ["--backend=native", "-O2"],
];

/// Items are kept, statements run, an expression prints its value, and a binding — with every
/// later assignment to it — persists into the inputs after it. A clean transcript warns about
/// nothing: the lints that judge one input (an unused `let`, a `mut` used later) are off.
#[test]
fn items_statements_and_values_persist_across_inputs() {
    let transcript = "\
let x = 5;
x + 1
fn sq(a: i32) -> i32 { a * a }
sq(x)
let y = 2.5;
y * 2.0
\"hi\"
[1, 2, 3]
struct P { x: i32, y: f32 }
P { x: 1, y: 0.5 }
let mut acc = 0;
for i in 0..4 { acc = acc + i; }
acc
acc = acc * 10;
acc
print(acc)
1
";
    for mode in MODES {
        let (code, out, err) = repl(&mode, transcript);
        assert_eq!(code, Some(0), "{mode:?}: {err}");
        // 5 + 1; 5 * 5; 2.5 * 2.0 is 5.0 (a float keeps its point); the string quoted; the array;
        // the struct; 0 + 1 + 2 + 3; that times ten, twice (echoed, then printed by `print`, a
        // `()` tail that echoes nothing); and the replay of `print(acc)` does not print again.
        assert_eq!(
            out, "6\n25\n5.0\n\"hi\"\n[1, 2, 3]\nP { x: 1, y: 0.5 }\n6\n60\n60\n1\n",
            "{mode:?}: {err}"
        );
        assert_eq!(err, "", "{mode:?}: a clean session reports nothing");
    }
}

/// An input that fails — to compile, at run time, or by ending the program — is reported and
/// dropped, and the session goes on from where it was. A diagnostic points into the input that
/// holds the problem, `<repl:N>`, with lines counted from that input's own first line.
#[test]
fn a_failing_input_is_reported_and_the_session_continues() {
    let transcript = "\
let a = [1, 2, 3];
let b: i32 = \"no\";
b
a[1]
assert(a[0] == 2);
a[0]
exit(3);
fn bad() -> i32 {
    let q: i32 = 1.5;
    q
}
a[2]
";
    for mode in MODES {
        let (code, out, err) = repl(&mode, transcript);
        assert_eq!(code, Some(0), "{mode:?}: {err}");
        // Only `a` survives: a[1], a[0] and a[2] print; `b`, the failed assert and `exit` are gone.
        assert_eq!(out, "2\n1\n3\n", "{mode:?}: {err}");
        for want in [
            "error[E0401]",
            "--> <repl:2>:1:1",
            "error[E0301]: cannot find `b` in this scope",
            "--> <repl:3>:1:1",
            "assertion failed",
            "exit status 3",
            // The bad function's error is on line 2 of input 8, at the `let`.
            "--> <repl:8>:2:5",
        ] {
            assert!(err.contains(want), "{mode:?}: missing {want:?} in:\n{err}");
        }
    }
}

/// An input continues on the next line while a `(`, `[` or `{` is open, or a block comment is; a
/// bracket inside a string or a comment does not count, and a string that is not closed on its
/// line is that input's error rather than a reason to keep reading — even after a bracket opened
/// before it on its line, which the broken string leaves nothing to close. A bracket opened on an
/// earlier line still holds the input open, so a function with a broken string is one input.
#[test]
fn an_input_continues_while_a_bracket_or_comment_is_open() {
    let transcript = "\
fn fact(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    n * fact(n - 1)
}
let xs = [
    fact(3),
    fact(5),
];
xs
let s = \"an open ( inside a string\";
s
/* a comment (
   that ends here */ 7
// a line with nothing but a comment is no input
let bad = \"no end
7 * 6
print(\"no end
1 + 1
fn h() -> i32 {
    let t = \"no end
    5
}
h()
";
    let (code, out, err) = repl(&[], transcript);
    assert_eq!(code, Some(0), "{err}");
    // 3! and 5!; the string, quoted; the comment's two lines are one input whose value is 7; the
    // unclosed string is input 7 (the comment-only line took no number) and the next line runs;
    // so does the line after `print("no end` (input 9). `fn h` is input 11, all four lines of it:
    // its `5` is not an input of its own (it would print), and `h` is never defined (input 12).
    assert_eq!(
        out, "[6, 120]\n\"an open ( inside a string\"\n7\n42\n2\n",
        "{err}"
    );
    for want in [
        "--> <repl:7>:1:11",
        "--> <repl:9>:1:7",
        "--> <repl:11>:2:13",
        "cannot find function `h`",
    ] {
        assert!(err.contains(want), "missing {want:?} in:\n{err}");
    }
    assert_eq!(err.matches("error[E0102]").count(), 3, "{err}");
    // Every diagnostic is in one of the four failing inputs.
    for line in err.lines().filter(|l| l.contains("--> <repl:")) {
        assert!(
            ["<repl:7>:", "<repl:9>:", "<repl:11>:", "<repl:12>:"]
                .iter()
                .any(|at| line.contains(at)),
            "{line}\nin:\n{err}"
        );
    }
}

/// An attribute on a line of its own — the corpus's usual layout — waits for the item or statement
/// on the lines after it, blank lines included, and stays on it: the item keeps its attributes
/// (`:items` shows them, and sema checks them: an unknown one is refused), and a loop keeps
/// `@generic`.
#[test]
fn an_attribute_on_its_own_line_belongs_to_what_follows() {
    let transcript = "\
@strict_fp
fn f(a: f32) -> f32 { a * 2.0 }
f(1.5)
@export(\"twice\")
@inline

fn g(a: i32) -> i32 { a + a }
g(4)
let mut s = 0;
@generic
for i in 0..4 { s = s + i; }
s
@bogus
fn k() -> i32 { 1 }
k()
:items
";
    for mode in MODES {
        let (code, out, err) = repl(&mode, transcript);
        assert_eq!(code, Some(0), "{mode:?}: {err}");
        // 1.5 * 2.0; 4 + 4; 0 + 1 + 2 + 3; then `:items`, each item from its first attribute.
        assert_eq!(
            out,
            "3.0\n8\n6\n\
             @strict_fp\nfn f(a: f32) -> f32 { a * 2.0 }\n\
             @export(\"twice\")\n@inline\n\nfn g(a: i32) -> i32 { a + a }\n",
            "{mode:?}: {err}"
        );
        // `@bogus` reaches sema with `fn k` (input 8), which is refused, so input 9 finds no `k`.
        for want in [
            "error[E0306]",
            "--> <repl:8>:1:",
            "cannot find function `k`",
        ] {
            assert!(err.contains(want), "{mode:?}: missing {want:?} in:\n{err}");
        }
        assert_eq!(err.matches("error[").count(), 2, "{mode:?}: {err}");
    }
}

/// `:items` prints the kept items in definition order; a later item of the same name replaces the
/// earlier one; `:reset` forgets every item and binding.
#[test]
fn items_are_listed_replaced_and_reset() {
    let transcript = "\
fn f() -> i32 { 1 }
struct Q { v: i32 }
let k = 3;
:items
fn f() -> i32 { 2 }
f() + k
:items
:reset
k
f()
:items
1
";
    let (code, out, err) = repl(&[], transcript);
    assert_eq!(code, Some(0), "{err}");
    assert_eq!(
        out,
        "fn f() -> i32 { 1 }\nstruct Q { v: i32 }\n\
         5\n\
         struct Q { v: i32 }\nfn f() -> i32 { 2 }\n\
         1\n",
        "{err}"
    );
    assert!(
        err.contains("cannot find `k` in this scope") && err.contains("`f`"),
        "{err}"
    );
}

/// `:type <expr>` prints the type sema gives the expression after the session's inputs, and
/// runs nothing.
#[test]
fn type_prints_the_type_without_running() {
    let transcript = "\
let x = 5;
let v: [f32; 3] = [1.0, 2.0, 3.0];
fn h(a: f32) -> f32 { a }
fn note() { print(99); }
let w: str = \"wk\";
w
:type x
:type v
:type v[0] * 2.0
:type h
:type h(1.0) > 0.0
:type \"s\"
:type w
:type note()
:type let z = 1;
:type nope
:type
";
    for mode in MODES {
        let (code, out, err) = repl(&mode, transcript);
        assert_eq!(code, Some(0), "{mode:?}: {err}");
        // `w`'s value, quoted; a literal is a `*u8` and a `str` binding a `str` (not the `[]u8` it
        // is laid out as); `note()` is typed, not run: `()` and no 99.
        assert_eq!(
            out, "\"wk\"\ni32\n[f32; 3]\nf32\nfn(f32) -> f32\nbool\n*u8\nstr\n()\n",
            "{mode:?}: {err}"
        );
        assert_eq!(
            err.matches("`:type` takes one expression").count(),
            2,
            "{mode:?}: {err}"
        );
        assert!(err.contains("cannot find `nope`"), "{mode:?}: {err}");
    }
}

/// The replay that carries the bindings forward is silent on both streams, and a program in a
/// session reads end of input: the session's own lines are not the program's to consume.
#[test]
fn the_replay_is_silent_and_standard_input_is_the_sessions() {
    let transcript = "\
print(1);
eprintln(2);
let mut buf: [u8; 8] = [0; 8];
read_stdin_u8(buf)
let n = 10;
n
";
    for mode in MODES {
        let (code, out, err) = repl(&mode, transcript);
        assert_eq!(code, Some(0), "{mode:?}: {err}");
        // `read_stdin_u8` reads nothing (0), and the line after it is still the session's.
        assert_eq!(out, "1\n0\n10\n", "{mode:?}: {err}");
        assert_eq!(
            err, "2\n",
            "{mode:?}: the replayed eprintln is not repeated"
        );
    }
}

/// A redefinition that breaks an earlier input — at compile time or when the replay runs — is
/// refused, and the session keeps the definition that works.
#[test]
fn a_redefinition_that_breaks_an_earlier_input_is_refused() {
    let transcript = "\
fn g() -> i32 { 1 }
let r: i32 = g();
fn g() -> f32 { 1.0 }
r
fn d() -> i32 { 1 }
let q = d();
assert(q == 1);
fn d() -> i32 { 0 }
q + r
";
    let (code, out, err) = repl(&[], transcript);
    assert_eq!(code, Some(0), "{err}");
    assert_eq!(out, "1\n2\n", "{err}");
    // The f32 `g` breaks input 2's annotation; the zero `d` breaks input 7's assert on replay.
    assert!(err.contains("--> <repl:2>:1:1"), "{err}");
    assert!(
        err.contains("no longer run with these items: assertion failed"),
        "{err}"
    );
}

/// A binding is not a stored value: every input recomputes it by replaying the statements that made
/// it, against the items as they are then. A redefinition that breaks nothing is accepted, without a
/// word, and changes every binding computed from the item it replaces.
#[test]
fn a_redefinition_recomputes_the_bindings_made_from_it() {
    let transcript = "\
fn rate() -> i32 { 2 }
let tax = 100 * rate();
tax
fn rate() -> i32 { 3 }
tax
";
    for mode in MODES {
        let (code, out, err) = repl(&mode, transcript);
        assert_eq!(code, Some(0), "{mode:?}: {err}");
        // 100 * 2 when `tax` was made; 100 * 3 once `rate` is redefined, though no input assigned
        // to `tax` in between.
        assert_eq!(out, "200\n300\n", "{mode:?}: {err}");
        assert_eq!(err, "", "{mode:?}: the redefinition is accepted silently");
    }
}

/// A value with no printed form shows its type; a top-level `return` or `defer`, and a `main`, are
/// refused; `:quit` ends the session before the lines after it.
#[test]
fn unprintable_values_refused_inputs_and_quit() {
    let transcript = "\
fn sq(a: f32) -> f32 { a * a }
sq
return 1;
defer print(1);
fn main() -> i32 { 0 }
:help
:frobnicate
sq(1.5)
:quit
sq(2.0)
";
    for mode in MODES {
        let (code, out, err) = repl(&mode, transcript);
        assert_eq!(code, Some(0), "{mode:?}: {err}");
        let (help, rest) = out.split_at(out.find("Commands:").expect("the help text"));
        assert!(
            help.starts_with("<fn(f32) -> f32>\nWukong REPL."),
            "{mode:?}: {out}"
        );
        assert!(rest.contains(":type <expr>"), "{mode:?}: {out}");
        // 1.5 squared; nothing after `:quit` runs.
        assert!(rest.ends_with("\n2.25\n"), "{mode:?}: {out}");
        for want in [
            "a top-level `return`",
            "a top-level `defer`",
            "`main` is the session's own entry point",
            "unknown command `:frobnicate`",
        ] {
            assert!(err.contains(want), "{mode:?}: missing {want:?} in:\n{err}");
        }
    }
}

/// `--error-format=json` keeps the diagnostic stream one JSON object per line in a session too.
#[test]
fn json_diagnostics_stay_json() {
    let (code, out, err) = repl(
        &["--error-format=json"],
        "let b: i32 = \"no\";\n1\n:bogus\n",
    );
    assert_eq!(code, Some(0), "{err}");
    assert_eq!(out, "1\n");
    assert!(!err.trim().is_empty());
    for line in err.lines() {
        assert!(
            line.starts_with('{') && line.ends_with('}'),
            "not a JSON line: {line}"
        );
    }
    assert!(err.contains("<repl:1>"), "{err}");
}
