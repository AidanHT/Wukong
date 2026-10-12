//! The matcher's own tests: the regex dialect, every directive's positive and negative cases, and
//! the error reports. Each expectation is FileCheck's documented behaviour, worked by hand.

use super::regex::{self, Node};
use super::{check, Options};

fn opts() -> Options {
    Options::default()
}

#[track_caller]
fn ok_with(checks: &str, input: &str, o: &Options) {
    if let Err(e) = check("t.txt", checks, input, o) {
        panic!("expected the input to match\n--- checks ---\n{checks}\n--- input ---\n{input}\n--- report ---\n{e}");
    }
}

#[track_caller]
fn ok(checks: &str, input: &str) {
    ok_with(checks, input, &opts());
}

#[track_caller]
fn fail_with(checks: &str, input: &str, o: &Options) -> String {
    match check("t.txt", checks, input, o) {
        Ok(()) => panic!("expected a failure\n--- checks ---\n{checks}\n--- input ---\n{input}"),
        Err(e) => e,
    }
}

#[track_caller]
fn fail(checks: &str, input: &str) -> String {
    fail_with(checks, input, &opts())
}

#[track_caller]
fn assert_has(report: &str, want: &str) {
    assert!(
        report.contains(want),
        "the report should contain {want:?}:\n{report}"
    );
}

// ---------------------------------------------------------------------------------------------
// The regex dialect.
// ---------------------------------------------------------------------------------------------

/// The `(start, end)` char span of the first match of `re` in `text`, or `None`.
fn search(re: &str, text: &str) -> Option<(usize, usize)> {
    let node = regex::parse(re).unwrap_or_else(|e| panic!("{re}: {e}"));
    let chars: Vec<char> = text.chars().collect();
    regex::find(&node, 0, &chars, 0, true)
        .unwrap()
        .map(|f| (f.start, f.end))
}

#[test]
fn regex_literals_classes_and_shorthands() {
    assert_eq!(search("abc", "xxabcx"), Some((2, 5)));
    assert_eq!(search("a.c", "a\u{e9}c"), Some((0, 3)));
    assert_eq!(search("[0-9]+", "v123 = add"), Some((1, 4)));
    assert_eq!(search("[^ ]+", "  word next"), Some((2, 6)));
    assert_eq!(search("[]a]+", "x]a]y"), Some((1, 4)));
    assert_eq!(search("[a-]+", "x-a-y"), Some((1, 4)));
    assert_eq!(search("[[:digit:]]+", "ab12c"), Some((2, 4)));
    assert_eq!(search("[[:alpha:]_]+", "1ab_c2"), Some((1, 5)));
    assert_eq!(search(r"\d+", "v42"), Some((1, 3)));
    assert_eq!(search(r"\w+", "  fadd.f32"), Some((2, 6)));
    assert_eq!(search(r"\s+", "a \t b"), Some((1, 4)));
    assert_eq!(search(r"\S+", "  ab "), Some((2, 4)));
    assert_eq!(search(r"[\d.]+", "x1.5y"), Some((1, 4)));
    assert_eq!(search(r"\.\*\(", "a.*(b"), Some((1, 4)));
    assert_eq!(search("x", "abc"), None);
}

#[test]
fn regex_anchors_alternation_and_boundaries() {
    assert_eq!(search("^ab", "ab ab"), Some((0, 2)));
    assert_eq!(search("^b", "ab"), None);
    assert_eq!(search("b$", "b ab"), Some((3, 4)));
    assert_eq!(search("(fadd|fmul) v", "x fmul v1"), Some((2, 8)));
    assert_eq!(search("(?:a|b)c", "xbc"), Some((1, 3)));
    // Alternation is leftmost-first: `a` wins over `ab` at the same start.
    assert_eq!(search("a|ab", "ab"), Some((0, 1)));
    assert_eq!(search(r"\badd\b", "fadd add"), Some((5, 8)));
    assert_eq!(search(r"\Badd", "add fadd"), Some((5, 8)));
}

#[test]
fn regex_repetition_greedy_lazy_and_counted() {
    assert_eq!(search("a*", "aaab"), Some((0, 3)));
    assert_eq!(search("a*?b", "aaab"), Some((0, 4)));
    assert_eq!(search("<.+?>", "<a><b>"), Some((0, 3)));
    assert_eq!(search("<.+>", "<a><b>"), Some((0, 6)));
    assert_eq!(search("a?b", "b"), Some((0, 1)));
    assert_eq!(search("a{2}", "a aa"), Some((2, 4)));
    assert_eq!(search("a{2,}", "aaaa"), Some((0, 4)));
    assert_eq!(search("a{1,2}", "aaa"), Some((0, 2)));
    assert_eq!(
        search("x{,2}", "x{,2}"),
        Some((0, 5)),
        "not a count: literal braces"
    );
    // An empty iteration stops the loop instead of spinning.
    assert_eq!(search("(a*)*b", "aab"), Some((0, 3)));
    assert_eq!(search("(|a)+b", "ab"), Some((0, 2)));
    assert_eq!(search("(^)*x", "x"), Some((0, 1)));
}

#[test]
fn regex_backreference_matches_the_captured_text() {
    let node = Node::Concat(vec![
        Node::Group(Some(0), Box::new(regex::parse("[a-z]+").unwrap())),
        Node::Char('-'),
        Node::Backref(0),
    ]);
    let chars: Vec<char> = "foo-bar foo-foo".chars().collect();
    let f = regex::find(&node, 1, &chars, 0, true).unwrap().unwrap();
    assert_eq!((f.start, f.end), (8, 15));
    assert_eq!(f.caps, vec![Some((8, 11))]);
}

#[test]
fn regex_end_anchor_respects_a_cut_line() {
    let node = regex::parse("b$").unwrap();
    let chars: Vec<char> = "ab".chars().collect();
    assert!(regex::find(&node, 0, &chars, 0, true).unwrap().is_some());
    // The same text, cut short at a region end that is not the line's end: no `$` there.
    assert!(regex::find(&node, 0, &chars, 0, false).unwrap().is_none());
}

#[test]
fn regex_syntax_errors_are_reported() {
    for (re, want) in [
        ("(ab", "unclosed `(`"),
        ("ab)", "unmatched `)`"),
        ("[ab", "unclosed `[`"),
        ("*a", "nothing to repeat"),
        (r"\q", "unsupported escape"),
        ("a\\", "trailing `\\`"),
        ("[z-a]", "reversed range"),
        ("[[:nope:]]", "unknown class"),
        ("a{3,1}", "m < n"),
        ("a{5000}", "above 1000"),
    ] {
        let err = regex::parse(re).expect_err(re);
        assert!(err.0.contains(want), "{re}: {err}");
    }
}

#[test]
fn regex_step_budget_turns_blowup_into_an_error() {
    let node = regex::parse("(a|aa)*c").unwrap();
    let chars: Vec<char> = "a".repeat(60).chars().collect();
    let err = regex::find(&node, 0, &chars, 0, true).expect_err("must run out of budget");
    assert!(err.0.contains("backtracking steps"), "{err}");
}

/// A one-character repetition tries its ends in a loop, not one stack frame per character: on
/// this test thread's default stack, a 300 000-character line would overflow a recursive `.*`.
#[test]
fn regex_long_lines_do_not_recurse_per_character() {
    let line = format!("a{}b", "c".repeat(300_000));
    assert_eq!(search("a.*b", &line), Some((0, 300_002)));
    assert_eq!(search("a.*?b", &line), Some((0, 300_002)));
    assert_eq!(search("c{2,5}b", &line), Some((299_996, 300_002)));
    let hex = format!("static blob = hex {}\n", "0f".repeat(100_000));
    ok("CHECK: static {{.*}} = hex {{[0-9a-f]+}}{{$}}", &hex);
    fail("CHECK: static {{.*}} = hex {{[0-9a-f]+}}g", &hex);
}

// ---------------------------------------------------------------------------------------------
// CHECK, CHECK-NEXT, CHECK-SAME, CHECK-EMPTY, CHECK-COUNT.
// ---------------------------------------------------------------------------------------------

#[test]
fn check_matches_in_order_and_may_share_a_line() {
    ok("CHECK: a\nCHECK: c", "a\nb\nc\n");
    ok("CHECK: foo\nCHECK: bar", "foo bar\n");
    let r = fail("CHECK: c\nCHECK: a", "a\nb\nc\n");
    assert_has(
        &r,
        "t.txt:2:8: error: CHECK: expected string not found in input",
    );
    // The scan starts where `c` ended: the end of line 3.
    assert_has(&r, "<stdin>:3:2: note: scanning from here");
}

#[test]
fn check_next_needs_exactly_the_next_line() {
    ok("CHECK: a\nCHECK-NEXT: b\nCHECK-NEXT: c", "a\nb\nc\n");
    let same = fail("CHECK: a\nCHECK-NEXT: b", "a b\n");
    assert_has(
        &same,
        "t.txt:2:13: error: CHECK-NEXT: is on the same line as the previous match",
    );
    let skipped = fail("CHECK: a\nCHECK-NEXT: c", "a\nb\nc\n");
    assert_has(
        &skipped,
        "CHECK-NEXT: is not on the line after the previous match",
    );
    assert_has(&skipped, "<stdin>:3:1: note: the match is here");
    assert_has(
        &skipped,
        "<stdin>:2:1: note: the line after the previous match is here",
    );
}

#[test]
fn check_same_stays_on_the_line() {
    ok(
        "CHECK: fn f(\nCHECK-SAME: i32\nCHECK-SAME: -> f32",
        "fn f(v0 i32) -> f32 {\n",
    );
    let r = fail("CHECK: fn f(\nCHECK-SAME: ret", "fn f(v0 i32) {\n  ret\n");
    assert_has(
        &r,
        "CHECK-SAME: is not on the same line as the previous match",
    );
}

#[test]
fn check_empty_matches_the_next_line_only_when_blank() {
    ok("CHECK: a\nCHECK-EMPTY:\nCHECK-NEXT: b", "a\n\nb\n");
    let r = fail("CHECK: a\nCHECK-EMPTY:", "a\nb\n\n");
    assert_has(
        &r,
        "CHECK-EMPTY: is not on the line after the previous match",
    );
    fail("CHECK: a\nCHECK-EMPTY:", "a\nb\n");
}

#[test]
fn check_count_matches_n_times_in_a_row() {
    ok("CHECK-COUNT-3: x\nCHECK: end", "x x\nx\nend\n");
    let r = fail("CHECK-COUNT-4: x", "x x\nx\n");
    assert_has(
        &r,
        "CHECK-COUNT-4: expected string not found in input (match 4 of 4)",
    );
}

// ---------------------------------------------------------------------------------------------
// CHECK-NOT, CHECK-DAG, CHECK-LABEL.
// ---------------------------------------------------------------------------------------------

#[test]
fn check_not_holds_between_its_neighbours_only() {
    let checks = "CHECK: a\nCHECK-NOT: x\nCHECK: b";
    ok(checks, "a\ny\nb\n");
    ok(checks, "x\na\nb\nx\n");
    let r = fail(checks, "a\nx\nb\n");
    assert_has(
        &r,
        "t.txt:2:12: error: CHECK-NOT: excluded string found in input",
    );
    assert_has(&r, "<stdin>:2:1: note: found here");
    // Leading -NOT: from the start; trailing: to the end.
    fail("CHECK-NOT: x\nCHECK: a", "x\na\n");
    ok("CHECK-NOT: x\nCHECK: a", "a\nx\n");
    fail("CHECK: a\nCHECK-NOT: x", "a\nx\n");
    ok("CHECK: a\nCHECK-NOT: x", "x\na\n");
    // A -NOT on the matched line itself: only the text between the two matches counts.
    ok("CHECK: a\nCHECK-NOT: x\nCHECK: b", "xa b\n");
    fail("CHECK: a\nCHECK-NOT: x\nCHECK: b", "a x b\n");
}

#[test]
fn check_dag_matches_in_any_order_without_overlap() {
    ok("CHECK-DAG: a\nCHECK-DAG: b\nCHECK: c", "b\na\nc\n");
    // The next positive directive starts after the furthest -DAG match.
    let r = fail("CHECK-DAG: a\nCHECK-DAG: b\nCHECK: c", "c\na\nb\n");
    assert_has(&r, "t.txt:3:8: error: CHECK: expected string not found");
    // Two -DAGs never claim one occurrence.
    fail("CHECK-DAG: x\nCHECK-DAG: x", "x\n");
    ok("CHECK-DAG: x\nCHECK-DAG: x", "x\nx\n");
    // ... nor overlapping text: `ab` and `bc` cannot share the `b`.
    fail("CHECK-DAG: ab\nCHECK-DAG: bc", "abc\n");
    ok("CHECK-DAG: ab\nCHECK-DAG: bc", "abc bc\n");
    let missing = fail("CHECK: start\nCHECK-DAG: q", "start\na\n");
    assert_has(&missing, "CHECK-DAG: expected string not found in input");
    assert_has(&missing, "<stdin>:1:6: note: scanning from here");
}

#[test]
fn check_not_between_dag_runs_separates_them() {
    let checks = "CHECK-DAG: a\nCHECK-NOT: z\nCHECK-DAG: b";
    ok(checks, "a\nb\nz\n");
    let r = fail(checks, "a\nz\nb\n");
    assert_has(&r, "CHECK-NOT: excluded string found in input");
    // The runs keep their order: `b` must come after `a`'s run.
    fail(checks, "b\na\n");
}

#[test]
fn check_label_partitions_the_input() {
    let input = "fn f\n  x\nfn g\n  y\n";
    ok(
        "CHECK-LABEL: fn f\nCHECK: x\nCHECK-LABEL: fn g\nCHECK: y",
        input,
    );
    // `y` exists, but not in f's region.
    let r = fail("CHECK-LABEL: fn f\nCHECK: y\nCHECK-LABEL: fn g", input);
    assert_has(
        &r,
        "t.txt:2:8: error: CHECK: expected string not found in input",
    );
    // A -NOT is confined to its region too.
    ok(
        "CHECK-LABEL: fn f\nCHECK-NOT: y\nCHECK-LABEL: fn g\nCHECK: y",
        input,
    );
    fail("CHECK-LABEL: fn g\nCHECK-NOT: y", input);
    // A missing label stops everything, naming the label.
    let r = fail("CHECK-LABEL: fn h\nCHECK: x", input);
    assert_has(
        &r,
        "t.txt:1:14: error: CHECK-LABEL: expected string not found in input",
    );
    // A -NEXT right after a label is relative to the label's line.
    ok("CHECK-LABEL: fn g\nCHECK-NEXT: y", input);
    fail("CHECK-LABEL: fn f\nCHECK-NEXT: fn g", input);
}

#[test]
fn a_failed_region_does_not_stop_the_others() {
    let input = "fn f\n  x\nfn g\n  y\n";
    let r = fail(
        "CHECK-LABEL: fn f\nCHECK: q1\nCHECK-LABEL: fn g\nCHECK: q2",
        input,
    );
    assert_has(&r, "t.txt:2:8:");
    assert_has(&r, "t.txt:4:8:");
}

// ---------------------------------------------------------------------------------------------
// Patterns: regexes, variables, @LINE, whitespace.
// ---------------------------------------------------------------------------------------------

#[test]
fn regex_pieces_in_a_pattern() {
    ok("CHECK: v{{[0-9]+}} = add", "  v12 = add v1, v2\n");
    ok("CHECK: {{fadd|fmul}} v1", "v3 = fmul v1, v2\n");
    // The alternation stays inside its braces: `abc{{x|z}}def` is not `abcx|zdef`.
    fail("CHECK: abc{{x|z}}def", "abcx\n");
    ok("CHECK: {{a{2}}}b", "aab\n");
    // A pattern never matches across a line break.
    fail("CHECK: a{{.*}}b", "a\nb\n");
    // Literal regex metacharacters outside braces are literal.
    ok("CHECK: [2 x f32] (v1)", "alloca [2 x f32] (v1)\n");
}

#[test]
fn variables_bind_and_substitute() {
    let checks = "CHECK: [[X:v[0-9]+]] = const\nCHECK: ret [[X]]";
    ok(checks, "v3 = const\nret v3\n");
    let r = fail(checks, "v3 = const\nret v4\n");
    assert_has(&r, "note: with \"X\" equal to \"v3\"");
    // Brackets inside the regex nest: `[[V:[a-z]+]]` ends after the `+`.
    ok("CHECK: [[V:[a-z]+]]=[[V]]", "ab=ab\n");
    // A use after a binding in the same pattern refers to that binding's text.
    ok("CHECK: [[A:[a-z]+]]-[[A]]", "foo-bar foo-foo\n");
    // A later binding replaces an earlier one.
    ok(
        "CHECK: [[X:v[0-9]+]] = a\nCHECK: [[X:v[0-9]+]] = b\nCHECK: use [[X]]",
        "v1 = a\nv2 = b\nuse v2\n",
    );
    // A bound value is literal text, never a regex.
    fail("CHECK: [[X:a.c]]\nCHECK: [[X]]", "a.c\nabc\n");
    let r = fail("CHECK: [[Y]]", "x\n");
    assert_has(&r, "uses undefined variable `Y`");
}

#[test]
fn dag_matches_bind_variables_for_later_directives() {
    ok(
        "CHECK-DAG: [[A:v[0-9]+]] = const 1\nCHECK-DAG: [[B:v[0-9]+]] = const 2\nCHECK: add [[B]], [[A]]",
        "v7 = const 2\nv4 = const 1\nv9 = add v7, v4\n",
    );
}

#[test]
fn line_pseudo_variable_names_the_directive_line() {
    ok(
        "first\nCHECK: at [[@LINE]] after [[@LINE-1]] before [[@LINE+1]]",
        "at 2 after 1 before 3\n",
    );
    let r = fail("CHECK: [[@LINE-5]]", "x\n");
    assert_has(&r, "before the file starts");
    let r = fail("CHECK: [[@FILE]]", "x\n");
    assert_has(&r, "unknown pseudo variable");
}

#[test]
fn blanks_match_any_run_of_blanks_unless_strict() {
    ok("CHECK: a b", "a    b\n");
    ok("CHECK: a b", "a\tb\n");
    fail("CHECK: a b", "ab\n");
    ok("CHECK:     a   ", "a\n");
    let strict = Options {
        strict_whitespace: true,
        ..opts()
    };
    ok_with("CHECK: a b", "a b\n", &strict);
    fail_with("CHECK: a b", "a  b\n", &strict);
}

#[test]
fn crlf_input_is_line_structured() {
    ok("CHECK: a\nCHECK-NEXT: b", "a\r\nb\r\n");
    ok("CHECK: {{a$}}", "a\r\n");
    ok("CHECK: a\r\nCHECK-NEXT: b\r\n", "a\nb\n");
}

// ---------------------------------------------------------------------------------------------
// Options.
// ---------------------------------------------------------------------------------------------

#[test]
fn prefixes_select_directives() {
    let o2 = Options {
        prefixes: vec!["O2".into()],
        ..opts()
    };
    // `CHECK:` lines are not O2's.
    ok_with("CHECK: nowhere\nO2: b", "b\n", &o2);
    let both = Options {
        prefixes: vec!["CHECK".into(), "O2".into()],
        ..opts()
    };
    ok_with("CHECK: a\nO2: b", "a\nb\n", &both);
    fail_with("O2: b\nCHECK: a", "a\nb\n", &both);
    // A prefix must start at a word boundary.
    let r = fail("XCHECK: a\nCHECK: b", "a\n");
    assert_has(&r, "t.txt:2:8:");
    // The longest prefix wins: `CHECK-O2:` is its own prefix, not CHECK plus a suffix.
    let nested = Options {
        prefixes: vec!["CHECK".into(), "CHECK-O2".into()],
        ..opts()
    };
    ok_with("CHECK-O2: a\nCHECK-NEXT: b", "a\nb\n", &nested);
}

#[test]
fn implicit_check_not_applies_everywhere() {
    let o = Options {
        implicit_check_not: vec!["call".into()],
        ..opts()
    };
    let checks = "CHECK: one\nCHECK: two";
    ok_with(checks, "one\ntwo\n", &o);
    let r = fail_with(checks, "one\ncall x\ntwo\n", &o);
    assert_has(
        &r,
        "<implicit-check-not>:1:1: error: IMPLICIT-CHECK-NOT: excluded string found",
    );
    assert_has(&r, "<stdin>:2:1: note: found here");
    fail_with(checks, "call\none\ntwo\n", &o);
    fail_with(checks, "one\ntwo\ncall\n", &o);
    // The positive matches themselves may contain it.
    ok_with("CHECK: one call two", "one call two\n", &o);
}

#[test]
fn empty_input_is_an_error_unless_allowed() {
    let r = fail("CHECK-NOT: x", "");
    assert_has(&r, "the input is empty");
    ok_with(
        "CHECK-NOT: x",
        "",
        &Options {
            allow_empty: true,
            ..opts()
        },
    );
}

// ---------------------------------------------------------------------------------------------
// Reports and check-file errors.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_failed_match_shows_the_directive_and_the_nearest_candidate() {
    let r = fail(
        "CHECK: a\nCHECK: v3 = fmul v1, v2",
        "a\nv3 = fadd v1, v2\nv4 = fmul v1, v9\n",
    );
    assert_has(
        &r,
        "t.txt:2:8: error: CHECK: expected string not found in input",
    );
    assert_has(&r, "CHECK: v3 = fmul v1, v2\n       ^");
    assert_has(&r, "<stdin>:1:2: note: scanning from here");
    // Line 3 differs in two characters, line 2 in three.
    assert_has(
        &r,
        "<stdin>:3:1: note: possible intended match here\nv4 = fmul v1, v9\n^",
    );
}

#[test]
fn a_hopeless_pattern_shows_no_candidate() {
    let r = fail("CHECK: zzzzzzzz", "a\nb\n");
    assert!(!r.contains("possible intended match"), "{r}");
}

#[test]
fn check_file_errors_name_their_line() {
    for (checks, want) in [
        (
            "x\nCHECK: {{abc",
            "t.txt:2:8: error: CHECK: a `{{` regex has no closing `}}`",
        ),
        ("CHECK: [[X", "has no closing `]]`"),
        ("CHECK: {{}}", "an empty `{{}}` regex"),
        ("CHECK: {{a(}}", "unclosed `(`"),
        ("CHECK:", "found an empty pattern"),
        ("CHECK-EMPTY: x", "takes no pattern"),
        ("CHECK-COUNT-0: x", "at least 1"),
        ("CHECK-NEXT: a", "with no positive directive before it"),
        (
            "CHECK-DAG: a\nCHECK-SAME: b",
            "with no positive directive before it",
        ),
        (
            "CHECK-LABEL: [[X:a]]",
            "a label can neither define nor use a variable",
        ),
        (
            "CHECK: [[X:a]]\nCHECK-LABEL: [[X]]",
            "a label can neither define nor use",
        ),
        (
            "CHECK-NOT: [[X:a]]",
            "a -NOT pattern cannot define a variable",
        ),
        ("CHECK: [[X:a]][[X:b]]", "defined twice in one pattern"),
        ("CHECK: [[1X:a]]", "invalid variable name"),
        ("CHECK: [[X:]]", "empty regex"),
        (
            "CHECK: a\n// CHECK-NXT: b",
            "t.txt:2: error: `CHECK-NXT` looks like a `CHECK` directive",
        ),
        (
            "CHECK: a\n; CHECK-NEXT b",
            "`CHECK-NEXT` looks like a `CHECK` directive",
        ),
        (
            "CHECK: a\n// CHECK : b",
            "`CHECK` looks like a `CHECK` directive",
        ),
        (
            "nothing here",
            "no check directives found with prefix `CHECK:`",
        ),
    ] {
        let r = fail(checks, "a\n");
        assert_has(&r, want);
    }
    // Prose that merely mentions a directive mid-sentence is not a typo.
    ok("// The CHECK-NOT below is the point.\nCHECK: a", "a\n");
}

#[test]
fn invalid_prefixes_are_rejected() {
    for prefixes in [vec![], vec!["1A".to_string()], vec!["A".into(), "A".into()]] {
        let o = Options { prefixes, ..opts() };
        check("t.txt", "A: x", "x\n", &o).expect_err("an invalid prefix list");
    }
}
