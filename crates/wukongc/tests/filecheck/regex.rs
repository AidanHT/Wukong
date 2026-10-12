//! A small backtracking regular-expression engine for FileCheck patterns.
//!
//! The workspace locks no `regex` crate and the tree takes no new dependency, so the `{{..}}` and
//! `[[NAME:..]]` pieces of a check pattern are compiled here. The dialect is the POSIX ERE subset
//! LLVM FileCheck patterns are written in, plus the Perl shorthands everyone types anyway:
//!
//! | syntax | meaning |
//! |---|---|
//! | `x`, `\x` (`x` punctuation) | the literal character |
//! | `.` | any character but a newline |
//! | `[abc]` `[a-z]` `[^..]` | a class; `]` first is literal, `-` first or last is literal |
//! | `[[:alpha:]]` and `alnum digit xdigit upper lower space blank punct word` | POSIX classes |
//! | `\d \w \s` / `\D \W \S` | digit, word character, whitespace / their complements |
//! | `\n \t \r` | the control characters |
//! | `^` `$` | the start / end of the line being searched |
//! | `\b` `\B` | a word boundary / not one |
//! | `*` `+` `?` `{n}` `{n,}` `{n,m}` | greedy repetition; a trailing `?` makes it lazy |
//! | `(..)` `(?:..)` | grouping (nothing reads a group's text, so both only group) |
//! | `a\|b` | alternation, leftmost first |
//!
//! Matching is leftmost-first backtracking (Perl semantics, not POSIX leftmost-longest): the first
//! start position that matches wins, and at that position the first alternative and the greediest
//! repetition that let the rest match. A repetition that matches empty stops iterating, so `(a*)*`
//! terminates, and a step budget turns a pathological pattern into an error instead of a hang.
//!
//! A pattern searches one line at a time (the caller hands it a line, possibly cut short at the end
//! of the region being searched), so nothing here ever matches across a newline.

use std::fmt;

/// The most backtracking steps one search may take before it is reported as too expensive.
const STEP_LIMIT: u64 = 20_000_000;

/// The largest count a `{n,m}` repetition may name.
const MAX_REPEAT: u32 = 1000;

/// A regex syntax error, or a search that ran out of budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexError(pub String);

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One class item: an inclusive character range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Class {
    pub negated: bool,
    pub ranges: Vec<(char, char)>,
}

impl Class {
    fn of(ranges: &[(char, char)]) -> Class {
        Class {
            negated: false,
            ranges: ranges.to_vec(),
        }
    }

    pub fn matches(&self, c: char) -> bool {
        let inside = self.ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi);
        // A negated class never matches a newline, as in a POSIX `REG_NEWLINE` regex.
        if self.negated {
            !inside && c != '\n'
        } else {
            inside
        }
    }
}

const DIGIT: &[(char, char)] = &[('0', '9')];
const WORD: &[(char, char)] = &[('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')];
const SPACE: &[(char, char)] = &[('\t', '\r'), (' ', ' ')];

/// A parsed pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    Empty,
    Char(char),
    Any,
    Class(Class),
    LineStart,
    LineEnd,
    WordBoundary(bool),
    Concat(Vec<Node>),
    Alt(Vec<Node>),
    Repeat {
        node: Box<Node>,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    },
    /// A group; `Some(slot)` records the span it matched in capture slot `slot`.
    Group(Option<usize>, Box<Node>),
    /// The text capture slot `slot` matched, again.
    Backref(usize),
}

impl Node {
    /// A node matching `s` literally.
    pub fn literal(s: &str) -> Node {
        Node::Concat(s.chars().map(Node::Char).collect())
    }

    /// One or more blanks (space or tab): what a run of whitespace in check text matches.
    pub fn blanks() -> Node {
        Node::Repeat {
            node: Box::new(Node::Class(Class::of(&[('\t', '\t'), (' ', ' ')]))),
            min: 1,
            max: None,
            greedy: true,
        }
    }
}

/// Parse `src` as a regex. Groups inside it never capture.
pub fn parse(src: &str) -> Result<Node, RegexError> {
    let mut p = Parser {
        s: src.chars().collect(),
        i: 0,
    };
    let node = p.alt()?;
    if p.i < p.s.len() {
        // Only an unbalanced `)` stops `alt` early.
        return Err(RegexError(format!(
            "unmatched `)` at offset {} in regex `{src}`",
            p.i
        )));
    }
    Ok(node)
}

struct Parser {
    s: Vec<char>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn src(&self) -> String {
        self.s.iter().collect()
    }

    fn err(&self, msg: &str) -> RegexError {
        RegexError(format!(
            "{msg} at offset {} in regex `{}`",
            self.i,
            self.src()
        ))
    }

    fn alt(&mut self) -> Result<Node, RegexError> {
        let mut alts = vec![self.concat()?];
        while self.peek() == Some('|') {
            self.i += 1;
            alts.push(self.concat()?);
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        })
    }

    fn concat(&mut self) -> Result<Node, RegexError> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.quantified(atom)?);
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.pop().unwrap(),
            _ => Node::Concat(items),
        })
    }

    fn quantified(&mut self, mut atom: Node) -> Result<Node, RegexError> {
        loop {
            let (min, max) = match self.peek() {
                Some('*') => {
                    self.i += 1;
                    (0, None)
                }
                Some('+') => {
                    self.i += 1;
                    (1, None)
                }
                Some('?') => {
                    self.i += 1;
                    (0, Some(1))
                }
                Some('{') => match self.counted()? {
                    Some(bounds) => bounds,
                    // Not a well-formed `{n,m}`: the `{` is a literal, read by the next atom.
                    None => return Ok(atom),
                },
                _ => return Ok(atom),
            };
            let greedy = if self.peek() == Some('?') {
                self.i += 1;
                false
            } else {
                true
            };
            atom = Node::Repeat {
                node: Box::new(atom),
                min,
                max,
                greedy,
            };
        }
    }

    /// `{n}`, `{n,}` or `{n,m}` at `self.i`, consumed; `None` (nothing consumed) when the brace
    /// does not start one.
    fn counted(&mut self) -> Result<Option<(u32, Option<u32>)>, RegexError> {
        let start = self.i;
        let mut j = self.i + 1;
        let number = |j: &mut usize, s: &[char]| -> Option<u32> {
            let from = *j;
            while *j < s.len() && s[*j].is_ascii_digit() {
                *j += 1;
            }
            if *j == from || *j - from > 6 {
                return None;
            }
            s[from..*j].iter().collect::<String>().parse().ok()
        };
        let Some(min) = number(&mut j, &self.s) else {
            return Ok(None);
        };
        let max = match self.s.get(j) {
            Some('}') => Some(min),
            Some(',') => {
                j += 1;
                if self.s.get(j) == Some(&'}') {
                    None
                } else {
                    let Some(max) = number(&mut j, &self.s) else {
                        return Ok(None);
                    };
                    if self.s.get(j) != Some(&'}') {
                        return Ok(None);
                    }
                    Some(max)
                }
            }
            _ => return Ok(None),
        };
        self.i = j + 1;
        if min > MAX_REPEAT || max.is_some_and(|m| m > MAX_REPEAT) {
            self.i = start;
            return Err(self.err(&format!("repetition count above {MAX_REPEAT}")));
        }
        if max.is_some_and(|m| m < min) {
            self.i = start;
            return Err(self.err("repetition `{n,m}` with m < n"));
        }
        Ok(Some((min, max)))
    }

    fn atom(&mut self) -> Result<Node, RegexError> {
        let c = self.peek().expect("atom at end");
        self.i += 1;
        Ok(match c {
            '(' => {
                if self.s[self.i..].starts_with(&['?', ':']) {
                    self.i += 2;
                }
                let inner = self.alt()?;
                if self.peek() != Some(')') {
                    return Err(self.err("unclosed `(`"));
                }
                self.i += 1;
                Node::Group(None, Box::new(inner))
            }
            '[' => Node::Class(self.class()?),
            '.' => Node::Any,
            '^' => Node::LineStart,
            '$' => Node::LineEnd,
            '\\' => self.escape(false)?,
            '*' | '+' | '?' => {
                self.i -= 1;
                return Err(self.err(&format!("`{c}` has nothing to repeat")));
            }
            // Includes a `{` that does not open a well-formed `{n,m}`: it is a literal.
            _ => Node::Char(c),
        })
    }

    /// The escape after a `\` (already consumed). Inside a class, `\b` is not a boundary.
    fn escape(&mut self, in_class: bool) -> Result<Node, RegexError> {
        let Some(c) = self.peek() else {
            return Err(self.err("trailing `\\`"));
        };
        self.i += 1;
        let class = |ranges: &[(char, char)], negated: bool| {
            Node::Class(Class {
                negated,
                ranges: ranges.to_vec(),
            })
        };
        Ok(match c {
            'd' => class(DIGIT, false),
            'D' => class(DIGIT, true),
            'w' => class(WORD, false),
            'W' => class(WORD, true),
            's' => class(SPACE, false),
            'S' => class(SPACE, true),
            'n' => Node::Char('\n'),
            't' => Node::Char('\t'),
            'r' => Node::Char('\r'),
            'b' if !in_class => Node::WordBoundary(true),
            'B' if !in_class => Node::WordBoundary(false),
            c if c.is_ascii_alphanumeric() => {
                self.i -= 1;
                return Err(self.err(&format!("unsupported escape `\\{c}`")));
            }
            c => Node::Char(c),
        })
    }

    /// A bracket expression after its `[` (already consumed).
    fn class(&mut self) -> Result<Class, RegexError> {
        let mut negated = false;
        if self.peek() == Some('^') {
            negated = true;
            self.i += 1;
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let Some(c) = self.peek() else {
                return Err(self.err("unclosed `[`"));
            };
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            // A POSIX class name: `[:digit:]`.
            if c == '[' && self.s.get(self.i + 1) == Some(&':') {
                let rest: String = self.s[self.i + 2..].iter().collect();
                let Some(end) = rest.find(":]") else {
                    return Err(self.err("unclosed `[:` class name"));
                };
                let name = &rest[..end];
                let set: &[(char, char)] = match name {
                    "alpha" => &[('A', 'Z'), ('a', 'z')],
                    "digit" => DIGIT,
                    "alnum" => &[('0', '9'), ('A', 'Z'), ('a', 'z')],
                    "xdigit" => &[('0', '9'), ('A', 'F'), ('a', 'f')],
                    "upper" => &[('A', 'Z')],
                    "lower" => &[('a', 'z')],
                    "space" => SPACE,
                    "blank" => &[('\t', '\t'), (' ', ' ')],
                    "punct" => &[('!', '/'), (':', '@'), ('[', '`'), ('{', '~')],
                    "word" => WORD,
                    _ => return Err(self.err(&format!("unknown class `[:{name}:]`"))),
                };
                ranges.extend_from_slice(set);
                self.i += 2 + name.chars().count() + 2;
                continue;
            }
            let lo = self.class_char()?;
            let lo = match lo {
                ClassChar::Set(set, false) => {
                    ranges.extend(set);
                    continue;
                }
                ClassChar::Set(..) => {
                    return Err(self.err("a negated shorthand (`\\D`, `\\W`, `\\S`) in a class"))
                }
                ClassChar::One(c) => c,
            };
            // `a-z`, but a `-` right before the closing `]` is a literal.
            if self.peek() == Some('-') && self.s.get(self.i + 1).is_some_and(|&c| c != ']') {
                self.i += 1;
                let ClassChar::One(hi) = self.class_char()? else {
                    return Err(self.err("a class shorthand cannot end a range"));
                };
                if hi < lo {
                    return Err(self.err(&format!("reversed range `{lo}-{hi}`")));
                }
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
        Ok(Class { negated, ranges })
    }

    fn class_char(&mut self) -> Result<ClassChar, RegexError> {
        let c = self.peek().expect("class char at end");
        self.i += 1;
        if c != '\\' {
            return Ok(ClassChar::One(c));
        }
        Ok(match self.escape(true)? {
            Node::Char(c) => ClassChar::One(c),
            Node::Class(cl) => ClassChar::Set(cl.ranges, cl.negated),
            _ => unreachable!("escape inside a class is a char or a class"),
        })
    }
}

enum ClassChar {
    One(char),
    Set(Vec<(char, char)>, bool),
}

/// A successful search: the matched span and every capture slot's span, in char indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub start: usize,
    pub end: usize,
    pub caps: Vec<Option<(usize, usize)>>,
}

/// Search `text[from..]` for the leftmost match of `node`, which uses `slots` capture slots.
/// `text` is one line, or the leading part of one: `$` matches at `text.len()` only when
/// `ends_line`.
pub fn find(
    node: &Node,
    slots: usize,
    text: &[char],
    from: usize,
    ends_line: bool,
) -> Result<Option<Found>, RegexError> {
    let mut m = Matcher {
        text,
        ends_line,
        caps: vec![None; slots],
        steps: 0,
        exhausted: false,
    };
    for start in from..=text.len() {
        m.caps.iter_mut().for_each(|c| *c = None);
        let mut end = None;
        let hit = m.m(node, start, &mut |_, j| {
            end = Some(j);
            true
        });
        if m.exhausted {
            return Err(RegexError(format!(
                "the pattern took more than {STEP_LIMIT} backtracking steps on one line; \
                 simplify it"
            )));
        }
        if hit {
            return Ok(Some(Found {
                start,
                end: end.expect("a match records its end"),
                caps: m.caps,
            }));
        }
    }
    Ok(None)
}

type Cont<'k, 't> = dyn FnMut(&mut Matcher<'t>, usize) -> bool + 'k;

struct Matcher<'t> {
    text: &'t [char],
    ends_line: bool,
    caps: Vec<Option<(usize, usize)>>,
    steps: u64,
    exhausted: bool,
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

impl<'t> Matcher<'t> {
    /// Match `n` at `i`, then the continuation `k` from wherever `n` ended. True as soon as one
    /// way through both succeeds; every capture a failed path wrote is restored.
    fn m(&mut self, n: &Node, i: usize, k: &mut Cont<'_, 't>) -> bool {
        self.steps += 1;
        if self.steps > STEP_LIMIT {
            self.exhausted = true;
        }
        if self.exhausted {
            return false;
        }
        let len = self.text.len();
        match n {
            Node::Empty => k(self, i),
            Node::Char(c) => i < len && self.text[i] == *c && k(self, i + 1),
            Node::Any => i < len && self.text[i] != '\n' && k(self, i + 1),
            Node::Class(cl) => i < len && cl.matches(self.text[i]) && k(self, i + 1),
            Node::LineStart => i == 0 && k(self, i),
            Node::LineEnd => i == len && self.ends_line && k(self, i),
            Node::WordBoundary(want) => {
                let before = i > 0 && is_word(self.text[i - 1]);
                let after = i < len && is_word(self.text[i]);
                (before != after) == *want && k(self, i)
            }
            Node::Concat(nodes) => self.seq(nodes, i, k),
            Node::Alt(alts) => {
                for a in alts {
                    if self.m(a, i, k) {
                        return true;
                    }
                    if self.exhausted {
                        return false;
                    }
                }
                false
            }
            Node::Repeat {
                node,
                min,
                max,
                greedy,
            } => self.rep(node, *min, *max, *greedy, 0, i, k),
            Node::Group(None, inner) => self.m(inner, i, k),
            Node::Group(Some(slot), inner) => {
                let slot = *slot;
                self.m(inner, i, &mut |s: &mut Matcher<'t>, j| {
                    let prev = s.caps[slot];
                    s.caps[slot] = Some((i, j));
                    if k(s, j) {
                        true
                    } else {
                        s.caps[slot] = prev;
                        false
                    }
                })
            }
            Node::Backref(slot) => {
                let Some((a, b)) = self.caps[*slot] else {
                    return false;
                };
                let w = b - a;
                i + w <= len && self.text[a..b] == self.text[i..i + w] && k(self, i + w)
            }
        }
    }

    fn seq(&mut self, nodes: &[Node], i: usize, k: &mut Cont<'_, 't>) -> bool {
        match nodes.split_first() {
            None => k(self, i),
            Some((first, rest)) => {
                self.m(first, i, &mut |s: &mut Matcher<'t>, j| s.seq(rest, j, k))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rep(
        &mut self,
        node: &Node,
        min: u32,
        max: Option<u32>,
        greedy: bool,
        count: u32,
        i: usize,
        k: &mut Cont<'_, 't>,
    ) -> bool {
        let can_more = max.is_none_or(|m| count < m);
        let can_stop = count >= min;
        let one_more = |s: &mut Matcher<'t>, k: &mut Cont<'_, 't>| {
            s.m(node, i, &mut |s2: &mut Matcher<'t>, j| {
                // An iteration that matched nothing cannot make progress: once the minimum is
                // met, refuse it rather than loop.
                if j == i && count >= min {
                    return false;
                }
                s2.rep(node, min, max, greedy, count + 1, j, k)
            })
        };
        if greedy {
            (can_more && one_more(self, k)) || (can_stop && !self.exhausted && k(self, i))
        } else {
            (can_stop && k(self, i)) || (can_more && !self.exhausted && one_more(self, k))
        }
    }
}
