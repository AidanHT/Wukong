//! Rustc-style terminal rendering of [`Diagnostic`]s.
//!
//! The labels of a diagnostic are drawn in one frame per source file, a source line at a time:
//! every line that carries a label is printed once, with all of its labels beneath it. Single-line
//! labels on one line share an underline row (`^` for the primary, `-` for a secondary); the
//! rightmost one's message ends that row, and the others hang below on vertical bars, the rightmost
//! first (see [`single_line_rows`]). Every multi-line label is drawn whole, with connectors in a
//! left margin that has one column per concurrently open multi-line label (see
//! [`assign_connector_columns`]): `/` or a start marker where it opens, `|` down its lines, and an
//! end marker under its last character carrying its message. The lines between two labelled lines
//! are printed only inside a connector, and a long run of them is elided around `...`.

use std::cmp::Reverse;
use std::collections::BTreeSet;

use crate::{Diagnostic, NoteKind};
use wukong_span::SourceMap;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const BLUE: &str = "\x1b[1;34m"; // gutter / frame color

/// One character cell of an annotation row, with the ANSI color it is painted in (`None`: plain).
type Cell = (char, Option<&'static str>);

/// Renders diagnostics to strings. `color` toggles ANSI escapes (off for tests and pipes).
pub struct Renderer {
    pub color: bool,
}

impl Renderer {
    pub fn new(color: bool) -> Renderer {
        Renderer { color }
    }

    fn paint(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("{code}{s}{RESET}")
        } else {
            s.to_string()
        }
    }

    /// `cells` as text, each run of one color painted once.
    fn paint_cells(&self, cells: &[Cell]) -> String {
        let mut out = String::new();
        let mut i = 0;
        while i < cells.len() {
            let style = cells[i].1;
            let mut run = String::new();
            while i < cells.len() && cells[i].1 == style {
                run.push(cells[i].0);
                i += 1;
            }
            match style {
                Some(code) => out.push_str(&self.paint(code, &run)),
                None => out.push_str(&run),
            }
        }
        out
    }

    /// Render a single diagnostic to a multi-line string (no trailing newline).
    pub fn render(&self, d: &Diagnostic, sm: &SourceMap) -> String {
        let mut out = String::new();

        // Header: `error[E0501]: message`
        let head = match d.code {
            Some(code) => format!("{}[{}]", d.severity.header(), code),
            None => d.severity.header().to_string(),
        };
        out.push_str(&self.paint(d.severity.ansi(), &head));
        out.push_str(&self.paint(BOLD, &format!(": {}", d.message)));

        // The labels, sorted by position and grouped into one frame per source file.
        let mut labels: Vec<&crate::Label> =
            d.labels.iter().filter(|l| !l.span.is_dummy()).collect();
        labels.sort_by_key(|l| (l.span.source.0, l.span.lo));
        let mut files: Vec<FileFrame> = Vec::new();
        for l in labels {
            let ann = annotate(l, sm);
            match files.last_mut() {
                Some(f) if f.source == l.span.source => f.anns.push(ann),
                _ => files.push(FileFrame {
                    source: l.span.source,
                    anns: vec![ann],
                    columns: 0,
                    lines: Vec::new(),
                }),
            }
        }
        for f in &mut files {
            f.columns = assign_connector_columns(&mut f.anns);
            f.lines = frame_lines(&f.anns);
        }
        // One connector column per concurrently open multi-line label, plus a space before the
        // code, shared by every frame so all the echoed source text stays aligned; none at all
        // when no label spans lines, which keeps the single-line form exactly as it always was.
        let columns = files.iter().map(|f| f.columns).max().unwrap_or(0);
        let margin = if columns > 0 { columns + 1 } else { 0 };

        // Gutter width is driven by the largest line number printed, and room for `...` when a
        // frame elides lines.
        let mut width = 1;
        for line in files.iter().flat_map(|f| &f.lines) {
            width = width.max(match line {
                Some(n) => n.to_string().len(),
                None => ELLIPSIS.len(),
            });
        }
        let pad = " ".repeat(width);
        let bar = self.paint(BLUE, "|");

        // `--> file:line:col` for the primary span.
        if let Some(span) = d.primary_span() {
            if !span.is_dummy() {
                let loc = sm.span_location(span);
                let arrow = self.paint(BLUE, "-->");
                out.push('\n');
                out.push_str(&format!(
                    "{pad}{arrow} {}:{}:{}",
                    sm.name(span.source),
                    loc.line,
                    loc.col
                ));
            }
        }

        // A frame in a *different file* than the one currently shown gets a `::: file:line:col`
        // header first, so a secondary label in an imported file is never printed under the root
        // file's `-->` arrow with only its own line number (the long-standing cross-file bug: E0300
        // duplicate-across-imports showed the second definition's line with no hint it lived in
        // another file).
        if !files.is_empty() {
            // The `-->` arrow established the primary span's file as the current context.
            let mut cur_source = d.primary_span().map(|s| s.source);
            out.push('\n');
            out.push_str(&format!("{pad} {bar}"));
            for f in &files {
                if Some(f.source) != cur_source {
                    cur_source = Some(f.source);
                    let loc = sm.span_location(f.anns[0].label.span);
                    let colons = self.paint(BLUE, ":::");
                    out.push('\n');
                    out.push_str(&format!(
                        "{pad} {colons} {}:{}:{}",
                        sm.name(f.source),
                        loc.line,
                        loc.col
                    ));
                    out.push('\n');
                    out.push_str(&format!("{pad} {bar}"));
                }
                let frame = Frame {
                    renderer: self,
                    sm,
                    width,
                    bar: &bar,
                    columns,
                    margin,
                };
                frame.render_file(&mut out, f);
            }
            out.push('\n');
            out.push_str(&format!("{pad} {bar}"));
        }

        // Trailing notes / helps: `= note: ...`
        for (kind, text) in &d.notes {
            let tag = match kind {
                NoteKind::Note => "note",
                NoteKind::Help => "help",
            };
            out.push('\n');
            let eq = self.paint(BLUE, "=");
            out.push_str(&format!("{pad} {eq} {}: {text}", self.paint(BOLD, tag)));
        }

        // Structured suggestions: a `help:` line carrying the message and, when the fix is a
        // concrete one-line replacement, the replacement text itself in backticks. The precise
        // span+replacement travels in the JSON stream (see `to_json`) for an editor to apply; the
        // terminal form stays compact and never tries to splice a multi-line edit into one line.
        for s in &d.suggestions {
            out.push('\n');
            let eq = self.paint(BLUE, "=");
            let tag = self.paint(BOLD, "help");
            let single_line = !s.replacement.contains('\n');
            if single_line && !s.replacement.is_empty() {
                out.push_str(&format!(
                    "{pad} {eq} {tag}: {}: `{}`",
                    s.message, s.replacement
                ));
            } else if single_line && s.replacement.is_empty() {
                // A deletion: there is no text to show, so name the action.
                out.push_str(&format!("{pad} {eq} {tag}: {} (remove it)", s.message));
            } else {
                out.push_str(&format!("{pad} {eq} {tag}: {}", s.message));
            }
        }

        out
    }
}

/// The labels of one source file, laid out for its frame.
struct FileFrame<'a> {
    source: wukong_span::SourceId,
    /// Sorted by position.
    anns: Vec<Ann<'a>>,
    /// How many connector columns this file's multi-line labels need.
    columns: usize,
    /// The rows of source the frame echoes, in order: a line number, or `None` for the `...` that
    /// stands for elided lines (see [`frame_lines`]).
    lines: Vec<Option<u32>>,
}

/// One label placed in its file: its lines, and its columns in *rendered* terms (a tab counts
/// [`TAB_WIDTH`], a wide character 2), 0-based from the start of the echoed source text.
struct Ann<'a> {
    label: &'a crate::Label,
    /// The line of its first character, and of its last (see [`last_char_location`]). They differ
    /// exactly for a multi-line label.
    first: u32,
    last: u32,
    /// The column of its first character on `first`.
    start: usize,
    /// A single-line label: one past the last column of its underline. A multi-line label: the
    /// column of its last character on `last`.
    end: usize,
    /// A multi-line label that only whitespace precedes on `first`: its code row opens with `/`
    /// instead of a start-marker row.
    opens_line: bool,
    /// A multi-line label's connector column in the margin (see [`assign_connector_columns`]).
    column: usize,
}

impl Ann<'_> {
    fn multi_line(&self) -> bool {
        self.last > self.first
    }

    /// `^` and red for the primary label, `-` and blue for a secondary one.
    fn mark(&self) -> (char, &'static str) {
        if self.label.primary {
            ('^', span_color(true))
        } else {
            ('-', span_color(false))
        }
    }

    /// Is this multi-line label's connector drawn in the margin of the rows *below* code line `n`
    /// (before any end or start marker of that line)? From the line after it opens, or from its
    /// own line when it opens with `/`, through its last line, whose end marker closes it.
    fn open_below(&self, n: u32) -> bool {
        self.multi_line()
            && ((self.first < n && n <= self.last) || (self.first == n && self.opens_line))
    }
}

/// Place `l` in its file. A single-line label's underline is the rendered width of its span on its
/// line, at least one column; one whose span runs into its line's terminator (its last character
/// is the `\n`) is underlined to one column past the line's text.
fn annotate<'a>(l: &'a crate::Label, sm: &SourceMap) -> Ann<'a> {
    let span = l.span;
    let lo = sm.span_location(span);
    let last = last_char_location(sm, span);
    let text = sm.line_text(span.source, lo.line);
    let before = (lo.col - 1) as usize;
    let start = rendered_width(text, 0, before);
    if last.line > lo.line {
        let end_text = sm.line_text(span.source, last.line);
        return Ann {
            label: l,
            first: lo.line,
            last: last.line,
            start,
            end: rendered_width(end_text, 0, (last.col - 1) as usize),
            opens_line: text.chars().take(before).all(char::is_whitespace),
            column: 0,
        };
    }
    let hi = sm.location(span.source, span.hi);
    let len = if hi.line == lo.line {
        rendered_width(text, before, (hi.col - 1) as usize)
    } else {
        rendered_width(text, before, text.chars().count() + 1)
    };
    Ann {
        label: l,
        first: lo.line,
        last: lo.line,
        start,
        end: start + len.max(1),
        opens_line: false,
        column: 0,
    }
}

/// Give every multi-line label a connector column: the leftmost one no label already there shares
/// a line with. Labels are taken in order of their first line, then column, then the longer first,
/// so an enclosing label sits left of the labels it encloses, and two labels on disjoint lines
/// share a column. Returns how many columns the file needs.
fn assign_connector_columns(anns: &mut [Ann]) -> usize {
    let mut order: Vec<usize> = (0..anns.len()).filter(|&i| anns[i].multi_line()).collect();
    order.sort_by_key(|&i| (anns[i].first, anns[i].start, Reverse(anns[i].last)));
    let mut taken: Vec<Vec<(u32, u32)>> = Vec::new();
    for i in order {
        let (first, last) = (anns[i].first, anns[i].last);
        let free = |lines: &Vec<(u32, u32)>| lines.iter().all(|&(f, l)| l < first || last < f);
        let column = match taken.iter().position(free) {
            Some(c) => c,
            None => {
                taken.push(Vec::new());
                taken.len() - 1
            }
        };
        taken[column].push((first, last));
        anns[i].column = column;
    }
    taken.len()
}

/// The source rows a file's frame echoes, in order: every line a label starts or ends on, and the
/// lines between two of them when a multi-line label spans the whole gap. A gap of more than
/// `MAX_CONNECTOR_LINES - 2` lines keeps its first and last line around one `...` (`None`), so a
/// lone multi-line label of more than [`MAX_CONNECTOR_LINES`] lines prints its first two and last
/// two. Lines between labels that no connector spans are not printed; the line numbers show the
/// jump.
fn frame_lines(anns: &[Ann]) -> Vec<Option<u32>> {
    let anchors: Vec<u32> = anns
        .iter()
        .flat_map(|a| [a.first, a.last])
        .collect::<BTreeSet<u32>>()
        .into_iter()
        .collect();
    let mut lines = Vec::new();
    for (k, &a) in anchors.iter().enumerate() {
        lines.push(Some(a));
        let Some(&b) = anchors.get(k + 1) else {
            break;
        };
        let spanned = anns
            .iter()
            .any(|x| x.multi_line() && x.first <= a && b <= x.last);
        if !spanned || b == a + 1 {
            continue;
        }
        if elides(a, b) {
            lines.extend([Some(a + 1), None, Some(b - 1)]);
        } else {
            lines.extend((a + 1..b).map(Some));
        }
    }
    lines
}

/// Everything a frame's rows share.
struct Frame<'r> {
    renderer: &'r Renderer,
    sm: &'r SourceMap,
    width: usize,
    bar: &'r str,
    /// Connector columns in the margin (shared by every file's frame).
    columns: usize,
    /// The margin's width: the connector columns plus one space, or 0 without connectors.
    margin: usize,
}

impl Frame<'_> {
    /// Echo every row of `f`'s frame, each labelled line followed by its labels' rows.
    fn render_file(&self, out: &mut String, f: &FileFrame) {
        let mut prev = 0;
        for &line in &f.lines {
            let Some(n) = line else {
                // The elided lines: the gutter shows `...` and the margin the connectors that run
                // through them.
                let mut cells = self.margin_cells(|c| {
                    f.anns
                        .iter()
                        .find(|a| {
                            a.column == c && a.multi_line() && a.first <= prev && prev < a.last
                        })
                        .map(|a| ('|', a.mark().1))
                });
                trim_end(&mut cells);
                let dots = self
                    .renderer
                    .paint(BLUE, &format!("{ELLIPSIS:<width$}", width = self.width));
                out.push('\n');
                out.push_str(&format!(
                    "{dots} {} {}",
                    self.bar,
                    self.renderer.paint_cells(&cells)
                ));
                // `trim_end` cannot remove the space before an empty margin; do it here.
                if cells.is_empty() {
                    out.pop();
                }
                continue;
            };
            prev = n;
            self.render_line(out, f, n);
        }
    }

    /// Code line `n` of `f`'s file and the rows of every label on it.
    fn render_line(&self, out: &mut String, f: &FileFrame, n: u32) {
        let text = self.sm.line_text(f.source, n);
        let multi = || f.anns.iter().filter(|a| a.multi_line());
        // The code row's margin: `/` where a connector opens on this line's text, `|` where one
        // runs through it.
        // (Labels on disjoint lines share a column, so look for the one on this line.)
        let margin = self.margin_cells(|c| {
            multi().filter(|a| a.column == c).find_map(|a| {
                if a.first == n && a.opens_line {
                    Some(('/', a.mark().1))
                } else if a.first < n && n <= a.last {
                    Some(('|', a.mark().1))
                } else {
                    None
                }
            })
        });
        let num = self
            .renderer
            .paint(BLUE, &format!("{n:>width$}", width = self.width));
        out.push('\n');
        out.push_str(&format!(
            "{num} {} {}{}",
            self.bar,
            self.renderer.paint_cells(&margin),
            expand_tabs(text)
        ));

        // The connectors drawn in the margin of the rows below, by column (the color of each).
        let mut open: Vec<Option<&'static str>> = (0..self.columns)
            .map(|c| {
                multi()
                    .find(|a| a.column == c && a.open_below(n))
                    .map(|a| a.mark().1)
            })
            .collect();

        // The single-line labels: one underline row and their hanging messages.
        let singles: Vec<&Ann> = f
            .anns
            .iter()
            .filter(|a| !a.multi_line() && a.first == n)
            .collect();
        if !singles.is_empty() {
            for row in single_line_rows(&singles) {
                let mut cells = self.open_margin(&open);
                cells.extend(row);
                self.push_row(out, cells);
            }
        }

        // The end markers of the connectors that close on this line, innermost (rightmost)
        // first, so each runs left past the columns still open: `|____^ message`.
        let mut ending: Vec<&Ann> = multi().filter(|a| a.last == n).collect();
        ending.sort_by_key(|a| Reverse(a.column));
        for a in ending {
            let (mark, color) = a.mark();
            let mut cells = self.open_margin(&open);
            cells.truncate(a.column);
            cells.push(('|', Some(color)));
            put_run(&mut cells, self.margin + a.end, '_', color);
            cells.push((mark, Some(color)));
            if !a.label.message.is_empty() {
                cells.push((' ', Some(color)));
                cells.extend(a.label.message.chars().map(|ch| (ch, Some(color))));
            }
            open[a.column] = None;
            self.push_row(out, cells);
        }

        // The start markers of the connectors that open mid-line, outermost (leftmost) first:
        // ` ____^` running from the connector's column to the label's first character.
        let mut starting: Vec<&Ann> = multi().filter(|a| a.first == n && !a.opens_line).collect();
        starting.sort_by_key(|a| a.column);
        for a in starting {
            let (mark, color) = a.mark();
            let mut cells = self.open_margin(&open);
            cells.truncate(a.column);
            cells.push((' ', None));
            put_run(&mut cells, self.margin + a.start, '_', color);
            cells.push((mark, Some(color)));
            open[a.column] = Some(color);
            self.push_row(out, cells);
        }
    }

    /// A margin of `columns` cells, each `cell(column)` or blank, then the space before the code;
    /// empty without connectors.
    fn margin_cells(&self, cell: impl Fn(usize) -> Option<(char, &'static str)>) -> Vec<Cell> {
        let mut cells: Vec<Cell> = (0..self.columns)
            .map(|c| match cell(c) {
                Some((ch, color)) => (ch, Some(color)),
                None => (' ', None),
            })
            .collect();
        if self.margin > 0 {
            cells.push((' ', None));
        }
        cells
    }

    /// The margin of an annotation row: `|` in each open connector's column.
    fn open_margin(&self, open: &[Option<&'static str>]) -> Vec<Cell> {
        self.margin_cells(|c| open[c].map(|color| ('|', color)))
    }

    /// One annotation row under the gutter, its trailing blanks dropped.
    fn push_row(&self, out: &mut String, mut cells: Vec<Cell>) {
        trim_end(&mut cells);
        out.push('\n');
        out.push_str(&format!(
            "{} {} {}",
            " ".repeat(self.width),
            self.bar,
            self.renderer.paint_cells(&cells)
        ));
    }
}

/// The rows under one source line for its single-line labels: one underline row, then the
/// messages that do not fit on it, each hanging off a vertical bar at its label's first column.
///
/// ```text
///   |     --- -  -  ^ the rightmost label's message ends the underline row
///   |     |   |  |
///   |     |   |  the next one to the left
///   |     |   and the next
///   |     the leftmost
/// ```
///
/// Underlines are drawn widest first and a primary over a secondary of the same width, so a label
/// nested inside another stays visible inside it. The rightmost label (by first column, then last)
/// puts its message on the underline row when no underline reaches past its own; every other
/// label with a message hangs, and so does that one when one does. A label without a message only
/// underlines.
fn single_line_rows(anns: &[&Ann]) -> Vec<Vec<Cell>> {
    let max_end = anns.iter().map(|a| a.end).max().unwrap_or(0);
    let mut underline: Vec<Cell> = vec![(' ', None); max_end];
    let mut draw = anns.to_vec();
    draw.sort_by_key(|a| (Reverse(a.end - a.start), a.label.primary));
    for a in draw {
        let (mark, color) = a.mark();
        for cell in &mut underline[a.start..a.end] {
            *cell = (mark, Some(color));
        }
    }
    let mut by_position = anns.to_vec();
    by_position.sort_by_key(|a| (a.start, a.end, a.label.primary));
    let rightmost = by_position[by_position.len() - 1];
    let inline = !rightmost.label.message.is_empty() && rightmost.end == max_end;
    if inline {
        let color = Some(rightmost.mark().1);
        underline.push((' ', color));
        underline.extend(rightmost.label.message.chars().map(|ch| (ch, color)));
    }
    let mut rows = vec![underline];

    // Right to left, so each message's row has only bars to its left.
    let hanging: Vec<&Ann> = by_position
        .iter()
        .rev()
        .copied()
        .filter(|a| !a.label.message.is_empty() && !(inline && std::ptr::eq(*a, rightmost)))
        .collect();
    if hanging.is_empty() {
        return rows;
    }
    let bars = |anns: &[&Ann]| {
        let mut row: Vec<Cell> = Vec::new();
        for a in anns {
            put(&mut row, a.start, ('|', Some(a.mark().1)));
        }
        row
    };
    rows.push(bars(&hanging));
    for (i, a) in hanging.iter().enumerate() {
        let mut row = bars(&hanging[i + 1..]);
        let color = Some(a.mark().1);
        for (k, ch) in a.label.message.chars().enumerate() {
            put(&mut row, a.start + k, (ch, color));
        }
        rows.push(row);
    }
    rows
}

/// Set `row[col]`, padding the row with blanks to reach it.
fn put(row: &mut Vec<Cell>, col: usize, cell: Cell) {
    if row.len() <= col {
        row.resize(col + 1, (' ', None));
    }
    row[col] = cell;
}

/// Fill `row` with `ch` from its current end up to (not including) column `to`.
fn put_run(row: &mut Vec<Cell>, to: usize, ch: char, color: &'static str) {
    while row.len() < to {
        row.push((ch, Some(color)));
    }
}

/// Drop trailing blank cells.
fn trim_end(cells: &mut Vec<Cell>) {
    while cells.last().is_some_and(|c| c.0 == ' ') {
        cells.pop();
    }
}

/// The most lines a lone multi-line label prints in full; a longer one keeps its first two and
/// last two lines around one [`ELLIPSIS`] line.
const MAX_CONNECTOR_LINES: u32 = 6;

/// The gutter text of the line that stands for a connector frame's elided middle.
const ELLIPSIS: &str = "...";

/// Whether the gap between labelled lines `first` and `last`, inside a connector, elides its
/// middle: the frame over `first..=last` would print more than [`MAX_CONNECTOR_LINES`] lines.
fn elides(first: u32, last: u32) -> bool {
    last - first + 1 > MAX_CONNECTOR_LINES
}

/// The location of a span's *last* character: the char just before its exclusive end `hi`, or
/// `lo` for an empty span. A span whose `hi` is column 1 of the next line (one that includes the
/// line terminator) therefore ends on the line before. Steps back to a char boundary, so a span
/// ending in a multi-byte character is located at that character's first byte.
fn last_char_location(sm: &SourceMap, span: wukong_span::Span) -> wukong_span::Location {
    let src = sm.source(span.source);
    let mut end = span.hi.saturating_sub(1).max(span.lo);
    while end > span.lo && !src.is_char_boundary(end as usize) {
        end -= 1;
    }
    sm.location(span.source, end)
}

/// How many columns a tab occupies when a source line is rendered.
///
/// The line and the caret padding beneath it must agree on this. Emitting a raw `\t` would let the
/// reader's terminal pick the width instead, which the padding cannot know — so both sides expand
/// tabs here, as rustc does.
const TAB_WIDTH: usize = 4;

/// Rendered width of the chars of `s` in the 0-based char range `[start, end)`, counting a tab as
/// [`TAB_WIDTH`] columns.
///
/// Columns past the end of `s` count one each, so a span that reaches the stripped line terminator
/// pads exactly as it did before tabs were expanded.
fn rendered_width(s: &str, start: usize, end: usize) -> usize {
    let mut width = 0usize;
    let mut i = 0usize;
    for c in s.chars() {
        if i >= end {
            break;
        }
        if i >= start {
            width += if c == '\t' {
                TAB_WIDTH
            } else {
                char_display_width(c)
            };
        }
        i += 1;
    }
    width + end.saturating_sub(i.max(start))
}

/// Terminal cell width of a single character: 2 for East-Asian wide and fullwidth forms (and the
/// common wide emoji blocks), 1 otherwise. A tab is handled separately by the caller.
///
/// This is a small dependency-free table — enough to keep carets aligned under CJK text or an emoji
/// in a comment or string, which is the drift the char-count column measure produced. It is not a
/// full Unicode width implementation; zero-width combining marks are counted as 1 (rare in source).
fn char_display_width(c: char) -> usize {
    let u = c as u32;
    let wide = matches!(u,
        0x1100..=0x115F       // Hangul Jamo
        | 0x2E80..=0x303E     // CJK radicals, Kangxi, CJK symbols/punctuation
        | 0x3041..=0x33FF     // Hiragana, Katakana, CJK compat
        | 0x3400..=0x4DBF     // CJK Ext A
        | 0x4E00..=0x9FFF     // CJK Unified Ideographs
        | 0xA000..=0xA4CF     // Yi
        | 0xAC00..=0xD7A3     // Hangul Syllables
        | 0xF900..=0xFAFF     // CJK Compatibility Ideographs
        | 0xFE30..=0xFE4F     // CJK Compatibility Forms
        | 0xFF00..=0xFF60     // Fullwidth Forms
        | 0xFFE0..=0xFFE6     // Fullwidth signs
        | 0x1F300..=0x1FAFF   // emoji & pictographs
        | 0x20000..=0x3FFFD   // CJK Ext B+
    );
    if wide {
        2
    } else {
        1
    }
}

/// `s` with each tab replaced by [`TAB_WIDTH`] spaces, so it lines up with [`rendered_width`].
fn expand_tabs(s: &str) -> String {
    if !s.contains('\t') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + TAB_WIDTH);
    for c in s.chars() {
        if c == '\t' {
            for _ in 0..TAB_WIDTH {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn span_color(primary: bool) -> &'static str {
    if primary {
        "\x1b[1;31m" // red for primary
    } else {
        "\x1b[1;34m" // blue for secondary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Diagnostic;
    use wukong_span::{SourceMap, Span};

    fn sm_with(src: &str) -> (SourceMap, wukong_span::SourceId) {
        let mut sm = SourceMap::new();
        let id = sm.add("test.wk", src);
        (sm, id)
    }

    #[test]
    fn renders_header_location_and_caret() {
        let (sm, id) = sm_with("fn main() {\n    matmul(a, b);\n}\n");
        // underline "matmul" on line 2 (cols 5..11)
        let span = Span::new(id, 16, 22);
        let d = Diagnostic::error("shape mismatch")
            .with_code("E0501")
            .primary(span, "K must equal P")
            .help("transpose b");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0501]: shape mismatch
 --> test.wk:2:5
  |
2 |     matmul(a, b);
  |     ^^^^^^ K must equal P
  |
  = help: transpose b";
        assert_eq!(r, expected);
    }

    #[test]
    fn tab_indented_line_aligns_with_its_caret() {
        // Two leading tabs. The rendered source line and the caret padding beneath it must agree
        // on how wide a tab is; printing a raw `\t` and padding with one space per column puts the
        // carets (tabstop - 1) columns left of the construct they are supposed to mark.
        let (sm, id) = sm_with("fn main() {\n\t\tmatmul(a, b);\n}\n");
        let span = Span::new(id, 14, 20); // "matmul" on line 2
        let d = Diagnostic::error("shape mismatch")
            .with_code("E0501")
            .primary(span, "K must equal P");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0501]: shape mismatch
 --> test.wk:2:3
  |
2 |         matmul(a, b);
  |         ^^^^^^ K must equal P
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn renders_without_labels() {
        let (sm, _id) = sm_with("x");
        let d = Diagnostic::error("could not find input file").note("check the path");
        let r = Renderer::new(false).render(&d, &sm);
        assert_eq!(
            r,
            "error: could not find input file\n  = note: check the path"
        );
    }

    #[test]
    fn secondary_label_in_another_file_gets_a_colon_header() {
        let mut sm = SourceMap::new();
        let a = sm.add("a.wk", "aaa\n");
        let b = sm.add("b.wk", "bbb\n");
        let d = Diagnostic::error("duplicate definition")
            .with_code("E0300")
            .primary(Span::new(a, 0, 3), "first")
            .secondary(Span::new(b, 0, 3), "second");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0300]: duplicate definition
 --> a.wk:1:1
  |
1 | aaa
  | ^^^ first
  ::: b.wk:1:1
  |
1 | bbb
  | --- second
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn caret_aligns_under_a_wide_character() {
        // 你 is an East-Asian wide character (2 terminal cells); the caret under the following `x`
        // must be padded by 2, not 1, or it drifts one column left.
        let (sm, id) = sm_with("你x\n");
        let span = Span::new(id, 3, 4); // the `x` (你 is 3 UTF-8 bytes)
        let d = Diagnostic::error("bad").primary(span, "here");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: bad
 --> test.wk:1:2
  |
1 | 你x
  |   ^ here
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn renders_a_suggestion_as_a_help_line() {
        use crate::Applicability;
        let (sm, id) = sm_with("fn main() {\n    let x = 0;\n    x = 1;\n}\n");
        let span = Span::new(id, 20, 21); // the `x` in `let x`
        let d = Diagnostic::error("cannot assign twice to immutable binding `x`")
            .with_code("E0304")
            .primary(Span::new(id, 31, 32), "cannot assign")
            .suggest_with(
                span,
                "consider making this binding mutable",
                "mut x",
                Applicability::MachineApplicable,
            );
        let r = Renderer::new(false).render(&d, &sm);
        assert!(
            r.ends_with("= help: consider making this binding mutable: `mut x`"),
            "{r}"
        );
    }

    #[test]
    fn color_adds_escapes() {
        let (sm, id) = sm_with("abc");
        let d = Diagnostic::error("e").primary(Span::new(id, 0, 1), "");
        let colored = Renderer::new(true).render(&d, &sm);
        assert!(colored.contains("\x1b["));
        let plain = Renderer::new(false).render(&d, &sm);
        assert!(!plain.contains("\x1b["));
    }

    /// The span from the first occurrence of `from` to the end of the first occurrence of `to` at
    /// or after it.
    fn span_between(src: &str, id: wukong_span::SourceId, from: &str, to: &str) -> Span {
        let lo = src.find(from).unwrap();
        let hi = lo + src[lo..].find(to).unwrap() + to.len();
        Span::new(id, lo as u32, hi as u32)
    }

    // The multi-line snapshots below are derived by hand from the connector rules on
    // `render_connector`: margin `"  "`, `"| "` or `"/ "` between `| ` and the text; a start
    // marker `{pad} |  ` + `_` x (s+1) + mark, with s the rendered width before the span; an end
    // marker `{pad} | |` + `_` x (e+1) + mark + message, with e the rendered width before the
    // span's last char. The mark sits in the column of the char it points at.

    #[test]
    fn multi_line_span_starting_mid_line_gets_a_start_marker() {
        let src = "fn main() {\n    let y = add(\n        1,\n        2);\n}\n";
        let (sm, id) = sm_with(src);
        // `add(` .. `)`: 'a' is col 13 of line 2 (s = 12), ')' col 10 of line 4 (e = 9).
        let span = span_between(src, id, "add(", ")");
        let d = Diagnostic::error("call spans three lines").primary(span, "this call");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: call spans three lines
 --> test.wk:2:13
  |
2 |       let y = add(
  |  _____________^
3 | |         1,
4 | |         2);
  | |__________^ this call
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn multi_line_span_opening_its_line_gets_a_slash() {
        let src = "fn main() {\n    match a {\n        0 => 1,\n    }\n}\n";
        let (sm, id) = sm_with(src);
        // `match` (only whitespace before it on line 2) .. the `}` at col 5 of line 4 (e = 4).
        let span = span_between(src, id, "match", "    }");
        let d = Diagnostic::error("non-exhaustive match")
            .with_code("E0405")
            .primary(span, "patterns not covered");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0405]: non-exhaustive match
 --> test.wk:2:5
  |
2 | /     match a {
3 | |         0 => 1,
4 | |     }
  | |_____^ patterns not covered
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_ten_line_span_elides_its_middle_and_widens_the_gutter() {
        let src = "fn main() {\n    let v = [\n        1,\n        2,\n        3,\n        4,\n        5,\n        6,\n        7,\n        8,\n    ];\n}\n";
        let (sm, id) = sm_with(src);
        // `[` at col 13 of line 2 (s = 12) .. `]` at col 5 of line 11 (e = 4): ten lines, so lines
        // 2, 3, `...`, 10 and 11 are printed, and the gutter is 3 wide for the `...`.
        let span = span_between(src, id, "[", "]");
        let d = Diagnostic::error("long literal").primary(span, "array literal");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: long literal
   --> test.wk:2:13
    |
  2 |       let v = [
    |  _____________^
  3 | |         1,
... | |
 10 | |         8,
 11 | |     ];
    | |_____^ array literal
    |";
        assert_eq!(r, expected);
    }

    #[test]
    fn seven_lines_elide_and_six_do_not() {
        // The elision threshold: a 6-line label prints every line, a 7-line one elides lines 3..=5.
        let six = "a\nb\nc\nd\ne\nf\n";
        let (sm, id) = sm_with(six);
        let d = Diagnostic::error("six").primary(Span::new(id, 0, 11), "");
        let expected = "\
error: six
 --> test.wk:1:1
  |
1 | / a
2 | | b
3 | | c
4 | | d
5 | | e
6 | | f
  | |_^
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);

        let seven = "a\nb\nc\nd\ne\nf\ng\n";
        let (sm, id) = sm_with(seven);
        let d = Diagnostic::error("seven").primary(Span::new(id, 0, 13), "");
        let expected = "\
error: seven
   --> test.wk:1:1
    |
  1 | / a
  2 | | b
... | |
  6 | | f
  7 | | g
    | |_^
    |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);
    }

    #[test]
    fn a_single_line_label_inside_a_connector_shares_its_line() {
        let src = "fn main() {\n    let a = 1;\n    match a {\n        0 => 1,\n    }\n}\n";
        let (sm, id) = sm_with(src);
        let a = src.find("a = 1").unwrap() as u32; // col 9 of line 2
        let zero = src.find("0 =>").unwrap() as u32; // col 9 of line 4, inside the match
        let d = Diagnostic::error("non-exhaustive match")
            .with_code("E0405")
            .primary(span_between(src, id, "match", "    }"), "not exhaustive")
            .secondary(Span::new(id, a, a + 1), "defined here")
            .secondary(Span::new(id, zero, zero + 1), "only this value is covered");
        let r = Renderer::new(false).render(&d, &sm);
        // Line 4 is printed once, inside the connector, with the `0` label under it and the
        // connector's `|` running past it in the margin. (It used to be printed a second time, in a
        // frame of its own after the connector's.) Line 2 is outside the connector, so its margin
        // is blank.
        let expected = "\
error[E0405]: non-exhaustive match
 --> test.wk:3:5
  |
2 |       let a = 1;
  |           - defined here
3 | /     match a {
4 | |         0 => 1,
  | |         - only this value is covered
5 | |     }
  | |_____^ not exhaustive
  |";
        assert_eq!(r, expected);
    }

    /// The span of the first occurrence of `text`.
    fn span_of(src: &str, id: wukong_span::SourceId, text: &str) -> Span {
        let lo = src.find(text).unwrap();
        Span::new(id, lo as u32, (lo + text.len()) as u32)
    }

    /// The one-character span of the first character of the first occurrence of `text`.
    fn first_char_of(src: &str, id: wukong_span::SourceId, text: &str) -> Span {
        let lo = src.find(text).unwrap() as u32;
        Span::new(id, lo, lo + 1)
    }

    // The single-line snapshots below are derived by hand from `single_line_rows`: one underline
    // row, the rightmost label's message ending it when no underline reaches past its own, and
    // every other message hanging, rightmost first, at its label's first column under a row of
    // `|` bars.

    #[test]
    fn two_labels_on_one_line_share_one_frame() {
        let src = "fn main() {\n    let x: i32 = \"a\" + b;\n}\n";
        let (sm, id) = sm_with(src);
        // `i32` at rendered columns 11..14, `"a"` at 17..20 (header column 18).
        let d = Diagnostic::error("mismatched types")
            .with_code("E0401")
            .primary(span_of(src, id, "\"a\""), "expected `i32`, found `*u8`")
            .secondary(span_of(src, id, "i32"), "expected due to this");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0401]: mismatched types
 --> test.wk:2:18
  |
2 |     let x: i32 = \"a\" + b;
  |            ---   ^^^ expected `i32`, found `*u8`
  |            |
  |            expected due to this
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn three_labels_on_one_line_hang_right_to_left() {
        let src = "fn main() {\n    foo(a, b, c);\n}\n";
        let (sm, id) = sm_with(src);
        // `foo` 4..7, `a` 8..9, `b` 11..12, `c` 14..15. `b` has no message: it only underlines.
        let d = Diagnostic::error("three arguments")
            .primary(first_char_of(src, id, "c)"), "third")
            .secondary(span_of(src, id, "foo"), "callee")
            .secondary(first_char_of(src, id, "a,"), "first")
            .secondary(first_char_of(src, id, "b,"), "");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: three arguments
 --> test.wk:2:15
  |
2 |     foo(a, b, c);
  |     --- -  -  ^ third
  |     |   |
  |     |   first
  |     callee
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_label_nested_in_another_on_one_line_stays_visible() {
        let src = "fn main() {\n    let total = scale(x) * 2;\n}\n";
        let (sm, id) = sm_with(src);
        // The primary covers `scale(x) * 2` (16..28) and the secondary `x` (22..23) inside it. The
        // wider underline is drawn first, so the `-` shows through; the secondary does not reach
        // the row's end, so neither message fits on it and both hang.
        let d = Diagnostic::error("overflow")
            .primary(span_of(src, id, "scale(x) * 2"), "this product overflows")
            .secondary(first_char_of(src, id, "x)"), "x is large");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: overflow
 --> test.wk:2:17
  |
2 |     let total = scale(x) * 2;
  |                 ^^^^^^-^^^^^
  |                 |     |
  |                 |     x is large
  |                 this product overflows
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_primary_and_a_secondary_on_one_span() {
        let src = "fn main() {\n    f(x);\n}\n";
        let (sm, id) = sm_with(src);
        // Both on `x` (6..7): the primary's `^` is drawn over the secondary's `-`, and the primary
        // is the rightmost by the tie-break, so its message ends the row and the secondary hangs.
        let d = Diagnostic::error("use of moved value")
            .secondary(span_of(src, id, "x"), "and used here")
            .primary(span_of(src, id, "x"), "moved here");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: use of moved value
 --> test.wk:2:7
  |
2 |     f(x);
  |       ^ moved here
  |       |
  |       and used here
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn hanging_bars_and_messages_take_their_label_colors() {
        let src = "fn main() {\n    let x: i32 = \"a\" + b;\n}\n";
        let (sm, id) = sm_with(src);
        let d = Diagnostic::error("e")
            .primary(span_of(src, id, "\"a\""), "p")
            .secondary(span_of(src, id, "i32"), "s");
        let r = Renderer::new(true).render(&d, &sm);
        let (blue, red) = ("\x1b[1;34m", "\x1b[1;31m");
        let row = |s: &str| format!("  {blue}|{RESET} {s}");
        for want in [
            row(&format!("           {blue}---{RESET}   {red}^^^ p{RESET}")),
            row(&format!("           {blue}|{RESET}")),
            row(&format!("           {blue}s{RESET}")),
        ] {
            assert!(r.contains(&format!("{want}\n")), "{want:?} in {r:?}");
        }
    }

    #[test]
    fn a_secondary_multi_line_label_gets_connectors_with_dashes() {
        // The primary label is single-line, so the first multi-line label, a secondary, is drawn
        // with connectors, marked `-` like any secondary.
        let src = "fn main() {\n    let t = (1,\n             2);\n    t.5;\n}\n";
        let (sm, id) = sm_with(src);
        let five = src.find("5;").unwrap() as u32; // col 7 of line 4
        let d = Diagnostic::error("no field `5` on a 2-tuple")
            .primary(Span::new(id, five, five + 1), "no field 5")
            .secondary(span_between(src, id, "(1", ")"), "tuple defined here");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: no field `5` on a 2-tuple
 --> test.wk:4:7
  |
2 |       let t = (1,
  |  _____________-
3 | |              2);
  | |_______________- tuple defined here
4 |       t.5;
  |         ^ no field 5
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn tab_indented_multi_line_span_aligns_its_markers() {
        let src = "fn main() {\n\tlet y = add(\n\t\t1,\n\t);\n}\n";
        let (sm, id) = sm_with(src);
        // `add(` is col 10 of line 2, rendered at width 4 + 8 = 12; `)` is col 2 of line 4,
        // rendered at width 4. The header keeps the char column, 10.
        let span = span_between(src, id, "add(", ")");
        let d = Diagnostic::error("bad call").primary(span, "this call");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: bad call
 --> test.wk:2:10
  |
2 |       let y = add(
  |  _____________^
3 | |         1,
4 | |     );
  | |_____^ this call
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_span_ending_at_a_newline_ends_on_the_line_before() {
        // `hi` at column 1 of line 2: the last char is line 1's `\n`, so the label is single-line
        // and renders exactly as before connectors existed (cut, one column past the text).
        let (sm, id) = sm_with("abc\ndef\n");
        let d = Diagnostic::error("e").primary(Span::new(id, 0, 4), "here");
        let expected = "\
error: e
 --> test.wk:1:1
  |
1 | abc
  | ^^^^ here
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);

        // `hi` at column 1 of line 10: the label ends at line 9's `\n` (col 10, e = 9), so line 10
        // is neither printed nor counted in the gutter width.
        let src = "fn main() {\n    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n    let e = 5;\n    let f = 6;\n    g(a,\n      b);\n}\n";
        let (sm, id) = sm_with(src);
        let lo = src.find("g(a").unwrap() as u32;
        let hi = src.rfind('}').unwrap() as u32;
        let d = Diagnostic::error("e").primary(Span::new(id, lo, hi), "ends at the newline");
        let expected = "\
error: e
 --> test.wk:8:5
  |
8 | /     g(a,
9 | |       b);
  | |__________^ ends at the newline
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);
    }

    #[test]
    fn every_multi_line_label_gets_connectors() {
        let src = "fn main() {\n    let a = 1;\n    let b = 2;\n    g(a,\n      b,\n      a);\n    let c = 3;\n    f(a,\n      b,\n      c);\n}\n";
        let (sm, id) = sm_with(src);
        // The secondary `g(..)` (lines 4-6) and the primary `f(..)` (lines 8-10) share no line, so
        // they share the one connector column; each opens its line (`/`) and ends at the `)` in
        // column 8 (e = 7: `|` and eight `_`). Line 7 lies between them, inside neither, so it is
        // not printed; line 10 widens the gutter to 2. (The second multi-line label used to be cut
        // to its first line.)
        let d = Diagnostic::error("two multi-line labels")
            .secondary(span_between(src, id, "g(a", ")"), "the first call")
            .primary(span_between(src, id, "f(a", ")"), "the second call");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: two multi-line labels
  --> test.wk:8:5
   |
 4 | /     g(a,
 5 | |       b,
 6 | |       a);
   | |________- the first call
 8 | /     f(a,
 9 | |       b,
10 | |       c);
   | |________^ the second call
   |";
        assert_eq!(r, expected);
    }

    #[test]
    fn nested_multi_line_labels_get_a_connector_column_each() {
        let src =
            "fn main() {\n    let r = outer(\n        inner(\n            1),\n        2);\n}\n";
        let (sm, id) = sm_with(src);
        // `outer(` .. `2)` spans lines 2-5 (s = 12, e = 9) and opens first, so it takes column 0;
        // `inner(` .. `1)` spans lines 3-4 inside it (s = 8, opening its line, e = 13) and takes
        // column 1. Two columns make the margin three wide: `outer`'s start marker runs `_` from
        // column 1 to its mark at 3 + 12; `inner` opens with `/` in column 1; `inner` closes first
        // (`|` of `outer`, then its own `|`, `_` to its mark at 3 + 13); `outer` closes on line 5.
        let d = Diagnostic::error("nested calls")
            .primary(span_between(src, id, "outer(", "2)"), "outer call")
            .secondary(span_between(src, id, "inner(", "1)"), "inner call");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: nested calls
 --> test.wk:2:13
  |
2 |        let r = outer(
  |  ______________^
3 | |/         inner(
4 | ||             1),
  | ||______________- inner call
5 | |          2);
  | |___________^ outer call
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn single_line_labels_on_a_multi_line_labels_first_middle_and_last_lines() {
        let src = "fn main() {\n    let y = add(x,\n        1,\n        z);\n}\n";
        let (sm, id) = sm_with(src);
        // `add(` .. `)` spans lines 2-4 (s = 12, e = 9); `x` (16), `1` and `z` (8) are on its
        // first, middle and last lines. Each line prints once: its single-line labels first, then
        // the connector's end marker (line 4), and its start marker after line 2's label.
        let d = Diagnostic::error("bad call")
            .primary(span_between(src, id, "add(", ")"), "the call")
            .secondary(first_char_of(src, id, "x,"), "x here")
            .secondary(first_char_of(src, id, "1,"), "one")
            .secondary(first_char_of(src, id, "z)"), "z here");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: bad call
 --> test.wk:2:13
  |
2 |       let y = add(x,
  |                   - x here
  |  _____________^
3 | |         1,
  | |         - one
4 | |         z);
  | |         - z here
  | |__________^ the call
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_connector_through_a_labelled_line_elides_each_long_gap() {
        let src = "fn main() {\n    let v = [\n        1,\n        2,\n        3,\n        4,\n        5,\n        6,\n        7,\n        8,\n        9,\n        10,\n    ];\n}\n";
        let (sm, id) = sm_with(src);
        // `[` (line 2, s = 12) .. `]` (line 13, e = 4), with a label on `2` (line 4, column 8).
        // The gap 2..4 has one line, printed; the gap 4..13 has eight, more than four, so only its
        // first (5) and last (12) are printed around `...`.
        let d = Diagnostic::error("long literal")
            .primary(span_between(src, id, "[", "]"), "array literal")
            .secondary(first_char_of(src, id, "2,"), "this one");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: long literal
   --> test.wk:2:13
    |
  2 |       let v = [
    |  _____________^
  3 | |         1,
  4 | |         2,
    | |         - this one
  5 | |         3,
... | |
 12 | |         10,
 13 | |     ];
    | |_____^ array literal
    |";
        assert_eq!(r, expected);
    }

    #[test]
    fn json_positions_of_a_multi_line_span_are_unchanged() {
        // Only the human form (and its `rendered` copy) draws connectors; `line`/`col` stay the
        // span's `lo` and `line_end`/`col_end` its exclusive `hi`, byte offsets untouched.
        let src = "fn main() {\n    let y = add(\n        1,\n        2);\n}\n";
        let (sm, id) = sm_with(src);
        let span = span_between(src, id, "add(", ")");
        assert_eq!((span.lo, span.hi), (24, 50));
        let d = Diagnostic::error("call spans three lines").primary(span, "this call");
        let j = crate::to_json(&d, &sm);
        assert!(
            j.contains(
                "\"line\":2,\"col\":13,\"primary\":true,\"label\":\"this call\",\
                 \"line_end\":4,\"col_end\":11,\"byte_start\":24,\"byte_end\":50}"
            ),
            "{j}"
        );
        assert!(j.contains("\\n  | |__________^ this call\\n"), "{j}");
    }

    #[test]
    fn header_column_is_the_char_column_json_reports() {
        // The `-->` column is the 1-based char column, the same number as JSON `col` — the
        // editor/LSP jump contract. Display width only places the caret: a tab (4 cells) and a
        // wide char (2 cells) put `x` at display column 7, but its header column is 3.
        let (sm, id) = sm_with("\t你x = 1;\n");
        let span = Span::new(id, 4, 5); // `x`: tab (1 byte) + 你 (3 bytes)
        let d = Diagnostic::error("bad").primary(span, "here");
        let expected = "\
error: bad
 --> test.wk:1:3
  |
1 |     你x = 1;
  |       ^ here
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);
        assert!(crate::to_json(&d, &sm).contains("\"line\":1,\"col\":3,"));
    }

    #[test]
    fn connectors_take_the_label_color_and_the_gutter_stays_blue() {
        let src = "fn main() {\n    match a {\n        0 => 1,\n    }\n}\n";
        let (sm, id) = sm_with(src);
        let d = Diagnostic::error("e").primary(span_between(src, id, "match", "    }"), "m");
        let r = Renderer::new(true).render(&d, &sm);
        let (blue, red) = ("\x1b[1;34m", "\x1b[1;31m");
        assert!(
            r.contains(&format!(
                "{blue}2{RESET} {blue}|{RESET} {red}/{RESET}     match a {{"
            )),
            "{r:?}"
        );
        assert!(
            r.contains(&format!(
                "{blue}3{RESET} {blue}|{RESET} {red}|{RESET}         0 => 1,"
            )),
            "{r:?}"
        );
        assert!(
            r.contains(&format!("  {blue}|{RESET} {red}|_____^ m{RESET}")),
            "{r:?}"
        );
    }
}
