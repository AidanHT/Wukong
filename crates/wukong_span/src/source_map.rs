//! The source map: owns loaded source text and translates byte offsets into line/column.

use crate::{SourceId, Span};

/// A 1-based line/column position, suitable for display in diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub line: u32,
    pub col: u32,
}

/// One loaded source file: its name, its text, and a precomputed index of line starts.
pub struct SourceFile {
    pub name: String,
    pub src: String,
    /// Byte offset of the first character of each line. Always starts with `0`.
    line_starts: Vec<u32>,
}

impl SourceFile {
    fn new(name: String, src: String) -> SourceFile {
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push((i + 1) as u32);
            }
        }
        SourceFile {
            name,
            src,
            line_starts,
        }
    }

    /// 0-based index of the line containing `offset`.
    fn line_index(&self, offset: u32) -> usize {
        match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        }
    }
}

/// Owns every source file in a compilation and hands out [`SourceId`]s.
#[derive(Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> SourceMap {
        SourceMap::default()
    }

    /// Load a source file, returning its id. The id is just its insertion index.
    pub fn add(&mut self, name: impl Into<String>, src: impl Into<String>) -> SourceId {
        let id = SourceId(self.files.len() as u32);
        self.files.push(SourceFile::new(name.into(), src.into()));
        id
    }

    fn file(&self, id: SourceId) -> &SourceFile {
        &self.files[id.0 as usize]
    }

    pub fn source(&self, id: SourceId) -> &str {
        &self.file(id).src
    }

    pub fn name(&self, id: SourceId) -> &str {
        &self.file(id).name
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// The source text covered by `span`.
    pub fn span_text(&self, span: Span) -> &str {
        &self.file(span.source).src[span.lo as usize..span.hi as usize]
    }

    /// Translate a byte offset within a file into a 1-based line/column.
    ///
    /// The column counts Unicode scalar values (chars) from the line start, so multi-byte
    /// UTF-8 sequences advance the column by one, not by their byte length.
    pub fn location(&self, id: SourceId, offset: u32) -> Location {
        let f = self.file(id);
        let line_idx = f.line_index(offset);
        let line_start = f.line_starts[line_idx] as usize;
        let col = f.src[line_start..offset as usize].chars().count() as u32;
        Location {
            line: line_idx as u32 + 1,
            col: col + 1,
        }
    }

    /// The start location of a span (its `lo`).
    pub fn span_location(&self, span: Span) -> Location {
        self.location(span.source, span.lo)
    }

    /// The text of a given 1-based line, with the trailing newline stripped.
    pub fn line_text(&self, id: SourceId, line: u32) -> &str {
        let f = self.file(id);
        let i = (line - 1) as usize;
        let start = f.line_starts[i] as usize;
        let end = f
            .line_starts
            .get(i + 1)
            .map(|&x| x as usize)
            .unwrap_or(f.src.len());
        f.src[start..end].trim_end_matches(['\n', '\r'])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line_locations() {
        let mut sm = SourceMap::new();
        let id = sm.add("a.wk", "fn main");
        assert_eq!(sm.location(id, 0), Location { line: 1, col: 1 });
        assert_eq!(sm.location(id, 3), Location { line: 1, col: 4 });
    }

    #[test]
    fn multi_line_locations() {
        let mut sm = SourceMap::new();
        let id = sm.add("a.wk", "abc\ndef\nghi");
        // 'd' is byte 4, start of line 2.
        assert_eq!(sm.location(id, 4), Location { line: 2, col: 1 });
        // 'h' is byte 9.
        assert_eq!(sm.location(id, 9), Location { line: 3, col: 2 });
        assert_eq!(sm.line_text(id, 2), "def");
    }

    #[test]
    fn multibyte_columns() {
        let mut sm = SourceMap::new();
        // "é" is two bytes in UTF-8; the 'x' after it is byte offset 2 but column 2.
        let id = sm.add("a.wk", "éx");
        assert_eq!(sm.location(id, 2), Location { line: 1, col: 2 });
    }

    #[test]
    fn span_text_roundtrip() {
        let mut sm = SourceMap::new();
        let id = sm.add("a.wk", "let x = 42;");
        let sp = Span::new(id, 4, 5);
        assert_eq!(sm.span_text(sp), "x");
        assert_eq!(sm.name(id), "a.wk");
    }
}
