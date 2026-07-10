//! `wukong_span` — source positions, the source map, and string interning.
//!
//! These are the foundational, dependency-free types shared by every later stage of the
//! compiler. A [`Span`] is a half-open byte range within a [`SourceId`]; the [`SourceMap`]
//! owns the actual text and translates byte offsets into human-readable line/column
//! [`Location`]s. The [`Interner`] turns repeated identifier strings into cheap copyable
//! [`Symbol`]s.

pub mod fxhash;
mod intern;
mod source_map;

pub use fxhash::{FxHashMap, FxHashSet, FxHasher};
pub use intern::{Interner, Symbol};
pub use source_map::{Location, SourceFile, SourceMap};

/// Identifies one source file within a [`SourceMap`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub u32);

impl std::fmt::Debug for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "src#{}", self.0)
    }
}

/// A half-open byte range `[lo, hi)` within a specific source file.
///
/// Spans are deliberately small (12 bytes) and `Copy` so they can be attached to every token
/// and AST node without a second thought.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub source: SourceId,
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub const fn new(source: SourceId, lo: u32, hi: u32) -> Self {
        Span { source, lo, hi }
    }

    /// A placeholder span for compiler-synthesized nodes that have no backing source text.
    pub const fn dummy() -> Self {
        Span {
            source: SourceId(u32::MAX),
            lo: 0,
            hi: 0,
        }
    }

    pub const fn is_dummy(&self) -> bool {
        self.source.0 == u32::MAX
    }

    pub const fn len(&self) -> u32 {
        self.hi - self.lo
    }

    pub const fn is_empty(&self) -> bool {
        self.lo == self.hi
    }

    /// The smallest span covering both `self` and `other`. They must share a source.
    pub fn to(self, other: Span) -> Span {
        debug_assert_eq!(
            self.source, other.source,
            "cannot merge spans from different sources"
        );
        Span {
            source: self.source,
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// A zero-width span at this span's start — useful for "expected token here" diagnostics.
    pub const fn shrink_to_lo(self) -> Span {
        Span {
            source: self.source,
            lo: self.lo,
            hi: self.lo,
        }
    }

    /// A zero-width span at this span's end.
    pub const fn shrink_to_hi(self) -> Span {
        Span {
            source: self.source,
            lo: self.hi,
            hi: self.hi,
        }
    }
}

impl std::fmt::Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_dummy() {
            write!(f, "<dummy>")
        } else {
            write!(f, "{}..{}@{}", self.lo, self.hi, self.source.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_len_and_empty() {
        let s = Span::new(SourceId(0), 3, 7);
        assert_eq!(s.len(), 4);
        assert!(!s.is_empty());
        assert!(Span::new(SourceId(0), 5, 5).is_empty());
    }

    #[test]
    fn span_merge() {
        let a = Span::new(SourceId(0), 3, 7);
        let b = Span::new(SourceId(0), 10, 12);
        let m = a.to(b);
        assert_eq!((m.lo, m.hi), (3, 12));
        // merge is order-independent
        assert_eq!(b.to(a), m);
    }

    #[test]
    fn dummy_span() {
        assert!(Span::dummy().is_dummy());
        assert!(!Span::new(SourceId(0), 0, 0).is_dummy());
    }

    #[test]
    fn shrink() {
        let s = Span::new(SourceId(1), 4, 9);
        assert_eq!(s.shrink_to_lo(), Span::new(SourceId(1), 4, 4));
        assert_eq!(s.shrink_to_hi(), Span::new(SourceId(1), 9, 9));
    }
}
