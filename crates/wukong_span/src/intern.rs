//! A simple string interner.
//!
//! Identifiers appear over and over in source; interning maps each distinct string to a small
//! [`Symbol`] (a `u32`) that is `Copy`, `Eq`, and `Hash`, so the rest of the compiler can
//! compare and store names cheaply instead of cloning `String`s everywhere.

use crate::fxhash::FxHashMap;

/// A handle to an interned string. Cheap to copy, compare, and hash.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(pub u32);

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sym#{}", self.0)
    }
}

/// Interns strings into [`Symbol`]s and resolves them back.
#[derive(Default)]
pub struct Interner {
    lookup: FxHashMap<Box<str>, Symbol>,
    strings: Vec<Box<str>>,
}

impl Interner {
    pub fn new() -> Interner {
        Interner::default()
    }

    /// Intern `s`, returning a stable [`Symbol`]. Interning the same text twice yields the
    /// same symbol.
    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(&sym) = self.lookup.get(s) {
            return sym;
        }
        let sym = Symbol(self.strings.len() as u32);
        let boxed: Box<str> = s.into();
        self.strings.push(boxed.clone());
        self.lookup.insert(boxed, sym);
        sym
    }

    /// The symbol for `s` **if it has already been interned**, without minting a new one.
    ///
    /// The read-only half of [`Interner::intern`], for the passes that hold a shared `&Interner`
    /// and must ask "does this name exist?" rather than create it (`wukong_mir_build`'s recognizers
    /// resolve a struct-field buffer base `l.w` through the synthetic name `"l.w"`, pre-interned by
    /// `lower_program`, which is the only place that holds `&mut Interner`). A `None` therefore
    /// means "not a name this compilation minted", which every caller must treat as a decline.
    pub fn get(&self, s: &str) -> Option<Symbol> {
        self.lookup.get(s).copied()
    }

    /// Resolve a symbol back to its string.
    ///
    /// A `Symbol` is only meaningful inside the interner that minted it: this indexes `strings`
    /// directly, so a foreign symbol panics if its index is out of range and — worse — silently
    /// resolves to the *wrong* string if it happens to be in range. There is one interner per
    /// compilation; never persist or cross-compare a `Symbol` value.
    pub fn resolve(&self, sym: Symbol) -> &str {
        &self.strings[sym.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_and_roundtrip() {
        let mut i = Interner::new();
        let a = i.intern("matmul");
        let b = i.intern("tensor");
        let a2 = i.intern("matmul");
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(i.resolve(a), "matmul");
        assert_eq!(i.resolve(b), "tensor");
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn get_is_read_only() {
        let mut i = Interner::new();
        assert_eq!(i.get("matmul"), None);
        let a = i.intern("matmul");
        assert_eq!(i.get("matmul"), Some(a));
        // `get` must not mint: the miss above left the table untouched.
        assert_eq!(i.len(), 1);
        assert_eq!(i.get("l.wq"), None);
    }

    #[test]
    fn empty_string_interns() {
        let mut i = Interner::new();
        let e = i.intern("");
        assert_eq!(i.resolve(e), "");
    }
}
