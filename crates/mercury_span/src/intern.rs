//! A simple string interner.
//!
//! Identifiers appear over and over in source; interning maps each distinct string to a small
//! [`Symbol`] (a `u32`) that is `Copy`, `Eq`, and `Hash`, so the rest of the compiler can
//! compare and store names cheaply instead of cloning `String`s everywhere.

use std::collections::HashMap;

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
    lookup: HashMap<Box<str>, Symbol>,
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

    /// Resolve a symbol back to its string. Panics on a symbol from a different interner.
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
    fn empty_string_interns() {
        let mut i = Interner::new();
        let e = i.intern("");
        assert_eq!(i.resolve(e), "");
    }
}
