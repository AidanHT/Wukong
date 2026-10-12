//! A simple string interner.
//!
//! Identifiers appear over and over in source; interning maps each distinct string to a small
//! [`Symbol`] (a `u32`) that is `Copy`, `Eq`, and `Hash`, so the rest of the compiler can
//! compare and store names cheaply instead of cloning `String`s everywhere.

use std::sync::{Mutex, PoisonError};

use crate::fxhash::FxHashMap;

/// The first [`Symbol`] value [`Interner::fresh`] hands out. Interned strings count up from zero and
/// fresh symbols from here, so the two ranges never meet (an interner would need 2^31 distinct
/// strings first).
const FRESH_FLOOR: u32 = 1 << 31;

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
    /// The symbols [`Interner::fresh`] minted, in order: entry `i` is `Symbol(FRESH_FLOOR + i)`, and
    /// holds the interned symbol whose text it displays as. Behind a lock because minting happens
    /// through a shared reference (semantic analysis holds `&Interner`).
    fresh: Mutex<Vec<Symbol>>,
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

    /// A new symbol, distinct from every other symbol this interner has handed out or will, that
    /// displays as `like`'s text — through a shared reference, unlike [`Interner::intern`].
    ///
    /// It names something the source spells with an existing name but that must not be confused
    /// with it: semantic analysis gives the value a local `n` has when `f::<n>(..)` is called a fresh
    /// dimension symbol, so a shape derived from it never unifies with a later value of `n`, while
    /// every diagnostic still calls it `n`. A fresh symbol is not text: [`Interner::get`] never
    /// returns one, and interning its text yields `like`'s symbol (or that of the name it was minted
    /// like), never the fresh one. Minting `like` a fresh symbol displays as the name that one was
    /// minted like.
    pub fn fresh(&self, like: Symbol) -> Symbol {
        let mut fresh = self.fresh.lock().unwrap_or_else(PoisonError::into_inner);
        let base = match like.0.checked_sub(FRESH_FLOOR) {
            Some(i) => fresh[i as usize],
            None => like,
        };
        let sym = Symbol(FRESH_FLOOR + fresh.len() as u32);
        fresh.push(base);
        sym
    }

    /// Whether [`Interner::fresh`] minted `sym`.
    pub fn is_fresh(&self, sym: Symbol) -> bool {
        sym.0 >= FRESH_FLOOR
    }

    /// Resolve a symbol back to its string (a [`fresh`](Interner::fresh) one to the text of the name
    /// it was minted like).
    ///
    /// A `Symbol` is only meaningful inside the interner that minted it: this indexes `strings`
    /// directly, so a foreign symbol panics if its index is out of range and — worse — silently
    /// resolves to the *wrong* string if it happens to be in range. There is one interner per
    /// compilation; never persist or cross-compare a `Symbol` value.
    pub fn resolve(&self, sym: Symbol) -> &str {
        let sym = match sym.0.checked_sub(FRESH_FLOOR) {
            Some(i) => self.fresh.lock().unwrap_or_else(PoisonError::into_inner)[i as usize],
            None => sym,
        };
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

    /// A fresh symbol is a new identity with an old name: it differs from the name it was minted
    /// like and from every other fresh symbol, displays as that name, and is never found by text.
    #[test]
    fn fresh_symbols_are_distinct_and_display_as_their_base() {
        let mut i = Interner::new();
        let n = i.intern("n");
        let a = i.fresh(n);
        let b = i.fresh(n);
        assert_ne!(a, n);
        assert_ne!(a, b);
        assert!(i.is_fresh(a) && i.is_fresh(b) && !i.is_fresh(n));
        assert_eq!(i.resolve(a), "n");
        assert_eq!(i.resolve(b), "n");
        // Minted like a fresh symbol, it displays as the original name.
        let c = i.fresh(a);
        assert_eq!(i.resolve(c), "n");
        // The text still means the interned symbol, and minting interned nothing.
        assert_eq!(i.get("n"), Some(n));
        assert_eq!(i.intern("n"), n);
        assert_eq!(i.len(), 1);
        // Interning after minting never collides with a fresh symbol.
        let m = i.intern("m");
        assert!(![a, b, c].contains(&m));
        assert_eq!(i.resolve(m), "m");
    }

    #[test]
    fn empty_string_interns() {
        let mut i = Interner::new();
        let e = i.intern("");
        assert_eq!(i.resolve(e), "");
    }
}
