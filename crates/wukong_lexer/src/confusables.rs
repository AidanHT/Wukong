//! Unicode characters that look like ASCII punctuation, operators or whitespace (`E0105`).
//!
//! Kernel code is routinely pasted from papers, word processors and chat clients, which substitute
//! `−` (U+2212 MINUS SIGN) for `-`, `×` for `*`, curly quotes for `"`, and a no-break or zero-width
//! space for a plain space. Before this table every such character was an opaque
//! `E0101 unexpected character` printed through `escape_default` (`\u{2212}`), and the parser then
//! reported a second error at the same place. Now the lexer names the code point and the ASCII
//! character it was mistaken for, and — where that ASCII character is one token — keeps lexing as
//! if it had been written, so one pasted character costs exactly one diagnostic.
//!
//! The table is closed and small on purpose (the common paste artifacts, after rustc's
//! `unicode_chars.rs`), not Unicode's full `confusables.txt`: a character not listed here is still
//! reported, as `E0101` with its code point.

use crate::TokenKind;

/// What the lexer does after reporting a confusable character.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Recover {
    /// Produce the ASCII token the character was mistaken for (`−` lexes as `-`).
    Token(TokenKind),
    /// Treat the character as whitespace and keep scanning.
    Trivia,
    /// Produce an error token: the ASCII character opens a construct (a string or char literal)
    /// the lexer cannot reconstruct from a lookalike.
    Error,
}

/// One table entry: the character, its Unicode name, the ASCII text it imitates, and the recovery.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Confusable {
    pub ch: char,
    pub name: &'static str,
    /// The ASCII replacement; empty for a zero-width character (delete it).
    pub ascii: &'static str,
    pub recover: Recover,
}

const fn tok(ch: char, name: &'static str, ascii: &'static str, kind: TokenKind) -> Confusable {
    Confusable {
        ch,
        name,
        ascii,
        recover: Recover::Token(kind),
    }
}

const fn space(ch: char, name: &'static str, ascii: &'static str) -> Confusable {
    Confusable {
        ch,
        name,
        ascii,
        recover: Recover::Trivia,
    }
}

const fn quote(ch: char, name: &'static str, ascii: &'static str) -> Confusable {
    Confusable {
        ch,
        name,
        ascii,
        recover: Recover::Error,
    }
}

const TABLE: &[Confusable] = &[
    // Dashes and minus signs.
    tok('\u{2212}', "MINUS SIGN", "-", TokenKind::Minus),
    tok('\u{2010}', "HYPHEN", "-", TokenKind::Minus),
    tok('\u{2011}', "NON-BREAKING HYPHEN", "-", TokenKind::Minus),
    tok('\u{2012}', "FIGURE DASH", "-", TokenKind::Minus),
    tok('\u{2013}', "EN DASH", "-", TokenKind::Minus),
    tok('\u{2014}', "EM DASH", "-", TokenKind::Minus),
    tok('\u{FE63}', "SMALL HYPHEN-MINUS", "-", TokenKind::Minus),
    tok('\u{FF0D}', "FULLWIDTH HYPHEN-MINUS", "-", TokenKind::Minus),
    // Multiplication and division.
    tok('\u{00D7}', "MULTIPLICATION SIGN", "*", TokenKind::Star),
    tok('\u{2217}', "ASTERISK OPERATOR", "*", TokenKind::Star),
    tok('\u{22C5}', "DOT OPERATOR", "*", TokenKind::Star),
    tok('\u{FF0A}', "FULLWIDTH ASTERISK", "*", TokenKind::Star),
    tok('\u{00F7}', "DIVISION SIGN", "/", TokenKind::Slash),
    tok('\u{2215}', "DIVISION SLASH", "/", TokenKind::Slash),
    tok('\u{2044}', "FRACTION SLASH", "/", TokenKind::Slash),
    tok('\u{FF0F}', "FULLWIDTH SOLIDUS", "/", TokenKind::Slash),
    tok('\u{FF0B}', "FULLWIDTH PLUS SIGN", "+", TokenKind::Plus),
    // Comparison, assignment and arrows.
    tok('\u{FF1D}', "FULLWIDTH EQUALS SIGN", "=", TokenKind::Eq),
    tok('\u{2260}', "NOT EQUAL TO", "!=", TokenKind::Ne),
    tok('\u{2264}', "LESS-THAN OR EQUAL TO", "<=", TokenKind::Le),
    tok('\u{2265}', "GREATER-THAN OR EQUAL TO", ">=", TokenKind::Ge),
    tok('\u{FF1C}', "FULLWIDTH LESS-THAN SIGN", "<", TokenKind::Lt),
    tok(
        '\u{FF1E}',
        "FULLWIDTH GREATER-THAN SIGN",
        ">",
        TokenKind::Gt,
    ),
    tok('\u{2192}', "RIGHTWARDS ARROW", "->", TokenKind::Arrow),
    tok(
        '\u{21D2}',
        "RIGHTWARDS DOUBLE ARROW",
        "=>",
        TokenKind::FatArrow,
    ),
    tok(
        '\u{FF01}',
        "FULLWIDTH EXCLAMATION MARK",
        "!",
        TokenKind::Bang,
    ),
    // Delimiters and separators.
    tok(
        '\u{FF08}',
        "FULLWIDTH LEFT PARENTHESIS",
        "(",
        TokenKind::LParen,
    ),
    tok(
        '\u{FF09}',
        "FULLWIDTH RIGHT PARENTHESIS",
        ")",
        TokenKind::RParen,
    ),
    tok(
        '\u{FF3B}',
        "FULLWIDTH LEFT SQUARE BRACKET",
        "[",
        TokenKind::LBracket,
    ),
    tok(
        '\u{FF3D}',
        "FULLWIDTH RIGHT SQUARE BRACKET",
        "]",
        TokenKind::RBracket,
    ),
    tok(
        '\u{FF5B}',
        "FULLWIDTH LEFT CURLY BRACKET",
        "{",
        TokenKind::LBrace,
    ),
    tok(
        '\u{FF5D}',
        "FULLWIDTH RIGHT CURLY BRACKET",
        "}",
        TokenKind::RBrace,
    ),
    tok('\u{FF0C}', "FULLWIDTH COMMA", ",", TokenKind::Comma),
    tok('\u{FF1B}', "FULLWIDTH SEMICOLON", ";", TokenKind::Semi),
    tok('\u{037E}', "GREEK QUESTION MARK", ";", TokenKind::Semi),
    tok('\u{FF1A}', "FULLWIDTH COLON", ":", TokenKind::Colon),
    tok('\u{FF0E}', "FULLWIDTH FULL STOP", ".", TokenKind::Dot),
    tok('\u{3002}', "IDEOGRAPHIC FULL STOP", ".", TokenKind::Dot),
    // Quotation marks: a lookalike cannot open a literal, so it is an error token.
    quote('\u{201C}', "LEFT DOUBLE QUOTATION MARK", "\""),
    quote('\u{201D}', "RIGHT DOUBLE QUOTATION MARK", "\""),
    quote('\u{FF02}', "FULLWIDTH QUOTATION MARK", "\""),
    quote('\u{2018}', "LEFT SINGLE QUOTATION MARK", "'"),
    quote('\u{2019}', "RIGHT SINGLE QUOTATION MARK", "'"),
    // Whitespace that is not ASCII whitespace.
    space('\u{00A0}', "NO-BREAK SPACE", " "),
    space('\u{2009}', "THIN SPACE", " "),
    space('\u{202F}', "NARROW NO-BREAK SPACE", " "),
    space('\u{3000}', "IDEOGRAPHIC SPACE", " "),
    space('\u{200B}', "ZERO WIDTH SPACE", ""),
    space('\u{FEFF}', "ZERO WIDTH NO-BREAK SPACE", ""),
];

/// The table entry for `ch`, if it is a known confusable.
pub(crate) fn lookup(ch: char) -> Option<&'static Confusable> {
    TABLE.iter().find(|c| c.ch == ch)
}

/// A Greek letter's spelled-out name (`α` → `alpha`), for the E0101 help on a non-ASCII identifier
/// character: ML code is written with α, β, γ, ε, σ and friends, which Wukong identifiers cannot
/// contain.
pub(crate) fn greek_name(ch: char) -> Option<&'static str> {
    Some(match ch {
        'α' | 'Α' => "alpha",
        'β' | 'Β' => "beta",
        'γ' | 'Γ' => "gamma",
        'δ' | 'Δ' => "delta",
        'ε' | 'ϵ' | 'Ε' => "epsilon",
        'ζ' | 'Ζ' => "zeta",
        'η' | 'Η' => "eta",
        'θ' | 'Θ' => "theta",
        'ι' | 'Ι' => "iota",
        'κ' | 'Κ' => "kappa",
        'λ' | 'Λ' => "lambda",
        'μ' | 'µ' | 'Μ' => "mu",
        'ν' | 'Ν' => "nu",
        'ξ' | 'Ξ' => "xi",
        'π' | 'Π' => "pi",
        'ρ' | 'Ρ' => "rho",
        'σ' | 'ς' | 'Σ' => "sigma",
        'τ' | 'Τ' => "tau",
        'φ' | 'ϕ' | 'Φ' => "phi",
        'χ' | 'Χ' => "chi",
        'ψ' | 'Ψ' => "psi",
        'ω' | 'Ω' => "omega",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry is a distinct non-ASCII character whose replacement is ASCII, and every token
    /// replacement is spelled exactly like the token it produces: the diagnostic tells the user to
    /// type `ascii`, and the parser then sees that token, so the two must agree.
    #[test]
    fn table_is_well_formed() {
        for (i, c) in TABLE.iter().enumerate() {
            assert!(!c.ch.is_ascii(), "{:?} is ASCII", c.ch);
            assert!(c.ascii.is_ascii(), "{} has a non-ASCII replacement", c.name);
            assert!(c.name.is_ascii() && !c.name.is_empty());
            assert!(
                TABLE[..i].iter().all(|d| d.ch != c.ch),
                "{} is listed twice",
                c.name
            );
            if let Recover::Token(kind) = c.recover {
                assert_eq!(kind.glyph(), Some(c.ascii), "{}", c.name);
            }
        }
        assert!(TABLE.len() >= 30, "the table shrank to {}", TABLE.len());
    }
}
