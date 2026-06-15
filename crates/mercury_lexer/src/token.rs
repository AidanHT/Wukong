//! Token kinds and the [`Token`] type.

use mercury_span::Span;

/// The lexical category of a token. Literal *values* are not stored here — they live in the
/// source text covered by the token's [`Span`] — so this enum is a cheap `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TokenKind {
    // Literals & names
    Ident,
    Int,
    Float,
    Str,
    Char,

    // Keywords
    Fn,
    Let,
    Mut,
    If,
    Else,
    While,
    For,
    In,
    Loop,
    Match,
    Return,
    Break,
    Continue,
    Struct,
    Enum,
    Impl,
    Trait,
    Module,
    Import,
    As,
    Const,
    Defer,
    Pub,
    Extern,
    Step,
    Where,
    True,
    False,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    // Punctuation
    Comma,
    Semi,
    Colon,
    ColonColon,
    Dot,
    DotDot,
    DotDotEq,
    Arrow,
    FatArrow,
    At,
    Question,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Bang,
    Tilde,
    Shl,
    Shr,
    AmpAmp,
    PipePipe,
    Eq,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,

    // Sentinels
    Eof,
    /// An unrecognized byte; carries a diagnostic and allows the lexer to keep going.
    Error,
}

impl TokenKind {
    /// Map an identifier string to its keyword kind, if any.
    pub fn keyword(s: &str) -> Option<TokenKind> {
        use TokenKind::*;
        Some(match s {
            "fn" => Fn,
            "let" => Let,
            "mut" => Mut,
            "if" => If,
            "else" => Else,
            "while" => While,
            "for" => For,
            "in" => In,
            "loop" => Loop,
            "match" => Match,
            "return" => Return,
            "break" => Break,
            "continue" => Continue,
            "struct" => Struct,
            "enum" => Enum,
            "impl" => Impl,
            "trait" => Trait,
            "module" => Module,
            "import" => Import,
            "as" => As,
            "const" => Const,
            "defer" => Defer,
            "pub" => Pub,
            "extern" => Extern,
            "step" => Step,
            "where" => Where,
            "true" => True,
            "false" => False,
            _ => return None,
        })
    }

    pub fn is_keyword(self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Fn | Let | Mut | If | Else | While | For | In | Loop | Match | Return | Break
                | Continue | Struct | Enum | Impl | Trait | Module | Import | As | Const | Defer
                | Pub | Extern | Step | Where | True | False
        )
    }

    /// A short human-readable description, used in "expected X" diagnostics. Punctuation
    /// returns its glyph in backticks-free form; categories return a noun.
    pub fn describe(self) -> &'static str {
        use TokenKind::*;
        match self {
            Ident => "identifier",
            Int => "integer literal",
            Float => "float literal",
            Str => "string literal",
            Char => "char literal",
            Eof => "end of file",
            Error => "invalid token",
            other => other.glyph().unwrap_or("token"),
        }
    }

    /// The canonical source text for fixed tokens (keywords and punctuation).
    pub fn glyph(self) -> Option<&'static str> {
        use TokenKind::*;
        Some(match self {
            Fn => "fn",
            Let => "let",
            Mut => "mut",
            If => "if",
            Else => "else",
            While => "while",
            For => "for",
            In => "in",
            Loop => "loop",
            Match => "match",
            Return => "return",
            Break => "break",
            Continue => "continue",
            Struct => "struct",
            Enum => "enum",
            Impl => "impl",
            Trait => "trait",
            Module => "module",
            Import => "import",
            As => "as",
            Const => "const",
            Defer => "defer",
            Pub => "pub",
            Extern => "extern",
            Step => "step",
            Where => "where",
            True => "true",
            False => "false",
            LParen => "(",
            RParen => ")",
            LBrace => "{",
            RBrace => "}",
            LBracket => "[",
            RBracket => "]",
            Comma => ",",
            Semi => ";",
            Colon => ":",
            ColonColon => "::",
            Dot => ".",
            DotDot => "..",
            DotDotEq => "..=",
            Arrow => "->",
            FatArrow => "=>",
            At => "@",
            Question => "?",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            Amp => "&",
            Pipe => "|",
            Caret => "^",
            Bang => "!",
            Tilde => "~",
            Shl => "<<",
            Shr => ">>",
            AmpAmp => "&&",
            PipePipe => "||",
            Eq => "=",
            EqEq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            PlusEq => "+=",
            MinusEq => "-=",
            StarEq => "*=",
            SlashEq => "/=",
            PercentEq => "%=",
            AmpEq => "&=",
            PipeEq => "|=",
            CaretEq => "^=",
            ShlEq => "<<=",
            ShrEq => ">>=",
            Ident | Int | Float | Str | Char | Eof | Error => return None,
        })
    }

    /// The variant name, used by the `--emit=tokens` dump and snapshot tests.
    pub fn name(self) -> &'static str {
        use TokenKind::*;
        match self {
            Ident => "Ident",
            Int => "Int",
            Float => "Float",
            Str => "Str",
            Char => "Char",
            Fn => "Fn",
            Let => "Let",
            Mut => "Mut",
            If => "If",
            Else => "Else",
            While => "While",
            For => "For",
            In => "In",
            Loop => "Loop",
            Match => "Match",
            Return => "Return",
            Break => "Break",
            Continue => "Continue",
            Struct => "Struct",
            Enum => "Enum",
            Impl => "Impl",
            Trait => "Trait",
            Module => "Module",
            Import => "Import",
            As => "As",
            Const => "Const",
            Defer => "Defer",
            Pub => "Pub",
            Extern => "Extern",
            Step => "Step",
            Where => "Where",
            True => "True",
            False => "False",
            LParen => "LParen",
            RParen => "RParen",
            LBrace => "LBrace",
            RBrace => "RBrace",
            LBracket => "LBracket",
            RBracket => "RBracket",
            Comma => "Comma",
            Semi => "Semi",
            Colon => "Colon",
            ColonColon => "ColonColon",
            Dot => "Dot",
            DotDot => "DotDot",
            DotDotEq => "DotDotEq",
            Arrow => "Arrow",
            FatArrow => "FatArrow",
            At => "At",
            Question => "Question",
            Plus => "Plus",
            Minus => "Minus",
            Star => "Star",
            Slash => "Slash",
            Percent => "Percent",
            Amp => "Amp",
            Pipe => "Pipe",
            Caret => "Caret",
            Bang => "Bang",
            Tilde => "Tilde",
            Shl => "Shl",
            Shr => "Shr",
            AmpAmp => "AmpAmp",
            PipePipe => "PipePipe",
            Eq => "Eq",
            EqEq => "EqEq",
            Ne => "Ne",
            Lt => "Lt",
            Le => "Le",
            Gt => "Gt",
            Ge => "Ge",
            PlusEq => "PlusEq",
            MinusEq => "MinusEq",
            StarEq => "StarEq",
            SlashEq => "SlashEq",
            PercentEq => "PercentEq",
            AmpEq => "AmpEq",
            PipeEq => "PipeEq",
            CaretEq => "CaretEq",
            ShlEq => "ShlEq",
            ShrEq => "ShrEq",
            Eof => "Eof",
            Error => "Error",
        }
    }
}

/// A lexed token: a kind plus the source span it covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Token {
        Token { kind, span }
    }
}
