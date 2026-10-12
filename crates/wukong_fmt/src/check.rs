//! The self-check: the four laws every output must satisfy before it is returned.
//!
//! - **(a) It parses cleanly.** Re-lexing and re-parsing the output with the *same* interner
//!   produces no lexer or parser error and no parser lint (`W0030`, `W0031`).
//! - **(b) It is the same AST.** The `Debug` rendering of the re-parsed module equals the
//!   original's once every span is masked (`lo..hi@src` → `_`). NodeIds and symbols are compared
//!   exactly, so a reformat that would renumber a node, or intern a different name, is caught.
//! - **(c) It has the same comments**, in the same order, with the same text.
//! - **(d) It has the same identifier, literal, keyword and `@` tokens** (the sorted multiset of
//!   `Ident`, `Int`, `Float`, `Str`, `Char` and `Label` texts, every keyword, and `@`). This catches
//!   source text the AST does not keep: a struct field's or trait method's attributes (a bare
//!   `@extern` holds no identifier at all), an enum's discriminant repr, a `pub` the parser
//!   discards before `import` or `impl`, and keyword tokens skipped inside an `extern` block.
//!   Punctuation is not counted; the printer refuses punctuation junk in an `extern` block itself.
//!
//! A failure names the law and about 80 characters of context. These laws are never relaxed to
//! let a file through: the printer is fixed instead, or the file stays refused.

use wukong_ast::Module;
use wukong_lexer::{Comment, CommentKind, Token, TokenKind};
use wukong_span::{Interner, SourceId};

/// The input side of the comparison.
pub(crate) struct Original<'a> {
    pub(crate) src: &'a str,
    pub(crate) tokens: &'a [Token],
    pub(crate) comments: &'a [Comment],
    pub(crate) module: &'a Module,
    pub(crate) source: SourceId,
}

pub(crate) fn self_check(
    orig: &Original<'_>,
    interner: &mut Interner,
    out: &str,
) -> Result<(), String> {
    // (a)
    let (tokens, comments, lex) = wukong_lexer::tokenize_with_comments(out, orig.source);
    if let Some(d) = lex.first() {
        return Err(format!(
            "self-check (a) failed: the output does not lex ({}) near `{}`",
            d.message,
            context(out, d.primary_span().map(|s| s.lo as usize).unwrap_or(0))
        ));
    }
    let parsed = wukong_parser::parse_module_tokens_with_lints(&tokens, out, interner, 0);
    if let Some(d) = parsed.diags.first().or(parsed.lints.first()) {
        return Err(format!(
            "self-check (a) failed: the output does not re-parse cleanly ({}{}) near `{}`",
            d.code.map(|c| format!("{c}: ")).unwrap_or_default(),
            d.message,
            context(out, d.primary_span().map(|s| s.lo as usize).unwrap_or(0))
        ));
    }

    // (b)
    let want = mask_spans(&format!("{:?}", orig.module));
    let got = mask_spans(&format!("{:?}", parsed.module));
    if want != got {
        let at = first_difference(&want, &got);
        return Err(format!(
            "self-check (b) failed: the output re-parses to a different AST: expected `{}`, got `{}`",
            context(&want, at),
            context(&got, at)
        ));
    }

    // (c)
    let texts = |src: &str, cs: &[Comment]| -> Vec<String> {
        cs.iter()
            .map(|c| match c.kind {
                CommentKind::Line => c.text(src).to_string(),
                CommentKind::Block => c.text(src).replace("\r\n", "\n"),
            })
            .collect()
    };
    let (cw, cg) = (texts(orig.src, orig.comments), texts(out, &comments));
    if cw != cg {
        let k = cw.iter().zip(&cg).take_while(|(a, b)| a == b).count();
        return Err(format!(
            "self-check (c) failed: comment {} differs: expected `{}`, got `{}`",
            k + 1,
            cw.get(k)
                .map(|s| clip(s))
                .unwrap_or_else(|| "<none>".into()),
            cg.get(k)
                .map(|s| clip(s))
                .unwrap_or_else(|| "<none>".into())
        ));
    }

    // (d)
    let multiset = |src: &str, ts: &[Token]| -> Vec<(u8, String)> {
        let mut v: Vec<(u8, String)> = ts
            .iter()
            .filter_map(|t| {
                let k = law_d_class(t.kind)?;
                Some((k, src[t.span.lo as usize..t.span.hi as usize].to_string()))
            })
            .collect();
        v.sort();
        v
    };
    let (tw, tg) = (multiset(orig.src, orig.tokens), multiset(out, &tokens));
    if let Some((text, want, got)) = first_count_difference(&tw, &tg) {
        return Err(format!(
            "self-check (d) failed: `{text}` occurs {want} time(s) in the source but {got} in the \
             output (the AST does not keep some source text here)"
        ));
    }
    Ok(())
}

/// The tokens law (d) counts, by class: identifiers, literals, labels, every keyword, and `@`.
/// These are what the printer must reproduce exactly. A keyword or an `@` can carry text the AST
/// never stores: a bare `@extern` field attribute, a `pub` the parser discards before `import`, a
/// keyword `parse_extern` skips. Punctuation, delimiters and operators are not counted: the
/// printer adds and removes `;`, `,` and parentheses as the canonical style requires, and law (b)
/// covers what they mean.
///
/// The match is exhaustive with no `_ =>` arm, so a new token kind fails this crate's build until
/// it is classified here.
fn law_d_class(k: TokenKind) -> Option<u8> {
    use TokenKind as T;
    Some(match k {
        T::Ident => 0,
        T::Int => 1,
        T::Float => 2,
        T::Str => 3,
        T::Char => 4,
        T::Label => 5,
        T::Fn
        | T::Let
        | T::Mut
        | T::If
        | T::Else
        | T::While
        | T::For
        | T::In
        | T::Loop
        | T::Match
        | T::Return
        | T::Break
        | T::Continue
        | T::Struct
        | T::Enum
        | T::Impl
        | T::Trait
        | T::Module
        | T::Import
        | T::As
        | T::Const
        | T::Defer
        | T::Pub
        | T::Extern
        | T::Step
        | T::Where
        | T::True
        | T::False => 6,
        T::At => 7,
        T::LParen
        | T::RParen
        | T::LBrace
        | T::RBrace
        | T::LBracket
        | T::RBracket
        | T::Comma
        | T::Semi
        | T::Colon
        | T::ColonColon
        | T::Dot
        | T::DotDot
        | T::DotDotEq
        | T::Arrow
        | T::FatArrow
        | T::Question
        | T::Plus
        | T::Minus
        | T::Star
        | T::Slash
        | T::Percent
        | T::Amp
        | T::Pipe
        | T::Caret
        | T::Bang
        | T::Tilde
        | T::Shl
        | T::Shr
        | T::AmpAmp
        | T::PipePipe
        | T::Eq
        | T::EqEq
        | T::Ne
        | T::Lt
        | T::Le
        | T::Gt
        | T::Ge
        | T::PlusEq
        | T::MinusEq
        | T::StarEq
        | T::SlashEq
        | T::PercentEq
        | T::AmpEq
        | T::PipeEq
        | T::CaretEq
        | T::ShlEq
        | T::ShrEq
        | T::Eof
        | T::Error => return None,
    })
}

/// The first token text (in sorted order) whose count differs between two sorted multisets, with
/// its count in each.
fn first_count_difference(
    want: &[(u8, String)],
    got: &[(u8, String)],
) -> Option<(String, usize, usize)> {
    let count = |v: &[(u8, String)], at: usize| v[at..].iter().take_while(|t| **t == v[at]).count();
    let (mut i, mut j) = (0, 0);
    while i < want.len() || j < got.len() {
        let next = match (want.get(i), got.get(j)) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let w = if want.get(i) == Some(next) {
            count(want, i)
        } else {
            0
        };
        let g = if got.get(j) == Some(next) {
            count(got, j)
        } else {
            0
        };
        if w != g {
            return Some((next.1.clone(), w, g));
        }
        i += w;
        j += g;
    }
    None
}

/// Replace every `Span` debug rendering (`lo..hi@src`, or `<dummy>`) with `_`.
pub(crate) fn mask_spans(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let mut copied = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
            // digits `..` digits `@` digits
            let mut j = i;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if b.get(j) == Some(&b'.') && b.get(j + 1) == Some(&b'.') {
                let mut k = j + 2;
                let d0 = k;
                while k < b.len() && b[k].is_ascii_digit() {
                    k += 1;
                }
                if k > d0 && b.get(k) == Some(&b'@') {
                    let mut m = k + 1;
                    let d1 = m;
                    while m < b.len() && b[m].is_ascii_digit() {
                        m += 1;
                    }
                    if m > d1 {
                        out.push_str(&s[copied..i]);
                        out.push('_');
                        copied = m;
                        i = m;
                        continue;
                    }
                }
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out.push_str(&s[copied..]);
    out.replace("<dummy>", "_")
}

fn first_difference(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// About 80 characters of `s` around byte `at`, on one line.
fn context(s: &str, at: usize) -> String {
    let mut lo = at.saturating_sub(40).min(s.len());
    while !s.is_char_boundary(lo) {
        lo -= 1;
    }
    let mut hi = (at + 40).min(s.len());
    while !s.is_char_boundary(hi) {
        hi += 1;
    }
    s[lo..hi].replace(['\n', '\r'], " ")
}

fn clip(s: &str) -> String {
    let mut t: String = s.chars().take(80).collect();
    if t.len() < s.len() {
        t.push('…');
    }
    t.replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::{first_count_difference, law_d_class, mask_spans};
    use wukong_lexer::TokenKind;

    /// `@` and every keyword are counted (a bare `@extern` holds no identifier); punctuation is
    /// not, because the printer adds and removes it.
    #[test]
    fn law_d_counts_keywords_and_at_but_not_punctuation() {
        for k in [
            TokenKind::At,
            TokenKind::Extern,
            TokenKind::Pub,
            TokenKind::Where,
        ] {
            assert!(law_d_class(k).is_some(), "{k:?}");
        }
        for k in [
            TokenKind::Semi,
            TokenKind::Comma,
            TokenKind::LParen,
            TokenKind::Eof,
        ] {
            assert!(law_d_class(k).is_none(), "{k:?}");
        }
    }

    #[test]
    fn a_count_difference_names_the_token_and_both_counts() {
        let v = |xs: &[(u8, &str)]| -> Vec<(u8, String)> {
            xs.iter().map(|(k, s)| (*k, s.to_string())).collect()
        };
        let a = v(&[(0, "x"), (6, "pub"), (6, "pub"), (7, "@")]);
        assert_eq!(first_count_difference(&a, &a), None);
        assert_eq!(
            first_count_difference(&a, &v(&[(0, "x"), (6, "pub"), (7, "@")])),
            Some(("pub".to_string(), 2, 1))
        );
        assert_eq!(
            first_count_difference(&a, &v(&[(0, "x"), (6, "pub"), (6, "pub")])),
            Some(("@".to_string(), 1, 0))
        );
        assert_eq!(
            first_count_difference(&v(&[(0, "x")]), &v(&[(0, "x"), (0, "y")])),
            Some(("y".to_string(), 0, 1))
        );
    }

    #[test]
    fn spans_are_masked_and_nothing_else() {
        assert_eq!(
            mask_spans("Ident { sym: sym#3, span: 10..14@0 }, id: n12, Int(64), [1, 2]"),
            "Ident { sym: sym#3, span: _ }, id: n12, Int(64), [1, 2]"
        );
        assert_eq!(mask_spans("<dummy> 0..0@4294967295"), "_ _");
    }
}
