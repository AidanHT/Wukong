//! The self-check: the four laws every output must satisfy before it is returned.
//!
//! - **(a) It parses cleanly.** Re-lexing and re-parsing the output with the *same* interner
//!   produces no lexer or parser error and no parser lint (`W0030`, `W0031`).
//! - **(b) It is the same AST.** The `Debug` rendering of the re-parsed module equals the
//!   original's once every span is masked (`lo..hi@src` → `_`). NodeIds and symbols are compared
//!   exactly, so a reformat that would renumber a node, or intern a different name, is caught.
//! - **(c) It has the same comments**, in the same order, with the same text.
//! - **(d) It has the same identifier and literal tokens** (sorted multiset of `Ident`, `Int`,
//!   `Float`, `Str`, `Char` and `Label` texts). This catches source text the AST does not keep: a
//!   struct field's or trait method's attributes, an enum's discriminant repr, tokens skipped
//!   inside an `extern` block.
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
                let k = match t.kind {
                    TokenKind::Ident => 0,
                    TokenKind::Int => 1,
                    TokenKind::Float => 2,
                    TokenKind::Str => 3,
                    TokenKind::Char => 4,
                    TokenKind::Label => 5,
                    _ => return None,
                };
                Some((k, src[t.span.lo as usize..t.span.hi as usize].to_string()))
            })
            .collect();
        v.sort();
        v
    };
    let (tw, tg) = (multiset(orig.src, orig.tokens), multiset(out, &tokens));
    if tw != tg {
        let missing = tw.iter().find(|t| !tg.contains(t));
        let extra = tg.iter().find(|t| !tw.contains(t));
        let what = match (missing, extra) {
            (Some(m), _) => format!("`{}` is missing from the output", m.1),
            (None, Some(x)) => format!("`{}` was added to the output", x.1),
            (None, None) => "a token occurs a different number of times".to_string(),
        };
        return Err(format!(
            "self-check (d) failed: the identifier/literal tokens differ: {what} (the AST does \
             not keep some source text here)"
        ));
    }
    Ok(())
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
    use super::mask_spans;

    #[test]
    fn spans_are_masked_and_nothing_else() {
        assert_eq!(
            mask_spans("Ident { sym: sym#3, span: 10..14@0 }, id: n12, Int(64), [1, 2]"),
            "Ident { sym: sym#3, span: _ }, id: n12, Int(64), [1, 2]"
        );
        assert_eq!(mask_spans("<dummy> 0..0@4294967295"), "_ _");
    }
}
