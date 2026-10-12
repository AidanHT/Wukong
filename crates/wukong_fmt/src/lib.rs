//! `wukong_fmt` — the Wukong source formatter behind `wukongc fmt`.
//!
//! An AST pretty-printer that keeps comments. Its one guarantee is that it never changes what a
//! program means: every result is re-lexed and re-parsed and must reproduce the input's AST
//! exactly (NodeIds included, spans aside), every comment in order, and every identifier and
//! literal token. When any of that fails the source is refused with [`FmtError::Unsupported`] and
//! nothing is rewritten — the formatter never emits a guess.
//!
//! The pipeline ([`format_source`]):
//!
//! 1. `wukong_lexer::tokenize_with_comments` — the token stream plus the comment side channel
//!    plain `tokenize` throws away. Any lexer error is [`FmtError::Syntax`].
//! 2. `wukong_parser::parse_module_tokens_with_lints` — any parser *error* is
//!    [`FmtError::Syntax`]; parser lints (`W0030`, `W0031`) in the input are allowed, and the
//!    output never has them.
//! 3. The printer (`printer.rs`, comment placement in `comments.rs`) — exhaustive over every AST
//!    enum with no `_ =>` arm, so new syntax fails this crate's build until it is printed.
//! 4. The self-check (`check.rs`) — the four laws above; a failure is
//!    [`FmtError::Unsupported`] naming the law and the place.
//! 5. Line endings: a file whose first line break is `\r\n` is written with `\r\n` throughout; a
//!    leading byte-order mark is kept.

mod check;
mod comments;
mod printer;

#[cfg(test)]
mod tests;

use wukong_diag::Diagnostic;
use wukong_span::{Interner, SourceId};

/// Formatting options.
#[derive(Clone, Debug)]
pub struct FmtOptions {
    /// The line width a signature, call, struct literal or array literal is wrapped at.
    pub max_width: usize,
}

impl Default for FmtOptions {
    fn default() -> Self {
        FmtOptions { max_width: 100 }
    }
}

/// Why a source was not formatted. In every case the caller must leave the file untouched.
#[derive(Clone, Debug)]
pub enum FmtError {
    /// The source does not lex or parse: the lexer's and parser's error diagnostics (warnings are
    /// not included). Render them as a compile would.
    Syntax(Vec<Diagnostic>),
    /// The formatter cannot prove its output means the same program: the message names the first
    /// failing self-check law (or the construct it cannot print) with some context.
    Unsupported(String),
}

impl std::fmt::Display for FmtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FmtError::Syntax(d) => write!(f, "{} syntax error(s)", d.len()),
            FmtError::Unsupported(m) => f.write_str(m),
        }
    }
}

/// Format one source file. `source` is the id its spans carry (any value works; diagnostics in a
/// [`FmtError::Syntax`] cite it). Returns the formatted text — identical to `src` when the file is
/// already formatted — or why it was refused.
pub fn format_source(src: &str, source: SourceId, opts: &FmtOptions) -> Result<String, FmtError> {
    let (tokens, comments, lex_diags) = wukong_lexer::tokenize_with_comments(src, source);
    let lex_errors: Vec<Diagnostic> = lex_diags.into_iter().filter(|d| d.is_error()).collect();
    if !lex_errors.is_empty() {
        return Err(FmtError::Syntax(lex_errors));
    }
    let mut interner = Interner::new();
    let parsed = wukong_parser::parse_module_tokens_with_lints(&tokens, src, &mut interner, 0);
    let errors: Vec<Diagnostic> = parsed.diags.into_iter().filter(|d| d.is_error()).collect();
    if !errors.is_empty() {
        return Err(FmtError::Syntax(errors));
    }
    let module = parsed.module;

    let ctx = printer::Src::new(src, &tokens, &comments, &interner);
    let body = printer::print_module(&ctx, opts, &module).map_err(FmtError::Unsupported)?;

    let crlf = first_line_break_is_crlf(src);
    let mut out = String::with_capacity(body.len() + body.len() / 16 + 3);
    if src.starts_with('\u{FEFF}') {
        out.push('\u{FEFF}');
    }
    if crlf {
        for (i, line) in body.split('\n').enumerate() {
            if i > 0 {
                out.push_str("\r\n");
            }
            out.push_str(line);
        }
    } else {
        out.push_str(&body);
    }

    check::self_check(
        &check::Original {
            src,
            tokens: &tokens,
            comments: &comments,
            module: &module,
            source,
        },
        &mut interner,
        &out,
    )
    .map_err(FmtError::Unsupported)?;
    Ok(out)
}

/// Whether the first line break in `src` is a `\r\n`.
fn first_line_break_is_crlf(src: &str) -> bool {
    match src.find('\n') {
        Some(i) => i > 0 && src.as_bytes()[i - 1] == b'\r',
        None => false,
    }
}
