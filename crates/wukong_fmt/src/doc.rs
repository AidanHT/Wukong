//! `wukongc --emit=doc`: a file's documentation as Markdown.
//!
//! The page is a pure function of the source: the file's title (its `module` path, or the name
//! the caller gives it), its `//!` docs, then every item in source order — a `##` heading, the
//! item's signature in a `wk` code block, and its `///` docs. A struct's fields, an enum's
//! variants, a trait's or an `impl` block's methods and an `extern` block's declarations follow
//! as a list, each entry with its own docs. Doc text is copied through as written (it is
//! Markdown), unindented the way `Docs::text` defines.
//!
//! Every signature is printed by the formatter's own printer (`printer.rs`), so it reads exactly
//! as `wukongc fmt` would write it — parameters wrap at the same width — except that comments
//! and blank lines are left out and a function stops before its body. Where the printer cannot
//! print a construct faithfully, the source text stands in.

use crate::printer::{Printer, Src};
use crate::{FmtError, FmtOptions};
use wukong_ast::*;
use wukong_diag::Diagnostic;
use wukong_span::{Interner, SourceId, Span};

/// The Markdown documentation of one source file. `file_name` titles a file with no `module`
/// header. A file that does not lex or parse is [`FmtError::Syntax`], with its errors.
pub fn document(
    src: &str,
    source: SourceId,
    file_name: &str,
    opts: &FmtOptions,
) -> Result<String, FmtError> {
    let (tokens, lex_diags) = wukong_lexer::tokenize(src, source);
    let mut interner = Interner::new();
    let parsed = wukong_parser::parse_module_tokens_with_lints(&tokens, src, &mut interner, 0);
    let errors: Vec<Diagnostic> = lex_diags
        .into_iter()
        .chain(parsed.diags)
        .filter(|d| d.is_error())
        .collect();
    if !errors.is_empty() {
        return Err(FmtError::Syntax(errors));
    }
    let ctx = Src::new(src, &tokens, &[], &interner);
    let w = Writer {
        ctx: &ctx,
        opts,
        src,
    };
    Ok(w.module(&parsed.module, file_name))
}

struct Writer<'a> {
    ctx: &'a Src<'a>,
    opts: &'a FmtOptions,
    src: &'a str,
}

impl<'a> Writer<'a> {
    fn module(&self, m: &Module, file_name: &str) -> String {
        let mut blocks: Vec<String> = Vec::new();
        blocks.push(match &m.name {
            Some(name) => {
                let path: Vec<&str> = name
                    .segments
                    .iter()
                    .map(|s| self.ctx.interner.resolve(s.sym))
                    .collect();
                format!("# Module {}", code(&path.join(".")))
            }
            None => format!("# {}", code(file_name)),
        });
        if !m.docs.is_empty() {
            blocks.push(m.docs.text());
        }
        let mut i = 0;
        while i < m.items.len() {
            let it = &m.items[i];
            // An `impl` block's methods follow its marker as flattened `fn` items.
            let methods = match &it.kind {
                ItemKind::Impl(d) => &m.items[i + 1..(i + 1 + d.methods.len()).min(m.items.len())],
                _ => &[],
            };
            self.item(it, methods, &mut blocks);
            i += 1 + methods.len();
        }
        let mut out = blocks.join("\n\n");
        out.push('\n');
        out
    }

    fn item(&self, it: &Item, methods: &[Item], blocks: &mut Vec<String>) {
        blocks.push(format!("## {}", code(&self.heading(it))));
        blocks.push(format!("```wk\n{}\n```", self.signature(it)));
        if !it.docs.is_empty() {
            blocks.push(it.docs.text());
        }
        let (title, entries): (&str, Vec<String>) = match &it.kind {
            ItemKind::Struct(d) => (
                "Fields",
                d.fields.iter().map(|f| self.field_entry(f)).collect(),
            ),
            ItemKind::Enum(d) => (
                "Variants",
                d.variants
                    .iter()
                    .map(|v| {
                        let sig = self.flat(|p| p.variant(v), v.span);
                        let fields: Vec<String> = match &v.data {
                            VariantData::Struct(fs) => fs
                                .iter()
                                .filter(|f| !f.docs.is_empty())
                                .map(|f| self.field_entry(f))
                                .collect(),
                            VariantData::Unit | VariantData::Tuple(_) => Vec::new(),
                        };
                        entry(&sig, &v.docs, &fields)
                    })
                    .collect(),
            ),
            ItemKind::Trait(t) => (
                "Methods",
                t.methods
                    .iter()
                    .enumerate()
                    .map(|(k, f)| {
                        let sig = self.flat(|p| p.fn_signature(f, &f.name), f.name.span);
                        entry(&sig, t.method_docs.get(k).unwrap_or(&Docs::default()), &[])
                    })
                    .collect(),
            ),
            ItemKind::Impl(d) => (
                "Methods",
                methods
                    .iter()
                    .zip(&d.methods)
                    .filter_map(|(mi, m)| match &mi.kind {
                        ItemKind::Fn(f) => {
                            let sig = self.flat(|p| p.fn_signature(f, &m.name), mi.span);
                            Some(entry(&sig, &mi.docs, &[]))
                        }
                        _ => None,
                    })
                    .collect(),
            ),
            ItemKind::Extern(x) => (
                "Declarations",
                x.items
                    .iter()
                    .enumerate()
                    .map(|(k, f)| {
                        let sig = self.flat(|p| p.fn_signature(f, &f.name), f.name.span);
                        entry(&sig, x.item_docs.get(k).unwrap_or(&Docs::default()), &[])
                    })
                    .collect(),
            ),
            ItemKind::Fn(_) | ItemKind::Const(_) | ItemKind::Import(_) | ItemKind::TypeAlias(_) => {
                ("", Vec::new())
            }
        };
        if !entries.is_empty() {
            blocks.push(format!("### {title}"));
            let loose = entries.iter().any(|e| e.contains('\n'));
            blocks.push(entries.join(if loose { "\n\n" } else { "\n" }));
        }
    }

    /// The item's `##` heading: its kind and name (an `impl` block's header, an `extern` block's
    /// ABI, an import's path).
    fn heading(&self, it: &Item) -> String {
        let name = |id: &Ident| self.ctx.interner.resolve(id.sym).to_string();
        match &it.kind {
            ItemKind::Fn(f) => format!("fn {}", name(&f.name)),
            ItemKind::Struct(d) => format!("struct {}", name(&d.name)),
            ItemKind::Enum(d) => format!("enum {}", name(&d.name)),
            ItemKind::Const(c) => format!("const {}", name(&c.name)),
            ItemKind::TypeAlias(a) => format!("type {}", name(&a.name)),
            ItemKind::Trait(t) => format!("trait {}", name(&t.name)),
            ItemKind::Import(im) => format!("import {}", self.path(&im.path)),
            ItemKind::Extern(x) => format!("extern {}", self.ctx.interner.resolve(x.abi)),
            ItemKind::Impl(d) => match &d.trait_ref {
                Some(tr) => format!("impl {} for {}", self.path(tr), self.path(&d.self_ty)),
                None => format!("impl {}", self.path(&d.self_ty)),
            },
        }
    }

    fn path(&self, p: &Path) -> String {
        let segs: Vec<&str> = p
            .segments
            .iter()
            .map(|s| self.ctx.interner.resolve(s.sym))
            .collect();
        segs.join(".")
    }

    /// The item's signature as the formatter prints it: a function without its body, an `impl`
    /// block's header without its methods, any other item whole; attributes on the line above.
    fn signature(&self, it: &Item) -> String {
        let mut p = Printer::new(self.ctx, self.opts, false);
        p.blank_lines = false;
        let body_lo = match &it.kind {
            ItemKind::Fn(f) => {
                p.attrs_line(&it.attrs, it.span.lo);
                p.fn_signature(f, &f.name);
                f.body.as_ref().map(|b| b.span.lo)
            }
            ItemKind::Impl(d) => {
                p.attrs_line(&it.attrs, it.span.lo);
                p.w("impl ");
                if let Some(tr) = &d.trait_ref {
                    p.dotted_path(tr);
                    p.w(" for ");
                }
                p.dotted_path(&d.self_ty);
                Some(d.self_ty.span.hi)
            }
            ItemKind::Struct(_)
            | ItemKind::Enum(_)
            | ItemKind::Const(_)
            | ItemKind::Import(_)
            | ItemKind::Extern(_)
            | ItemKind::Trait(_)
            | ItemKind::TypeAlias(_) => {
                p.item(it, &[], u32::MAX);
                None
            }
        };
        if p.err.is_some() || p.failed {
            // The source text instead: through the body's start for a function or `impl`.
            let hi = body_lo.unwrap_or(it.span.hi).max(it.span.lo);
            return self
                .text(Span::new(it.span.source, it.span.lo, hi))
                .to_string();
        }
        p.out.trim_end().to_string()
    }

    /// `f` printed on one line, or the source text of `fallback` when it cannot be.
    fn flat(&self, f: impl FnOnce(&mut Printer<'a>), fallback: Span) -> String {
        let p = Printer::new(self.ctx, self.opts, true);
        p.flat_of(f).unwrap_or_else(|| {
            self.text(fallback)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
    }

    fn field_entry(&self, f: &Field) -> String {
        let sig = self.flat(|p| p.field_decl(f), f.span);
        entry(&sig, &f.docs, &[])
    }

    fn text(&self, sp: Span) -> &'a str {
        self.src
            .get(sp.lo as usize..sp.hi as usize)
            .unwrap_or_default()
            .trim()
    }
}

/// One list entry: `- `sig``, then its docs and any nested entries, indented under it.
fn entry(sig: &str, docs: &Docs, nested: &[String]) -> String {
    let mut out = format!("- {}", code(sig));
    let mut body: Vec<String> = Vec::new();
    if !docs.is_empty() {
        body.push(docs.text());
    }
    if !nested.is_empty() {
        let loose = nested.iter().any(|e| e.contains('\n'));
        body.push(nested.join(if loose { "\n\n" } else { "\n" }));
    }
    for part in body {
        out.push_str("\n\n");
        out.push_str(&indent(&part, "  "));
    }
    out
}

/// Every non-empty line of `text` prefixed with `pad`.
fn indent(text: &str, pad: &str) -> String {
    text.split('\n')
        .map(|l| {
            if l.is_empty() {
                String::new()
            } else {
                format!("{pad}{l}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `s` as Markdown inline code: delimited by one more backtick than its longest run of them, and
/// padded with a space where it starts or ends with one.
fn code(s: &str) -> String {
    let mut longest = 0;
    let mut run = 0;
    for c in s.chars() {
        run = if c == '`' { run + 1 } else { 0 };
        longest = longest.max(run);
    }
    let fence = "`".repeat(longest + 1);
    if longest > 0 {
        format!("{fence} {s} {fence}")
    } else {
        format!("{fence}{s}{fence}")
    }
}

#[cfg(test)]
#[path = "doc_tests.rs"]
mod tests;
