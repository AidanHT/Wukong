//! Item-level parsing: the module header, functions, structs, enums, consts, imports, and
//! extern blocks, plus generics and parameter lists.

use super::Parser;
use wukong_ast::*;
use wukong_diag::Diagnostic;
use wukong_lexer::{Token, TokenKind};
use wukong_span::{Interner, SourceId};

use TokenKind as T;

/// Parse a whole module from already-lexed tokens (the driver's entry point — it tokenizes once
/// and renders lexer diagnostics itself, so this returns only parser diagnostics).
pub fn parse_module_tokens(
    tokens: &[Token],
    src: &str,
    interner: &mut Interner,
) -> (Module, Vec<Diagnostic>) {
    let (m, diags, _) = parse_module_tokens_from(tokens, src, interner, 0);
    (m, diags)
}

/// Like [`parse_module_tokens`], but `NodeId`s are allocated starting at `first_node_id`, and the
/// first id *after* the last allocated one is returned alongside the module.
///
/// This is the multi-file entry point: `NodeId`s are only unique within one `Parser` run, yet sema
/// and mir_build key their side tables (`types`, `consts`, …) by `NodeId` across the whole merged
/// program. The import loader threads the returned watermark into the next file's parse so every
/// file's ids occupy a disjoint range and the merged module has globally unique `NodeId`s.
pub fn parse_module_tokens_from(
    tokens: &[Token],
    src: &str,
    interner: &mut Interner,
    first_node_id: u32,
) -> (Module, Vec<Diagnostic>, u32) {
    let out = parse_module_tokens_with_lints(tokens, src, interner, first_node_id);
    (out.module, out.diags, out.next_node)
}

/// Everything one parse produces: see [`parse_module_tokens_with_lints`].
pub struct ParseOutput {
    pub module: Module,
    /// Parser errors (the lexer's diagnostics are the caller's, as for the other entry points).
    pub diags: Vec<Diagnostic>,
    /// The first `NodeId` after the last one allocated (see [`parse_module_tokens_from`]).
    pub next_node: u32,
    /// The parser's lint findings, as warnings with their lint code: `W0030` (a statement's `;` is
    /// missing) and `W0031` (a `(`, `[` or touching `-` at the start of a line continues the
    /// expression above it). Their levels (`-A/-W/-D`, `@allow` …) are not applied here.
    pub lints: Vec<Diagnostic>,
}

/// [`parse_module_tokens_from`], plus the parser's lint findings in [`ParseOutput::lints`] — the
/// driver's entry point. The other entry points drop the lints, so a caller that asserts a parse
/// produced no diagnostic is not failed by a warning.
pub fn parse_module_tokens_with_lints(
    tokens: &[Token],
    src: &str,
    interner: &mut Interner,
    first_node_id: u32,
) -> ParseOutput {
    let mut p = Parser::new(tokens, src, interner);
    p.next_node = first_node_id;
    let module = p.module();
    ParseOutput {
        module,
        diags: std::mem::take(&mut p.diags),
        next_node: p.next_node,
        lints: std::mem::take(&mut p.lints),
    }
}

/// Convenience entry that lexes and parses in one step (tests / standalone use). Returns both
/// lexer and parser diagnostics.
pub fn parse_module(
    src: &str,
    source: SourceId,
    interner: &mut Interner,
) -> (Module, Vec<Diagnostic>) {
    let (tokens, mut diags) = wukong_lexer::tokenize(src, source);
    let (m, pdiags) = parse_module_tokens(&tokens, src, interner);
    diags.extend(pdiags);
    (m, diags)
}

impl Parser<'_> {
    pub(crate) fn module(&mut self) -> Module {
        let start = self.span();
        let name = if self.at(T::Module) {
            self.bump();
            let path = self.parse_dotted_path();
            self.eat(T::Semi);
            Some(path)
        } else {
            None
        };
        let mut items = Vec::new();
        while !self.at(T::Eof) {
            let before = self.pos;
            items.extend(self.parse_item());
            // Guarantee forward progress even if recovery stalled.
            if self.pos == before && !self.at(T::Eof) {
                self.bump();
            }
        }
        Module {
            name,
            items,
            span: start.to(self.prev_span()),
        }
    }

    /// One source item. Usually exactly one AST item; an `impl` block yields its marker item plus
    /// one flattened `fn` item per method, and an unparseable item yields none.
    fn parse_item(&mut self) -> Vec<Item> {
        let start = self.span();
        let attrs = self.parse_attrs();
        let is_pub = self.eat(T::Pub);
        let kind = match self.kind() {
            T::Fn => ItemKind::Fn(self.parse_fn(is_pub)),
            T::Struct => ItemKind::Struct(self.parse_struct(is_pub)),
            T::Enum => ItemKind::Enum(self.parse_enum(is_pub)),
            T::Const => ItemKind::Const(self.parse_const(is_pub)),
            T::Import => ItemKind::Import(self.parse_import()),
            T::Extern => ItemKind::Extern(self.parse_extern(is_pub)),
            T::Trait => ItemKind::Trait(self.parse_trait(is_pub)),
            T::Impl => return self.parse_impl(start, attrs),
            _ => {
                let sp = self.span();
                self.error(
                    sp,
                    "E0208",
                    format!(
                        "expected an item (fn, struct, enum, const, import, extern, trait, impl), \
                         found {}",
                        self.found()
                    ),
                );
                self.recover_item();
                return Vec::new();
            }
        };
        vec![Item {
            id: self.nid(),
            attrs,
            kind,
            span: start.to(self.prev_span()),
        }]
    }

    fn recover_item(&mut self) {
        loop {
            match self.kind() {
                T::Eof
                | T::Fn
                | T::Struct
                | T::Enum
                | T::Const
                | T::Import
                | T::Extern
                | T::Trait
                | T::Impl
                | T::Pub
                | T::At => break,
                _ => {
                    self.bump();
                }
            }
        }
    }

    fn parse_generics(&mut self) -> Vec<GenericParam> {
        let mut gs = Vec::new();
        if !self.eat(T::Lt) {
            return gs;
        }
        while !self.at(T::Gt) && !self.at(T::Eof) {
            let start = self.span();
            let mut bounds = Vec::new();
            let kind = if self.eat(T::Const) {
                let name = self.ident();
                self.expect(T::Colon);
                let ty = self.parse_type();
                GenericParamKind::Const { name, ty }
            } else {
                let name = self.ident();
                // `<T: Shape + Named>` — trait bounds on a type parameter.
                if self.eat(T::Colon) {
                    bounds = self.parse_bounds();
                }
                GenericParamKind::Type(name)
            };
            gs.push(GenericParam {
                kind,
                bounds,
                span: start.to(self.prev_span()),
            });
            if !self.eat(T::Comma) {
                break;
            }
        }
        self.expect(T::Gt);
        gs
    }

    /// `Bound (+ Bound)*` — the trait list after a `:` in a generic parameter or `where` clause.
    fn parse_bounds(&mut self) -> Vec<Path> {
        let mut bounds = vec![self.parse_dotted_path()];
        while self.eat(T::Plus) {
            bounds.push(self.parse_dotted_path());
        }
        bounds
    }

    /// Is the token `n` ahead the identifier `self`?
    fn at_self_word(&self, n: usize) -> bool {
        self.nth(n) == T::Ident && {
            let sp = self.tokens[self.pos + n].span;
            &self.src[sp.lo as usize..sp.hi as usize] == "self"
        }
    }

    /// A method's receiver, when the parameter list starts with one: `self`, `mut self`, `&self`
    /// or `&mut self`, desugared to a parameter named `self` of type `receiver`. An aggregate is
    /// passed by reference already, so `&self` means `self` and `&mut self` means `mut self` —
    /// the in-place, caller-visible mutation `mut` opts an aggregate parameter into. `self: T`
    /// (an explicit type) is an ordinary parameter and is left to `parse_params`.
    fn parse_receiver(&mut self, receiver: &Path) -> Option<Param> {
        let start = self.span();
        let (skip, mutable) = match self.kind() {
            T::Amp if self.nth(1) == T::Mut && self.at_self_word(2) => (2, true),
            T::Amp if self.at_self_word(1) => (1, false),
            T::Mut if self.at_self_word(1) => (1, true),
            T::Ident if self.at_self_word(0) => (0, false),
            _ => return None,
        };
        if self.nth(skip + 1) == T::Colon {
            return None;
        }
        for _ in 0..skip {
            self.bump();
        }
        let name = self.ident();
        let mut path = receiver.clone();
        path.span = name.span;
        for seg in &mut path.segments {
            seg.span = name.span;
        }
        let ty = TypeExpr {
            id: self.nid(),
            kind: TypeKind::Path(path),
            span: name.span,
        };
        Some(Param {
            id: self.nid(),
            attrs: Vec::new(),
            mutable,
            name,
            ty,
            span: start.to(self.prev_span()),
        })
    }

    fn parse_params(&mut self, receiver: Option<&Path>) -> Vec<Param> {
        let mut params = Vec::new();
        self.expect(T::LParen);
        if let Some(r) = receiver {
            if let Some(p) = self.parse_receiver(r) {
                params.push(p);
                if !self.eat(T::Comma) {
                    self.expect(T::RParen);
                    return params;
                }
            }
        }
        while !self.at(T::RParen) && !self.at(T::Eof) {
            let start = self.span();
            let attrs = self.parse_attrs();
            // Optional `mut`: `fn f(mut p: T)` marks the parameter mutable (reassignable, and for
            // an aggregate — passed by reference — mutable in place with the change visible to the
            // caller). Without it the parameter is immutable (sema enforces this, E0304).
            let mutable = self.eat(T::Mut);
            let name = self.ident();
            self.expect(T::Colon);
            let ty = self.parse_type();
            params.push(Param {
                id: self.nid(),
                attrs,
                mutable,
                name,
                ty,
                span: start.to(self.prev_span()),
            });
            if !self.eat(T::Comma) {
                break;
            }
        }
        self.expect(T::RParen);
        params
    }

    fn parse_fn(&mut self, is_pub: bool) -> FnDecl {
        self.parse_fn_with(is_pub, None)
    }

    /// A `fn` item; `receiver` is the type a leading `self` parameter has (inside an `impl` or a
    /// `trait`), and `None` everywhere else.
    fn parse_fn_with(&mut self, is_pub: bool, receiver: Option<&Path>) -> FnDecl {
        self.bump(); // fn
        let name = self.ident();
        let mut generics = self.parse_generics();
        let params = self.parse_params(receiver);
        let ret = if self.eat(T::Arrow) {
            Some(self.parse_type())
        } else {
            None
        };
        // `where T: Shape + Named, U: Other` — folded into the named generic parameter's bounds,
        // exactly as if it had been written `<T: Shape + Named>`. A clause naming no type
        // parameter of this function is reported rather than dropped. Any other entry is a
        // dimension predicate (`K % 16 == 0`, `M <= 4096`), kept for sema to check at each call; it
        // is parsed with struct literals off, so the body's `{` ends it.
        let mut where_preds = Vec::new();
        if self.eat(T::Where) {
            while !self.at(T::LBrace) && !self.at(T::Eq) && !self.at(T::Semi) && !self.at(T::Eof) {
                if !(self.at(T::Ident) && self.nth(1) == T::Colon) {
                    where_preds.push(self.parse_cond());
                    if !self.eat(T::Comma) {
                        break;
                    }
                    continue;
                }
                let gname = self.ident();
                self.expect(T::Colon);
                let bounds = self.parse_bounds();
                let target = generics
                    .iter_mut()
                    .find(|g| matches!(&g.kind, GenericParamKind::Type(id) if id.sym == gname.sym));
                match target {
                    Some(g) => g.bounds.extend(bounds),
                    None => {
                        let nm = self.interner.resolve(gname.sym).to_string();
                        self.error(
                            gname.span,
                            "E0315",
                            format!(
                                "`where` clause names `{nm}`, which is not a type parameter of \
                                 this function"
                            ),
                        );
                    }
                }
                if !self.eat(T::Comma) {
                    break;
                }
            }
        }
        let body = if self.at(T::LBrace) {
            Some(self.parse_block())
        } else if self.eat(T::Eq) {
            let e = self.parse_expr();
            let sp = e.span;
            self.eat(T::Semi);
            Some(Block {
                id: self.nid(),
                stmts: Vec::new(),
                tail: Some(Box::new(e)),
                span: sp,
            })
        } else {
            self.eat(T::Semi);
            None
        };
        FnDecl {
            name,
            is_pub,
            generics,
            params,
            ret,
            where_preds,
            body,
        }
    }

    /// `trait Name { fn m(self, …) -> R; … }`. Inside, `Self` stays the placeholder `Self`, which
    /// sema substitutes with each implementing type.
    fn parse_trait(&mut self, is_pub: bool) -> TraitDecl {
        self.bump(); // trait
        let name = self.ident();
        let self_sym = self.interner.intern("Self");
        let receiver = Path {
            segments: vec![Ident {
                sym: self_sym,
                span: name.span,
            }],
            span: name.span,
        };
        let mut methods = Vec::new();
        self.expect(T::LBrace);
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            let _attrs = self.parse_attrs();
            if self.at(T::Fn) {
                methods.push(self.parse_fn_with(false, Some(&receiver)));
            } else {
                let sp = self.span();
                self.error(
                    sp,
                    "E0208",
                    format!(
                        "expected a method signature (`fn …;`) in a trait, found {}",
                        self.found()
                    ),
                );
                self.recover_member();
            }
        }
        self.expect(T::RBrace);
        TraitDecl {
            name,
            is_pub,
            methods,
        }
    }

    /// `impl Type { fn … }` or `impl Trait for Type { fn … }`. Each method becomes a top-level
    /// function item named `Type.method` (its receiver, if any, desugared to `self: Type`, and
    /// every `Self` in the block read as `Type`), preceded by an [`ItemKind::Impl`] marker that
    /// records the block for sema.
    fn parse_impl(&mut self, start: wukong_span::Span, attrs: Vec<Attr>) -> Vec<Item> {
        self.bump(); // impl
        let first = self.parse_dotted_path();
        let (trait_ref, self_ty) = if self.eat(T::For) {
            (Some(first), self.parse_dotted_path())
        } else {
            (None, first)
        };
        let ty_name = self_ty.segments.last().map(|s| s.sym);
        let mut out = Vec::new();
        let mut methods = Vec::new();
        self.expect(T::LBrace);
        let saved = std::mem::replace(&mut self.self_ty, ty_name);
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            let mstart = self.span();
            let mattrs = self.parse_attrs();
            let is_pub = self.eat(T::Pub);
            if !self.at(T::Fn) {
                let sp = self.span();
                self.error(
                    sp,
                    "E0208",
                    format!(
                        "expected a method (`fn`) in an `impl` block, found {}",
                        self.found()
                    ),
                );
                self.recover_member();
                continue;
            }
            let mut f = self.parse_fn_with(is_pub, Some(&self_ty));
            let short = f.name;
            if let Some(ty) = ty_name {
                let mangled = format!(
                    "{}.{}",
                    self.interner.resolve(ty),
                    self.interner.resolve(short.sym)
                );
                f.name = Ident {
                    sym: self.interner.intern(&mangled),
                    span: short.span,
                };
            }
            methods.push(ImplMethod {
                name: short,
                fn_name: f.name.sym,
            });
            out.push(Item {
                id: self.nid(),
                attrs: mattrs,
                kind: ItemKind::Fn(f),
                span: mstart.to(self.prev_span()),
            });
        }
        self.self_ty = saved;
        self.expect(T::RBrace);
        let marker = Item {
            id: self.nid(),
            attrs,
            kind: ItemKind::Impl(ImplDecl {
                self_ty,
                trait_ref,
                methods,
            }),
            span: start.to(self.prev_span()),
        };
        out.insert(0, marker);
        out
    }

    /// Skip to the next member of a `trait`/`impl` body after a parse error: the next `fn`, `@`,
    /// `pub`, or the closing `}` (left for the caller).
    fn recover_member(&mut self) {
        loop {
            match self.kind() {
                T::Eof | T::RBrace | T::Fn | T::Pub | T::At => break,
                _ => {
                    self.bump();
                }
            }
        }
    }

    fn parse_struct(&mut self, is_pub: bool) -> StructDecl {
        self.bump(); // struct
        let name = self.ident();
        let generics = self.parse_generics();
        let mut fields = Vec::new();
        self.expect(T::LBrace);
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            let _attrs = self.parse_attrs(); // field attributes accepted, not yet stored
            let fstart = self.span();
            let is_pub = self.eat(T::Pub);
            let fname = self.ident();
            self.expect(T::Colon);
            let ty = self.parse_type();
            fields.push(Field {
                name: fname,
                is_pub,
                ty,
                span: fstart.to(self.prev_span()),
            });
            if !self.eat(T::Comma) {
                break;
            }
        }
        self.expect(T::RBrace);
        StructDecl {
            name,
            is_pub,
            generics,
            fields,
        }
    }

    fn parse_enum(&mut self, is_pub: bool) -> EnumDecl {
        self.bump(); // enum
        let name = self.ident();
        if self.eat(T::Colon) {
            let _repr = self.parse_type(); // discriminant repr accepted, not yet stored
        }
        let generics = self.parse_generics();
        let mut variants = Vec::new();
        self.expect(T::LBrace);
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            let vstart = self.span();
            let vname = self.ident();
            let data = if self.at(T::LParen) {
                self.bump();
                let mut tys = Vec::new();
                while !self.at(T::RParen) && !self.at(T::Eof) {
                    tys.push(self.parse_type());
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RParen);
                VariantData::Tuple(tys)
            } else if self.at(T::LBrace) {
                self.bump();
                let mut fs = Vec::new();
                while !self.at(T::RBrace) && !self.at(T::Eof) {
                    let fstart = self.span();
                    let is_pub = self.eat(T::Pub);
                    let fname = self.ident();
                    self.expect(T::Colon);
                    let ty = self.parse_type();
                    fs.push(Field {
                        name: fname,
                        is_pub,
                        ty,
                        span: fstart.to(self.prev_span()),
                    });
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RBrace);
                VariantData::Struct(fs)
            } else {
                VariantData::Unit
            };
            let discriminant = if self.eat(T::Eq) {
                Some(self.parse_expr())
            } else {
                None
            };
            variants.push(Variant {
                name: vname,
                data,
                discriminant,
                span: vstart.to(self.prev_span()),
            });
            if !self.eat(T::Comma) {
                break;
            }
        }
        self.expect(T::RBrace);
        EnumDecl {
            name,
            is_pub,
            generics,
            variants,
        }
    }

    fn parse_const(&mut self, is_pub: bool) -> ConstDecl {
        self.bump(); // const
        let name = self.ident();
        self.expect(T::Colon);
        let ty = self.parse_type();
        self.expect(T::Eq);
        let value = self.parse_expr();
        self.eat(T::Semi);
        ConstDecl {
            name,
            is_pub,
            ty,
            value,
        }
    }

    fn parse_import(&mut self) -> Import {
        self.bump(); // import
        let path = self.parse_dotted_path();
        let mut alias = None;
        let mut items = None;
        if self.eat(T::As) {
            alias = Some(self.ident());
        } else if self.at(T::Dot) && self.nth(1) == T::LBrace {
            self.bump(); // .
            self.bump(); // {
            let mut names = Vec::new();
            while !self.at(T::RBrace) && !self.at(T::Eof) {
                names.push(self.ident());
                if !self.eat(T::Comma) {
                    break;
                }
            }
            self.expect(T::RBrace);
            items = Some(names);
        }
        self.eat(T::Semi);
        Import { path, alias, items }
    }

    /// `[pub] extern ["C"] { [pub] fn ...; ... }` or `[pub] extern ["C"] fn ...;`. A `pub` on the
    /// block exports every declaration in it; a `pub` on one declaration exports just that one.
    fn parse_extern(&mut self, is_pub: bool) -> ExternBlock {
        self.bump(); // extern
        let abi = if self.at(T::Str) {
            let s = self.intern_span(self.span());
            self.bump();
            s
        } else {
            self.interner.intern("\"C\"")
        };
        let mut items = Vec::new();
        if self.eat(T::LBrace) {
            while !self.at(T::RBrace) && !self.at(T::Eof) {
                let item_pub = self.at(T::Pub) && self.nth(1) == T::Fn;
                if item_pub {
                    self.bump(); // pub
                }
                if self.at(T::Fn) {
                    items.push(self.parse_fn(is_pub || item_pub));
                } else {
                    self.bump();
                }
            }
            self.expect(T::RBrace);
        } else if self.at(T::Fn) {
            items.push(self.parse_fn(is_pub));
        }
        ExternBlock { abi, items }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_ast::print;

    fn parse(src: &str) -> String {
        let mut i = Interner::new();
        let (m, diags) = parse_module(src, SourceId(0), &mut i);
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
        print::print_module(&m, &i)
    }

    #[test]
    fn module_and_fn() {
        let out = parse("module demo\nfn add(a: i32, b: i32) -> i32 { return a + b; }\n");
        assert!(out.starts_with("module demo\n"));
        assert!(out.contains("fn add"));
        assert!(out.contains("param a: i32"));
        assert!(out.contains("ret i32"));
        assert!(out.contains("binary +"));
    }

    #[test]
    fn generic_fn_with_tensors() {
        let out = parse("fn mm<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N]) {}\n");
        assert!(out.contains("type-param M"));
        assert!(out.contains("param a: Tensor[f32, M, K]"));
    }

    #[test]
    fn struct_and_const() {
        let out = parse("struct Vec3 { x: f32, y: f32, z: f32 }\nconst N: usize = 8;\n");
        assert!(out.contains("struct Vec3"));
        assert!(out.contains("field x: f32"));
        assert!(out.contains("const N: usize"));
    }

    #[test]
    fn parallel_for_attribute() {
        let out = parse("fn k<N>(x: Tensor[f32, N]) { @parallel @simd for i in 0..N { } }\n");
        assert!(out.contains("@parallel"));
        assert!(out.contains("@simd"));
        assert!(out.contains("for"));
        assert!(out.contains("range"));
    }

    /// `@extern("C")` (examples/vadd.wk) still parses — its name is a keyword token.
    #[test]
    fn extern_attribute_still_parses() {
        let out = parse("@export(\"vadd_f32\")\n@extern(\"C\")\nfn vadd(a: *f32) {}\n");
        assert!(out.contains("@extern"), "{out}");
        assert!(out.contains("@export"), "{out}");
    }

    /// A stray `@` must cost one diagnostic at the `@` rather than eat the `fn` that follows it.
    /// Before, this file produced a single E0208 pointing at `main` and the module contained only
    /// `helper` — `fn main` had been silently deleted.
    #[test]
    fn stray_at_does_not_consume_the_following_item() {
        let mut i = Interner::new();
        let src =
            "module t\nfn helper() -> i32 { return 1; }\n@\nfn main() -> i32 { return helper(); }\n";
        let (module, diags) = parse_module(src, SourceId(0), &mut i);
        assert!(!diags.is_empty(), "a stray `@` must be diagnosed");
        let names: Vec<String> = module
            .items
            .iter()
            .filter_map(|it| match &it.kind {
                ItemKind::Fn(f) => Some(i.resolve(f.name.sym).to_string()),
                _ => None,
            })
            .collect();
        assert!(
            names.contains(&"main".to_string()),
            "lost `main`: {names:?}"
        );
        assert!(
            names.contains(&"helper".to_string()),
            "lost `helper`: {names:?}"
        );
    }

    /// A stray `@` inside a block must not eat the `let` that follows it.
    #[test]
    fn stray_at_does_not_consume_the_following_statement() {
        let mut i = Interner::new();
        let src = "fn main() -> i32 {\n    @\n    let x: i32 = 5;\n    return x;\n}\n";
        let (module, diags) = parse_module(src, SourceId(0), &mut i);
        assert!(!diags.is_empty(), "a stray `@` must be diagnosed");
        let out = print::print_module(&module, &i);
        assert!(out.contains("let"), "the `let` binding was shredded: {out}");
        assert!(out.contains("type i32"), "{out}");
    }

    /// Struct functional update `P { x: 9, ..b }` is unsupported syntax. It must cost ONE accurate
    /// diagnostic and leave the parser in sync: before, it produced five (E0201, E0200, E0202,
    /// E0200, E0208), the last proving the parser had fallen out of the body into module scope.
    #[test]
    fn struct_functional_update_reports_once_and_stays_in_sync() {
        let mut i = Interner::new();
        let src = "struct P { x: i32, y: i32 }\nfn main() -> i32 {\n    let b: P = P { x: 1, y: 2 };\n    let c: P = P { x: 9, ..b };\n    return c.x;\n}\n";
        let (module, diags) = parse_module(src, SourceId(0), &mut i);
        assert_eq!(
            diags.iter().filter(|d| d.is_error()).count(),
            1,
            "one construct, one diagnostic, got {diags:?}"
        );
        // Still in sync: both items survived, so the parser never fell back to module scope.
        assert_eq!(module.items.len(), 2, "lost an item: {diags:?}");

        // A base-less `..` and a trailing `..` cost one diagnostic too — the recovery must not ask
        // for an expression that is not there and eat the `}` closing the literal.
        for body in ["P { .. }", "P { x: 1, .. }", "P { ..b }"] {
            let src =
                format!("struct P {{ x: i32, y: i32 }}\nfn main() -> i32 {{ let c: P = {body}; return c.x; }}\n");
            let mut i = Interner::new();
            let (m, d) = parse_module(&src, SourceId(0), &mut i);
            assert_eq!(
                d.iter().filter(|x| x.is_error()).count(),
                1,
                "`{body}` should cost one diagnostic, got {d:?}"
            );
            assert_eq!(m.items.len(), 2, "`{body}` lost an item");
        }
    }

    /// An attribute argument with a missing value must not swallow the `)` that closes the argument
    /// list. Before, the `)` became the value (the printer showed `@parallel(grain = ))`) and the
    /// sole diagnostic was "expected `)`" pointing at the `for` on the next line.
    #[test]
    fn attribute_missing_value_keeps_the_closing_paren() {
        let mut i = Interner::new();
        let src = "fn main() -> i32 {\n    let mut s: i32 = 0;\n    @parallel(grain = )\n    for i in 0..4 { s += i; }\n    return s;\n}\n";
        let (module, diags) = parse_module(src, SourceId(0), &mut i);
        assert!(
            diags.iter().any(|d| d.code == Some("E0207")),
            "expected E0207 at the missing value, got {diags:?}"
        );
        assert!(
            !diags.iter().any(|d| d.code == Some("E0200")),
            "the `)` must not have been consumed, got {diags:?}"
        );
        let out = print::print_module(&module, &i);
        assert!(out.contains("for"), "the loop was lost: {out}");
    }

    /// An `impl` block flattens into one top-level `fn Type.method` per method, preceded by the
    /// marker item; the receiver desugars to `self: Type`, and `Self` reads as the type.
    #[test]
    fn impl_block_flattens_into_named_functions() {
        let out = parse(
            "struct V { x: f32 }\n\
             impl V {\n\
                 fn get(self) -> f32 { return self.x; }\n\
                 fn bump(&mut self, k: f32) { self.x = self.x + k; }\n\
                 fn peek(&self) -> f32 { return self.x; }\n\
                 fn make() -> Self { return Self { x: 1.0 }; }\n\
             }\n",
        );
        assert!(out.contains("impl V"), "{out}");
        assert!(out.contains("method get = fn V.get"), "{out}");
        assert!(out.contains("fn V.get"), "{out}");
        assert!(out.contains("param self: V"), "{out}");
        assert!(out.contains("param mut self: V"), "{out}");
        assert!(out.contains("fn V.make"), "{out}");
        assert!(out.contains("ret V"), "{out}");
        assert!(
            !out.contains("Self"),
            "every `Self` is the impl's type: {out}"
        );
    }

    /// `impl Trait for Type`, a trait of signatures with receivers, bounds in the generic list and
    /// a `where` clause folded into them, and a function type.
    #[test]
    fn traits_bounds_and_function_types_parse() {
        let out = parse(
            "trait Shape { fn area(self) -> f32; fn grow(mut self, k: f32); fn unit() -> Self; }\n\
             struct S { a: f32 }\n\
             impl Shape for S {\n\
                 fn area(self) -> f32 { return self.a; }\n\
                 fn grow(mut self, k: f32) { self.a = self.a * k; }\n\
                 fn unit() -> S { return S { a: 1.0 }; }\n\
             }\n\
             fn total<T: Shape + Shape, U>(x: T, y: U) -> f32 where U: Shape { return 0.0; }\n\
             fn apply(f: fn(f32, i32) -> f32, g: fn(f32)) -> f32 { return f(1.0, 2); }\n",
        );
        assert!(out.contains("trait Shape"), "{out}");
        assert!(out.contains("fn area(self: Self) -> f32"), "{out}");
        assert!(out.contains("fn grow(mut self: Self, k: f32)"), "{out}");
        assert!(out.contains("fn unit() -> Self"), "{out}");
        assert!(out.contains("impl Shape for S"), "{out}");
        assert!(out.contains("type-param T: Shape + Shape"), "{out}");
        assert!(out.contains("type-param U: Shape"), "{out}");
        assert!(out.contains("param f: fn(f32, i32) -> f32"), "{out}");
        assert!(out.contains("param g: fn(f32)"), "{out}");
    }

    /// A `where` clause naming no type parameter of its function is reported, not dropped.
    #[test]
    fn a_where_clause_on_an_unknown_parameter_is_reported() {
        let mut i = Interner::new();
        let src = "fn f<T>(x: T) -> i32 where U: Shape { return 0; }\n";
        let (_, diags) = parse_module(src, SourceId(0), &mut i);
        assert!(diags.iter().any(|d| d.code == Some("E0315")), "{diags:?}");
    }
}
