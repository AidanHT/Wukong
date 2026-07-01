//! Item-level parsing: the module header, functions, structs, enums, consts, imports, and
//! extern blocks, plus generics and parameter lists.

use super::Parser;
use mercury_ast::*;
use mercury_diag::Diagnostic;
use mercury_lexer::{Token, TokenKind};
use mercury_span::{Interner, SourceId};

use TokenKind as T;

/// Parse a whole module from already-lexed tokens (the driver's entry point — it tokenizes once
/// and renders lexer diagnostics itself, so this returns only parser diagnostics).
pub fn parse_module_tokens(
    tokens: &[Token],
    src: &str,
    interner: &mut Interner,
) -> (Module, Vec<Diagnostic>) {
    let mut p = Parser::new(tokens, src, interner);
    let m = p.module();
    (m, std::mem::take(&mut p.diags))
}

/// Convenience entry that lexes and parses in one step (tests / standalone use). Returns both
/// lexer and parser diagnostics.
pub fn parse_module(
    src: &str,
    source: SourceId,
    interner: &mut Interner,
) -> (Module, Vec<Diagnostic>) {
    let (tokens, mut diags) = mercury_lexer::tokenize(src, source);
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
            if let Some(item) = self.parse_item() {
                items.push(item);
            }
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

    fn parse_item(&mut self) -> Option<Item> {
        let start = self.span();
        let attrs = self.parse_attrs();
        let is_pub = self.eat(T::Pub);
        let kind = match self.kind() {
            T::Fn => ItemKind::Fn(self.parse_fn(is_pub)),
            T::Struct => ItemKind::Struct(self.parse_struct(is_pub)),
            T::Enum => ItemKind::Enum(self.parse_enum(is_pub)),
            T::Const => ItemKind::Const(self.parse_const(is_pub)),
            T::Import => ItemKind::Import(self.parse_import()),
            T::Extern => ItemKind::Extern(self.parse_extern()),
            _ => {
                let sp = self.span();
                self.error(
                    sp,
                    "E0208",
                    format!(
                        "expected an item (fn, struct, enum, const, import, extern), found {}",
                        self.kind().describe()
                    ),
                );
                self.recover_item();
                return None;
            }
        };
        Some(Item {
            id: self.nid(),
            attrs,
            kind,
            span: start.to(self.prev_span()),
        })
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
            let kind = if self.eat(T::Const) {
                let name = self.ident();
                self.expect(T::Colon);
                let ty = self.parse_type();
                GenericParamKind::Const { name, ty }
            } else {
                GenericParamKind::Type(self.ident())
            };
            gs.push(GenericParam {
                kind,
                span: start.to(self.prev_span()),
            });
            if !self.eat(T::Comma) {
                break;
            }
        }
        self.expect(T::Gt);
        gs
    }

    fn parse_params(&mut self) -> Vec<Param> {
        let mut params = Vec::new();
        self.expect(T::LParen);
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
        self.bump(); // fn
        let name = self.ident();
        let generics = self.parse_generics();
        let params = self.parse_params();
        let ret = if self.eat(T::Arrow) {
            Some(self.parse_type())
        } else {
            None
        };
        // `where` bounds are accepted but not yet enforced.
        if self.at(T::Where) {
            while !matches!(self.kind(), T::LBrace | T::Semi | T::Eq | T::Eof) {
                self.bump();
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
            body,
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

    fn parse_extern(&mut self) -> ExternBlock {
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
                if self.at(T::Fn) {
                    items.push(self.parse_fn(false));
                } else {
                    self.bump();
                }
            }
            self.expect(T::RBrace);
        } else if self.at(T::Fn) {
            items.push(self.parse_fn(false));
        }
        ExternBlock { abi, items }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_ast::print;

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
}
