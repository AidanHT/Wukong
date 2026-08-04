# Wukong — Code Map

A per-feature index of the **Wukong compiler source code**. Every entry names one
independently-improvable feature and gives the exact file and line range that implements it,
so an agent can jump straight to the code with no searching.

**Scope.** First-party Wukong source only — Rust crates under `crates/`, the `.wk` test and
example corpus, and first-party measurement tooling. Documentation, vendored dependencies,
`target/`, and data blobs are deliberately excluded.

**How to use this file.** Sections are numbered work groups `G01`–`G44`. Each group is sized
for a single agent to own end-to-end: hand an agent one group id (or a few related ones) and it
has a complete, self-contained slice of the compiler to interrogate, test and improve.

**Generated from** branch `main` at commit `4f2f6de` — 1446 verified feature entries across 44 groups.
Every line range in this file was mechanically checked to exist within its file at generation time.

---

## Group index

**Frontend**

| Group | Area | Features |
|---|---|---|
| [G01](#g01--lexing-tokens-spans--string-interning) | Lexing, Tokens, Spans & String Interning | 20 |
| [G02](#g02--ast-representation--parsing) | AST Representation & Parsing | 42 |
| [G03](#g03--semantic-analysis-resolution-type-checking-generics-exhaustiveness) | Semantic Analysis: resolution, type checking, generics, exhaustiveness | 47 |
| [G04](#g04--shape--tensor-type-system) | Shape & Tensor Type System | 21 |
| [G05](#g05--diagnostics-catalogue-rendering-json-output) | Diagnostics: catalogue, rendering, JSON output | 20 |

**Middle end**

| Group | Area | Features |
|---|---|---|
| [G06](#g06--mir-data-model-instructions-builder-printer-verifier) | MIR Data Model: instructions, builder, printer, verifier | 35 |
| [G07](#g07--mir-lowering-core-monomorphization-function-lowering-abi-aggregates) | MIR Lowering Core: monomorphization, function lowering, ABI, aggregates | 43 |
| [G08](#g08--mir-lowering-control-flow-expressions-calls-coercions-intrinsics) | MIR Lowering: control flow, expressions, calls, coercions, intrinsics | 41 |
| [G09](#g09--optimizer-ssa-construction-and-scalar-passes) | Optimizer: SSA construction and scalar passes | 43 |

**Kernel recognition**

| Group | Area | Features |
|---|---|---|
| [G10](#g10--recognizer-infrastructure-nest-matchers-affine-analysis-dim-canonicalization) | Recognizer Infrastructure: nest matchers, affine analysis, dim canonicalization | 18 |
| [G11](#g11--recognizers-gemm--matmul--gemv-family-and-epilogue-fusion) | Recognizers: GEMM / matmul / GEMV family and epilogue fusion | 21 |
| [G12](#g12--recognizers-elementwise-vmath-velem-horner-dequant) | Recognizers: elementwise, vmath, velem, horner, dequant | 44 |
| [G13](#g13--recognizers-normalizations-softmax--layernorm--rmsnorm-and-backwards) | Recognizers: normalizations (softmax / LayerNorm / RMSNorm) and backwards | 30 |
| [G14](#g14--recognizers-reductions-arg-reductions-column-reductions-scans) | Recognizers: reductions, arg-reductions, column reductions, scans | 26 |
| [G15](#g15--recognizers-attention-embedding-scatter-losses-other-ml-ops) | Recognizers: attention, embedding, scatter, losses, other ML ops | 22 |
| [G16](#g16--parallel-region-scanning-loop-outlining-parallel-dispatch) | @parallel: region scanning, loop outlining, parallel dispatch | 42 |
| [G17](#g17--ast-level-auto-vectorizer-128256-bit-and-fma-contraction) | AST-level Auto-Vectorizer (128/256-bit) and FMA contraction | 39 |

**Backends**

| Group | Area | Features |
|---|---|---|
| [G18](#g18--cranelift-native-backend-mir-to-clif-calling-convention-object-emit-linking) | Cranelift Native Backend: MIR to CLIF, calling convention, object emit, linking | 45 |
| [G19](#g19--cranelift-raw-avx2-encoder-and-backend-fuzzing) | Cranelift: raw AVX2 encoder and backend fuzzing | 16 |
| [G20](#g20--interpreter) | Interpreter | 32 |
| [G21](#g21--driver-cli-wukongc-module-loader-emit-modes-llvm-textual-backend) | Driver, CLI (wukongc), module loader, emit modes, LLVM textual backend | 18 |

**CPU runtime**

| Group | Area | Features |
|---|---|---|
| [G22](#g22--runtime-gemm--gemv--gevm-core) | Runtime: GEMM / GEMV / GEVM core | 27 |
| [G23](#g23--runtime-elementwise-vmath-velem-scans) | Runtime: elementwise, vmath, velem, scans | 43 |
| [G24](#g24--runtime-normalizations-and-softmax) | Runtime: normalizations and softmax | 17 |
| [G25](#g25--runtime-reductions-and-arg-reductions) | Runtime: reductions and arg-reductions | 22 |
| [G26](#g26--runtime-quantization-and-low-precision-bf16--f16--int8--int4) | Runtime: quantization and low precision (bf16 / f16 / int8 / int4) | 44 |
| [G27](#g27--runtime-attention-rope-embedding-convpool-transpose) | Runtime: attention, RoPE, embedding, conv/pool, transpose | 24 |
| [G28](#g28--runtime-backward--training-kernels-and-losses) | Runtime: backward / training kernels and losses | 25 |
| [G29](#g29--runtime-dispatch-table-abi-thread-pool-environment-knobs) | Runtime: dispatch table, ABI, thread pool, environment knobs | 47 |

**Autodiff**

| Group | Area | Features |
|---|---|---|
| [G30](#g30--autodiff-reverse-mode-mir-to-mir-tape-vjp-rules-fused-optimizers) | Autodiff: reverse-mode MIR to MIR, tape, VJP rules, fused optimizers | 35 |

**GPU**

| Group | Area | Features |
|---|---|---|
| [G31](#g31--gpu-device-management-memory-host-dispatch-api) | GPU: device management, memory, host dispatch API | 13 |
| [G32](#g32--gpu-gemm-and-tensor-core-ptx-kernels) | GPU: GEMM and tensor-core PTX kernels | 48 |
| [G33](#g33--gpu-attention-and-flash-attention-ptx) | GPU: attention and flash-attention PTX | 28 |
| [G34](#g34--gpu-quantized-ptx-int8--int4--fp8) | GPU: quantized PTX (int8 / int4 / fp8) | 39 |
| [G35](#g35--gpu-conv-winograd-norm-optimizer-ptx) | GPU: conv, winograd, norm, optimizer PTX | 32 |
| [G36](#g36--gpu-mir-to-ptx-lowering-gpu-native-backend) | GPU: MIR to PTX lowering (gpu-native backend) | 32 |
| [G37](#g37--gpu-megakernel-fusion-cuda-graphs-device-pool-cubin-cache-autotuner) | GPU: megakernel, fusion, CUDA graphs, device pool, cubin cache, autotuner | 21 |
| [G38](#g38--gpu-serving-stack-paged-kv-continuous-batching-tp-sim) | GPU: serving stack (paged KV, continuous batching, TP sim) | 14 |
| [G39](#g39--gpu-training-resident-path-and-backward-ptx) | GPU: training-resident path and backward PTX | 13 |
| [G40](#g40--gpu-correctness-gates-and-peer-baselines) | GPU: correctness gates and peer baselines | 75 |

**Harnesses**

| Group | Area | Features |
|---|---|---|
| [G41](#g41--benchmarks-xbench-cross-language-harness-and-model-bench) | Benchmarks: xbench cross-language harness and model bench | 59 |
| [G42](#g42--benchmarks-compile-time-measurement-profiling--dev-perf-probes) | Benchmarks: compile-time measurement, profiling & dev perf probes | 20 |
| [G43](#g43--test-corpus-run--fail--imports-differential-and-opt-invariance-gates) | Test corpus: run / fail / imports differential and opt-invariance gates | 92 |
| [G44](#g44--example-wk-programs-and-the-gpt-2-end-to-end-pipeline) | Example .wk programs and the GPT-2 end-to-end pipeline | 11 |

---

# Frontend

## G01 — Lexing, Tokens, Spans & String Interning

*Files:* `crates/wukong_lexer/src/lib.rs`, `crates/wukong_lexer/src/token.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_span/src/fxhash.rs`, `crates/wukong_span/src/intern.rs`, `crates/wukong_span/src/lib.rs`, `crates/wukong_span/src/source_map.rs`

**`crates/wukong_lexer/src/lib.rs`**

- **Non-aborting tokenize driver** — `crates/wukong_lexer/src/lib.rs:13-27` — Scans until Eof, always appending a terminating Eof token, and returns accumulated diagnostics instead of bailing, so parsing always gets a stream.
- **Token-stream dump for --emit=tokens** — `crates/wukong_lexer/src/lib.rs:29-38` — Renders each token as `Kind "text"` by byte-slicing the source with the token span; snapshot-tested and used as the lexer's observable output.
- **Unknown-byte error recovery with UTF-8 width decoding** — `crates/wukong_lexer/src/lib.rs:105-117` — Consumes a whole multi-byte character, escapes it into an E0101 message, emits an Error token and continues, so one stray glyph costs one diagnostic; also `crates/wukong_lexer/src/lib.rs:424-437`.
- **Trivia skipping with depth-nested block comments** — `crates/wukong_lexer/src/lib.rs:119-160` — Loops over spaces/tabs/CR/LF, `//` to end-of-line, and `/*…*/` with a depth counter supporting nesting; unterminated block comment reports E0103 and stops.
- **ASCII identifier scan and keyword table** — `crates/wukong_lexer/src/lib.rs:162-172` — Consumes `[A-Za-z0-9_]` then maps the text through a 28-entry keyword table; non-ASCII identifiers are impossible, also `crates/wukong_lexer/src/token.rs:109-178`.
- **Numeric literal scanner with radix, separators, exponent and suffix** — `crates/wukong_lexer/src/lib.rs:174-250` — Handles `0x/0b/0o`, `_` separators, fraction only when a digit follows the dot (so `1..5` splits), exponent, and type suffixes.
- **Float-vs-int classification quirks** — `crates/wukong_lexer/src/lib.rs:231-249` — Any alphabetic suffix beginning with `f` forces Float (so `1foo` lexes Float), while the radix branch returns Int early, never validating digits or suffixes.
- **String literal delimiting with newline/EOF cutoff** — `crates/wukong_lexer/src/lib.rs:252-270` — Consumes to the closing quote, treats bare newline or EOF as unterminated (E0102) and stops there, keeping the error local to one line.
- **Backslash-escape consumer** — `crates/wukong_lexer/src/lib.rs:272-312` — Delimits `\xHH`, `\u{…}` and one-char escapes UTF-8-aware; `\u{` scanning bails at `"`/`'`/newline so a malformed escape cannot swallow the terminator; value validation deferred to mir_build.
- **Loop-label vs char-literal disambiguation** — `crates/wukong_lexer/src/lib.rs:314-339` — Lookahead measures the identifier after `'` and only emits Label when no closing quote follows, reproducing Rust's rule without backtracking the cursor.
- **Char literal scan and E0104 recovery** — `crates/wukong_lexer/src/lib.rs:341-359` — Consumes one escape or one whole UTF-8 char then requires a closing quote; empty `''` is silently accepted, and `'ab'` errors while leaving the cursor mid-literal.
- **Maximal-munch operator/punctuation table** — `crates/wukong_lexer/src/lib.rs:361-421` — One match arm per glyph, ordered so 3-byte (`..=`, `<<=`, `>>=`) beats 2-byte beats 1-byte; advances `pos` by the matched length directly.
- **Lexer and span unit-test suites** — `crates/wukong_lexer/src/lib.rs:439-540` — Cover keyword/ident splitting, number forms, ranges, operators, escapes, nested comments, error recovery and the dump snapshot; span/source-map tests check merge, multibyte columns, interning, also `crates/wukong_span/src/source_map.rs:117-156`.
**`crates/wukong_lexer/src/token.rs`**

- **TokenKind/Token data model** — `crates/wukong_lexer/src/token.rs:5-107` — A `Copy` kind-plus-span pair carrying no literal payload; values are recovered later from the source map, keeping tokens 16 bytes, also `crates/wukong_lexer/src/token.rs:373-384`.
- **Parallel describe/glyph/name spelling tables** — `crates/wukong_lexer/src/token.rs:180-370` — Three hand-written per-variant tables feeding diagnostics, canonical source text, and the token dump; adding a TokenKind requires editing all three or output desynchronises.
**`crates/wukong_mir_build/src/lib.rs`**

- **Literal decoding** — `crates/wukong_mir_build/src/lib.rs:24988-25110` — Radix-prefixed and suffixed integer parsing, separator-stripping float parsing, and char/string escape decoding including saturating `\u{…}` so an over-long escape cannot panic the compiler.
**`crates/wukong_span/src/fxhash.rs`**

- **Dependency-free FxHash hasher and map aliases** — `crates/wukong_span/src/fxhash.rs:1-90` — Rotate-xor-multiply word hasher replacing SipHash across the front-end for speed; seedless and thus deterministic, with iteration-order safety argued by the `--emit=mir` gate.
**`crates/wukong_span/src/intern.rs`**

- **String interner** — `crates/wukong_span/src/intern.rs:19-56` — Maps each distinct string to a `Copy` u32 Symbol via an FxHashMap plus index vector; stores every string twice (key and value clone), an obvious memory lever.
**`crates/wukong_span/src/lib.rs`**

- **Span type and span arithmetic** — `crates/wukong_span/src/lib.rs:27-104` — 12-byte `Copy` half-open byte range with `dummy` sentinel (SourceId u32::MAX), `to` merge, and `shrink_to_lo/hi`; cross-source merge is only a debug_assert.
**`crates/wukong_span/src/source_map.rs`**

- **SourceMap: file ownership, line index, location and line text** — `crates/wukong_span/src/source_map.rs:12-115` — Precomputes line starts per file, binary-searches them for 1-based line/column with char-counted columns, and slices span text and trimmed line text for diagnostics.

## G02 — AST Representation & Parsing

*Files:* `crates/wukong_ast/src/expr.rs`, `crates/wukong_ast/src/item.rs`, `crates/wukong_ast/src/lib.rs`, `crates/wukong_ast/src/print.rs`, `crates/wukong_ast/src/stmt.rs`, `crates/wukong_ast/src/ty.rs`, `crates/wukong_parser/src/items.rs`, `crates/wukong_parser/src/lib.rs`

**`crates/wukong_ast/src/expr.rs`**

- **Expression node and raw-source-text literals** — `crates/wukong_ast/src/expr.rs:6-59` — Expr{id,kind,span} plus Int/Float/Str/Char stored as unparsed interned source text (suffixes included), path refs, unary/binary, call with generic args, multi-index, field/tuple-field, cast; also `crates/wukong_ast/src/expr.rs:90-92`.
- **Aggregate and repeat literal constructors** — `crates/wukong_ast/src/expr.rs:60-71` — StructLit with functional-update `..rest`, ArrayLit, `[value; count]` ArrayRepeat and TupleLit, the surface forms feeding aggregate lowering and slot sizing.
- **Value-producing control flow with labels, guards and break-values** — `crates/wukong_ast/src/expr.rs:72-108` — Block/If/Match/labeled `loop` are expressions whose type joins arm and break values; MatchArm carries an optional `if` guard; also `crates/wukong_ast/src/stmt.rs:39-42`.
- **Operator glyph tables for UnOp, BinOp and AssignOp** — `crates/wukong_ast/src/expr.rs:110-176` — Single source of operator spelling (incl. `&mut ` with trailing space, which printers must trim) shared by printer and diagnostics; also `crates/wukong_ast/src/expr.rs:225-257`.
- **BinOp::fold_const_len shared const-array-length oracle** — `crates/wukong_ast/src/expr.rs:178-223` — One u64 folder both sema's eval_usize and mir_build's const_usize_expr call, preventing slot-size/bounds desync; div, rem, over-wide shift and comparisons yield 0 rather than panicking.
**`crates/wukong_ast/src/item.rs`**

- **Item, FnDecl and by-reference `mut` parameters** — `crates/wukong_ast/src/item.rs:6-46` — Six top-level item kinds; FnDecl allows a bodiless declaration, and Param.mutable opts an aggregate parameter into caller-visible in-place mutation.
- **Generic parameter kinds: type/dimension vs explicit const** — `crates/wukong_ast/src/item.rs:48-61` — A bare `<T>`/`<M>` doubles as a type or a symbolic tensor dimension, while `const N: usize` is spelled explicitly, driving monomorphization and hidden dim params.
- **Struct, enum and const declarations with explicit discriminants** — `crates/wukong_ast/src/item.rs:63-108` — Generic struct/enum declarations, per-variant Unit/Tuple/Struct payload data, an optional discriminant expression, and constant items with a required type annotation.
- **Imports and extern ABI blocks** — `crates/wukong_ast/src/item.rs:110-122` — Import encodes dotted path, optional alias, and optional selective `{x, y}` list; ExternBlock groups bodiless FnDecls under an interned ABI symbol.
- **Attribute grammar and the misleading `Attr::args` helper** — `crates/wukong_ast/src/item.rs:124-157` — Attr/AttrArg/AttrVal encode `@simd`, `@tile(64)`, `@parallel(grain = 1)`, `@export("n")`; DIVERGENCE: `args()`'s doc claims key lookup but merely returns the whole slice.
**`crates/wukong_ast/src/lib.rs`**

- **AST root: NodeId, Ident, Path, Module** — `crates/wukong_ast/src/lib.rs:1-68` — Data-only crate re-exporting four node modules; NodeId(u32) with DUMMY sentinel keys sema side tables so the tree is never rewritten.
**`crates/wukong_ast/src/print.rs`**

- **AST printer harness and inline renderers** — `crates/wukong_ast/src/print.rs:1-147` — Depth-indented emitter with print_module/print_expr/print_type entry points; type_str renders types source-like, expr_inline handles only literals/paths/binary/unary and degrades to `<expr>`.
- **Module and item printing (lossy by design)** — `crates/wukong_ast/src/print.rs:149-280` — Renders attributes, generics, params and fn bodies, but drops enum variant payloads and discriminants, import aliases and selective lists, extern fn signatures, and Tiled layout extents.
- **Statement, pattern and expression tree printing** — `crates/wukong_ast/src/print.rs:282-628` — Full recursive `--emit=ast` rendering of every StmtKind, ForIter, PatKind and ExprKind, with a single unit test covering only an empty named module.
**`crates/wukong_ast/src/stmt.rs`**

- **Block, Stmt and per-statement attribute carriage** — `crates/wukong_ast/src/stmt.rs:6-22` — Braced block with statement list plus optional tail-expression value; every statement carries its own Attr vector, the hook `@parallel`/`@simd` loop annotations ride on.
- **Statement kinds and for-loop iteration forms** — `crates/wukong_ast/src/stmt.rs:24-70` — let/assign-op/expr/return/labeled break-with-value/continue/defer/while/for, and ForIter covering `start..end`, inclusive, open-ended and `step s` ranges or an arbitrary iterable.
- **Pattern family for match and let bindings** — `crates/wukong_ast/src/stmt.rs:72-106` — Wildcard, ident, tuple, unit, integer literal with parser-folded negation, char literal, bool, or-patterns, enum-discriminant paths, and half-open/inclusive range patterns.
- **Data-carrying variant patterns with payload destructuring** — `crates/wukong_ast/src/stmt.rs:107-129` — `PatKind::Variant` plus VariantPat::Tuple/Struct and FieldPat encode `Op::Add(x,y)` and `Circle{r}` shorthand, matching discriminant and every sub-pattern together.
**`crates/wukong_ast/src/ty.rs`**

- **Surface type syntax (TypeExpr / TypeKind)** — `crates/wukong_ast/src/ty.rs:1-40` — Named paths, integer const-generic arguments, unit, `*T`/`&T` with mutability, `[]T` slices, `[T; N]` arrays with an expression length, tuples, and SIMD vector types.
**`crates/wukong_parser/src/items.rs`**

- **Module entry points, NodeId watermarking, item dispatch and recovery** — `crates/wukong_parser/src/items.rs:12-135` — Three parse entries thread a `first_node_id` watermark so multi-file imports get globally unique NodeIds; `recover_item` resyncs to the next item keyword after E0208.
- **Function declarations: params, `mut`, `where` skip, `= expr` body** — `crates/wukong_parser/src/items.rs:164-233` — Parameter list with per-param attributes and `mut`, optional `-> T`, a `where` clause that is skipped token-by-token and never enforced, and an expression-bodied form desugared to a tail block.
- **Struct and enum declarations** — `crates/wukong_parser/src/items.rs:235-336` — Fields with `pub` and discarded field attributes; enum variants support unit/tuple/struct payloads and explicit discriminants, but the `enum E: u8` repr type is parsed before generics and thrown away.
- **Const, import and extern-block items** — `crates/wukong_parser/src/items.rs:338-401` — `const NAME: T = expr;`, dotted imports with `as` alias or `.{a, b}` selective lists, and `extern "ABI"` blocks defaulting to `"C"` that keep only `fn` signatures.
**`crates/wukong_parser/src/lib.rs`**

- **Parser state, cursor and node/span helpers** — `crates/wukong_parser/src/lib.rs:44-253` — Token cursor (`tok`/`nth`/`bump`/`eat`/`expect`), NodeId allocation, span interning, and `ident_like` which accepts keywords as attribute names.
- **Struct-literal ambiguity gate (`no_struct_lit`)** — `crates/wukong_parser/src/lib.rs:83-98` — Suppresses `Name { … }` struct-literal parsing in `if`/`while`/`for`/`match`/guard heads and re-enables it inside any delimited context, also at `crates/wukong_parser/src/lib.rs:784-843` and `crates/wukong_parser/src/lib.rs:1275-1277`.
- **Type grammar (pointer, ref, slice, array, tuple, unit)** — `crates/wukong_parser/src/lib.rs:257-338` — Recursive type parser where `[]T` is a slice and `[T; len]` an array with an expression length; a 1-tuple collapses unless a trailing comma is present.
- **Pratt precedence climbing for binary operators** — `crates/wukong_parser/src/lib.rs:487-524` — Left-associative binding-power loop over 18 operators with 9 precedence levels; tables at `crates/wukong_parser/src/lib.rs:1655-1693`.
- **Cast level between binary and prefix** — `crates/wukong_parser/src/lib.rs:530-547` — Places `as` looser than every unary prefix but tighter than every binop so `*p as T` is `(*p) as T`, fixing a historical ptrtoint-then-load miscompile.
- **Prefix unary operators including `~` alias and `&`/`&mut`** — `crates/wukong_parser/src/lib.rs:549-596` — Neg, Not (`!` and `~` both map to `UnOp::Not`), Deref, and Ref/RefMut, with the depth choke point wrapping every operand.
- **Postfix chain: calls, multi-index, fields, turbofish, `::` paths** — `crates/wukong_parser/src/lib.rs:598-751` — Loops call args, comma-separated multi-dimensional `a[i, k]` indexing, `.field`, `.0` tuple fields, `::<T,…>(…)` turbofish (E0205 if no `(` follows), and `::name`.
- **Primary expressions: literals, grouping, tuple/array literals, `sizeof`/`alignof`** — `crates/wukong_parser/src/lib.rs:753-843` — Int/float/str/char/bool literals, parenthesized-vs-tuple disambiguation preserving the inner node id, `[a,b]` and `[v; n]` repeat forms; `sizeof[T]`/`alignof[T]` at `crates/wukong_parser/src/lib.rs:885-900`.
- **`loop` as a value expression and labeled loops** — `crates/wukong_parser/src/lib.rs:844-884` — The single loop-parsing site; `'l: loop` stays an expression while a label followed by anything else reports E0200 and recovers with an empty tuple.
- **Struct literals and `Enum::Variant { … }` payload literals** — `crates/wukong_parser/src/lib.rs:937-960` — Field-init list parsing gated by a no-consumption lookahead for `Ident (:: Ident)+ {`; `rest` is always `None`, so functional-update `..base` syntax is unimplemented despite the AST field, lookahead at `crates/wukong_parser/src/lib.rs:1365-1394`.
- **`if` / `else if` chains as expressions** — `crates/wukong_parser/src/lib.rs:962-1000` — Condition parsed in no-struct-lit mode, else branch recurses directly into `parse_if` and therefore carries its own depth charge separate from the prefix choke point.
- **`match` expression with arm guards** — `crates/wukong_parser/src/lib.rs:1002-1051` — Pattern, optional `if` guard parsed as a condition head, `=>` body, optional comma, plus a forward-progress bump so a depth-tripped arm cannot spin the loop forever.
- **Block and statement dispatcher with forward-progress guard** — `crates/wukong_parser/src/lib.rs:1055-1226` — Routes let/const/return/break/continue/defer/while/for/labeled-loop/expression statements, decides tail-expression vs statement at `}`, and bumps if a sub-parse consumed nothing.
- **Jump statements with labels and break values** — `crates/wukong_parser/src/lib.rs:1112-1155` — `return`/`break`/`continue`/`defer`, where `break 'outer v` carries both an optional target label and an optional value; label text is interned without the leading quote at `crates/wukong_parser/src/lib.rs:1288-1294`.
- **Binding statements: `let`, local `const`, and compound assignment** — `crates/wukong_parser/src/lib.rs:1237-1258` — `let [mut] pat [: T] [= init]` with optional semicolon; a local `const X: T = v;` desugars to an immutable let at `crates/wukong_parser/src/lib.rs:1090-1111`; 11 assignment operators at `crates/wukong_parser/src/lib.rs:1695-1711`.
- **Loop statements and range iteration with `step`** — `crates/wukong_parser/src/lib.rs:1260-1320` — `while`/`for` (optionally labeled via the `nth(2) != Loop` guard at `crates/wukong_parser/src/lib.rs:1167-1184`) and `a..b`/`a..=b` ranges with an omissible end and an optional `step` clause.
- **Pattern grammar: or-patterns, ranges, variants, literals** — `crates/wukong_parser/src/lib.rs:1324-1546` — Top-level `A | B`, range tails, `Enum::Variant(..)` tuple and `{ f, g: p }` struct payloads with field shorthand, wildcard, int/char/bool and sign-folded negative literals; the range doc says integer-only but the code also accepts a `char` lower bound.
- **Attribute grammar (`@name(args)`) including `@parallel`** — `crates/wukong_parser/src/lib.rs:1550-1636` — Generic attribute parser with int/string/word/key=value arguments; `@parallel`, `@simd` and `@extern` get no special-case handling here and are interpreted downstream.
- **Glued tuple-index recovery (`t.0.0`)** — `crates/wukong_parser/src/lib.rs:1644-1653` — Splits a lexer-glued `N.M` float token back into two tuple-field accesses, declining any float with an exponent or suffix; use site `crates/wukong_parser/src/lib.rs:642-670`.

## G03 — Semantic Analysis: resolution, type checking, generics, exhaustiveness

*Files:* `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_parser/src/items.rs`, `crates/wukong_sema/src/lib.rs`, `crates/wukong_sema/src/shape.rs`

**`crates/wukong_mir_build/src/lib.rs`**

- **Value-type generic detection** — `crates/wukong_mir_build/src/lib.rs:97-156` — `ty_has_generic`, `collect_named_generics` and `fn_type_generics` isolate generics used as values (never dims) and re-order them into declaration order for positional mangling.
- **Compile-time array-length evaluation** — `crates/wukong_mir_build/src/lib.rs:18370-18417` — Depth-bounded folding of literals, const paths, binary arithmetic and enum discriminants, deliberately mirroring sema's `eval_usize` so alloca sizes match bounds checks.
**`crates/wukong_parser/src/items.rs`**

- **Generic parameter lists (type and const params)** — `crates/wukong_parser/src/items.rs:137-162` — Parses `<T, const N: usize>` shared by fn, struct and enum declarations, supplying the symbolic dims that Tensor shape annotations reference.
**`crates/wukong_sema/src/lib.rs`**

- **Builtin nominal name tables (alloc/read/write/now_ns)** — `crates/wukong_sema/src/lib.rs:27-68` — Maps `alloc_<T>`, `read_<T>`/`write_<T>`, and `now_ns` spellings to element scalars so sema and mir_build cannot drift on the builtin surface.
- **Def/DefKind/EnumVariant resolved-definition model** — `crates/wukong_sema/src/lib.rs:70-152` — Name-indexed `DefMap` of functions, consts, structs and enums (variants carry resolved i64 discriminant plus unit/tuple/struct payload) consumed by every later check and by mir_build.
- **Two-phase checking with the `checking_bodies` gate** — `crates/wukong_sema/src/lib.rs:294-325` — `check` runs collect, recursive-type and recursive-const passes, then flips a flag so undeclared-dim diagnostics fire once per site with every generic and const already known; the flag is documented at `crates/wukong_sema/src/lib.rs:355-362`.
- **Collection pass registering all top-level items** — `crates/wukong_sema/src/lib.rs:391-477` — Walks module items (including `extern` blocks) lowering signatures, struct fields and enum payloads under the item's generic scope before any body is checked, giving order-independent resolution.
- **Enum discriminant resolution with 32-bit range gate** — `crates/wukong_sema/src/lib.rs:417-465` — Auto-increments (wrapping) or takes explicit `= <int>` discriminants and rejects any outside i32 with E0401, since enum values lower to i32 and would silently truncate.
- **Recursive-by-value struct/enum cycle check (E0402)** — `crates/wukong_sema/src/lib.rs:484-520` — DFS over by-value containment (pointer fields break cycles) rejecting infinite-size types that would stack-overflow mir_build's layout pass; helpers at `crates/wukong_sema/src/lib.rs:158-169` and `crates/wukong_sema/src/lib.rs:526-544`.
- **Recursive const-initializer cycle check (E0403)** — `crates/wukong_sema/src/lib.rs:553-594` — DFS over the const-reference graph rejecting self-dependent consts before mir_build's use-site inliner recurses forever; the reference collector is `crates/wukong_sema/src/lib.rs:175-292`.
- **Declaration-ordered generic parameter list** — `crates/wukong_sema/src/lib.rs:596-621` — `collect_fn` builds `FnSig.generics` from the AST rather than the `generics` HashSet, so a turbofish binds arguments by position deterministically across runs.
- **Duplicate-definition diagnostic with both spans (E0300)** — `crates/wukong_sema/src/lib.rs:623-649` — `register` rejects a redefined top-level name and labels the first definition too, which matters because the import loader splices multiple files into one flat namespace.
- **AST type-syntax lowering to semantic Ty** — `crates/wukong_sema/src/lib.rs:653-723` — Lowers paths/scalars, pointers, refs, slices, arrays, tuples, SIMD vectors and tensors (with layout), raising E0302 when a vector or tensor element is not a scalar.
- **Compile-time usize evaluator for array lengths and dims** — `crates/wukong_sema/src/lib.rs:826-863` — Folds int literals, const references, binary arithmetic (`BinOp::fold_const_len`) and C-style enum discriminants, mirroring mir_build's `const_usize_expr` so slot sizes and bounds checks agree; enum access helper at `crates/wukong_sema/src/lib.rs:804-818`.
- **Top-level const initializer checking (E0401)** — `crates/wukong_sema/src/lib.rs:885-917` — Type-checks a const's initializer against its annotation at module scope, re-stamping only all-literal expressions, and records the expression for mir_build's inlining.
- **Function body entry: param binding, duplicate params, definite return** — `crates/wukong_sema/src/lib.rs:919-982` — Resets all four per-scope stacks in lockstep, rejects duplicate parameter names (E0300), records non-`mut` params, and requires a value-returning function not to fall off its end (E0401).
- **Aggregate / noncomputable / kind-clash classifiers** — `crates/wukong_sema/src/lib.rs:1063-1127` — Distinguishes C-style enums (integer discriminants) from tagged unions and aggregates, powering the `==` rejection, unary/binary operand rejection, unit-condition rejection and scalar-vs-pointer clash checks.
- **`as` cast validity rules** — `crates/wukong_sema/src/lib.rs:1134-1211` — Permits scalar↔scalar, enum→int, int→pointer and same-pointee pointer retypes only, rejecting pointer→int, aggregate reinterprets and int→enum (which could forge a discriminant reaching an unreachable match arm).
- **Struct-literal completeness and per-field checking** — `crates/wukong_sema/src/lib.rs:1218-1320` — Rejects unknown, duplicate and missing fields, adapts and range-checks literal field values, enforces array-field initializer length, and demands explicit casts for mismatched scalar fields.
- **Immutable local and parameter enforcement (E0304)** — `crates/wukong_sema/src/lib.rs:1334-1395` — Tracks non-`mut` lets and params per scope and, for aggregate params passed by reference, rejects mutation through field/index projections while stopping the walk at a deref; enforcement site `crates/wukong_sema/src/lib.rs:1610-1653`.
- **Lexical scope stack and value name resolution** — `crates/wukong_sema/src/lib.rs:1403-1437` — Resolves a single-segment name innermost-scope-first, then generics (as compile-time `usize`), then the def map, then treats scalar/`Tensor`/vector type names as lenient namespaces; scope push/pop at `crates/wukong_sema/src/lib.rs:1322-1332`.
- **`let` statement typing and annotation compatibility** — `crates/wukong_sema/src/lib.rs:1454-1513` — Joins annotation and initializer, re-stamps only genuinely all-literal initializers (avoiding dropping a narrow operand's widening), binds the pattern and updates immutability marks.
- **Assignment statement rule battery** — `crates/wukong_sema/src/lib.rs:1514-1677` — Literal adaptation and range check, non-literal scalar-mismatch rejection, compound-assign bans on aggregates/pointers/tensors and int-place-float-value, kind-clash rejection, and whole-tensor shape unification.
- **`return` statement checks** — `crates/wukong_sema/src/lib.rs:1689-1754` — Rejects returning a value from a `-> ()` function, returning `()` where a type is declared, and a bare `return;` where a value is required, plus shape and literal-range checks.
- **Loop context: labels, break/continue scoping, break-value join** — `crates/wukong_sema/src/lib.rs:1758-1820` — Resolves a break/continue to the innermost or labeled loop frame (E0303 otherwise), rejects `break v` in statement-position loops, and joins break value types with shape unification; frame type `crates/wukong_sema/src/lib.rs:366-377`, loop typing `crates/wukong_sema/src/lib.rs:2983-2996`.
- **For-iterand element typing and pattern binding** — `crates/wukong_sema/src/lib.rs:1855-1918` — Ranges take the start's type (defaulting to `usize`), arrays and slices bind their element type, and `bind_pattern` binds ident/tuple/or/variant patterns while literal patterns bind nothing.
- **Enum-variant destructuring patterns** — `crates/wukong_sema/src/lib.rs:1925-2033` — Resolves `Enum::Variant(..)`/`{..}` patterns, reporting unknown variants (E0301), payload-kind and arity mismatches (E0401), and binds payload sub-patterns to declared types with `Unknown` recovery.
- **Tuple-payload variant construction as a call** — `crates/wukong_sema/src/lib.rs:2039-2116` — Intercepts `Enum::Variant(args)` before ordinary call typing, checking arity and per-field types and rejecting unit/struct variants spelled as calls; per-field checker at `crates/wukong_sema/src/lib.rs:2121-2141`.
- **Unsuffixed-literal adaptation and re-stamping** — `crates/wukong_sema/src/lib.rs:2185-2291` — Decides whether a literal (through unary minus, all-constant arithmetic, arrays, repeats and tuples) adopts an annotation, then rewrites the side-table types so MIR sees one consistent width.
- **Narrow-type literal range check (E0401)** — `crates/wukong_sema/src/lib.rs:2299-2337` — Recurses into aggregate literals and flags a folded constant outside an i8..u32 target range instead of letting it wrap identically on both backends; bounds table `crates/wukong_sema/src/lib.rs:3359-3370`, folder `crates/wukong_sema/src/lib.rs:3439-3488`.
- **Unary operator checks** — `crates/wukong_sema/src/lib.rs:2397-2478` — Rejects dereferencing a definite non-pointer, unary `-`/`!` on aggregates/unit/functions and bitwise complement on floats, and re-types `-9223372036854775808` as i64::MIN.
- **Binary operator check battery** — `crates/wukong_sema/src/lib.rs:2480-2700` — Rejects aggregate/unit operands, pointer and whole-tensor arithmetic, bool arithmetic, float bitwise/shift, chained `==`/ordered comparisons via bool operands, and mixed signed/unsigned integer comparisons.
- **Field access, enum-variant paths and tuple index bounds** — `crates/wukong_sema/src/lib.rs:2717-2804` — Resolves `E::V` paths (rejecting a bare payload-carrying variant), auto-derefs pointer/reference struct bases for field lookup, and range-checks tuple field indices with E0501.
- **Struct and enum-struct-variant literal typing** — `crates/wukong_sema/src/lib.rs:2811-2876` — Types every field value, routes `Enum::Variant { .. }` to the shared struct-literal checker, and gives a declared struct its nominal type while unknown names stay lenient.
- **`if`/`match` arm merging with shape agreement** — `crates/wukong_sema/src/lib.rs:2907-2970` — Types condition and arms in per-arm scopes, unifies each arm's tensor/vector shape against the running result before joining, and triggers the exhaustiveness check.
- **Match exhaustiveness prover (E0405)** — `crates/wukong_sema/src/lib.rs:3005-3073` — Conservatively rejects a match only when incompleteness is certain: enums need every variant covered by name or discriminant span, bools need both cases, other scalars need a catch-all; collectors at `crates/wukong_sema/src/lib.rs:3255-3312`.
- **Result-type join with generic-literal adaptation** — `crates/wukong_sema/src/lib.rs:3109-3175` — `join`/`join_scalar` promote to the wider (float-preferring) scalar and `compatible` allows the one array→slice unsizing, while the binop tail at `crates/wukong_sema/src/lib.rs:2681-2699` lets a literal adapt to a generic-typed peer on either side.
- **Divergence analysis for the definite-return rule** — `crates/wukong_sema/src/lib.rs:3196-3245` — Conservative `block_diverges`/`stmt_diverges`/`expr_diverges` (a `loop` diverges only when break-free, a `match` only when catch-all and all arms diverge), with the break scanner at `crates/wukong_sema/src/lib.rs:3319-3357`.
- **Literal text well-formedness and default scalar widening** — `crates/wukong_sema/src/lib.rs:3497-3542` — Rejects malformed int/float literals that used to lower silently to 0, while `int_lit_scalar` widens an unsuffixed literal from i32 to i64/u64 by magnitude; typing sites `crates/wukong_sema/src/lib.rs:2347-2370`, defaults `crates/wukong_sema/src/lib.rs:3372-3408`, parsers `crates/wukong_sema/src/lib.rs:3544-3601`.
**`crates/wukong_sema/src/shape.rs`**

- **Math-intrinsic return typing table** — `crates/wukong_sema/src/shape.rs:18-49` — Types ~50 unlowered math builtins as float-of-first-arg while `abs`/`round`/`floor`/`ceil`/`trunc` preserve integer types; doc-comment claims only six intrinsics.
- **Resolved user-fn call path with literal argument adaptation** — `crates/wukong_sema/src/shape.rs:67-107` — Looks up a single-segment callee, retypes unsuffixed numeric literal arguments to concrete scalar parameters, range-checks them, then hands off to `check_fn_call`.
- **Math-intrinsic argument kind gate (E0401)** — `crates/wukong_sema/src/shape.rs:112-139` — Rejects pointer/ref/unit/fn/aggregate arguments to math intrinsics, closing an interp-vs-native divergence where invalid MIR ran silently on raw aggregate values.
- **Heap builtin typing: `alloc_<T>` / `free`** — `crates/wukong_sema/src/shape.rs:145-205` — Types `alloc_f32(n)` as `[]f32` and `free(s)` as unit, enforcing arity (E0503) and integer-count / slice-argument types (E0401) before structural lowering.
- **File-I/O builtin typing: `read_<T>` / `write_<T>`** — `crates/wukong_sema/src/shape.rs:216-265` — Types the blob I/O family as `i64`, requiring a `*u8` path and a `[]T` buffer whose element scalar exactly matches the name's dtype.
- **Builtin arity/printability gates: `now_ns`, `print`/`println`, `assert`** — `crates/wukong_sema/src/shape.rs:266-344` — Pins zero-arg `now_ns`, rejects multi-arg or `()`-valued print, and pins `assert` arity, each closing a documented backend divergence.
- **Modeled slice `.len()` method typing** — `crates/wukong_sema/src/shape.rs:351-366` — Types `s.len()` on a slice as `i64` so `for i in 0..s.len()` avoids a mixed-width `Cmp` the MIR verifier rejects; other methods stay lenient.
- **Compile-time index constant folding with const and enum-discriminant resolution** — `crates/wukong_sema/src/shape.rs:755-783` — Resolves a literal, a top-level `const` chain (depth-capped at 32), or an enum variant discriminant so static bounds checks see through named constants.

## G04 — Shape & Tensor Type System

*Files:* `crates/wukong_ast/src/ty.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_parser/src/lib.rs`, `crates/wukong_sema/src/lib.rs`, `crates/wukong_sema/src/shape.rs`, `crates/wukong_types/src/lib.rs`

**`crates/wukong_ast/src/ty.rs`**

- **Tensor type syntax: dims and physical layout** — `crates/wukong_ast/src/ty.rs:41-73` — `Tensor[f32, M, N]` with per-dim Int/Named/Dynamic(`?`) extents and an optional Contiguous/ColMajor/Strided/Tiled layout, the syntactic root of the symbolic shape system.
**`crates/wukong_mir_build/src/lib.rs`**

- **Hidden symbolic-dim parameter ABI** — `crates/wukong_mir_build/src/lib.rs:158-197` — `collect_symbolic_dims`/`symbolic_dim_params` derive the deterministic dedup'd `Dim::Var` list threaded as hidden i64 params; a HashSet here would be a nondeterministic-ABI miscompile.
- **Hidden symbolic-dim runtime parameters for generic tensor functions** — `crates/wukong_mir_build/src/lib.rs:2577-2589` — Prepends one i64 param per symbolic dim and binds it under its dim symbol so strides, `dim_value`, and plain uses resolve through ordinary lookup; ABI mirrors the call site.
- **Tensor row-major stride computation and multi-index flattening** — `crates/wukong_mir_build/src/lib.rs:15011-15190` — Flattens `t[i,j,…]` to one GEP using static strides when interior dims are constants, else builds strides at runtime from hidden dim params, declining `Dynamic` dims and non-contiguous layouts.
**`crates/wukong_parser/src/lib.rs`**

- **Shape-annotation type forms: `vec[T,N]`, `Tensor[…]`, `f32x8`** — `crates/wukong_parser/src/lib.rs:340-382` — Text-peek recognition of the three tensor/SIMD spellings before falling back to a dotted path; the `f32x8` splitter and scalar whitelist live at `crates/wukong_parser/src/lib.rs:1714-1743`.
- **Tensor dimension and layout annotations** — `crates/wukong_parser/src/lib.rs:397-464` — Dims parse as integer, `?` dynamic, or symbolic name; a trailing `.contiguous`/`.col_major`/`.strided`/`.tiled(a,b)` layout is validated with E0204 on an unknown name.
**`crates/wukong_sema/src/lib.rs`**

- **Tensor dimension resolution: generic vs const vs unknown (E0504)** — `crates/wukong_sema/src/lib.rs:725-773` — A dim name resolves to a symbolic `Var` for a declared generic, to a folded `Const` for a top-level const, else E0504 with a Levenshtein did-you-mean; hint code at `crates/wukong_sema/src/lib.rs:780-794` and `crates/wukong_sema/src/lib.rs:3086-3100`.
- **Return-value checking: rigid shape, kind clash, scalar agreement** — `crates/wukong_sema/src/lib.rs:989-1036` — Unifies the declared return shape against the returned value with `rigid=true` so a generic function cannot lie about its output shape, and rejects pointer/aggregate-vs-scalar and non-literal scalar mismatches.
- **Operator/assignment/arm shape unification** — `crates/wukong_sema/src/lib.rs:1045-1057` — `check_binop_shapes` rigidly unifies tensor-tensor and vector-vector operand shapes, the hook reused by assignment, `if`/`match` arm merges and `break` value joins.
**`crates/wukong_sema/src/shape.rs`**

- **Call checking: arity, turbofish generic binding, dim/type substitution maps** — `crates/wukong_sema/src/shape.rs:369-446` — Builds the `dims`/`tys` substitutions from explicit generic args (int literal to `Dim::Const`, scalar name to type, otherwise `Dim::Var`) then unifies every parameter.
- **Unification core: tensor element/rank/dim plus cross-kind mismatch reporting** — `crates/wukong_sema/src/shape.rs:448-498` — Compares tensor element scalars (E0502) and ranks (E0501) then per-axis dims; kind clashes and the kind-namer live at `crates/wukong_sema/src/shape.rs:622-674` and `crates/wukong_sema/src/shape.rs:52-64`.
- **Array-to-tensor decay with const-product divisibility check** — `crates/wukong_sema/src/shape.rs:499-573` — Accepts an array for a tensor parameter but binds a rank-1 symbolic dim to the array length and requires the length be a multiple of the const-dim product.
- **Structural unify arms: pointer, ref, slice-unsizing, vector, scalar** — `crates/wukong_sema/src/shape.rs:574-621` — Recurses through pointer/ref pointees and slice/array element types, and reports exact vector lane/element and scalar type mismatches as E0401.
- **Dimension unification with rigid vs inference modes** — `crates/wukong_sema/src/shape.rs:680-753` — Rigid mode (body/return checks) demands identity so a generic cannot lie about its output shape; inference mode binds free vars but refuses to bind `Dynamic`.
- **Index typing: rank check, static bounds, bool index and non-indexable base rejection** — `crates/wukong_sema/src/shape.rs:785-881` — Checks index count against tensor rank, bounds-checks constant indices against const dims and array lengths, and rejects scalar/struct/tuple bases.
- **Dimension equality relation** — `crates/wukong_sema/src/shape.rs:892-899` — Treats `Dynamic` as matching anything, `Const` equal by value and `Var` equal only by symbol, making `?` the deliberate escape hatch in both unify modes.
- **Turbofish dimension-literal parser** — `crates/wukong_sema/src/shape.rs:901-907` — Takes the leading ASCII digits of a generic-argument token and `unwrap_or(0)`, so any unparsable dim text silently becomes dimension zero with no diagnostic.
- **Generic substitution machinery: type-generic inference and shape substitution** — `crates/wukong_sema/src/shape.rs:913-973` — `infer_type_generics` binds `Ty::Named` parameter positions (first binding wins) recursing through structure; `apply_subst` rewrites `Dim::Var`s and named types through the whole type.
**`crates/wukong_types/src/lib.rs`**

- **Scalar type lattice** — `crates/wukong_types/src/lib.rs:10-106` — The sixteen primitive scalars with name round-tripping, byte size/align and float/int/signed predicates; note `is_int()` is true for `Char`, which therefore passes integer-argument gates.
- **Dim, Shape and the symbolic-dim ABI source of truth** — `crates/wukong_types/src/lib.rs:108-139` — `Dim::{Const,Var,Dynamic}` plus `Shape::symbolic_dims`, which lists `Var` dims in axis order and defines the hidden `i64` runtime-dim parameters for symbolic-generic functions.
- **Semantic type vocabulary `Ty` and tensor `Layout`** — `crates/wukong_types/src/lib.rs:141-212` — The full type enum including shape-typed `Tensor`, SIMD `Vector`, `Unknown`/`Error` leniency sentinels; the `Layout` variants are carried but never compared by unification.

## G05 — Diagnostics: catalogue, rendering, JSON output

*Files:* `crates/wukong_diag/src/catalog.rs`, `crates/wukong_diag/src/json.rs`, `crates/wukong_diag/src/lib.rs`, `crates/wukong_diag/src/render.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_parser/src/lib.rs`, `crates/wukong_types/src/lib.rs`

**`crates/wukong_diag/src/catalog.rs`**

- **Code-range convention and its gaps** — `crates/wukong_diag/src/catalog.rs:7-14` — Doc-comment declares the E00xx/E01xx/…/C0xxx numbering scheme, but E0001 is emitted by no compiler stage (only diag's own test) and E0404 is missing entirely between E0403 and E0405.
- **Case-insensitive explanation lookup** — `crates/wukong_diag/src/catalog.rs:23-32` — `explain` upper-cases the query and linearly scans the catalogue; `all()` exposes the whole table for listing, both re-exported through `wukong_driver` for the `wukongc --explain` CLI path.
- **Stable error-code catalogue table** — `crates/wukong_diag/src/catalog.rs:44-251` — Static array of 30 `Explanation` entries (E0001, E01xx lexer, E02xx parser, E03xx resolution, E04xx types, E05xx shapes, C0001) each with title and multi-sentence fix guidance backing `--explain`.
- **Catalogue quality gate test** — `crates/wukong_diag/src/catalog.rs:253-272` — Unit tests assert case-insensitivity, that every code starts with `E`/`C`, has a non-empty title, and a body longer than 20 chars, preventing thin placeholder explanations.
**`crates/wukong_diag/src/json.rs`**

- **Hand-rolled JSON string escaper** — `crates/wukong_diag/src/json.rs:18-42` — Dependency-free escaping of quotes, backslash, `\n\r\t` and any control char below 0x20 as `\u00xx`; note it does not escape DEL or emit `\uXXXX` surrogate pairs, passing raw UTF-8 through.
- **JSON Lines diagnostic emitter** — `crates/wukong_diag/src/json.rs:44-99` — Serializes one diagnostic to a single-line object with severity, optional code, message, a `spans` array (file/line/col/primary/label, dummy spans skipped) and a `notes` array, for `--error-format=json` tooling.
**`crates/wukong_diag/src/lib.rs`**

- **Severity taxonomy and ANSI color palette** — `crates/wukong_diag/src/lib.rs:18-50` — Five severities (Bug/Error/Warning/Note/Help) with header text and hard-coded bold ANSI codes; Note/Help are only reachable via `Diagnostic::new`, having no constructor helper.
- **Diagnostic / Label / NoteKind data model** — `crates/wukong_diag/src/lib.rs:52-86` — Core carrier struct: severity, optional `&'static str` code, message, ordered labels (primary vs secondary flag), and note/help pairs, shared by both renderers.
- **Fluent diagnostic builder and primary-span rule** — `crates/wukong_diag/src/lib.rs:88-156` — Chainable `error/warning/bug/with_code/primary/secondary/note/help` constructors plus `primary_span`, which falls back to the first label when no primary exists — the anchor both the `-->` line and tests depend on.
- **DiagnosticSink error accounting** — `crates/wukong_diag/src/lib.rs:158-198` — Collects diagnostics and counts errors (Bug counts as an error); `take()` drains and resets the counter, so a caller that drains mid-compile loses the has-errors signal.
**`crates/wukong_diag/src/render.rs`**

- **Color toggle and span colors** — `crates/wukong_diag/src/render.rs:10-27` — `Renderer { color }` gates every escape through `paint`, giving byte-identical plain output for pipes and golden tests; primary/secondary underline colors are chosen by `span_color`, also at `crates/wukong_diag/src/render.rs:143-149`.
- **Rustc-style header and source-location line** — `crates/wukong_diag/src/render.rs:29-65` — Builds `error[E0501]: message` from severity plus optional code, then a padded `--> file:line:col` arrow derived from `primary_span`, skipped entirely for dummy spans.
- **Gutter sizing, dummy-span filtering and label ordering** — `crates/wukong_diag/src/render.rs:40-80` — Gutter width comes from the largest label line number; dummy spans are dropped and remaining labels sorted by (source id, lo) so frames print in source order, one frame per label with no same-line merging.
- **Trailing note/help lines** — `crates/wukong_diag/src/render.rs:82-91` — Emits `= note:` / `= help:` rows aligned to the gutter after the source frames, with the tag bolded and the `=` painted in the frame color.
- **Caret/underline geometry** — `crates/wukong_diag/src/render.rs:96-140` — Computes caret run length from the span's first line only (multi-line spans clamp to `line_chars+1 - col`), pads with `col-1` spaces, and uses `^` red for primary vs `-` blue for secondary; tabs in source misalign the underline.
- **Golden-output regression tests** — `crates/wukong_diag/src/render.rs:151-204` — Exact-string tests pin the caret frame, the no-label form, and the color-on/off contract; companion tests cover JSON well-formedness and escaping at `crates/wukong_diag/src/json.rs:101-132` and the builder/sink at `crates/wukong_diag/src/lib.rs:200-229`.
**`crates/wukong_mir_build/src/lib.rs`**

- **C0001 "not yet supported by codegen" hard-error gate** — `crates/wukong_mir_build/src/lib.rs:2337-2347` — Emits a hard diagnostic when lowering meets a construct the front end accepted, so the compiler refuses to emit broken MIR instead of silently miscompiling.
**`crates/wukong_parser/src/lib.rs`**

- **Recursion-depth budget and one-shot E0209 latch** — `crates/wukong_parser/src/lib.rs:159-193` — MAX_DEPTH=1024 charged by every recursive production; `too_deep` latches `depth_exceeded`, which silences all follow-on recovery diagnostics so one clean error is emitted.
- **Depth-charge sites across the grammar** — `crates/wukong_parser/src/lib.rs:487-524` — Iterative binop/cast folds, prefix, type, if, match, block and pattern productions each charge and restore `depth`, at `crates/wukong_parser/src/lib.rs:530-565`, `crates/wukong_parser/src/lib.rs:962-975`, `crates/wukong_parser/src/lib.rs:1055-1077`, `crates/wukong_parser/src/lib.rs:1396-1410`.
**`crates/wukong_types/src/lib.rs`**

- **Type rendering for diagnostics** — `crates/wukong_types/src/lib.rs:275-325` — Renders every `Ty` variant to source-like text, resolving interned symbols and printing dims as constant, variable name, or `?`, which every shape error message consumes.

# Middle end

## G06 — MIR Data Model: instructions, builder, printer, verifier

*Files:* `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir/src/builder.rs`, `crates/wukong_mir/src/inst.rs`, `crates/wukong_mir/src/lib.rs`, `crates/wukong_mir/src/print.rs`, `crates/wukong_mir/src/verify.rs`, `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Hand-built SIMD MIR verifier + execution equality test** — `crates/wukong_codegen_cranelift/src/tests.rs:4341-4420` — Constructs vector load/splat/fadd/store MIR with `Builder`, asserts `verify_function` is clean, and requires the interpreter's lane-wise arena and real SSE vectors to agree.
**`crates/wukong_mir_build/src/lib.rs`**

- **Merge-param representation and slice fat-pointer layout** — `crates/wukong_mir_build/src/lib.rs:1543-1554` — `merge_repr_ty` demotes Array merge params to Ptr so mem2reg's verifier accepts aggregate-valued if/match; slice layout constants live at `crates/wukong_mir_build/src/lib.rs:1537-1541`.
**`crates/wukong_mir/src/builder.rs`**

- **Builder construction lifecycle and value/block arenas** — `crates/wukong_mir/src/builder.rs:19-56` — Auto-creates the entry block, hands out monotonic ValueIds/BlockIds, tracks a "current" block and reassembles a Function in `finish`, also `crates/wukong_mir/src/builder.rs:77-90` and `crates/wukong_mir/src/builder.rs:177-188`.
- **Mid-construction introspection accessors** — `crates/wukong_mir/src/builder.rs:40-44` — `func_name`, `value_type` and `block` expose name, value types and block params before `finish()`, letting lowering distinguish aggregate base pointers from scalars, also `crates/wukong_mir/src/builder.rs:63-75`.
- **Builder parameter and instruction-append API** — `crates/wukong_mir/src/builder.rs:92-127` — `add_param` (entry-block param doubling as function param), `block_param` for merge points, and push/build/build_void which allocate the result value from the requested type.
- **Entry-block alloca hoisting** — `crates/wukong_mir/src/builder.rs:129-139` — `alloca` appends the slot to the entry block rather than the current block so every stack slot dominates all uses and never lands inside a loop body.
- **Builder terminator helpers and declared-return-type accessor** — `crates/wukong_mir/src/builder.rs:141-175` — set_term/ret/br/cond_br write the current block's terminator (default Unreachable), and `ret_type` lets the lowerer coerce tail values so `Ret` is well-typed.
**`crates/wukong_mir/src/inst.rs`**

- **BinOp set with explicit signedness** — `crates/wukong_mir/src/inst.rs:6-58` — Eighteen integer/float/bitwise/shift operators separating SDiv/UDiv and SRem/URem, with `name()` for printing and `is_float()` for backend dispatch.
- **CmpOp predicate set** — `crates/wukong_mir/src/inst.rs:60-108` — Sixteen i1-producing predicates covering signed, unsigned and *ordered-only* float comparisons; no unordered (uno/ueq) predicates exist, so NaN-aware comparisons cannot be expressed.
- **CastKind conversion set** — `crates/wukong_mir/src/inst.rs:110-145` — Twelve conversions (sext/zext/trunc, fp↔int both signednesses, fpext/fptrunc, bitcast, ptr↔int) naming the full legal cast surface backends must implement.
- **Core value, arithmetic and memory op set** — `crates/wukong_mir/src/inst.rs:149-174` — ConstInt/ConstFloat, Bin/Cmp/Neg/Not/Cast/Select plus Alloca, typed Load, Store and element-strided Gep, the scalar backbone every backend and the interpreter must cover.
- **Call and address-of ops (Call, FuncAddr, GlobalAddr)** — `crates/wukong_mir/src/inst.rs:175-188` — Direct symbol calls plus pure function-address and static-data-address materialization, the mechanism behind @parallel outlined-body callbacks and string literals.
- **SIMD/math primitive ops: Splat, Fma, Sqrt** — `crates/wukong_mir/src/inst.rs:189-202` — Scalar-to-lane broadcast, single-rounding fused multiply-add contracted from `x + y*z`, and hardware sqrt, each mirrored bit-exactly by the interpreter.
- **Round op with RoundMode bit-exactness contract** — `crates/wukong_mir/src/inst.rs:203-237` — Nearest/Floor/Ceil/Trunc rounding where Nearest is explicitly ties-to-even (`round_ties_even`, not `round`) so interpreter and Cranelift agree bit-for-bit.
- **Terminator family with branch arguments** — `crates/wukong_mir/src/inst.rs:246-262` — Ret/Br/CondBr/Unreachable, where Br and CondBr carry per-edge argument lists feeding the destination block's params, replacing phi nodes.
**`crates/wukong_mir/src/lib.rs`**

- **Value/block identifiers and the block-param SSA block** — `crates/wukong_mir/src/lib.rs:25-43` — Newtype u32 ids with custom Debug printing `v#`/`bb#`, and blocks carrying typed params instead of phi nodes, also `crates/wukong_mir/src/lib.rs:130-137`.
- **MirType lattice and Scalar→MirType lowering** — `crates/wukong_mir/src/lib.rs:45-82` — Signless integer/float/Ptr/Vec/Array/Void type enum plus `from_scalar`, collapsing u/i pairs, usize/isize to I64 and Char to I32.
- **MirType predicates and textual notation** — `crates/wukong_mir/src/lib.rs:84-127` — `is_int`/`is_float`/`is_vector`/`lane_type` classify SIMD arithmetic by lane type, and `display` renders the `<4 x f32>` vector and `[N x T]` array spellings the whole toolchain greps for.
- **Function representation: value-type arena and accessors** — `crates/wukong_mir/src/lib.rs:139-164` — Function holds params, ret type, block vector, a ValueId-indexed `value_types` arena, entry block and its vec_kernels; `value_type`/`block` index directly and panic out of range.
- **MirLevel lowering-stage invariant (comment-vs-code divergence)** — `crates/wukong_mir/src/lib.rs:166-171` — Declares High/Low stages, but `MirLevel::High` is constructed nowhere in the repo and `Program::new` starts at Low, so the documented High→Low advance never happens.
- **Program container plus rodata StaticData blobs** — `crates/wukong_mir/src/lib.rs:173-206` — Read-only NUL-terminated string blobs addressed by `Op::GlobalAddr` so returned `*u8` stays valid, with a linear-scan `Program::function` name lookup.
**`crates/wukong_mir/src/print.rs`**

- **Deterministic MIR text printer for --emit=mir** — `crates/wukong_mir/src/print.rs:6-48` — Renders each function header, block labels with typed params and terminators, but silently omits `Program::statics` and `Program::level`, so emitted text cannot round-trip string literals, also `crates/wukong_mir/src/print.rs:110-140`.
- **Exhaustive op renderer fmt_op** — `crates/wukong_mir/src/print.rs:62-108` — Covers all twenty Op variants with per-op syntax; ConstFloat uses plain Display so `1.0` prints as `1`, making float and int constants indistinguishable apart from the type suffix.
- **Builder/printer round-trip unit gate** — `crates/wukong_mir/src/print.rs:142-168` — Builds a two-param add function and asserts the exact printed text, the only in-crate guard against silent drift in value numbering or MIR syntax.
**`crates/wukong_mir/src/verify.rs`**

- **MIR verifier entry points and per-function driver** — `crates/wukong_mir/src/verify.rs:13-67` — Walks every function's blocks collecting defined values, checks each block's index matches its `BlockId`, then verifies each op and terminator, returning plain error strings.
- **Flow-insensitive use-before-def check** — `crates/wukong_mir/src/verify.rs:44-55` — `defined` is populated from ALL blocks before checking, so `use_val` only catches values defined nowhere; it cannot detect dominance violations or use-before-def across blocks.
- **Silent-skip typing when `value_types` is short** — `crates/wukong_mir/src/verify.rs:69-96` — `ty()` returns `None` for out-of-range ValueIds and every checker (`expect_ty`, `check_result_is`, `result_ty`) then passes silently, so a missing type entry disables all type verification for that value.
- **Result-presence rules per opcode** — `crates/wukong_mir/src/verify.rs:98-107` — Requires a result for every op except `Store`/`Call`/`VecKernelCall` and forbids one on `Store`; the doc-comment above lists only Store and Call, omitting the VecKernelCall exemption the code actually applies.
- **Constant, arithmetic and negation typing rules** — `crates/wukong_mir/src/verify.rs:110-150` — Checks `ConstInt`/`ConstFloat` tag types, forces both `Bin` operands to equal the result type, and classifies float-vs-int by `lane_type()`, exempting `Xor`/`And`/`Or` so bitwise ops on `i1` booleans pass.
- **Comparison and vector-mask width invariants** — `crates/wukong_mir/src/verify.rs:151-184` — Requires both `Cmp` operands to share a type, the predicate's float-ness to match the lane type, and a `Vec(_,n)` compare to yield an n-lane mask; scalar compares must yield `I1`.
- **Select mask-shape invariant** — `crates/wukong_mir/src/verify.rs:196-222` — Both `Select` arms must equal the result type; a vector result demands a same-lane-count vector mask and returns early, while a scalar result demands an `I1` condition.
- **Memory-op pointer and index typing** — `crates/wukong_mir/src/verify.rs:223-248` — `Alloca`/`FuncAddr`/`GlobalAddr`/`Gep` must yield `Ptr`, `Load`/`Store`/`Gep` bases must be `Ptr`, `Load` result must match its annotated type, and `Gep` indices must be integral; also `crates/wukong_mir/src/verify.rs:271-276`.
- **VecKernelCall ABI verification** — `crates/wukong_mir/src/verify.rs:254-270` — Enforces the 256-bit kernel call contract: `ptrs` and `scalars` must be `Ptr` and the element count `n` must be an integer, matching the fixed `fn(*const *mut u8, *const f32, u64)` kernel ABI.
- **Float-intrinsic and splat shape checks** — `crates/wukong_mir/src/verify.rs:277-329` — `Fma`/`Sqrt`/`Round` require a float lane type and all operands equal to the result, and `Splat` requires a vector result whose lane type equals the scalar operand's type.
- **Terminator and CFG-edge invariants** — `crates/wukong_mir/src/verify.rs:347-411` — Checks `ret` value type against `Function::ret` (and void-ness for `ret` with no value), `cond_br` condition is `I1`, branch targets exist, and every edge's arg count and types match the target block's parameters.

## G07 — MIR Lowering Core: monomorphization, function lowering, ABI, aggregates

*Files:* `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_types/src/lib.rs`

**`crates/wukong_mir_build/src/lib.rs`**

- **Alloca-per-local lowering strategy (stale module doc)** — `crates/wukong_mir_build/src/lib.rs:1-10` — Header states every local gets an alloca promoted later by mem2reg, but falsely claims tensor/SIMD/parallel constructs are unlowered placeholders.
- **Array-repeat unroll threshold** — `crates/wukong_mir_build/src/lib.rs:26-28` — `REPEAT_UNROLL_LIMIT = 8` decides whether `[v; n]` becomes straight-line stores or a fill loop; a pure IR-size versus speed tuning knob.
- **Mono instantiation table** — `crates/wukong_mir_build/src/lib.rs:42-66` — Carries instance_of keys, per-function value-type generics, ordered instance list, and the string blob table threaded into lowering; `tg()` gates all call redirection.
- **Generic type substitution walker** — `crates/wukong_mir_build/src/lib.rs:68-95` — `subst_ty` structurally rewrites `Ty::Named` generics through ptr/ref/slice/array/tuple/fn, short-circuiting on an empty substitution; used for both params and returns.
- **Monomorphization name mangling** — `crates/wukong_mir_build/src/lib.rs:199-215` — `mono_type_name` builds stable ASCII type names (`p_`, `r_`, `s_`, `aN_`, `tN_` prefixes) with an unnamed `"x"` catch-all that can alias distinct exotic types.
- **Generic argument binding and canonical key** — `crates/wukong_mir_build/src/lib.rs:217-254` — `bind_generics` unifies param against concrete arg types first-binding-wins; `canon_type_args` refuses under-constrained instantiations, keeping partially generic calls out of the instance set.
- **Transitive monomorphization collector** — `crates/wukong_mir_build/src/lib.rs:256-499` — `MonoCollector` walks every AST statement/expression from concrete callers, then drains a worklist so generic-calls-generic resolves; instances are a Vec for deterministic emission order.
- **Static string `.rodata` interning** — `crates/wukong_mir_build/src/lib.rs:501-537` — Collects every string literal from bodies and const initializers, NUL-terminates, dedups by content, and mints `wukong$rodata$N` blobs so returned `*u8` never dangles, with mirror walkers at `crates/wukong_mir_build/src/lib.rs:539-669`.
- **Monomorphized instance emission and par-region append** — `crates/wukong_mir_build/src/lib.rs:1196-1214` — Re-lowers each generic template once per collected substitution under its mangled name, then appends outlined parallel-region bodies last since backends resolve by name.
- **Function lowering entry: signature substitution and parameter ABI** — `crates/wukong_mir_build/src/lib.rs:1372-1459` — Resolves sema signature under the mono substitution, binds hidden dim params, then declares params contiguously: aggregates bound directly as base pointers, scalars stored into allocas.
- **Aggregate return sret ABI** — `crates/wukong_mir_build/src/lib.rs:1397-1426` — Struct/tuple/slice returns become void plus a hidden leading destination pointer; the fall-through path deep-copies only genuine pointer tails, guarding an -O0-versus-O2 verifier ICE, at `crates/wukong_mir_build/src/lib.rs:1462-1497`.
- **Aggregate classification for ABI selection** — `crates/wukong_mir_build/src/lib.rs:1501-1526` — Free-standing `ty_is_aggregate`/`enum_data_carrying` mirror the lowerer's registry-aware classification so the sret decision can be made before any Builder exists.
- **VariantCtor payload borrow + slice fat-pointer layout** — `crates/wukong_mir_build/src/lib.rs:1528-1541` — Enum-variant initializer view (unit/tuple/struct) consumed by `construct_enum_into`, plus the 16-byte slice layout constants (data ptr at 0, i64 len at 8).
- **FnLowerer state struct** — `crates/wukong_mir_build/src/lib.rs:2293-2334` — The per-function lowering context: scope stack, labeled loop stack, sret slot, generic substitution and monomorphization table, plus the dispatch-mode flags `parallel_fn` and `par`.
- **Lexical scope stack for local name→(slot, MirType) binding** — `crates/wukong_mir_build/src/lib.rs:2351-2370` — Push/pop/bind/lookup over a stack of hash maps resolving names innermost-first; every local, param, and hidden dim binding flows through it.
- **Registry-aware semantic-type → MIR-type lowering (`mir_ty_of`)** — `crates/wukong_mir_build/src/lib.rs:2422-2454` — Resolves generic params via subst and maps named structs, data-carrying enums, tuples, and slices to sized `[N x i8]` byte buffers instead of the free `mir_ty`'s I32 fallback.
- **Annotated-local slot typing from AST type expressions (`mir_ty_of_ann`)** — `crates/wukong_mir_build/src/lib.rs:2463-2519` — Single resolver for `let x: T` slots covering scalars, structs, data enums, const-length arrays, padded tuples, 16-byte slices, pointers, and vectors; unknown paths silently default to I32.
- **Deterministic zero-initialization of no-initializer `let` slots** — `crates/wukong_mir_build/src/lib.rs:2528-2552` — Typed zero store for scalars, unrolled-or-loop fill for scalar arrays, recursive zeroing for aggregates; deliberately skips Ptr/Vec slots, matching interpreter zero memory, with `const_zero` at `crates/wukong_mir_build/src/lib.rs:3236-3245`.
- **Aggregate-by-pointer parameter ABI (`param_abi`)** — `crates/wukong_mir_build/src/lib.rs:2557-2562` — Any type whose MIR form is an Array byte buffer is passed as a base pointer while scalars pass by value, defining the whole aggregate calling convention.
- **Registry-aware size and alignment of semantic types** — `crates/wukong_mir_build/src/lib.rs:2594-2635` — `ty_size`/`ty_align` resolve named structs and both C-style and tagged enums through sema and recurse into arrays and tuples, which `Ty::size_of` alone cannot do.
- **Padded field-layout authority shared by structs, tuples, and enum payloads** — `crates/wukong_mir_build/src/lib.rs:2641-2654` — `aggregate_layout` accumulates round-up offsets, total size, and max alignment, returning None if any field is genuinely unsized.
- **Declared-struct layout accessor family** — `crates/wukong_mir_build/src/lib.rs:2660-2704` — `struct_layout`, `struct_field_tys`, `struct_size`, and `struct_align` project the shared aggregate layout into MIR-typed, semantic-typed, size, and alignment views used by init, copy, and access paths.
- **Tuple destructuring pattern binding with nested recursion** — `crates/wukong_mir_build/src/lib.rs:2716-2735` — Binds each sub-pattern to its field's byte-offset place, recursing into nested tuple patterns and ignoring wildcards; the preceding doc comment at 2706-2710 actually describes `struct_field_place` and is orphaned onto this function.
- **C-style enum variant access folded to its discriminant constant** — `crates/wukong_mir_build/src/lib.rs:2740-2759` — `enum_variant_value` recognizes `E::B` as a single-segment path field access on a declared enum and yields the variant's integer discriminant for direct constant lowering.
- **Data-carrying enum representation predicate and tagged-union layout** — `crates/wukong_mir_build/src/lib.rs:2765-2787` — One predicate decides byte-buffer vs i32-discriminant representation everywhere; `enum_layout` computes size/align/payload-offset as an i32 tag plus the union of all variant payloads.
- **Enum variant field offset resolution (positional and named)** — `crates/wukong_mir_build/src/lib.rs:2793-2828` — Absolute payload-relative offsets per variant field, plus a name-tagged variant for struct-payload literals, with `field_base_data_enum` disambiguating data-enum variant paths at `crates/wukong_mir_build/src/lib.rs:2833-2846`.
- **Slice fat-pointer construction and interp-safe copy** — `crates/wukong_mir_build/src/lib.rs:2860-2915` — Array sources store `{base, static len}` (unsizing) while slice sources copy the two fields as typed Ptr/I64, never as 16 i8 loads which would truncate a length ≥128 under interpreter slot memory; `let_is_slice` gates it at `crates/wukong_mir_build/src/lib.rs:2851-2858`.
- **Enum variant value construction into a tagged-union buffer** — `crates/wukong_mir_build/src/lib.rs:2938-2992` — Allocates or reuses the buffer, zeroes padding and inactive bytes for backend-deterministic contents, stores the i32 tag, then routes tuple or named payload fields through `init_field`, erroring on unknown field names.
- **Place base pointer for explicit-deref aggregate access** — `crates/wukong_mir_build/src/lib.rs:3002-3010` — For `(*p).f`/`(*p)[i]` uses `p`'s pointer value rather than a Load of the whole aggregate, avoiding invalid `gep [N x i8]` MIR that previously segfaulted at native -O0.
- **Struct field place resolution with auto-deref** — `crates/wukong_mir_build/src/lib.rs:3012-3035` — Resolves the struct symbol through `Ty::Named`, `*T`, or `&T`, GEPs the declared field offset, and falls back to a dummy alloca plus a C0001 error for non-struct bases.
- **Struct literal initialization and the shared `init_field` primitive** — `crates/wukong_mir_build/src/lib.rs:3041-3107` — Routes literal fields to declared offsets by name regardless of literal order, recursing nested struct/tuple/array literals directly into place and deep-copying or coercing everything else.
- **Interp-exact aggregate deep copy (`emit_copy`)** — `crates/wukong_mir_build/src/lib.rs:3115-3190` — Copies structs/tuples/arrays leaf-by-leaf with the same GEP discipline as access, tagged enums byte-by-byte via `emit_copy_bytes`, and slices as typed fat-pointer fields, because a flat memcpy breaks the interpreter's slot memory.
- **By-pointer aggregate read convention and element GEP** — `crates/wukong_mir_build/src/lib.rs:3196-3218` — `load_or_addr` returns the pointer itself for `MirType::Array` fields instead of loading the buffer through a register, and `gep_elem` builds the element-typed constant-index GEP.
- **Block lowering driver with ordered kernel-recognition probe chain** — `crates/wukong_mir_build/src/lib.rs:3249-3364` — Walks statements, attempting each recognizer window in a documented disjointness order and skipping the consumed statement count, otherwise falling back to plain statement lowering.
- **Statement lowering dispatcher (lower_stmt)** — `crates/wukong_mir_build/src/lib.rs:6732-7004` — The single entry point routing every StmtKind to its lowering, owning the `terminated` flag that suppresses dead branches after return/break/continue.
- **`let` slot typing and initializer routing** — `crates/wukong_mir_build/src/lib.rs:6734-6858` — Five-arm slot-type matrix (annotation vs init, slice fat-pointer, nested aggregates) plus per-shape init lowering, deep-copy, tuple destructuring, and deterministic zero-init.
- **Assignment lowering: whole-aggregate copy and scalar coercion** — `crates/wukong_mir_build/src/lib.rs:6859-6925` — Aggregate stores route through a temp then emit_copy (fixing self-referential literals); scalar stores coerce to the place type and re-round bf16/f16 compound results.
- **Return lowering: sret aggregate copy and return-type coercion** — `crates/wukong_mir_build/src/lib.rs:6929-6946` — Aggregate returns deep-copy into the caller buffer and return void; scalars coerce to the declared return type, also at `crates/wukong_mir_build/src/lib.rs:7056-7063`.
- **Aggregate initialization and field addressing** — `crates/wukong_mir_build/src/lib.rs:14819-15004` — Array-literal/repeat init with deep-copy for aggregate elements, an unroll-vs-fill-loop threshold, the byte-offset `field_ptr` GEP primitive, and tuple literal init/field places.
- **Lvalue/place lowering (`lower_place`)** — `crates/wukong_mir_build/src/lib.rs:15192-15263` — Resolves names, derefs, array/slice indexing (loading the slice fat pointer's data pointer), tuple/struct fields and multi-dim tensor indices to an address plus element type.
- **Hidden symbolic-dim call ABI** — `crates/wukong_mir_build/src/lib.rs:16269-16390` — Computes per-call runtime dimension arguments from turbofish then argument shapes, first-binding-wins, emitting C0001 and a placeholder zero when a dim is undetermined.
- **MIR type layout and ABI decay** — `crates/wukong_mir_build/src/lib.rs:18306-18363` — Maps sema types to MIR (tuples and aggregates become flat byte-buffer arrays, arrays decay to pointers at calls) and computes byte size/alignment for no-init buffers.
**`crates/wukong_types/src/lib.rs`**

- **Sized-type layout: size, align and tuple field offsets** — `crates/wukong_types/src/lib.rs:214-273` — Computes byte sizes with a shared `round_up` accumulation, treats a slice as a 16-byte fat pointer, and exposes `tuple_offsets` as the aggregate-GEP source of truth.

## G08 — MIR Lowering: control flow, expressions, calls, coercions, intrinsics

*Files:* `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_mir_build/src/lib.rs`**

- **merge_repr_ty control-flow merge representation** — `crates/wukong_mir_build/src/lib.rs:1543-1554` — Maps an `Array` result type to `Ptr` for `if`/`match` value merge params, fixing the -O0-passes/-O2-verifier-rejects divergence on aggregate-valued branches.
- **Runtime intrinsic symbols: alloc, file I/O, clock, printing** — `crates/wukong_mir_build/src/lib.rs:2115-2138` — Heap alloc/free, the 12 typed raw little-endian read/write entry points returning count/-1/-2, `now_ns`, plus string and unsigned print entry points, also `crates/wukong_mir_build/src/lib.rs:1729-1737`.
- **Subst-aware expression typing with slice `.len()` i64 override** — `crates/wukong_mir_build/src/lib.rs:2374-2415` — `expr_ty` substitutes monomorphized generics; `expr_mir` special-cases zero-arg `.len()` on a slice to I64 because sema types it Unknown, preventing a verifier-rejecting i32/i64 compare.
- **print/println argument-kind predicates (string vs unsigned)** — `crates/wukong_mir_build/src/lib.rs:2381-2394` — `is_string_arg` routes `*u8` to byte rendering and `is_unsigned_int_arg` routes unsigned scalars to magnitude printing, excluding bool and signed/float.
- **Array→slice unsizing coercion at call arguments** — `crates/wukong_mir_build/src/lib.rs:2922-2934` — When a parameter is `[]T` and the argument is `[T; N]`, materializes a fresh 16-byte fat-pointer temporary and passes its address; every other argument lowers verbatim.
- **Enum discriminants treated as signed in coercions and casts** — `crates/wukong_mir_build/src/lib.rs:3220-3234` — `signed()` reports true for a `Ty::Named` that resolves to an enum so `E::Neg as i64` sign-extends, while struct names stay unsigned; previously a gate-blind both-backend miscompile.
- **break/continue lowering, loop-target resolution, and unimplemented `defer`** — `crates/wukong_mir_build/src/lib.rs:6957-7002` — Branches to the resolved loop's continue/exit block passing a coerced (or zero) merge argument; `defer` only reports "unsupported" and lowers for effects, with `find_loop` at `crates/wukong_mir_build/src/lib.rs:7035-7048`.
- **Statement-position control-flow lowering (lower_expr_stmt, lower_while, lower_loop)** — `crates/wukong_mir_build/src/lib.rs:7007-7201` — Builds header/body/exit blocks, pushes the label-and-target loop stack, and dispatches statement-position if/block/loop expressions to their value-discarding forms.
- **Value-producing `loop` lowering (lower_loop_value)** — `crates/wukong_mir_build/src/lib.rs:7209-7244` — Gives the exit block a typed merge param that each `break v` feeds, using merge_repr_ty so aggregates flow as pointers; break-less loops yield a dummy zero.
- **For-over-array desugar fallback** — `crates/wukong_mir_build/src/lib.rs:9854-9875` — Non-range iterators route to `lower_for_array`, rewriting `for p in arr` into an indexed range loop for statically-sized arrays and emitting an unsupported diagnostic otherwise.
- **Loop-counter width unification** — `crates/wukong_mir_build/src/lib.rs:9877-9890` — Drives the scalar loop by the wider of the start/end integer types so `for i in 0..n` with an i64 `n` cannot emit verifier-invalid `cmp.i32 i32, i64` MIR.
- **Scalar for-loop CFG lowering with latch-targeted `continue`** — `crates/wukong_mir_build/src/lib.rs:9924-10000` — Builds header/body/latch/exit blocks, selects Slt/Sle/Ult/Ule from inclusivity and signedness, and registers the latch (not the header) as the `continue` target so the step always runs.
- **`for x in array/slice` desugaring** — `crates/wukong_mir_build/src/lib.rs:10674-10808` — Lowers iteration over an array or slice into the indexed range CFG, hoisting base pointer and trip count; the doc comment claims fixed-size arrays only, but slices are handled.
- **Boolean-context condition normalization (`lower_bool_cond`)** — `crates/wukong_mir_build/src/lib.rs:14653-14677` — Normalizes float conditions to `!= 0.0` and wide integer conditions to `!= 0` so `cond_br` always gets an `i1`, closing a three-way interp/native/-O0-vs-O2 truthiness divergence.
- **`if` statement and value lowering with merge params** — `crates/wukong_mir_build/src/lib.rs:14679-14813` — Value form coerces each arm to the joined merge type (aggregates flow as `Ptr`) and terminates the merge `Unreachable` when both arms diverge, avoiding a -O0-only verify ICE.
- **Expression lowering dispatch and literal materialization** — `crates/wukong_mir_build/src/lib.rs:15267-15443` — Lowers int/float/bool/char/string literals (strings prefer a `.rodata` `GlobalAddr`, falling back to a frame-local NUL-terminated buffer), inlines top-level consts, and routes all other expression forms.
- **`match` lowering as a guarded if-else chain** — `crates/wukong_mir_build/src/lib.rs:15453-15556` — Evaluates the scrutinee once, tests each arm, stops at an unconditional catch-all, and emits `Unreachable` (not a zero default) for the non-matching fallthrough; arm bodies at `crates/wukong_mir_build/src/lib.rs:16002-16031`.
- **Pattern binding walks and variant payload field resolution** — `crates/wukong_mir_build/src/lib.rs:15562-15625` — Copies scalar bindings into fresh slots while aggregates bind by address, recursing through tuple and variant sub-patterns; variant payload offsets at `crates/wukong_mir_build/src/lib.rs:15894-15994`.
- **Pattern test lowering (`pattern_cond`, guards, aggregate scrutinees)** — `crates/wukong_mir_build/src/lib.rs:15658-15860` — Emits equality tests for int/char/bool/enum patterns, signed/unsigned range bounds, AND over tuple fields and OR over alternatives; guard normalization at `crates/wukong_mir_build/src/lib.rs:15630-15652`.
- **Enum discriminant handling for C-style vs data-carrying enums** — `crates/wukong_mir_build/src/lib.rs:15865-15888` — Loads the i32 discriminant from offset 0 of a buffer scrutinee but compares a C-style scalar scrutinee directly; disc resolution at `crates/wukong_mir_build/src/lib.rs:15789-15802`.
- **Unary lowering with pointer-to-aggregate deref rule** — `crates/wukong_mir_build/src/lib.rs:16033-16068` — `*p` on an aggregate pointee yields the pointer itself rather than an invalid `load [N x i8]` that hung the native JIT or panicked mem2reg.
- **Binary lowering, comparison type join and half-result rounding** — `crates/wukong_mir_build/src/lib.rs:16070-16110` — Coerces both compare operands to a numeric join and both arithmetic operands to the result type, then re-rounds bf16/f16 results via `FpTrunc`; helpers at `crates/wukong_mir_build/src/lib.rs:16200-16227`.
- **Short-circuit `&&`/`||` as a real CFG diamond** — `crates/wukong_mir_build/src/lib.rs:16117-16147` — Lowers to cond_br plus an i1 merge param instead of a bitwise and/or, so an unsafe or side-effecting RHS never runs when the LHS decides the result.
- **Monomorphized call-target resolution** — `crates/wukong_mir_build/src/lib.rs:16234-16257` — Binds generic params against argument types, canonicalizes the type-arg key, and redirects a call to its monomorphic instance plus concrete return type.
- **lower_call dispatch chain** — `crates/wukong_mir_build/src/lib.rs:16392-16588` — Ordered resolution of enum ctors, slice `len`, user fns (sret/void/scalar), heap, file-I/O, now_ns, sdpa, math builtins, then print/assert intrinsics with string/unsigned/bool routing.
- **Heap builtins alloc_<T>/free** — `crates/wukong_mir_build/src/lib.rs:16601-16670` — Lowers typed allocation to `wukong_rt_alloc(count, size, is_float)` with a MIR negative-length clamp wrapped in a fresh 16-byte slice fat pointer; `free` unwraps it.
- **Typed file-I/O intrinsic family** — `crates/wukong_mir_build/src/lib.rs:16682-16719` — Maps `read_/write_<T>` to twelve runtime symbols passing `(path, data, len)` from a slice fat pointer; the doc comment lists only four types while code covers six.
- **now_ns clock and ABI coercion helper** — `crates/wukong_mir_build/src/lib.rs:16729-16773` — Zero-arg monotonic nanosecond intrinsic plus `lower_coerced`, the shared lower-then-coerce used by every kernel call that must pin an argument's MIR type.
- **Math-intrinsic lowering dispatch** — `crates/wukong_mir_build/src/lib.rs:16780-17088` — Fifty-plus builtins routed to primitive MIR or hand-expanded emitters, with an integer-argument short circuit for abs/rounding and float coercion via `lower_math_arg`; extra sites `crates/wukong_mir_build/src/lib.rs:17095-17129`.
- **Forward activation emitter family** — `crates/wukong_mir_build/src/lib.rs:17152-17215` — sigmoid/tanh/silu/gelu plus elu, leaky_relu, softplus, mish, selu, tanhshrink, hardsigmoid, hardswish, softsign, logsigmoid, each mirroring its AVX2 kernel op-for-op; extra sites `crates/wukong_mir_build/src/lib.rs:17299-17453`.
- **Activation backward emitter family** — `crates/wukong_mir_build/src/lib.rs:17221-17297` — Six `dy·act'(x)` gradients (silu, gelu, sigmoid, tanh, elu, softplus) written to match the runtime `*_bwd8` FMA order so inlined equals dispatched.
- **erf via Abramowitz–Stegun** — `crates/wukong_mir_build/src/lib.rs:17458-17527` — Compare/select abs and odd-sign handling, Horner FMA polynomial in t, and a reused exp(−x²), enabling exact (erf-based) GELU without IEEE bit surgery.
- **Composed transcendental family** — `crates/wukong_mir_build/src/lib.rs:17532-17649` — sinh/cosh, asinh/acosh/atanh, and Kahan-corrected expm1/log1p built from shared exp/log; cbrt, atan2 (NaN at origin unlike libm) and overflow-safe hypot at `crates/wukong_mir_build/src/lib.rs:17883-17963`.
- **Cephes sin/cos polynomial core** — `crates/wukong_mir_build/src/lib.rs:17651-17761` — Quadrant reduction by magic rounding, three-part π/2 split, separate sin/cos minimax polys, and a float-mask quadrant blend that enables RoPE.
- **Cephes atan core and derived inverse trig** — `crates/wukong_mir_build/src/lib.rs:17765-17848` — Two breakpoint masks with blended reduced candidates, odd degree-3 FMA poly, offset and sign restore; tan/asin/acos compose on it at `crates/wukong_mir_build/src/lib.rs:17852-17878`.
- **exp core: 8-bucket table reduction** — `crates/wukong_mir_build/src/lib.rs:17981-18088` — Magic-add rounding, integer bias extraction, Cody-Waite remainder, a three-level select tree replacing vpermps, and shift-free 2^e reconstruction mirroring the runtime kernel value-for-value; extra sites `crates/wukong_mir_build/src/lib.rs:17134-17150`, `crates/wukong_mir_build/src/lib.rs:18231-18258`.
- **log core: 8-bucket table reduction** — `crates/wukong_mir_build/src/lib.rs:18125-18228` — Bit-pattern split straddling 1.0, exact k·2^23 conversion, select-tree R/L lookups, Estrin degree-5 tail, and hi/lo ln2 reconstruction; f64 demotes/promotes at `crates/wukong_mir_build/src/lib.rs:18093-18109`.
- **Cast lowering semantics** — `crates/wukong_mir_build/src/lib.rs:18260-18300` — Numeric-to-bool becomes `!= 0` truthiness rather than low-bit truncation, and float→int signedness is taken from the target type to stop an interpreter/native divergence.
- **AST-to-MIR operator mapping tables** — `crates/wukong_mir_build/src/lib.rs:18419-18531` — Arithmetic, compound-assign and comparison operator selection by float/signed-ness; the cast-kind and numeric-join tables live at `crates/wukong_mir_build/src/lib.rs:24625-24701`.
- **Math-intrinsic name table and mask/lane type helpers** — `crates/wukong_mir_build/src/lib.rs:24707-24861` — The MathIntrinsic enum, its string lookup, rounding-mode mapping, reduction-op tag, and the compare-mask/float-lane type constructors used by every emitter.
- **Transcendental constant tables** — `crates/wukong_mir_build/src/lib.rs:24863-24977` — exp/log bucket tables, Cody-Waite ln2 splits, erf, Cephes sin/cos and atan coefficients written as f64 decimals that round to the runtime kernel's exact f32 bits.

## G09 — Optimizer: SSA construction and scalar passes

*Files:* `crates/wukong_opt/src/cache.rs`, `crates/wukong_opt/src/cfg.rs`, `crates/wukong_opt/src/cse.rs`, `crates/wukong_opt/src/dce.rs`, `crates/wukong_opt/src/dom.rs`, `crates/wukong_opt/src/dse.rs`, `crates/wukong_opt/src/fxhash.rs`, `crates/wukong_opt/src/inline.rs`, `crates/wukong_opt/src/lib.rs`, `crates/wukong_opt/src/licm.rs`, `crates/wukong_opt/src/mem2reg.rs`, `crates/wukong_opt/src/phi.rs`, `crates/wukong_opt/src/simplify.rs`, `crates/wukong_opt/src/simplify_cfg.rs`

**`crates/wukong_opt/src/cache.rs`**

- **CfgAnalyses lazy per-function analysis cache** — `crates/wukong_opt/src/cache.rs:26-119` — Computes preds/rpo/idom/df/dom-children once each and reuses them across every pass and fixpoint iteration until `invalidate`, with paired accessors (`idoms_and_preds`, `df_and_children`) for passes needing two at once.
- **`all_reachable` prune-skip fast path** — `crates/wukong_opt/src/cache.rs:86-93` — Compares cached RPO length to block count so cse/licm/mem2reg skip a reachability DFS in the steady state; consumers at `crates/wukong_opt/src/cse.rs:33-36`, `crates/wukong_opt/src/licm.rs:31-34`, `crates/wukong_opt/src/mem2reg.rs:39-43`.
**`crates/wukong_opt/src/cfg.rs`**

- **CFG primitives: successors, predecessors, RPO, reachability** — `crates/wukong_opt/src/cfg.rs:13-70` — Successors dedupe a `cond_br` with identical arms into one edge; predecessors are unique in source order; RPO/reachability use a recursive DFS that can overflow the stack on very deep CFGs.
- **Unreachable-block pruning with compaction and renumbering** — `crates/wukong_opt/src/cfg.rs:75-116` — Removes entry-unreachable blocks, remaps every terminator target and the function entry to compacted ids via a `BTreeMap`, returning whether anything moved; prerequisite for all dominance analysis.
**`crates/wukong_opt/src/cse.rs`**

- **CSE pass driver and global rewrite application** — `crates/wukong_opt/src/cse.rs:25-70` — Prunes unreachable blocks, collects all alloca ids once, numbers values over the dominator tree, then rewrites every op and terminator use through the resolved map.
- **Allocation-free pure-op value-numbering key** — `crates/wukong_opt/src/cse.rs:82-109` — A hashable `Key` enum replacing the old `format!` string, keying floats by raw bits and sub-kinds by `as u8` discriminant, built by `pure_key` at `crates/wukong_opt/src/cse.rs:189-215`.
- **Dominator-scoped value numbering with block-exit scope teardown** — `crates/wukong_opt/src/cse.rs:111-174` — Adds pure-op keys on block entry and removes exactly the keys that block introduced when leaving its dominator subtree, so reuse is legal without per-candidate dominance checks.
- **Intra-block load forwarding with conservative alias invalidation** — `crates/wukong_opt/src/cse.rs:118-146` — Tracks the current value per alloca slot within one block; a store through an unknown pointer or any call/vector-kernel call clears all slots, and allocas themselves are never value-numbered.
**`crates/wukong_opt/src/dce.rs`**

- **Dead-code elimination liveness sweep** — `crates/wukong_opt/src/dce.rs:11-42` — One non-iterative pass building a used-value set then dropping result-producing instructions that are neither used nor side-effecting; never removes dead block parameters or dead blocks.
**`crates/wukong_opt/src/dom.rs`**

- **Cooper–Harvey–Kennedy immediate dominators** — `crates/wukong_opt/src/dom.rs:16-67` — Iterative idom fixpoint over a precomputed RPO/pred map with an `intersect` walk up the dom tree, skipping unreachable or not-yet-processed predecessors.
- **Dominance frontiers and dominator-tree children** — `crates/wukong_opt/src/dom.rs:72-108` — Frontier computation via the runner-up-the-idom-chain rule (only for join blocks with >=2 preds) plus child lists, which together drive mem2reg phi placement and CSE's dominator walk.
**`crates/wukong_opt/src/dse.rs`**

- **Intra-block dead-store elimination** — `crates/wukong_opt/src/dse.rs:19-81` — Marks a store dead when a later store to the same alloca overwrites it unread; any load of the slot, unknown-pointer access, or call clears pending stores, and block-exit survivors are kept for successors.
**`crates/wukong_opt/src/fxhash.rs`**

- **FxHash hasher for optimizer maps** — `crates/wukong_opt/src/fxhash.rs:17-86` — Dependency-free rotate-xor-multiply hasher plus `FxHashMap`/`FxHashSet` aliases, chosen because every key is a small program-derived integer; deterministic iteration order is relied on for byte-identical MIR output.
**`crates/wukong_opt/src/inline.rs`**

- **Leaf-only inlining policy with size, growth and dead-callee rules** — `crates/wukong_opt/src/inline.rs:26-104` — Inlines only callees that call no user function and have at most 40 instructions, stops a caller at 5000 instructions, and drops inlined callees that are no longer called anywhere.
- **Call-site search and repeated splice loop** — `crates/wukong_opt/src/inline.rs:106-121` — Rescans the whole caller from block 0 for the first inlinable call after every splice, giving quadratic behaviour on call-heavy functions while snapshotted callee bodies prevent cascade inlining.
- **SSA call-site splice with continuation block** — `crates/wukong_opt/src/inline.rs:123-192` — Remaps every callee value and block to fresh caller ids, splits the call block so the tail becomes a continuation taking the result as a parameter, and turns each `ret` into a branch via `crates/wukong_opt/src/inline.rs:194-229`.
- **Inlining ignores function-local vec_kernels tables** — `crates/wukong_opt/src/inline.rs:169-190` — Callee `Op::VecKernelCall` instructions are copied with their `kernel` index unchanged even though that index addresses the callee's own `Function::vec_kernels`, which inlining never merges into the caller, see `crates/wukong_mir/src/lib.rs:148-153`.
**`crates/wukong_opt/src/lib.rs`**

- **Pass trait and analysis-cache contract** — `crates/wukong_opt/src/lib.rs:42-51` — Defines the function-level transform interface: `run_function(&mut Function, &mut CfgAnalyses) -> bool`, where returning true means mutated and CFG-restructuring passes must invalidate the cache.
- **Standard pipeline and opt-level gating** — `crates/wukong_opt/src/lib.rs:111-132` — `-O0` empty, `-O1` adds mem2reg/simplify/simplify-cfg/simplify-phis/dce, `-O2` appends cse/dse/licm; `optimize` also runs whole-program inlining at `-O2`+ at `crates/wukong_opt/src/lib.rs:212-219`.
- **Fixpoint driver with per-pass clean-tracking** — `crates/wukong_opt/src/lib.rs:143-203` — Runs passes repeatedly, skipping passes already reporting no-change and re-dirtying all on any mutation, capped at 100 iterations; claims bit-identical MIR to a naive fixpoint.
- **Per-pass MIR verify-each debug gate** — `crates/wukong_opt/src/lib.rs:182-192` — Under `debug_assertions` every executed pass is followed by `verify_function`, asserting well-formedness and naming the culprit pass; compiled out of release so release bugs escape.
- **Shared use-visiting and side-effect helpers** — `crates/wukong_opt/src/lib.rs:244-391` — `map_op_uses`/`map_term_uses`/`each_op_use`/`each_term_use`/`has_side_effects` are the single exhaustive match over every `Op`/`Terminator` operand, so a new MIR op must be added here or all passes silently miss it.
**`crates/wukong_opt/src/licm.rs`**

- **LICM pass driver** — `crates/wukong_opt/src/licm.rs:23-51` — Prunes unreachable blocks, skips single-block functions, and hoists per natural loop only when a usable preheader exists, using cached idoms and predecessors simultaneously.
- **Natural-loop discovery from back edges** — `crates/wukong_opt/src/licm.rs:53-88` — Finds back edges `n -> h` where `h` dominates `n`, grows the body by backward reachability, merges loops sharing a header, and sorts by header id for deterministic MIR (M12).
- **Idom-chain dominance test** — `crates/wukong_opt/src/licm.rs:90-103` — Walks the immediate-dominator chain upward from `b` to the entry self-loop, an O(depth) query reused for both back-edge detection and preheader validation.
- **Preheader eligibility rule** — `crates/wukong_opt/src/licm.rs:105-130` — Requires exactly one out-of-loop predecessor that branches unconditionally to the header and dominates it; no preheader is ever synthesized, so many loops are simply skipped.
- **Speculation whitelist for hoisting** — `crates/wukong_opt/src/licm.rs:132-155` — Permits only side-effect-free non-trapping ops (consts, cmp, cast, gep, select, fma, sqrt, round) and rejects loads, stores, calls, allocas and integer div/rem which can trap on zero.
- **Fixpoint invariant-hoisting rounds** — `crates/wukong_opt/src/licm.rs:157-231` — Alternates a read phase finding instructions whose operands are all loop-external or already hoisted with a mutate phase moving them, iterating to closure over block-id-sorted bodies for determinism.
**`crates/wukong_opt/src/mem2reg.rs`**

- **mem2reg pass driver and reachability precondition** — `crates/wukong_opt/src/mem2reg.rs:29-51` — Runs SSA promotion only after pruning unreachable blocks (dominance is undefined otherwise), invalidating the CFG analysis cache when pruning renumbered blocks.
- **Promotable-slot escape analysis** — `crates/wukong_opt/src/mem2reg.rs:53-107` — Accepts only scalar int/float allocas whose pointer appears solely as a load/store address; any gep, call arg, stored-value or terminator use disqualifies the slot.
- **Iterated dominance frontier phi placement** — `crates/wukong_opt/src/mem2reg.rs:127-148` — Places per-slot phis as fresh block parameters at the IDF of store-containing blocks using a worklist, recording `(phi value, slot)` order to match later edge arguments.
- **Deferred three-phase edit application** — `crates/wukong_opt/src/mem2reg.rs:176-253` — Collects edits during the read-only walk then applies edge phi-args, index-keyed load/store/alloca deletion, and use rewriting in an order that keeps original instruction indices valid.
- **Dominator-tree renaming walk with per-slot definition stacks** — `crates/wukong_opt/src/mem2reg.rs:255-369` — Recursively renames loads to reaching defs and stores to new defs, pushing/popping per-slot stacks; recursion is unbounded so very deep dominator trees risk stack overflow.
- **Read-before-write zero materialization** — `crates/wukong_opt/src/mem2reg.rs:277-293` — Mints one cached zero constant per MIR type when a slot is read with an empty def stack, matching the interpreter's zero-initialized memory instead of an undef value, materialized at `crates/wukong_opt/src/mem2reg.rs:219-242`.
- **Rewrite-chain resolution with cycle guards** — `crates/wukong_opt/src/mem2reg.rs:371-383` — Follows `load -> reaching def` (and CSE's `value -> canonical`) chains to a fixed point with self-reference and 100k/10k iteration guards, sibling at `crates/wukong_opt/src/cse.rs:176-187`.
**`crates/wukong_opt/src/phi.rs`**

- **Dead and trivial block-parameter elimination** — `crates/wukong_opt/src/phi.rs:24-121` — Fixpoint removing unused block params and params whose incoming args (ignoring self-refs) are a single value, rewriting all uses and every incoming edge via `crates/wukong_opt/src/phi.rs:124-161`; entry params are exempt as they are the function signature.
**`crates/wukong_opt/src/simplify_cfg.rs`**

- **Constant branch folding** — `crates/wukong_opt/src/simplify_cfg.rs:45-87` — Turns a `cond_br` on a known `ConstInt` into a `br` to the taken side carrying that side's args, and also collapses a `cond_br` whose targets and arguments are identical.
- **Straight-line block merging** — `crates/wukong_opt/src/simplify_cfg.rs:92-136` — Repeatedly fuses `A: br B` into A when A is B's only predecessor, substituting B's params with A's edge args; enlarges blocks so the intra-block CSE/DSE passes see more, then prunes.
**`crates/wukong_opt/src/simplify.rs`**

- **Simplify pass driver: const table, substitution chains, borrow-avoiding classification** — `crates/wukong_opt/src/simplify.rs:31-167` — Single forward sweep building a value-to-constant map, rewriting operands through a union-find-ish `subst` chain (`crates/wukong_opt/src/simplify.rs:169-178`), copying only Copy operands into an `Act` enum to avoid cloning heap `Op` fields.
- **Float fold precision gating and bf16/f16 fold refusal** — `crates/wukong_opt/src/simplify.rs:193-216` — `round_float_to_ty` narrows every f32 fold step through `as f32` so folded chains match runtime f32, and bf16/f16 arithmetic is deliberately left unfolded because MIR consts carry unrounded literals.
- **Width-correct integer constant folding** — `crates/wukong_opt/src/simplify.rs:218-273` — Folds all integer binops in `i128` then re-masks to the MIR type, with shift counts masked to width and unsigned ops reading a zero-extended window; helpers `mask`/`int_bits`/`uval` at `crates/wukong_opt/src/simplify.rs:435-474`.
- **Comparison folding: width-aware and NaN-safe self-compare** — `crates/wukong_opt/src/simplify.rs:287-339` — `fold_cmp` folds at the *operand* type width so signed/unsigned predicates match the runtime register, and `fold_cmp_self` folds `x cmp x` only for integer predicates since NaN breaks the float ones.
- **Algebraic identity table** — `crates/wukong_opt/src/simplify.rs:341-433` — Peephole rules for x+0, x-0, x-x, x*1, x*0, x/1, x%1, x|0, x|x, x^0, x^x, x&0, x&x and shift-by-zero, emitting either a value replacement or a fresh zero constant.

# Kernel recognition

## G10 — Recognizer Infrastructure: nest matchers, affine analysis, dim canonicalization

*Files:* `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_mir_build/src/lib.rs`**

- **GemmSyms runtime-symbol table** — `crates/wukong_mir_build/src/lib.rs:1722-2139` — A single `Copy` struct of ~180 pre-interned runtime entry-point symbols threaded through every lowerer, the central registry every kernel recognizer dispatches against.
- **Window primitives: `as_range0_for` / `let_init`** — `crates/wukong_mir_build/src/lib.rs:3422-3445` — Pure shape matchers for `for v in 0..N` (literal-0 start, exclusive, no step) and `let name = init`, the entry gate of every fused-run recognizer; also `crates/wukong_mir_build/src/lib.rs:3448-3461`.
- **Reduction-seed soundness gates** — `crates/wukong_mir_build/src/lib.rs:3766-3807` — Accepts only seeds provably neutral (`x[0]`, batched `x[r*C]` with a `+0` peel, or `<=-1e30`); the cummax/cummin twin generalizes to `>=1e30` but omits the `+0` peel; also `crates/wukong_mir_build/src/lib.rs:5658-5692`.
- **Scalar-binding matchers: reciprocal, log, log-sum-exp** — `crates/wukong_mir_build/src/lib.rs:3810-3826` — Recognize `let inv = 1.0/s`, `let ls = log(s)` and `m + log(s)` (either order), the scalar glue chaining reduction passes to normalize passes; also `crates/wukong_mir_build/src/lib.rs:4238-4251`, `crates/wukong_mir_build/src/lib.rs:4339-4363`.
- **Kernel operand base-pointer resolution (kernel_base_ptr)** — `crates/wukong_mir_build/src/lib.rs:7253-7269` — Loads through Ptr slots and `[]T` fat pointers (detected by the i8-array SLICE_SIZE shape) but passes fixed-array slots straight through, keeping old MIR byte-identical.
- **Slice-blind operand lookup in most kernel emitters** — `crates/wukong_mir_build/src/lib.rs:7466-7472` — Only emit_norm/emit_sgemm/emit_gemv use kernel_base_ptr; every other emitter uses raw `lookup`, so a `[]f32` slice operand would pass the fat-pointer buffer, also `crates/wukong_mir_build/src/lib.rs:7607-7613`.
- **Batched-matmul base-offset GEP helper (`offset_base`)** — `crates/wukong_mir_build/src/lib.rs:8321-8350` — Lowers and i64-coerces each batch/head offset term, sums them, and GEPs the base pointer by f32 elements, returning base unchanged when no offset exists.
- **Kernel operand offset and dimension materialization (offset_base, dim_value)** — `crates/wukong_mir_build/src/lib.rs:8325-8366` — Sums batch/head index terms into one f32-strided GEP and materializes literal or runtime dimensions as i64, both bailing cleanly when out of scope.
- **Kernel operand base-pointer resolution rule (`kernel_base_ptr` vs raw `lookup`)** — `crates/wukong_mir_build/src/lib.rs:8745-8751` — Reduction paths resolve array operands through `kernel_base_ptr` to fix the 1-D-tensor slot-address divergence, while the argreduce/axpby/vmath-narrow emitters here still pass `self.lookup(..).0` directly, at `crates/wukong_mir_build/src/lib.rs:9393-9399`, `crates/wukong_mir_build/src/lib.rs:8635-8637`, `crates/wukong_mir_build/src/lib.rs:9135-9139`, `crates/wukong_mir_build/src/lib.rs:9337-9339`.
- **Ordered recognizer cascade in `lower_for`** — `crates/wukong_mir_build/src/lib.rs:9557-9853` — A ~30-probe ordered chain (matmul, residual epilogue, i8/lowp GEMM, GEMV, GEVM, transpose, pool2d, col reductions, backwards, losses, RoPE, dequant, scans, embedding, scatter, reductions, axpby, bias-bcast, batched norm/vmath) whose ordering encodes disjointness claims that a stress-tester must re-verify.
- **Row-major batched index matcher** — `crates/wukong_mir_build/src/lib.rs:10823-10873` — `index_off`/`index_by_loopvar`/`is_mul_of` recognize `base[j]` or `base[row*cols + j]` in either factor order, the shared shape all vmath recognizers need.
- **Loop-invariant coefficient and affine term classifier (`velem_coeff`/`velem_term`)** — `crates/wukong_mir_build/src/lib.rs:11597-11641` — Splits an additive term into scaled array read or invariant f32 bias, using conservative `expr_mentions` so per-element factors are rejected.
- **Coefficient lowering with kernel defaults (`lower_coeff`)** — `crates/wukong_mir_build/src/lib.rs:11810-11822` — Lowers an optional coefficient expr to f32 or synthesizes a default constant, the single policy point for absent `a`/`b`/`c` scalars.
- **Conservative symbol-mention and structural-equality predicates** — `crates/wukong_mir_build/src/lib.rs:18536-18654` — Escape analysis that returns true for unmodeled expression kinds so fusion declines safely, plus the elementwise-subset structural equality at `crates/wukong_mir_build/src/lib.rs:18719-18769`.
- **Dim canonicalization and 2-D index parsing** — `crates/wukong_mir_build/src/lib.rs:18961-19025` — `Dim::Lit`/`Dim::Var` compares strides symbolically so runtime-sized matmuls still dispatch; `mul_with_row`/`match_row_col` disambiguate `row*stride + col` using the known row.
- **Index-term flattening and operand primitives** — `crates/wukong_mir_build/src/lib.rs:19343-19438` — Additive/multiplicative flatteners, literal-multiplication distribution, `match_row_col_off` with leftover base offsets, plus single-index/zero-literal/f32 predicates; multiplicative twin at `crates/wukong_mir_build/src/lib.rs:19879-19902`.
- **Shape-typed operand matchers** — `crates/wukong_mir_build/src/lib.rs:19535-19609` — Derives a rank-2 contiguous tensor's inner stride so idiomatic `a[i,k]` two-index spellings dispatch to the same GEMM kernel as the flat `a[i*K+k]` form.
- **Loop-shape primitives for fusion** — `crates/wukong_mir_build/src/lib.rs:24521-24581` — Block-tail and branch-value extraction plus the half-open unit-step `for` and range-bound accessors every recognizer builds on.

## G11 — Recognizers: GEMM / matmul / GEMV family and epilogue fusion

*Files:* `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Batched-matmul offset and fused linear-epilogue value gates** — `crates/wukong_codegen_cranelift/src/tests.rs:2916-2953` — Hand-computes multi-head `Q·Kᵀ` outputs and the `act(x·Wᵀ+b)` epilogue results (relu/identity/gelu/silu, bias-free SwiGLU) because both backends run the same fused kernel, extra site `crates/wukong_codegen_cranelift/src/tests.rs:2961-3005`.
- **GEMM value gates against independent Rust references (beta=0 / beta=1)** — `crates/wukong_codegen_cranelift/src/tests.rs:3686-3736` — Recomputes the matmul in Rust for sizes hitting MR/NR remainders so a beta misclassification or dropped C is caught where a shared-kernel differential would be blind, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:4044-4063`, `crates/wukong_codegen_cranelift/src/tests.rs:4257-4302`.
- **GEMM recognizer dispatch gates (sgemm / sgemm_nt / _parallel + negatives)** — `crates/wukong_codegen_cranelift/src/tests.rs:3885-3924` — Proves the ikj-accumulate, ijk-dot, zero-init and shape-typed `Tensor[f32,N,N]` spellings all reach the tuned kernels while a wrong B stride never does, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:3744-3810`, `crates/wukong_codegen_cranelift/src/tests.rs:3966-4040`, `crates/wukong_codegen_cranelift/src/tests.rs:3932-3962`.
- **GEMV / GEVM dispatch, negative-case and decode-attention gates** — `crates/wukong_codegen_cranelift/src/tests.rs:4449-4547` — Pins α-scaled `wukong_sgemv_alpha`, `wukong_sgevm_f32` and their parallel twins, rejects additive stores, indexed scales, outer-indexed weights and aliased outputs, and asserts a full KV-decode body dispatches all three stages, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:4556-4663`, `crates/wukong_codegen_cranelift/src/tests.rs:4668-4698`.
**`crates/wukong_mir_build/src/lib.rs`**

- **f32 GEMM/GEMV/GEVM dispatch symbols** — `crates/wukong_mir_build/src/lib.rs:1738-1787` — NN/NT/TN matmuls, fused-epilogue Linear, α-scaled NT, GEMV, α-scaled GEMV and vector·matrix, each with a bit-identical `_parallel` twin, also `crates/wukong_mir_build/src/lib.rs:1727-1728`.
- **bf16/f16 mixed-precision GEMM symbols** — `crates/wukong_mir_build/src/lib.rs:1884-1912` — Half-input NT, fused-epilogue NT and TN weight-gradient GEMMs that losslessly widen then reuse the tuned f32 kernels, keeping bit-for-bit equality with the widened f32 GEMM.
- **Matmul epilogue fusion probes at statement level** — `crates/wukong_mir_build/src/lib.rs:3259-3277` — Tries f32 GEMM+bias/ReLU, bf16/f16 low-precision GEMM+activation, and int8 GEMM+per-channel-dequant fusions in that order, each folding a matmul plus its follow-up loop into one call.
- **f32 GEMM dispatch emitter (emit_sgemm)** — `crates/wukong_mir_build/src/lib.rs:7271-7377` — Selects among NN/NT/TN serial and parallel kernels, applies batch base GEPs, and routes fused bias or alpha-scaled forms, declining unsupported shapes to the scalar nest.
- **Alpha scale materialization (alpha_value)** — `crates/wukong_mir_build/src/lib.rs:7384-7402` — Turns a peeled alpha into an f32 value from a literal, an in-scope local load, or a top-level `const`, with locals shadowing consts.
- **GEMV and GEVM emitters (emit_gemv, emit_gevm)** — `crates/wukong_mir_build/src/lib.rs:7411-7489` — Emit matrix-vector and vector-matrix kernels with optional alpha; GEVM always passes alpha (defaulting 1.0) and parallelizes by output-column stripes to avoid reassociation.
- **f32 Linear+epilogue fusion (try_fuse_matmul_epilogue, emit_sgemm_epi)** — `crates/wukong_mir_build/src/lib.rs:8138-8213` — Fuses `C = act(A·Bᵀ + bias)` into one writeback for plain 2-D nests only; a comment calls the epilogue kernel serial though emit picks nt_epi_par under @parallel.
- **MatmulNest plan and recognizer dispatcher** — `crates/wukong_mir_build/src/lib.rs:18781-18817` — Carries operands, dims, beta, both transpose flags, per-operand batch offsets, alpha and fused bias; `recognize_matmul` tries ikj, ijk, memacc at `crates/wukong_mir_build/src/lib.rs:19690-19700`.
- **Matmul store-shape peels: alpha scale and fused bias** — `crates/wukong_mir_build/src/lib.rs:18819-18876` — Extracts a loop-invariant f32 scale (literal or symbol) or a per-column `bias[offset+j]` from the store so the GEMM writeback absorbs it; `crates/wukong_mir_build/src/lib.rs:18884-18932`.
- **Matmul product and zero-init matchers** — `crates/wukong_mir_build/src/lib.rs:19442-19527` — Classifies A/B factors in either order with normal or transposed layouts and optional batch offsets, and verifies the per-row zero-init that makes beta zero; `crates/wukong_mir_build/src/lib.rs:19615-19687`.
- **Bias/activation epilogue matcher** — `crates/wukong_mir_build/src/lib.rs:19704-19873` — Matches a following `C = act(C [+ bias[j]])` double loop against the matmul's dims and output, encoding identity/relu/gelu/silu codes; identity requires a bias, activations do not.
- **int8 quantized nn.Linear nest** — `crates/wukong_mir_build/src/lib.rs:20079-20287` — Matches the u8×i8→i32 NT dot-product nest, enforcing operand signedness because the kernel zero-extends A and sign-extends B, and rejecting output aliasing.
- **ijk dot-product matmul recognizers** — `crates/wukong_mir_build/src/lib.rs:20523-20683` — The `let s` scalar-accumulator form carrying batch offsets, alpha and fused bias, plus the memory-accumulator variant that would otherwise fall through to fully scalar code at `crates/wukong_mir_build/src/lib.rs:20703-20832`.
- **Fused residual projection recognizer** — `crates/wukong_mir_build/src/lib.rs:20842-21032` — Detects a store that reads its own output back, mapping the transformer skip connection to the beta=1 fused-epilogue kernel; whole-function probe at `crates/wukong_mir_build/src/lib.rs:24361-24375`.
- **ikj accumulate matmul recognizer** — `crates/wukong_mir_build/src/lib.rs:21035-21192` — The optionally-hoisted `let aik` form with beta chosen by the presence of a zero-init loop; offsets, alpha and transposed A are structurally unreachable here.
- **GEMV and GEVM recognizers** — `crates/wukong_mir_build/src/lib.rs:21196-21350` — Row-major `y[i] = Σ a[i*N+j]·x[j]` with optional alpha, and its structurally disjoint strided transpose `out[j] = Σ w[i]·a[i*N+j]` at `crates/wukong_mir_build/src/lib.rs:21355-21512`.
- **Whole-function matmul lowering wrappers** — `crates/wukong_mir_build/src/lib.rs:21515-21603` — Rebuilds a recognized matmul function as a thin kernel tail-call binding hidden dim params first; int8 and low-precision probes at `crates/wukong_mir_build/src/lib.rs:21607-21643` and `crates/wukong_mir_build/src/lib.rs:24381-24439`.

## G12 — Recognizers: elementwise, vmath, velem, horner, dequant

*Files:* `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Transcendental / erf / trig expansion differential** — `crates/wukong_codegen_cranelift/src/tests.rs:2045-2089` — Exercises exp clamps, wide-magnitude log, pow, erf odd symmetry and quadrant-reduced sin/cos in scalar and vectorized form, plus the two-array sigmoid-gate VM2 path, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:2846-2873`, `crates/wukong_codegen_cranelift/src/tests.rs:2878-2908`.
- **vmath activation dispatch with libm golden accuracy** — `crates/wukong_codegen_cranelift/src/tests.rs:2096-2160` — Beyond native==interp (both call the same kernel) it pins exp/tanh/sigmoid/silu/gelu to fixed ~1-ULP golden digits, covering non-multiple-of-8 tails and the fused adjacent-loop case, extra site `crates/wukong_codegen_cranelift/src/tests.rs:2166-2179`.
- **Streaming velem and bias-broadcast dispatch gates** — `crates/wukong_codegen_cranelift/src/tests.rs:3048-3081` — Requires saxpy, ReLU, nested-if ReLU6 and `x[i*C+j]+b[j]` to lower to a single `wukong_velem_f32`/`wukong_bias_bcast_f32` call and match closed-form references across vector-body and tail sizes, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:3418-3446`, `crates/wukong_codegen_cranelift/src/tests.rs:3652-3680`, `crates/wukong_codegen_cranelift/src/tests.rs:3088-3149`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Broadcast-bias @parallel probe** — `crates/wukong_mir_build/src/lib.rs:1326-1370` — Same throwaway-lowerer trick for `match_bias_bcast`, preserving the two-level `out[i*C+j] = act(x + b[j])` nest so the multicore bias kernel fires.
- **Elementwise vmath/velem/horner/bias-broadcast symbols** — `crates/wukong_mir_build/src/lib.rs:1788-1827` — 256-bit AVX2 transcendental (f32/bf16/f16, one- and two-input), streaming affine+activation, Horner polynomial and broadcast-bias kernels with parallel twins.
- **Mirrored runtime op-code constant space** — `crates/wukong_mir_build/src/lib.rs:2143-2218` — Hand-duplicated `VM_*`/`VM2_*`/`VE_*`/`DQ_*`/bias-activation codes with no compile-time cross-check against the runtime crate (note the gap at 4, filled later by `VMATH_RELU`), also `crates/wukong_mir_build/src/lib.rs:2271-2291`.
- **Recognizer plan structs for streaming maps and dequant** — `crates/wukong_mir_build/src/lib.rs:2220-2269` — `VTerm`/`VElemPlan`/`DequantPlan`/`DequantPerchanNest` carry operand *symbols* plus borrowed loop-invariant coefficient exprs, deferring base-pointer resolution to avoid the tensor-param GEP-off-slot bug.
- **Shared low-precision axpby term matcher (`match_lowp_axpby_sum`)** — `crates/wukong_mir_build/src/lib.rs:8942-9007` — Matches `a*(x[k] as f32) + b*(y[k] as f32)` with either coefficient implicit and either factor order, rejecting coefficients that use `k` and mixed bf16/f16 input precisions.
- **f32-output low-precision axpby recognizer (`match_lowp_axpby`)** — `crates/wukong_mir_build/src/lib.rs:9009-9045` — Requires a single-statement body storing to an f32 `out[k]` and delegates the RHS to the shared sum matcher; note its intended doc-comment and `#[allow(type_complexity)]` at 8931-8941 are misattached to `match_lowp_axpby_sum`.
- **Half-output axpby recognizer (`match_lowp_axpby_narrow`)** — `crates/wukong_mir_build/src/lib.rs:9047-9105` — Matches `out[k] = (<axpby sum>) as bf16/f16` demanding the store target, the cast type and both input precisions all be the same half width so one narrowing kernel applies.
- **bf16/f16-in f32-out axpby emitter (`try_emit_bf16_axpby`)** — `crates/wukong_mir_build/src/lib.rs:9107-9167` — Emits `wukong_axpby_{bf16,f16}(x,y,out,n,a,b)` for `0..n` loops, materializing implicit coefficients as literal 1.0 and coercing explicit ones to f32.
- **Half-in/half-out axpby emitter (`try_emit_lowp_axpby_narrow`)** — `crates/wukong_mir_build/src/lib.rs:9169-9230` — Same shape as the f32-output emitter but calls `axpby_{bf16,f16}_out`, halving write traffic; probed after the f32-output form on the claim the target shapes are disjoint.
- **Half-output activation recognizer (`match_vmath_narrow`)** — `crates/wukong_mir_build/src/lib.rs:9232-9305` — Matches `out[k] = (f((x[k] as f32))) as bf16/f16`, resolving `f` via `vmath_opcode_of`, requiring a unary f32-typed call and identical half widths on input and output.
- **Half-output activation emitter (`try_emit_lowp_vmath_narrow`)** — `crates/wukong_mir_build/src/lib.rs:9307-9355` — Emits `wukong_vmath_{bf16,f16}_out(x,out,n,op)` with the opcode as an i64 constant, the KV-cache/activation-store path at 4 bytes per element.
- **vmath intrinsic opcode table** — `crates/wukong_mir_build/src/lib.rs:10879-10936` — Maps exactly 35 unary math intrinsics to `VMATH_*` kernel codes, shared by the f32 and half-output recognizers so they cannot drift.
- **Unary activation statement recognizer** — `crates/wukong_mir_build/src/lib.rs:10940-11010` — Matches `out[j] = f(x[j])` over f32, folds the product form `x*sigmoid(x)` into `VMATH_SILU`, and accepts bf16/f16 inputs widened by cast; also `crates/wukong_mir_build/src/lib.rs:11016-11036`.
- **Whole-body vmath match** — `crates/wukong_mir_build/src/lib.rs:11042-11061` — Requires every body statement to be an independent recognized activation with both operand symbols in scope, emitting no MIR so it is safe to probe speculatively.
- **vmath kernel call emission** — `crates/wukong_mir_build/src/lib.rs:11066-11116` — Emits one full-range kernel call per statement in source order, selecting bf16/f16/serial/multicore variants and GEP-ing each base by the start index.
- **Flat and batched activation loop dispatch** — `crates/wukong_mir_build/src/lib.rs:11124-11137` — `try_vmath_for` handles `for j in lo..hi`; `try_emit_batched_vmath` collapses a zero-based `[R,C]` nest into one flat `[0,R*C)` call; also `crates/wukong_mir_build/src/lib.rs:11151-11219`.
- **Batched vmath nest recognizer (`try_emit_batched_vmath`)** — `crates/wukong_mir_build/src/lib.rs:11151-11219` — Matches `for r in 0..R { for j in 0..C { out[r*C+j]=f(x[r*C+j]) } }` and emits one flat `[0,R*C)` vmath call, assuming row-major contiguity.
- **Bias-broadcast activation opcode table (`bias_activation_code`)** — `crates/wukong_mir_build/src/lib.rs:11221-11241` — Maps twelve transcendental/saturating `MathIntrinsic`s to runtime `VMATH_*` codes, deliberately excluding trig/log as implausible post-bias activations.
- **Bias activation peeler (`peel_bias_act`)** — `crates/wukong_mir_build/src/lib.rs:11243-11272` — Strips bare-add, `f(add)` call, or if-form ReLU off a bias value, remapping the velem `VE_RELU` code into the bias kernel's `VMATH_RELU`.
- **Broadcast-bias nest matcher (`match_bias_bcast_inner`/`match_bias_bcast`)** — `crates/wukong_mir_build/src/lib.rs:11274-11391` — Pure matcher for `out[i*C+j]=act(x[i*C+j]+b[j])`, relying on `index_off` and `index_by_loopvar` being disjoint to disambiguate operand order.
- **Bias-broadcast dispatch (`try_emit_bias_bcast`)** — `crates/wukong_mir_build/src/lib.rs:11393-11428` — Lowers a recognized bias nest to `wukong_bias_bcast_f32[_par]`, covering the row-invariant bias the affine velem matcher declines.
- **Gated-FFN (SwiGLU/GeGLU/GLU) recognizer** — `crates/wukong_mir_build/src/lib.rs:11443-11479` — Detects `out[j]=act(a[j])*b[j]` in either factor order and emits `VMATH2_*_GATE`, always passing the activated operand first as the kernel requires.
- **Two-argument transcendental and activation-backward opcodes** — `crates/wukong_mir_build/src/lib.rs:11480-11511` — Maps `pow`/`atan2`/`hypot` plus six `*_backward` intrinsics to `VMATH2_*`, requiring both args and result f32 unit-stride reads.
- **vmath2 loop recognizer and emitter (`match_vmath2_body`/`emit_vmath2_calls`/`try_vmath2_for`)** — `crates/wukong_mir_build/src/lib.rs:11513-11595` — Accepts a multi-statement body of independent two-input maps and emits one GEP'd `wukong_vmath2_f32` call per statement over `[s,e)`.
- **Streaming affine map matcher (`match_velem_affine`)** — `crates/wukong_mir_build/src/lib.rs:11643-11694` — Flattens the RHS add-tree into at most two array reads plus one bias, building a `VElemPlan` covering saxpy/scale/residual/bias/copy.
- **Elementwise binary map matcher (`match_velem_binary`)** — `crates/wukong_mir_build/src/lib.rs:11696-11742` — Recognizes `out[j]=x[j]*y[j]` and `x[j]/y[j]` as `VE_HADAMARD`/`VE_DIV`, the both-operands-indexed forms the affine matcher refuses.
- **ReLU/ReLU6 activation peeler (`peel_velem_act`)** — `crates/wukong_mir_build/src/lib.rs:11744-11782` — Recursively unwraps `if INNER>0 {INNER} else {0}` and the ReLU6 clamp, requiring structural equality between guard operand and then-branch value.
- **velem body recognizer and loop driver (`match_velem_body`/`try_velem_for`)** — `crates/wukong_mir_build/src/lib.rs:11784-11808` — Tries affine then binary, bare then ReLU-peeled, then lowers loop bounds to i64 and emits the kernel call; also `crates/wukong_mir_build/src/lib.rs:11881-11898`.
- **velem call emission and parallel-variant rule (`emit_velem_call`)** — `crates/wukong_mir_build/src/lib.rs:11824-11879` — GEPs bases by `s`, reuses the `x` pointer for an unread `y`, defaults `b` to 1.0 only when `y` is present, and selects the rayon kernel inside mixed `@parallel` functions.
- **Dequant activation peeler (`peel_dequant_act`)** — `crates/wukong_mir_build/src/lib.rs:11902-11932` — Recognizes `fmax(x,0)` in either order as `DQ_RELU` plus `gelu`/`silu` calls, falling back to `DQ_ID` rather than failing.
- **Dequant product matcher (`match_dequant_mul`)** — `crates/wukong_mir_build/src/lib.rs:11957-11984` — Matches `(q[j] as f32)*scale` in either factor order, reusing `velem_coeff` to require the scale be loop-invariant f32.
- **Streaming dequant recognizer (`match_dequant_body`/`try_dequant_for`)** — `crates/wukong_mir_build/src/lib.rs:11986-12016` — Builds a `DequantPlan` from a single-statement int-input map, disjoint from velem by construction; loop driver at `crates/wukong_mir_build/src/lib.rs:12061-12077`.
- **Dequant call emission with per-width GEP stride (`emit_dequant_call`)** — `crates/wukong_mir_build/src/lib.rs:12018-12059` — Strides `q` by its source element type and `out` by f32; the doc comment claims the serial kernel is always used yet the code selects `dequant_par` when `parallel_fn`.
- **Per-channel dequant nest matcher (`match_dequant_perchan`)** — `crates/wukong_mir_build/src/lib.rs:12104-12190` — Matches `out[i*C+j]=act((q[i*C+j] as f32)*scale[j])`, resolving rows/cols through `as_dim` rather than lowered exprs, so only dim-expressible bounds qualify.
- **Per-channel dequant emission (`emit_dequant_perchan`)** — `crates/wukong_mir_build/src/lib.rs:12192-12223` — Emits one whole-matrix `wukong_dequant_perchan_f32[_parallel]` call; the plan's `q_elem` field is explicitly discarded via `let _`, making it dead state.
- **Horner step matcher (`match_horner_step`)** — `crates/wukong_mir_build/src/lib.rs:12225-12270` — Matches `r = r*v + Ck` in either multiply and add operand order, requiring `Ck` be loop-invariant f32 by `expr_mentions`.
- **Horner polynomial body recognizer (`match_vhorner_body`)** — `crates/wukong_mir_build/src/lib.rs:12272-12324` — Requires at least four statements shaped let-element, let-seed, N steps, store, collecting coefficients highest-degree first from body-local scalars.
- **Horner emission with stack coefficient array (`emit_vhorner`)** — `crates/wukong_mir_build/src/lib.rs:12326-12377` — Allocas an f32 array, stores each lowered coefficient, then calls `wukong_vhorner_f32` with GEP'd input/output pointers and the coefficient count.
- **Horner loop driver (`try_vhorner_for`)** — `crates/wukong_mir_build/src/lib.rs:12379-12395` — Coerces loop bounds to i64 and emits the Horner kernel, returning false to fall through to the generic 128-bit vectorizer.
- **bf16/f16→f32 axpby chunk emission** — `crates/wukong_mir_build/src/lib.rs:12592-12636` — Inline recognizer arm GEPing half-width inputs and f32 output by the chunk start and calling `axpby_bf16`/`axpby_f16` for mixed-precision residual-add/saxpy.
- **Per-channel int8 dequant epilogue** — `crates/wukong_mir_build/src/lib.rs:19895-20076` — Recognizes `out = act((c as f32)*scale_a*scale_b[j] + bias[j])` in any factor association, with optional per-tensor scale and bias, pinned to the int8 nest's dims.

## G13 — Recognizers: normalizations (softmax / LayerNorm / RMSNorm) and backwards

*Files:* `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_runtime/src/norm.rs`

**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Norm recognizer routing gates (plain / affine / batched / parallel)** — `crates/wukong_codegen_cranelift/src/tests.rs:4069-4123` — Pins affine vs plain kernel selection, rejects an affine-looking scaled softmax, and requires batched `for r { row }` RMSNorm/LayerNorm/softmax and their multicore twins to keep dispatching, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:4132-4194`, `crates/wukong_codegen_cranelift/src/tests.rs:4202-4249`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Batched-norm @parallel probe** — `crates/wukong_mir_build/src/lib.rs:1280-1324` — Builds a throwaway `FnLowerer` with a never-finished Builder to run the pure `match_batched_norm` recognizer, keeping fused per-row norm dispatch from being outlined away.
- **Normalization forward and backward symbols** — `crates/wukong_mir_build/src/lib.rs:1841-1857` — Fused softmax/LayerNorm/RMSNorm, their affine γ/β variants and batched parallel twins, plus the softmax/RMSNorm/LayerNorm input-gradient kernels, also `crates/wukong_mir_build/src/lib.rs:2007-2020`.
- **Fused normalization recognition window chain** — `crates/wukong_mir_build/src/lib.rs:3288-3327` — Probes log-softmax (6 stmts) before softmax (7 or fused 6), then LayerNorm, RMSNorm, and L2-norm, emitting one `wukong_norm_f32` call; softmax forms are restricted to in-place while the others allow out-of-place.
- **Running-max reduce body matcher** — `crates/wukong_mir_build/src/lib.rs:3464-3503` — Matches `m = fmax(m, x[v])` with either argument order, returning the reduced array, batch-offset aware; the shared first pass of softmax, log-softmax, xent and KD loss.
- **In-place exp-of-centered body matcher** — `crates/wukong_mir_build/src/lib.rs:3506-3550` — Matches `x[v] = exp(x[v] - m)` requiring f32 MIR type and identical array on both sides, the split softmax's second pass.
- **Scalar sum / sum-of-squares body matchers** — `crates/wukong_mir_build/src/lib.rs:3553-3603` — `s += x[v]` in both `+=` and `s = s + x` spellings (array-discovering variant names LayerNorm's row) plus `s += x[v]*x[v]` for RMSNorm; also `crates/wukong_mir_build/src/lib.rs:3607-3654`.
- **Affine (gamma/beta) peeler** — `crates/wukong_mir_build/src/lib.rs:3670-3711` — Strips `core * gamma[v] + beta[v]` in any operand order off a normalize RHS, requiring loop-var-indexed arrays distinct from the data; also `crates/wukong_mir_build/src/lib.rs:3658-3663`.
- **Normalize-scale body matcher (`match_scale_body`)** — `crates/wukong_mir_build/src/lib.rs:3720-3761` — Matches `out[v] = x[v]*inv` or `x[v]/den` plus optional affine, returning the destination array so both in-place and out-of-place `out = norm(x)` forms dispatch.
- **Split-window softmax recognizer (`match_softmax`)** — `crates/wukong_mir_build/src/lib.rs:3848-3893` — Fuses the 6-or-7-statement max/exp/sum/normalize window into one `wukong_norm_f32` call, requiring identical `0..N` bounds and that internal scalars never escape.
- **Softmax normalize-tail matcher (reciprocal or divide)** — `crates/wukong_mir_build/src/lib.rs:3907-3933` — Accepts either `let inv=1/s; x[i]*=inv` (2 stmts) or the textbook `x[i]/=s` (1 stmt), the latter canonicalized to reciprocal-multiply inside the kernel; also `crates/wukong_mir_build/src/lib.rs:3936-3967`.
- **Fused-window softmax recognizer (`match_softmax_fused`)** — `crates/wukong_mir_build/src/lib.rs:3984-4026` — Recognizes the one-fewer-pass spelling where exp-store and sum share a loop, folding to the identical norm kernel call as the split form.
- **Fused exp-store-and-sum body matcher** — `crates/wukong_mir_build/src/lib.rs:4038-4124` — Handles both the 3-statement `let e = exp(...)` form and the 2-statement re-read form, verifying the summed value is exactly the just-stored element.
- **Log-softmax window recognizer** — `crates/wukong_mir_build/src/lib.rs:4132-4174` — Six-statement max/Σexp/log/subtract window folded to `NORM_LOGSOFTMAX`, with its final in-place `x[v] = (x[v]-m) - ls` body matcher; also `crates/wukong_mir_build/src/lib.rs:5957-5991`.
- **Sum-of-exp-of-centered body matcher** — `crates/wukong_mir_build/src/lib.rs:4178-4235` — Matches `s += exp(x[v]-m)` with f32 type check and no in-place write, the shared Σexp pass of log-softmax, xent forward/backward, logsumexp and KD loss.
- **Recompute-softmax body matcher** — `crates/wukong_mir_build/src/lib.rs:4726-4760` — Matches `dx[v] = exp(x[v]-m) * invZ` writing a separate array, built on the reusable centered-exp predicate; also `crates/wukong_mir_build/src/lib.rs:4703-4721`.
- **Fused-norm kernel call emitter (emit_norm)** — `crates/wukong_mir_build/src/lib.rs:5998-6095` — Emits wukong_norm_f32 / norm_affine_f32 (or _parallel when batched inside @parallel), routing in-place vs out-of-place dst and null-pointer gamma/beta as integer zero.
- **Static eps resolution for the norm ABI** — `crates/wukong_mir_build/src/lib.rs:6100-6157` — Resolves the fused-norm epsilon to f32 bits from an inline literal, a nearest non-`mut` `let`, or a shadow-checked top-level `const`.
- **Row-count divisor pinning (count_as_f32 / recip_of_count / scaled_by_inv_count / match_mean)** — `crates/wukong_mir_build/src/lib.rs:6163-6215` — Proves a mean divisor equals the loop trip count across `/(n as f32)`, literal-vs-literal, and multiply-by-reciprocal spellings, preventing bogus norm fusion.
- **LayerNorm body matchers (is_centered, match_var_body, match_shift_scale_body)** — `crates/wukong_mir_build/src/lib.rs:6218-6327` — Match the centered value, the squared-deviation accumulation, and the normalize store with optional peeled affine gamma/beta and either multiply or divide spelling.
- **Reciprocal-std bindings (as_rsqrt_arg, match_inv_rstd, match_rstd)** — `crates/wukong_mir_build/src/lib.rs:6330-6459` — Recognize `1/sqrt(sum/n+eps)`, `rsqrt(...)`, or plain `sqrt(...)` denominators including the no-eps form (eps bits 0), yielding the eps the kernel needs.
- **LayerNorm 7-statement window recognizer (match_layernorm)** — `crates/wukong_mir_build/src/lib.rs:6479-6533` — Matches sum/mean/variance/rstd/normalize statement chains with identical `0..N` bounds and rejects windows whose internal scalars are read afterwards.
- **RMSNorm 4-statement window recognizer (match_rmsnorm)** — `crates/wukong_mir_build/src/lib.rs:6547-6589` — Matches mean-square sum, rstd binding, and affine scale store; supports batched row-offset indexing and out-of-place destinations.
- **L2-normalize recognizer and its bindings (match_inv_l2norm, match_l2den, match_l2norm)** — `crates/wukong_mir_build/src/lib.rs:6596-6725` — RMSNorm minus the `/N` divisor; structurally disjoint so windows match at most one, and any affine wrapper is deliberately declined.
- **Batched row-normalization recognizer (`match_batched_norm`)** — `crates/wukong_mir_build/src/lib.rs:9474-9541` — Probes softmax, fused softmax, LayerNorm, RMSNorm and L2-norm windows offset-indexed by `r*C` requiring every body statement be consumed; its doc comment claiming "RMSNorm only for now" contradicts the four families actually matched.
- **Batched norm dispatch (`try_emit_batched_norm`)** — `crates/wukong_mir_build/src/lib.rs:9543-9555` — Passes the recognized data/dst arrays, per-row width, eps, NORM_* opcode and optional gamma/beta to `emit_norm` with `rows` taken from the outer loop bound.
- **Batched softmax-backward recognizer** — `crates/wukong_mir_build/src/lib.rs:22905-23119` — Matches the per-row dot followed by `dx = y·(dy − s)`, verifying the dot ranges over the same two arrays and that dx aliases neither input.
- **Batched RMSNorm-backward recognizer** — `crates/wukong_mir_build/src/lib.rs:23452-23820` — A seven-statement per-row shape pinning both reductions, the rsqrt-of-mean-square eps binding, and the coefficient algebra; shared free helpers at `crates/wukong_mir_build/src/lib.rs:23413-23450`.
- **Batched LayerNorm-backward recognizer** — `crates/wukong_mir_build/src/lib.rs:23822-24106` — A thirteen-statement per-row shape matching four reductions, mean/rstd bindings, and the `rstd·(g − m1 − xhat·m2)` gradient with xhat/centered-read predicates.
**`crates/wukong_runtime/src/norm.rs`**

- **Duplicated log-softmax implementation** — `crates/wukong_runtime/src/norm.rs:92-134` — The `NORM_LOGSOFTMAX` scalar/AVX2 pair is a second, independently maintained copy of the same algorithm shipped as its own kernel, a real drift hazard; see `crates/wukong_runtime/src/logsoftmax.rs:94-115`.

## G14 — Recognizers: reductions, arg-reductions, column reductions, scans

*Files:* `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **`@parallel` reduction family gate with exact goldens** — `crates/wukong_codegen_cranelift/src/tests.rs:2487-2614` — Covers dot, ssd, sum, max, min, absmax, abssum and MAE folds where native splits across cores while the interpreter runs the serial kernel, with integer-exact golden values distinguishing RED_ABSDIFF from RED_SSD.
- **Reduction vectorization gates: reassociation, f64 reference and the 256-bit trip threshold** — `crates/wukong_codegen_cranelift/src/tests.rs:3247-3276` — Checks the reassociated dot against an f64 reference within `8·√n·ε`, asserts `veckernel` appears only at trips ≥2048, and pins int/ssd/fmax/fmin folds bit-exact across backends, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:3284-3334`, `crates/wukong_codegen_cranelift/src/tests.rs:3155-3239`, `crates/wukong_codegen_cranelift/src/tests.rs:3338-3412`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Reduction and arg-reduction dispatch symbols** — `crates/wukong_mir_build/src/lib.rs:1828-1840` — Deterministic parallel f32 reduction plus global and per-row/per-column argmax/argmin kernels, all documented as bit-identical serial-vs-parallel so the interp oracle stays exact, also `crates/wukong_mir_build/src/lib.rs:2050-2065`.
- **Transpose and column-reduction/statistics symbols** — `crates/wukong_mir_build/src/lib.rs:1963-2006` — Cache-blocked f32 and u16 transposes plus the strided axis-0 family (sum, max, min, absmax, mean, sumsq, L2, RMS) gcc leaves scalar, each with a parallel twin.
- **Scan/prefix dispatch symbols** — `crates/wukong_mir_build/src/lib.rs:2066-2088` — Per-row cumsum (Hillis-Steele, reassociating so the kernel is its own oracle), cumprod, cummax/cummin and the first-order linear-recurrence selective scan for SSM-style models.
- **Flat 1-D scan recognition (cumsum, cumprod, linear recurrence)** — `crates/wukong_mir_build/src/lib.rs:3336-3353` — Matches the two-statement `let acc` plus writeback-loop form that defeats the generic vectorizer and folds it into the serial scan kernel, complementing the batched for-statement path.
- **Per-row argmax/argmin nest recognizer** — `crates/wukong_mir_build/src/lib.rs:4929-5013` — Matches the seeded strict-compare top-1 scan (inner loop may start at 0 or 1) requiring an i32 output array, with its order-flexible `bv/bi` update body; also `crates/wukong_mir_build/src/lib.rs:5019-5094`.
- **Cumsum (inclusive prefix sum) recognizers, batched and flat** — `crates/wukong_mir_build/src/lib.rs:5138-5175` — Shared accumulator/store pair matcher plus batched `for r` and single-row wrappers emitting `wukong_cumsum_f32`, defeating the loop-carried recurrence gcc keeps scalar; also `crates/wukong_mir_build/src/lib.rs:5111-5131`, `crates/wukong_mir_build/src/lib.rs:5184-5198`.
- **Cumprod (inclusive prefix product) recognizers, batched and flat** — `crates/wukong_mir_build/src/lib.rs:5238-5274` — Multiplicative twin of cumsum with a `1.0` seed and `*` accumulate, bit-exact because a bare product is not reassociated; also `crates/wukong_mir_build/src/lib.rs:5212-5232`, `crates/wukong_mir_build/src/lib.rs:5281-5295`.
- **Scan accumulate-step matchers (add / multiply)** — `crates/wukong_mir_build/src/lib.rs:5340-5376` — Recognize `acc = acc + x[..i]` and `p = p * x[..i]` in both compound and binary spellings with either operand order, returning the data array; also `crates/wukong_mir_build/src/lib.rs:5300-5336`.
- **Linear-recurrence / selective-scan recognizers, batched and flat** — `crates/wukong_mir_build/src/lib.rs:5419-5456` — Shared `h = a[t]*h + b[t]; out[t] = h` pair matcher plus batched and 1-D wrappers emitting `wukong_lrscan_f32`, the SSM/Mamba/EMA kernel; also `crates/wukong_mir_build/src/lib.rs:5393-5414`, `crates/wukong_mir_build/src/lib.rs:5463-5479`.
- **Recurrence-step and gated-carry matchers** — `crates/wukong_mir_build/src/lib.rs:5485-5520` — Split the `Add` into a gated carry `a[..]·h` and a plain input read `b[..]` in either order, identifying the gate and input arrays; also `crates/wukong_mir_build/src/lib.rs:5524-5546`.
- **Cumulative max/min scan recognizer** — `crates/wukong_mir_build/src/lib.rs:5562-5594` — Matches `m = fmax/fmin(m, x[..]); out[..] = m` per row into `wukong_cummax/cummin_f32`, bit-exact since max/min select an input; also `crates/wukong_mir_build/src/lib.rs:5598-5653`.
- **Column-reduction dispatch table (emit_colsum)** — `crates/wukong_mir_build/src/lib.rs:7631-7661` — Maps eight column-reduction op codes (sum/max/min/maxabs/mean/sumsq/l2/rms) times serial/parallel onto runtime symbols with sum as the catch-all default.
- **Arg-reduction emitters (emit_colarg, emit_rowarg)** — `crates/wukong_mir_build/src/lib.rs:7851-7891` — Emit the i32-output per-column and per-row argmax/argmin kernels, four-way selection over max-vs-min and serial-vs-parallel.
- **Scan kernel emitters (cumsum, cumprod, lrscan, cummax/cummin)** — `crates/wukong_mir_build/src/lib.rs:7895-7978` — Emit per-row prefix sum/product, linear-recurrence scan, and cumulative max/min; note cumsum and cumprod share the same CumsumNest type.
- **Reduction-body recognizer (match_reduction_kernel)** — `crates/wukong_mir_build/src/lib.rs:8373-8523` — Classifies a single-statement fold into RED_DOT/SSD/SUM/SUMABS/ABSDIFF/MAX/MIN/MAXABS, requiring the index be exactly the loop variable and the addend free of the accumulator.
- **Argmax/argmin loop-body recognizer (`match_argreduce_kernel`)** — `crates/wukong_mir_build/src/lib.rs:8528-8597` — Pure AST match of `if x[k] >|< bv { bv = x[k]; bi = k }` (single stmt or tail, order-flexible assigns, `bi = k as int` allowed) returning value/index symbols, RED_ARGMAX/RED_ARGMIN and the array base.
- **Argreduce kernel dispatch with branchless seed reconcile (`try_emit_argreduce`)** — `crates/wukong_mir_build/src/lib.rs:8599-8709` — Emits one `wukong_argreduce_f32` call then folds its (value,index) against the running `(bv,bi)` via `Cmp+And/Or+Select` using a lowest-index tie-break so the result is seed-independent for any `0..n` loop.
- **Argreduce type/scope preconditions and `@parallel` twin selection** — `crates/wukong_mir_build/src/lib.rs:8609-8651` — Requires literal-0 start, an f32 `bv` slot, an integer `bi` slot (I64/I32/I16/I8) and an in-scope array, and picks `gemm.argreduce_par` inside a `@parallel` function claiming bit-identical ascending chunk folds.
- **`@parallel` f32 reduction dispatch (`try_emit_parallel_reduction`)** — `crates/wukong_mir_build/src/lib.rs:8711-8795` — Turns `for k in 0..n { s += f(x[k],y[k]) }` into one `sred_par` call combined into the accumulator additively, or by `Cmp(Fogt/Folt)+Select` for RED_MAX/RED_MIN/RED_MAXABS.
- **bf16/f16 mixed-precision reduction recognizer (`match_lowp_reduction`)** — `crates/wukong_mir_build/src/lib.rs:8797-8929` — Matches `s += (x[k] as f32)` (sum) and `(x[k] as f32)*(y[k] as f32)` (dot) through a shared widening-load closure that pins index `k`, cast target f32, and equal bf16/f16 precision on both operands.
- **Widened running max/min/absmax sub-pattern** — `crates/wukong_mir_build/src/lib.rs:8845-8886` — Inside `match_lowp_reduction`, recognizes `m = fmax/fmin(m, (x[k] as f32))` either operand order plus `fmax(m, abs(...))` as RED_MAXABS, the symmetric int8 quant scale over half-precision weights.
- **Low-precision reduction kernel-selection matrix (`try_emit_lowp_reduction`)** — `crates/wukong_mir_build/src/lib.rs:9357-9472` — Selects among eight dot/sum symbols keyed on (`parallel_fn`, is_f16) or an op-coded `reduce_{bf16,f16}[_par]` call for the max family, then folds the result into the f32 accumulator.
- **Column-reduction family recognizer** — `crates/wukong_mir_build/src/lib.rs:22310-22679` — Eight strided axis-0 folds (sum, max, min, absmax, mean, sumsq, L2, RMS) derived from a fold classifier plus a finalize classifier that rejects finalizes on non-additive bases.
- **Per-column arg-reduction recognizer** — `crates/wukong_mir_build/src/lib.rs:22681-22903` — Matches the strided argmax/argmin nest returning row indices with strict-compare tie-breaking, requiring a distinct i32 output and a row-zero seed.

## G15 — Recognizers: attention, embedding, scatter, losses, other ML ops

*Files:* `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_runtime/src/scatter.rs`

**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Embedding gather gate** — `crates/wukong_codegen_cranelift/src/tests.rs:2418-2449` — Requires `out[t,:] = weight[ids[t],:]` to dispatch to `wukong_embedding_f32(_parallel)` past the T>64 multicore threshold while a plain elementwise copy must not, extra site `crates/wukong_codegen_cranelift/src/tests.rs:2455-2479`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Loss, RoPE and log-partition symbols** — `crates/wukong_mir_build/src/lib.rs:2021-2049` — Softmax cross-entropy forward/backward, RoPE forward/backward, batched log-sum-exp, KL divergence, entropy and soft-label distillation loss, all row-independent so parallel equals serial.
- **Embedding, scatter-add, pooling and attention symbols** — `crates/wukong_mir_build/src/lib.rs:2089-2114` — Token-row gather, its scatter-add gradient dual, 2D max/avg pooling over `[C,H,W]`, and the fused SDPA kernel bundled here so it rides the GEMM symbol path.
- **Cross-entropy forward nest recognizer** — `crates/wukong_mir_build/src/lib.rs:4272-4335` — Recognizes the 5-statement per-row `loss[r] = m + log(s) - x[r*C+target[r]]` nest into `wukong_xent_fwd_f32`, rejecting aliasing among logits, labels and loss.
- **Label-gather index matcher (`match_xent_gather`)** — `crates/wukong_mir_build/src/lib.rs:4368-4396` — Matches the data-dependent `x[r*C + target[r]]` read with either addend order and an optional `as` cast, pinning the row stride via `is_mul_of`.
- **Embedding-lookup nest recognizer** — `crates/wukong_mir_build/src/lib.rs:4414-4471` — Matches the two-deep `out[t*H+d] = weight[ids[t]*H+d]` gather with f32 data / distinct output checks, dispatching the first layer of every LLM.
- **Indirect row-gather index matchers** — `crates/wukong_mir_build/src/lib.rs:4478-4530` — Match `weight[ids[t]*H + d]` with stride pinned to the inner bound, and the `ids[t]` selector requiring an i32 array with cast peeling; also `crates/wukong_mir_build/src/lib.rs:4535-4546`.
- **Embedding emitter with vocab sentinel** — `crates/wukong_mir_build/src/lib.rs:4554-4582` — Emits `wukong_embedding_f32[_parallel]` passing a hardcoded `1<<48` table-height sentinel so the kernel's out-of-range clamp never fires, mimicking the unchecked source.
- **Scatter-add (embedding-gradient) nest recognizer** — `crates/wukong_mir_build/src/lib.rs:4597-4667` — Matches `grad_w[ids[t]*H+d] += grad_out[t*H+d]`, requiring three distinct arrays and a statically-sized `grad_w` array type so the real table height is recoverable.
- **Scatter-add emitter with real V** — `crates/wukong_mir_build/src/lib.rs:4673-4699` — Emits `wukong_scatter_add_f32[_parallel]` computing `V = total/H` by MIR UDiv, because the parallel kernel partitions output rows and a sentinel would serialize it.
- **Cross-entropy backward nest recognizer** — `crates/wukong_mir_build/src/lib.rs:4782-4852` — Matches the 7-statement `softmax(x) - onehot(target)` gradient including the trailing `dx[r*C+target[r]] -= 1.0` scatter, dispatching `wukong_xent_bwd_f32`.
- **Batched log-sum-exp nest recognizer** — `crates/wukong_mir_build/src/lib.rs:4860-4911` — Xent minus the gather: max + Σexp then `out[r] = m + log(s)` folded to `wukong_logsumexp_f32` for CRF / log-partition workloads.
- **Row-reduction loss head and log-index helper** — `crates/wukong_mir_build/src/lib.rs:5711-5748` — Shared 3-statement `for r { let s=0; for i { s += term }; out[r]=... }` skeleton whose doc comment mislabels the returned tuple as `(…term, store_value)` when it actually returns `(…out, term)`; also `crates/wukong_mir_build/src/lib.rs:5695-5704`.
- **KL-divergence loss nest recognizer** — `crates/wukong_mir_build/src/lib.rs:5753-5803` — Matches `out[r] = Σ p·(log p − log q)` on the shared row-reduce head, dispatching `wukong_kldiv_f32`; it rejects `out` aliasing p or q but not p aliasing q.
- **Shannon-entropy loss nest recognizer** — `crates/wukong_mir_build/src/lib.rs:5807-5856` — Matches `out[r] = −Σ p·log p` requiring the store be a unary negation of the accumulator, dispatching `wukong_entropy_f32` for RL policy entropy.
- **Soft-label (distillation) cross-entropy recognizer** — `crates/wukong_mir_build/src/lib.rs:5862-5953` — Eight-statement nest reusing xent's max/Σexp/lse prefix then `out[r] = Σ q·(lse − x)`, dispatching `wukong_kd_loss_f32`.
- **Log-sum-exp emitter (emit_logsumexp)** — `crates/wukong_mir_build/src/lib.rs:7830-7847` — Emits the numerically-stable row log-sum-exp kernel over the shared two-pointer plus rows/cols ABI, with a parallel variant.
- **Divergence and soft-label loss emitters (kldiv, entropy, kd_loss)** — `crates/wukong_mir_build/src/lib.rs:7981-8042` — Emit row-wise KL divergence, entropy, and soft-label cross-entropy kernels over the two- or three-pointer plus rows/cols ABI.
- **Fused sdpa attention lowering** — `crates/wukong_mir_build/src/lib.rs:16749-16766` — Recognizes the 8-argument `sdpa` builtin and emits one `wukong_attention_f32` call, avoiding materialization of the S×S score matrix on both backends.
- **ML-nest plan structs and @parallel interceptor probes** — `crates/wukong_mir_build/src/lib.rs:23131-23367` — Declares xent, logsumexp, kldiv, entropy, kd-loss, rowarg, cumsum, lrscan, cumminmax, embedding and scatter plans; throwaway-lowerer probes run before the outliner at `crates/wukong_mir_build/src/lib.rs:23371-23409`.
**`crates/wukong_runtime/src/scatter.rs`**

- **scatter-add row accumulate (AVX2 + scalar twin)** — `crates/wukong_runtime/src/scatter.rs:45-97` — grad_w_row += grad_out_row eight f32 per step with a scalar tail, feature-detected, one IEEE add per lane so both paths agree bit-for-bit.
- **scatter_rows output-row-range filter and out-of-range id guard** — `crates/wukong_runtime/src/scatter.rs:99-132` — Scans all T ids ascending and applies only ids inside [r0,r1), which doubles as the id<v bound; negative or oversized ids are silently skipped.

## G16 — @parallel: region scanning, loop outlining, parallel dispatch

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_runtime/src/bias.rs`, `crates/wukong_runtime/src/cumminmax.rs`, `crates/wukong_runtime/src/cumprod.rs`, `crates/wukong_runtime/src/cumsum.rs`, `crates/wukong_runtime/src/entropy.rs`, `crates/wukong_runtime/src/lib.rs`, `crates/wukong_runtime/src/lrscan.rs`, `crates/wukong_runtime/src/pool2d.rs`, `crates/wukong_runtime/src/rope.rs`, `crates/wukong_runtime/src/rope_bwd.rs`, `crates/wukong_runtime/src/scatter.rs`, `crates/wukong_runtime/src/velem.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **@parallel-for runtime entry lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1066-1075` — Lowers `wukong_parallel_for(n, body_ptr, env_ptr)` calls, the vehicle by which outlined loop bodies reach the thread pool.
**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **`@parallel` region outlining and head-loop gate** — `crates/wukong_codegen_cranelift/src/tests.rs:4706-4782` — A mid-function head loop with body-local scratch must outline into `wukong_parallel_for` and be bit-exact three ways (vs its serial twin in-program, vs the interpreter, and -O0 vs every level) for even and ragged head counts, extra site `crates/wukong_codegen_cranelift/src/tests.rs:4424-4440`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Pre-interned @parallel region symbol pool** — `crates/wukong_mir_build/src/lib.rs:851-869` — Mints `wukong$par$0..PAR_REGION_MAX` up front only when some item carries `@parallel`, because the lowerer holds a shared `&Interner` and cannot intern lazily.
- **Whole-function @parallel kernel interception ladder** — `crates/wukong_mir_build/src/lib.rs:880-1189` — Twenty ordered probes (matmul, int8, lowp, gemv/gevm, transpose, pool, colsum, norms, losses, scans, embedding, scatter, bias-bcast) route recognized bodies to multicore kernels instead of the generic loop outliner.
- **@parallel attribute predicate** — `crates/wukong_mir_build/src/lib.rs:1218-1223` — String-compares each item attribute name against "parallel"; the single gate every whole-function interception and outlining decision consults.
- **Outlinable parallel-loop recognizer** — `crates/wukong_mir_build/src/lib.rs:1225-1278` — `parallel_spec` demands all-array params, a single-statement body, an exclusive step-less range starting at literal 0, and an identifier pattern before outlining is allowed.
- **Whole-function @parallel outliner** — `crates/wukong_mir_build/src/lib.rs:1556-1700` — Splits `@parallel fn { for i in 0..hi {..} }` into an outlined `(start,end,env)` body plus a wrapper calling `wukong_parallel_for`, the all-core vehicle for kernel dispatch.
- **Parallel env pointer-array marshalling** — `crates/wukong_mir_build/src/lib.rs:1655-1686` — Wrapper allocas an `[Ptr; k]` env, stores every parameter into it, and the body reloads each slot as a base pointer; scalar params are stored/reloaded as `Ptr` yet bound with their real type, also `crates/wukong_mir_build/src/lib.rs:1607-1623`.
- **ParRegions pre-interned outlined-symbol pool** — `crates/wukong_mir_build/src/lib.rs:1702-1720` — Fixed 32-slot pool of `wukong$par$<n>` names (deliberately not `wukong_`-prefixed to dodge dispatch greps) for mid-function parallel regions; overflow silently stays serial, also `crates/wukong_mir_build/src/lib.rs:861-861`, `crates/wukong_mir_build/src/lib.rs:10060-10060`.
- **`@parallel` propagation into statement-level kernel dispatch** — `crates/wukong_mir_build/src/lib.rs:9565-9569` — Each recognizer emitter receives `self.parallel_fn` to select the multicore kernel twin; the matmul site documents a fixed bug where this was hardcoded false, leaving `@parallel` transformer GEMMs serial.
- **Mid-function `@parallel` region legality contract** — `crates/wukong_mir_build/src/lib.rs:10002-10059` — The mixed-radix-digit soundness argument (single- and two-digit tiled forms, body-local scratch privacy, decline list, autodiff loud-failure note) governing the outliner probed last at `crates/wukong_mir_build/src/lib.rs:9905-9922`.
- **@parallel region entry gate** — `crates/wukong_mir_build/src/lib.rs:10060-10089` — Gates mid-function outlining on ident pattern, literal `0..N` bounds and `n >= 2`, then runs legality analysis before emit.
- **Per-array common access signature (pass 1)** — `crates/wukong_mir_build/src/lib.rs:10113-10129` — Requires every read and write of a written captured array to decompose under one identical loop-var signature, else declines.
- **Mixed-radix digit extent bounding (pass 2)** — `crates/wukong_mir_build/src/lib.rs:10130-10167` — Proves per-iteration slice disjointness by bounding residual index terms below the digit strides for direct `hh*C` and tiled `(hh/C,hh%C)` forms.
- **Region control-flow legality walk** — `crates/wukong_mir_build/src/lib.rs:10173-10264` — Scoped block walk rejecting `return`/`defer`, labeled or region-depth `break`/`continue`, while nesting depth tracked so inner-loop breaks stay legal.
- **Region let-binding aliasing rejection** — `crates/wukong_mir_build/src/lib.rs:10191-10212` — Declines any local whose initializer type or annotation is slice/pointer/reference, since such a local could launder writes past the write rules.
- **Digit-derived local modeling** — `crates/wukong_mir_build/src/lib.rs:10213-10233` — Records locals whose initializer is an affine combination of the region var's div/mod digits so tiled index terms classify; evaluated before binding, invalidated on reassignment.
- **Inner for-loop env with literal bounds** — `crates/wukong_mir_build/src/lib.rs:10265-10325` — Pushes each inner loop variable with an optional literal non-negative unit-step range so index classification can bound its contribution; non-literal ranges become outer-only.
- **Assignment-target classification** — `crates/wukong_mir_build/src/lib.rs:10331-10370` — Body-local places are private, single-index writes to captured fixed-size arrays are recorded for the proof, and everything else (scalar captures, whole arrays) declines.
- **Conservative region expression walk** — `crates/wukong_mir_build/src/lib.rs:10372-10480` — Whitelists literals/paths/binops/casts/aggregates/if/block, allows only pure math-intrinsic calls, rejects deref/ref, match and loop, with a declining catch-all.
- **Bare-name read classification** — `crates/wukong_mir_build/src/lib.rs:10485-10499` — Region var and locals are free, non-runtime names inline, others become captures; a whole-array read records an opaque access fatal only if written.
- **Capture binding-shape filter** — `crates/wukong_mir_build/src/lib.rs:10504-10516` — Accepts only fixed-size arrays and int/float scalars as captures, rejecting Ptr-slot tensors and vectors that could alias other captures.
- **Region shadowing guard and pattern binder** — `crates/wukong_mir_build/src/lib.rs:10518-10544` — Declines rebinding the region var or an enclosing loop var, sheds stale digit models on rebind, and binds ident/wildcard/tuple patterns only.
- **Outlined-call env pointer table** — `crates/wukong_mir_build/src/lib.rs:10550-10607` — Reserves an outlined symbol from a fixed pool (exhaustion falls back to serial) and packs capture base/slot pointers into a uniform array passed to `wukong_parallel_for`.
- **Outlined body function construction** — `crates/wukong_mir_build/src/lib.rs:10609-10662` — Builds `f(start,end,env)` with a fresh FnLowerer that reloads captures from env and lowers the untouched body via `lower_ranged_loop`, deliberately setting `parallel_fn: false` to keep serial kernels bit-exact.
- **`@parallel` per-chunk kernel dispatch cascade (`try_vectorize_ranged`)** — `crates/wukong_mir_build/src/lib.rs:12553-12649` — Re-runs vmath/velem/dequant/axpby/horner recognizers on already-lowered chunk bounds so each thread's slice becomes one 256-bit kernel call, giving multicore × SIMD.
- **Pre-lowered-bounds ranged loop (`lower_ranged_loop`)** — `crates/wukong_mir_build/src/lib.rs:14565-14635` — The `@parallel` outliner's per-thread loop: narrows i64 bounds to the index type, tries vectorization first, and lowers an unlabeled loop with a separate latch for `continue`.
- **@parallel region scan state and digit modeling** — `crates/wukong_mir_build/src/lib.rs:19027-19172` — Tracks scopes, captures, per-access environments, and locals bound to affine combinations of the region variable's div/mod digits, invalidating them on rebind and scope exit.
- **Mixed-radix disjointness proof for parallel regions** — `crates/wukong_mir_build/src/lib.rs:19179-19281` — Decomposes each written index into a `T(c)` or `DivMod` signature, rejecting duplicated digits, mismatched divisors and non-injective equal coefficients; extent bounding at `crates/wukong_mir_build/src/lib.rs:19290-19336`.
**`crates/wukong_runtime/src/bias.rs`**

- **Parallel bias over fixed 8-row bands** — `crates/wukong_runtime/src/bias.rs:176-220` — Splits rows into `RBAND=8` contiguous bands, each recursively calling the serial kernel on a disjoint slice; matrices of ≤8 rows run serially.
**`crates/wukong_runtime/src/cumminmax.rs`**

- **cumminmax serial/parallel drivers and the four exported entry points** — `crates/wukong_runtime/src/cumminmax.rs:195-299` — Shared ext-parameterised drivers with a 256-row parallel gate, exposed as cummax/cummin serial and _parallel C symbols mapping rows across rayon.
**`crates/wukong_runtime/src/cumprod.rs`**

- **cumprod parallel row-chunk split with CUMPROD_PAR_MIN** — `crates/wukong_runtime/src/cumprod.rs:75-124` — Below 64 rows serial, else one contiguous row chunk per rayon thread running the identical interleaved kernel, bit-identical to serial.
**`crates/wukong_runtime/src/cumsum.rs`**

- **cumsum serial and per-row parallel drivers with CUMSUM_PAR_MIN** — `crates/wukong_runtime/src/cumsum.rs:131-185` — Below 256 rows serial, else one rayon task per row (finer than cumprod/lrscan's chunking) with each row a disjoint output sub-slice.
**`crates/wukong_runtime/src/entropy.rs`**

- **Row-parallel `_parallel` entry family** — `crates/wukong_runtime/src/entropy.rs:146-180` — Seven near-identical rayon row maps with a PAR_MIN=8 serial fallback and raw pointers smuggled across the closure as `usize`, also at `crates/wukong_runtime/src/layernorm_bwd.rs:342-390`, `crates/wukong_runtime/src/rmsnorm_bwd.rs:250-297`, `crates/wukong_runtime/src/xent.rs:178-215`, `crates/wukong_runtime/src/xent_bwd.rs:208-250`, `crates/wukong_runtime/src/kd_loss.rs:222-262`, `crates/wukong_runtime/src/kldiv.rs:146-184`.
**`crates/wukong_runtime/src/lib.rs`**

- **`wukong_parallel_for` C-ABI region entry point** — `crates/wukong_runtime/src/lib.rs:397-467` — The symbol the backend lowers `@parallel for` to: clamps `n<=0`, reads the worker count inside the installed pool, and forks `body(start,end,env)` chunks that must be index-independent.
- **Dynamic granule claim queue for parallel-for** — `crates/wukong_runtime/src/lib.rs:433-454` — Cache-line-isolated (128 B) atomic counter hands out `ceil(n/(workers*8))` granules so fast P-cores absorb E-core straggling, with a static equal-chunk split as the fallback branch at `crates/wukong_runtime/src/lib.rs:455-465`.
**`crates/wukong_runtime/src/lrscan.rs`**

- **lrscan parallel row-chunk split and LRSCAN_PAR_MIN** — `crates/wukong_runtime/src/lrscan.rs:102-174` — Below 64 rows runs serial, else slices rows into one contiguous chunk per rayon thread (bare global pool, not the wuk pool) preserving the 4-row interleave.
**`crates/wukong_runtime/src/pool2d.rs`**

- **Parallel pooling channel fan-out with POOL_PAR_MIN gate** — `crates/wukong_runtime/src/pool2d.rs:390-447` — Below 4 channels runs serial; otherwise rayon maps whole channel planes across cores via `usize`-laundered pointers, bit-identical since output planes are disjoint.
**`crates/wukong_runtime/src/rope_bwd.rs`**

- **Parallel RoPE backward with ROPE_BWD_PAR_MIN gate** — `crates/wukong_runtime/src/rope_bwd.rs:160-205` — Same 8-row threshold and rayon disjoint-row fan-out as the forward, bit-identical to the serial entry because rows never combine.
**`crates/wukong_runtime/src/rope.rs`**

- **Parallel RoPE with ROPE_PAR_MIN row-count gate** — `crates/wukong_runtime/src/rope.rs:154-198` — Below 8 rows delegates to the serial entry; otherwise rayon maps disjoint rows, laundering raw pointers through `usize` to cross the Send boundary, staying bit-identical to serial.
**`crates/wukong_runtime/src/scatter.rs`**

- **Deterministic parallel scatter by output-row split** — `crates/wukong_runtime/src/scatter.rs:162-215` — Splits V weight-gradient rows (not tokens) across cores so colliding tokens never race, avoiding atomics and keeping ascending-ti fold order; serial entry and SCATTER_PAR_MIN=64 at `crates/wukong_runtime/src/scatter.rs:136-160`.
**`crates/wukong_runtime/src/velem.rs`**

- **velem multicore fixed-VCHUNK decomposition** — `crates/wukong_runtime/src/velem.rs:361-427` — Splits [0,n) into thread-count-independent 8192-element chunks on the unified wuk pool, inheriting one whole-array NT decision so parallel equals serial bit-for-bit.

## G17 — AST-level Auto-Vectorizer (128/256-bit) and FMA contraction

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir/src/inst.rs`, `crates/wukong_mir/src/vec_kernel.rs`, `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Hardware FMA, sqrt, round and splat opcodes** — `crates/wukong_codegen_cranelift/src/lib.rs:745-775` — Lowers `Splat`/`Fma`/`Sqrt`/`Round` to single Cranelift instructions (scalar or 128-bit vector), with `Nearest` meaning ties-to-even to mirror the interpreter.
**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Ignored same-run 256-vs-128 throughput A/B benches** — `crates/wukong_codegen_cranelift/src/tests.rs:36-81` — Compiles one kernel twice under `WUKONG_P4_NO_256` and reports the best-of-5 alternating ratio; a perf sanity probe, not a gate, extra site `crates/wukong_codegen_cranelift/src/tests.rs:89-129`.
- **256-bit vectorizer differential coverage sweep** — `crates/wukong_codegen_cranelift/src/tests.rs:137-187` — Crosses twelve elementwise body shapes with sixteen trip counts straddling the 8-lane boundary in both `for` and `while` form, asserting native==interp and -O0==-O3 each time.
- **FMA contraction and inline-transcendental vectorization gates** — `crates/wukong_codegen_cranelift/src/tests.rs:3474-3510` — Proves `x + y*z` prints as `fma` in MIR and that native `fma` equals the interpreter's `mul_add` bit-for-bit, plus vectorized inline exp/sigmoid/sin bodies where scalar operands must splat to all lanes, extra site `crates/wukong_codegen_cranelift/src/tests.rs:1403-1436`.
- **Loop-fusion structural equivalence gate** — `crates/wukong_codegen_cranelift/src/tests.rs:3515-3566` — Counts MIR instructions to assert two adjacent same-range elementwise loops lower to exactly the same program as the hand-fused single loop, then checks the value.
- **General AVX2 recipe + counting-while normalization gates** — `crates/wukong_codegen_cranelift/src/tests.rs:3573-3602` — Asserts a real `vec_kernels` entry is synthesized then checks bit-exactness against the interpreter, including that a normalized `while` preserves the post-loop counter for the empty-run case, extra site `crates/wukong_codegen_cranelift/src/tests.rs:3608-3646`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Vectorized-body load CSE cache** — `crates/wukong_mir_build/src/lib.rs:2309-2312` — `vec_loads` keys already-loaded vectors by canonical index text so a twice-read `x[i]` loads once, invalidated between unroll copies and after any store.
- **Adjacent same-range elementwise loop fusion (`try_fuse_run`)** — `crates/wukong_mir_build/src/lib.rs:3379-3416` — Finds the maximal run of `for` loops over a structurally identical range then lowers the longest prefix (≥2) whose concatenated body the vectorizer accepts, using vectorizability as the dependence-safety proof.
- **Counting-while to for-range normalization (try_normalize_counting_while, is_unit_incr)** — `crates/wukong_mir_build/src/lib.rs:7073-7156` — Rewrites `while i < N { …; i += 1 }` into the vectorizable range form, committing only after a pure vectorizability check, then restores `i` by select.
- **Auto-vectorizer probe cascade for `for j in a..b`** — `crates/wukong_mir_build/src/lib.rs:12398-12469` — Ordered attempt of vmath/vmath2/velem/vhorner/dequant kernel recognizers, then reduction analysis, then the generic vectorizer, bailing to scalar lowering.
- **Float/int reduction recognizer (`reduction_of`)** — `crates/wukong_mir_build/src/lib.rs:12476-12549` — Matches `s += e` / `s = s+e` / `s = fmax|fmin(s,e)` single-statement bodies, requiring lane type == accumulator type, returning addend, lane, width and fold kind.
- **Statement-level vectorizability analysis + loop-carried dependence check** — `crates/wukong_mir_build/src/lib.rs:12653-12724` — Validates let/assign-only bodies, unit-stride stores, index-independence from inner temps, and rejects any written array read at a second distinct index.
- **Intrinsic-shadowing precedence rule (`vectorizable_intrinsic`)** — `crates/wukong_mir_build/src/lib.rs:12731-12746` — Resolves a callee to a `MathIntrinsic` only when no user `fn` of that name exists, mirroring `lower_call` so the vectorizer never lowers a shadowed call as math.
- **Value-expression vectorizability check (`vec_check_value`)** — `crates/wukong_mir_build/src/lib.rs:12748-12932` — Pins one shared lane type across array reads, invariant scalars, literals, +-*/, neg, if-conversion and ~45 math intrinsics, restricting transcendentals to an f32 lane.
- **SIMD integer-division bail** — `crates/wukong_mir_build/src/lib.rs:12799-12808` — Declines to vectorize `/` on an integer lane because Cranelift x86 has no vector sdiv/udiv, which would be a native verifier panic while the interpreter ran it lane-wise.
- **If-conversion shifted-index speculative-load guard** — `crates/wukong_mir_build/src/lib.rs:12814-12848` — Refuses if-conversion when either branch reads `x[i±k]`, since a compare+blend evaluates both arms in every lane and would OOB-load in guard-false boundary lanes.
- **Raw-AVX2 `VecRecipe` capture and op lowering** — `crates/wukong_mir_build/src/lib.rs:12948-13120` — Flattens an f32 body into streams, pre-loaded invariant scalars and a linear `VecOp` list with load CSE and store-forwarding; only `sqrt`, arithmetic and cmp/select are expressible.
- **256-bit kernel call marshalling (`emit_veckernel_for`)** — `crates/wukong_mir_build/src/lib.rs:13126-13253` — Computes `n = trip & -8` (zero when under 8), stores stream base pointers and scalar values into stack arrays, issues `VecKernelCall`, then runs the scalar tail.
- **256→128-bit fallback policy, `WUKONG_P4_NO_256` kill-switch and register-pressure gate** — `crates/wukong_mir_build/src/lib.rs:13259-13311` — Tries the raw-AVX2 recipe only for f32 lanes when the env knob is unset and `kern.pressure().fits()`, else emits the CLIF 128-bit strips.
- **128-bit strip-mined vector loop + scalar tail emitters** — `crates/wukong_mir_build/src/lib.rs:13317-13443` — Emits an unrolled `VEC_UNROLL`-group strip, a single-vector strip and a scalar remainder over one shared index slot, clearing the load cache per unroll copy.
- **256-bit AVX2 reduction kernel (recipe + emission)** — `crates/wukong_mir_build/src/lib.rs:13453-13676` — Builds a fold recipe (fusing `x*y` into `vfmadd` for sums only), rejects Splat/Const folded values, calls the kernel and combines its horizontal result with the initial accumulator.
- **128-bit reduction with per-unroll lane accumulators** — `crates/wukong_mir_build/src/lib.rs:13684-13805` — Initializes `VEC_UNROLL` accumulators to the fold identity (0 / -inf / +inf), runs two strips, then horizontally folds every lane into the scalar accumulator.
- **`VEC256_REDUCTION_MIN_TRIP` trip-size heuristic** — `crates/wukong_mir_build/src/lib.rs:13697-13710` — Uses the 256-bit reduction only when both loop bounds are compile-time constants and the trip is large enough to amortize the out-of-line call; also `crates/wukong_mir_build/src/lib.rs:12454-12463`.
- **Reduction strip and scalar-remainder emitters** — `crates/wukong_mir_build/src/lib.rs:13811-13964` — Fold `unroll` independent vector groups into separate accumulator slots per iteration, then a scalar `while j < end { s = fold(s, addend) }` remainder.
- **Fold semantics and FMA accumulate (`reduce_combine_*`, `vec_accumulate`, `scalar_accumulate`)** — `crates/wukong_mir_build/src/lib.rs:13968-14087` — Sum uses FAdd/Add, fmax/fmin use the same compare+select the scalar intrinsic emits, and `acc + x*y` contracts to one `Fma` for floats.
- **Vector body lowering with load cache and intermediate-forwarding** — `crates/wukong_mir_build/src/lib.rs:14090-14241` — Lowers validated let/assign statements to lane ops, caches vector loads per body copy, clears the cache on any store then re-seeds it with the just-stored unit-stride value.
- **Per-lane math intrinsic expansion table** — `crates/wukong_mir_build/src/lib.rs:14244-14496` — Expands ~45 intrinsics (exp/log family, trig, activations and six activation backwards) into inline vector polynomials, with int-lane guards making `abs` a select and rounding the identity.
- **Vector FMA contraction (`vec_try_fma`)** — `crates/wukong_mir_build/src/lib.rs:14512-14545` — Contracts a float `x + y*z` or `y*z + x` inside a vector body into a single lane-wise `Op::Fma`, the matmul/saxpy inner-loop win.
- **Scalar FMA contraction (`try_contract_fma`)** — `crates/wukong_mir_build/src/lib.rs:16153-16198` — Contracts float `x + y*z` into one `Op::Fma` lowering operands in source order, matched by the interpreter's `mul_add` so both backends stay bit-identical.
- **Affine stride and shifted-index if-conversion guard** — `crates/wukong_mir_build/src/lib.rs:18659-18713` — Rejects speculative SIMD if-conversion when a branch reads a nonzero-offset array element, which would fault past the array end in masked-off lanes.
- **Vectorizer recipe with load CSE and store forwarding** — `crates/wukong_mir_build/src/lib.rs:24444-24480` — Interns unit-stride streams and invariant scalars by structural equality and caches per-stream values so repeated reads reuse one load; index keys at `crates/wukong_mir_build/src/lib.rs:18936-18958`.
- **Vectorizer width and unroll tuning knobs** — `crates/wukong_mir_build/src/lib.rs:24484-24545` — 128-bit register width with 4× unroll (Cranelift cannot legalize wider lanes) and the measured 2048-trip threshold gating the out-of-line 256-bit reduction kernel.
- **For-body fusion block concatenation** — `crates/wukong_mir_build/src/lib.rs:24586-24621` — Concatenates a run of loop bodies preserving NodeIds and re-appends each trailing expression as a statement, fixing a silent both-backends-agree miscompile.
**`crates/wukong_mir/src/inst.rs`**

- **VecKernelCall op and the per-function AVX2 kernel arena** — `crates/wukong_mir/src/inst.rs:209-222` — Side-effecting call to a synthesized 256-bit kernel by function-local index (invisible to SSA renaming), marshalling stream pointers, invariant scalars and a multiple-of-8 trip count, also `crates/wukong_mir/src/lib.rs:148-153` and `crates/wukong_mir/src/builder.rs:45-51`.
**`crates/wukong_mir/src/vec_kernel.rs`**

- **VecKernel recipe data model and kernel ABI** — `crates/wukong_mir/src/vec_kernel.rs:1-66` — Defines the backend-agnostic 256-bit kernel recipe (name, stream/scalar counts, SSA op list, unroll, optional reduce) plus its `VecOp` instruction set of load/splat/const/bin/fma/sqrt/neg/cmp/select/store; also `crates/wukong_mir/src/vec_kernel.rs:126-138`.
- **Reduction fold semantics, identities and fused-FMA accumulation** — `crates/wukong_mir/src/vec_kernel.rs:68-124` — `VecRedOp::fold` deliberately mirrors `vmaxps`/`vminps` operand order so ties, ±0 and NaN agree with machine code, and `VecReduce.fma` collapses a product addend into one fused rounding.
- **Per-lane reference semantics with store-to-load forwarding** — `crates/wukong_mir/src/vec_kernel.rs:140-222` — The single source of truth both the interpreter oracle and the AVX2 assembler must match bit-for-bit, including sign-bit `Neg`, single-rounded `mul_add`, 1.0/0.0 compare masks, and forwarding a just-stored stream value to later loads.
- **Fixed reassociation oracle for reductions** — `crates/wukong_mir/src/vec_kernel.rs:246-309` — Reproduces the assembler's exact float grouping — U unrolled lane accumulators, a single-group cleanup, lane-wise combine into copy 0, then a sequential lane-0..7 horizontal fold — so results are reproducible across backends.
- **Shared register-pressure analysis and unroll/feasibility gating** — `crates/wukong_mir/src/vec_kernel.rs:311-434` — Computes hoisted broadcasts, sign-mask need, last-use liveness (with reduction operands pinned past the end) and peak live registers, single-sourced so the vectorizer's fit gate and the assembler's plan can never disagree; rejects `Const` ops and >4 streams outright; also `crates/wukong_mir/src/vec_kernel.rs:224-244`.

# Backends

## G18 — Cranelift Native Backend: MIR to CLIF, calling convention, object emit, linking

*Files:* `crates/wukong_codegen_cranelift/src/backend.rs`, `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_runtime/src/lib.rs`

**`crates/wukong_codegen_cranelift/src/backend.rs`**

- **`CraneliftBackend` `Backend` trait implementation** — `crates/wukong_codegen_cranelift/src/backend.rs:1-26` — Adapts `jit_run` to the shared `Artifact::Executed { exit_code, stdout }` shape so driver and differential gate treat native and interpreter interchangeably.
**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Print/assert runtime shims and intrinsic classification** — `crates/wukong_codegen_cranelift/src/lib.rs:43-91` — Captures generated-code output into a process-global `OUTPUT` buffer plus `ASSERT_FAILED`/`RUN_LOCK` so native bytes match the interpreter oracle exactly; see also `crates/wukong_codegen_cranelift/src/lib.rs:281-305` and `crates/wukong_codegen_cranelift/src/lib.rs:1721-1767`.
- **MIR type and byte-layout mapping with overflow-safe sizing** — `crates/wukong_codegen_cranelift/src/lib.rs:309-343` — Maps MIR types to CLIF (`i1`→`i8`, `f16`/`bf16`→`f32`, arrays→pointer) and returns `None` on >4 GiB layouts, turned into a clean diagnostic at `crates/wukong_codegen_cranelift/src/lib.rs:440-458`.
- **Comparison-predicate mapping matching interpreter semantics** — `crates/wukong_codegen_cranelift/src/lib.rs:345-375` — Translates MIR `CmpOp` to `IntCC`/`FloatCC`, deliberately mapping `Fone` to unordered `NotEqual` (Rust `f64::ne`) rather than ordered `one` for bit-exact parity.
- **Function translation driver: RPO scheduling and block-parameter SSA wiring** — `crates/wukong_codegen_cranelift/src/lib.rs:469-522` — Computes reverse postorder, appends entry/interior block params typed from MIR, lowers each block, then seals all blocks; successor computation dedups identical CondBr targets at `crates/wukong_codegen_cranelift/src/lib.rs:388-402` and translator state lives at `crates/wukong_codegen_cranelift/src/lib.rs:407-438`.
- **Vector-aware Neg and Select lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:574-627` — Dispatches `Neg` on the vector lane type (avoiding invalid `ineg` on float vectors) and lowers `Select` as `bitselect` blend for vectors, unifying mismatched float widths for scalars.
- **Address materialization: alloca, gep, static-data and function addresses** — `crates/wukong_codegen_cranelift/src/lib.rs:628-636` — Creates aligned explicit stack slots, scales gep indices by element size, and materializes `.rodata` string-blob and function addresses; see `crates/wukong_codegen_cranelift/src/lib.rs:377-386`, `crates/wukong_codegen_cranelift/src/lib.rs:694-700`, `crates/wukong_codegen_cranelift/src/lib.rs:733-744`.
- **Per-result value normalization** — `crates/wukong_codegen_cranelift/src/lib.rs:777-791` — After every instruction, masks `i1` results to their low bit and re-coerces float results to the declared width, keeping `vmap` types canonical and matching interpreter normalization.
- **Float binary-op coercion and true-fmod remainder** — `crates/wukong_codegen_cranelift/src/lib.rs:794-824` — Computes float ops in the result type with operands coerced, and routes `FRem` to runtime `fmod` shims (`crates/wukong_codegen_cranelift/src/lib.rs:93-102`) instead of the drifting trunc identity.
- **Integer division/remainder safety guard** — `crates/wukong_codegen_cranelift/src/lib.rs:837-875` — Rewrites zero and (signed) −1 divisors to 1, then patches results so div-by-zero yields 0 and `INT_MIN/-1` wraps, avoiding hardware traps and matching the interpreter.
- **Cast lowering with rounding at widen/narrow boundaries** — `crates/wukong_codegen_cranelift/src/lib.rs:879-941` — Handles all `CastKind`s; `FpExt` re-rounds bf16/f16 sources and `FpTrunc` demotes-then-rounds, defending against optimizer store-bypass that previously made -O2 disagree with -O0.
- **Saturating float→int conversion for narrow targets** — `crates/wukong_codegen_cranelift/src/lib.rs:943-979` — Works around x64 Cranelift's inability to target `i8`/`i16` by converting to `i32` saturating then clamping and reducing, reproducing Rust `as` semantics including NaN and out-of-range.
- **User-call fast path and argument coercion helpers** — `crates/wukong_codegen_cranelift/src/lib.rs:1060-1065` — Dispatches known user functions through pre-declared FuncRefs before any runtime-name matching; the `coerce_to_i64/f64/f32` helpers used by every kernel ABI live at `crates/wukong_codegen_cranelift/src/lib.rs:1770-1801`.
- **Terminator lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1803-1843` — Emits `return`, `jump`/`brif` with block arguments converted to `BlockArg::Value`, and lowers `Unreachable` to a user trap code rather than falling through.
- **Function signature construction and calling convention** — `crates/wukong_codegen_cranelift/src/lib.rs:2024-2039` — Builds a Cranelift `Signature` from MIR params/return using the module's default call conv, silently dropping `Void`-typed parameters and returns.
- **Target ISA construction and flag selection** — `crates/wukong_codegen_cranelift/src/lib.rs:2063-2099` — Configures `opt_level=speed`, per-caller PIC (JIT needs non-PIC, objects want PIC), and inline stack probing so large local arrays touch guard pages without an external `__chkstk`.
- **Runtime import declaration and module population** — `crates/wukong_codegen_cranelift/src/lib.rs:2101-2600` — Builds ~25 shared runtime signatures and declares every `RT_*` symbol as an import before defining user bodies; the `RtFuncs` FuncId registry holding them is at `crates/wukong_codegen_cranelift/src/lib.rs:1848-2022`.
- **Runtime-kernel extern import table (tail)** — `crates/wukong_codegen_cranelift/src/lib.rs:2601-2865` — Declares ~130 more `wukong_*` runtime kernels as `Linkage::Import` FuncIds into the module, each reusing one of a dozen shared signature classes.
- **Kernel ABI signature-class aliasing** — `crates/wukong_codegen_cranelift/src/lib.rs:2649-2662` — Unrelated kernel families deliberately share one Cranelift signature because only the ABI shape matters, so element types are invisible to the linker; also `crates/wukong_codegen_cranelift/src/lib.rs:2786-2793`, `crates/wukong_codegen_cranelift/src/lib.rs:2821-2833`.
- **Kernels declared with no `@parallel` twin** — `crates/wukong_codegen_cranelift/src/lib.rs:2723-2731` — vhorner, vmath2 and attention are imported serial-only while every sibling family declares a `_parallel` variant, capping all-core dispatch for those ops; also `crates/wukong_codegen_cranelift/src/lib.rs:2794-2796`.
- **Declaration error-handling divergence: three `.unwrap()` sites** — `crates/wukong_codegen_cranelift/src/lib.rs:2803-2811` — The three axpby bf16/f16 declares panic on failure while every neighbouring declare returns `Err` via `map_err`, an inconsistent failure mode.
- **User-function pre-declaration pass** — `crates/wukong_codegen_cranelift/src/lib.rs:2867-2876` — Declares every MIR function as `Linkage::Export` with `signature_of` before any body is built so calls resolve regardless of definition order.
- **Read-only static data blobs for string literals** — `crates/wukong_codegen_cranelift/src/lib.rs:2878-2892` — Defines each `program.statics` blob as non-writable `Linkage::Local` data including the trailing NUL, giving returned `*u8` a real `.rodata` address instead of a dead stack frame.
- **Serial-vs-parallel codegen gate** — `crates/wukong_codegen_cranelift/src/lib.rs:2930-2938` — Chooses `define_functions_parallel` only when the caller requested parallelism and the program has more than one function, else the serial path.
- **Per-function CLIF construction driver** — `crates/wukong_codegen_cranelift/src/lib.rs:2950-2965` — Builds one MIR function into a complete Cranelift IR function against a read-only `Decls` view, then runs `FnTranslator::translate` and finalizes; also `crates/wukong_codegen_cranelift/src/lib.rs:3637-3658`.
- **Deterministic FuncRef / GlobalValue import ordering** — `crates/wukong_codegen_cranelift/src/lib.rs:2970-2991` — Imports callees, statics and vec-kernels by iterating the source-order Vecs rather than the HashMaps, keeping FuncRef indices and hence the CLIF reproducible.
- **`rt_refs` string-keyed runtime FuncRef table** — `crates/wukong_codegen_cranelift/src/lib.rs:2992-3635` — Imports all ~200 runtime symbols into each function's DFG keyed by their `&'static str` name, the lookup surface `FnTranslator` uses to emit any kernel call.
- **Layout-overflow reported as a clean error** — `crates/wukong_codegen_cranelift/src/lib.rs:3652-3657` — A type whose byte layout overflows `u32` sets `layout_err` during translation and aborts before the half-built function is defined, instead of panicking.
- **`Decls`: Sync read-only view of module declarations** — `crates/wukong_codegen_cranelift/src/lib.rs:3664-3713` — Re-implements `declare_func_in_func` / `declare_data_in_func` verbatim over `ModuleDeclarations` so CLIF building borrows the module immutably and can run on rayon workers.
- **Serial function definition (two-phase build-then-define)** — `crates/wukong_codegen_cranelift/src/lib.rs:3715-3756` — Builds every function's CLIF while borrowing `declarations()`, then compiles and defines each in source order through the module's own `define_function`.
- **Parallel per-function codegen with in-order definition** — `crates/wukong_codegen_cranelift/src/lib.rs:3758-3824` — Rayon `map_init` gives each worker its own builder/context, compiles to owned bytes plus `ModuleReloc`s with a default `ControlPlane`, then defines them serially for byte-identical objects.
- **`JitProgram` handle: locked run, output capture, memory ownership** — `crates/wukong_codegen_cranelift/src/lib.rs:3826-3882` — Holds the JIT module and entry code pointer, serializes runs on `RUN_LOCK`, clears/clones the shared `OUTPUT` buffer, turns `ASSERT_FAILED` into an `Err`, and frees code memory on drop.
- **Entry-point return-ABI transmute** — `crates/wukong_codegen_cranelift/src/lib.rs:3884-3910` — Dispatches on the MIR return type to transmute the code pointer to a void/f64/i64/i32 `extern "C"` fn; float entries are truncated via `f() as i64`.
- **512 MiB big-stack worker for native runs** — `crates/wukong_codegen_cranelift/src/lib.rs:3912-3931` — Runs JIT'd code on a scoped thread with interpreter-matching stack headroom and re-raises panics, so deep recursion does not diverge between backends.
- **`jit_compile` runtime symbol binding table** — `crates/wukong_codegen_cranelift/src/lib.rs:3934-4457` — Roughly 250 `builder.symbol` registrations bind every `RT_*` name to a real `wukong_runtime` (or local shim) function pointer for the non-PIC JIT module.
- **JIT never uses parallel codegen** — `crates/wukong_codegen_cranelift/src/lib.rs:4457-4460` — Both JIT entry points hard-code `parallel = false` into `populate_module`, so the rayon codegen path benefits object emission only; also `crates/wukong_codegen_cranelift/src/lib.rs:5032-5034`.
- **JIT entry-point contract and return-type capture** — `crates/wukong_codegen_cranelift/src/lib.rs:4462-4476` — Rejects an entry with parameters (freeing JIT memory first) and stores the entry's MIR return type so `invoke_code` picks the right ABI.
- **`jit_run` one-shot compile-and-execute** — `crates/wukong_codegen_cranelift/src/lib.rs:4479-4486` — Thin wrapper composing `jit_compile` and `JitProgram::run`, the exact shape the differential gate compares against the interpreter.
- **`JitModuleHandle` raw function-pointer access** — `crates/wukong_codegen_cranelift/src/lib.rs:4488-4510` — Exposes any compiled function's finalized code pointer by symbol for callers that know the ABI (kernel timing, fuzzing), valid until the handle drops.
- **`jit_module`: duplicated 500-line symbol table** — `crates/wukong_codegen_cranelift/src/lib.rs:4513-5039` — Compiles all functions for pointer access but re-lists every runtime symbol binding a second time, so any new kernel must be added in two places or silently fails to resolve here.
- **`EmitOptions` byte-identity compile-time levers and their env knobs** — `crates/wukong_codegen_cranelift/src/lib.rs:5041-5063` — Verifier and parallel-codegen toggles asserted to never change emitted bytes, defaulted by `WUKONG_CL_VERIFY` and the `WUKONG_PAR_CODEGEN=0` kill-switch; also `crates/wukong_codegen_cranelift/src/lib.rs:2047-2061`.
- **Object emission entry points (PIC, timed)** — `crates/wukong_codegen_cranelift/src/lib.rs:5077-5120` — Builds a PIC ISA and `ObjectModule`, populates it, and emits object bytes with runtime symbols left as undefined imports for the external link step.
**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Guard-page probestack gate for large stack frames** — `crates/wukong_codegen_cranelift/src/tests.rs:438-452` — A ~200 KB local array must run without faulting at shallow stack depth and still match the interpreter at -O0/2/3.
- **Object-byte identity and determinism gates for emit levers** — `crates/wukong_codegen_cranelift/src/tests.rs:4833-4907` — Asserts serial-vs-parallel per-function codegen, the IR-verifier toggle and eight repeated parallel emits all produce bit-identical object bytes over a five-program corpus, extra site `crates/wukong_codegen_cranelift/src/tests.rs:4810-4831`.
**`crates/wukong_runtime/src/lib.rs`**

- **C-ABI f16 narrow/widen shims for Cranelift** — `crates/wukong_runtime/src/lib.rs:222-241` — `wukong_f32_to_f16_bits` / `wukong_f16_bits_to_f32` exist because Cranelift has no f16 promote/demote, so every native f16 store/load is a call.

## G19 — Cranelift: raw AVX2 encoder and backend fuzzing

*Files:* `crates/wukong_codegen_cranelift/src/avx2.rs`, `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`

**`crates/wukong_codegen_cranelift/src/avx2.rs`**

- **vcmpps predicate table matching Rust float comparison NaN semantics** — `crates/wukong_codegen_cranelift/src/avx2.rs:51-61` — Maps Eq/Lt/Le/Ge/Gt to ordered-signalling imm8s but Ne to NEQ_UQ so `!=` is true on NaN, keeping the machine code bit-equal to the interpreter oracle.
- **AVX2 emitter coverage gate / bail-out contract** — `crates/wukong_codegen_cranelift/src/avx2.rs:83-116` — Rejects bodies with >4 streams, an unfolded `Const`, or peak-live+hoist exceeding 16 YMM regs so the caller keeps the 128-bit CLIF fallback.
- **YMM register-allocation plan (dense low-first assignment)** — `crates/wukong_codegen_cranelift/src/avx2.rs:118-153` — Lays out unroll×per_group body banks, then reduction accumulators, then hoisted broadcasts, and derives the Win64 callee-saved xmm6..15 spill set from the highest register touched.
- **Kernel ABI plumbing: stream base GPRs, prologue/epilogue, vzeroupper** — `crates/wukong_codegen_cranelift/src/avx2.rs:164-179` — Spills used callee-saved xmm halves with `vmovups` (no shadow space, no stack alignment need), loads stream bases into r9/r10/r11, and ends `vzeroupper; ret`; also `crates/wukong_codegen_cranelift/src/avx2.rs:45-49`, `crates/wukong_codegen_cranelift/src/avx2.rs:209-213`, `crates/wukong_codegen_cranelift/src/avx2.rs:263-272`.
- **Loop-invariant scalar broadcast and in-register constant synthesis** — `crates/wukong_codegen_cranelift/src/avx2.rs:181-189` — Hoists each distinct scalar via `vbroadcastss` from `[rdx]` and builds the -0.0 negation sign mask with `vpcmpeqd`+`vpslld` instead of a data section.
- **Reduction pipeline: identity init, per-copy fold, ordered horizontal fold** — `crates/wukong_codegen_cranelift/src/avx2.rs:190-208` — Seeds accumulators with 0/−∞/+∞ built in-register, folds each unroll copy (fusing `vfmadd231ps` when the recipe marks an fma product), then spills 8 lanes and folds them sequentially into xmm0; also `crates/wukong_codegen_cranelift/src/avx2.rs:245-261`, `crates/wukong_codegen_cranelift/src/avx2.rs:397-431`.
- **Two-tier strip-mined loop structure with unsigned trip test** — `crates/wukong_codegen_cranelift/src/avx2.rs:216-243` — Emits an unrolled main loop (step unroll×8) falling into a single-vector cleanup loop, each guarded by `lea rax,[i+step]; cmp rax,r8; ja`, leaving the scalar tail to the caller.
- **Hazard-free free-list register allocator inside a lane group** — `crates/wukong_codegen_cranelift/src/avx2.rs:283-330` — Allocates the result register before freeing operands whose last use is this op, so a result can never alias a still-live operand even across the two-instruction FMA.
- **AVX2 instruction repertoire for recipe ops** — `crates/wukong_codegen_cranelift/src/avx2.rs:319-395` — Lowers Load/Store (`vmovups` base+rcx*4+disp), Bin (`vaddps`/`vsubps`/`vmulps`/`vdivps`), `vsqrtps`, `vxorps` negation, `vcmpps`, and `vblendvps` select with deliberately swapped src1/src2.
- **saxpy bring-up probe kernel and raw-bytes install proof** — `crates/wukong_codegen_cranelift/src/avx2.rs:433-466` — Hand-encodes `a*x+y` with `vfmadd213ps` to prove the assemble → `define_function_bytes` → finalize → call seam works before the recipe compiler exists; also `crates/wukong_codegen_cranelift/src/avx2.rs:477-525`.
- **Differential oracle harness: assembled kernel vs VecKernel::eval_lane / eval_reduction** — `crates/wukong_codegen_cranelift/src/avx2.rs:530-579` — JIT-installs the raw bytes and asserts every full-vector lane (and the reduction's f32 return) is bit-for-bit equal to the shared backend-agnostic reference; also `crates/wukong_codegen_cranelift/src/avx2.rs:584-614`.
- **AVX2 recipe test corpus (elementwise, reduction, aliasing, bail-out)** — `crates/wukong_codegen_cranelift/src/avx2.rs:622-988` — Covers sum/dot/fused-dot/sumsq/weighted/fmax/fmin reductions, saxpy, diff-of-squares, sqrt-div NaN/Inf, store-to-load forwarding through memory, negation, relu, in-place, 4-stream rdx base, and the 5-stream bail.
**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **VecKernelCall bridge to raw-AVX2 synthesized kernels** — `crates/wukong_codegen_cranelift/src/lib.rs:705-732` — Calls per-function assembled 256-bit AVX2 kernels through pre-declared FuncRefs with a (ptrs, scalars, n:i64) ABI, binding an f32 result only for reduction kernels.
- **Synthesized AVX2 vector-kernel installation as raw machine bytes** — `crates/wukong_codegen_cranelift/src/lib.rs:2894-2928` — Assembles each function's `vec_kernels` via `avx2::assemble_kernel` and installs them with `define_function_bytes` (align 16, no relocs) under a `(ptrs, scalars, n)` signature, f32-returning when the kernel reduces.
**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Cranelift 256-bit legalization tripwire and op probe** — `crates/wukong_codegen_cranelift/src/tests.rs:210-262` — Builds an `f32x8` function and asserts Cranelift still rejects it (the justification for `VEC_REG_BYTES=16` and the raw-AVX2 emitter); the companion probe sweeps fourteen 256-bit ops under `catch_unwind`, extra site `crates/wukong_codegen_cranelift/src/tests.rs:268-432`.
**`crates/wukong_mir_build/src/lib.rs`**

- **256-bit dispatch gate: WUKONG_P4_NO_256 kill-switch and reduction trip threshold** — `crates/wukong_mir_build/src/lib.rs:13269-13292` — Tries the AVX2 recipe only for f32 lanes when the env var is unset, and gates the 256-bit reduction on a compile-time-known trip ≥ 2048; also `crates/wukong_mir_build/src/lib.rs:13697-13710`, `crates/wukong_mir_build/src/lib.rs:24513-24513`, `crates/wukong_codegen_cranelift/src/lib.rs:2894-2928`.

## G20 — Interpreter

*Files:* `crates/wukong_interp/src/lib.rs`

**`crates/wukong_interp/src/lib.rs`**

- **Runtime `Value` representation and coercions** — `crates/wukong_interp/src/lib.rs:16-48` — Five-variant tagged value (i128 int, f64 float, slot-index Ptr, VecRef handle, Unit) with lossy `as_int`/`as_float`/`truthy` coercions used by every op.
- **Backend impl and public run entry points** — `crates/wukong_interp/src/lib.rs:55-72` — Implements `Backend::compile` as immediate execution returning `Artifact::Executed`, with `run`/`run_with_output` building a fresh `Interp` per program and returning exit code plus captured stdout, also `crates/wukong_interp/src/lib.rs:141-176`.
- **512 MiB worker-thread stack for deep recursion** — `crates/wukong_interp/src/lib.rs:178-196` — Runs the host-stack-recursive tree-walker in a scoped thread with a reserved 512 MiB stack and re-raises panics, so deep Wukong recursion hits the step guard instead of aborting.
- **Typed f32/f64 buffer kernel-entry ABI** — `crates/wukong_interp/src/lib.rs:226-364` — Copies caller slices into flat memory, calls the kernel with one pointer per parameter, then copies slots back, enabling full-output-buffer differential fuzzing against the native backend.
- **int8 kernel-entry ABI** — `crates/wukong_interp/src/lib.rs:366-425` — Lays u8/i8/i32 buffers contiguously with zero/sign-extension into `Value::Int` slots, runs a three-pointer quantized GEMM kernel, and reads i32 results back for the int8 fuzzer.
- **Interpreter state, register-file pooling and frame entry** — `crates/wukong_interp/src/lib.rs:427-464` — `Interp` holds flat slot memory, a recycled per-depth register-file pool, an edge-argument scratch vector, the vector arena and static-blob address cache; `run_function` resizes and binds params.
- **Block-walking exec loop, 100M step guard and terminators** — `crates/wukong_interp/src/lib.rs:466-531` — Executes instructions per block with a 100-million-step infinite-loop guard, handling Ret/Br/CondBr/Unreachable and snapshot-before-write block-argument passing at `crates/wukong_interp/src/lib.rs:3857-3871`.
- **Per-result width normalization with the `Load` exemption** — `crates/wukong_interp/src/lib.rs:476-504` — Masks integer results to their declared MIR width and rounds sub-f64 floats to f32, but exempts `Op::Load` so byte-wise aggregate copies don't truncate wider payload slots.
- **Lane-wise vector evaluation and the vector side-arena** — `crates/wukong_interp/src/lib.rs:552-649` — Bin/Cmp/Neg/Not/Cast/Select/Splat/Fma/Sqrt/Round all branch on a `Vec` result type to evaluate per lane with lane-type rounding, backed by `push_vec`/`vec_lanes` at `crates/wukong_interp/src/lib.rs:848-916`.
- **Slot-indexed memory model: alloca, slot_count and Gep striding** — `crates/wukong_interp/src/lib.rs:650-657` — Alloca pushes one typed-zero slot per scalar leaf (recursing arrays) and Gep strides `base + index*slot_count(elem)`, mirroring native byte sizing, also `crates/wukong_interp/src/lib.rs:705-713`, `crates/wukong_interp/src/lib.rs:3890-3917`.
- **Load/Store including vector gather/scatter and half round-on-read** — `crates/wukong_interp/src/lib.rs:658-704` — Scalar loads return the slot verbatim (bf16/f16 re-rounded on read), vector loads gather n contiguous slots into a VecRef, vector stores scatter lanes back.
- **VecKernelCall: interpreting synthesized AVX2 kernels** — `crates/wukong_interp/src/lib.rs:724-812` — Resolves stream base pointers and invariant scalars from memory, then replays the vectorizer's recipe element-by-element (`eval_lane`) or as a pinned-reassociation reduction (`eval_reduction`).
- **Function addresses via FUNC_TAG and the sequential parallel_for vehicle** — `crates/wukong_interp/src/lib.rs:813-823` — `FuncAddr` returns `Ptr(FUNC_TAG + func_index)`; `wukong_parallel_for` decodes it and runs the whole index range in one serial body call, also `crates/wukong_interp/src/lib.rs:50-53`, `crates/wukong_interp/src/lib.rs:1212-1230`.
- **GlobalAddr static-blob materialization with address caching** — `crates/wukong_interp/src/lib.rs:824-847` — Materializes a string literal's bytes into never-freed memory once and caches the base by symbol, so pointer equality and post-frame validity match native `.rodata`.
- **Print/assert host intrinsics including `*u8` and unsigned forms** — `crates/wukong_interp/src/lib.rs:918-977` — print/println, print_u (low 64 bits as u64) and print_str (NUL-scan through slot memory, UTF-8 lossy) append to captured stdout; failed `assert` returns an error.
- **Heap alloc/free model and the monotonic clock intrinsic** — `crates/wukong_interp/src/lib.rs:978-1010` — `wukong_rt_alloc` appends `count` typed zeros to the arena (float or int zero per flag), `wukong_rt_free` is a deliberate no-op, and `wukong_now_ns` calls the same runtime clock native links.
- **File-I/O intrinsic family (typed little-endian blobs, read and write)** — `crates/wukong_interp/src/lib.rs:1011-1211` — Six read and six write widths (f32/i32/i64/u8/f64/i8) reconstruct the NUL-terminated path from memory, decode/encode headerless LE elements, and return -1 open-fail, -2 io-fail, else element count.
- **f32 GEMM marshalling family (NN/NT/TN, alpha, fused epilogue)** — `crates/wukong_interp/src/lib.rs:1231-1328` — Copies slot memory into real f32 buffers, calls the same serial `wukong_runtime` GEMM the native backend uses (accelerator first for NT beta=0), and writes C back, also `crates/wukong_interp/src/lib.rs:1701-1827`.
- **GEMV and GEVM marshalling** — `crates/wukong_interp/src/lib.rs:1329-1458` — `sgemv`, α-scaled `sgemv_alpha` and the vector·matrix `sgevm_f32` all marshal operands, call the serial runtime kernel (claimed bit-identical to parallel), and write the result vector back.
- **int8 GEMM, fused dequant GEMM and dequant kernels** — `crates/wukong_interp/src/lib.rs:1459-1700` — Recovers exact u8/i8 bytes from `Value::Int` low bits, handles per-column scales plus an optional bias distinguished by `Ptr`-vs-`Int(0)` variant, and rebuilds width-matched raw byte buffers for the DQ_* dequant kernels.
- **Elementwise kernel marshalling: vmath, vmath2, velem, bias_bcast, vhorner** — `crates/wukong_interp/src/lib.rs:1828-1876` — Reads all inputs before writing outputs (making in-place x==out correct) and calls the identical runtime kernels, also `crates/wukong_interp/src/lib.rs:2099-2143`, `crates/wukong_interp/src/lib.rs:2833-2988`.
- **Loss kernel marshalling: cross-entropy, KL/KD, logsumexp, entropy** — `crates/wukong_interp/src/lib.rs:2320-2415` — Softmax cross-entropy forward/backward read i32 labels from int slots; KL/KD share an (a,b,out,rows,cols) shape and logsumexp/entropy share (x,out,rows,cols), also `crates/wukong_interp/src/lib.rs:2654-2705`, `crates/wukong_interp/src/lib.rs:2510-2551`.
- **Scan marshalling: cumsum/cummax/cummin/cumprod and linear recurrence** — `crates/wukong_interp/src/lib.rs:2603-2653` — Per-row inclusive prefix scans select a runtime kernel by name prefix and buffer all input first for in-place safety; the first-order selective scan is at `crates/wukong_interp/src/lib.rs:2190-2235`.
- **Reduction and arg-reduction marshalling** — `crates/wukong_interp/src/lib.rs:2989-3058` — `sreduce` (dot/ssd/sum, accelerator-capable) and `argreduce` return scalars; row/col argmax/argmin write i32 index buffers and the eight column reductions write per-column floats, also `crates/wukong_interp/src/lib.rs:2552-2602`, `crates/wukong_interp/src/lib.rs:2032-2098`.
- **bf16/f16 exact-bit reconstruction discipline across half kernels** — `crates/wukong_interp/src/lib.rs:3059-3347` — Half values live as their rounded f32, so every half kernel re-derives exact u16 bits via `f32_to_{bf16,f16}_bits`; covers reductions, axpby, half-output arms, vmath halves and const materialization at `crates/wukong_interp/src/lib.rs:541-549`, `crates/wukong_interp/src/lib.rs:2706-2832`.
- **Low-precision GEMM and fused-epilogue GEMM marshalling** — `crates/wukong_interp/src/lib.rs:3348-3530` — bf16/f16 NT and TN GEMMs plus their bias+activation epilogue variants share one u16 marshalling path, branching only on which runtime kernel widens the bits.
- **Fused norm marshalling: softmax/LayerNorm/RMSNorm, affine, and backwards** — `crates/wukong_interp/src/lib.rs:3531-3673` — Plain and affine norms pass eps as raw f32 bits and null gamma/beta by value-variant; the shared 7-arg RMS/LayerNorm backward and softmax backward arms sit at `crates/wukong_interp/src/lib.rs:2236-2319`, `crates/wukong_interp/src/lib.rs:2144-2189`.
- **Embedding gather with derived table extent, and scatter-add backward** — `crates/wukong_interp/src/lib.rs:3674-3737` — Because the recognizer passes a 2^48 vocab sentinel, the interpreter derives `v_eff = max(id)+1` to bound weight marshalling; scatter-add pre-loads grad_w so accumulation is in place, `crates/wukong_interp/src/lib.rs:3738-3797`.
- **ML-op marshalling: attention, RoPE, transpose, pool2d** — `crates/wukong_interp/src/lib.rs:3798-3847` — Flash attention, RoPE forward/backward, blocked f32 transpose, the precision-agnostic u16 transpose done as a pure Value permutation, and max/avg pool with output-length guards, also `crates/wukong_interp/src/lib.rs:2416-2509`, `crates/wukong_interp/src/lib.rs:1877-2031`.
- **Scalar integer/float operation semantics matched to Cranelift** — `crates/wukong_interp/src/lib.rs:4023-4138` — Wrapping add/sub/mul, divide-by-zero yielding 0, width-masked shift counts, logical `LShr` over the unsigned width window, and signed/unsigned compares; helpers `mask`/`int_bits`/`uval` at `crates/wukong_interp/src/lib.rs:3919-3973`.
- **Cast semantics: single-rounded int→float, saturating fp→int, half grids, bitcast** — `crates/wukong_interp/src/lib.rs:4140-4214` — ZExt reads the source's own width, SiToFp/UiToFp round once via `int_to_float`, FpToSi/FpToUi saturate like `fcvt_*_sat`, and FpExt/FpTrunc round bf16/f16 at the observation boundary.
- **In-crate oracle unit tests** — `crates/wukong_interp/src/lib.rs:4216-4416` — Twelve tests covering loops, recursive fib, 1000-deep recursion (the big-stack guard), stdout capture, the f32 and int8 kernel ABIs, step loops, array parameters, casts and failing asserts.

## G21 — Driver, CLI (wukongc), module loader, emit modes, LLVM textual backend

*Files:* `crates/wukong_backend/src/lib.rs`, `crates/wukong_codegen_llvm/src/lib.rs`, `crates/wukong_driver/src/lib.rs`, `crates/wukong_driver/src/loader.rs`, `crates/wukong_runtime/src/lib.rs`, `crates/wukongc/src/main.rs`

**`crates/wukong_backend/src/lib.rs`**

- **`Backend`/`Artifact` seam** — `crates/wukong_backend/src/lib.rs:12-35` — Declares the trait the doc claims makes the driver backend-agnostic, but the driver never references it and calls concrete backend functions directly instead.
**`crates/wukong_codegen_llvm/src/lib.rs`**

- **Block-parameter SSA to LLVM phi bridge** — `crates/wukong_codegen_llvm/src/lib.rs:29-60` — Collects per-target incoming `(pred, args)` edges, collapsing a `CondBr` whose arms share a target into one predecessor; consumed by `crates/wukong_codegen_llvm/src/lib.rs:195-212`.
- **Textual LLVM IR module emission with `.rodata` string constants** — `crates/wukong_codegen_llvm/src/lib.rs:84-157` — Emits private byte-escaped constants for statics then each function signature/body; deliberately text IR, not libLLVM, to avoid a build-time dependency.
- **LLVM instruction lowering including Splat, Fma, Sqrt, Round and VecKernelCall** — `crates/wukong_codegen_llvm/src/lib.rs:214-418` — Inlines constants as operands and emits multi-line expansions for Splat/Fma; intrinsic `declare`s for sqrt/round are knowingly omitted so the text is not assemblable as-is.
**`crates/wukong_driver/src/lib.rs`**

- **Emit-stage taxonomy and `--emit` string parsing** — `crates/wukong_driver/src/lib.rs:22-53` — Enumerates tokens/ast/mir-high/mir/grad/llvm-ir/obj/exe and maps CLI spellings (`mir-low` aliases `mir`) to the stage the pipeline stops at.
- **`compile()` pipeline orchestration with per-stage early exits** — `crates/wukong_driver/src/lib.rs:169-374` — Sequences read/lex/parse/import-load/sema/mir_build/optimize/backend, deliberately dumping `tokens` and `ast` for the ROOT file only before imports are spliced.
- **Autodiff forces at least -O1 regardless of requested level** — `crates/wukong_driver/src/lib.rs:296-313` — `--emit=grad`/`--train` bump `opt_level` to 1 because the transform requires single-block SSA input, then divert before any backend runs.
- **Pre-backend MIR verifier gate** — `crates/wukong_driver/src/lib.rs:316-331` — Verifies every function before `--run` on any backend so an invalid -O0 lowering is a diagnosable ICE rather than a native SIGSEGV; also `crates/wukong_driver/src/lib.rs:996-1006`.
- **Backend selection and honest GPU capability errors** — `crates/wukong_driver/src/lib.rs:332-434` — Routes `--run` to interp/Cranelift-JIT/GPU-offload/GPU-lower, and without the `gpu` feature returns a build-capability error instead of silently falling back to CPU.
- **Native object/exe emission with rustc-then-`cc` link ladder** — `crates/wukong_driver/src/lib.rs:467-560` — Writes the Cranelift object, tries the rustc link, and only on `Unavailable` emits `WUKONG_RT_C` (`crates/wukong_driver/src/lib.rs:439-463`) and links via `$CC`.
- **rustc-driven link path and the generated `#![no_main]` Rust runtime shim** — `crates/wukong_driver/src/lib.rs:639-686` — Locates `libwukong_runtime.rlib` next to the compiler exe and links through rustc so kernel symbols and `.rodata` relocations resolve; shim at `crates/wukong_driver/src/lib.rs:578-630`.
- **`--emit=grad` surface and `--grad-wrt` validation** — `crates/wukong_driver/src/lib.rs:696-757` — Resolves the loss function by name, validates requested indices are in-range pointer params, and defaults to every buffer param when the list is empty.
- **Buffer element-count recovery from semantic types for `--train`** — `crates/wukong_driver/src/lib.rs:763-805` — Reads `[f32;N]`/`Tensor` shapes out of the sema signature before params erase to `Ptr`, rejecting symbolic dims and slices with a clear error.
**`crates/wukong_driver/src/loader.rs`**

- **Multi-file import loader: dotted-path resolution, DFS splice, cycle dedup** — `crates/wukong_driver/src/loader.rs:38-148` — Resolves `import a.b.c` to `<root-dir>/a/b/c.wk`, dedups by canonicalized path so cycles/diamonds load once, splices items into one flat namespace, emits E0305 per missing file.
**`crates/wukong_runtime/src/lib.rs`**

- **`rt_path` C-string path decoder** — `crates/wukong_runtime/src/lib.rs:536-551` — Turns the intrinsics' NUL-terminated `*const u8` into a `PathBuf`, returning `None` (reported as `-1`) for null or non-UTF-8 so a wrong file is never opened.
- **Typed file-I/O intrinsic family via `rt_file_io!`** — `crates/wukong_runtime/src/lib.rs:553-644` — Macro generating headerless little-endian read/write pairs with a frozen `-1`/`-2`/`n` return contract and `min(len, file_size/sizeof)` truncation, instantiated for six element types at `crates/wukong_runtime/src/lib.rs:646-651`.
**`crates/wukongc/src/main.rs`**

- **`wukongc` CLI argument parser, usage text, big-stack thread, and `--explain`** — `crates/wukongc/src/main.rs:8-234` — Hand-rolled flag parsing for every emit/backend/opt/train/color/error-format knob; `emit_explicit` is tracked then discarded at line 232, a dead flag.
- **256 MB compile worker thread** — `crates/wukongc/src/main.rs:66-75` — Runs the whole compile off the main thread so deeply nested but legal input hits the parser's E0209 limit instead of silently overflowing the stack.

# CPU runtime

## G22 — Runtime: GEMM / GEMV / GEVM core

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_runtime/src/gemm.rs`, `crates/wukong_runtime/src/gemv.rs`, `crates/wukong_runtime/src/gevm.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **GEMM/GEMV/GEVM call-lowering family** — `crates/wukong_codegen_cranelift/src/lib.rs:1076-1181` — Routes ~30 sgemm variants (NN/NT/TN, bf16/f16, epilogue, α-scaled, parallel) plus gemv/gevm by symbol name and arity onto four shared signatures.
**`crates/wukong_runtime/src/gemm.rs`**

- **BLIS blocking constants (MR/NR/MC/KC/NC)** — `crates/wukong_runtime/src/gemm.rs:18-31` — Register tile 6×16 and cache blocks MC=144, KC=384, NC=4080, each justified by measured L1/L2/L3 residency; every kernel and scratch size derives from them.
- **Size-adaptive K-block splitter `select_kc`** — `crates/wukong_runtime/src/gemm.rs:77-94` — Splits K into the fewest equal-ish 4-aligned blocks ≤ KC so tails stay fat; all kernels call it so serial and parallel group K identically.
- **Fused `Epilogue` value type: alpha, bias, activation codes** — `crates/wukong_runtime/src/gemm.rs:393-456` — Carries `act(alpha·x + bias[j])` with identity/ReLU/GELU/SiLU codes, a scalar `apply` reusing vmath, and `shift` for column-origin rebasing across blocks.
- **Public C-ABI GEMM entry family (nn/nt/nt_epi/nt_alpha × serial/parallel)** — `crates/wukong_runtime/src/gemm.rs:458-577` — Eight `#[no_mangle]` symbols the compiler's matmul recognizer lowers to, all funneling into one dispatcher with beta/epilogue flags; parallel twins at `crates/wukong_runtime/src/gemm.rs:683-741`.
- **Central dispatcher `gemm_dispatch` (feature, size and path selection)** — `crates/wukong_runtime/src/gemm.rs:579-673` — Rejects non-positive dims, runtime-detects AVX2+FMA, applies the MAC gate, short-circuits 1-worker pools to serial, then routes among the five parallel shapes.
- **TN weight-gradient GEMM `C = Aᵀ·B` via transpose prepass** — `crates/wukong_runtime/src/gemm.rs:743-812` — O(m·k) `transpose_into` scratch pass then reuse of the tuned NN kernel, so backward `dW = dYᵀ·X` inherits bit-exact accumulation order.
- **Legacy fork-join parallel kernel `sgemm_avx2_parallel`** — `crates/wukong_runtime/src/gemm.rs:1165-1277` — Per-K-block parallel pack plus a row-panel `par_iter` compute region, retained behind `WUKONG_GEMM_FORKJOIN=1` as the adjacent-run A/B baseline for newer shapes.
- **Persistent broadcast region path (`GemmRegion`, barrier phases)** — `crates/wukong_runtime/src/gemm.rs:1304-1412` — One `broadcast` per call with atomic pack/compute claim counters and in-region barriers replacing per-block fork-joins; soundness rests on documented rayon-core internals; worker at `crates/wukong_runtime/src/gemm.rs:1414-1531`.
- **2D block-shape policy `select_2d_block_shape`** — `crates/wukong_runtime/src/gemm.rs:1580-1656` — MR/NR-aligned candidate ladder plus a skinny-M single-row rule and tail rebalancing, choosing L2-resident C blocks; pinned by `crates/wukong_runtime/src/gemm.rs:3850-3896`.
- **Static range-split 2D block path `sgemm_2d_blocks`** — `crates/wukong_runtime/src/gemm.rs:1658-1770` — One rayon task per C block with per-worker packing and no barriers, the `WUKONG_GEMM_DYN=0` baseline; per-block whole-K body at `crates/wukong_runtime/src/gemm.rs:1772-1824`.
- **Skinny shared-A regime (single block row, call-shared packed A)** — `crates/wukong_runtime/src/gemm.rs:1826-1866` — Packs A once per call when nbi==1 and under 8 MB, refuted as a win but kept as an instrument; buffer/cap at `crates/wukong_runtime/src/gemm.rs:1560-1578`, pack phase at `crates/wukong_runtime/src/gemm.rs:2008-2047`.
- **False-sharing-isolated claim counter `PaddedCounter`** — `crates/wukong_runtime/src/gemm.rs:1868-1874` — 128-byte-aligned atomic wrapper so the single cross-worker write of the dynamic path never shares a line with caller stack data.
- **LPT steal-order block enumeration `dyn_block_order`** — `crates/wukong_runtime/src/gemm.rs:1876-1895` — Optional permutation listing full blocks before edge remainders so the join straggle is bounded by a small tail; permutation invariant tested at `crates/wukong_runtime/src/gemm.rs:3922-3956`.
- **Dynamic atomic claim-queue scheduler `sgemm_2d_blocks_dyn` (default)** — `crates/wukong_runtime/src/gemm.rs:1897-2140` — One relaxed `fetch_add` per C block gives block-granular self-scheduling on the asymmetric P/E box, halving straggler variance while keeping bit-exactness.
- **Retired shared-pack 2D path `sgemm_2d_shared`** — `crates/wukong_runtime/src/gemm.rs:2142-2292` — Packs whole-matrix A/B panels once per K-block into caller scratch, then one task per C block over contiguous slices; now instrument-only after per-block packing won.
- **Portable scalar fallback `sgemm_scalar`** — `crates/wukong_runtime/src/gemm.rs:2294-2336` — Triple-loop reference used when AVX2/FMA is absent, handling bt, beta and the epilogue with a different (unblocked) accumulation order than the tuned path.
- **Serial five-loop AVX2 kernel `sgemm_avx2` with reusable pack scratch** — `crates/wukong_runtime/src/gemm.rs:2338-2420` — The jc/pc/ic loop nest applying beta on the first K-block and the epilogue on the last; thread-local buffer reuse at `crates/wukong_runtime/src/gemm.rs:381-391`.
- **Panel packing routines (A, B, Bᵀ) with contiguous-read ordering and edge zero-pad** — `crates/wukong_runtime/src/gemm.rs:2422-2558` — Produce `[kc][MR]`/`[kc][NR]` micropanels; the contraction-innermost read order is what makes the transposed nn.Linear operand affordable.
- **Macro-kernel tile sweep with per-column epilogue shifting** — `crates/wukong_runtime/src/gemm.rs:2685-2725` — Iterates NR-column then MR-row panels of a macro block, rebasing the bias pointer per column panel so the fused epilogue lands on the right columns.
- **AVX2/FMA `micro_6x16` microkernel: 12 accumulators, ×4 K-unroll, B and C prefetch** — `crates/wukong_runtime/src/gemm.rs:2727-2846` — Load-broadcast A against two 256-bit B lanes, unrolled fourfold with one B prefetch per group, plus a vector full-tile writeback fast path.
- **Vectorized fused-epilogue writeback and scalar edge-tile path** — `crates/wukong_runtime/src/gemm.rs:2848-2939` — Full tiles apply beta, alpha, bias and gelu8/silu8 entirely in 256-bit lanes; partial or unmatched tiles spill to a 6×16 buffer and apply the scalar epilogue.
- **AVX-512 microkernel twin `micro_6x16_avx512`** — `crates/wukong_runtime/src/gemm.rs:2941-3029` — Six zmm accumulators halving the FMA count for full no-epilogue tiles, argued bit-identical by lane-width equivalence and dead code on this box; twin test at `crates/wukong_runtime/src/gemm.rs:3281-3325`.
**`crates/wukong_runtime/src/gemv.rs`**

- **GEMV per-row dot family (4-accumulator AVX2 + scalar twin + feature dispatch)** — `crates/wukong_runtime/src/gemv.rs:21-106` — Folds one row 32 elements/step across four independent `_mm256_fmadd_ps` chains, then a fixed hsum tree and scalar tail; scalar fallback at `crates/wukong_runtime/src/gemv.rs:85-91` uses a different (non-bit-equal) order.
- **`wukong_sgemv` / `wukong_sgemv_alpha` entry points with α-on-store** — `crates/wukong_runtime/src/gemv.rs:112-158` — Row-major `[M,N]·[N]` dispatch symbols; the α form applies exactly one multiply per output element and skips it entirely when `alpha == 1.0`.
**`crates/wukong_runtime/src/gevm.rs`**

- **GEVM i-outer restructured column fold (AVX2 broadcast-FMA + scalar `mul_add` twin)** — `crates/wukong_runtime/src/gevm.rs:41-141` — Rewrites the strided `wᵀ·A` reduction as row-streaming `out[j..] += w[i]·A[i,j..]`, 8 columns/step, so lanes, tail and scalar path are bit-equal.
- **`wukong_sgevm_f32` serial entry and single-pass α finalize** — `crates/wukong_runtime/src/gevm.rs:143-162` — Public vector-times-matrix symbol whose α scale is applied once per column after its full fold, skipped for `alpha == 1.0`, at `crates/wukong_runtime/src/gevm.rs:24-39`.

## G23 — Runtime: elementwise, vmath, velem, scans

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_runtime/src/bias.rs`, `crates/wukong_runtime/src/cumminmax.rs`, `crates/wukong_runtime/src/cumprod.rs`, `crates/wukong_runtime/src/cumsum.rs`, `crates/wukong_runtime/src/lrscan.rs`, `crates/wukong_runtime/src/velem.rs`, `crates/wukong_runtime/src/vmath.rs`, `crates/wukong_runtime/src/xent.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Elementwise vmath/velem/vhorner/bias-bcast/axpby call lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1182-1202` — Lowers the streaming elementwise kernel family, including the bf16/f16 input and output twins that reuse identical signatures; further arms at `crates/wukong_codegen_cranelift/src/lib.rs:1409-1450` and `crates/wukong_codegen_cranelift/src/lib.rs:1562-1579`.
**`crates/wukong_runtime/src/bias.rs`**

- **Fused-activation sharing with the vmath dispatch (`BIAS_ACT_NONE` sentinel)** — `crates/wukong_runtime/src/bias.rs:27-52` — Reuses `vmath::apply1`/`vmath8_for` so `act(x+b)` is bit-for-bit `wukong_vmath_f32(x+b)`; unknown op codes silently degenerate to identity.
- **Broadcast-bias AVX2 kernel with ×4 unroll and macro-instantiated activation** — `crates/wukong_runtime/src/bias.rs:54-127` — Resolves the 8-lane activation once, then instantiates the row loop per branch so the hot ×4-unrolled 32-column body carries no per-block branch.
- **`wukong_bias_bcast_f32` entry point and scalar fallback** — `crates/wukong_runtime/src/bias.rs:140-174` — Public `[rows,cols]` broadcast-bias symbol requiring both AVX2 and FMA detection before the vector path, else the scalar reference at `crates/wukong_runtime/src/bias.rs:43-52`.
**`crates/wukong_runtime/src/cumminmax.rs`**

- **Ext extremum abstraction: identity and tie/NaN-matching fold** — `crates/wukong_runtime/src/cumminmax.rs:29-70` — Max/Min enum supplying -inf/+inf identities and the (a>b)?a:b / (a<b)?a:b scalar folds that replicate _mm256_max_ps/min_ps lane semantics exactly.
- **cummax/cummin Hillis-Steele identity-filled scan and row carry** — `crates/wukong_runtime/src/cumminmax.rs:89-177` — Same three-step cross-lane shift as cumsum but folding with max/min and filling vacated lanes with the identity, giving a bit-exact (non-reassociated) running extremum; twin at `crates/wukong_runtime/src/cumminmax.rs:72-87`.
**`crates/wukong_runtime/src/cumprod.rs`**

- **cumprod row scan and 4-row interleaved ILP block** — `crates/wukong_runtime/src/cumprod.rs:26-73` — Strict left-to-right prefix product from identity 1.0 with four in-flight mul chains; unlike cumsum it is bit-exact to naive since a product is not reassociated.
**`crates/wukong_runtime/src/cumsum.rs`**

- **Hillis-Steele in-register 8-lane inclusive prefix sum** — `crates/wukong_runtime/src/cumsum.rs:41-77` — Three permutevar8x32 cross-128-bit shifts with blendv zero-fill and adds, the balanced tree that breaks the loop-carried prefix-sum dependency.
- **cumsum row driver: block scan + broadcast carry + scalar tail** — `crates/wukong_runtime/src/cumsum.rs:87-113` — Adds the running carry to each scanned block, re-extracts carry from lane 7, and finishes cols%8 left-to-right; twin/dispatch at `crates/wukong_runtime/src/cumsum.rs:25-39` and `crates/wukong_runtime/src/cumsum.rs:115-129`.
**`crates/wukong_runtime/src/lrscan.rs`**

- **lrscan single-row first-order linear recurrence** — `crates/wukong_runtime/src/lrscan.rs:38-58` — h = a[t]*h + b[t] as separate mul then add (two roundings, deliberately not mul_add) with in-place aliasing safety, the SSM/EMA scan primitive.
- **lrscan 4-row interleaved ILP block** — `crates/wukong_runtime/src/lrscan.rs:60-100` — Runs four independent row carries down a shared t to fill ports a single latency-bound chain leaves idle, with a <4-row scalar tail and no reassociation; oracles at `crates/wukong_runtime/src/lrscan.rs:176-299`.
**`crates/wukong_runtime/src/velem.rs`**

- **velem op-code ISA (VE_ID/RELU/RELU6/USE_Y/HADAMARD/DIV)** — `crates/wukong_runtime/src/velem.rs:36-50` — Bit-flag opcode contract shared with the mir_build recognizer selecting activation (low byte) and compute mode (affine, Hadamard, quotient) plus y-read flag.
- **Scalar twin elem1/act1 and the identity-affine fast path** — `crates/wukong_runtime/src/velem.rs:113-165` — Scalar element semantics mirroring maxps/minps tie/NaN behaviour and skipping the FMA when a==1,c==0,y unread, the lane-for-lane oracle for the AVX2 body.
- **velem 256-bit AVX2/FMA streaming map kernel** — `crates/wukong_runtime/src/velem.rs:242-359` — Unrolled-x4 (32 elements/step) compute/store macro kernel doing affine FMA or mul/div plus clamp, the core width and ILP win over Cranelift's 128-bit vectorizer.
- **NT store mechanics: 32-byte align prologue, sfence, gated prefetch** — `crates/wukong_runtime/src/velem.rs:275-353` — Scalar peel until out is 32B-aligned (vmovntps faults otherwise), sfence after weakly ordered stores, and prefetch only in the DRAM regime; distance at `crates/wukong_runtime/src/velem.rs:108-110`.
- **vhorner entry and scalar Horner reference** — `crates/wukong_runtime/src/velem.rs:429-475` — Polynomial map out[i]=poly(x[i]) with a mul_add Horner chain, feature-detected AVX2 dispatch, and degenerate handling for ncoeff<=1.
- **vhorner AVX2 six-chain ILP kernel with MAX_HORNER stack splat** — `crates/wukong_runtime/src/velem.rs:477-572` — Pre-splats up to 32 coefficients on the stack and runs six independent 8-lane FMA chains (48 elem/step) to hide the latency-bound Horner recurrence.
**`crates/wukong_runtime/src/vmath.rs`**

- **VM_* one-input opcode numbering** — `crates/wukong_runtime/src/vmath.rs:24-59` — 36 contiguous `i64` op codes (VM_EXP=0 … VM_CBRT=35) shared verbatim with the mir_build elementwise recognizer, defining the whole dispatch namespace.
- **Table-reduced exp core (constants + scalar twin + AVX2 lanes)** — `crates/wukong_runtime/src/vmath.rs:86-129` — 8-bucket `e^x=2^e·T[j]·poly(r)` reduction with vpermps table lookup, Cody-Waite ln2/8 split and 3-FMA cubic; twins at `crates/wukong_runtime/src/vmath.rs:248-275` and `crates/wukong_runtime/src/vmath.rs:1555-1586`.
- **Table-reduced log core (constants + scalar twin + AVX2 lanes)** — `crates/wukong_runtime/src/vmath.rs:131-184` — bit-arithmetic `ln(x)=k·ln2+L[j]+poly(s)` split at LOG_OFF so bucket 4 straddles 1.0 exactly; twins at `crates/wukong_runtime/src/vmath.rs:285-318` and `crates/wukong_runtime/src/vmath.rs:1590-1625`.
- **Base-conversion family exp2/log2/exp10/log10** — `crates/wukong_runtime/src/vmath.rs:214-219` — four ops implemented as pre/post multiplies around the shared exp/log using `f64 as f32` constants matching splat_const_f; bodies at `crates/wukong_runtime/src/vmath.rs:516-539` and `crates/wukong_runtime/src/vmath.rs:1869-1897`.
- **Scalar activation twin family (tanh…hardswish, softsign, logsigmoid)** — `crates/wukong_runtime/src/vmath.rs:320-442` — fifteen scalar activations composed on exp1/log1 with stable softplus and NaN-matching min/max clamps, serving as AVX2 tail and no-AVX2 fallback.
- **Cephes sin/cos with 3-part π/2 range reduction** — `crates/wukong_runtime/src/vmath.rs:451-489` — quadrant reduction via add-magic rounding plus separate sin/cos minimax polys blended by float-eq quadrant masks; AVX2 twin at `crates/wukong_runtime/src/vmath.rs:1786-1839`.
- **erf via Abramowitz–Stegun 7.1.26** — `crates/wukong_runtime/src/vmath.rs:496-514` — degree-5 rational-in-t times `exp(-x²)` with compare-select abs and odd-function sign fixup, giving exact-GELU capability; AVX2 twin at `crates/wukong_runtime/src/vmath.rs:1846-1867`.
- **Hyperbolic and inverse-hyperbolic family** — `crates/wukong_runtime/src/vmath.rs:541-577` — sinh/cosh/asinh/acosh/atanh built on exp1/log1, with asinh using abs+copysign to dodge the large-negative cancellation; AVX2 twins at `crates/wukong_runtime/src/vmath.rs:1899-1954`.
- **Cephes atan 3-region reduction** — `crates/wukong_runtime/src/vmath.rs:585-607` — branchless fold to |x| with two breakpoints, degree-3 odd poly, π/4 or π/2 offset and bit-or copysign; AVX2 twin at `crates/wukong_runtime/src/vmath.rs:1961-1987`.
- **Composed inverse trig tan/asin/acos** — `crates/wukong_runtime/src/vmath.rs:609-630` — tan as sin/cos and asin as `atan(x/√(1−x²))` with acos as π/2−asin, inheriting atan's ≈1-ULP grade; AVX2 twins at `crates/wukong_runtime/src/vmath.rs:1989-2014`.
- **All-real cbrt via exp∘log with zero guard** — `crates/wukong_runtime/src/vmath.rs:638-643` — evaluates `e^{ln|x|/3}`, blends the `|x|==0 → 0` case to avoid log(0) garbage, then bit-or restores sign; AVX2 twin at `crates/wukong_runtime/src/vmath.rs:2020-2029`.
- **Kahan-corrected expm1/log1p** — `crates/wukong_runtime/src/vmath.rs:645-676` — `(u−1)·x/ln(u)` and `ln(u)·x/(u−1)` corrections with a branchless `u==1 → x` guard recovering small-x relative accuracy; AVX2 twins at `crates/wukong_runtime/src/vmath.rs:2031-2055`.
- **apply1 scalar op dispatch** — `crates/wukong_runtime/src/vmath.rs:683-723` — single match mapping all 36 VM_* codes to scalar twins, reused by bias.rs fused activation, returning identity `x` for an unrecognized op.
- **Parallel vmath variant with pool-invariant chunking** — `crates/wukong_runtime/src/vmath.rs:751-797` — fixed 16384-element chunks over the rayon pool so output is bit-identical to serial regardless of thread count, falling back below the chunk size or on a width-1 pool.
- **vmath8_for AVX2 kernel-pointer table** — `crates/wukong_runtime/src/vmath.rs:809-851` — returns the 8-lane function pointer per op code, shared by the f32, bf16 and f16 dispatchers plus bias.rs, so all paths apply the identical activation.
- **vmath_avx2 main loop structure** — `crates/wukong_runtime/src/vmath.rs:884-970` — NT decision, scalar alignment prologue, optional ×6 body, ×4 (32-elem) ILP body, 8-wide remainder, sfence and scalar tail, all bit-identical.
- **VM2_* two-input opcode namespace** — `crates/wukong_runtime/src/vmath.rs:977-999` — a separate 12-code namespace covering pow/atan2/hypot, six activation backwards and three gated-FFN forms, documented per-op with its positional argument meaning.
- **Two-input transcendentals pow/atan2/hypot** — `crates/wukong_runtime/src/vmath.rs:1001-1039` — pow as exp∘log, atan2 with a sign-aware quadrant π adjust, hypot scaled by max(|a|,|b|) with a zero guard; AVX2 twins at `crates/wukong_runtime/src/vmath.rs:1163-1209`.
- **vmath2 dispatch and AVX2 pipeline** — `crates/wukong_runtime/src/vmath.rs:1393-1449` — three-stream NT gate, alignment prologue, ×4 body, remainder, sfence and tail; scalar dispatch `crates/wukong_runtime/src/vmath.rs:1143-1159`, pointer table `crates/wukong_runtime/src/vmath.rs:1339-1359`, entry `crates/wukong_runtime/src/vmath.rs:1367-1391`.
- **AVX2 8-lane activation family (relu8…hardswish8)** — `crates/wukong_runtime/src/vmath.rs:1629-1779` — the 256-bit lane twins mirroring each scalar activation op-for-op, including selu's multiply order and blendv-vs-branch equivalence, keeping lane==scalar bit-exact.
- **libm-parity tolerance gate** — `crates/wukong_runtime/src/vmath.rs:2064-2161` — sweeps 4096 points per op against std references with per-op mixed abs+rel tolerances; extra oracles for erf, inverse hyperbolics, inverse trig and vmath2 at `crates/wukong_runtime/src/vmath.rs:2166-2304`.
- **Lane==scalar-tail bit-exactness gate family** — `crates/wukong_runtime/src/vmath.rs:2364-2404` — asserts kernel output equals apply1 bit-for-bit on non-multiple-of-8 lengths (log ops listed then skipped as out-of-domain); NT-regime variants at `crates/wukong_runtime/src/vmath.rs:2935-2968`.
- **exp/log table self-derivation gates** — `crates/wukong_runtime/src/vmath.rs:2412-2427` — re-derives every R/L and T entry, SCALE, MBIAS, the Cody-Waite pair and Chebyshev-shifted P2 from scratch and pins bits; exp half at `crates/wukong_runtime/src/vmath.rs:2505-2533`.
- **Dense f64-reference accuracy sweeps** — `crates/wukong_runtime/src/vmath.rs:2438-2474` — 500k log points assert log relative error <1e-6 and ln(1)==0; 3M exp points assert <2.4e-7 and exp(0)==1 at `crates/wukong_runtime/src/vmath.rs:2543-2578`.
- **Monotonicity and saturation gates** — `crates/wukong_runtime/src/vmath.rs:2481-2495` — scans ±256 consecutive f32 around every log bucket edge for no-decrease; exp pins ±∞/NaN clamp behaviour, 0x7F3504A4 saturation and n-edge monotonicity at `crates/wukong_runtime/src/vmath.rs:2588-2620`.
**`crates/wukong_runtime/src/xent.rs`**

- **Shared vmath transcendental reuse for bit-exactness** — `crates/wukong_runtime/src/xent.rs:31-34` — Losses import `exp1`/`exp8`/`log1`/`log8` from vmath rather than carrying private polynomials, so loss bits match softmax and log-softmax; also `crates/wukong_runtime/src/kldiv.rs:30-33`, `crates/wukong_runtime/src/entropy.rs:37-40`.

## G24 — Runtime: normalizations and softmax

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_runtime/src/logsoftmax.rs`, `crates/wukong_runtime/src/norm.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Fused normalization and norm-backward call lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1580-1614` — Lowers `wukong_norm_f32` and the affine gamma/beta variant (nullable pointers, eps as raw bits, op selector); RMSNorm/LayerNorm backward at `crates/wukong_codegen_cranelift/src/lib.rs:1389-1408`.
**`crates/wukong_runtime/src/logsoftmax.rs`**

- **Shared log-sum-exp core (scalar + AVX2)** — `crates/wukong_runtime/src/logsoftmax.rs:57-92` — Factors `off = m + log(Σexp(x−m))` into one routine reused by both log-softmax and log-sum-exp, keeping their `m`/`s` bits identical; AVX2 twin at `crates/wukong_runtime/src/logsoftmax.rs:119-158`.
- **Log-softmax / log-sum-exp row wrappers and feature dispatch** — `crates/wukong_runtime/src/logsoftmax.rs:94-224` — Thin per-row writers (`x−off` vector+tail, or a single scalar store) behind `is_x86_feature_detected` gates selecting AVX2 or the scalar twin.
**`crates/wukong_runtime/src/norm.rs`**

- **Norm op-code ABI constants** — `crates/wukong_runtime/src/norm.rs:26-31` — Five i64 op codes (softmax, layernorm, rmsnorm, logsoftmax, l2norm) shared verbatim with the mir_build recognizer, selecting the per-row routine inside one C-ABI kernel.
- **Fixed-order 8-lane horizontal combines hsum8/hmax8** — `crates/wukong_runtime/src/norm.rs:33-47` — Balanced-tree sum/max over the 8 lane accumulators, called by both scalar and AVX2 paths so their reductions agree bit-for-bit; duplicated at `crates/wukong_runtime/src/logsoftmax.rs:39-53`.
- **Scalar stable softmax row** — `crates/wukong_runtime/src/norm.rs:51-90` — Three fused passes (lane-wise max, exp(x−m) written to out while summing, reciprocal scale) with an alias-safe in-place contract and no eps use at all.
- **Scalar stable log-softmax row** — `crates/wukong_runtime/src/norm.rs:92-134` — Reuses softmax's max/Σexp lane structure then folds both subtractions into one `x−(m+log s)` rounding, deliberately not writing `out` during the sum so in-place rows stay valid.
- **Scalar LayerNorm row** — `crates/wukong_runtime/src/norm.rs:136-173` — Mean pass then an explicit `mul_add` squared-deviation variance (to match the AVX2 fmadd), normalizing with `1/sqrt(var+eps)`; eps is added inside the sqrt, not clamped.
- **Scalar RMSNorm row** — `crates/wukong_runtime/src/norm.rs:175-199` — Single fmadd sum-of-squares pass scaled by 1/n, then `x * 1/sqrt(mean_sq+eps)`, skipping the mean-subtraction entirely.
- **Scalar L2-norm row** — `crates/wukong_runtime/src/norm.rs:201-228` — RMSNorm's byte-identical sum-of-squares without the `1/n` factor, giving the unit-norm projection `x/sqrt(Σx²+eps)` for embeddings and cosine similarity.
- **AVX2 softmax row with vectorized exp8** — `crates/wukong_runtime/src/norm.rs:232-278` — 8-wide max, `exp8`-based exp+store+accumulate, and 8-wide rescale, with scalar `exp1` tails folded into the same lanes so it matches the scalar twin bit-for-bit.
- **AVX2 log-softmax row** — `crates/wukong_runtime/src/norm.rs:280-328` — Same vector max/Σexp as softmax but no intermediate store; broadcasts `off = m + log1(sum)` and emits `x − off` 8-wide, keeping one scalar log as the bit-exact pivot.
- **AVX2 LayerNorm row** — `crates/wukong_runtime/src/norm.rs:330-376` — Vector mean, `_mm256_fmadd_ps` variance accumulator with a `mul_add` scalar tail, then an 8-wide `(x−mean)*inv` writeback mirroring the scalar path exactly.
- **AVX2 RMSNorm and L2-norm rows** — `crates/wukong_runtime/src/norm.rs:378-443` — Twin fmadd sum-of-squares kernels differing only by the `invn` mean factor, sharing the broadcast-reciprocal-sqrt writeback and scalar tails.
- **Scalar affine LayerNorm/RMSNorm rows** — `crates/wukong_runtime/src/norm.rs:445-536` — Reductions duplicated verbatim from the plain twins, with per-column gamma/beta folded into one `mul_add` and null pointers meaning scale 1 / shift 0.
- **AVX2 affine LayerNorm/RMSNorm rows** — `crates/wukong_runtime/src/norm.rs:538-662` — Same reductions as the plain AVX2 kernels plus an `_mm256_fmadd_ps(norm, gamma, beta)` writeback with the null-pointer tests hoisted out of the loop into ones/zeros splats.
- **Per-row dispatchers with silent unsupported-op no-op** — `crates/wukong_runtime/src/norm.rs:664-723` — Runtime `avx2`+`fma` detection choosing vector or scalar twins per op code, but every unrecognized op (including softmax through the affine path) silently returns leaving `out` untouched.

## G25 — Runtime: reductions and arg-reductions

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_runtime/src/colarg.rs`, `crates/wukong_runtime/src/colreduce.rs`, `crates/wukong_runtime/src/lowp.rs`, `crates/wukong_runtime/src/reduce.rs`, `crates/wukong_runtime/src/rowarg.rs`, `crates/wukong_runtime/src/xent.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Shared (ptr,ptr,i64,i64) kernel family lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1203-1238` — One arm covers logsumexp, entropy, row/col arg-reductions and four cumulative scans because they share the vmath signature; transpose and eight column reductions follow at `crates/wukong_codegen_cranelift/src/lib.rs:1251-1312`.
- **Value-returning reduction kernel lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1503-1561` — Binds call results for `sreduce` (f32), `argreduce` (i64 index), and the bf16/f16 dot/sum/max-family reductions, the only kernels here whose result feeds back into MIR.
**`crates/wukong_runtime/src/colarg.rs`**

- **Strict-compare scalar per-column arg scan** — `crates/wukong_runtime/src/colarg.rs:41-80` — Seeds best row 0 from `x[0,j]` then scans rows ascending with strict `>`/`<` so ties keep the lowest row; the initial `out[j]=0` store is dead.
- **Single-pass L1-resident AVX2 column arg-reduce** — `crates/wukong_runtime/src/colarg.rs:82-168` — Holds `best_val`/`best_idx` for the whole column range in scratch and reads each element once; row indices are tracked as f32, exact only below 2^24 rows.
**`crates/wukong_runtime/src/colreduce.rs`**

- **ColKind taxonomy and per-kind seed rule** — `crates/wukong_runtime/src/colreduce.rs:15-36` — Eight column kinds where additive kinds seed 0 and fold rows `[0,rows)` while Max/Min/MaxAbs seed row 0 and fold `[1,rows)`.
- **Per-column statistics finalize pass** — `crates/wukong_runtime/src/colreduce.rs:38-67` — Applies `/rows`, `sqrt`, or `sqrt(_/rows)` once per output column after the fold; the Mean divisor is confusingly named `inv` yet used as a divide.
- **AVX2 row-major streaming column-reduction kernel** — `crates/wukong_runtime/src/colreduce.rs:69-210` — Streams `x` row-major folding 8 columns per step into cache-resident `out`, hoisting the kind match out of the row loop; squares use mul+add, not FMA.
- **Scalar column-reduction twin and range dispatch** — `crates/wukong_runtime/src/colreduce.rs:212-283` — Same i-ascending per-column order and identical fold expressions as the vector path; runtime AVX2 selection at `crates/wukong_runtime/src/colreduce.rs:285-307`.
- **Sixteen exported column-reduction C entries** — `crates/wukong_runtime/src/colreduce.rs:347-564` — colsum/colmax/colmin/colmaxabs/colmean/coll2/colrms/colsumsq each with a serial and a parallel `#[no_mangle]` wrapper, all guarding `rows<=0 || cols<=0`.
**`crates/wukong_runtime/src/lowp.rs`**

- **Deterministic chunked parallel half reductions** — `crates/wukong_runtime/src/lowp.rs:483-665` — `par_chunk_reduce` cuts `[0,n)` on fixed `RCHUNK` boundaries and folds partials in ascending chunk order, so results are thread-count-independent though reassociated versus the flat serial kernel.
**`crates/wukong_runtime/src/reduce.rs`**

- **Reduction op-code table and dual-ABI split** — `crates/wukong_runtime/src/reduce.rs:30-50` — Eleven `RED_*` codes shared with the MIR recognizer; codes 7/8 (argmax/argmin) ride an `-> i64` ABI while the rest ride `-> f32`.
- **Fold algebra: identity, pairwise combine, fixed-order 8-lane tree** — `crates/wukong_runtime/src/reduce.rs:52-87` — `ident` seeds 0/∓∞ and `fold2` defines `+`/`(a>b)?a:b` matching `_mm256_max_ps`; `hcombine8` is the balanced tree at `crates/wukong_runtime/src/reduce.rs:119-130`.
- **Deterministic fixed-chunk decomposition and serial entry** — `crates/wukong_runtime/src/reduce.rs:89-94` — `RCHUNK = 8192` elements independent of thread count is the determinism contract; the ascending serial fold lives at `crates/wukong_runtime/src/reduce.rs:227-247`.
- **Per-element contribution semantics for all nine value reductions** — `crates/wukong_runtime/src/reduce.rs:96-117` — Encodes dot/ssd/sum/sumsq/max/min/maxabs/sumabs/absdiff as mul_add or sign-mask-abs forms; unknown op codes silently return the accumulator unchanged.
- **Portable scalar chunk twin and feature dispatch** — `crates/wukong_runtime/src/reduce.rs:150-175` — Eight scalar lane accumulators mirroring the AVX2 register lane-for-lane, selected by runtime avx2+fma detection at `crates/wukong_runtime/src/reduce.rs:132-148`.
- **AVX2 chunk reduction kernel** — `crates/wukong_runtime/src/reduce.rs:177-225` — One `__m256` accumulator per chunk with a per-op intrinsic arm, then scalar tail plus `hcombine8`, deliberately bit-identical to the scalar twin.
- **Arg-reduction tie-break total order** — `crates/wukong_runtime/src/reduce.rs:298-320` — `arg_fold` uses a strict value compare plus an explicit lower-index rule, making the fold associative so any chunk split returns the same index.
- **Scalar arg-reduce twin and i32-index dispatch guard** — `crates/wukong_runtime/src/reduce.rs:351-384` — Eight ILP lane candidates collapsed ascending; the dispatcher at `crates/wukong_runtime/src/reduce.rs:322-349` refuses AVX2 when `hi > i32::MAX` since indices ride i32 lanes.
- **AVX2 four-accumulator arg-reduce kernel** — `crates/wukong_runtime/src/reduce.rs:386-454` — 32 elements per iteration via `cmp_ps` + two `blendv`; unconsumed lanes carry index sentinel −1, so an all-(−∞) input diverges from the scalar twin.
- **Whole-array argmax/argmin C entries, serial and parallel** — `crates/wukong_runtime/src/reduce.rs:456-529` — Both use the same `RCHUNK` split and ascending `arg_fold`; empty or all-NaN input yields `usize::MAX` cast to −1.
**`crates/wukong_runtime/src/rowarg.rs`**

- **Per-row argmax/argmin kernels** — `crates/wukong_runtime/src/rowarg.rs:31-119` — Scalar ascending scan plus an AVX2 path seeding eight lanes from real data (no ±∞ sentinel), collapsing left-to-right through the shared `arg_fold`.
**`crates/wukong_runtime/src/xent.rs`**

- **Duplicated fixed-order hsum8/hmax8 lane combiners** — `crates/wukong_runtime/src/xent.rs:36-50` — Balanced-tree horizontal sum and max copied verbatim into all seven files instead of shared, the invariant that makes scalar and AVX2 reductions bit-identical; also `crates/wukong_runtime/src/entropy.rs:42-48`.

## G26 — Runtime: quantization and low precision (bf16 / f16 / int8 / int4)

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_runtime/src/dequant.rs`, `crates/wukong_runtime/src/gemm.rs`, `crates/wukong_runtime/src/i8gemm.rs`, `crates/wukong_runtime/src/lib.rs`, `crates/wukong_runtime/src/lowp.rs`, `crates/wukong_runtime/src/transpose.rs`, `crates/wukong_runtime/src/vmath.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **bf16/f16 narrow memory load-widen and store-narrow path** — `crates/wukong_codegen_cranelift/src/lib.rs:637-693` — Loads 2-byte halves and widens (bf16 by `<<16` bitcast, f16 via runtime shim), and rounds+truncates on store, since register type is `f32`.
- **bf16/f16 grid rounding at constants and observation boundaries** — `crates/wukong_codegen_cranelift/src/lib.rs:981-1023` — Inline bit-exact `round_bf16` arithmetic (with NaN quieting) and `half`-crate shim calls for f16, plus rounded constant materialization at `crates/wukong_codegen_cranelift/src/lib.rs:535-551` so -O2 cannot diverge from -O0 via store bypass.
- **Int8 GEMM and dequant call lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1615-1627` — Lowers quantized `i8gemm_nt`, its 10-arg fused-dequant epilogue form, and the flat/per-channel dequant kernels; the latter three at `crates/wukong_codegen_cranelift/src/lib.rs:1675-1720`.
**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Quantized and low-precision kernel gates (int8 GEMM, dequant, bf16/f16)** — `crates/wukong_codegen_cranelift/src/tests.rs:2261-2294` — Sweeps i8/u8/i32 widths × four activations × serial/@parallel for dequant and per-channel dequant, plus u8×i8 GEMM signedness rejection, bf16 dot/sum, axpby and narrowing half-output kernels with golden values, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:2186-2251`, `crates/wukong_codegen_cranelift/src/tests.rs:2369-2411`, `crates/wukong_codegen_cranelift/src/tests.rs:2621-2731`, `crates/wukong_codegen_cranelift/src/tests.rs:2737-2841`.
**`crates/wukong_mir_build/src/lib.rs`**

- **int8 quantized GEMM and dequant symbols** — `crates/wukong_mir_build/src/lib.rs:1858-1883` — u8×i8→i32 NT GEMM, the fused GEMM+per-channel-dequant that keeps the i32 accumulator in registers, and the standalone flat/per-channel dequant maps with width codes.
- **Half-precision reductions, axpby and narrowing-store symbols** — `crates/wukong_mir_build/src/lib.rs:1913-1962` — bf16/f16 dot/sum/max-family reductions with parallel twins the interpreter marshals *directly* (unlike f32, whose interp calls the serial form), plus half-in/half-out axpby and activations.
- **Low-precision plain GEMM emitters (emit_i8gemm, emit_lowp_gemm)** — `crates/wukong_mir_build/src/lib.rs:7495-7563` — Emit int8 u8xi8 NT and an 8-way bf16/f16 x NT/TN x serial/parallel kernel selection with beta hard-coded 0 for the dot-product form.
- **bf16/f16 Linear+epilogue fusion (try_fuse_lowp_matmul_epilogue, emit_lowp_gemm_epi)** — `crates/wukong_mir_build/src/lib.rs:8051-8130` — Fuses a half-precision NT matmul with its bias/activation loop into one nt_epi call, declining TN nests that have no half epilogue kernel.
- **int8 GEMM + per-channel dequant fusion with liveness gate** — `crates/wukong_mir_build/src/lib.rs:8222-8319` — Fuses the i32 accumulator away only when it is a block-local `let` unmentioned after the dequant loop; folds scales, optional bias, and activation into registers.
- **Integer cast operand with sema-derived signedness (`dequant_cast_operand`)** — `crates/wukong_mir_build/src/lib.rs:11934-11955` — Reads i8/u8/i32 width from the sema scalar because `MirType` collapses i8 and u8, selecting sign- versus zero-extension; batched twin at `crates/wukong_mir_build/src/lib.rs:12079-12102`.
- **bf16/f16 mixed-precision matmul nest** — `crates/wukong_mir_build/src/lib.rs:20289-20517` — Matches half-precision operands widened to f32 with an f32 accumulator in NT or TN form, requiring identical precision and exactly one transposed operand.
**`crates/wukong_runtime/src/dequant.rs`**

- **Dequant op-code encoding and scalar reference** — `crates/wukong_runtime/src/dequant.rs:32-115` — packs activation in the low byte and input width (`DQ_I8`/`DQ_U8`/`DQ_I32`) in the next, with `load_i32`/`deq1`/`act1` as the single scalar truth shared by the AVX2 tail and the interpreter.
- **AVX2 dequant with macro-flattened width loops** — `crates/wukong_runtime/src/dequant.rs:141-265` — works around `#[target_feature]` helpers not inlining on stable by expanding one 4×-unrolled 32-elem/step loop per input width; GELU/SiLU take a separate per-lane path that stack-round-trips each vector and never uses NT stores.
- **`wukong_dequant_f32` entry and 8-aligned parallel chunking** — `crates/wukong_runtime/src/dequant.rs:267-344` — detects AVX2 once, and the parallel form rounds chunk sizes up to a multiple of 8 via a `Send`/`Sync` `Addr` wrapper so each worker's NT prologue peels disjointly.
- **Per-channel dequant (`scale[j]` broadcast down rows)** — `crates/wukong_runtime/src/dequant.rs:346-535` — the quantized `nn.Linear` writeback shape, with a per-width macro row kernel loading the scale vector instead of a splat (issuing `sfence` per row when NT) and a per-row multicore twin.
- **Dequant correctness gates** — `crates/wukong_runtime/src/dequant.rs:537-670` — bit-for-bit AVX2 == scalar == parallel across all widths and activations at a length crossing the NT prologue and tail, plus a rational/f64 reference past 2²⁴ and degenerate-length cases.
**`crates/wukong_runtime/src/gemm.rs`**

- **bf16/f16 widen-prepass GEMM family (NT, NT+epilogue, TN)** — `crates/wukong_runtime/src/gemm.rs:814-1163` — Twelve entry points losslessly widening u16 halves into f32 scratch and delegating to the f32 kernels, so half-precision results equal the widened f32 GEMM bit-for-bit.
**`crates/wukong_runtime/src/i8gemm.rs`**

- **Scalar u8×i8 dot reference plus AVX2 `vpmaddwd` dot** — `crates/wukong_runtime/src/i8gemm.rs:36-108` — widens 16 bytes at a time to i16 and folds with two ILP chains into i32 lanes; wrapping i32 is associative mod 2³² so no fixed combine order is needed, with the in-register horizontal sum at lines 100-108.
- **1×4 register-blocked AVX2 int8 dot and row driver** — `crates/wukong_runtime/src/i8gemm.rs:110-190` — one A-row chunk widened once and reused across four B-rows with four accumulator chains, plus the scalar `C=A·Bᵀ` nest fallback at `crates/wukong_runtime/src/i8gemm.rs:192-204`.
- **AVX-VNNI `vpdpbusd` int8 dot family** — `crates/wukong_runtime/src/i8gemm.rs:206-308` — single-instruction 32-product multiply-accumulate in `dot_i8_vnni`/`dot4_i8_vnni` with a `gemm_row_nt_vnni` row driver, ~3-4× denser than the widen+madd path and still bit-exact.
- **2×4 VNNI register tile (two A-rows × four B-rows)** — `crates/wukong_runtime/src/i8gemm.rs:310-418` — each B-row chunk is loaded once and fed to both A-rows, halving B traffic across eight independent `vpdpbusd` chains, driven by `gemm_2rows_nt_vnni`.
- **`wukong_i8gemm_nt` SIMD-tier dispatch** — `crates/wukong_runtime/src/i8gemm.rs:420-471` — detects VNNI then AVX2 exactly once per call and runs the 2-row tile (odd row via the 1×4 driver) or the AVX2 row loop, keeping the hot loop branch-free.
- **Multicore int8 GEMM row map** — `crates/wukong_runtime/src/i8gemm.rs:473-522` — hoists feature detection out of the rayon loop and maps independent C rows across cores; note it always uses the 1×4 `gemm_row_nt_vnni`, not the serial path's 2×4 tile, despite the doc claiming identical per-row code.
- **Per-channel dequant epilogue row** — `crates/wukong_runtime/src/i8gemm.rs:524-558` — `dst[j] = act(i32·scale_a·scale_b[j] + bias[j])` in the recognizer's exact left-associated multiply order, with act codes 0/1/2/3 reusing `vmath::{gelu1,silu1}`.
- **Fused int8 GEMM + dequant (`wukong_i8gemm_nt_deq`)** — `crates/wukong_runtime/src/i8gemm.rs:560-625` — computes each C row into an L1-resident i32 scratch and dequantizes immediately so the accumulator never round-trips to DRAM, with the per-row multicore twin at `crates/wukong_runtime/src/i8gemm.rs:627-689`.
- **int8 GEMM bit-exactness test suite** — `crates/wukong_runtime/src/i8gemm.rs:691-944` — deterministic full-range fills pin scalar == AVX2 == VNNI == 2×4 tile at every K, fused == unfused dequant for all four activations, serial == parallel, an i64 reference, and identical i32 overflow wrapping.
**`crates/wukong_runtime/src/lib.rs`**

- **bf16 round-to-nearest-even conversion trio** — `crates/wukong_runtime/src/lib.rs:169-196` — Single shared bf16 pack/unpack/round used by interpreter, Cranelift-emitted CLIF and the bf16 GEMM packer, with an explicit NaN-quieting branch before the RNE bias.
- **IEEE f16 conversion trio backed by the half crate** — `crates/wukong_runtime/src/lib.rs:198-220` — f16 bits/round helpers deliberately routed through `half` so they match the F16C `vcvtps2ph` the runtime reduction kernels use bit-for-bit.
**`crates/wukong_runtime/src/lowp.rs`**

- **Half kind shim (widen/narrow single source of truth)** — `crates/wukong_runtime/src/lowp.rs:19-59` — `Half::{F16,Bf16}` dispatches every scalar widen/narrow through the crate-root `f16_bits_to_f32`/`bf16_bits_to_f32`/`f32_to_{bf16,f16}_bits` shims so interpreter and native round identically.
- **Deterministic 8-lane scalar twins for half sum/dot** — `crates/wukong_runtime/src/lowp.rs:61-96` — `sum_scalar`/`dot_scalar` keep exactly the 8 logical lane accumulators and fixed combine tree the SIMD path uses, making SIMD == scalar bit-for-bit and serving as the no-AVX2 fallback, with the combine at `crates/wukong_runtime/src/lowp.rs:26-30`.
- **Shared SIMD widen helpers (`vcvtph2ps`, bf16 `<<16` bit-extend)** — `crates/wukong_runtime/src/lowp.rs:100-120` — `pub(crate)` `widen_f16`/`widen_bf16` are the one definition every bf16/f16 dispatch path (including `vmath.rs`) widens through, keeping all half kernels bit-consistent.
- **AVX2 narrowing-store helpers (8- and 16-lane bf16/f16 packs)** — `crates/wukong_runtime/src/lowp.rs:128-199` — `bf16_round8` does round-to-nearest-even with a NaN-quieting blend, `narrow_{bf16,f16}` store 8 lanes and `narrow_{bf16,f16}_pack` build 32-byte blocks enabling one non-temporal store.
- **bf16/f16 sum and dot reductions with f32 accumulator** — `crates/wukong_runtime/src/lowp.rs:201-275` — AVX2/F16C+FMA lane-accumulator kernels that stream half the bytes of the f32 twin, exported through the C-ABI entry points at `crates/wukong_runtime/src/lowp.rs:411-479`.
- **bf16/f16 max/min/absmax reduction family** — `crates/wukong_runtime/src/lowp.rs:277-407` — `rfold` mirrors `reduce.rs::fold2` (absmax = max of `andnot(-0.0,v)`) and the AVX2/F16C folds are exact since widen and max/min round nothing, feeding `wukong_reduce_{bf16,f16}`.
- **Half-input / f32-output axpby (`out = a·x + b·y`)** — `crates/wukong_runtime/src/lowp.rs:667-781` — AVX2 `mul`+`fmadd` matched by a scalar twin in the same op order, streaming 8 bytes/elem instead of 12; note the f16 fallback reuses the misleadingly named `axpby_bf16_scalar` generic.
- **`stream_narrow!` streaming-narrow skeleton** — `crates/wukong_runtime/src/lowp.rs:816-867` — macro expanded inside the caller's `#[target_feature]` fn giving 32 lanes/iter, a scalar alignment prologue for `vmovntdq`, `sfence`, cacheable fallback, and 8-lane plus scalar tails.
- **All-half streaming axpby (bf16/f16 in and out)** — `crates/wukong_runtime/src/lowp.rs:869-906` — 6 bytes/elem versus an all-f32 axpby's 12, with the scalar twin at `crates/wukong_runtime/src/lowp.rs:790-798` and C-ABI entries at `crates/wukong_runtime/src/lowp.rs:953-1011`.
- **Half-output activation kernels (`vmath_*_out`)** — `crates/wukong_runtime/src/lowp.rs:908-951` — reuse `vmath::vmath8_for`/`apply1` and fall back to the scalar path for ops with no 8-lane twin, exported at `crates/wukong_runtime/src/lowp.rs:1013-1059` for the transformer activation/KV-write path.
- **SIMD-equals-scalar and f64-reference gates for the half kernels** — `crates/wukong_runtime/src/lowp.rs:1072-1199` — bit-for-bit twin checks over tail sizes for sum/dot/reduce/axpby, extended to the narrowing-output kernels at `crates/wukong_runtime/src/lowp.rs:1264-1338` and ULP-bounded f64 references at `crates/wukong_runtime/src/lowp.rs:1340-1361` and `crates/wukong_runtime/src/lowp.rs:1522-1544`.
- **Narrowing-store == scalar shim gate** — `crates/wukong_runtime/src/lowp.rs:1201-1262` — pins both the 8-lane and 16-lane pack narrows against `f32_to_{bf16,f16}_bits` over zeros, subnormals, f16 overflow, ±inf and NaN, the gate that keeps half-output kernels oracle-consistent.
- **Parallel-equals-sequential-chunked determinism gate** — `crates/wukong_runtime/src/lowp.rs:1546-1693` — `seq_chunked` recomputes the exact `RCHUNK` fold without rayon and the tests assert bit equality, repeat-call determinism, and exact max/min/absmax.
**`crates/wukong_runtime/src/transpose.rs`**

- **f32 and u16 (bf16/f16) transpose dispatch symbols** — `crates/wukong_runtime/src/transpose.rs:97-150` — Four `#[no_mangle]` entries (serial/parallel × f32/u16) over the shared generic core, exploiting that a transpose moves raw bits and is precision-agnostic.
**`crates/wukong_runtime/src/vmath.rs`**

- **bf16/f16-input activation kernels** — `crates/wukong_runtime/src/vmath.rs:1451-1549` — lossless widen (shift / F16C vcvtph2ps) feeding the same vmath8_for kernel to halve input bandwidth; their doc says "28-op" though 36 ops exist, and they omit the ×4 unroll and NT stores.
- **Low-precision equals widened-f32 gate** — `crates/wukong_runtime/src/vmath.rs:2747-2791` — runs 18 ops through the bf16 and f16 kernels and asserts bit equality with the f32 kernel on pre-widened inputs, the property interpreter marshalling relies on.

## G27 — Runtime: attention, RoPE, embedding, conv/pool, transpose

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_runtime/src/attention.rs`, `crates/wukong_runtime/src/embedding.rs`, `crates/wukong_runtime/src/pool2d.rs`, `crates/wukong_runtime/src/rope.rs`, `crates/wukong_runtime/src/transpose.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Embedding, attention, scatter-add and pool2d call lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1628-1674` — Lowers the token-gather, fused online-softmax SDPA (the only kernel with no `_parallel` twin), and embedding-gradient scatter; 2D pooling at `crates/wukong_codegen_cranelift/src/lib.rs:1266-1282`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Data-movement kernel emitters (emit_pool2d, emit_transpose)** — `crates/wukong_mir_build/src/lib.rs:7571-7625` — Emit max/avg pool2d with kernel and stride dims, and f32/u16 cache-blocked transpose; the pool selector treats any non-POOL_MAX code as average.
- **RoPE forward/backward emitter (emit_rope)** — `crates/wukong_mir_build/src/lib.rs:7804-7826` — Emits rotary-embedding kernels over `(x, inv_freq, out, rows, half)`, choosing forward or backward and serial or multicore from the nest flag.
- **Matrix transpose recognizer** — `crates/wukong_mir_build/src/lib.rs:21646-21760` — Pins both strides to the opposite loop bound, rejects in-place aliasing, and selects the f32 or precision-agnostic u16 blocked kernel by element scalar type.
- **2-D max/avg pooling recognizer** — `crates/wukong_mir_build/src/lib.rs:21762-22308` — Strictly decomposes the channel/window index into H, W, SH, SW, classifies the fold as fmax or sum, and pins the avg divisor to the true window count.
- **RoPE forward and backward recognizer** — `crates/wukong_mir_build/src/lib.rs:24108-24356` — Matches the half-split rotation with inline per-element cos/sin, requiring stride equal to twice the half width, and picks forward or inverse rotation by sign pattern.
**`crates/wukong_runtime/src/attention.rs`**

- **Fused attention public entry + AVX2/FMA dispatch gate** — `crates/wukong_runtime/src/attention.rs:19-53` — `wukong_attention_f32(q,k,v,o,s,d,scale,causal)` returns silently on non-positive dims, then runtime-detects avx2+fma to pick the tiled kernel, else the scalar twin.
- **Online-softmax (flash) recurrence, scalar reference** — `crates/wukong_runtime/src/attention.rs:55-101` — Per query keeps running max `m`, denominator `l` and a `D`-wide rescaled accumulator so the `S×S` score matrix is never materialized; `l==0` yields zeros.
- **ATT_QB=32 query-block tiling heuristic** — `crates/wukong_runtime/src/attention.rs:115-119` — Fixed 32-query block holds stats plus a `QB·D` accumulator L1-resident so each `k_j`/`v_j` load is reused across the block; an untuned magic constant.
- **AVX2 query-block-tiled flash-attention kernel** — `crates/wukong_runtime/src/attention.rs:121-216` — Keys outer, queries inner; 8-wide FMA dot plus 8-wide `acc·corr + p·v` rescale with scalar `D%8` tails, causal `j>i` skipped by `continue`; helper `crates/wukong_runtime/src/attention.rs:103-113`.
**`crates/wukong_runtime/src/embedding.rs`**

- **Embedding row-gather copy (AVX2 8-wide load/store + scalar twin)** — `crates/wukong_runtime/src/embedding.rs:26-74` — Copies one `H`-wide weight row per token id, 8 f32 per step with a scalar tail; pure data movement so both paths are bit-identical.
- **Out-of-range id guard zeroing the output row** — `crates/wukong_runtime/src/embedding.rs:76-132` — Ids outside `[0, v)` write an all-zero row instead of reading out of bounds, applied identically in the AVX2, scalar and parallel paths.
- **Embedding entry points and rayon row chunking (`EMBEDDING_PAR_MIN`=64)** — `crates/wukong_runtime/src/embedding.rs:134-206` — Serial and parallel gather symbols; the parallel form splits `T` rows into one chunk per thread, and both take `t`/`h`/`v` as `usize`, unlike the sibling kernels' `i64` ABI.
**`crates/wukong_runtime/src/pool2d.rs`**

- **AVX2 unit-horizontal-stride pooling band (sw==1)** — `crates/wukong_runtime/src/pool2d.rs:47-113` — Folds eight output columns at once from one contiguous `_mm256_loadu_ps` per window cell, using a true `_mm256_div_ps` (not reciprocal-multiply) so avg matches scalar bit-for-bit.
- **Even-lane deinterleave primitive for stride-2 gathers** — `crates/wukong_runtime/src/pool2d.rs:115-135` — Two `_mm256_permutevar8x32_ps` with index `{0,2,4,6,0,2,4,6}` plus a `0b11110000` blend extract eight stride-2 columns from sixteen contiguous floats, purely lane shuffles.
- **AVX2 stride-2 pooling band with ow_band bounds cutoff** — `crates/wukong_runtime/src/pool2d.rs:137-218` — The dominant CNN downsample path: precomputes how many leading output columns satisfy `2*ox+(kw-1)+16 <= w` for the two-vector load, sending the remainder to the scalar tail.
- **PoolKind fold semantics and shared scalar window fold** — `crates/wukong_runtime/src/pool2d.rs:220-266` — Single definition of a `kh×kw` fold in `(dy,dx)` ascending order — max seeded from cell (0,0), avg summed then divided by `kh·kw` — reused by kernel, tail and tests; enum at `crates/wukong_runtime/src/pool2d.rs:38-45`.
- **No-padding output-shape rule and window-fit guard** — `crates/wukong_runtime/src/pool2d.rs:332-345` — `out_dims` computes `oh=(h-kh)/sh+1`, `ow=(w-kw)/sw+1` and returns `None` for any non-positive dim or oversized window, making the kernel write nothing rather than fault.
- **Serial pooling driver and the four public pool entry points** — `crates/wukong_runtime/src/pool2d.rs:347-388` — Loops channel planes at `c*h*w` / `c*oh*ow` offsets; the max/avg × serial/parallel `#[no_mangle]` wrappers differ only by `PoolKind` at `crates/wukong_runtime/src/pool2d.rs:449-535`.
**`crates/wukong_runtime/src/rope.rs`**

- **RoPE scalar row twin (half-split rotation)** — `crates/wukong_runtime/src/rope.rs:40-65` — Rotates pair `(x[j], x[j+half])` by `theta = pos*inv_freq[j]` using shared `vmath::sincos1` and two `mul_add`s chosen to match the AVX2 fnmadd/fmadd exactly.
- **RoPE AVX2 row kernel with inline vectorized sin/cos** — `crates/wukong_runtime/src/rope.rs:67-110` — Eight pairs per step via `vmath::sin8`/`cos8` (the win: no libm vectorized `sincosf`), one FMA per output, scalar `half%8` tail through identical ops.
- **RoPE serial entry point and per-row AVX2 gate** — `crates/wukong_runtime/src/rope.rs:127-152` — `wukong_rope_f32` walks rows of a `[rows,2*half]` tensor using row index as absolute position, no-ops on non-positive dims, and permits `out` aliasing `x`; gate at `crates/wukong_runtime/src/rope.rs:112-125`.
**`crates/wukong_runtime/src/transpose.rs`**

- **Cache-blocked generic transpose core (B=32 tiles)** — `crates/wukong_runtime/src/transpose.rs:13-54` — Tiles `dst[j,i]=src[i,j]` into 32×32 blocks so both tiles stay L1-resident; generic over `Copy` so one core serves every element width.
- **Parallel transpose over row blocks with `TRANSPOSE_PAR_MIN`** — `crates/wukong_runtime/src/transpose.rs:56-95` — Spreads `B`-row blocks across cores above 65536 elements; splitting only on rows means few-row wide matrices get one task despite passing the gate.

## G28 — Runtime: backward / training kernels and losses

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_runtime/src/entropy.rs`, `crates/wukong_runtime/src/kd_loss.rs`, `crates/wukong_runtime/src/kldiv.rs`, `crates/wukong_runtime/src/layernorm_bwd.rs`, `crates/wukong_runtime/src/rmsnorm_bwd.rs`, `crates/wukong_runtime/src/rope_bwd.rs`, `crates/wukong_runtime/src/softmax_bwd.rs`, `crates/wukong_runtime/src/vmath.rs`, `crates/wukong_runtime/src/xent.rs`, `crates/wukong_runtime/src/xent_bwd.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Backward and loss kernel call family (3 ptr + 2 i64)** — `crates/wukong_codegen_cranelift/src/lib.rs:1313-1388` — Lowers vmath2, softmax-backward, linear-recurrence scan, cross-entropy forward/backward and RoPE forward/backward on one shared shape; KL/KD losses at `crates/wukong_codegen_cranelift/src/lib.rs:1239-1250`.
**`crates/wukong_mir_build/src/lib.rs`**

- **Backward and loss kernel emitters (softmax_bwd, rmsnorm_bwd, layernorm_bwd, xent, xent_bwd)** — `crates/wukong_mir_build/src/lib.rs:7667-7799` — Emit the batched training kernels, passing rows/cols and (for norms) the compile-time eps bit pattern, each bailing to the scalar nest when unbound.
**`crates/wukong_runtime/src/entropy.rs`**

- **Shannon entropy row reduction** — `crates/wukong_runtime/src/entropy.rs:60-110` — Computes `−Σ p·log p` using a deliberately split mul-then-add rather than FMA so scalar and AVX2 round identically; `0·log 0` is left unconditional and undefined.
**`crates/wukong_runtime/src/kd_loss.rs`**

- **Soft-label knowledge-distillation cross-entropy row** — `crates/wukong_runtime/src/kd_loss.rs:72-114` — Three-pass max / Σexp / Σq·(lse−x) reduction generalizing hard-label xent; q normalization is the caller's responsibility since the formula is applied verbatim.
- **KD-loss AVX2 twin with zero-subtract negation** — `crates/wukong_runtime/src/kd_loss.rs:128-179` — Builds `-(q·x)` as `_mm256_sub_ps(zero, q*x)` inside an fmadd to mirror the scalar `-(qj*x)`, a signed-zero-sensitive construction for one-hot q rows.
**`crates/wukong_runtime/src/kldiv.rs`**

- **Per-row KL divergence kernel** — `crates/wukong_runtime/src/kldiv.rs:54-106` — Folds `p·(log p − log q)` with a deliberate non-FMA multiply into one 8-lane accumulator; there is no `p == 0` guard, so zero probabilities yield NaN.
**`crates/wukong_runtime/src/layernorm_bwd.rs`**

- **LayerNorm-backward four per-row reductions** — `crates/wukong_runtime/src/layernorm_bwd.rs:50-114` — Recomputes mean, rstd, s1/C and s2/C from `x` in four fixed 8-lane accumulators, FMA-contracting variance and Σg·xhat; saved forward stats are never assumed.
- **LayerNorm-backward dx writeback** — `crates/wukong_runtime/src/layernorm_bwd.rs:116-145` — Elementwise `dx = rstd*(g - s1c - xhat*s2c)` computed as inner-then-scale with left-associated subtraction deliberately chosen to mirror the AVX2 sub-sub-mul ordering.
- **Optional-gamma null-pointer-means-one convention** — `crates/wukong_runtime/src/layernorm_bwd.rs:133-139` — A null `gamma` pointer selects an unbranched `dy` load per element in both reduction and apply passes, checked once per row; also `crates/wukong_runtime/src/rmsnorm_bwd.rs:68-108`.
- **LayerNorm-backward AVX2 twins** — `crates/wukong_runtime/src/layernorm_bwd.rs:155-282` — 256-bit stats and apply that spill their `__m256` accumulators into `[f32;8]` and reuse the scalar `hsum8` and scalar tail, forcing lane-for-lane bit equality.
**`crates/wukong_runtime/src/rmsnorm_bwd.rs`**

- **RMSNorm-backward fused two-reduction row** — `crates/wukong_runtime/src/rmsnorm_bwd.rs:57-110` — One pass folds Σx² and Σg·x into two 8-lane accumulators, then applies `dx = r*fma(x, -(r²·s/C), g)`, one FMA per element.
- **RMSNorm-backward AVX2 twin** — `crates/wukong_runtime/src/rmsnorm_bwd.rs:123-196` — AVX2/FMA row with identical lane folding and negated-coef FMA apply; its doc-comment permits `dx` aliasing `dy` while the public entry doc at `crates/wukong_runtime/src/rmsnorm_bwd.rs:220-248` forbids overlapping `x`.
**`crates/wukong_runtime/src/rope_bwd.rs`**

- **RoPE backward scalar row (transpose rotation)** — `crates/wukong_runtime/src/rope_bwd.rs:44-69` — Applies the inverse rotation `[[c,s],[-s,c]]` to the upstream gradient, differing from the forward only in the two combine signs, using the same shared Cephes `sincos1`.
- **RoPE backward AVX2 row kernel** — `crates/wukong_runtime/src/rope_bwd.rs:71-114` — Eight gradient pairs per step with `sin8`/`cos8` and `fmadd`/`fnmadd` mirroring the scalar twin lane-for-lane, plus a scalar `half%8` tail.
- **RoPE backward serial entry and AVX2 gate** — `crates/wukong_runtime/src/rope_bwd.rs:131-158` — `wukong_rope_bwd_f32` walks `[rows,2*half]` gradient rows with row index as position, no-ops on non-positive dims, allows `dx` aliasing `g`; gate at `crates/wukong_runtime/src/rope_bwd.rs:116-129`.
**`crates/wukong_runtime/src/softmax_bwd.rs`**

- **Softmax-backward elementwise apply kernels** — `crates/wukong_runtime/src/softmax_bwd.rs:25-58` — AVX2 (avx2 only, no fma) `y*(dy−s)` sub-then-mul over 8 lanes with a scalar twin, deliberately FMA-free so vector and tail agree lane-for-lane.
- **Softmax-backward row routine reusing the proven dot** — `crates/wukong_runtime/src/softmax_bwd.rs:60-76` — Delegates `s = Σ y·dy` to `wukong_sreduce_f32(RED_DOT)` so the kernel introduces no new float-accumulation order, then applies the elementwise tail.
- **Softmax-backward C-ABI entries and parallel threshold** — `crates/wukong_runtime/src/softmax_bwd.rs:78-141` — Serial row loop plus a rayon row-map gated by `SOFTMAX_BWD_PAR_MIN = 8`; the safety doc forbids `dx` overlapping `y` although the code loads both operands before storing each block.
**`crates/wukong_runtime/src/vmath.rs`**

- **Fused activation-backward kernels (six ops)** — `crates/wukong_runtime/src/vmath.rs:1041-1117` — `dx = dy·act'(x)` for silu/gelu/sigmoid/tanh/elu/softplus folding the upstream multiply into the 256-bit derivative; AVX2 twins at `crates/wukong_runtime/src/vmath.rs:1242-1335`.
- **Gated-FFN kernels (SwiGLU/GeGLU/GLU)** — `crates/wukong_runtime/src/vmath.rs:1119-1137` — `out = act(a)·b` reusing the forward silu/gelu/sigmoid so the gate is bit-identical to the unfused composition; AVX2 twins at `crates/wukong_runtime/src/vmath.rs:1211-1240`.
- **Backward/gate f64 closed-form oracle** — `crates/wukong_runtime/src/vmath.rs:2627-2693` — checks the six backward kernels against independent f64 derivatives plus bit-exact tail match, proving real gradients; gate equivalent at `crates/wukong_runtime/src/vmath.rs:2699-2741`.
**`crates/wukong_runtime/src/xent_bwd.rs`**

- **Cross-entropy backward softmax-minus-onehot** — `crates/wukong_runtime/src/xent_bwd.rs:68-105` — Stores exp(x−m) into `dx` while accumulating Z, scales by 1/Z, then subtracts exactly 1.0 at the target column; supports in-place `dx == x`.
- **Cross-entropy backward AVX2 twin** — `crates/wukong_runtime/src/xent_bwd.rs:118-164` — Store-then-accumulate `exp8` pass plus a separate normalize sweep that re-reads `dx`, reproducing softmax's writeback bit-for-bit before the scalar onehot fixup.
**`crates/wukong_runtime/src/xent.rs`**

- **Softmax cross-entropy forward row loss** — `crates/wukong_runtime/src/xent.rs:64-93` — Stable row-max plus Σexp(x−m) reduction capped by one `log1` and a scalar target-logit gather, yielding `lse − x[target]` without materializing softmax.
- **Cross-entropy forward AVX2 twin** — `crates/wukong_runtime/src/xent.rs:105-136` — `exp8` 8-lane body with an `exp1` tail folded into the same lanes, so the `lse` bits match a dispatched softmax's reduction exactly.

## G29 — Runtime: dispatch table, ABI, thread pool, environment knobs

*Files:* `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_runtime/src/bias.rs`, `crates/wukong_runtime/src/colarg.rs`, `crates/wukong_runtime/src/colreduce.rs`, `crates/wukong_runtime/src/dequant.rs`, `crates/wukong_runtime/src/gemm.rs`, `crates/wukong_runtime/src/gemv.rs`, `crates/wukong_runtime/src/gevm.rs`, `crates/wukong_runtime/src/layernorm_bwd.rs`, `crates/wukong_runtime/src/lib.rs`, `crates/wukong_runtime/src/logsoftmax.rs`, `crates/wukong_runtime/src/lowp.rs`, `crates/wukong_runtime/src/norm.rs`, `crates/wukong_runtime/src/pool2d.rs`, `crates/wukong_runtime/src/reduce.rs`, `crates/wukong_runtime/src/rowarg.rs`, `crates/wukong_runtime/src/velem.rs`, `crates/wukong_runtime/src/vmath.rs`, `crates/wukong_runtime/src/xent.rs`

**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **Runtime symbol-name constant table** — `crates/wukong_codegen_cranelift/src/lib.rs:104-279` — Roughly 180 `RT_*` string constants naming every dispatched CPU kernel and its `_parallel` twin, the single source of truth shared by JIT binding and object imports.
- **Heap allocation and monotonic clock intrinsics** — `crates/wukong_codegen_cranelift/src/lib.rs:1451-1475` — Lowers `wukong_rt_alloc(count, elem_size, is_float)->ptr`, `wukong_rt_free`, and the zero-argument `wukong_now_ns()->i64` used for in-process benchmarking.
- **File-I/O intrinsic family lowering** — `crates/wukong_codegen_cranelift/src/lib.rs:1476-1502` — One arm lowers all twelve read/write × {f32,f64,i32,i64,i8,u8} calls sharing a (path, data, len)->i64 status ABI; the nearby comment mis-states "eight dtypes/directions".
- **Codegen environment knobs: verifier and parallel codegen** — `crates/wukong_codegen_cranelift/src/lib.rs:2041-2061` — `WUKONG_CL_VERIFY` toggles the check-only IR verifier (default on only in debug builds) and `WUKONG_PAR_CODEGEN=0` forces serial per-function compilation.
**`crates/wukong_mir_build/src/lib.rs`**

- **Runtime kernel symbol pool (GemmSyms)** — `crates/wukong_mir_build/src/lib.rs:679-840` — Interns ~150 runtime entry-point names (GEMM/GEMV, vmath, norms, scans, losses, quant, file-I/O, alloc) once, each with a `_parallel` twin; also smuggles print_str/print_u.
**`crates/wukong_runtime/src/bias.rs`**

- **Non-temporal store gate, alignment peel, and software prefetch** — `crates/wukong_runtime/src/bias.rs:129-138` — Switches to `vmovntps` once total streamed bytes (x + out) reach 10 MiB, with a scalar 32-byte-alignment prologue and `sfence`, prologue at `crates/wukong_runtime/src/bias.rs:87-96`.
**`crates/wukong_runtime/src/colarg.rs`**

- **Column arg-reduce stripe parallelism and exported entries** — `crates/wukong_runtime/src/colarg.rs:209-244` — 8-aligned disjoint output stripes per core below a 256-column serial threshold (`crates/wukong_runtime/src/colarg.rs:36-39`); four `#[no_mangle]` entries at `crates/wukong_runtime/src/colarg.rs:246-294`.
**`crates/wukong_runtime/src/colreduce.rs`**

- **Column-stripe parallel split for column reductions** — `crates/wukong_runtime/src/colreduce.rs:309-345` — Splits columns into 8-aligned disjoint stripes below/above a 256-column threshold; uses the global rayon pool, not `run_on_wuk_pool` like reduce.rs.
**`crates/wukong_runtime/src/dequant.rs`**

- **Dequant non-temporal store threshold** — `crates/wukong_runtime/src/dequant.rs:117-122` — `use_nt` switches to `vmovntps` once `n·(in_bytes+4)` reaches the 10 MiB `NT_MIN_BYTES` at `crates/wukong_runtime/src/dequant.rs:56-60`, avoiding read-for-ownership on tensor-sized streams.
**`crates/wukong_runtime/src/gemm.rs`**

- **Parallel gate `PAR_MIN_MACS` / `WUKONG_GEMM_PAR_MIN_MACS`** — `crates/wukong_runtime/src/gemm.rs:33-54` — MAC-count threshold (2^23) below which the parallel dispatch collapses to the serial AVX2 kernel; env-overridable once-read knob consumed in `gemm_dispatch`.
- **Work-scaled per-task MAC budget `min_task_macs`** — `crates/wukong_runtime/src/gemm.rs:56-75` — Caps effective worker count for the shared-pack path so near-gate problems get few large blocks; doc claims 1 Mi best but the coded default is 4 Mi, used at `crates/wukong_runtime/src/gemm.rs:2185-2192`.
- **Private physical-core rayon pool (`gemm_pool`), now the crate-wide unified pool** — `crates/wukong_runtime/src/gemm.rs:96-141` — Builds a HyperThread-free 16 MiB-stack pool once, falling back to the global pool when physical ≥ logical; other runtime kernels route here too.
- **Worker CPU-pinning probe `WUKONG_GEMM_AFFINITY`** — `crates/wukong_runtime/src/gemm.rs:143-167` — Windows SetThreadAffinityMask start-handler pinning P/E cores, kept default-off as a recorded refutation (measured 25-35% slower than free migration); also `crates/wukong_runtime/src/gemm.rs:135-137`.
- **Parallel-path selector knob family (FORKJOIN / 2D / 2D_SHARED / DYN / STEAL_ORDER / SKINNY_A)** — `crates/wukong_runtime/src/gemm.rs:169-292` — Six OnceLock env gates choosing among five parallel GEMM shapes, each documenting the adjacent-run A/B measurement that made it default or refuted it.
- **Block-grain override `WUKONG_GEMM_TASK_MACS` and `blocks_target_from`** — `crates/wukong_runtime/src/gemm.rs:294-327` — Replaces the default 3×nworkers block target with macs/budget, clamped to [1, 2^20]; pure policy pinned by a unit test at `crates/wukong_runtime/src/gemm.rs:3902-3914`.
- **C-tile prefetch knob `WUKONG_GEMM_PF_C`** — `crates/wukong_runtime/src/gemm.rs:329-341` — Default-on gate for the microkernel's per-row C prefetch, hiding writeback demand misses at large N; consumed at `crates/wukong_runtime/src/gemm.rs:2750-2755`.
- **Cooperative pack region `pack_ab_par` with byte gate and split-region A/B** — `crates/wukong_runtime/src/gemm.rs:2577-2683` — Fuses A- and B-panel packing into one rayon region above `WUKONG_PACK_PAR_MIN_KB` bytes, else packs serially; threshold at `crates/wukong_runtime/src/gemm.rs:2560-2575`.
**`crates/wukong_runtime/src/gemv.rs`**

- **Parallel GEMV: rows across cores with `GEMV_PAR_MIN_ROWS` gate** — `crates/wukong_runtime/src/gemv.rs:160-242` — Rayon maps independent rows onto cores through the same per-row routine (bit-identical to serial), falling back to serial below 64 rows, threshold at `crates/wukong_runtime/src/gemv.rs:200-203`.
**`crates/wukong_runtime/src/gevm.rs`**

- **GEVM column-stripe parallelism (8-aligned stripes, `GEVM_PAR_MIN`=256)** — `crates/wukong_runtime/src/gevm.rs:164-219` — Splits output columns (never rows) into per-core stripes rounded up to multiples of 8, so every column keeps its full ascending-i chain and bits match serial.
**`crates/wukong_runtime/src/layernorm_bwd.rs`**

- **Integer-only kernel ABI and degenerate-shape no-op guard** — `crates/wukong_runtime/src/layernorm_bwd.rs:320-340` — Every entry takes i64 rows/cols, returns silently when either is `<= 0`, and decodes eps from `f32::from_bits(eps_bits as u32)`; also `crates/wukong_runtime/src/rmsnorm_bwd.rs:228-248`.
**`crates/wukong_runtime/src/lib.rs`**

- **Stale crate-header contract doc** — `crates/wukong_runtime/src/lib.rs:1-6` — Header claims the runtime is "Arena, parallel_for, wukong_sgemm" for "native (LLVM) builds", but the real surface is ~150 Cranelift-linked kernels.
- **Kernel module wiring and C-ABI re-export surface** — `crates/wukong_runtime/src/lib.rs:8-167` — Declares 33 kernel modules and re-exports every `wukong_*` symbol plus op-code constants, mostly as serial/`_parallel` pairs the Cranelift backend links by name.
- **Bump Arena allocator** — `crates/wukong_runtime/src/lib.rs:243-291` — Owned-Vec arena with power-of-two-aligned `alloc` returning offsets, overflow-checked, `None` on exhaustion, and all-at-once `reset` for kernel scratch.
- **Sequential reference `parallel_for`** — `crates/wukong_runtime/src/lib.rs:293-307` — Despite the name this Rust-level helper is strictly sequential with a `step<=0` early return; it is the deterministic reference, not the native lowering target.
- **`EnvAddr` Send/Sync pointer smuggler** — `crates/wukong_runtime/src/lib.rs:309-315` — Wraps the raw closure-env address as a `usize` so rayon workers can carry it, justified by disjoint output indices and the blocking join.
- **Global rayon pool with 16 MiB worker stacks** — `crates/wukong_runtime/src/lib.rs:317-344` — `Once`-guarded `build_global` that every parallel entry must call first, because outlined `@parallel` bodies privatize ~1.5 MiB of stack scratch that rayon's default 2 MiB stacks cannot hold.
- **`WUKONG_POOL_UNIFY` pool-unification knob** — `crates/wukong_runtime/src/lib.rs:346-359` — Default-ON `OnceLock` flag routing model kernels onto the private physical-core GEMM pool to kill private/global park-unpark churn; `=0` restores global-pool routing as an A/B instrument.
- **`run_on_wuk_pool` unified-pool installer** — `crates/wukong_runtime/src/lib.rs:361-375` — Installs work on the x86_64 private GEMM pool when unification is on and it exists, else falls back to the global pool, relying on rayon's inline-nested `install` semantics.
- **`wuk_pool_width` serial-vs-parallel dispatch policy** — `crates/wukong_runtime/src/lib.rs:377-395` — Reports the unified pool's worker count so `_parallel` kernels can take the width==1 serial fast path, explicitly excluding `wukong_parallel_for` which needs the 16 MiB worker stacks.
- **`WUKONG_PFOR_DYN` scheduling kill-switch** — `crates/wukong_runtime/src/lib.rs:469-477` — Read-once default-ON flag selecting dynamic claiming over the static split; scheduling-only, so results are bit-identical either way.
- **Zeroed heap allocation with hidden size header** — `crates/wukong_runtime/src/lib.rs:479-514` — `wukong_rt_alloc` backs `alloc_<T>(n)` with a 16-byte header storing the layout size, guaranteeing 16-byte-aligned zeroed data and null on zero/negative/overflowing sizes; `elem_is_float` is accepted but ignored.
- **`wukong_rt_free` header-reconstructing deallocator** — `crates/wukong_runtime/src/lib.rs:516-534` — Reads the size back out of the 16-byte header to rebuild the `Layout`, treating null as a no-op; misuse is UB natively but silently survives in the interpreter.
- **`wukong_now_ns` monotonic in-process clock** — `crates/wukong_runtime/src/lib.rs:653-667` — Lazily captured `Instant` epoch in a `OnceLock` gives non-decreasing nanoseconds for in-program benchmarking; programs must print only differences to stay backend-identical.
**`crates/wukong_runtime/src/logsoftmax.rs`**

- **Batched logsoftmax/logsumexp entries with LOGSOFTMAX_PAR_MIN** — `crates/wukong_runtime/src/logsoftmax.rs:226-338` — Four C-ABI entries where the parallel forms fall back to serial below 8 rows and otherwise use rayon's *global* pool, diverging from norm.rs's `run_on_wuk_pool` routing.
**`crates/wukong_runtime/src/lowp.rs`**

- **Half-output NT threshold and prefetch-distance knobs** — `crates/wukong_runtime/src/lowp.rs:800-815` — `HALFOUT_NT_MIN_BYTES` (10 MiB, compared against `6·n`) flips the narrowing store to non-temporal past L3 and `HALFOUT_PF_AHEAD` sets the 256-element input software prefetch.
**`crates/wukong_runtime/src/norm.rs`**

- **wukong_norm_f32 serial entry and eps-as-bits ABI** — `crates/wukong_runtime/src/norm.rs:725-748` — All-integer C ABI passing eps as `f32::to_bits` in an i64, guarding non-positive rows/cols as a no-op and striding rows by `cols`.
- **wukong_norm_f32_parallel pool routing** — `crates/wukong_runtime/src/norm.rs:750-795` — Forks rows on the unified `run_on_wuk_pool` with pointers smuggled as usize, short-circuiting only on `wuk_pool_width()<=1` and, unlike the sibling files, applying no minimum-row threshold.
- **Affine norm C-ABI entries, serial and parallel** — `crates/wukong_runtime/src/norm.rs:797-885` — Row-major `[rows,cols]` driver with row-shared length-`cols` gamma/beta (nullable, round-tripped through usize across rayon) and the same pool-width serial fast path.
**`crates/wukong_runtime/src/pool2d.rs`**

- **Pooling stride-specialisation dispatch and generic scalar twin** — `crates/wukong_runtime/src/pool2d.rs:268-330` — `pool_channel` routes `sw==1` and `sw==2` to the AVX2 bands only when runtime AVX2 is detected; every other stride (including sw>2) falls to the any-stride scalar nest.
**`crates/wukong_runtime/src/reduce.rs`**

- **Multicore reduction entry with pool gating** — `crates/wukong_runtime/src/reduce.rs:249-296` — Rayon indexed `map().collect()` preserves chunk order, then folds ascending; falls back to serial when chunks < 2 or `wuk_pool_width() <= 1`.
**`crates/wukong_runtime/src/rowarg.rs`**

- **Row arg-reduce parallel map and exported entries** — `crates/wukong_runtime/src/rowarg.rs:152-180` — Maps the per-row routine across rows with rayon above a 64-row threshold (`crates/wukong_runtime/src/rowarg.rs:26-29`); four C entries at `crates/wukong_runtime/src/rowarg.rs:182-229`.
**`crates/wukong_runtime/src/velem.rs`**

- **Non-temporal store policy: NT_MIN_BYTES / use_nt / velem_streams** — `crates/wukong_runtime/src/velem.rs:52-80` — NT stores engage once total streamed bytes (streams x n x 4, streams counted from the opcode) reach 10 MiB, keying on working set not length.
- **WUKONG_VELEM_PAR_MIN parallel-gate knob** — `crates/wukong_runtime/src/velem.rs:91-106` — OnceLock-cached env override (default 256 Ki elements) below which the parallel entry runs serial; also bails when the pool width is 1.
**`crates/wukong_runtime/src/vmath.rs`**

- **wukong_vmath_f32 C entry point and feature gate** — `crates/wukong_runtime/src/vmath.rs:731-749` — runtime `is_x86_feature_detected!` avx2+fma check selecting the vector kernel else a scalar loop; note the AVX2 path leaves `out` untouched for unknown ops while the scalar path writes identity.
- **Non-temporal store crossover heuristic** — `crates/wukong_runtime/src/vmath.rs:853-869` — `use_nt` streams stores once total bytes across all live arrays reach 10 MiB (L3), copied verbatim from velem so both kernels cross identically.
- **WUKONG_VMATH_UNROLL=6 A/B knob** — `crates/wukong_runtime/src/vmath.rs:871-882` — OnceLock-cached env switch selecting a ×6 (48-element) unroll variant ahead of the shipped ×4 body, for FMA-port versus register-pressure measurement.
**`crates/wukong_runtime/src/xent.rs`**

- **Per-row AVX2+FMA runtime feature-detect dispatchers** — `crates/wukong_runtime/src/xent.rs:142-151` — Seven thin `*_row` selectors calling `is_x86_feature_detected!` per row (not hoisted per call), also at `crates/wukong_runtime/src/layernorm_bwd.rs:289-308`, `crates/wukong_runtime/src/rmsnorm_bwd.rs:202-218`, `crates/wukong_runtime/src/entropy.rs:116-125`.

# Autodiff

## G30 — Autodiff: reverse-mode MIR to MIR, tape, VJP rules, fused optimizers

*Files:* `crates/wukong_autodiff/src/lib.rs`, `crates/wukong_autodiff/src/optim.rs`, `crates/wukong_autodiff/src/tape.rs`, `crates/wukong_autodiff/src/tests.rs`, `crates/wukong_driver/src/lib.rs`

**`crates/wukong_autodiff/src/lib.rs`**

- **`grad()` entry point and forward-contract validation** — `crates/wukong_autodiff/src/lib.rs:54-88` — Rejects multi-block functions and non-`Ptr` `wrt` params, interns kernel symbols, then drives replay/seed/reverse to produce `{name}_grad`.
- **Vjp state model: five maps plus gradient-param appending** — `crates/wukong_autodiff/src/lib.rs:94-158` — Holds `fwd_to_new`/`adj`/`grad_buf`/`buf_adj`/`buf_count`/`contributed`, appends one `Ptr` grad param per `wrt`, and pre-indexes defining ops and alloca element counts.
- **Forward replay with full operand remapping** — `crates/wukong_autodiff/src/lib.rs:166-180` — Re-emits every forward instruction into the new value space so the reverse pass can reference any intermediate; the remap covers all 20 `Op` variants at `crates/wukong_autodiff/src/lib.rs:420-463`.
- **Loss seeding and reverse-order driver** — `crates/wukong_autodiff/src/lib.rs:183-214` — Requires a `ret <loss>` float terminator, seeds its adjoint to 1.0, then walks instructions in reverse applying VJP rules and emits `ret void`.
- **Kernel-call routing and loud refusal of unrecognized buffer writers** — `crates/wukong_autodiff/src/lib.rs:218-240` — Routes recognized kernel calls to the tape, refuses any void-result unknown call, and skips stores (assumed loss sink) rather than silently zeroing gradients.
- **Scalar VJP rule table (fadd/fsub/fmul/fdiv/neg/fma/sqrt/select/const leaf)** — `crates/wukong_autodiff/src/lib.rs:252-330` — The straight-line SSA arithmetic rules; sqrt reuses the replayed forward root, constants terminate adjoints, anything else errors via `op_name` at `crates/wukong_autodiff/src/lib.rs:467-490`.
- **Load-adjoint routing into gradient buffers (read-add-write)** — `crates/wukong_autodiff/src/lib.rs:334-359` — Accumulates a loaded element's adjoint into the matching grad buffer element so repeated loads sum; silently drops loads from non-`wrt` inputs.
- **Buffer-pointer provenance resolver** — `crates/wukong_autodiff/src/lib.rs:364-378` — Accepts only a bare parameter or a one-level `gep` of a parameter, erroring on anything deeper so a gradient is never misrouted.
- **Adjoint accumulation and scalar emit helpers** — `crates/wukong_autodiff/src/lib.rs:382-417` — `accum` initializes-or-sums an SSA value's adjoint; thin `fadd/fmul/fdiv/neg/select/fconst` builders keep every rule short and type-preserving.
**`crates/wukong_autodiff/src/optim.rs`**

- **Fused AdamW update kernel as MIR** — `crates/wukong_autodiff/src/optim.rs:45-99` — One counted-loop pass doing moments, bias correction, sqrt and decoupled weight decay; hyperparameter buffer layout at `crates/wukong_autodiff/src/optim.rs:20-31`, loop helper `crates/wukong_autodiff/src/optim.rs:103-120`, size-keyed name `crates/wukong_autodiff/src/optim.rs:161-163`.
**`crates/wukong_autodiff/src/tape.rs`**

- **Interned kernel symbol table and `is_kernel` recognition gate** — `crates/wukong_autodiff/src/tape.rs:56-108` — Names the nine recognized runtime kernels including the four `_parallel` twins that must mirror mir_build's recognizers; the acceptance test is at `crates/wukong_autodiff/src/tape.rs:135-145`.
- **Buffer-argument canonicalization and constant-operand extraction** — `crates/wukong_autodiff/src/tape.rs:153-170` — Peels whole-buffer `gep(buf, 0)` through casts so both lowering spellings key one buffer-adjoint entry; `const_i64` at `crates/wukong_autodiff/src/tape.rs:808-816` demands compile-time dims/op codes.
- **Kernel-call VJP dispatcher** — `crates/wukong_autodiff/src/tape.rs:173-197` — Canonicalizes every buffer operand then routes to the sreduce/sgemm_nt/vmath/velem/norm rule, accepting serial and `@parallel` symbol variants identically.
- **Reduction VJP: SUM, DOT and SSD scalar-to-buffer seeding** — `crates/wukong_autodiff/src/tape.rs:204-259` — Turns the scalar loss adjoint into buffer gradients via velem fill/scale/affine calls; the error string claims only SUM and SSD though DOT is implemented.
- **`sgemm_nt` (nn.Linear) VJP: dA = dC·B and dB = dCᵀ·A** — `crates/wukong_autodiff/src/tape.rs:267-297` — Emits two `wukong_sgemm` calls with `beta` accumulation, transposing dC through the synthesized flat loop at `crates/wukong_autodiff/src/tape.rs:716-727` and helper `crates/wukong_autodiff/src/tape.rs:654-669`.
- **Activation VJP routing: fused vmath2 backward vs synthesized loop** — `crates/wukong_autodiff/src/tape.rs:304-339` — Sends silu/gelu/elu/softplus to the fused `wukong_vmath2_f32` `*_BWD` codes and the algebraic activations to a loop; code map at `crates/wukong_autodiff/src/tape.rs:113-121`, op-code constants at `crates/wukong_autodiff/src/tape.rs:21-48`.
- **Streaming affine (velem) VJP, identity activation only** — `crates/wukong_autodiff/src/tape.rs:347-375` — Emits `dx = a·dout`, `dy = b·dout` (the residual-add / saxpy backward), gating on the `USE_Y` flag and rejecting non-identity activations loudly.
- **Norm VJP dispatcher with eps-bit decoding** — `crates/wukong_autodiff/src/tape.rs:384-415` — Selects softmax/LayerNorm/RMSNorm backward and reconstructs the f32 eps from its i64 bit pattern; all three use the per-row `sreduce` helper at `crates/wukong_autodiff/src/tape.rs:529-548`.
- **Softmax backward `dx = y ⊙ (dy − Σ dy·y)`** — `crates/wukong_autodiff/src/tape.rs:418-441` — Nested rows×cols loops with one per-row DOT reduction, reusing the forward normalized output instead of recomputing exponentials.
- **LayerNorm backward with recomputed row statistics** — `crates/wukong_autodiff/src/tape.rs:447-487` — Recovers variance as `E[x²] − μ²` from SUM and SUMSQ row reductions, so the forward norm must not overwrite `x` in place.
- **RMSNorm backward `dx = (1/r)(dy − y·mean(dy·y))`** — `crates/wukong_autodiff/src/tape.rs:492-525` — Recomputes the row RMS from a SUMSQ reduction plus eps, then combines elementwise in the inner loop.
- **Algebraic activation-derivative loop body** — `crates/wukong_autodiff/src/tape.rs:552-600` — Emits `dx[i] = dout[i]·f'` for relu (mask on `x`), sigmoid (`y(1−y)`), tanh (`1−y²`) and exp (`y`), erroring on any other op code.
- **Gradient-buffer allocation policy** — `crates/wukong_autodiff/src/tape.rs:607-624` — Resolves a forward buffer to an existing adjoint, the appended grad param, a fresh sized alloca, or `None` for non-differentiated inputs; unknown intermediate sizes error.
- **Multi-contribution policy: `single()` refusal vs matmul `beta` accumulation** — `crates/wukong_autodiff/src/tape.rs:629-645` — Overwrite-style ops error loudly on a second gradient contribution, while matmul writes switch `beta` 0→1 so fan-out inputs accumulate correctly.
- **velem emit helpers: fill, scale, affine** — `crates/wukong_autodiff/src/tape.rs:672-710` — Three one-call spellings of `wukong_velem_f32` that express broadcast-fill, scalar scaling, and two-input affine combines used by every reduction and velem VJP.
- **Synthesized counted-loop builder (`open_loop`/`close_loop`)** — `crates/wukong_autodiff/src/tape.rs:729-760` — Builds header/body/exit blocks with an i64 block-param induction variable, the only control flow the backward introduces; the `Loop` handle is at `crates/wukong_autodiff/src/tape.rs:126-131`.
**`crates/wukong_autodiff/src/tests.rs`**

- **f64 finite-difference correctness gate** — `crates/wukong_autodiff/src/tests.rs:51-124` — Runs analytic and central-difference gradients on the interpreter and asserts elementwise agreement plus tight closed-form checks; the core oracle for every scalar rule.
- **End-to-end trainability tests (scalar MLP and kernel MLP2)** — `crates/wukong_autodiff/src/tests.rs:340-461` — Assert monotone SGD loss descent and >10× reduction, proving gradients drive real learning; the two-GEMM kernel version lives at `crates/wukong_autodiff/src/tests.rs:1177-1338`.
- **f32 tensor-tape gates: dual FD+closed-form and FD-only** — `crates/wukong_autodiff/src/tests.rs:554-587` — Pairs a loose f32 finite difference with a tight f64 closed form; the FD-only variant at `crates/wukong_autodiff/src/tests.rs:866-887` adds a non-trivially-zero guard against vacuous passes.
- **Pre-norm transformer block composite VJP test** — `crates/wukong_autodiff/src/tests.rs:1091-1163` — Chains rmsnorm→sgemm_nt→silu→sum so one reverse pass exercises all four kernel backward families together, FD-gated.
- **AdamW reference gate and optimizer training test** — `crates/wukong_autodiff/src/tests.rs:1371-1508` — Five bias-corrected steps checked against an f64 AdamW reference, then 300 steps against a teacher target asserting substantial loss reduction.
- **`@parallel` region must decline loudly (coupling contract test)** — `crates/wukong_autodiff/src/tests.rs:1517-1564` — Builds mir_build's `wukong_parallel_for` env-pack shape and asserts `grad` returns the "no VJP rule" error rather than a silently zero gradient.
**`crates/wukong_driver/src/lib.rs`**

- **`--train` fwd→bwd→optimizer loop with SGD and fused AdamW** — `crates/wukong_driver/src/lib.rs:809-990` — Seeds buffers from a deterministic LCG, builds one AdamW kernel per distinct trainable size, runs the gradient kernel each step, and prints the loss trajectory.
- **Finite-difference correctness gate for emitted backwards** — `crates/wukong_driver/src/lib.rs:1138-1164` — Central-difference gradient over every element versus the analytic gradient with rel+abs tolerance; the only real oracle since recognized backward kernels are gate-blind.
- **Backward-rides-the-tuned-kernel MIR assertions** — `crates/wukong_driver/src/lib.rs:1295-1470` — Prints the `{loss, loss_grad}` MIR and asserts silu/gelu backwards call `wukong_vmath2_f32` and the FFN block emits exactly three `wukong_sgemm` adjoints.

# GPU

## G31 — GPU: device management, memory, host dispatch API

*Files:* `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/lib.rs`, `crates/wukong_codegen_gpu/src/ptx_optim.rs`, `crates/wukong_driver/src/gpu_accel.rs`, `crates/wukong_interp/src/lib.rs`

**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **Gpu handle: context, stream and in-process PTX module cache** — `crates/wukong_codegen_gpu/src/gpu.rs:18-61` — Owns one CUDA context, default stream and a `&'static str`-keyed module map so PTX is JIT-loaded once per process per kernel family.
- **Device attribute probes: SM count, L2 size, peak HBM bandwidth** — `crates/wukong_codegen_gpu/src/gpu.rs:92-130` — Queries `cuDeviceGetAttribute` for grid sizing and honest perf denominators; `peak_hbm_gbs` reproduces deviceQuery's `2×memClock×busWidth/8`, `sm_count` defaults to 20.
- **Process-wide GPU singleton with graceful no-device skip** — `crates/wukong_codegen_gpu/src/gpu.rs:133-147` — A `OnceLock<Mutex<Option<Gpu>>>` serializes all driver calls across test threads and yields `None` (skip, not fail) on GPU-less machines.
- **Sticky-fault device-lost detection and primary-context reset** — `crates/wukong_codegen_gpu/src/gpu.rs:149-207` — Drops the poisoned context, calls `cuDevicePrimaryCtxReset_v2`, and on failure latches a `DEVICE_LOST` flag so corpus runs skip loudly instead of cascading false failures.
- **Bit-exact saxpy and vadd host entries** — `crates/wukong_codegen_gpu/src/gpu.rs:209-242` — Copy-in/launch/copy-out wrappers whose fused `fma.rn` saxpy and IEEE add are documented to match CPU references bit-for-bit, making them cheap correctness gates.
- **4×-ILP streaming-copy launch grid and HBM bandwidth kernel** — `crates/wukong_codegen_gpu/src/gpu.rs:244-266` — `stream_cfg` divides the float4 count by four to route work through the kernel's 4-wide fast path (~84%→90% of peak); grid-stride keeps correctness grid-independent.
- **f32→f16→f32 cast round-trip entry** — `crates/wukong_codegen_gpu/src/gpu.rs:268-283` — Exercises the `cast_f32_f16` kernel claimed round-to-nearest-even identical to `half::f16::from_f32`, the glue for resident fp16 pipeline stage boundaries.
- **GPU vmath activation dispatch and support gate** — `crates/wukong_codegen_gpu/src/gpu.rs:285-324` — Maps `VM_*` codes to six PTX entries (relu/exp/sigmoid/tanh/silu/gelu); `vmath_supported` duplicates that list so offload falls back to CPU instead of panicking.
- **Deterministic GPU reduction with fixed decomposition** — `crates/wukong_codegen_gpu/src/gpu.rs:326-383` — Fixed 256×256 grid plus ascending host-side combine makes sum/dot/max run-to-run identical; `RED_MAX` is hand-copied from the runtime, a desync landmine.
**`crates/wukong_codegen_gpu/src/lib.rs`**

- **Crate public surface and `gpu` feature gating** — `crates/wukong_codegen_gpu/src/lib.rs:1-134` — Declares every GPU module behind `#[cfg(feature = "gpu")]` (so a plain `cargo test` compiles an empty crate), exports `GPU_ENABLED`, `Gpu`/`available`, `lower_jit_run`, and leaves `paged_kv` un-gated as pure host policy.
**`crates/wukong_codegen_gpu/src/ptx_optim.rs`**

- **Grid-stride occupancy sizing heuristic** — `crates/wukong_codegen_gpu/src/ptx_optim.rs:209-218` — 256-thread blocks capped at 32 blocks per SM and never more blocks than elements; shared by every memory-bound optimizer and backward kernel.
**`crates/wukong_driver/src/gpu_accel.rs`**

- **`GpuAccel` accelerator bridge with structural decline rules** — `crates/wukong_driver/src/gpu_accel.rs:29-138` — Implements interp's `Accelerator` for gemm/vmath/norm/reduce/epilogue, counting device calls and declining (never silently CPU-erroring) on unsupported opcodes or unaligned `beta/m/n/k` shapes.
**`crates/wukong_interp/src/lib.rs`**

- **Accelerator offload seam for the GPU backend** — `crates/wukong_interp/src/lib.rs:74-139` — Five-method trait (sgemm_nt, vmath, norm, sreduce, sgemm_nt_epi) whose `None` return means fall back to the CPU kernel; wired at `crates/wukong_interp/src/lib.rs:198-224`, `crates/wukong_interp/src/lib.rs:1270-1282`, `crates/wukong_interp/src/lib.rs:1745-1756`, `crates/wukong_interp/src/lib.rs:3554-3568`.

## G32 — GPU: GEMM and tensor-core PTX kernels

*Files:* `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/ptx.rs`, `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs`, `crates/wukong_codegen_gpu/src/ptx_gemm.rs`, `crates/wukong_codegen_gpu/src/ptx_wmma.rs`

**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **f32 GEMM entries: 16×16 tiled NN/NT plus register-blocked 64×64** — `crates/wukong_codegen_gpu/src/gpu.rs:385-485` — Baseline non-tensor-core `A·Bᵀ` and `A·B` launchers with their two launch-config helpers, the tolerance-gated reference shape for every later variant, also `crates/wukong_codegen_gpu/src/gpu.rs:422-458`.
- **WMMA multi-tile vs single-tile entry picker** — `crates/wukong_codegen_gpu/src/gpu.rs:487-512` — Chooses the fragment-reuse `<base>_mt` kernel when M,N divide the warp tile, else the any-16-multiple base, returning name and grid together.
- **fp16 GEMM regime dispatch ladder** — `crates/wukong_codegen_gpu/src/gpu.rs:514-600` — Six-arm size dispatch keyed on the fp16 A+B working set (≥48MB v2cs cliff, ≥16MB swizzled w24 workhorse, ≤1024 pipe64_s6, pipe128_s4, `_sm`, `_mt`).
- **Shared-memory-staged WMMA GEMM family and configs** — `crates/wukong_codegen_gpu/src/gpu.rs:602-649` — `_sm` 64×64, `_sm128` single-buffered 128×128 and the `cp.async` double-buffered twins, each with a matching CTA-per-tile launch config, also `crates/wukong_codegen_gpu/src/gpu.rs:697-779` and `crates/wukong_codegen_gpu/src/gpu.rs:993-1029`.
- **Static-shape fp16 SMEM GEMM (per-shape PTX, bypasses module cache)** — `crates/wukong_codegen_gpu/src/gpu.rs:651-695` — Bakes M/N/K into generated PTX and raw-loads it so ptxas folds strides; picks the 128 tile only at M,N≥4096.
- **WMMA `sm_db` fused epilogue family: activation, bias, bias+activation** — `crates/wukong_codegen_gpu/src/gpu.rs:781-991` — Two shared launchers plus seven thin wrappers fold relu/silu/gelu and a per-column bias into the C store, removing cuBLAS's second HBM round-trip kernel.
- **WMMA fused residual (accumulator-seeded skip connection)** — `crates/wukong_codegen_gpu/src/gpu.rs:857-894` — Passes an M×N residual that seeds the f32 wmma accumulators, making the transformer skip add free instead of a separate add kernel.
- **Raster-aware pipeline launch config and multi-stage `cp.async` GEMM** — `crates/wukong_codegen_gpu/src/gpu.rs:1031-1077` — `pipe_cfg` emits a 1-D grid when the variant self-rasterizes (`raster>0`) and 2-D otherwise; `gemm_nt_f16_pipe` drives any `PipeCfg` variant.
- **Cliff-variant dispatch route into `CLIFF_VARIANTS`** — `crates/wukong_codegen_gpu/src/gpu.rs:1079-1122` — Looks up a named cliff kernel, asserts its divisibility, and launches a 1-D rasterized grid identical to the gate and A/B bench shapes.
- **`mma.sync` workhorse fused bias/activation epilogue family** — `crates/wukong_codegen_gpu/src/gpu.rs:1124-1248` — Register-level bias+act folded onto the fastest large-GEMM base; note the wrappers all target `..._swz_bias*` entries while several doc comments name the unswizzled ones.
- **Fused bias+residual epilogues on both GEMM bases** — `crates/wukong_codegen_gpu/src/gpu.rs:1250-1331` — Down-proj / output-proj sublayer output collapsing two post-GEMM kernels into the store, on the mma workhorse and the ≤1024³ `pipe_64_s6` champion.
- **Size-aware fused Linear dispatch** — `crates/wukong_codegen_gpu/src/gpu.rs:1333-1375` — Routes `act(A·Bᵀ+bias)` to the deep WMMA pipe at M,N≤1024 and the mma workhorse above, so the fused Linear wins at every size.
- **Fused SwiGLU/GeGLU dual-B gate kernels (fp16 and bf16)** — `crates/wukong_codegen_gpu/src/gpu.rs:1377-1509` — One 128×64 kernel stages x once and feeds two GEMMs plus the gate multiply, collapsing cuBLAS's three-kernel chain; optional per-branch biases add two launch args.
- **Recorded refutation: the ldmatrix XOR-swizzle does not transfer to the gate tile** — `crates/wukong_codegen_gpu/src/gpu.rs:1481-1489` — A measured note pinning production to the padded gate base after swz read 0.89× at 2048³, geomean ~1.00× — an anti-regression guardrail.
- **bf16 GEMM regime dispatch and entry-derived launch geometry** — `crates/wukong_codegen_gpu/src/gpu.rs:1550-1651` — bf16 twin of the fp16 ladder (≥16MB swizzled workhorse else `_mt`), with a `w22swz` name-suffix check that rewrites the warp grid to 2×2.
- **bf16 fused epilogue families (mma bias/act, bias+residual, `sm_db` act/bias)** — `crates/wukong_codegen_gpu/src/gpu.rs:1653-1808` — Training-dtype twins of every fp16 fused variant, sharing the f32-accumulator epilogue so only input quantization differs, also `crates/wukong_codegen_gpu/src/gpu.rs:1511-1548`.
- **Resident fused SiLU FFN block** — `crates/wukong_codegen_gpu/src/gpu.rs:2739-2812` — Runs `x + SiLU(RMSNorm(x)·W1ᵀ)·W2ᵀ` fully on device, folding the SiLU into the up-projection store and the residual into the down-projection accumulator.
- **Tensor-core GEMM variant reference gates (f16/bf16, sm/sm128/db/pipe/swizzle)** — `crates/wukong_codegen_gpu/src/gpu.rs:5189-5447` — Seven gates hold every WMMA/mma.sync staging variant to the same 1e-2/2e-3 (bf16 2e-2/1e-2) band, with per-variant shapes chosen to hit single-K-tile prologues, ring-buffer wrap, and rectangular multi-CTA stores.
- **Standalone `mma.sync.m16n8k16` fragment-layout probe** — `crates/wukong_codegen_gpu/src/gpu.rs:6494-6569` — Ships inline hand-written PTX computing one 16×8 tile from f16-exact small integers and checks it against a CPU matmul, de-risking the per-lane A/B/D register layout before any kernel builds on it.
**`crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs`**

- **Transposable naive training GEMM module (NN/NT/TN)** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:577-636` — Generates one entry per transpose mode by swapping only the A/B index expressions, so forward NT and gradient NN/TN need no operand shuffling; host wrapper at `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:658-679`.
- **Device GEMM routing to the register-blocked kernel** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:833-855` — NN and NT ride `ptx_gemm`'s 64x64 reg-blocked kernel directly while TN transposes A first, so the resident step never touches the naive kernel.
- **Mixed-precision fp16 tensor-core GEMM routing** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:966-1007` — Narrows or transpose-narrows operands to f16 to fit NT-only WMMA, keeps master weights f32 so no loss scaling is needed, and falls back to f32 for non-16-multiple dims; WMMA dispatch at `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:913-952`.
**`crates/wukong_codegen_gpu/src/ptx_gemm.rs`**

- **Register-blocked f32 GEMM PTX generator** — `crates/wukong_codegen_gpu/src/ptx_gemm.rs:26-140` — 16×16 threads compute a 64×64 tile with a 4×4 per-thread micro-tile, cooperative BK=16 A/B panels in SMEM, fully unrolled 16-FMA inner product, ragged M/N/K guarded.
- **Register-blocked GEMM module and host tile exports** — `crates/wukong_codegen_gpu/src/ptx_gemm.rs:143-156` — Caches one module holding `gemm_nt_rb` (A·Bᵀ) and `gemm_nn_rb` (A·B) with per-entry label tags, exporting TILE_M/TILE_N so the host sizes the grid.
**`crates/wukong_codegen_gpu/src/ptx_wmma.rs`**

- **Per-warp multi-tile WMMA GEMM generator (`entry`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:27-150` — Emits `C=A·Bᵀ` where each warp owns a tm×tn grid of m16n16k16 tiles loaded straight from global, reusing fragments across tm·tn MMAs; tile constants at `crates/wukong_codegen_gpu/src/ptx_wmma.rs:2382-2387`.
- **`PipeCfg` tuning-knob struct with SMEM budget and name lookup** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:184-219` — Carries bm/bn/bk, warp grid, stages, raster, mma-vs-wmma and pad; `smem_bytes` = stages·(bm+bn)·(bk+pad)·2, which over-counts for swizzled kernels that force ldp=bk.
- **Per-regime pipeline variant tables (`PIPE_VARIANTS`, `PIPE_BF16`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:221-265` — The swept per-size winners (deep BK=16 pipe L2-resident, 128×128 BK=32 r16 mma workhorse spilling) that generation, gating, sweeps and dispatch all iterate; lookup `crates/wukong_codegen_gpu/src/ptx_wmma.rs:267-274`.
- **GEMM-cliff experiment family (`CliffCfg`, `CLIFF_VARIANTS`, `gemm_cliff_ptx`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:284-369` — Ten same-shape mma.sync candidates varying swizzle/pad, stages, raster band, launch bounds and store mode in a module separate from the dispatched one.
- **`CliffCfg::mma_per_warp` ILP metric — comment/code divergence** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:310-312` — Doc says mma/iteration is `tm·tn·(bk/16)` but the body returns only `(bm/(16·wm))·(bn/(8·wn))`, omitting the bk/16 k-step factor.
- **Shared-memory-staged WMMA GEMM (`entry_smem`) and its CTA-tile constants** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:371-537` — CTA cooperatively stages bm×BK and bn×BK tiles via 128-bit `ld.global.v4`/`st.shared.v4`, then all warps compute from SMEM; shapes at `crates/wukong_codegen_gpu/src/ptx_wmma.rs:159-177`.
- **Static-shape specialization of the SMEM WMMA kernel** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:443-450` — Bakes M/N/K as PTX immediates so ptxas strength-reduces `×K`/`×N` strides and knows the trip count, keeping the launcher signature identical; wrapper at `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1972-1990`.
- **Fused activation epilogue (`Act`) on f32 accumulators** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:539-585` — relu/silu/gelu applied register-wise pre-store using Ada SFU `ex2/tanh/rcp.approx` with constants mirroring `ptx::vmath_ptx`, so fused equals unfused under the same tolerance gate.
- **cp.async double-buffered WMMA GEMM (`entry_smem_db`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:587-857` — Two SMEM buffers toggled by XOR of the power-of-two tile size; prefetches the next K-tile with `cp.async` while tensor cores consume the current, `wait_group 1` gating.
- **Fused residual via `wmma.load.c` accumulator seeding** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:699-715` — Seeds accumulators from `residual[M,N]` in the same opaque fragment layout the final store uses, so `out = residual + A·Bᵀ` costs no HBM round-trip; pipe twin `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1028-1043`.
- **Fused per-column bias via SMEM store-back scratch (WMMA path)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:795-837` — Works around WMMA's opaque fragment map by re-storing each tile into a per-warp 1 KiB smemA slot, re-reading by explicit (row,col), adding bias then activating; pipe twin `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1133-1176`.
- **N-stage cp.async software-pipelined WMMA GEMM (`entry_smem_pipe`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:859-1192` — Ring of `stages` SMEM buffers with `wait_group stages-2`, one `bar.sync` per K-iteration fencing both RAW and WAR, and configurable wide `bk` feeding bk/16 WMMA steps.
- **Threadblock rasterization remap (column-band-of-G order)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1011-1023` — Remaps a 1-D CTA grid into bands of `raster` N-tile columns with a runtime edge-band width, compacting co-scheduled CTAs' L2 footprint; twins at `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1374-1381` and `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1708-1715`.
- **Epilogue store-vectorization mode (`Store::Scalar/V2/V2Cs`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1194-1208` — Folds each lane's adjacent-column D pair into one `st.global.v2.f32`, optionally with the `.cs` evict-first hint; bit-identical output, emitter at `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1515-1521`.
- **`mma.sync.m16n8k16` hand-placed-fragment GEMM (`entry_mma_pipe`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1210-1560` — The route past the WMMA ceiling: explicit lane→(groupID,tid) fragment addressing, same cp.async pipeline and raster, tm m16 × tn n8 sub-tiles per warp.
- **Launch-bounds occupancy knob (`min_ctas` → `.maxntid`/`.minnctapersm`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1238-1246` — Forces ptxas to cap registers so `min_ctas` CTAs co-reside per SM; `0` emits nothing, keeping byte-identical legacy PTX; emission at `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1311-1313`.
- **Padded-SMEM bank-conflict-free b32 fragment loads (pad knob)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1263-1268` — `pad=8` makes the row stride hit 32 distinct banks for the 8 grp × 4 tid lanes, trading SMEM footprint (occupancy) for conflict-freedom; load site `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1475-1497`.
- **XOR-swizzled no-pad SMEM + `ldmatrix` gathers (`swz`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1270-1282` — Chunk column XOR `((row>>1)&(nc-1))` on both cp.async stores and `ldmatrix.x4`/`.x2` reads, conflict-free at higher occupancy; lane phases `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1389-1400`, store `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1423-1427`, read `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1457-1474`.
- **Register-level bias/activation/residual epilogue on mma.sync D-fragments** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1527-1555` — Uses the known D map (d0,d2 at gcol; d1,d3 at gcol+1) to add bias, activate, then add residual entirely in registers — no SMEM scratch, unlike the WMMA path.
- **Fused gated-FFN dual-B SwiGLU/GeGLU generator (`entry_mma_gate`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1562-1901` — Shares one staged x tile across Wg and Wu, keeps two accumulator sets in a 128×64 register/SMEM-neutral tile, folds `act(gate)⊙up` into the store.
- **Tensor-core roofline probe kernel** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1903-1970` — One A/B fragment loaded from global then `iters·ROOFLINE_ACC` independent MMAs, accumulators folded to keep chains live, measuring throughput not latency; module `crates/wukong_codegen_gpu/src/ptx_wmma.rs:2372-2380`.
- **fp16 tensor-core module assembly (`wmma_f16_ptx`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1992-2238` — One cached module instantiating ~40 entries: single/multi-tile, sm, sm128, db, every PIPE_VARIANT, swz and w22swz twins, and the bias/act/residual/gate fused families.
- **bf16 tensor-core module assembly (`wmma_bf16_ptx`)** — `crates/wukong_codegen_gpu/src/ptx_wmma.rs:2240-2369` — Mirrors the fp16 catalogue for the training dtype via the precision-generic generators, adding the mma workhorse, swz/w22swz twins, fused bias/residual and gated-FFN entries.
**`crates/wukong_codegen_gpu/src/ptx.rs`**

- **Tiled shared-memory f32 GEMM (`gemm_nn`/`gemm_nt`)** — `crates/wukong_codegen_gpu/src/ptx.rs:720-942` — Baseline 16×16 CTA staging TILE=16 A/B tiles in SMEM with one C element per thread; out-of-range threads load zeros to keep `bar.sync` uniform.

## G33 — GPU: attention and flash-attention PTX

*Files:* `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/ptx.rs`, `crates/wukong_codegen_gpu/src/ptx_flash.rs`

**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **Flash tiled/untiled crossover planner** — `crates/wukong_codegen_gpu/src/gpu.rs:1892-1922` — Returns entry name and matched launch config together at `seq >= FLASH_TILE_MIN`, with a `_forced` variant so the gate can exercise both bit-identical kernels.
- **Tensor-core flash applicability rule, entry choice and config** — `crates/wukong_codegen_gpu/src/gpu.rs:1924-1972` — Requires D∈{64,128}, S%16==0, S≥512; picks the hand-packed feed at D=64 and the `ldmatrix` PV feed at D=128; the cfg docstring names stale `flash_d64_w`/`_w4` kernels.
- **Warp-specialized flash routing knob and matched plan seam** — `crates/wukong_codegen_gpu/src/gpu.rs:1974-2036` — `WUKONG_FLASH_WS=0` kill-switch read once via `OnceLock`, an S≥4096-only per-D win table, and `wmma_flash_plan` pairing entry with the 2-warp grid to avoid barrier deadlock.
- **Flash-attention host entry and launch seam** — `crates/wukong_codegen_gpu/src/gpu.rs:2038-2094` — Validates Q/K/V shapes and `SUPPORTED_D`, then runs the online-softmax kernel that never materializes the seq×seq scores; `flash_attn_run` is the gate's injection point.
- **Multi-head attention seam with head-major layout shims** — `crates/wukong_codegen_gpu/src/gpu.rs:3196-3241` — Cast-transposes Q/K/V to f16 `[H,S,dh]`, launches the WMMA flash with `grid.y=heads`, transposes back; shim helpers at `crates/wukong_codegen_gpu/src/gpu.rs:3144-3188`.
- **f64 attention oracles (`ref_attn`, `ref_attn_causal`)** — `crates/wukong_codegen_gpu/src/gpu.rs:6388-6447` — Materialized two-pass-softmax single-head references, the independent oracle every flash kernel gate compares against, with the causal variant masking `j>i` to −∞ before the max.
- **Flash-attention correctness gate family** — `crates/wukong_codegen_gpu/src/gpu.rs:6449-6492` — Forces both f32 dispatch lanes at ragged seq/head-dim combinations; the f16 tensor-core gates (wmma `_w`/`_w4`, register-resident `_m`, causal `_mc`, multi-head grid.y, pipelined `_mp`/`_mpc`, multi-warp `_mp4`/`_mp8`) span `crates/wukong_codegen_gpu/src/gpu.rs:6571-6935`.
- **Flash A/B tuning benches under one pinned clock** — `crates/wukong_codegen_gpu/src/gpu.rs:6937-7470` — Five `#[ignore]` harnesses (pipe-vs-mma, multiwarp-vs-mp, throughput, tiled-vs-untiled, mma-vs-wmma) that assert output agreement first, then re-pin the clock with a 1024³ GEMM hammer before each best-of-5 timing.
**`crates/wukong_codegen_gpu/src/ptx_flash.rs`**

- **Flash tiling/occupancy tuning constants and crossover rule** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:39-60` — FLASH_WARPS=2, FLASH_TWARPS=8 and FLASH_TILE_MIN=0 encode the measured A/B verdict that tiled always wins, so untiled is generated but never dispatched.
- **Untiled per-lane f32 flash kernel generator (`flash_d{D}`)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:64-151` — Emits one-warp-per-query-row PTX where 32 lanes split head dim, butterfly-reduce each score, and run online softmax streaming K/V from global.
- **Key-block-tiled SMEM flash kernel (`flash_d{D}_t`)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:156-266` — Eight warps cooperatively stage BK=1024/D keys into a uniform 8 KB SMEM slab, cutting K/V traffic 8x; barriers run unpredicated so ragged CTAs cannot deadlock.
- **WMMA tensor-core flash with SMEM S/P/PV/O round-trip** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:287-445` — One warp per 16 rows does Q·Kᵀ and P·V via wmma, storing every fragment to SMEM and doing softmax under an explicit lane==row map; helper `crates/wukong_codegen_gpu/src/ptx_flash.rs:269-272`.
- **Wide-key WMMA flash (`flash_d{d}_w{nkb}`) amortising per-step barriers** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:457-617` — Processes WK=16·nkb keys per softmax step to cut SMEM round-trips nkb×, requiring S%WK==0; tile width set by `crates/wukong_codegen_gpu/src/ptx_flash.rs:2805-2807`.
- **Register-resident `mma.sync` FA2 kernel (`flash_d{d}_m`)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:657-845` — Keeps O/m/l in registers across the whole K loop with the hand-placed m16n8k16 lane map, and reuses the score D-layout directly as the P A-fragment with no SMEM bounce.
- **Multi-head dispatch by folding a `ctaid.y` head base into Q/K/V/O pointers** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:707-712` — Makes every mma kernel handle `[H,S,D]` layouts with unchanged per-row addressing and a no-op for single-head grid.y=1 callers; repeated at `crates/wukong_codegen_gpu/src/ptx_flash.rs:1940-1943`.
- **Causal masking: diagonal-block loop stop plus per-score `selp` −inf mask** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:733-770` — Halves work by ending the key loop at `kb == row` and masking only the diagonal block's out-of-range keys; mirrored at `crates/wukong_codegen_gpu/src/ptx_flash.rs:1046-1057` and `crates/wukong_codegen_gpu/src/ptx_flash.rs:2351-2362`.
- **Online-softmax register recurrence (max, shfl-bfly, corr rescale, denominator)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:772-818` — The shared per-tile math: per-row local max over four keys, two-offset butterfly reduce, `ex2.approx` correction, in-place O rescale, packed f16 probability fragments; replicated verbatim in every mma variant.
- **`cp.async` double-buffered SMEM-staged flash (`flash_d{d}_mp`/`_mpc`)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:873-1146` — Ping-pongs two K+V slabs with `wait_group 1` so block kb+1's copy overlaps block kb's compute; single-buffer occupancy probe `_m1` at `crates/wukong_codegen_gpu/src/ptx_flash.rs:979-989`.
- **`ldmatrix` conflict-free fragment feed (`_lm` variants)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:1021-1035` — Replaces bank-conflicting strided `ld.shared` with warp-collective `ldmatrix.x2` for K and `.x2.trans` for V; PV half at `crates/wukong_codegen_gpu/src/ptx_flash.rs:1099-1113`.
- **Software-pipelined QKᵀ-ahead kernel (`flash_d{d}_msp`), a measured negative** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:1171-1365` — Separate K/V pools stage K one tile further ahead so QKᵀ(i+1) issues before softmax(i); documented as a wash-to-slight-loss, retained as evidence.
- **Head-dim warp-split kernel (`flash_d{d}_hs`), a measured negative** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:1390-1558` — Two warps share 16 query rows, each owning half the output head dim to halve O accumulators; occupancy doubled but the redundant QKᵀ made it 11-20% slower.
- **Multi-warp-CTA shared-staging flash (`flash_d{d}_mp{warps}`)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:1576-1753` — Four or eight warps on disjoint query blocks share one staged K/V slab, with a predicated cooperative stage and a second barrier protecting the buffer from a lapping warp.
- **Wide-key-tile pipelined flash (`flash_d{d}_mpw{nkb}`), a measured negative** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:1777-1966` — Softmaxes BK=16·nkb keys at once and routes each score n-tile into the right P k-tile slot; lost 0.65-0.91× because occupancy dominates.
- **Fused-RoPE flash (`flash_d{d}_mprope`) — the library-can't-fuse lever** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:1987-2177` — Rotates Q once at load and each K fragment after `ld.shared` using interleaved GPT-J pairs from host cos/sin tables; helpers at `crates/wukong_codegen_gpu/src/ptx_flash.rs:2041-2052`.
- **Warp-specialized ping-pong flash (`flash_d{d}_ws*`) with named-barrier phase machine** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:2223-2488` — Two warps on disjoint query tiles run anti-phase via `bar.sync 1/2,64` so one warp's mma hides the other's SFU softmax; loop bodies at `crates/wukong_codegen_gpu/src/ptx_flash.rs:2438-2477`.
- **Three-stage `cp.async` ring twin of the ws kernel (`flash_d{d}_ws3`)** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:2506-2733` — Adds a third buffer so staging runs two blocks ahead with `wait_group 1`, restoring whole-step prefetch distance at 1.5× SMEM and lower occupancy.
- **Flash PTX module assembly and `OnceLock` cache** — `crates/wukong_codegen_gpu/src/ptx_flash.rs:2738-2803` — Concatenates ~30 kernel entries under one `.version 7.8/.target sm_89` header; its doc-comment lists only five kernels and is badly out of date versus the body.
**`crates/wukong_codegen_gpu/src/ptx.rs`**

- **Multi-head attention layout shims and f32→f16 cast** — `crates/wukong_codegen_gpu/src/ptx.rs:110-280` — `cast_transpose_qkv` folds narrowing into the `[S,H·dh]`→`[H,S,dh]` transpose the flash kernel wants, `transpose_attn_out` inverts it, `cast_f32_f16` is the plain narrowing pass.

## G34 — GPU: quantized PTX (int8 / int4 / fp8)

*Files:* `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/paged_attention.rs`, `crates/wukong_codegen_gpu/src/ptx_fp8.rs`, `crates/wukong_codegen_gpu/src/ptx_int4.rs`, `crates/wukong_codegen_gpu/src/ptx_int8.rs`

**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **8-bit mma.sync single-tile fragment validators** — `crates/wukong_codegen_gpu/src/gpu.rs:3502-3523` — fp8 E4M3 `m16n8k32` 16x8 tile checking the manual A-row / B-column fragment layout, with the bit-exact u8xi8->i32 int8 twin at `crates/wukong_codegen_gpu/src/gpu.rs:4150-4166`.
- **fp8 GEMM three-way dispatch ladder** — `crates/wukong_codegen_gpu/src/gpu.rs:3529-3589` — Routes E4M3 `A·Bᵀ` to the pipelined kernel when the 128-tile divides, else a fragment-reuse multi-tile kernel, else the single-tile kernel.
- **W4A16 int4 weight-only decode GEMM** — `crates/wukong_codegen_gpu/src/gpu.rs:3598-3643` — Unpacks group-wise 4-bit weights to fp16 on the fly into the identical fp16 MMA, dispatching symmetric or zero-point entries by `QuantWeight.zeros`.
- **W4A16 deterministic split-K for thin-M decode** — `crates/wukong_codegen_gpu/src/gpu.rs:3652-3702` — `sk` K-splits write disjoint f32 planes then a fixed-order reduce sums them, giving run-to-run bit-identical results a float atomicAdd split-K could not.
- **W4A16 static-shape specialization** — `crates/wukong_codegen_gpu/src/gpu.rs:3710-3757` — Bakes M/N/K as PTX constants and raw-loads a fresh per-shape module so ptxas strength-reduces the stride multiplies a library cannot see.
- **fp8 pipelined GEMM warp-tile and stage heuristic** — `crates/wukong_codegen_gpu/src/gpu.rs:3921-3971` — Chooses the 3-stage w64 tile for `M,N<=2048, K>=96`, the 2-stage w64 otherwise, and a 64-row `_m64` entry when 128 does not divide M.
- **fp8 fused bias/activation/residual epilogue family** — `crates/wukong_codegen_gpu/src/gpu.rs:3979-4031` — One shared launcher plus four thin wrappers folding bias and relu/silu/gelu into the fp8 store; the bias+residual variant is at `crates/wukong_codegen_gpu/src/gpu.rs:4037-4074`.
- **fp8 dual-B SwiGLU/GeGLU fused gate** — `crates/wukong_codegen_gpu/src/gpu.rs:4083-4127` — Feeds one staged x tile to two weight GEMMs and fuses the gate multiply into the store, collapsing three kernels into one; wrappers at `crates/wukong_codegen_gpu/src/gpu.rs:4129-4137`.
- **int8 W8A8 tensor-core GEMM with multi-tile fast path** — `crates/wukong_codegen_gpu/src/gpu.rs:4174-4225` — Exact-mod-2^32 `u8 x i8 -> i32` GEMM choosing the fragment-reuse multi-tile kernel when the block divides evenly, else the single-tile kernel.
- **int8 SMEM-staged cp.async regime dispatch** — `crates/wukong_codegen_gpu/src/gpu.rs:4245-4323` — Picks ldmatrix+XOR-swizzle w64 (M,N>=2048), plain 64x64 swz, hand-placed 128 (non-swz, >=4096) or hand-placed 64 tiles; cfg helper at `crates/wukong_codegen_gpu/src/gpu.rs:4228-4234`.
- **int8 static-shape swizzle GEMM** — `crates/wukong_codegen_gpu/src/gpu.rs:4332-4367` — Bakes M/N/K into the ldmatrix+swizzle kernel and raw-loads it, reusing the same 64x64 / 128x128 regime rule as the dynamic dispatch.
- **int8 split-K via red.global.add.u32** — `crates/wukong_codegen_gpu/src/gpu.rs:4378-4410` — `gridDim.z` K-splits fold partials into a pre-zeroed C by integer atomic reduce, staying bit-exact because integer addition commutes.
- **int8 GEMM with fused per-channel dequant** — `crates/wukong_codegen_gpu/src/gpu.rs:4417-4485` — Folds `f32(acc)·scale[j]` into the C store across w64/swz/hand-placed tiles, removing the separate i32->f32 dequant kernel cuBLAS int8 requires.
- **W4A16 int4 correctness gates (dense + split-K)** — `crates/wukong_codegen_gpu/src/gpu.rs:4689-4768` — Gates symmetric and zero-point int4 decode against the exact f64 dequant reference at 1e-2/2e-3, plus run-to-run bit-identity and static-vs-dynamic kernel bit-equality; split-K twin at `crates/wukong_codegen_gpu/src/gpu.rs:4770-4804`.
- **W4A16 split-K decode occupancy bench** — `crates/wukong_codegen_gpu/src/gpu.rs:4806-4887` — Times sk=1 against sk∈{2,4,8} (GEMM *plus* the reduce kernel, so the speedup is honest end-to-end) on decode-shaped small-M/large-K problems, best-of-8×50 device-resident.
- **fp8 `m16n8k32` tensor-core tile gate** — `crates/wukong_codegen_gpu/src/gpu.rs:10322-10360` — Feeds asymmetric e4m3-exact integer data (so a lane/layout bug cannot hide behind all-ones) through `fp8_tile` and compares to an f64 reference over the same e4m3-rounded bits at 1e-3.
**`crates/wukong_codegen_gpu/src/paged_attention.rs`**

- **int8-KV paged decode attention with factored-out per-(token,head) scales** — `crates/wukong_codegen_gpu/src/paged_attention.rs:228-339` — Loads `s8` cache bytes, keeps the dot in raw ints then multiplies once by scaleK, and folds scaleV into p; host mirror at `crates/wukong_codegen_gpu/src/paged_attention.rs:381-397`, launcher `crates/wukong_codegen_gpu/src/paged_attention.rs:346-376`.
- **int8 KV-append: warp amax reduction, scale store, quantize-and-scatter** — `crates/wukong_codegen_gpu/src/paged_attention.rs:536-627` — Butterfly-merges per-lane |x| maxima, has lane 0 store both f32 scales, then clamps `cvt.rni` quotients to ±127; device rounds ties-even where the host rounds half-away-from-zero.
**`crates/wukong_codegen_gpu/src/ptx_fp8.rs`**

- **fp8 E4M3 tile probe plus naive and fragment-reuse GEMMs** — `crates/wukong_codegen_gpu/src/ptx_fp8.rs:57-119` — The validated single-warp e4m3 `m16n8k32` layout and its two global-load GEMM scalings; see `crates/wukong_codegen_gpu/src/ptx_fp8.rs:887-994` and `crates/wukong_codegen_gpu/src/ptx_fp8.rs:787-880`.
- **Padded-SMEM multistage cp.async fp8 pipe generator** — `crates/wukong_codegen_gpu/src/ptx_fp8.rs:154-382` — The workhorse e4m3 GEMM: stages-deep ring, 16-byte SMEM row padding for conflict-free b32 fragment loads, rasterization, and per-warp tm×tn `mma.sync` blocking.
- **Fused fp8 epilogues: bias, activation, residual** — `crates/wukong_codegen_gpu/src/ptx_fp8.rs:355-377` — Applies per-column bias, an `Act::epilogue` transcendental, and an f32 residual add entirely in D-fragment registers before the store; module variants at `crates/wukong_codegen_gpu/src/ptx_fp8.rs:660-700`.
- **fp8 dual-B gated-FFN kernel (SwiGLU/GeGLU/GLU)** — `crates/wukong_codegen_gpu/src/ptx_fp8.rs:393-617` — Stages one x tile against two weight tiles, interleaves the independent gate/up `mma`s for ILP, and multiplies post-activation in one kernel; variants `crates/wukong_codegen_gpu/src/ptx_fp8.rs:705-713`.
- **fp8 pipeline config sweep hook and regime-tuned variants** — `crates/wukong_codegen_gpu/src/ptx_fp8.rs:725-751` — Builds the pipe at arbitrary tile/stage/raster settings; the shipped 64×64-warp and 3-stage picks live at `crates/wukong_codegen_gpu/src/ptx_fp8.rs:763-775`, tuning constants `crates/wukong_codegen_gpu/src/ptx_fp8.rs:127-139`, small-M tile `crates/wukong_codegen_gpu/src/ptx_fp8.rs:646-659`.
**`crates/wukong_codegen_gpu/src/ptx_int4.rs`**

- **Marlin/AWQ interleaved int4 nibble layout and QuantWeight container** — `crates/wukong_codegen_gpu/src/ptx_int4.rs:44-100` — Packs 8 weights per u32 as even-low/odd-high nibble pairs so one `lop3` yields an f16x2, with host pack/unpack twins and group/word accessors.
- **Host group-wise int4 quantizers (symmetric and asymmetric)** — `crates/wukong_codegen_gpu/src/ptx_int4.rs:109-135` — Symmetric stores offset-binary `u=q+8` against the rounded fp16 scale; asymmetric seeds the range with 0 so the zero-point stays representable, at `crates/wukong_codegen_gpu/src/ptx_int4.rs:144-175`.
- **W4A16 SMEM-staged WMMA GEMM generator** — `crates/wukong_codegen_gpu/src/ptx_int4.rs:245-482` — Stages an fp16 A tile and an on-the-fly-dequantized B tile into shared memory, then runs stock `wmma.mma.m16n16k16` f32-accumulate; module at `crates/wukong_codegen_gpu/src/ptx_int4.rs:494-503`.
- **lop3 int4→f16x2 fast unpack with unified zero-offset** — `crates/wukong_codegen_gpu/src/ptx_int4.rs:414-436` — Extracts a nibble pair as `(1024+u)` f16x2 via one `lop3`, then `sub.f16x2` the 0x6400|Z subtrahend and `mul.f16x2` the scale, dequantizing two weights per op.
- **W4A16 split-K with fixed-order reduction kernel** — `crates/wukong_codegen_gpu/src/ptx_int4.rs:510-559` — Each z-CTA writes a disjoint f32 partial plane and a grid-stride reduce kernel sums planes in fixed z order for run-to-run determinism; GEMM side `crates/wukong_codegen_gpu/src/ptx_int4.rs:350-359`, module `crates/wukong_codegen_gpu/src/ptx_int4.rs:568-577`.
**`crates/wukong_codegen_gpu/src/ptx_int8.rs`**

- **int8 m16n8k32 fragment-layout probe** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:20-82` — Single-warp `mma.sync.m16n8k32.row.col.s32.u8.s8.s32` tile pinning the exact per-lane u8/i8 A/B and i32 D addressing, the validated basis every other int8 kernel reuses.
- **Hand-placed SMEM + cp.async double-buffered int8 GEMM generator** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:274-450` — Emits a BK=32 two-buffer pipeline where CTAs stage the next A/B slab while tensor cores consume the current; wrappers at `crates/wukong_codegen_gpu/src/ptx_int8.rs:192-268`.
- **ldmatrix + XOR-swizzle conflict-free int8 GEMM generator** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:492-797` — BK=64 swizzled SMEM (`chunk XOR ((row>>1)&3)`) gathered by `ldmatrix.x4/.x2`, removing the hand-placed path's 2-way bank conflicts; codegen legality asserts at `crates/wukong_codegen_gpu/src/ptx_int8.rs:521-553`.
- **Static-shape (M/N/K baked) quantized-GEMM specialization** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:602-609` — Emits dims as `mov` immediates so ptxas strength-reduces hot-loop strides and knows the trip count; entries `crates/wukong_codegen_gpu/src/ptx_int8.rs:837-849` and int4 twin `crates/wukong_codegen_gpu/src/ptx_int4.rs:335-342`, `crates/wukong_codegen_gpu/src/ptx_int4.rs:585-595`.
- **Threadblock rasterization for L2 locality** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:615-627` — Maps a 1-D CTA grid into raster-wide N-tile bands so co-resident CTAs share a compact A/B footprint; entries `crates/wukong_codegen_gpu/src/ptx_int8.rs:851-873` and `crates/wukong_codegen_gpu/src/ptx_int8.rs:932-939`.
- **Multistage cp.async SMEM ring on the swizzle path** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:690-724` — Generalizes the XOR double-buffer to a stages-deep add-and-wrap ring; ring advance `crates/wukong_codegen_gpu/src/ptx_int8.rs:749-759`, and the documented measured-negative 3-stage entry `crates/wukong_codegen_gpu/src/ptx_int8.rs:913-930`.
- **Fused per-channel int8 dequant epilogue** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:772-784` — Folds `out=f32(Σu8·i8)·scale[col]` into the C store in registers, eliminating the second HBM pass cuBLAS int8 requires; also `crates/wukong_codegen_gpu/src/ptx_int8.rs:429-439`.
- **int8 split-K with deterministic integer atomic fold** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:786-791` — Each gridDim.z CTA reduces its K-slice into C via `red.global.add.u32`, order-independent and bit-exact; K-range setup `crates/wukong_codegen_gpu/src/ptx_int8.rs:628-633`, entry `crates/wukong_codegen_gpu/src/ptx_int8.rs:812-815`.
- **int8 tile/warp-shape config surface and w64 default** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:885-893` — Exposes arbitrary (bm,bn,wm,wn,raster) with tile-encoded entry names, plus the shipped winning 128×128 CTA / 64×64 warp tile at `crates/wukong_codegen_gpu/src/ptx_int8.rs:902-911`.
- **int8 global-load GEMM pair (naive + fragment-reuse)** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:955-1062` — `C=A·Bᵀ` u8×i8→i32 with one 16×8 tile per warp, plus the TM×TN multi-tile variant reusing each A frag across N; see `crates/wukong_codegen_gpu/src/ptx_int8.rs:86-188`.
- **Separate BK=32 multistage int8 generator** — `crates/wukong_codegen_gpu/src/ptx_int8.rs:1073-1252` — An independent stages-deep ring over the hand-placed (non-swizzled) fragments keeping the 2-barrier overwrite discipline; s3/s4 64×64 and 128×128 wrappers at `crates/wukong_codegen_gpu/src/ptx_int8.rs:1255-1288`.

## G35 — GPU: conv, winograd, norm, optimizer PTX

*Files:* `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/ptx.rs`, `crates/wukong_codegen_gpu/src/ptx_conv.rs`, `crates/wukong_codegen_gpu/src/ptx_norm.rs`, `crates/wukong_codegen_gpu/src/ptx_optim.rs`, `crates/wukong_codegen_gpu/src/ptx_winograd.rs`

**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **Fused row-normalization host entry (softmax/layernorm/rmsnorm)** — `crates/wukong_codegen_gpu/src/gpu.rs:1858-1890` — Dispatches `NORM_*` codes to one-warp-per-row PTX with deterministic warp-butterfly reductions, a grid of exactly `rows` blocks of 32 threads.
- **SMEM-tiled conv2d launch configuration** — `crates/wukong_codegen_gpu/src/gpu.rs:2096-2111` — Computes the per-output-tile grid from `TILE_P`/`TILE_Q` and the channel block factor `kblock(k)`, shared by both the launcher and the conv peer bench.
- **conv2d SMEM-tiled vs naive dispatch** — `crates/wukong_codegen_gpu/src/gpu.rs:2120-2167` — Valid stride-1 f32 conv2d picking the shape-specialized SMEM-tiled PTX when `tiled_applies`, else a one-thread-per-output naive kernel, launch grid at `crates/wukong_codegen_gpu/src/gpu.rs:2099-2111`.
- **fp16 tensor-core implicit-GEMM conv2d** — `crates/wukong_codegen_gpu/src/gpu.rs:2193-2227` — Host-rounds X/W to f16 and runs one warp per `WMMA_BM×WMMA_BN` output tile with f32 accumulate; grid helper at `crates/wukong_codegen_gpu/src/gpu.rs:2172-2185`.
- **Split-K implicit-GEMM conv plus deterministic plane reduce** — `crates/wukong_codegen_gpu/src/gpu.rs:2250-2297` — Splits the `C*R*S` reduction across `sk` z-slices into disjoint partial planes then sums them in fixed order; auto factor chooser at `crates/wukong_codegen_gpu/src/gpu.rs:2304-2321`, cfg at `crates/wukong_codegen_gpu/src/gpu.rs:2231-2242`.
- **Winograd F(m,3x3) four-phase conv** — `crates/wukong_codegen_gpu/src/gpu.rs:2333-2406` — Filter/input transforms, one batched alpha^2 GEMM, output transform; DIVERGENCE: the doc-comment claims phase 3 reuses `conv2d_wmma`, but the code launches a dedicated `wino_bgemm`.
- **Fused conv + bias + activation epilogue** — `crates/wukong_codegen_gpu/src/gpu.rs:2415-2456` — Folds an optional per-channel bias and Relu/Silu/Gelu into the WMMA store epilogue, so the pointwise pass costs no extra `K*P*Q` HBM round-trip.
- **Strided and zero-padded implicit-GEMM conv variants** — `crates/wukong_codegen_gpu/src/gpu.rs:2462-2499` — Downsampling conv reading `(p*stride+r, q*stride+s)`, and the general affine padded form with in-kernel OOB-contributes-zero bounds checks at `crates/wukong_codegen_gpu/src/gpu.rs:2507-2545`.
- **Explicit-pad conv alternative (measured slower)** — `crates/wukong_codegen_gpu/src/gpu.rs:2557-2614` — Scatters X into a zeroed padded buffer then runs the dense valid strided kernel; documented as 1.0-1.12x slower than the bounds-checked path, kept as a gated A/B.
- **Affine conv split-K auto dispatch** — `crates/wukong_codegen_gpu/src/gpu.rs:2624-2679` — Uses `conv_splitk_factor_affine` with the device SM count to pick between the plain padded kernel and a padded split-K plus fixed-order reduce when the downsampled grid starves SMs.
- **conv2d_best top-level shape heuristic** — `crates/wukong_codegen_gpu/src/gpu.rs:2694-2720` — Routes to Winograd only for valid 3x3 with `C>=64`, `H,W>=28` and tile count `>=16`, else the affine or valid split-K auto implicit-GEMM paths.
- **Winograd relative-Frobenius gate** — `crates/wukong_codegen_gpu/src/gpu.rs:7582-7637` — Deliberately rejects per-element relative error (inverse-transform cancellation explodes it) in favour of ‖got−ref‖_F/‖ref‖_F ≤ 4e-3/8e-3 plus a coarse per-lane absolute backstop, for F(2,3) and F(4,3).
**`crates/wukong_codegen_gpu/src/ptx_conv.rs`**

- **Tiled-conv register-block factor and applicability gate** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:30-50` — `kblock` picks the largest of 8/4/2 dividing K and `tiled_applies` rejects shapes whose halo plus weights exceed a 44 KB shared-memory budget.
- **Shape-specialized SMEM-tiled direct conv2d generator** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:56-217` — Emits a `conv2d` PTX kernel with every extent baked in, staging the input halo plus KB weight windows per channel and fully unrolling the R*S FMA chain.
- **Naive one-thread-per-output conv2d reference kernel** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:271-381` — Fully dynamic c/r/s triple-loop conv taking C,H,W,K,R,S,P,Q as runtime params, kept as the honest worst-case baseline and universal fallback.
- **Auxiliary conv path kernels: explicit pad scatter and unfused bias+ReLU** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:526-581` — `pad_nchw_copy` scatters fp16 X into a pre-zeroed padded buffer so the dense kernel handles padding; the unfused `bias_relu` baseline it is measured against sits at `crates/wukong_codegen_gpu/src/ptx_conv.rs:223-266`.
- **fp16 tensor-core implicit-GEMM conv core generator** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:583-940` — Builds the whole `conv2d_wmma` family: 64x64 CTA tile, 2x2 warps, SMEM-staged weights plus on-the-fly im2col, `m16n16k16` WMMA with f32 accumulate; tile constants and the dispatch heuristic live at `crates/wukong_codegen_gpu/src/ptx_conv.rs:399-420`.
- **Hoisted K-independent im2col address decode** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:740-764` — Per B-staging slot precomputes `nn`, `p`, `q` and the folded `xpart` (or signed padded top-left coords) once outside the K-loop, so each K-step only decodes `(c,r,s)`; consumed at `crates/wukong_codegen_gpu/src/ptx_conv.rs:802-842`.
- **Strided and zero-padded affine im2col gather** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:816-833` — Switches the gather to per-tap signed coordinates guarded by a single unsigned compare per axis (negatives wrap high), giving zero-pad "same" and stride>1 downsample convs; wrappers at `crates/wukong_codegen_gpu/src/ptx_conv.rs:475-517`.
- **Fused conv bias plus activation epilogue** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:902-936` — The smemC drain adds a guarded per-output-channel bias and applies `Act::epilogue` on the f32 accumulators, eliminating a whole extra HBM round-trip over K*P*Q; entry point at `crates/wukong_codegen_gpu/src/ptx_conv.rs:457-468`.
- **Split-K conv partial planes and deterministic reduce** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:947-992` — Each `gridDim.z` slice writes its own disjoint M*N plane (no float atomics) and `conv_splitk_reduce` sums them in fixed ascending-z order for bit-reproducible results.
- **Split-K occupancy heuristic** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:1026-1045` — Picks the largest clean split landing CTA count in a `[2*SM, 5*SM]` window, requiring 16-divisible slices and `GK/sk>=64`, else keeps sk=1; contract asserted at `crates/wukong_codegen_gpu/src/ptx_conv.rs:1403-1429`.
- **Register double-buffered software-pipelined implicit-GEMM conv** — `crates/wukong_codegen_gpu/src/ptx_conv.rs:1071-1397` — Prefetches the next K-slice into registers before the MMAs and publishes into an XOR-toggled alternate SMEM buffer, halving barriers to one per K-step without needing cp.async contiguity.
**`crates/wukong_codegen_gpu/src/ptx_norm.rs`**

- **Warp-per-row fused GPU norms (softmax / LayerNorm / RMSNorm)** — `crates/wukong_codegen_gpu/src/ptx_norm.rs:1-149` — 32 lanes stride each row and all-reduce via `shfl.sync.bfly` (no shared memory); softmax uses SFU `ex2.approx`, LayerNorm uses the one-pass `E[x^2]-mu^2` variance.
**`crates/wukong_codegen_gpu/src/ptx_optim.rs`**

- **Fused AdamW and SGD optimizer-step PTX** — `crates/wukong_codegen_gpu/src/ptx_optim.rs:49-142` — Grid-stride kernels updating every parameter in one launch over contiguous `(w,g,m,v)`, mirroring `build_adamw_step` op-for-op with single-rounded `fma.rn`/`div.rn`/`sqrt.rn`; SGD twin at `crates/wukong_codegen_gpu/src/ptx_optim.rs:147-203`, hp layout at `crates/wukong_codegen_gpu/src/ptx_optim.rs:25-36`.
**`crates/wukong_codegen_gpu/src/ptx_winograd.rs`**

- **Winograd F(2,3)/F(4,3) f64 CPU reference driver** — `crates/wukong_codegen_gpu/src/ptx_winograd.rs:171-295` — Generic `winograd_fm3` transforms filters once, reduces `U (*) V` in the transform domain over channels, applies one inverse per tile; constant B/G/A sets at `crates/wukong_codegen_gpu/src/ptx_winograd.rs:68-121`, differential gate vs direct conv at `crates/wukong_codegen_gpu/src/ptx_winograd.rs:774-820`.
- **Winograd separable transform-map algebra and PTX transform kernels** — `crates/wukong_codegen_gpu/src/ptx_winograd.rs:417-572` — `filter_map`/`input_map`/`output_map` build outer-product coefficient matrices (`crates/wukong_codegen_gpu/src/ptx_winograd.rs:338-384`) that `emit_lincomb` (`crates/wukong_codegen_gpu/src/ptx_winograd.rs:388-405`) turns into nonzero-only FMA chains for the three transform kernels.
- **Winograd batched alpha-squared NN GEMM in one launch** — `crates/wukong_codegen_gpu/src/ptx_winograd.rs:583-749` — Folds all transform-position planes into `gridDim.z` with baked plane strides so the tiny per-plane grids run concurrently instead of serially starving the SMs.
**`crates/wukong_codegen_gpu/src/ptx.rs`**

- **Base elementwise PTX spikes: saxpy and vadd** — `crates/wukong_codegen_gpu/src/ptx.rs:11-104` — The canonical bounds-checked one-thread-per-element skeleton (`ntid/ctaid/tid` index, `cvta.to.global`, `mul.wide` addressing) every other hand-written kernel in the file copies.
- **Streaming copy bandwidth kernel with 4× ILP (`COPY_V4`)** — `crates/wukong_codegen_gpu/src/ptx.rs:295-367` — Four independent grid-stride-spaced `ld.global.v4.f32`/`st.global.v4.f32` groups per thread plus a scalar TAIL loop, sized to saturate HBM on a 1:1 copy.
- **GPU activation module generator (`vmath_ptx`) and `hexf` immediates** — `crates/wukong_codegen_gpu/src/ptx.rs:369-473` — Builds one cached module with relu/exp/sigmoid/tanh/silu/gelu entries from a shared prologue closure, using SFU approximations and `f32::to_bits`-derived constants.
- **Deterministic block reduction family (`REDUCE`)** — `crates/wukong_codegen_gpu/src/ptx.rs:482-710` — sum/dot/max entries grid-stride into per-thread accumulators, tree-reduce in a fixed 1024-byte shared array, and emit one partial per block for a fixed-order host combine.

## G36 — GPU: MIR to PTX lowering (gpu-native backend)

*Files:* `crates/wukong_codegen_gpu/src/lower.rs`

**`crates/wukong_codegen_gpu/src/lower.rs`**

- **gpu-native backend entry (`jit_run` / `GpuLowerBackend`)** — `crates/wukong_codegen_gpu/src/lower.rs:103-153` — Lowers a param-less entry function to PTX, tries the megakernel first (skippable via `WUKONG_GPU_NO_MEGA`), else JITs the single-thread lowering; also `crates/wukong_codegen_gpu/src/lower.rs:82-98`.
- **PTX module assembly (`emit_ptx`)** — `crates/wukong_codegen_gpu/src/lower.rs:157-223` — Emits `.version 7.8/.target sm_89` header, dedupes the `mrt_*` device helpers actually called, forward-declares every function, then bodies plus the `.visible .entry` wrapper.
- **Cooperative megakernel lowering mode** — `crates/wukong_codegen_gpu/src/lower.rs:232-291` — SPMD `wukong_mega` entry whose frame is a shared `.global` param and whose stores are `tid==0`-guarded; also `crates/wukong_codegen_gpu/src/lower.rs:478-519`, `crates/wukong_codegen_gpu/src/lower.rs:656-676`, `crates/wukong_codegen_gpu/src/lower.rs:3773-3786`.
- **Register-class model and MIR type mapping** — `crates/wukong_codegen_gpu/src/lower.rs:297-367` — Maps every MIR type to `%rd`/`%f`/`%fd`, defines ABI/mov suffixes, integer bit widths, byte sizes for gep/alloca, and sign-extended constant materialization.
- **Lazy virtual-register allocation and `.reg` count emission** — `crates/wukong_codegen_gpu/src/lower.rs:545-604` — Per-class counters mint `%rd/%r/%rs/%f/%fd/%p` names on first use of a ValueId, and the final counts become the function's `.reg <N>` declarations; state at `crates/wukong_codegen_gpu/src/lower.rs:382-424`.
- **Alloca frame layout and function preamble/ABI** — `crates/wukong_codegen_gpu/src/lower.rs:609-650` — Pre-scans allocas into one 8-byte-aligned `.local __frame` reached via `cvta.local`, binds the hidden `p_ctx` context pointer and each `.param`; signature at `crates/wukong_codegen_gpu/src/lower.rs:2212-2223`.
- **Instruction dispatch and the UNSUPPORTED-decline policy** — `crates/wukong_codegen_gpu/src/lower.rs:693-807` — Routes each MIR op to an emitter and declines `VecKernelCall`, `GlobalAddr` (strings) and scalar `Splat` with the classifiable `UNSUPPORTED:` prefix defined at `crates/wukong_codegen_gpu/src/lower.rs:68-73`.
- **Float remainder via an exact binary-scaling device fmod** — `crates/wukong_codegen_gpu/src/lower.rs:843-868` — `FRem` widens to f64 and calls `mrt_fmod`, which subtract-and-halves exactly so `1e18 % 3` is right where the `a-trunc(a/b)*b` identity fails; kernel at `crates/wukong_codegen_gpu/src/lower.rs:3347-3380`.
- **64-bit integer arithmetic with per-result width masking** — `crates/wukong_codegen_gpu/src/lower.rs:873-909` — All integer ops compute in `s64` then re-narrow via `mask_int`, with unsigned ops zero-masked and shift counts masked to width, mirroring the interpreter; helpers at `crates/wukong_codegen_gpu/src/lower.rs:963-994`.
- **Div/rem-by-zero guard** — `crates/wukong_codegen_gpu/src/lower.rs:945-959` — Substitutes divisor 1 and selects result 0 when `b==0`, reproducing the interpreter/Cranelift "no trap, yields 0" contract; note it does not guard `i64::MIN / -1`.
- **Compare and select lowering (i1 held in a 64-bit register)** — `crates/wukong_codegen_gpu/src/lower.rs:998-1057` — Float compares widen to the wider operand type and `!=` uses the unordered `neu` form; predicates are materialized to 0/1 via `selp`; condition-code tables at `crates/wukong_codegen_gpu/src/lower.rs:2178-2207`.
- **Cast lowering mirroring `wukong_interp::apply_cast`** — `crates/wukong_codegen_gpu/src/lower.rs:1075-1226` — Covers sext/zext/trunc, int→f64→f32 double rounding, bf16/f16 RNE round-trips, bitcasts, and float→narrow-int saturation clamped in a 32-bit temp; `fp_to_int_into` at `crates/wukong_codegen_gpu/src/lower.rs:1164-1204`.
- **Generic load/store per MIR type with bf16/f16 2-byte storage** — `crates/wukong_codegen_gpu/src/lower.rs:1237-1355` — Uses generic `ld`/`st` so stack and device-global pointers share code; DIVERGENCE: `i1` stores/loads 4 bytes while `size_of(I1)` is 1, so a gep-strided bool array would overlap.
- **Mega store guard with an explicit pointer-store exception** — `crates/wukong_codegen_gpu/src/lower.rs:1296-1324` — Every store is `@tid0`-predicated in megakernel mode except `Ptr` slots, which all threads must write or non-zero threads deref a null frame slot and fault.
- **SIMD `<N x T>` scalarization** — `crates/wukong_codegen_gpu/src/lower.rs:1360-1517` — Splits vectorized MIR into N lane registers reusing the scalar `*_into` emitters, striding vector loads/stores by lane size; declines vector block params and unhandled vector ops.
- **`wukong_parallel_for` lowered as one sequential chunk** — `crates/wukong_codegen_gpu/src/lower.rs:1585-1603` — Resolves the `func_addr` operand at compile time (PTX has no function pointers) and emits a direct `F(0, n, ctx)` call, correct because the outlined body is chunk-decomposable.
- **Mega cooperative call dispatch with `bar.sync` bracketing** — `crates/wukong_codegen_gpu/src/lower.rs:1620-1766` — Classifies each recognized op via `fusion::classify_call` and runs it cooperatively (reduce), chunked per-thread (vmath/velem/gemm/norm), or `tid==0`-serial, fenced by barriers.
- **PTX call convention emitters** — `crates/wukong_codegen_gpu/src/lower.rs:1785-1842` — Builds `.param` scopes, marshals args, optionally predicates the `call.uni` on `tid==0`, and binds the return; user-function calls thread the hidden ctx pointer at `crates/wukong_codegen_gpu/src/lower.rs:1938-1977`.
- **Chunked-cooperative call partitioning with per-arg strides** — `crates/wukong_codegen_gpu/src/lower.rs:1851-1918` — Computes a ceil-divided per-thread `[lo,hi)` slice, offsets each pointer arg by `lo*stride` and replaces the count, giving deterministic serial==parallel semantics; `Stride` at `crates/wukong_codegen_gpu/src/lower.rs:372-376`.
- **Device print/assert record buffer and host replay** — `crates/wukong_codegen_gpu/src/lower.rs:1979-2053` — Appends `(tag,payload)` records through an atomic counter with a capacity predicate; the host reformats them with the interpreter's exact `format!("{}\n")` at `crates/wukong_codegen_gpu/src/lower.rs:3742-3769`, layout at `crates/wukong_codegen_gpu/src/lower.rs:50-61`.
- **Terminators and the block-parameter edge convention** — `crates/wukong_codegen_gpu/src/lower.rs:2057-2103` — Maps Br/CondBr/Ret/Unreachable to `bra`/`@!p bra`/`ret`/`trap`; edge args copy through temporaries so permuted or self-referential edges are safe (`crates/wukong_codegen_gpu/src/lower.rs:2128-2145`).
- **Entry-kernel wrapper and exit-code propagation** — `crates/wukong_codegen_gpu/src/lower.rs:2227-2268` — The launched `.visible .entry` loads the ctx pointer, calls the entry `.func`, truncates a float return via `cvt.rzi.s64` and stores the exit code; mega twin at `crates/wukong_codegen_gpu/src/lower.rs:2107-2124`.
- **Recognized-runtime-call → `mrt_*` device helper dispatch table** — `crates/wukong_codegen_gpu/src/lower.rs:2290-2342` — Maps 20+ `wukong_*` runtime symbols (and their `_parallel` twins) to a PTX `.func` name, return class and definition text; call-site routing at `crates/wukong_codegen_gpu/src/lower.rs:1521-1613`.
- **vmath / vmath2 op-code constant gate** — `crates/wukong_codegen_gpu/src/lower.rs:2346-2348` — Requires the op selector to be a compile-time constant in `const_ints` and inside the implemented range (0..=35 single-arg, 0..=2 two-arg), else declines UNSUPPORTED; use site `crates/wukong_codegen_gpu/src/lower.rs:1546-1579`.
- **Serial reduction device kernel (`mrt_sreduce`)** — `crates/wukong_codegen_gpu/src/lower.rs:2354-2408` — Single predicated loop implementing dot/ssd/sum/sumsq/max/min/maxabs/sumabs/absdiff with op-specific identity init; sequential fold differs from the CPU's chunk tree only in reduction order.
- **Cooperative block reduction tree (`mrt_sreduce_coop`)** — `crates/wukong_codegen_gpu/src/lower.rs:2422-2514` — Thread-strided partials folded through a fixed shared-memory halving tree and broadcast to all threads, deterministic and atomic-free, requiring a power-of-two block; scratch at `crates/wukong_codegen_gpu/src/lower.rs:2413`.
- **Naive device GEMM helper family** — `crates/wukong_codegen_gpu/src/lower.rs:2518-2576` — Triple-loop `sgemm_nt` with beta accumulate, plus NN `sgemm`, exact int8 NT GEMM and the fused bias/relu/gelu/silu epilogue at `crates/wukong_codegen_gpu/src/lower.rs:2580-2638`, `crates/wukong_codegen_gpu/src/lower.rs:2642-2691`, `crates/wukong_codegen_gpu/src/lower.rs:3588-3679`.
- **Row-wise norm device kernels** — `crates/wukong_codegen_gpu/src/lower.rs:2699-2928` — `mrt_norm` implements softmax/layernorm/rmsnorm/log-softmax/l2norm with SFU `ex2/lg2` and eps passed as f32 bits; the affine γ/β variant with null-pointer defaults is at `crates/wukong_codegen_gpu/src/lower.rs:2934-3019`.
- **36-op activation kernel `mrt_vmath` with inline atan minimax** — `crates/wukong_codegen_gpu/src/lower.rs:3025-3232` — A branch table over op codes 0..35 using SFU approximations, with atan/asin/acos built from a 5-term minimax polynomial plus a `1/|x|` range reduction since PTX lacks those SFU ops.
- **Two-arg transcendental and streaming-elementwise kernels** — `crates/wukong_codegen_gpu/src/lower.rs:3238-3285` — `mrt_vmath2` does pow/atan2 (with quadrant fix)/hypot, and `mrt_velem` at `crates/wukong_codegen_gpu/src/lower.rs:3293-3341` does hadamard/div/affine `a·x+b·y+c` with identity/relu/relu6 epilogues.
- **Mixed-precision helper generators that splice the f32 op table** — `crates/wukong_codegen_gpu/src/lower.rs:3397-3436` — `ptx_vmath_lowp` string-slices the committed `PTX_VMATH` dispatch between two literal markers so bf16/f16 share one source of truth; dot/sum/reduce/axpby twins at `crates/wukong_codegen_gpu/src/lower.rs:3439-3583`.
- **Device execution: PTX-hash module key, dump knob, single-thread launch** — `crates/wukong_codegen_gpu/src/lower.rs:3685-3731` — Leaks a hash-derived cache key so the shared `Gpu` module/cubin caches apply, honours `WUKONG_GPU_DUMP_PTX`, writes failing PTX to temp, and launches a 1×1 grid over the ctx buffer.

## G37 — GPU: megakernel, fusion, CUDA graphs, device pool, cubin cache, autotuner

*Files:* `crates/wukong_codegen_gpu/src/autotune.rs`, `crates/wukong_codegen_gpu/src/cubin.rs`, `crates/wukong_codegen_gpu/src/fusion.rs`, `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/graph.rs`, `crates/wukong_codegen_gpu/src/megakernel.rs`, `crates/wukong_codegen_gpu/src/pool.rs`

**`crates/wukong_codegen_gpu/src/autotune.rs`**

- **int8 GEMM autotune search and candidate set** — `crates/wukong_codegen_gpu/src/autotune.rs:32-199` — Nine shape-filtered kernels (smdb/swz 64/128, w64, raster-8, split-K 2/4/8) ranked best-of-6×50, each asserted byte-identical to the first before any timing counts.
- **On-disk autotune cache and cached dispatch** — `crates/wukong_codegen_gpu/src/autotune.rs:203-308` — Hand-rolled `<dtype> <m> <n> <k> = <config> <gflops>` text whose parser skips malformed lines so corruption degrades to re-tuning; tuned launcher `crates/wukong_codegen_gpu/src/autotune.rs:362-388`, 1.10-threshold regression revalidation `crates/wukong_codegen_gpu/src/autotune.rs:311-357`.
- **W4A16 split-K autotuner** — `crates/wukong_codegen_gpu/src/autotune.rs:397-553` — Times sk∈{1,2,4,8} (split GEMM plus fixed-order reduce) and cross-checks each against the un-split output within fp16 tolerance since int4 candidates are not bit-exact.
**`crates/wukong_codegen_gpu/src/cubin.rs`**

- **Persistent cubin cache** — `crates/wukong_codegen_gpu/src/cubin.rs:1-111` — Runs the in-driver `cuLink*` JIT to get SASS bytes, keys them by driver version plus PTX length and a `DefaultHasher` digest under `WUKONG_CUBIN_CACHE`, and writes atomically via temp-file rename with graceful fallback.
**`crates/wukong_codegen_gpu/src/fusion.rs`**

- **Cooperative-op symbol classifier** — `crates/wukong_codegen_gpu/src/fusion.rs:30-105` — Maps 13 `wukong_*` runtime symbol families (both serial and `_parallel` arms) to `CoopKind`, with print/println/assert as replayable side effects and everything else rejected.
- **Memory-taint fixpoint for SPMD safety** — `crates/wukong_codegen_gpu/src/fusion.rs:145-237` — Seeds Load and every Call result as tainted then propagates through operands and block params to a fixpoint, so loop-carried values converge.
- **Megakernel eligibility analysis** — `crates/wukong_codegen_gpu/src/fusion.rs:241-316` — Rejects entries with params, data-dependent branches, user calls, unrecognized symbols, tainted coop-op arguments, `wukong_parallel_for`, or zero recognized ops; unit gates at `crates/wukong_codegen_gpu/src/fusion.rs:355-458`.
**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **Persistent cubin cache load ladder** — `crates/wukong_codegen_gpu/src/gpu.rs:64-87` — Tries driver-tagged cached cubin, else compiles+persists one, else falls back to direct PTX JIT, so caching can never break a working load.
- **f32 GPU-resident transformer encoder layer** — `crates/wukong_codegen_gpu/src/gpu.rs:2842-2966` — Chains RMSNorm, three register-blocked GEMM projections, flash attention, vadd residuals and a vmath SiLU on device buffers with a single upload and download.
- **ResidentLayerF16 construction and shape gates** — `crates/wukong_codegen_gpu/src/gpu.rs:3052-3142` — Preloads every kernel once, narrows weights to f16, and asserts S/D/Dff multiples of 64, a supported flash head dim, and tensor-core flash availability when multi-head.
- **Fused-epilogue fp16 layer forward** — `crates/wukong_codegen_gpu/src/gpu.rs:3246-3322` — The production device chain where SiLU rides the up-projection store and both residuals seed the WMMA accumulator via `wmma.load.c`, eliminating three elementwise kernels.
- **ResidentModelF16 N-layer resident stack** — `crates/wukong_codegen_gpu/src/gpu.rs:3443-3496` — Chains N resident layers so each layer's device output feeds the next, giving one H2D and one D2H for a whole-model forward.
- **Pooled, alloc-free layer forward for graph capture** — `crates/wukong_codegen_gpu/src/gpu.rs:3775-3913` — Reimplements `forward_device` with every intermediate bump-allocated from a `DevicePool` on an explicit stream, so CUDA-graph capture records pure launches.
- **Cubin cache round-trip gate and module-load latency bench** — `crates/wukong_codegen_gpu/src/gpu.rs:6264-6347` — Asserts `ptx_to_cubin` emits a real ELF whose entries resolve after `Ptx::from_file`, then reports cold JIT vs driver-warm PTX vs cubin load best-of-10 and the ratio against a claimed 30–120 s Triton cold autotune.
- **Pooled and CUDA-graphed resident-layer bit-identical gates** — `crates/wukong_codegen_gpu/src/gpu.rs:13223-13269` — Poisons the pool slab with 0xFF so any read-before-write leaks NaN, then demands bit-for-bit equality with the eager forward; graph capture and whole-stack variants at `crates/wukong_codegen_gpu/src/gpu.rs:13338-13385`, `crates/wukong_codegen_gpu/src/gpu.rs:13900-13938`, helpers at `crates/wukong_codegen_gpu/src/gpu.rs:13279-13329` and `crates/wukong_codegen_gpu/src/gpu.rs:13863-13893`.
- **Multi-stream overlap pipeline, its serial-equality gate, and concurrency throughput** — `crates/wukong_codegen_gpu/src/gpu.rs:13488-13593` — Double-buffered pinned-staging H2D‖compute‖D2H pipeline with explicit events; gated bit-identical at `crates/wukong_codegen_gpu/src/gpu.rs:13623-13670` and measured at `crates/wukong_codegen_gpu/src/gpu.rs:13681-13711` and `crates/wukong_codegen_gpu/src/gpu.rs:13746-13797`, with graph decode latency at `crates/wukong_codegen_gpu/src/gpu.rs:13944-13981`.
**`crates/wukong_codegen_gpu/src/graph.rs`**

- **CUDA graph capture, instantiate and replay** — `crates/wukong_codegen_gpu/src/graph.rs:110-186` — Drives raw `begin/end_capture` + `cuGraphInstantiateWithFlags(0)` to avoid cudarc's forced AUTO_FREE flag, always ending capture even when recording fails; pinned staging buffer at `crates/wukong_codegen_gpu/src/graph.rs:46-105`.
**`crates/wukong_codegen_gpu/src/megakernel.rs`**

- **Cooperative megakernel launch path** — `crates/wukong_codegen_gpu/src/megakernel.rs:46-125` — Runs an eligible whole program as one 256-thread CTA, hashing PTX into a leaked module key and decoding the shared print/exit context buffer; block size at `crates/wukong_codegen_gpu/src/megakernel.rs:33-36`.
- **Megakernel corpus oracle gate with device-fault ledger** — `crates/wukong_codegen_gpu/src/megakernel.rs:183-296` — Runs every eligible tests/run program at -O0/-O3 against the interpreter, separating miscompiles from sticky CUDA faults and loudly listing programs skipped after device loss.
- **Megakernel same-run A/B bench family** — `crates/wukong_codegen_gpu/src/megakernel.rs:304-540` — Three ignored benches (reduce, vmath, row-chunked GEMM) cross-check mega==single==oracle before reporting only clock-invariant ratios; launch-structure bench at `crates/wukong_codegen_gpu/src/megakernel.rs:554-649`.
**`crates/wukong_codegen_gpu/src/pool.rs`**

- **Device bump-arena memory pool** — `crates/wukong_codegen_gpu/src/pool.rs:98-226` — One slab, aligned cursor bump, O(1) reset and high-water tracking; `PoolBuf` drop leaks rather than frees (`crates/wukong_codegen_gpu/src/pool.rs:54-96`), plus the NaN-poison aid at `crates/wukong_codegen_gpu/src/pool.rs:197-201`.

## G38 — GPU: serving stack (paged KV, continuous batching, TP sim)

*Files:* `crates/wukong_codegen_gpu/src/paged_attention.rs`, `crates/wukong_codegen_gpu/src/paged_kv.rs`, `crates/wukong_codegen_gpu/src/serving.rs`

**`crates/wukong_codegen_gpu/src/paged_attention.rs`**

- **Warp-cooperative paged decode-attention PTX generator (f16 cache)** — `crates/wukong_codegen_gpu/src/paged_attention.rs:56-171` — Lanes stride the context, walk the block table per position, and merge partial online-softmax states by butterfly; the module doc still claims a one-thread-per-(slot,head) kernel.
- **Paged-attention launch configuration and ABI wiring** — `crates/wukong_codegen_gpu/src/paged_attention.rs:186-216` — Maps `KvConfig` to the thirteen kernel params and grids `num_slots*heads` warps at `32*PAGED_ATTN_WARPS` threads; warp constant at `crates/wukong_codegen_gpu/src/paged_attention.rs:42`.
- **KV-append scatter kernel writing new tokens through the block table** — `crates/wukong_codegen_gpu/src/paged_attention.rs:406-464` — One thread per (slot, channel) narrows f32 to f16 and stores at the reserved `(phys, off)`, skipping inactive slots whose padded table points at live block 0; launcher `crates/wukong_codegen_gpu/src/paged_attention.rs:478-512`.
**`crates/wukong_codegen_gpu/src/paged_kv.rs`**

- **Paged KV geometry, offsets and budget guards** — `crates/wukong_codegen_gpu/src/paged_kv.rs:53-184` — One row-major offset rule shared by append and attention kernels, int8 scale-slab indexing, plus pre-allocation budget asserts and a max-context-within-budget solver.
- **Host block-table allocator with layout epoch** — `crates/wukong_codegen_gpu/src/paged_kv.rs:193-389` — LIFO free list, per-slot tables, append/reserve/free/locate and padded flattening; a monotone epoch bumped only on table changes lets steady-state decode skip table re-uploads.
- **Device paged cache with f16/int8 storage dispatch** — `crates/wukong_codegen_gpu/src/paged_kv.rs:399-571` — Allocates K/V slabs (int8 adding two per-(token,head) scale slabs), exposes split borrows for the append launcher, and panics loudly on wrong-dtype slab accessors.
**`crates/wukong_codegen_gpu/src/serving.rs`**

- **Pooled decode layer forward step** — `crates/wukong_codegen_gpu/src/serving.rs:57-279` — RMSNorm→QKV→cache append→paged attention→O-proj with fused residual→SiLU FFN, every scratch from the pool and `attn` allocated before the dtype match so bump order stays graph-stable.
- **Resident N-layer decode model execution** — `crates/wukong_codegen_gpu/src/serving.rs:283-418` — Ping-pongs two persistent `[Bcap,D]` buffers across layers with a pool reset per layer, using disjoint field borrows so cache storage, pool and metadata are live simultaneously.
- **Masked metadata advance and epoch-skipped upload** — `crates/wukong_codegen_gpu/src/serving.rs:435-495` — Appends only active slots, uploads the large flat block table solely when the layout epoch moved, and `upload_metadata` re-steers a captured graph by contents alone.
- **Continuous vs static batching admission policy** — `crates/wukong_codegen_gpu/src/serving.rs:529-672` — First-fit admission with a 64-deep look-ahead that avoids head-of-line blocking (documented fairness cost), and a `new_static` peer policy admitting only once the whole batch drains.
- **Capture-once graphed scheduler step** — `crates/wukong_codegen_gpu/src/serving.rs:679-763` — First call executes eagerly then records the launch half; later calls replay one `cuGraphLaunch`, asserting identical x/out pointers since masks and tables are re-read device contents.
- **Serving correctness gates** — `crates/wukong_codegen_gpu/src/serving.rs:996-1023` — Decode step vs f64 reference, plus append round-trip/mask, int8 quant exactness, block-layout and batch-composition bit-invariance, graph==eager, scheduler drain/conservation: `crates/wukong_codegen_gpu/src/serving.rs:883-925`, `crates/wukong_codegen_gpu/src/serving.rs:1086-1289`, `crates/wukong_codegen_gpu/src/serving.rs:1399-1806`, `crates/wukong_codegen_gpu/src/paged_kv.rs:580-758`.
- **Continuous-batching goodput sweep** — `crates/wukong_codegen_gpu/src/serving.rs:1822-2053` — One graph per Bcap re-steered by fill, timed interleaved best-of-N against a throttling clock, then a real graph-driven Scheduler drain comparing continuous against static batching.
- **Tensor-parallel partition simulation** — `crates/wukong_codegen_gpu/src/serving.rs:2078-2138` — Column-parallel N-splits are proven bit-identical to the unsplit GEMM while row-parallel K-splits match only within tolerance, mirroring a real ring all-reduce's reassociation.

## G39 — GPU: training-resident path and backward PTX

*Files:* `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs`, `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs`, `crates/wukong_codegen_gpu/src/train_resident.rs`

**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **fp8 backward GEMM (E5M2 grad x E4M3 weight)** — `crates/wukong_codegen_gpu/src/gpu.rs:4492-4525` — Rounds dY to the wide-range E5M2 gradient format and W to E4M3, accumulating f32 — the `dX`/`dW` building block of an fp8 training step.
- **Deterministic per-tensor amax calibration** — `crates/wukong_codegen_gpu/src/gpu.rs:4533-4555` — Grid-strided per-thread max-abs partials over a clamped 1024x256 grid with a host-side final max; atomics-free but not yet fully device-resident.
- **Device delayed-scaling fp8 quantize** — `crates/wukong_codegen_gpu/src/gpu.rs:4562-4593` — Casts `x·recip` to E5M2 or E4M3 wholly on GPU via Ada's packed `cvt.rn.satfinite` instruction, requiring an even element count since two f32 pack per convert.
**`crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs`**

- **Transpose and fused transpose-narrow-to-f16 backward kernels** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:36-151` — Grid-stride naive scatter transposes for the `dB = dC^T A` path, the second folding `cvt.rn.f16.f32` in so NT-only WMMA operand prep costs one launch.
- **Elementwise activation-backward kernel family and dispatch** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:180-310` — One PTX module with relu/sigmoid/tanh/exp entries under a uniform `(dout,x,y,dx,n)` signature, each reading only its needed operand; op-code routing at `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:313-354`.
- **Row-norm backward kernels (softmax / LayerNorm / RMSNorm)** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:433-509` — One warp per row builds up to four butterfly all-reduced row statistics then writes `dx` in a second strided pass, recomputing sigma from x; host launcher at `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:520-561`.
- **Training glue elementwise kernels** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:687-778` — `relu_fwd`, `mse_grad` (`dy = scale*(y-t)`), `scale_inplace`, and `causal_mask` (writes -inf above the diagonal), the small ops the resident step and attention backward need.
- **Materialized flash-attention backward chain** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:1184-1235` — Recomputes P rather than stashing it, then derives dV/dP/dS/dQ/dK from gated GEMM, scale, causal-mask and softmax-backward primitives with O(S^2) workspace.
**`crates/wukong_codegen_gpu/src/ptx_fp8_train.rs`**

- **fp8 backward GEMM with parametric operand types** — `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:79-160` — Re-emits the fragment-reuse fp8 GEMM with swappable `mma` type tokens, yielding the e5m2×e4m3 gradient and e5m2×e5m2 entries at `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:165-175`.
- **Deterministic per-tensor amax reduction** — `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:187-232` — Grid-stride max-abs writing one partial per thread with no atomics, giving a run-to-run-identical calibration statistic for delayed scaling.
- **Delayed-scaling quantization family (E5M2 codec, host and device)** — `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:261-314` — Device kernel packing two scaled f32 per `cvt.rn.satfinite.{e5m2x2,e4m3x2}.f32`; E5M2 codec `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:26-72`, scale/quantize helpers `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:238-254`, gates `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:332-366`.
**`crates/wukong_codegen_gpu/src/train_resident.rs`**

- **Resident MLP trainer buffers and step** — `crates/wukong_codegen_gpu/src/train_resident.rs:76-243` — Keeps weights, grads, AdamW moments and all activations in device buffers so forward, backward and the fused optimizer issue only launches; precision selector and GEMM dispatch at `crates/wukong_codegen_gpu/src/train_resident.rs:27-50`.
- **Gradient-checkpointing activation recompute** — `crates/wukong_codegen_gpu/src/train_resident.rs:178-183` — Regenerates `H_pre` and `H` from current weights inside the backward instead of stashing them, deterministically so gradients stay bit-identical to the stashed path.

## G40 — GPU: correctness gates and peer baselines

*Files:* `bench/pytorch/transformer_layer_peer.py`, `crates/wukong_codegen_gpu/src/baselines.rs`, `crates/wukong_codegen_gpu/src/diff.rs`, `crates/wukong_codegen_gpu/src/gpu.rs`, `crates/wukong_codegen_gpu/src/lower.rs`, `crates/wukong_codegen_gpu/src/paged_attention.rs`, `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs`, `crates/wukong_codegen_gpu/src/ptx_fp8.rs`, `crates/wukong_codegen_gpu/src/ptx_int4.rs`, `crates/wukong_codegen_gpu/src/ptx_optim.rs`, `crates/wukong_codegen_gpu/src/train_resident.rs`, `crates/wukong_driver/src/lib.rs`, `crates/wukong_xbench/src/model.rs`, `tools/fa2_sdpa_peer.py`

**`bench/pytorch/transformer_layer_peer.py`**

- **PyTorch transformer-layer GPU peer with f64 oracle** — `bench/pytorch/transformer_layer_peer.py:92-141` — Every torch path is validated against an f64 reference of the identical layer before its time is reported, with clock-repinning best-of-N timing, also `bench/pytorch/transformer_layer_peer.py:200-256`.
**`crates/wukong_codegen_gpu/src/baselines.rs`**

- **Peer availability probes** — `crates/wukong_codegen_gpu/src/baselines.rs:48-79` — Caches a `catch_unwind` probe of NVRTC+cuBLAS so a missing redist DLL skips instead of aborting; siblings for cuBLASLt, cuDNN and the torch venv at `crates/wukong_codegen_gpu/src/baselines.rs:1850-1861`, `crates/wukong_codegen_gpu/src/baselines.rs:1940-1947`, `crates/wukong_codegen_gpu/src/baselines.rs:2259-2276`.
- **NVRTC naive CUDA-C peer family (Tier A)** — `crates/wukong_codegen_gpu/src/baselines.rs:90-176` — Runtime-compiled hand-written kernels sharing one warm-up/sync timing shape: attention `crates/wukong_codegen_gpu/src/baselines.rs:188-300`, W4A16 `crates/wukong_codegen_gpu/src/baselines.rs:1096-1187`, conv `crates/wukong_codegen_gpu/src/baselines.rs:1198-1314`, int8 naive+dp4a `crates/wukong_codegen_gpu/src/baselines.rs:1331-1478`.
- **cuBLAS unfused attention chain peer** — `crates/wukong_codegen_gpu/src/baselines.rs:326-490` — Pre-FlashAttention bar: tensor-core QKᵀ, Wukong's own softmax and cast glue, then P·V, deliberately materializing the S×S scores to HBM that fused flash never pays.
- **cuBLAS row-major↔column-major NT mapping** — `crates/wukong_codegen_gpu/src/baselines.rs:496-572` — Encodes `Cᵀ = B̌ᵀ·Ǎ` so cuBLAS computes Wukong's `A·Bᵀ`; the f16-in/f32-out fairness variant that matches Wukong's store dtype is at `crates/wukong_codegen_gpu/src/baselines.rs:585-688`.
- **cuBLAS call-chain transformer layer and model** — `crates/wukong_codegen_gpu/src/baselines.rs:697-1077` — Substitutes cuBLAS for the six projections while reusing Wukong's identical norm/flash/cast/SiLU/vadd and multi-head transpose kernels, isolating GEMM quality plus unfusable residual/activation launches.
- **cuBLAS int8 IMMA peer with signedness caveat** — `crates/wukong_codegen_gpu/src/baselines.rs:1494-1585` — `cublasGemmEx` s8×s8→s32 tensor-core peer; since no mixed u8×s8 form exists, callers must restrict activations to [0,127] for the bit-exact cross-check.
- **cuBLASLt fp8 E4M3 matmul plan** — `crates/wukong_codegen_gpu/src/baselines.rs:1611-1843` — Raw-sys handle/descriptor/layout/heuristic plan with device 1.0 scales and RAII teardown, built once outside timing; K%16 and N%4 alignment asserted up front.
- **cuDNN conv2d peer with layout shims** — `crates/wukong_codegen_gpu/src/baselines.rs:1877-2083` — NCHW→NHWC/KRSC f16 transposes outside the timed loop, `TENSOR_OP_MATH` enabled, v7 heuristic algorithm chosen and named so the reported percentage discloses which engine ran.
- **int8 GEMM+dequant chain peer** — `crates/wukong_codegen_gpu/src/baselines.rs:2098-2190` — Hand-written per-channel dequant PTX plus cuBLAS int8, timed as one pair: the HBM round-trip and extra launch Wukong's fused epilogue removes.
- **PyTorch fused-SDPA subprocess peer** — `crates/wukong_codegen_gpu/src/baselines.rs:2217-2388` — Dumps f16 Q/K/V to a temp dir, runs a Python harness that CUDA-event-times cuDNN/efficient/math backends, and parses a flat key=value report; RoPE pipeline variant at `crates/wukong_codegen_gpu/src/baselines.rs:2399-2519`.
**`crates/wukong_codegen_gpu/src/diff.rs`**

- **Tolerance differential harness (`diff.rs`)** — `crates/wukong_codegen_gpu/src/diff.rs:1-107` — SplitMix64 PRNG for reproducible inputs plus f64-computed `err_stats`/`assert_close`/`assert_scalar_close` where a lane passes on either the absolute or relative bound.
**`crates/wukong_codegen_gpu/src/gpu.rs`**

- **fp16 tensor-core roofline self-measurement** — `crates/wukong_codegen_gpu/src/gpu.rs:1810-1856` — Register-resident `wmma.mma` loop with warm-up plus best-of-reps timing yields an internal FLOP/s ceiling the docstring admits under-estimates cuBLAS.
- **Unfused reference forward for fusion measurement** — `crates/wukong_codegen_gpu/src/gpu.rs:3331-3404` — Same layer with plain GEMMs plus separate vadd and silu launches, existing solely to quantify what the fused epilogues buy at full-layer level.
- **fp8 delayed-scaling quantize + amax gates** — `crates/wukong_codegen_gpu/src/gpu.rs:4600-4626` — Device `cvt`-based e5m2/e4m3 quantize must round-trip within 13%/7% relative of `x·recip`; companion amax gate asserts bit-exact `==` against CPU max-abs, extra site `crates/wukong_codegen_gpu/src/gpu.rs:4628-4643`.
- **fp8 backward GEMM (E5M2·E4M3) f64 gate** — `crates/wukong_codegen_gpu/src/gpu.rs:4644-4678` — Checks `gemm_nt_fp8_bwd` against an f64 reference decoding the same fp8 bits, using the honest `8·√K·ε` (min 2e-3) accumulation bound rather than an fp8-slack fudge.
- **`with_gpu` device-optional test harness** — `crates/wukong_codegen_gpu/src/gpu.rs:4680-4687` — Locks the global GPU mutex and either runs the body or prints `[skip] <name>: no CUDA device reachable`, so every gate below silently passes on CPU-only machines.
- **Driver JIT error-log diagnostics** — `crates/wukong_codegen_gpu/src/gpu.rs:4889-4923` — `cuModuleLoadDataEx` with `CU_JIT_ERROR_LOG_BUFFER` attached prints the real ptxas line behind `CUDA_ERROR_INVALID_PTX` and dumps the PTX to temp; generic twin hardcodes `wmma_f16_ptx()` at `crates/wukong_codegen_gpu/src/gpu.rs:6229-6262`.
- **Bit-exact device primitive gates (saxpy/vadd/copy-v4/f32→f16 cast)** — `crates/wukong_codegen_gpu/src/gpu.rs:4925-4990` — Four `to_bits()`-equality gates covering fused-multiply-add semantics, a non-block-multiple length, the grid-stride float4 copy, and `cvt.rn.f16.f32` vs `half::f16::from_f32` on ties/edges.
- **vmath and norm gates against the CPU runtime oracle** — `crates/wukong_codegen_gpu/src/gpu.rs:5000-5026` — Calls `wukong_runtime::wukong_vmath_f32` / `wukong_norm_f32` as the oracle so GPU SFU activations and fused softmax/LayerNorm/RMSNorm must match the CPU dispatch path, helpers at `crates/wukong_codegen_gpu/src/gpu.rs:4992-4998` and gate at `crates/wukong_codegen_gpu/src/gpu.rs:6349-6386`.
- **Reduction gate: f64 reference, exact max, determinism** — `crates/wukong_codegen_gpu/src/gpu.rs:5028-5070` — Over 2^20 positive elements sum/dot go against an f64 accumulation at 1e-3 rel, max must be bit-exact, and a repeat sum must be bit-identical.
- **f64 GEMM oracles and the c·√K·ε tolerance policy** — `crates/wukong_codegen_gpu/src/gpu.rs:5072-5161` — `ref_nt`/`ref_nn` f64 references plus the ragged-shape f32 GEMM gates using `max(8·√K·ε, 1e-4)`; the dtype-rounding oracle `ref_nt_rounded` every tensor-core gate shares lives at `crates/wukong_codegen_gpu/src/gpu.rs:5163-5187`.
- **Fused activation and residual epilogue gates** — `crates/wukong_codegen_gpu/src/gpu.rs:5449-5603` — relu/silu/gelu and `wmma.load.c` residual seeding must equal the activation of the f16-rounded reference GEMM, proving the load.c/store.d fragment maps are inverse and the SFU epilogue is correctly placed.
- **Fused bias(+act)(+residual) epilogue gates across four kernel families** — `crates/wukong_codegen_gpu/src/gpu.rs:5605-5954` — Gates the SMEM-scratch (wmma sm_db), register-level (mma workhorse), deep-pipe64, and bf16 mma bias epilogues as `act(round(A·Bᵀ)+bias[col])`, catching per-column map errors; bf16 sm_db twin at `crates/wukong_codegen_gpu/src/gpu.rs:6166-6227`.
- **Gated-FFN (SwiGLU/GeGLU/GLU) dual-B kernel gates** — `crates/wukong_codegen_gpu/src/gpu.rs:5956-6050` — Twenty fp16/bf16 variants (±bias, padded and no-pad `_swz`) checked as `act(round(x·Wgᵀ)[+bg])⊙(round(x·Wuᵀ)[+bu])` with compounded-error tolerances; fp8 lane at `crates/wukong_codegen_gpu/src/gpu.rs:6052-6112`, bf16 sm_db fused acts at `crates/wukong_codegen_gpu/src/gpu.rs:6114-6164`.
- **`ref_conv2d` f64 oracle and the valid-conv gates** — `crates/wukong_codegen_gpu/src/gpu.rs:7472-7580` — Direct 6-loop f64 convolution reference plus the f32 tiled and fp16 tensor-core implicit-GEMM gates, the latter using a `4·√(C·R·S)·2^-10` fp16-input error model.
- **Affine/epilogue/split-K conv gates and the best-path dispatch gate** — `crates/wukong_codegen_gpu/src/gpu.rs:7639-8056` — Six gates covering fused bias+act conv, strided, bounds-checked padded, explicit-pad scatter, auto split-K padded, and `conv2d_best` routing (Frobenius-gated whichever lane it picks); split-K determinism at `crates/wukong_codegen_gpu/src/gpu.rs:8320-8359`.
- **Conv peer benches vs naive CUDA-C, cuDNN and Winograd** — `crates/wukong_codegen_gpu/src/gpu.rs:8058-8318` — Fusion-vs-unfused and Winograd-vs-cuDNN harnesses that cross-check every contender before timing; further peer sweeps at `crates/wukong_codegen_gpu/src/gpu.rs:8361-8494`, `crates/wukong_codegen_gpu/src/gpu.rs:8496-8712`, `crates/wukong_codegen_gpu/src/gpu.rs:8714-8955`, `crates/wukong_codegen_gpu/src/gpu.rs:8957-9078`.
- **Transformer-layer f64 reference family** — `crates/wukong_codegen_gpu/src/gpu.rs:9080-9230` — `ref_rmsnorm`/`ref_silu`/`ref_transformer_layer{,_f16,_f16_mha}`/`ref_model_f16`/`ref_ffn` compose the primitive oracles; the f16 variants round only the four GEMM inputs, leaving norm/flash/residual in full precision to mirror the kernels.
- **Transformer layer and resident-model correctness gates** — `crates/wukong_codegen_gpu/src/gpu.rs:9232-9426` — Gates fused FFN, f32 layer, GPT-2-shaped 12-head layer, the f16 layer across both flash dispatch lanes, and a 4-deep resident model, each also asserting run-to-run bit-identity.
- **M12 bit-reproducibility gate and the cuBLAS reproducibility log** — `crates/wukong_codegen_gpu/src/gpu.rs:9428-9500` — A `twice_eq!` macro runs every reducing kernel family twice and demands identical bits; the peer comparison asserting Wukong stability while merely *logging* cuBLAS's is at `crates/wukong_codegen_gpu/src/gpu.rs:9502-9534`.
- **Layer/model throughput, fusion and scaling benches** — `crates/wukong_codegen_gpu/src/gpu.rs:9536-9743` — Clock-pinned (1500 warm GEMMs + a re-pin per measurement) benches for f32-vs-f16-vs-resident, depth scaling, fused-vs-unfused at `crates/wukong_codegen_gpu/src/gpu.rs:9950-10032`, and the flash-vs-GEMM scaling diagnostic at `crates/wukong_codegen_gpu/src/gpu.rs:9868-9948`.
- **cuBLAS call-chain peer gate and the M13 head-to-head benches** — `crates/wukong_codegen_gpu/src/gpu.rs:10034-10096` — Validates the `cublasGemmEx` f16-in/f32-out primitive, the whole chain layer, and chain-vs-Wukong agreement before any speed claim; benches decomposing GEMM-quality vs fusion at `crates/wukong_codegen_gpu/src/gpu.rs:10098-10214` and `crates/wukong_codegen_gpu/src/gpu.rs:10216-10320`, plus the depth sweep at `crates/wukong_codegen_gpu/src/gpu.rs:9745-9866`.
- **fp8 E4M3 GEMM tolerance gates (single/pipe/regime)** — `crates/wukong_codegen_gpu/src/gpu.rs:10363-10446` — Checks `gemm_nt_fp8`/`_pipe` against an E4M3-rounded f64 reference with a `max(8√K·ε, 2e-3)` relative bound, exercising the m64/w64/w64_s3 dispatch boundaries.
- **fp8 fused bias / activation / residual epilogue gates** — `crates/wukong_codegen_gpu/src/gpu.rs:10454-10557` — Verifies `gemm_nt_fp8_mma_bias{,_relu,_silu,_gelu}` and `_bias_residual` equal `act(rounded(A·Bᵀ)+bias[col])(+resid)`, gating the m16n8k32 D-fragment column map and per-element residual addressing.
- **Internal GEMM throughput ladder and fp16 roofline percentage** — `crates/wukong_codegen_gpu/src/gpu.rs:10588-10716` — Times naive vs register-blocked f32 vs f16/bf16 WMMA vs fp8 tensor-core kernels same-run, then reports % of the measured fp16 roofline at `crates/wukong_codegen_gpu/src/gpu.rs:11057-11112` with a sanity check at `crates/wukong_codegen_gpu/src/gpu.rs:12684-12694`.
- **cuBLASLt fp8 peer gate plus %-of-cuBLASLt scoreboard and config sweep** — `crates/wukong_codegen_gpu/src/gpu.rs:10802-10840` — Gates the cuBLASLt E4M3 peer against the same f64 oracle before timing, then reports Wukong pipe/mt as % of it; sweeps five tile/stage/raster configs at `crates/wukong_codegen_gpu/src/gpu.rs:10846-10994`.
- **fp8 tile and warp-tile dispatch levers (m64 vs default, w2×2 vs w2×4)** — `crates/wukong_codegen_gpu/src/gpu.rs:11002-11050` — Interleaved round-by-round A/B that cancels the shared clock, plus a checksum-equality gate and %-of-cuBLASLt warp-tiling sweep at `crates/wukong_codegen_gpu/src/gpu.rs:15218-15248` and `crates/wukong_codegen_gpu/src/gpu.rs:15257-15332`.
- **gemm_vs_peers tiered scoreboard (cuBLAS + NVRTC naive CUDA-C)** — `crates/wukong_codegen_gpu/src/gpu.rs:11127-11263` — Pre-gates both peers against the f64 oracle, hammers 40 warmup GEMMs to settle the mobile clock, then reports Wukong variants as % of cuBLAS with checksum cross-checks.
- **Multi-head layout shim bit-exact gate** — `crates/wukong_codegen_gpu/src/gpu.rs:11271-11325` — Asserts `cast_transpose_qkv` (f32 `[S,H·dh]`→f16 `[H,S,dh]`) and `transpose_attn_out` match a CPU reference bit-for-bit including f16 rounding, covering H=1 and the GPT-2 H=12 shape.
- **NVRTC nvcuda::wmma capability probe** — `crates/wukong_codegen_gpu/src/gpu.rs:11334-11378` — Empirically asks whether the toolkit-free redist NVRTC bundles `mma.h`, deciding whether a genuinely fused FA2-class CUDA-C peer is buildable; informational only, never fails.
- **cp.async pipeline-variant sweep and swizzle-vs-hand-placed SMEM A/B** — `crates/wukong_codegen_gpu/src/gpu.rs:11390-11483` — Times every `PIPE_VARIANTS` entry dividing each size against cuBLAS to pick the dispatched winner; the no-pad XOR-swizzle-vs-padded occupancy bet is isolated at `crates/wukong_codegen_gpu/src/gpu.rs:11490-11551`.
- **GEMM-cliff correctness gates (CLIFF_VARIANTS and dispatched w22swz)** — `crates/wukong_codegen_gpu/src/gpu.rs:11569-11599` — Every cliff candidate whose macro-tile divides the shape must match the f16-rounded f64 oracle (abs 1e-2, rel 2e-3); the production w22swz f16/bf16 entries gated at `crates/wukong_codegen_gpu/src/gpu.rs:11605-11650`, launch config at `crates/wukong_codegen_gpu/src/gpu.rs:11556-11562`.
- **GEMM-cliff A/B instruments: round-robin ratio-of-best and offline-ptxas comparison** — `crates/wukong_codegen_gpu/src/gpu.rs:11662-11774` — Times each candidate once per round with a two-slot cuBLAS self-noise sentinel and achieved CTAs/SM; the `WUKONG_PTXAS` standalone-assembler-vs-driver-JIT comparison lives at `crates/wukong_codegen_gpu/src/gpu.rs:11784-11907`.
- **Flash attention vs naive CUDA-C and unfused cuBLAS chain** — `crates/wukong_codegen_gpu/src/gpu.rs:11927-12098` — Pre-gates both peers against the f64 `ref_attn` oracle, then reports `flash_d64_mp` GFLOP/s and speedup over the score-materializing cuBLAS chain across S=512..4096.
- **Fusion-beats-call-chain bench family (activation, bias+act, SwiGLU gate, FFN block)** — `crates/wukong_codegen_gpu/src/gpu.rs:12109-12222` — Times fused epilogues against the two/three-kernel chains cuBLAS forces, proxying the epilogue HBM round-trip with `time_vmath`/`time_vadd`; siblings at `crates/wukong_codegen_gpu/src/gpu.rs:12235-12430`, `crates/wukong_codegen_gpu/src/gpu.rs:12857-12939`, `crates/wukong_codegen_gpu/src/gpu.rs:12440-12569`.
- **HBM bandwidth vs device-derived theoretical peak** — `crates/wukong_codegen_gpu/src/gpu.rs:12578-12654` — Bit-checks the copy kernel first, then measures copy/saxpy/reduce over 256 MB arrays as a % of `peak_hbm_gbs`, declaring the ≥90% milestone met or throttled.
- **Resident-launch timing wrapper family and peak-clock sampling helpers** — `crates/wukong_codegen_gpu/src/gpu.rs:12697-12841` — Near-identical `time_wmma{,_residual,_bias,_bias_residual}`/`time_gate` upload-once-launch-many timers, siblings `time_gemm`/`time_gemm_int8`/`time_w4a16`/`time_vadd`/`time_vmath`, and the best-of-N clock samplers `best_of`/`best_bw`/`min_latency`/`best_batch_pair` at `crates/wukong_codegen_gpu/src/gpu.rs:10560-10584`, `crates/wukong_codegen_gpu/src/gpu.rs:12944-12994`, `crates/wukong_codegen_gpu/src/gpu.rs:12662-12680`, `crates/wukong_codegen_gpu/src/gpu.rs:13392-13410`.
- **W4A16 int4 scoreboard with an honestly-empty Tier-B** — `crates/wukong_codegen_gpu/src/gpu.rs:13040-13184` — Compares int4-decode GEMM to a naive NVRTC W4A16 peer and to Wukong's own fp16 GEMM on the identical 64×64 tile, isolating weight-bandwidth win; timer at `crates/wukong_codegen_gpu/src/gpu.rs:13000-13025`.
- **int8 wrapping-i32 oracle and core bit-exact GEMM/dequant gates** — `crates/wukong_codegen_gpu/src/gpu.rs:13990-14004` — CPU reference accumulates `u8×i8` with wrapping i32 to match the tensor core mod-2³² contract exactly; consumed by the full-range equality gates at `crates/wukong_codegen_gpu/src/gpu.rs:14010-14050` and the fused per-channel dequant gate at `crates/wukong_codegen_gpu/src/gpu.rs:14056-14086`.
- **int8 peer scoreboard and the a127 signedness fixture** — `crates/wukong_codegen_gpu/src/gpu.rs:14212-14346` — Scores Wukong against cuBLAS IMMA, dp4a CUDA-C and naive CUDA-C; `int8_inputs_a127` restricts activations to [0,127] so u8×s8 and s8×s8 peers compute the identical matrix (`crates/wukong_codegen_gpu/src/gpu.rs:14182-14196`).
- **int8 kernel-variant bit-exact gate family (multistage, swizzle, split-K, raster, big-tile, w64)** — `crates/wukong_codegen_gpu/src/gpu.rs:14379-14427` — Each SMEM/pipeline/tiling variant must equal the i32 oracle element-for-element since only staging or tile ownership changes; siblings at `crates/wukong_codegen_gpu/src/gpu.rs:14436-14473`, `crates/wukong_codegen_gpu/src/gpu.rs:14481-14522`, `crates/wukong_codegen_gpu/src/gpu.rs:14620-14662`, `crates/wukong_codegen_gpu/src/gpu.rs:14763-14791`, `crates/wukong_codegen_gpu/src/gpu.rs:14966-15006`, `crates/wukong_codegen_gpu/src/gpu.rs:15093-15124`.
- **int8 %-of-cuBLAS tuning sweeps (stage depth, rasterization, big tile, w64)** — `crates/wukong_codegen_gpu/src/gpu.rs:14530-14612` — Interleaved same-run sweeps that pick the per-size dispatch winner after a bit-exact checksum cross-check; siblings at `crates/wukong_codegen_gpu/src/gpu.rs:14674-14756`, `crates/wukong_codegen_gpu/src/gpu.rs:14800-14876`, `crates/wukong_codegen_gpu/src/gpu.rs:14884-14955`, `crates/wukong_codegen_gpu/src/gpu.rs:15014-15087`.
- **int8 fused GEMM+dequant vs the cuBLAS GEMM+dequant chain** — `crates/wukong_codegen_gpu/src/gpu.rs:15135-15210` — Shows the register-store dequant costs ~0 while cuBLAS must launch a second kernel re-reading M×N i32 from HBM; marginal-cost bench at `crates/wukong_codegen_gpu/src/gpu.rs:14096-14149`.
- **Contention-robust internal A/B family (static-vs-dynamic, swizzle-vs-hand-placed, split-K occupancy)** — `crates/wukong_codegen_gpu/src/gpu.rs:15343-15406` — Wukong-internal same-family ratios that stay honest when the cuBLAS baseline is clock-corrupted; both kernels loaded through the same raw JIT path to avoid the int4 confound (`crates/wukong_codegen_gpu/src/gpu.rs:15462-15516`, `crates/wukong_codegen_gpu/src/gpu.rs:15571-15633`, `crates/wukong_codegen_gpu/src/gpu.rs:15642-15713`).
- **Static-shape (M/N/K baked) bit-exact gates for int8 and fp16** — `crates/wukong_codegen_gpu/src/gpu.rs:15413-15454` — Constant-folded-dim kernels must equal both the oracle and the dynamic kernel exactly, gating 64×64 through the public launcher and 128×128 directly; fp16 twin at `crates/wukong_codegen_gpu/src/gpu.rs:15523-15565`.
- **Attention fused-peer bench family vs PyTorch SDPA (cuDNN/cutlass)** — `crates/wukong_codegen_gpu/src/gpu.rs:15725-15877` — Feeds identical f16 tensors to both, gates output against the f64 oracle, and times Wukong by wall clock vs the peer's CUDA-event time so wins are conservative; variants/RoPE/causal/D=128/ldmatrix at `crates/wukong_codegen_gpu/src/gpu.rs:15882-15980`, `crates/wukong_codegen_gpu/src/gpu.rs:16955-17139`, `crates/wukong_codegen_gpu/src/gpu.rs:17152-17295`, `crates/wukong_codegen_gpu/src/gpu.rs:17751-17850`, `crates/wukong_codegen_gpu/src/gpu.rs:16602-16718`.
- **Flash-attention correctness gate family (wide, pipelined, head-split, RoPE, D=128, ldmatrix, warp-specialized)** — `crates/wukong_codegen_gpu/src/gpu.rs:15986-16034` — Each kernel variant is checked against `ref_attn`/`ref_attn_causal` at ragged and steady sequence lengths; the ws gate also pins the `WUKONG_FLASH_WS` route away from regressing regimes (`crates/wukong_codegen_gpu/src/gpu.rs:16351-16390`, `crates/wukong_codegen_gpu/src/gpu.rs:16725-16764`, `crates/wukong_codegen_gpu/src/gpu.rs:16870-16939`, `crates/wukong_codegen_gpu/src/gpu.rs:17302-17458`, `crates/wukong_codegen_gpu/src/gpu.rs:17472-17622`).
- **Flash internal clock-cancelling A/B family (wide, mp4, ldmatrix, single-buffer, pipelined, head-split, warp-specialized)** — `crates/wukong_codegen_gpu/src/gpu.rs:16042-16129` — Per-round the clock is pinned, both kernels timed in opposite orders, minima taken and the median ratio reported, so occupancy/feed-path levers are measured free of the ~7× clock swing (`crates/wukong_codegen_gpu/src/gpu.rs:16139-16238`, `crates/wukong_codegen_gpu/src/gpu.rs:16249-16344`, `crates/wukong_codegen_gpu/src/gpu.rs:16403-16499`, `crates/wukong_codegen_gpu/src/gpu.rs:16507-16593`, `crates/wukong_codegen_gpu/src/gpu.rs:16772-16863`, `crates/wukong_codegen_gpu/src/gpu.rs:17636-17744`).
**`crates/wukong_codegen_gpu/src/lower.rs`**

- **MIR→PTX coverage gate against the interpreter oracle** — `crates/wukong_codegen_gpu/src/lower.rs:3844-3978` — Runs every `tests/run/*.wk` at -O0 and -O3 through the backend, comparing exit code and float-tolerant output lines, and separates UNSUPPORTED skips from real driver faults and post-fault device-lost programs; PTX JIT-log diagnostic at `crates/wukong_codegen_gpu/src/lower.rs:3983-4020`.
**`crates/wukong_codegen_gpu/src/paged_attention.rs`**

- **f64 full-softmax CPU oracle for decode attention** — `crates/wukong_codegen_gpu/src/paged_attention.rs:677-728` — Recomputes scores, max-shifted softmax and weighted V in f64 per (slot, head), emitting zero rows for inactive slots exactly as the kernel does.
- **Host-only PTX shape and reference gates** — `crates/wukong_codegen_gpu/src/paged_attention.rs:736-791` — Assert the generated PTX unrolls accumulators exactly to head_dim, stages the query in shared, uses IEEE (not approx) division, stays ASCII, and balances braces.
- **GPU gates: tolerance vs f64 reference and bit-identity across physical block layouts** — `crates/wukong_codegen_gpu/src/paged_attention.rs:936-1007` — Ragged contexts including an empty slot are laid out ascending vs descending and asserted bit-equal; int8 twins at `crates/wukong_codegen_gpu/src/paged_attention.rs:1077-1128`, harness at `crates/wukong_codegen_gpu/src/paged_attention.rs:839-927`.
**`crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs`**

- **Backward-kernel correctness gates** — `crates/wukong_codegen_gpu/src/ptx_autodiff_bwd.rs:1252-1574` — Bit-exact transpose check, activation and f64 row-norm backward references, NN/NT/TN GEMM checks, an f16-rounded-operand f64 GEMM gate, and a two-pass-softmax f64 attention-backward reference.
**`crates/wukong_codegen_gpu/src/ptx_fp8.rs`**

- **E4M3 host encode/decode reference pair** — `crates/wukong_codegen_gpu/src/ptx_fp8.rs:12-51` — Round-to-nearest-even f32→e4m3 with 448 saturation and subnormal flush, plus a widener; note the decoder has no NaN case, so the encoder's 0x7f NaN reads back as 480.0.
**`crates/wukong_codegen_gpu/src/ptx_int4.rs`**

- **W4A16 bit-exact dequant reference and f64 oracle** — `crates/wukong_codegen_gpu/src/ptx_int4.rs:183-224` — Reconstructs weights exactly as the kernel does then contracts in f64, so the only residual error is tensor-core accumulation order; host gates at `crates/wukong_codegen_gpu/src/ptx_int4.rs:597-665`.
**`crates/wukong_codegen_gpu/src/ptx_optim.rs`**

- **AdamW correctness and convergence gates** — `crates/wukong_codegen_gpu/src/ptx_optim.rs:336-395` — Steps the GPU kernel and the real `build_adamw_step` MIR on the interpreter in lockstep for six steps counting bit-exact lanes; a 200-step convex-loss monotonicity test follows at `crates/wukong_codegen_gpu/src/ptx_optim.rs:430-460`.
**`crates/wukong_codegen_gpu/src/train_resident.rs`**

- **Resident-trainer gradient, checkpointing and learning gates** — `crates/wukong_codegen_gpu/src/train_resident.rs:349-549` — f64 closed-form MLP backprop reference in both f32 and f16 precision, a bit-identical stashed-vs-recomputed check that poisons the stashed buffers first, and loss-decrease tests.
- **cuBLAS peer baselines and M8 throughput benches** — `crates/wukong_codegen_gpu/src/train_resident.rs:894-977` — Ignored benches timing the fused WMMA resident step against a cuBLAS-fp16 eager chain (`crates/wukong_codegen_gpu/src/train_resident.rs:810-884`) with a parity noise band, plus GEMM percent-of-cuBLAS floors at `crates/wukong_codegen_gpu/src/train_resident.rs:561-728`.
**`crates/wukong_driver/src/lib.rs`**

- **GPU end-to-end differential gates over identical lowered MIR** — `crates/wukong_driver/src/lib.rs:1476-1834` — Runs linear, fused act/bias epilogue, activation, reduction and softmax on interp and GPU, asserting `calls >= 1` so a CPU fallback cannot masquerade as a pass.
**`crates/wukong_xbench/src/model.rs`**

- **PyTorch interpreter probe and MSVC vcvars bootstrap** — `crates/wukong_xbench/src/model.rs:868-934` — Probes `PYTHON`/`python`/`tools/torch-venv` and picks the NEWEST torch; the vcvars64 discovery, batfile env capture and `<base_prefix>\libs`-on-LIB child command are `crates/wukong_xbench/src/model.rs:936-1050`.
- **Generated PyTorch peer script with compile retry ladder** — `crates/wukong_xbench/src/model.rs:1112-1363` — Self-contained script rebuilding the identical f32 forward (F.linear/layer_norm/tanh-GELU/SDPA), timing eager and max-autotune-compiled at 1 and all threads coolest-first, with ATEN-pinning and fullgraph=False fallbacks all disclosed on stdout.
- **Torch peer marshalling, parsing and compiled-output gating** — `crates/wukong_xbench/src/model.rs:1387-1518` — Dumps the once-per-run weight blob plus per-config io blob, runs the script under the MSVC env, parses the tagged timing lines, and accepts a compiled output only at the right size; the LE dumper is `crates/wukong_xbench/src/model.rs:1097-1110`.
**`tools/fa2_sdpa_peer.py`**

- **RoPE-inclusive peer strengthening** — `tools/fa2_sdpa_peer.py:79-129` — Builds three RoPE variants (eager, zero-copy complex-multiply, compiled) and takes the minimum, deliberately making the peer as fast as possible so any Wukong claim stays conservative.
- **Fused FlashAttention-2 GPU peer harness** — `tools/fa2_sdpa_peer.py:131-201` — Times SDPA's flash, mem-efficient, cuDNN and math backends over the caller's raw Q/K/V dumps, choosing the fastest genuinely fused backend and emitting checksums for cross-checking.

# Harnesses

## G41 — Benchmarks: xbench cross-language harness and model bench

*Files:* `crates/wukong_runtime/src/attention.rs`, `crates/wukong_runtime/src/gemm.rs`, `crates/wukong_xbench/src/main.rs`, `crates/wukong_xbench/src/model.rs`, `tools/bench_gpt2_torch.py`, `tools/measure_gpt2.ps1`

**`crates/wukong_runtime/src/attention.rs`**

- **Attention fusion throughput probe vs Wukong's own GEMM path** — `crates/wukong_runtime/src/attention.rs:320-422` — Ignored best-of-20 benchmark comparing the fused kernel against `wukong_sgemm_nt` + row softmax + `wukong_sgemm`, isolating the no-materialization effect from SIMD quality.
**`crates/wukong_runtime/src/gemm.rs`**

- **Throughput probes: pinned single-core sweep, pool A/B, fused-epilogue traffic** — `crates/wukong_runtime/src/gemm.rs:3138-3279` — Ignored benches pinning CPU 0 at high priority and reporting best-of-N GFLOP/s, plus a logical-vs-physical pool ratio; epilogue probe at `crates/wukong_runtime/src/gemm.rs:4291-4348`.
**`crates/wukong_xbench/src/main.rs`**

- **Cross-language kernel ABI + Measure snapshot family** — `crates/wukong_xbench/src/main.rs:29-92` — Four extern-C kernel signatures (f32 3-ptr, u8/i8/i32, bf16-in/f32-scalar, all-half) and matching Measure structs carrying compile time, ns/call and a full output snapshot.
- **Full-buffer cross-language correctness oracles (tight + loose)** — `crates/wukong_xbench/src/main.rs:99-117` — `max_rel_err` returns worst relative element error with NaN/Inf/exact short-circuits and a 1e-6 floor; `relaxed_peer_ok` applies a magnitude-normalized 1e-2 bar and drops unusable relaxed peers, also `crates/wukong_xbench/src/main.rs:131-146`.
- **C(fast) relaxed-FP peer column and its reporting** — `crates/wukong_xbench/src/main.rs:124-178` — Recompiles the identical C source at `-ffast-math`, loose-checks it against Wukong, and prints the ratio alongside (never replacing) the honest-flags C column, also `crates/wukong_xbench/src/main.rs:182-205`.
- **Reduction-row gating list for the C(fast) column** — `crates/wukong_xbench/src/main.rs:209-219` — Hard-coded name match selects which elementwise rows get the extra `-ffast-math` peer; DIVERGENCE: the serial `argmax` row in the kernel table is omitted, so only its @parallel twin is normalized.
- **OpenMP C(omp) peer enablement probe** — `crates/wukong_xbench/src/main.rs:236-284` — One-time cached probe compiles, loads and runs an `omp_get_num_threads` DLL, disabling the multicore-C columns with a printed reason when gcc/libgomp/loader fail, flags at `crates/wukong_xbench/src/main.rs:225-229`.
- **Hand-written OpenMP twins of the elementwise rows** — `crates/wukong_xbench/src/main.rs:290-335` — Per-name C bodies with `parallel for`, `reduction(+/max:)` clauses and an argmax critical-section combine preserving lowest-index-wins serial semantics.
- **CLI substring filter and section dispatch** — `crates/wukong_xbench/src/main.rs:337-359` — First positional arg becomes a `contains` filter over kernel and section names, gating ~35 `want(...)`-guarded benchmark sections including the model bench, also `crates/wukong_xbench/src/main.rs:522-633`.
- **Elementwise-table driver loop** — `crates/wukong_xbench/src/main.rs:360-481` — Rebuilds shared x/y/out buffers per kernel, then times Wukong, gcc C, g++ C++, rustc, optional C(fast) and probed C(omp) through one identical loop and accumulates ratio vectors.
- **oneMKL VML transcendental peer** — `crates/wukong_xbench/src/main.rs:449-471` — Maps exp/log/tanh rows to `vsExp/vsLn/vsTanh`, drops the column when it disagrees with Wukong beyond 1e-3 (treated as an ABI mismatch), else prints the single-thread ratio.
- **matmul size sweep and its measurement env knobs** — `crates/wukong_xbench/src/main.rs:641-662` — Runs 256/512/1024/2048³, adds 4096³ under `XBENCH_HUGE`, and lets `XBENCH_MATMUL_SIZES` restrict the sweep for fast A/B iteration.
- **Skinny transformer-shape GEMM bench (library peers only)** — `crates/wukong_xbench/src/main.rs:672-731` — Six GPT-2 NT shapes at S=128/512 vs MKL 1c/all, reports % of measured roofline and asserts serial-vs-@parallel bit-exactness of the Wukong output.
- **Thermal-hygiene measurement ordering protocol** — `crates/wukong_xbench/src/main.rs:733-896` — Two coolest-first groups (Wuk1c+MKL1c adjacent, naive nests last in group one; MKL(all), C(omp), Wuk(par) last) so residual heat always lands on Wukong, with `XBENCH_NAIVE_HUGE` gating multi-second naive nests.
- **matmul ikj source triplet plus OpenMP twin** — `crates/wukong_xbench/src/main.rs:899-957` — Generates the same zero-init + ikj accumulate nest in Wukong (optionally `@parallel`), C, `#pragma omp parallel for` C, and Rust raw-pointer form for a like-for-like comparison.
- **matmul_tn weight-gradient bench (C=Aᵀ·B)** — `crates/wukong_xbench/src/main.rs:965-1097` — Column-strided A nest that Wukong folds to `wukong_sgemm_tn`; DIVERGENCE: takes `roof` but discards it (`let _ = roof`), so no roofline percentage is printed here.
- **GEMV memory-bound bench** — `crates/wukong_xbench/src/main.rs:1106-1194` — Three M×N shapes reported as A-stream GB/s with a magnitude-normalized (not pointwise) tolerance because the 8-wide row dot reassociates.
- **α-scaled attention-score GEMM bench with α-overhead probe** — `crates/wukong_xbench/src/main.rs:1203-1319` — Times `(Q·Kᵀ)·0.125` against an unscaled twin to show the folded α costs ~1.0× i.e. is free in the GEMM writeback.
- **nn.Linear NT bench against oneMKL and C(omp)** — `crates/wukong_xbench/src/main.rs:1324-1496` — Square `C=A·Bᵀ` dot-product nest measured under the same two-group thermal order with MKL 1c/all, tuned crate, C, C(fast), C(omp) and Rust columns; rectangular generator at `crates/wukong_xbench/src/main.rs:1449-1462`.
- **Fused FFN bench `C = silu(A·Bᵀ)`** — `crates/wukong_xbench/src/main.rs:1515-1647` — Two adjacent Wukong nests fused into one `sgemm_nt_epi` versus C/Rust GEMM plus a second scalar-`expf` pass, with an explicit disclosure that most of the ratio is the serial-reduction GEMM gap.
- **bf16 mixed-precision nn.Linear bench** — `crates/wukong_xbench/src/main.rs:1666-1790` — bf16 operands widened `as f32` with f32 accumulate over the bf16 harness, checked only on `c[0]`; DIVERGENCE: the preceding doc paragraph describes the int8 GEMM bench, not this function.
- **Transpose bench with bit-exact gate** — `crates/wukong_xbench/src/main.rs:1801-1935` — Cache-blocked `dst=srcᵀ` vs naive C/Rust and an OpenMP twin at 1024²/2048², compared by exact buffer equality since a permutation cannot reassociate.
- **Strided column-reduction benches (colsum, colmean/sumsq/L2/RMS)** — `crates/wukong_xbench/src/main.rs:1944-2083` — Column-outer folds gcc/rustc leave scalar, with C(fast)/C(omp) peers and bit-exact checks because both fold i-ascending, also `crates/wukong_xbench/src/main.rs:3210-3358`.
- **Broadcast bias-add bench** — `crates/wukong_xbench/src/main.rs:2090-2187` — `out[r,c]=x[r,c]+bias[c]` at two shapes reported as GB/s with an exact full-buffer comparison against C.
- **Dequant bench family (1-D and per-channel, i32/i8)** — `crates/wukong_xbench/src/main.rs:2197-2351` — 8M-element past-L3 sizes exercising NT stores, integer-exact mismatch reporting and a 4-way ratio printer; NOTE: the C++ peer hardcodes `g++` instead of `$CXX`, and the unused middle pointer aliases the output buffer.
- **Column extreme benches (colmax / colmin / colmaxabs)** — `crates/wukong_xbench/src/main.rs:2362-2480` — Parametrized op code generates all three Wukong/C/Rust sources; bit-exact cross-check since max/min select an input value.
- **Row and column arg-reduction benches (i32 index output)** — `crates/wukong_xbench/src/main.rs:2487-2608` — Argmax/argmin over rows and strided columns, comparing indices exactly by reinterpreting the f32 harness slots as i32, also `crates/wukong_xbench/src/main.rs:2610-2732`.
- **Loop-carried ILP recurrence benches (lrscan, cumprod)** — `crates/wukong_xbench/src/main.rs:2803-2885` — SSM/Mamba `h=a·h+b` scan and prefix product with contraction-bounded inputs, isolating Wukong's 4-row-interleaved ILP lever against C's single serial chain, also `crates/wukong_xbench/src/main.rs:2924-3005`.
- **Prefix-scan benches (cumsum, cummax/cummin)** — `crates/wukong_xbench/src/main.rs:3011-3093` — Loop-carried scans gcc/rustc keep scalar; cumsum uses a magnitude-normalized tolerance (the Hillis-Steele tree reassociates) while cum-extremes are bit-exact, generators at `crates/wukong_xbench/src/main.rs:2737-2764` and `crates/wukong_xbench/src/main.rs:3098-3208`.
- **Training backward bench family (softmax_bwd, rmsnorm_bwd, layernorm_bwd)** — `crates/wukong_xbench/src/main.rs:3368-3456` — Per-row multi-reduction gradients; the two norm backwards use the 4-pointer harness and a hand-rolled C(fast) peer, and layernorm_bwd alone delegates printing to the shared `report_ratio`, also `crates/wukong_xbench/src/main.rs:3498-3695`.
- **Cross-entropy forward-loss bench** — `crates/wukong_xbench/src/main.rs:3704-3763` — Row-max + Σexp + label gather where i32 targets ride through the harness's `*const f32` slot, checked magnitude-normalized because the lse reductions reassociate.
- **RoPE forward/backward bench pair** — `crates/wukong_xbench/src/main.rs:3806-3915` — Generates Wukong/C/Rust rotary-embedding nests at two shapes, times serial+@parallel, and flags mismatch by max|Δ|/max|C|; backward twin at `crates/wukong_xbench/src/main.rs:3994-4067`.
- **RoPE rotation bench** — `crates/wukong_xbench/src/main.rs:3812-3870` — Inline per-element cos/sin rotation over `[rows, 2·half]` with the standard `10000^(-k/half)` schedule, forcing C/Rust to scalar libm while Wukong uses 8-wide sin/cos.
- **Gated-FFN activation bench (SwiGLU/GeGLU)** — `crates/wukong_xbench/src/main.rs:4069-4133` — Times `out=act(a)*b` at N=2^20 with C/Rust spelling Wukong's tanh-approx gelu constants explicitly so the vmath2 gate op is compared like-for-like.
- **Row-wise loss benches (xent_bwd, kldiv, entropy, kd_loss)** — `crates/wukong_xbench/src/main.rs:4135-4263` — Emits per-row log/exp reduction sources in three languages with a per-case byte basis and a relaxed-FP C(fast) peer; softmax-xent backward variant at `crates/wukong_xbench/src/main.rs:3917-3992`.
- **Shared ratio reporters (`report_ratio`, `report`)** — `crates/wukong_xbench/src/main.rs:4265-4316` — Prints the Wuk/Wuk-par/C/Rust (+optional C(fast)) GB/s table, magnitude-normalized cross-check and both ratios; the wider kernel-table reporter with C++/C(omp) columns is `crates/wukong_xbench/src/main.rs:5561-5665`.
- **Activation-backward sweep (silu/gelu/sigmoid/tanh/elu/softplus)** — `crates/wukong_xbench/src/main.rs:4318-4453` — Builds the `dx=dy·act'(x)` loop plus hand-written scalar-libm C/Rust derivatives, exercising the VM2_*_BWD 256-bit dispatch at 1e-3 normalized tolerance.
- **int8 GEMM bench with exact-integer equality gate** — `crates/wukong_xbench/src/main.rs:4455-4599` — Times u8×i8→i32 NT GEMM at 512/1024, prints GOP/s and compile ms, and demands bit-exact full-buffer equality across Wuk/par/C/Rust (integers admit no tolerance).
- **bf16 mixed-precision reduction bench + bit helpers** — `crates/wukong_xbench/src/main.rs:4601-4770` — Round-to-nearest-even bf16 packing/widening helpers feed dot/sum benches at 2^20 and 2^24 to expose the bandwidth win of half storage with f32 accumulate.
- **All-half axpby narrowing-store two-lever bench** — `crates/wukong_xbench/src/main.rs:4772-4936` — A/Bs Wukong bf16-out against Wukong f32-out (write-traffic halving) and against a C/Rust all-half peer, cross-checked at a bf16-scale 8e-3 tolerance.
- **conv2d im2col+GEMM vs direct-convolution bench** — `crates/wukong_xbench/src/main.rs:4938-5073` — Wukong builds a function-local `[Cin·K·K, OH·OW]` column matrix then a GEMM nest while C/Rust run the six-deep direct nest; the doc says "checksum" but the code does a full-buffer max_rel_err.
- **Fused single-row norm sweep (7 variants)** — `crates/wukong_xbench/src/main.rs:5075-5343` — Emits softmax/logsoftmax/layernorm/rmsnorm/l2norm plus the two affine forms in the exact spellings the recognizers accept, with an ns/call table and a C(fast) column.
- **Batched-norm sweep with @parallel row and C(fast) gating** — `crates/wukong_xbench/src/main.rs:5345-5559` — Runs `[tokens,hidden]` norms at an L3-resident and a >L3 shape, timing serial and @parallel Wukong against one shared single-threaded C/Rust/C(fast) baseline validated via `relaxed_peer_ok`.
- **f32 measurement harness (3-pointer and 4-pointer)** — `crates/wukong_xbench/src/main.rs:5667-5792` — Drives the real parse→sema→mir_build→optimize(3)→Cranelift-JIT pipeline (and a gcc/rustc shared-library twin with filename sanitizing) returning compile time, ns/call and an output snapshot; 4-pointer variants at `crates/wukong_xbench/src/main.rs:5794-5922`.
- **Typed-ABI harness family (i8 / bf16 / half-out)** — `crates/wukong_xbench/src/main.rs:5924-6298` — Six near-identical JIT/extern harnesses for the `(u8,i8,i32)`, `(u16,u16,f32)` and all-`u16` kernel ABIs; the half-out pair alone skips zeroing the output before timing.
- **oneMKL dlopen peer (discovery, ILP64 sgemm, VML)** — `crates/wukong_xbench/src/main.rs:6379-6585` — Locates `mkl_rt` via `WUKONG_MKL_DLL`/conda layouts, resolves `cblas_sgemm_64` + `MKL_Set_Num_Threads` (by-value, not the Fortran binding) and optional `vsExp/vsLn/vsTanh`, leaking the library for process lifetime.
- **MKL standing report with degenerate-all-core guard** — `crates/wukong_xbench/src/main.rs:6587-6646` — Prints 1-core and @parallel ratios plus a full-buffer check, and suppresses the all-core ratio entirely when MKL(all) fails to beat MKL(1c) by 1.2× — an honesty guard against a measurement artifact.
- **Tuned-library GEMM peer and %-of-roofline standing** — `crates/wukong_xbench/src/main.rs:6648-6684` — Times the pure-Rust `matrixmultiply` sgemm at matching strides (transposed B read by stride swap); the accompanying standing printer is `crates/wukong_xbench/src/main.rs:6356-6377`.
- **Measured AVX2-FMA roofline** — `crates/wukong_xbench/src/main.rs:6686-6750` — 12 independent FMA accumulator chains warmed ≥400 ms (so turbo ramp cannot make "% of roofline" exceed 100%) then best-of-8, giving the clock-invariant denominator for GEMM claims.
- **The `kernels()` catalogue and kernel source templates** — `crates/wukong_xbench/src/main.rs:6757-7292` — ~40 Kernel records (streaming, reductions, argmax, 20+ transcendentals, fusion, and eight `@parallel` twins) each carrying byte traffic, note, and Wukong/C/Rust bodies; the template builders are `crates/wukong_xbench/src/main.rs:7404-7448`.
- **Streaming elementwise at >L3 (N=2^24)** — `crates/wukong_xbench/src/main.rs:7294-7402` — Runs saxpy/residual/scale/relu at 64 MiB per array to expose the non-temporal-store dispatch, with a tighter 1e-4 mismatch threshold than the rest of the suite.
**`crates/wukong_xbench/src/model.rs`**

- **Model config, FLOP accounting and deterministic weight init** — `crates/wukong_xbench/src/model.rs:113-134` — `Cfg` plus a MAC-convention per-layer FLOP formula and an LCG fill giving identical GPT-2-scaled weights to every language column; scratch/weight allocation at `crates/wukong_xbench/src/model.rs:195-279`.
- **Competent C peer translation unit** — `crates/wukong_xbench/src/model.rs:450-561` — The same 12-layer block as one gcc TU with `linear_nt`, two-pass affine LayerNorm, per-head slice extraction and Wukong's exact tanh-GELU constants, exporting `kbench`/`kfinal`.
- **Model compile pipeline and recognized-kernel dispatch scan** — `crates/wukong_xbench/src/model.rs:634-650` — `kernel_calls` counts `wukong_*` symbols in optimized-MIR text so a silent regression to scalar or serial kernels is printed, not hidden; pipeline + MIR capture at `crates/wukong_xbench/src/model.rs:567-632`, warnings at `crates/wukong_xbench/src/model.rs:1809-1821` and `crates/wukong_xbench/src/model.rs:1920-1945`.
- **Block ABIs and the 12-layer forward drivers** — `crates/wukong_xbench/src/model.rs:705-806` — Two ping-pong layer loops (C's 24-pointer `BlockFn` with caller per-head scratch, Wukong's 19-pointer `WukBlockFn` without it) plus the final LayerNorm; ABI declarations at `crates/wukong_xbench/src/model.rs:136-193`.
- **Power-state disclosure line** — `crates/wukong_xbench/src/model.rs:1052-1095` — Calls Win32 `GetSystemPowerStatus` and prints AC/BATTERY/unknown in the bench header, but only distinguishes AC-line status — it does not detect the charging-vs-full cap state.
- **`bench_model` driver, env knobs and result table** — `crates/wukong_xbench/src/model.rs:1524-1600` — Sweeps S=128/512 (overridable by `XBENCH_MODEL_S`); the column build, `XBENCH_MODEL_WUK_ONLY`/`TORCH_ONLY`/`NAIVE` gating, 9-column table, ratio lines and magnitude-normalized cross-checks are `crates/wukong_xbench/src/model.rs:1767-2227`.
**`tools/bench_gpt2_torch.py`**

- **PyTorch CPU GPT-2 peer with replayed token ids** — `tools/bench_gpt2_torch.py:56-104` — Times eager 1-thread, eager all-thread and compiled HF `GPT2Model` on the same synthetic id formula, then cross-checks the Wukong last-row logits for MATCH/DIVERGE.
**`tools/measure_gpt2.ps1`**

- **Power-gated repeatable measurement sweep** — `tools/measure_gpt2.ps1:41-86` — Classifies battery/charging/full-AC before and after each regime, takes the worse sample, and refuses to label all-core numbers reportable unless AC survived the heavy run, also `tools/measure_gpt2.ps1:120-167`.

## G42 — Benchmarks: compile-time measurement, profiling & dev perf probes

*Files:* `bench/kernels/matmul.wk`, `crates/wukong_bench/src/compile_time.rs`, `crates/wukong_bench/src/compile_vs.rs`, `crates/wukong_bench/src/main.rs`, `crates/wukong_bench/src/profile.rs`, `crates/wukong_codegen_cranelift/src/lib.rs`, `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_opt/src/lib.rs`, `crates/wukong_runtime/examples/gemm_scaling.rs`, `crates/wukong_runtime/examples/gemm_var.rs`, `crates/wukong_runtime/src/lowp.rs`, `crates/wukong_runtime/src/vmath.rs`, `crates/wukong_xbench/src/main.rs`, `crates/wukong_xbench/src/model.rs`

**`bench/kernels/matmul.wk`**

- **Compile-time and optimizer benchmark kernels** — `bench/kernels/matmul.wk:1-49` — Five scalar workloads (matmul for LICM, sum_to_n for mem2reg, primes and collatz for branch-heavy integer code, fib_rec for calls) driven by `wukong_bench` compile-time modes, also `bench/kernels/sum_to_n.wk:1-15`.
**`crates/wukong_bench/src/compile_time.rs`**

- **Compile-time per-pass attribution with an optimize_timed self-check** — `crates/wukong_bench/src/compile_time.rs:31-186` — Best-of-N front-end and optimizer timing plus a per-pass table, gated by requiring `optimize_timed` to print byte-identical MIR to `optimize`.
**`crates/wukong_bench/src/compile_vs.rs`**

- **`compile-vs` cross-compiler compile-to-object comparison** — `crates/wukong_bench/src/compile_vs.rs:47-357` — Times wukongc against gcc/g++/rustc same-run on three equivalent kernels, reporting clock-invariant ratios; C/C++ sources are deliberately header-free bare TUs for fairness.
**`crates/wukong_bench/src/main.rs`**

- **Optimizer-effectiveness bench doubling as a dual correctness gate** — `crates/wukong_bench/src/main.rs:90-264` — Reports MIR op reduction and interp-vs-native speedup while failing the process if -O0≠-O3 stdout/exit or Cranelift disagrees with the interpreter at -O3.
- **Adaptive batched timing with geometric-mean aggregation** — `crates/wukong_bench/src/main.rs:268-316` — Warms up then doubles reps until 50 ms elapses (cap 2^22) for both JIT and interpreter runs, so sub-microsecond programs are still measured stably.
**`crates/wukong_bench/src/profile.rs`**

- **`compile-profile` six-stage pipeline breakdown with backend split** — `crates/wukong_bench/src/profile.rs:107-311` — Times lex/parse/sema/mir_build/optimize/codegen in isolation by rebuilding each stage's input untimed, and splits the backend via `emit_object_timed` into codegen versus object-write.
- **`spawn-overhead` in-process-versus-process-spawn characterization** — `crates/wukong_bench/src/profile.rs:324-508` — Reports cold-start and warm best-of-N for in-memory object emission versus spawning `wukongc.exe`, deriving a spawn tax and isolating the rustc link cost of `--emit=exe`.
**`crates/wukong_codegen_cranelift/src/lib.rs`**

- **`ObjTimings` codegen-vs-object-write split** — `crates/wukong_codegen_cranelift/src/lib.rs:5065-5075` — Separates Cranelift isel/regalloc/emit time from COFF/ELF container serialization so `compile-profile` can attribute backend cost precisely.
**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Ignored backend compile-time A/B harness** — `crates/wukong_codegen_cranelift/src/tests.rs:4964-5158` — Walks tests/run, examples and bench/kernels, re-asserts byte identity per file, then reports geomean verifier and parallel-codegen ratios plus the codegen-vs-object-write split and ISA fixed cost, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:4922-4940`, `crates/wukong_codegen_cranelift/src/tests.rs:4944-4962`.
**`crates/wukong_opt/src/lib.rs`**

- **In-process optimizer timing harness** — `crates/wukong_opt/src/lib.rs:57-95` — `Timings`/`PassStat` accumulate per-pass wall time, call counts, inline time and max fixpoint iterations; `optimize_timed` wraps the identical pipeline at `crates/wukong_opt/src/lib.rs:225-239`.
**`crates/wukong_runtime/examples/gemm_scaling.rs`**

- **Serial-vs-parallel GEMM scaling probe** — `crates/wukong_runtime/examples/gemm_scaling.rs:1-98` — Dev instrument timing `wukong_sgemm_nt` against `wukong_sgemm_nt_parallel` at transformer GEMM shapes, printing GFLOP/s and parallel speedup in seconds instead of a full xbench run.
- **Adjacent-interleaved A/B timing discipline** — `crates/wukong_runtime/examples/gemm_scaling.rs:24-46` — `time_pair` measures serial and parallel adjacently each round so the laptop's thermal drift cancels in the per-round ratio, returning best-of plus median scaling.
**`crates/wukong_runtime/examples/gemm_var.rs`**

- **Mid-size GEMM run-variance probe (ABBA discipline)** — `crates/wukong_runtime/examples/gemm_var.rs:1-257` — Alternates Wukong-parallel and MKL-all batches per round with the order swapped every round, printing every round rather than a best-of, to attribute mid-size variance to machine state vs scheduling.
- **MKL peer loader with ILP64 by-value CBLAS binding** — `crates/wukong_runtime/examples/gemm_var.rs:22-90` — Resolves `mkl_rt.dll` via `WUKONG_MKL_DLL` or search, binding `cblas_sgemm_64`, `MKL_Set_Num_Threads` and `MKL_Get_Max_Threads`; the lowercase Fortran by-reference twin segfaults and is deliberately avoided.
- **Median and coefficient-of-variation statistics for run-variance reporting** — `crates/wukong_runtime/examples/gemm_var.rs:91-102` — Small helpers backing the probe's per-round spread output, the instrument used to root-cause the OS-straggler variance episodes on the worker pool.
**`crates/wukong_runtime/src/lowp.rs`**

- **Low-precision bandwidth benchmarks** — `crates/wukong_runtime/src/lowp.rs:1363-1520` — `#[ignore]`d best-of-5 harnesses comparing bf16/f16 sum and bf16-in axpby against autovectorized Rust and the tuned f32 kernels, reporting GB/s and speedup.
**`crates/wukong_runtime/src/vmath.rs`**

- **In-process throughput benches versus scalar libm** — `crates/wukong_runtime/src/vmath.rs:2802-2866` — ignored best-of-200 timing of eight ops against unvectorizable libm-call loops on an L2-resident working set; two-input version at `crates/wukong_runtime/src/vmath.rs:2874-2927`.
**`crates/wukong_xbench/src/main.rs`**

- **Run summary: geomean runtime/compile ratios, roofline and MKL banner** — `crates/wukong_xbench/src/main.rs:482-521` — Prints geomean Wukong-vs-C runtime and compile-time ratios, the measured single-core AVX2-FMA roofline used as GEMM's clock-invariant denominator, and MKL availability.
- **Adaptive best-of-N timer `time_ns`** — `crates/wukong_xbench/src/main.rs:6300-6354` — Warms 5 calls, doubles reps until a 50 ms block, then reports the fastest of 14 blocks; calls already exceeding 50 ms fall back to best-of-6 single calls.
**`crates/wukong_xbench/src/model.rs`**

- **Model timing instruments: big-stack worker and best-of-N forward timer** — `crates/wukong_xbench/src/model.rs:828-855` — `time_forward` warms once then samples to 4 s / 8 samples returning the fastest; `on_big_stack` runs columns on a 64 MiB-stack scoped thread spawned outside the timed region (`crates/wukong_xbench/src/model.rs:808-826`).

## G43 — Test corpus: run / fail / imports differential and opt-invariance gates

*Files:* `crates/wukong_codegen_cranelift/src/fuzz.rs`, `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs`, `crates/wukong_codegen_cranelift/src/tests.rs`, `crates/wukong_mir/src/verify.rs`, `crates/wukong_mir_build/src/lib.rs`, `crates/wukong_opt/src/lib.rs`, `crates/wukong_parser/src/lib.rs`, `crates/wukong_runtime/src/attention.rs`, `crates/wukong_runtime/src/bias.rs`, `crates/wukong_runtime/src/colarg.rs`, `crates/wukong_runtime/src/colreduce.rs`, `crates/wukong_runtime/src/cumsum.rs`, `crates/wukong_runtime/src/embedding.rs`, `crates/wukong_runtime/src/entropy.rs`, `crates/wukong_runtime/src/gemm.rs`, `crates/wukong_runtime/src/gemv.rs`, `crates/wukong_runtime/src/gevm.rs`, `crates/wukong_runtime/src/kd_loss.rs`, `crates/wukong_runtime/src/layernorm_bwd.rs`, `crates/wukong_runtime/src/lib.rs`, `crates/wukong_runtime/src/logsoftmax.rs`, `crates/wukong_runtime/src/norm.rs`, `crates/wukong_runtime/src/pool2d.rs`, `crates/wukong_runtime/src/reduce.rs`, `crates/wukong_runtime/src/rope.rs`, `crates/wukong_runtime/src/rope_bwd.rs`, `crates/wukong_runtime/src/scatter.rs`, `crates/wukong_runtime/src/softmax_bwd.rs`, `crates/wukong_runtime/src/transpose.rs`, `crates/wukong_runtime/src/velem.rs`, `crates/wukong_sema/src/lib.rs`, `crates/wukong_types/src/lib.rs`, `crates/wukong_xbench/src/model.rs`, `crates/wukongc/tests/determinism.rs`, `crates/wukongc/tests/emit.rs`, `crates/wukongc/tests/exe.rs`, `crates/wukongc/tests/fail.rs`, `crates/wukongc/tests/run.rs`, `tests/fail/deep_nesting.wk`, `tests/fail/generic_return_shape_lie.wk`, `tests/fail/lib/badlib.wk`, `tests/fail/matmul_dim.wk`, `tests/fail/rank.wk`, `tests/imports/diamond_root.wk`, `tests/run/axpby_half_out.wk`, `tests/run/colsum.wk`, `tests/run/decode_attn.wk`, `tests/run/exit_code.wk`, `tests/run/generic_shape_runtime.wk`, `tests/run/heap_alloc.wk`, `tests/run/i8_linear_dequant.wk`, `tests/run/import_multi.wk`, `tests/run/io_roundtrip_f32.wk`, `tests/run/linear_bias_gelu.wk`, `tests/run/matmul_f32.wk`, `tests/run/mut_param.wk`, `tests/run/now_ns_monotonic.wk`, `tests/run/parallel_head_loop.wk`, `tests/run/parallel_reduce.wk`, `tests/run/parallel_region_decline.wk`, `tests/run/rope.wk`, `tests/run/softmax_fused.wk`, `tests/run/transcendental_sweep.wk`, `tests/run/vec256_general.wk`, `tests/run/vmath2_dispatch.wk`, `tests/run/vmath_dispatch.wk`, `tests/run/xent_bwd.wk`

**`crates/wukong_codegen_cranelift/src/fuzz_grammar.rs`**

- **Grammar fuzzer generator state and lexical scope snapshot/restore** — `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:24-163` — Tracks scalars, arrays, struct/nested-struct/enum/tuple locals, protected loop counters and indentation, with a six-field `Marks` snapshot restored on block exit to keep names in scope.
- **Well-typed expression generator respecting every deliberate sema rejection** — `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:217-389` — Emits depth-bounded arithmetic, bitwise, shifts, casts, if-values and comparisons that are same-type only, divides only by positive literals, and reads arrays/aggregates only at provably in-bounds indices; also `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:166-214`.
- **Statement generator: 18 weighted forms targeting known miscompile classes** — `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:391-717` — Produces lets, cast-pinned assigns, array stores, depth-limited if/for/while, self-referential struct-literal assignment, nested `Box2` writes, exhaustive enum value-match, tuple destructuring, and by-ref/sret helper calls.
- **Whole-program assembly with sret and by-ref-aggregate helper functions** — `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:744-803` — Wraps generated bodies in shared `Pt`/`Box2`/`Col` types, `h0(mut p: Pt, k) -> f32` and `h1(a,b) -> Pt`, guarantees ≥3 prints, and returns a masked exit code; also `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:719-741`.
- **Five-way agreement matrix: interp O0/O2/O3 vs native O0/O3 plus MIR verification** — `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:805-844` — Compiles each seeded program through the real pipeline, verifies MIR at every level, and fails loudly with seed and full source on any exit-code/stdout divergence or compile error; also `crates/wukong_codegen_cranelift/src/fuzz_grammar.rs:858-929`.
**`crates/wukong_codegen_cranelift/src/fuzz.rs`**

- **Deterministic SplitMix64 PRNG and adversarial input regimes** — `crates/wukong_codegen_cranelift/src/fuzz.rs:19-66` — Dependency-free seeded generator drawing Normal, Positive, or Adversarial values (±0, ±1, ±Inf, NaN, denormals, 1e30) so any fuzz failure reproduces exactly on any machine.
- **Result-equivalence relation `same()` used by the full-buffer fuzzer** — `crates/wukong_codegen_cranelift/src/fuzz.rs:68-73` — Treats any two NaNs and any two zeros as equal, so the comparison is weaker than the module doc's claimed "bit-for-bit identical" bar (NaN payload and ±0 sign divergences pass).
- **Fuzz kernel corpus: ~45 source templates across elementwise, activations, reductions, fused norms, GEMM, tensors** — `crates/wukong_codegen_cranelift/src/fuzz.rs:77-532` — Each kernel is a parameterized Wukong source plus buffer-length and legal-regime metadata, deliberately including batched/affine norm forms that exercise the rows>1 marshalling path.
- **Full-buffer f32 interp-vs-native differential driver** — `crates/wukong_codegen_cranelift/src/fuzz.rs:536-609` — Runs every kernel at sizes straddling SIMD remainder boundaries, at -O0 and -O3, over all its regimes, comparing entire output buffers, with a >1000-run floor guarding against an emptied corpus.
- **int8 u8×i8→i32 GEMM full-buffer differential fuzzer** — `crates/wukong_codegen_cranelift/src/fuzz.rs:611-676` — Sizes hit the int8 microkernel's 16/32/48 K-chunk boundaries and odd tails; integer exactness makes plain `assert_eq!` on the whole C buffer the correct no-tolerance bar.
- **vmath f64-reference value oracle with per-function relative-error budgets** — `crates/wukong_codegen_cranelift/src/fuzz.rs:688-821` — Checks 36 transcendental/activation kernels against independent f64 references (1e-6 direct polys, 1e-4 exp-composed, 1e-2/1e-3 ill-conditioned), catching bugs both backends share; tan/tanhshrink/elu excluded by construction.
**`crates/wukong_codegen_cranelift/src/tests.rs`**

- **Differential run harness (jit / interp / jit_ok / compile_native)** — `crates/wukong_codegen_cranelift/src/tests.rs:6-29` — Parses, sema-checks, lowers and optimizes a source string then runs it natively or in the interpreter, returning (exit code, stdout) for every oracle in the file, extra site `crates/wukong_codegen_cranelift/src/tests.rs:189-203`.
- **Core smoke + interp-vs-native opt-invariance sweep** — `crates/wukong_codegen_cranelift/src/tests.rs:454-535` — Baseline JIT value checks (loops, recursion, print, casts, signed modulo, failing assert) plus the soundness sweep asserting native == interp at -O0/1/2/3.
- **Heap alloc/slice differential (`alloc_*`/`free`, fat pointers, DSE conservatism)** — `crates/wukong_codegen_cranelift/src/tests.rs:545-578` — Pins zero-init reads, cross-function `mut []T` mutation, negative-length clamp, len>127 re-binding and store-before-free against the interpreter at all four opt levels.
- **Aggregate memory-model and ABI differential family** — `crates/wukong_codegen_cranelift/src/tests.rs:584-705` — Tuples/structs/nested structs/arrays-of-aggregate plus by-value params, whole-struct assignment, sret returns and explicit-deref field places must agree byte-for-byte with the slot-indexed interpreter, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:881-989`, `crates/wukong_codegen_cranelift/src/tests.rs:1713-1786`.
- **Control-flow differential family (short-circuit, continue, labels, deep recursion)** — `crates/wukong_codegen_cranelift/src/tests.rs:782-874` — Uses captured stdout as side-effect evidence that `&&`/`||` skip their RHS, that `continue` runs the for-step, and that labeled break/continue target the outer loop, extra site `crates/wukong_codegen_cranelift/src/tests.rs:713-722`.
- **`match` semantics gate family (literals, guards, or/range/char/tuple patterns, exhaustiveness)** — `crates/wukong_codegen_cranelift/src/tests.rs:996-1030` — Covers if-else-chain lowering, discriminant tests, binding catch-alls and the `Unreachable` fallthrough of an exhaustive match, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:1264-1327`, `crates/wukong_codegen_cranelift/src/tests.rs:1443-1469`, `crates/wukong_codegen_cranelift/src/tests.rs:1793-1823`.
- **Literal / const / enum / char front-end value gates** — `crates/wukong_codegen_cranelift/src/tests.rs:1036-1058` — Pins radix literals, i32→i64 and u64-range default widening (asserted through stdout because both backends shared the wrong constant), aggregate-literal adaptation, const array lengths, enum discriminants and char literals, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:1532-1563`, `crates/wukong_codegen_cranelift/src/tests.rs:1691-1711`, `crates/wukong_codegen_cranelift/src/tests.rs:1605-1629`, `crates/wukong_codegen_cranelift/src/tests.rs:1828-1852`.
- **Cast, coercion and float-precision gates** — `crates/wukong_codegen_cranelift/src/tests.rs:1065-1090` — Float→narrow-int clamping, single-step int→f32 rounding, fp→int saturation and signedness, f32-precision const folding, `FRem`, return-type/mixed-branch coercion and int-operand math intrinsics, extra sites `crates/wukong_codegen_cranelift/src/tests.rs:1962-1982`, `crates/wukong_codegen_cranelift/src/tests.rs:4309-4336`, `crates/wukong_codegen_cranelift/src/tests.rs:1932-1956`, `crates/wukong_codegen_cranelift/src/tests.rs:1892-1926`, `crates/wukong_codegen_cranelift/src/tests.rs:1200-1257`.
- **`Tensor` 1-D param and wide loop-bound kernel gates** — `crates/wukong_codegen_cranelift/src/tests.rs:1335-1361` — Pins that recognizers load the base pointer out of a Tensor param slot rather than GEPing the slot address, and that `for i in 0..n` widens the counter to an i64/usize end bound, extra site `crates/wukong_codegen_cranelift/src/tests.rs:1369-1393`.
- **Stdout-observable gates: string literals and unsigned print** — `crates/wukong_codegen_cranelift/src/tests.rs:1860-1885` — Compares whole stdout buffers so `rt_print_str` escapes/UTF-8 and the `print_u` magnitude path (a high-bit u32/u64 must not print negative) match the interpreter, extra site `crates/wukong_codegen_cranelift/src/tests.rs:729-748`.
- **Dispatch-proof oracles: `lowered` + `lowered_calls` + MIR text search** — `crates/wukong_codegen_cranelift/src/tests.rs:3865-3881` — Lowers a program and asks whether any function emits a named runtime `Op::Call`, so recognizer tests prove kernel dispatch instead of a correct scalar fallback, extra site `crates/wukong_codegen_cranelift/src/tests.rs:3032-3041`.
**`crates/wukong_mir_build/src/lib.rs`**

- **In-crate lowering and consistency test suite** — `crates/wukong_mir_build/src/lib.rs:25112-25716` — Parallel-region outlining and five decline cases, plus emitter-side exp/log table cross-checks that catch dispatched-versus-inlined drift the differential gate cannot see.
**`crates/wukong_mir/src/verify.rs`**

- **Verifier regression tests** — `crates/wukong_mir/src/verify.rs:414-472` — Four Builder-driven unit tests pinning the valid-function baseline, i32/i64 operand mismatch detection, the void-`Call` exemption, and block-parameter arity mismatch on a branch.
**`crates/wukong_opt/src/lib.rs`**

- **Optimizer differential and structural test suite** — `crates/wukong_opt/src/lib.rs:393-769` — 13 in-crate tests gate -O0 vs -O2 result equality through the interpreter plus structural claims (all scalar allocas promoted, loop phi created, arrays stay in memory, CSE drops a mul, DSE drops a store, LICM hoists once, recursion not inlined), each re-verifying MIR.
**`crates/wukong_parser/src/lib.rs`**

- **Parser unit-test corpus and stack-overflow gates** — `crates/wukong_parser/src/lib.rs:1745-1982` — Precedence/cast/turbofish/type-rendering assertions, item-recovery test, and three depth tests spawning 64 MB stacks to prove pathological nesting yields exactly one E0209; item-level tests at `crates/wukong_parser/src/items.rs:404-449`.
**`crates/wukong_runtime/src/attention.rs`**

- **Attention naive-softmax differential oracle** — `crates/wukong_runtime/src/attention.rs:218-318` — Textbook materialized max-subtract softmax reference checked at 1e-4 relative tolerance over eight shapes straddling the 8-lane `D` remainder and both mask modes.
**`crates/wukong_runtime/src/bias.rs`**

- **Bias gates: lane/tail bit-equality, parallel identity, exact f64 add** — `crates/wukong_runtime/src/bias.rs:222-323` — Compares raw bit patterns across four activations, non-multiple-of-8 tails, an above-NT-threshold size, 523×97 band splits, and a single-rounding f64 reference.
**`crates/wukong_runtime/src/colarg.rs`**

- **Arg-reduction tie-break and lane-collapse test corpus** — `crates/wukong_runtime/src/colarg.rs:296-462` — Plants duplicate extrema across lanes, stripes, and tails to pin lowest-index semantics and scalar==AVX2; the row sibling is `crates/wukong_runtime/src/rowarg.rs:231-397`.
**`crates/wukong_runtime/src/colreduce.rs`**

- **Column-reduction naive oracle test** — `crates/wukong_runtime/src/colreduce.rs:566-674` — Asserts exact vector equality against an independent strided fold and serial==parallel over shapes straddling the 8-lane edge and the 256-column threshold.
**`crates/wukong_runtime/src/cumsum.rs`**

- **cumsum reassociated-reduction tolerance oracle** — `crates/wukong_runtime/src/cumsum.rs:187-315` — Holds the tree-scan to 1e-4 relative vs scalar and f64 references (not bit-exact), plus exact serial-vs-parallel and a focused cross-lane shift/zero-fill test.
**`crates/wukong_runtime/src/embedding.rs`**

- **Embedding gates: naive-gather equality, OOB rows, degenerate shapes** — `crates/wukong_runtime/src/embedding.rs:208-318` — Bit-exact comparison against a naive gather over nine shapes straddling the 8-lane and parallel thresholds, plus negative/over-large ids and zero-size no-ops.
**`crates/wukong_runtime/src/entropy.rs`**

- **Four-part differential oracle template per kernel** — `crates/wukong_runtime/src/entropy.rs:182-321` — Every file repeats scalar-vs-AVX2 bit equality, an independent f64 reference at 1e-5 relative tolerance, serial-vs-parallel bit equality, and a degenerate-shape no-op test; also `crates/wukong_runtime/src/kd_loss.rs:264-488`.
**`crates/wukong_runtime/src/gemm.rs`**

- **Naive-reference tolerance oracles for nn/nt/tn/lowp/epilogue/alpha** — `crates/wukong_runtime/src/gemm.rs:3035-3136` — Independent triple-loop references with √k-scaled tolerance covering ragged dims and beta accumulation; nt/tn variants at `crates/wukong_runtime/src/gemm.rs:3326-3475`, epilogue and alpha at `crates/wukong_runtime/src/gemm.rs:4350-4453`.
- **Serial-equals-parallel bit-exactness gate suite** — `crates/wukong_runtime/src/gemm.rs:3615-3849` — Byte-equality tests for every parallel shape across bt/beta/thread-count/steal-order/shared-A combinations, the invariant that lets the interpreter oracle call the serial kernel; also `crates/wukong_runtime/src/gemm.rs:3971-4289` and `crates/wukong_runtime/src/gemm.rs:4460-4489`.
**`crates/wukong_runtime/src/gemv.rs`**

- **GEMV correctness gates plus ignored bandwidth probe** — `crates/wukong_runtime/src/gemv.rs:244-396` — Tests an independent f64 reference, exact `alpha·plain` equality, byte-identity at `alpha==1`, serial==parallel, and an `--ignored` GB/s roofline probe over three shapes.
**`crates/wukong_runtime/src/gevm.rs`**

- **GEVM dual oracles: exact fma reference and independent f64 reference** — `crates/wukong_runtime/src/gevm.rs:221-335` — Asserts literal `assert_eq!` equality against a scalar `mul_add` fold (no tolerance) plus a tolerance-based f64 check, explicitly noting the recognizer path is gate-blind.
**`crates/wukong_runtime/src/kd_loss.rs`**

- **Closed-form analytic property gates** — `crates/wukong_runtime/src/kd_loss.rs:407-451` — One-hot KD equals hard-label xent, all-equal logits give log(C), CE-backward rows sum to zero, KL(p‖p) is zero, uniform entropy is log(C); also `crates/wukong_runtime/src/xent_bwd.rs:380-406`, `crates/wukong_runtime/src/kldiv.rs:299-317`, `crates/wukong_runtime/src/entropy.rs:291-306`.
**`crates/wukong_runtime/src/layernorm_bwd.rs`**

- **Norm-backward self-oracle circularity** — `crates/wukong_runtime/src/layernorm_bwd.rs:419-444` — The "reference" recomputes through the kernel's own scalar twins, so it pins AVX2==scalar but cannot catch a wrong formula; only the f64 test at `crates/wukong_runtime/src/layernorm_bwd.rs:557-608` does, also `crates/wukong_runtime/src/rmsnorm_bwd.rs:325-356`.
**`crates/wukong_runtime/src/lib.rs`**

- **Core runtime unit gates: alloc, arena, bf16, parallel-for** — `crates/wukong_runtime/src/lib.rs:669-806` — Pins clock monotonicity, alloc zeroing/alignment/degenerate-null, bf16 RNE and specials, arena alignment/exhaustion/reset, and that `wukong_parallel_for` covers every index exactly once.
- **File-I/O differential gates** — `crates/wukong_runtime/src/lib.rs:808-994` — Round-trips all six element types and pins the on-disk little-endian byte order, trailing-partial-element drop, `min(len,avail)`, untouched buffers on failure, and empty-write truncation.
**`crates/wukong_runtime/src/logsoftmax.rs`**

- **Log-softmax correctness battery** — `crates/wukong_runtime/src/logsoftmax.rs:340-527` — Scalar==AVX2 bit equality, f64 reference plus Σexp==1 and `logsoftmax == x − logsumexp` cross-checks, serial==parallel above the threshold, in-place equality, and degenerate-shape no-op.
**`crates/wukong_runtime/src/norm.rs`**

- **Norm bit-exactness twin gates** — `crates/wukong_runtime/src/norm.rs:907-1025` — Asserts scalar==AVX2, serial==parallel and in-place==out-of-place to raw bits across tail sizes 1..257 for all five ops; affine variants at `crates/wukong_runtime/src/norm.rs:1147-1255`.
- **Norm f64-reference oracles** — `crates/wukong_runtime/src/norm.rs:1027-1135` — Independent f64 recomputation of all five norms plus the softmax-sums-to-1 and exp(logsoftmax)==softmax invariants; affine oracle and the gamma=1/beta=0 anti-drift check at `crates/wukong_runtime/src/norm.rs:1257-1358`.
**`crates/wukong_runtime/src/pool2d.rs`**

- **Pooling exact-equality gates including a stride-2 AVX2 sweep** — `crates/wukong_runtime/src/pool2d.rs:537-793` — Asserts literal `Vec` equality against a naive nest over nine ragged configs, then drives `pool_channel_avx2_sw2` directly across six widths, three windows and two vertical strides.
**`crates/wukong_runtime/src/reduce.rs`**

- **Reduction correctness gates** — `crates/wukong_runtime/src/reduce.rs:531-722` — Four unit tests pin serial==parallel bit-for-bit, scalar==AVX2, an f64 tolerance reference, and lowest-index argmax across chunk boundaries; no ±∞ inputs are covered.
**`crates/wukong_runtime/src/rope_bwd.rs`**

- **RoPE backward oracle suite including forward-backward round-trip** — `crates/wukong_runtime/src/rope_bwd.rs:207-476` — Adds to the forward's checks a `rope_bwd(rope_fwd(x)) ≈ x` round-trip that pins the gradient as the true transpose rotation, not merely self-consistent.
**`crates/wukong_runtime/src/rope.rs`**

- **RoPE six-test property suite** — `crates/wukong_runtime/src/rope.rs:200-428` — Bit-exact scalar-vs-AVX2, bit-exact serial-vs-parallel, an independent f64 `sin_cos` reference, pair-norm preservation, in-place aliasing equality, and degenerate-shape no-op checks.
**`crates/wukong_runtime/src/scatter.rs`**

- **scatter-add collision and determinism test battery** — `crates/wukong_runtime/src/scatter.rs:217-397` — Naive reference plus heavy-collision id patterns, out-of-range skips, and degenerate shapes, all asserting exact serial==parallel bits on the real multicore path.
**`crates/wukong_runtime/src/softmax_bwd.rs`**

- **Softmax-backward dual oracles** — `crates/wukong_runtime/src/softmax_bwd.rs:143-261` — A self-oracle pinning AVX2-apply==scalar and serial==parallel, plus a genuinely external f64 reference on true row-normalized softmax inputs because the kernel is differential-gate-blind.
**`crates/wukong_runtime/src/transpose.rs`**

- **Transpose gates against the naive nested-loop permutation** — `crates/wukong_runtime/src/transpose.rs:152-226` — Bit-exact f32 and u16 comparisons over nine shapes straddling the 32-block edge, thin/square cases, and a size that trips the multicore path.
**`crates/wukong_runtime/src/velem.rs`**

- **velem/vhorner bit-exact oracle suite** — `crates/wukong_runtime/src/velem.rs:574-793` — Tail-vs-lane, saxpy-is-FMA, null-y, div, and serial-vs-parallel bit-equality tests across NT boundaries; the header cites a test `velem_scalar_matches_avx` that does not exist.
**`crates/wukong_sema/src/lib.rs`**

- **In-crate sema regression suite** — `crates/wukong_sema/src/lib.rs:3603-4099` — Thirty-odd tests pinning each error class (E0301/E0401/E0405/E0501/E0502/E0503) plus explicit no-false-positive corpora and a determinism test for generic parameter order.
**`crates/wukong_types/src/lib.rs`**

- **Type-layout unit gate** — `crates/wukong_types/src/lib.rs:335-418` — Six in-crate tests pinning scalar sizes, vector/array/nested-array sizes, tuple padding, `Scalar::from_name` round-trips and tensor display formatting.
**`crates/wukong_xbench/src/model.rs`**

- **Interpreter/native/@parallel differential bit-exactness gate** — `crates/wukong_xbench/src/model.rs:1602-1765` — Runs the identical reduced-config 12-layer forward through the interpreter, the JIT, and the `@parallel` JIT and requires max-rel-err exactly 0.0, returning the worst value for the bench header to flag.
- **Model unit tests: dispatch pinning and gate bit-exactness** — `crates/wukong_xbench/src/model.rs:2229-2318` — Asserts the serial block's six recognized kernels with no parallel region, the `@parallel` block's outlined region containing only serial per-head kernels, and that `interp_gate()` returns exactly `Some(0.0)`.
**`crates/wukongc/tests/determinism.rs`**

- **Cross-process MIR determinism gate** — `crates/wukongc/tests/determinism.rs:39-76` — Spawns three fresh processes per program (three random hash seeds) and requires byte-identical `--emit=mir -O2`, catching HashMap iteration order leaking into output.
**`crates/wukongc/tests/emit.rs`**

- **Emit-stage and example-program smoke gates** — `crates/wukongc/tests/emit.rs:43-87` — Front-end stages must succeed on all sources and mir-high/mir/llvm-ir on the run suite at -O0 and -O2 with no verifier ICE; also `crates/wukongc/tests/examples.rs:51-97`.
**`crates/wukongc/tests/exe.rs`**

- **Linked-executable AOT differential gate with clean skip** — `crates/wukongc/tests/exe.rs:25-107` — Links four curated fixtures via `--emit=exe` and compares stdout/exit to `--run`, reporting and skipping rather than failing when no linker toolchain exists.
**`crates/wukongc/tests/fail.rs`**

- **Compile-fail suite and imported-file span provenance test** — `crates/wukongc/tests/fail.rs:28-91` — Drives `--error-format=json --emit=mir` asserting each fixture's `EXPECT-CODE` appears, plus that an E0401 inside an imported file names `badlib.wk` line 6.
**`crates/wukongc/tests/run.rs`**

- **`tests/run` directive harness with per-run scratch directories** — `crates/wukongc/tests/run.rs:25-140` — Parses `// RUN:`/`// EXPECT-EXIT:`/`// EXPECT-OUT:` from each fixture and isolates every child process in its own temp CWD so relative-path file-I/O fixtures cannot collide.
- **Opt-invariance and native-equals-interpreter differential gates** — `crates/wukongc/tests/run.rs:144-236` — Three suites assert -O0 matches -O1/-O2/-O3 on interp, native matches interp at -O0/-O2, and native is itself opt-invariant, collecting all mismatches.
**`tests/fail/deep_nesting.wk`**

- **Parser robustness / stack-exhaustion fixtures** — `tests/fail/deep_nesting.wk:1-9` — 2000 nested parentheses must yield E0209 rather than a stack overflow that once crashed the compiler with exit 127 and no diagnostic.
**`tests/fail/generic_return_shape_lie.wk`**

- **Backend-divergence-prevention compile-fail fixtures** — `tests/fail/generic_return_shape_lie.wk:1-21` — Rejects a generic return-shape lie and an empty match that previously reached runtime, where the interpreter trapped but native read past the buffer or executed an illegal instruction, also `tests/fail/match_empty.wk:1-12`.
**`tests/fail/lib/badlib.wk`**

- **Imported-file diagnostic provenance fixture** — `tests/fail/lib/badlib.wk:1-8` — A deliberately ill-typed library whose E0401 must render with its own path and line 6, pinning per-`SourceId` span rendering in multi-file programs.
**`tests/fail/matmul_dim.wk`**

- **Compile-fail EXPECT-CODE protocol and error-code distribution** — `tests/fail/matmul_dim.wk:1-13` — 109 fixtures each pin one stable diagnostic code; the distribution is heavily skewed (56 E0401 type errors) leaving many catalogue codes with a single fixture.
**`tests/fail/rank.wk`**

- **Shape-checker compile-fail family (E0501-E0504)** — `tests/fail/rank.wk:1-11` — Twenty-nine fixtures pin rank mismatch, dimension mismatch, tensor-decay and unknown-dim rejection — the headline compile-time guarantee, also `tests/fail/matmul_dim.wk:1-13`.
**`tests/imports/diamond_root.wk`**

- **Loader unit-test fixtures: import cycle and diamond** — `tests/imports/diamond_root.wk:1-10` — Six files exercising A↔B cycles, a self-import no-op, and a diamond whose shared leaf must be spliced exactly once by canonical path, also `tests/imports/cycle_a.wk:1-11`.
**`tests/run/axpby_half_out.wk`**

- **Low-precision bf16 / f16 cluster** — `tests/run/axpby_half_out.wk:1-14` — Eighteen fixtures covering bf16/f16 GEMM, TN GEMM, FFN, reductions, min/max, transpose, narrowing output stores, literal grids and loop-carried recurrences at half width.
**`tests/run/colsum.wk`**

- **Reduction, column-reduction, arg-reduction and scan cluster** — `tests/run/colsum.wk:1-18` — Nine `col*.wk` plus `reduce*/rowargmax/argmax/cumsum/cumprod/prefix_sum/lrscan/l2norm/entropy` fixtures pin strided and index-tracking kernels that gcc leaves scalar, also `tests/run/cumsum.wk:1-18`.
**`tests/run/decode_attn.wk`**

- **GEMV / GEVM and single-token decode cluster** — `tests/run/decode_attn.wk:1-18` — Five `gemv*/gevm*` fixtures plus the KV-decode fixture pin matrix-vector, α-scaled GEMV and vector·matrix read-out dispatch, the CPU LLM serving hot path.
**`tests/run/exit_code.wk`**

- **Core scalar language and control-flow cluster** — `tests/run/exit_code.wk:1-8` — Fixtures for arithmetic, casts, radix/underscore/width-edge literals, loops with labels and break-values, recursion, shadowing, strings and `main`'s return becoming the process exit code, also `tests/run/assert_fail.wk:1-9`.
**`tests/run/generic_shape_runtime.wk`**

- **Symbolic-generic shape execution cluster** — `tests/run/generic_shape_runtime.wk:1-65` — Eight `generic*.wk` fixtures pin hidden i64 dim params, turbofish with runtime-computed dims, monomorphization, mixed literal/symbolic dims and symbolic matmul still reaching the GEMM kernel.
**`tests/run/heap_alloc.wk`**

- **Heap allocation, slices and pointer cluster** — `tests/run/heap_alloc.wk:1-31` — Six fixtures pin `alloc_*`/`free`, guaranteed zero-init on both backends, `.len()`, slice element typing and pointer deref — the only runtime-sized buffers in the language.
**`tests/run/i8_linear_dequant.wk`**

- **int8 GEMM and dequant fusion cluster** — `tests/run/i8_linear_dequant.wk:1-59` — Six `i8_*`/`dequant*` fixtures pin u8×i8 accumulation plus the per-channel dequant epilogue fused into one `wukong_i8gemm_nt_deq` writeback.
**`tests/run/import_multi.wk`**

- **Multi-file import corpus (transitive load, cycle, merged namespace)** — `tests/run/import_multi.wk:1-21` — A root importing `lib/mathlib.wk`, which imports `lib/deep.wk`, which imports mathlib back, proving cycle-breaking and one flat namespace, also `tests/run/lib/deep.wk:1-10`.
**`tests/run/io_roundtrip_f32.wk`**

- **File-I/O intrinsic round-trip cluster** — `tests/run/io_roundtrip_f32.wk:1-49` — Eight `io_*.wk` fixtures pin the headerless little-endian read/write contract for f32/f64/i8/i32/i64/u8 plus partial-read and missing-file (-1, buffer untouched) semantics, also `tests/run/io_missing_file.wk:1-27`.
**`tests/run/linear_bias_gelu.wk`**

- **nn.Linear + fused-epilogue cluster** — `tests/run/linear_bias_gelu.wk:1-47` — Eleven `linear_*.wk` fixtures pin `act(x·Wᵀ+b)` folding into one `wukong_sgemm_nt_epi` call with bias/ReLU/GELU/SiLU/residual epilogues and half-precision variants.
**`tests/run/matmul_f32.wk`**

- **Run-fixture directive protocol (RUN / EXPECT-EXIT / EXPECT-OUT)** — `tests/run/matmul_f32.wk:1-4` — Leading comment directives encode each program's CLI flags, required exit code and ordered stdout lines; 26 of 297 fixtures omit `// RUN:` and silently inherit the harness default `--run`, also `tests/run/io_roundtrip_f32.wk:1-12`.
- **GEMM / matmul recognizer coverage cluster** — `tests/run/matmul_f32.wk:1-30` — Nine `matmul*.wk` plus `gram/batched_matmul/tensor_matmul/scaled_gemm*/conv_im2col` fixtures pin the ikj-nest → `wukong_sgemm` dispatch across NN/TN/flat/accumulate spellings, also `tests/run/matmul_dynamic.wk:1-16`.
**`tests/run/mut_param.wk`**

- **Aggregate, enum and match coverage cluster** — `tests/run/mut_param.wk:1-30` — Roughly thirty fixtures for structs, tuples, arrays-of-aggregate, data-carrying enums, destructuring, value-position match/if merging and `mut` parameter by-reference semantics.
**`tests/run/now_ns_monotonic.wk`**

- **Differential-gate hygiene pattern for nondeterministic intrinsics** — `tests/run/now_ns_monotonic.wk:1-23` — Prints only the invariant `(b >= a)` rather than raw timestamps, so a wall-clock intrinsic can still be covered by interp-vs-native and opt-invariance gates.
**`tests/run/parallel_head_loop.wk`**

- **@parallel serial-twin bit-exactness oracle** — `tests/run/parallel_head_loop.wk:1-73` — The head loop runs twice — plain and `@parallel` — and prints the mismatch count, making zero the self-checking gate for the loop-outliner's disjointness proof.
**`tests/run/parallel_reduce.wk`**

- **@parallel dispatch coverage cluster** — `tests/run/parallel_reduce.wk:1-6` — Eleven `parallel_*.wk` plus `*_parallel.wk` fixtures pin the multicore variants of reduce, argmax, absmax, gemv, gemm, selu, velem, low-precision and divmod tiling.
**`tests/run/parallel_region_decline.wk`**

- **@parallel legality-decline fixture** — `tests/run/parallel_region_decline.wk:1-30` — A real cross-iteration dependence whose write index breaks the mixed-radix bound; the region matcher must decline, and any wrong parallelization races into different printed values.
**`tests/run/rope.wk`**

- **Attention / RoPE / embedding / scatter / loss cluster** — `tests/run/rope.wk:1-18` — Fixtures for sdpa, causal and multi-head attention, RoPE forward/backward, embedding gather, scatter-add, xent, kd_loss, kldiv and pooling pin the ML-op recognizers, also `tests/run/scatter_add.wk:1-18`.
**`tests/run/softmax_fused.wk`**

- **Normalization recognizer cluster (softmax / LayerNorm / RMSNorm)** — `tests/run/softmax_fused.wk:1-44` — Sixteen `*softmax*`/`*layernorm*`/`*rmsnorm*`/`norm_*` fixtures pin split, fused-window, batched, affine, out-of-place, no-eps and backward spellings all reaching `wukong_norm_*`.
**`tests/run/transcendental_sweep.wk`**

- **Breadth transcendental sweep with fused bodies and @parallel** — `tests/run/transcendental_sweep.wk:1-63` — Single-checksum fixture combining four kernels per loop body, inverse trig, and a parallel sweep, so any backend or opt-level divergence shifts one printed integer.
**`tests/run/vec256_general.wk`**

- **AST vectorizer coverage cluster (128-bit and 256-bit recipes)** — `tests/run/vec256_general.wk:1-41` — Eight `vec_*`/`vec256*` fixtures pin the general `VecKernelCall` recipe for stream×stream, in-place ReLU, sqrt-composition, int div, negation and while-loop forms that no named kernel claims.
**`tests/run/vmath_dispatch.wk`**

- **Activation / vmath elementwise cluster** — `tests/run/vmath_dispatch.wk:1-11` — Roughly forty per-op fixtures (gelu, silu, elu, selu, mish, softplus, hardswish, glu_gate, erf, trig, hyperbolics) pin the 35-op `wukong_vmath_f32` dispatch table one op at a time.
**`tests/run/vmath2_dispatch.wk`**

- **Two-input transcendental (vmath2) dispatch fixture** — `tests/run/vmath2_dispatch.wk:1-16` — Pins `pow`/`atan2`/`hypot` in an elementwise loop reaching the 256-bit `wukong_vmath2_f32`, whose expansion must mirror the inlined MIR op-for-op.
**`tests/run/xent_bwd.wk`**

- **Backward / training-kernel fixtures** — `tests/run/xent_bwd.wk:1-18` — Softmax, LayerNorm, RMSNorm, GELU, SiLU, ELU/softplus, gate and RoPE backward fixtures pin the VJP kernels that autodiff and hand-written training loops both dispatch.

## G44 — Example .wk programs and the GPT-2 end-to-end pipeline

*Files:* `crates/wukong_xbench/src/model.rs`, `examples/gemm.wk`, `examples/gpt2.wk`, `examples/gpt2_config.wk`, `examples/gpt2_forward_bench.wk`, `examples/gpt2_forward_bench_par.wk`, `examples/gpt2_infer.wk`, `examples/gpt2_infer_small.wk`, `examples/matmul.wk`, `tools/export_gpt2.py`, `tools/verify_gpt2.py`

**`crates/wukong_xbench/src/model.rs`**

- **Wukong GPT-2 block source generator** — `crates/wukong_xbench/src/model.rs:285-425` — Emits one pre-LN decoder block with every sub-op in its recognized spelling and per-head scratch declared inside the head loop so `@parallel` can outline it; the final LayerNorm module is `crates/wukong_xbench/src/model.rs:427-448`.
**`examples/gemm.wk`**

- **Runnable small-example set** — `examples/gemm.wk:1-41` — dot, fib, hello, relu, saxpy_array and gemm are compile-and-run demos; gemm doubles as the smallest end-to-end proof that a textbook nest becomes `wukong_sgemm`, also `examples/relu.wk:1-29`.
**`examples/gpt2_config.wk`**

- **Generated GPT-2 layout constant module** — `examples/gpt2_config.wk:1-46` — Element-index offsets for every tensor emitted by the exporter; the single source of truth all GPT-2 programs import, and hand-editing it silently desynchronizes them from the blob.
**`examples/gpt2_forward_bench_par.wk`**

- **GPT-2 all-core @parallel vehicle** — `examples/gpt2_forward_bench_par.wk:25-138` — The whole decoder block is one `@parallel fn` with head-private stack scratch and literal dims, ping-ponging two activation buffers so twelve layers land back in x0, also `examples/gpt2_forward_bench_par.wk:140-239`.
**`examples/gpt2_forward_bench.wk`**

- **GPT-2 124M single-core forward benchmark in recognized form** — `examples/gpt2_forward_bench.wk:1-264` — S=512 real-weight forward with per-head repacking, min-of-8 `now_ns` timing and a data-absent early return; its header still claims the LM head is unrecognized while lines 224-240 spell the dispatched GEMV.
**`examples/gpt2_infer_small.wk`**

- **Reduced-config GPT-2 correctness oracle** — `examples/gpt2_infer_small.wk:1-40` — Same pipeline and same flat-blob-plus-offset indexing at D=64/LAYERS=2/VOCAB=32 with synthetic in-loop weights, making the whole GPT-2 path runnable under the interp-vs-native gate.
**`examples/gpt2_infer.wk`**

- **Full GPT-2 124M inference program** — `examples/gpt2_infer.wk:1-218` — Loads the real flat weight blob and token ids, runs the exact forward, writes [SEQ,VOCAB] logits for the verifier and prints argmax 1757; interpreter-infeasible so it exits on a step limit.
**`examples/gpt2.wk`**

- **GPT-2 and Llama block showcases** — `examples/gpt2.wk:38-156` — Small runnable pre-norm blocks written entirely in recognized op-forms — LayerNorm+GELU MLP versus RMSNorm+RoPE+SwiGLU — as the readable specification of the dispatch surface, also `examples/llama_block.wk:25-159`.
**`examples/matmul.wk`**

- **Non-compiling aspirational syntax showcases** — `examples/matmul.wk:1-54` — matmul/softmax/vadd use `f32x8`, `row_ptr`, `@parallel(grain=1)` and have no `main`; all three fail with C0001 and pass the examples gate only because both backends fail identically, also `examples/vadd.wk:1-22`.
**`tools/export_gpt2.py`**

- **GPT-2 exporter with mandatory independent self-check** — `tools/export_gpt2.py:147-243` — Computes the offset table, transposes HF Conv1D to `[out,in]`, then re-reads the written blob and runs a numpy forward at the computed offsets to validate transposes and offsets together, also `tools/export_gpt2.py:249-288`.
**`tools/verify_gpt2.py`**

- **GPT-2 logits honesty gate** — `tools/verify_gpt2.py:56-89` — Compares the program's written logits against the HuggingFace reference and prints PASS only when relative error ≤1e-3 and both argmaxes equal 1757; numpy-only, no torch dependency.
