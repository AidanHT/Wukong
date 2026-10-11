//! Whole-program **buffer-parameter aliasing** facts.
//!
//! # The question this answers
//!
//! `fn mm(a: [f32; 16], b: [f32; 16], mut c: [f32; 16])` holding an `i/j/k` matmul nest is
//! recognized and lowered to one `wukong_sgemm(a, b, c, …)` call. That kernel writes `c` in a
//! *tiled* order while it is still reading `a` and `b`, so it equals the nest only when `c` is
//! disjoint from `a` and from `b`. Nothing in the language forbids `mm(x, y, x)`, and until this
//! module existed nothing in the compiler looked: the call was emitted with the alias visible in
//! the MIR (`call wukong_sgemm(v0, v9, v0, …)`) and both backends printed the same wrong answer —
//! `10 10 10 10 …` where the nest says `10 19 36 69 …`. Every differential, invariance and
//! determinism gate in the tree agrees on it, because both backends call the same kernel.
//!
//! `ty_is_buffer_param` already records the underlying fact: two DISTINCT parameters of array /
//! tensor / slice / pointer / reference / tuple / named type can name the same storage, because
//! all of them are passed by base pointer. What was missing is the other half — *whether any call
//! site in this program actually does that*.
//!
//! # The decision, and why
//!
//! Recognition is an optimization. The repo's own gate
//! `no_recognize_keeps_the_program_correct_on_both_backends` states the contract: "Recognition off
//! must change only *how* the program is compiled." So the language's meaning of an aliased
//! recognized call is **the meaning of the source nest** — not the kernel's, not a copy-in/copy-out
//! semantics invented by the interpreter's marshalling, and not a compile error (nothing about
//! `mm(x, y, x)` is ill-typed, and a diagnostic would reject programs that are perfectly well
//! defined as written).
//!
//! Making the three implementations agree on that meaning therefore has exactly one shape: a
//! recognizer whose kernel needs disjointness must run the scalar nest whenever disjointness does
//! not hold. That is what LLVM's `LoopIdiomRecognize` does when alias analysis cannot separate the
//! store range from the load range (it forms `memmove` where it can and bails otherwise), and what
//! Polly/LoopVersioning do when they cannot: they emit a runtime alias check and keep the original
//! loop on the may-alias arm. Both arms of that precedent exist here. This module supplies the
//! compile-time form — the whole-program facts below and [`plan_kernel_alias`], the one classifier
//! every multi-buffer emit site asks — and `kernel_guard.rs` supplies the run-time form for the
//! pairs the classifier answers `MayAlias`: an overlap test over every constrained operand's extent
//! that picks the kernel or the ORIGINAL source loop.
//!
//! # What the analysis is
//!
//! One conservative, whole-program, flow-insensitive pass, run once in `lower_program_with` before
//! any function is lowered. Wukong is a whole-program compiler: imports are merged before sema,
//! there are no globals (`ItemKind` has `Const` but no `static`), and a call resolves only through
//! a single-segment `Path` that names a `DefKind::Fn` — so there are no indirect calls and every
//! caller of every function *in the module* is visible. That makes "no call site aliases two of
//! `f`'s buffer params" a checkable whole-program property rather than an assumption — for a
//! program compiled from one root, which is what `--run` and `--emit=exe` build. It is NOT true of
//! a function entered from OUTSIDE the module, and that boundary is stated as the PREMISE bullet
//! under `# Known limits` rather than left implied.
//!
//! For each call `f(e0, …, en)` the pass classifies each argument's [`Origin`] and asks whether two
//! of them can be the same storage. A **place** (`x`, `x[i]`, `s.w`, `&x`) carries its root symbol
//! and its projection chain, so `f(l.wq, l.wk)` — the ordinary transformer-weights spelling — stays
//! provably disjoint while `f(x, x)` and `f(m, m[0])` do not. A **fresh** value (a struct/array/tuple
//! literal, or a call returning an aggregate by the sret ABI, or `alloc_f32(n)`) is storage nobody
//! else holds. Anything else is **unknown** and aliases conservatively.
//!
//! The result propagates transitively to a fixpoint: if `g`'s params `i`/`j` may alias and `g` calls
//! `f(p_i, p_j)`, then `f`'s corresponding params may alias too. Recursion terminates because the
//! per-function fact is a single monotone bit.
//!
//! **The traversal is one exhaustive walk** ([`walk_block`] / [`walk_stmt`] / [`walk_expr`], feeding
//! a single [`Node`] callback), and both `match`es over `StmtKind` and `ExprKind` are wildcard-free
//! so a new AST node is a compile error here rather than a silent hole. That is not decoration: the
//! first cut of this module had two overlapping walkers, one of which handed a block's `tail` only
//! to the *statement* path, so `fn go(a, b, mut c) { mm(a, b, c) }` — the same call minus its
//! semicolon — was invisible to the site collector and `mm` stayed "alias free". A `match` arm
//! guard was invisible for the same reason. An analysis with a hole a formatting choice walks
//! through is worse than no analysis, because it is trusted.
//!
//! # Two names for one buffer INSIDE a function
//!
//! The call-site fact above is only half of what `FnLowerer::all_provably_disjoint` needs; the other
//! half is that two distinct operand *symbols* in one function body are distinct storage. That is
//! not free either, because a `[]T` / `*T` / `&T` binding is a **view**: `lower_slice_from_source`
//! stores the source's base pointer into the fat pointer ("no element copy"), so after
//! `let mut sa: []f32 = a;` the names `sa` and `a` are one buffer, and a recognized nest written
//! over both would dispatch a disjointness-requiring kernel over overlapping storage — with no call
//! site aliasing anything, and in `main`, which has no parameters and is therefore *trivially*
//! alias-free under the call-site fact alone.
//!
//! So a function that **publishes a second name** for storage it can already reach needs a second
//! test. Both halves of it come out of the same walk ([`BodyScan`]).
//!
//! The republishing spellings are:
//!
//! * `let v: []T/&T/*T = <place>;` — the view binding above, recorded in `views`. It serves both
//!   tests: an ARGUMENT spelled `v` must resolve back to `<place>` before the call-site test can
//!   see the aliasing, and an OPERAND spelled `v` must resolve back to it before the body test can.
//!   Both go through one function, for the reason the next section gives.
//! * an aggregate literal field initialised from such a place and bound by a `let`:
//!   `let l = Layer { wq: swq };` publishes the operand `l.wq`, which `FnLowerer::field_base` names
//!   with the synthetic symbol `"<local>.<field>"`. `field_views` maps `(l, wq)` onto what `swq`
//!   views.
//! * `let l2: Layer = <place>;` — copying a whole struct is NOT covered by the value-semantics
//!   paragraph below. `emit_copy` walks the fields and sends a `[]T` one to
//!   `copy_slice_fat_ptr` — the 16-byte view, no element copy — so every borrowing field of the
//!   source gets a second operand name and each is recorded in `field_views` just as a literal's
//!   is. The value fields really do deep-copy, and nothing is recorded for them.
//! * `let (p, q) = <place>;` — a destructuring `let` copies each element, and a borrowing element
//!   copies as a fat pointer, so `p` is a second name for `t.0`. Recorded in `views` with a
//!   `Proj::Tuple` step, which keeps it precise: `t.0` and `t.1` cannot overlap, so the ordinary
//!   spelling publishes two names that do not collide, while destructuring one tuple TWICE
//!   publishes two that do.
//! * `for w in ws { … }` — the for-each element binding, which is the destructuring `let` one
//!   vocabulary away. `lower_for_array` binds an AGGREGATE element BY POINTER into the iterated
//!   buffer (`bind_slice(name, elem_ptr, …)`), and a `[]T` element arrives there too because a
//!   slice's `elem_mir` is its fat pointer, whose MIR shape is an `Array`; only a SCALAR element
//!   loads into a fresh slot and owns its storage. Recorded in `views` with a `Proj::Index` step,
//!   which keeps it precise the same way: ONE loop over `ws` publishes one name and costs nothing
//!   (`tests/run/for_each_array_of_slices.wk`), while `for w in ws { for u in ws { … } }`
//!   publishes two names that DO collide. `scan_body`'s `Node::Stmt` arm matched only `Let` and
//!   `Assign`, so this was a spelling the pass had no word for, and the double loop dispatched
//!   `wukong_sgemm` with A and C one buffer in a function no call site aliases anything in.
//!   Measured on the binary before this arm existed. Because `views` is keyed by NAME, a loop
//!   variable reused for a SECOND borrowing-element loop (`for w in ws { … } for w in vs { … }`)
//!   registers `w` twice and therefore taints, exactly as a rebound `let` view does — measured, one
//!   GEMM down to none, and it occurs nowhere in the four-root census.
//! * an assignment whose value is such a place (`p = &x;`), an array or tuple literal built from
//!   one, a struct functional update (`Layer { ..other }`), or a struct literal no `let` binds to a
//!   name. None of those can be attributed to a NAME flow-insensitively, so each taints the
//!   function outright — the coarse answer, kept for the shapes that have no precise one. Their
//!   shared gate is [`republishing_value`], and "such a place" is not all of it: a value this pass
//!   cannot classify at all counts too, because `Republish::Taint` means *cannot tell*. Reading it
//!   as "no" made `s = if flag { c } else { c };` and `let ws: [[]f32; 1] = [if flag { c } else
//!   { c }];` dispatch a GEMM over their own output, measured on both backends at -O0 and -O2.
//! * **an ARRAY initializing a `[]T` location** — `let l = L { w: c };` with `struct L { w: []f32 }`
//!   and `c: [f32; 16]`, and the same shape for an assignment `l.w = c;`. Sema's one implicit
//!   conversion unsizes the array, and `FnLowerer::init_field` then routes it to
//!   `lower_slice_from_source`, which stores `c`'s base pointer with no element copy — so `l.w` is
//!   a second name for `c` even though `c`'s own type borrows nothing. Every question of this kind
//!   goes through [`init_publishes_view`], which reads BOTH the destination's declared type and the
//!   value's; reading only the value's is what made this spelling invisible. It was invisible
//!   harmlessly until the same commit gave `init_field` its slice arm: before that the lowering
//!   read a fat pointer out of `c`'s float bytes and the program died rather than aliasing
//!   anything (`expected a pointer` on interp, exit 139 natively).
//! * a name given a view meaning TWICE (`let s: []f32 = c; … let s: []f32 = d;`). Both maps are
//!   keyed by NAME and last-writer-wins, so the second binding silently re-points every spelling of
//!   `s` — including the ones textually above it — at the wrong buffer. That taints too, and it
//!   poisons the name for the call-site half as well (`republish::ViewMaps::register`); a
//!   name-keyed flow-insensitive map has no sound precise answer here.
//!
//! The other half is the operand vocabulary. A recognizer names a buffer operand in exactly one
//! syntactic position — the BASE OF AN INDEX — and resolves it with one of two functions:
//! `single_path`, which accepts a bare local, or `buffer_base`, which accepts that plus the
//! one-level projection `l.wq` (named with the synthetic `"<local>.<field>"` symbol
//! `intern_field_bases` mints). Every helper that hands an emitter an operand symbol bottoms out in
//! one of those two over an `ExprKind::Index` base — `as_index1`, `match_operand_row_col[_off]`,
//! `affine_index`, the assignment targets — and a nest operand is by construction a buffer the
//! nest indexes. [`Operand`] is that vocabulary and [`BodyScan::operands`] records BOTH shapes at
//! every index base, so it is a superset of what any recognizer can name; the function is tainted
//! only when two DISTINCT operands resolve onto overlapping storage.
//!
//! # One resolver, and why it is a wall rather than a convention
//!
//! Resolving a name — an OPERAND or a call-site ARGUMENT, and there is no difference between them
//! — is a BOUNDED FIXPOINT over BOTH maps (`republish::ViewMaps::resolve`), not one pass over each
//! and not one map per path. A view can sit on top of a field view (`let v: []f32 = l.wq;` needs
//! the whole chain `v -> l.wq -> swq -> wq`; stopping short answers `l`, a root no other operand
//! shares) and a field view can sit on top of a view, so the two must alternate until neither
//! applies.
//!
//! That is a structural claim, and it is the fifth revision of it. There are two resolution PATHS
//! through this module and two republishing MAPS, and four consecutive revisions each fixed one
//! cell of that 2x2 and shipped the other three: the operand path walked `views` once; then `views`
//! to a fixpoint; then `field_views` was added to the operand path and the ARGUMENT path was found
//! and given `views` alone — the identical omission one frame up, which left
//! `let s: []f32 = c; let l: L = L { w: s }; let v: []f32 = l.w; mm(v, b, c)` dispatching
//! `wukong_sgemm` with A and C one buffer. Every surviving cell was that same defect: bit-identical
//! on interp and native at every `-O` level, which no differential, invariance or determinism gate
//! in the tree can see.
//!
//! A 2x2 has four places for a fifth mistake to live, so the 2x2 is deleted rather than patched.
//! Both maps are PRIVATE FIELDS of `republish::ViewMaps`; Rust privacy is per-module and a parent
//! module is not a descendant of its child, so nothing else in this file can read either map and
//! `republish::ViewMaps::resolve` is the only way to turn a name into a buffer identity. A new
//! consumer cannot resolve one map and forget the other, because it cannot reach a map at all —
//! and `tests::the_two_republishing_maps_are_read_in_exactly_one_place` reads this file back and
//! fails if either map's name appears in code outside the wall.
//!
//! That last distinction is not refinement for its own sake. The first cut of this test tainted on
//! the mere PRESENCE of a view, and the cost landed squarely on the shape the field vocabulary was
//! added for. `crates/wukong_xbench/wk/general/block_d_struct.wk`, the benchmark peer that holds a
//! transformer block's weights in a `struct Layer` the way a real model does, packs ten parameters
//! into ten `[]f32` views and then reads them only as `l.wq[…]`; presence-tainting declined all
//! NINE of its GEMM dispatches while the bare-parameter twin beside it kept them. That is not a
//! wrong answer, but it quietly turns a benchmark peer into a measurement of this pass, and nothing
//! in the ten-part gate compiles that directory. Under the operand test each of those ten names has
//! exactly ONE operand spelling, nothing overlaps, and the nine GEMMs stand.
//!
//! Every aggregate `let` NOT in the list above deep-copies (`emit_copy`, "the local owns its
//! storage (value semantics)") and publishes nothing, which is why `let b: [f32; 4] = a;` leaves
//! the function alone — an `[f32; N]` element really is copied element by element. This sentence
//! used to read "every OTHER aggregate `let`", full stop, and that was wrong about the one case
//! `emit_copy` exempts: it sends a `Ty::Slice` field to `copy_slice_fat_ptr`, which moves the
//! 16-byte view and no elements, so `let l2: Layer = l;` publishes every borrowing field. Measured
//! as a live wrong answer before the third bullet above existed. The taint stays one bit for the
//! whole function rather than a per-operand set, for the same reason the call-site fact is (below).
//!
//! # Precision, and why the coarseness is affordable
//!
//! The fact exposed is one bit per function ([`AliasFacts::alias_free`]), not a pair set: a function
//! that some call site can hand the same array twice — or that names one buffer twice itself — gets
//! no COMPILE-TIME disjointness for any pair of different operand names. What that costs is now an
//! overlap test, not the kernel: every such pair is `MayAlias`, and a recognizer under a versioning
//! context (`kernel_guard.rs`) dispatches whenever the buffers are disjoint at run time, which is
//! every call that does not actually alias. (A recognizer with no context — a probe, an outlined
//! `@parallel` chunk — still declines, as all of them did before versioning existed; the per-loop
//! cascade, the elementwise kernel probes and the statement windows all have one.) The one bit is
//! deliberate. A pair-precise fact would have to
//! survive the trip into an outlined `@parallel` body (whose recognizers see `env` captures, not
//! parameters) and into the recognizer probes (which build a `FnLowerer` with no parameter bindings
//! at all), and each of those seams is a place to get it silently wrong. One bit that rides on
//! `GemmSyms` — the struct whose own doc says it exists because it "is already threaded to every
//! recognizer and every recognizer probe" — cannot be lost on the way down.
//!
//! # Known limits
//!
//! Three kinds, and a reader must not take the list as uniformly safe. The ones marked CONSERVATIVE
//! are places the pass cannot tell and therefore taints: they cost a dispatch and nothing else. The
//! ones marked HOLE are spellings that DO reach one buffer twice and are still answered "disjoint".
//! Every hole listed is one the compiler already had before this module existed — it dispatched
//! every aliased spelling — so none of them is made reachable here; but this module does not close
//! them, and that is a weaker claim than "pre-existing holes this pass narrows rather than opens",
//! which is what this heading used to assert. The last entry is marked PREMISE: not a spelling the
//! pass mishandles, but a boundary on the sentence the whole analysis rests on. It is listed last
//! because it is the only limit here a reader cannot find by reading this file.
//!
//! **Every HOLE below was executed, not reasoned about.** Each was written as a `.wk` program over
//! the same 4x4 nest `tests/run/alias_matmul_inplace.wk` uses and run on the binary this revision
//! builds; each dispatched `wukong_sgemm` and printed the out-of-place product
//! `10 14 5 8 …` where the scalar nest says `10 14 23 32 …`, on interp and native at -O0 and -O2.
//! A bullet that says "cannot arrive that way" without that test is how the previous revision came
//! to reassure a reader about a spelling that does arrive: `match e { E::V(p) => … }` over
//! `enum E { V([]f32) }` was listed as a hole nothing reaches, and it reached it. It is now closed
//! (the CONSERVATIVE `match`-binding bullet below), and the rule for this list is: measure, or do
//! not claim.
//!
//! * CONSERVATIVE, and it reads in two directions that used to disagree. A buffer reaching a call
//!   through a `match`/`if`/`loop`/block **expression** value is `Unknown`. On the ARGUMENT path
//!   `Unknown` aliases everything, which is sound and occasionally pessimistic, and that is all
//!   this bullet used to say. On the in-body VIEW-BINDING path `Unknown` did the OPPOSITE until
//!   this revision, and that was a live wrong answer rather than a pessimism.
//!   `let v: []f32 = if flag { c } else { c };` IS a view: `let_is_slice` is tested BEFORE the
//!   aggregate deep-copy arm, so the binding goes to `lower_slice_from_source` and is a genuine
//!   fat-pointer view of whichever arm the merge produced. But an `if` is not a place and its type
//!   is a buffer param, so `origin_of` fell to its `_` arm and answered `Unknown`, the `let` arm's
//!   `if let Origin::At(p)` failed, and the arm returned having recorded nothing, tainted nothing
//!   and never entered `v` in `view_names`. `v` became its own root, and
//!   `fn mm(b, mut c, flag) { let v: []f32 = if flag { c } else { c }; <nest over v and b writing
//!   c> }` dispatched `wukong_sgemm` with A and C one buffer — `10 14 5 8 …` where the scalar nest
//!   says `10 14 23 32 …`, on interp and native at -O0 and -O2, in all four spellings (`if`, an
//!   UN-annotated `match` whose arms are the slice so that `inferred_view` fires alone, a bare
//!   `{ c }` block, and `loop { break c; }`).
//!
//!   The view binding was not the only site reading `Unknown` that way. FOUR more did, and every
//!   one was the same `if let Origin::At(p)` falling through: the struct-COPY arm
//!   (`let l2: L = if flag { l } else { l };`), a struct LITERAL's field value
//!   (`L { w: if flag { c } else { c } }`), a destructuring `let`'s initializer
//!   (`let (p, q) = if flag { t } else { t };`), and [`republishing_value`], which gates the
//!   `Assign` statement and the four aggregate-literal expression arms — so
//!   `s = if flag { c } else { c };`, `[if flag { c } else { c }]` and
//!   `[if flag { c } else { c }; 2]` all escaped. All six of those spellings were RUN, and each
//!   dispatched `wukong_sgemm` over its own output, identically on interp and native at -O0 and
//!   -O2. Five printed the out-of-place row above; the `ArrayRepeat` one printed a different wrong
//!   row, because its nest runs twice over the one buffer.
//!
//!   Those five sites, and the for-each iterand added in the same revision, now go through ONE
//!   three-way classifier, [`republish_of`], and match it exhaustively, so "record nothing" and
//!   "cannot tell" are separate arms with separate names rather than one `else`. Only a CALL is
//!   mapped to `Nothing`, because `alloc_f32(n)` is one and the CALL bullet below keeps it open
//!   deliberately. Cost, measured over the four-root census: 412 programs unchanged, 0 changed,
//!   and `block_d_struct.wk` keeps all nine of its GEMMs.
//!
//!   This is the same asymmetry the CALL bullet already named one frame down ("At a CALL SITE it is
//!   still conservative — so it is only the in-body operand test that lets it through") and did not
//!   generalise to the other `Unknown` producer, which is how the sixth instance of this class
//!   survived five passes — and then how four more SITES of it were found in one sweep, by asking
//!   the same question at every other consumer of an [`Origin`] in this file.
//! * CONSERVATIVE. A for-each loop this pass cannot attribute element-by-element taints whenever
//!   its element could be a buffer: an iterand whose sema type is neither `[T; N]` nor `[]T`
//!   (`lower_for_array` declines those to an `unsupported` diagnostic today, so no binding exists
//!   — but a widening there must not make the hole silent again), an iterand that is not a PLACE
//!   (`Fresh` says the aggregate is fresh, never that its elements are), and any binding pattern
//!   other than `Ident` or `_`. One more, which is a cost this rule ADDS rather than one it
//!   inherits: `views` is keyed by name, so reusing one loop variable for two borrowing-element
//!   loops registers it twice and taints the function the same way a rebound `let` view does. Only
//!   that last one is executed (one GEMM down to none); the three above it are read off their own
//!   arms rather than run, which for a CONSERVATIVE entry is the direction that cannot produce a
//!   wrong answer. None of the four occurs in the four-root census. The precise case, which is
//!   the one the corpus spells, is recorded in `views` with a `Proj::Index` step — see the
//!   republishing list above.
//! * CONSERVATIVE. A `match` whose SCRUTINEE type can hold a `[]T`/`*T`/`&T` leaf anywhere inside
//!   it, and any of whose arm patterns binds a name, taints the function outright. A pattern
//!   binding reaches into the scrutinee's storage, and the scrutinee itself is `Fresh` (a variant
//!   constructor's type is the enum, which does not borrow), so there is no chain back to what the
//!   payload was built from and no precise answer to give. Gated on the declared type, so every
//!   `match` over a scalar or a C-style enum — which is every `match` in the corpus — costs nothing.
//! * CONSERVATIVE. Two *different constant* indices into one array (`f(m[0], m[1])`) are treated as
//!   may-alias, because this pass does not evaluate index expressions.
//! * CONSERVATIVE. A `let` whose pattern this pass cannot attribute field-by-field taints when its
//!   initializer is a buffer-shaped place. `Ident` and `Tuple` are attributed precisely and
//!   `Wildcard`/`Unit` bind nothing, so what reaches the coarse arm today is only the refutable
//!   patterns a well-formed `let` is never spelled with — two of which (`Or`, `Variant`) do bind
//!   names where they are legal — plus any future binding pattern, which must never be answered
//!   "its own storage" by default.
//! * HOLE. A borrowing binding initialised from a CALL is not treated as a view. `let v: []f32 =
//!   alloc_f32(n)` is fresh storage and `let v: []f32 = pass_through(a)` is `a`, and this pass
//!   cannot tell them apart; tainting on the second would taint every `alloc_*` program in the
//!   corpus, so both are left as their own storage and the slice-returning user function is a
//!   stated hole. At a CALL SITE it is still conservative — [`origin_of`] classifies such a call
//!   `Unknown`, which aliases everything — so it is only the in-body operand test that lets it
//!   through. Measured: `fn mm(b, mut c) { let v: []f32 = pass_through(c); <nest over v and b
//!   writing c> }` dispatches.
//! * HOLE. An aggregate passed WHOLE into a callee carries its view across the call edge unseen.
//!   With `let l = Layer { w: sa };` (where `sa` views `a`), the call `f(l, a)` classifies `l` and
//!   `a` as two roots, so `f` stays alias free while `l.w` and `a` are one buffer, and a nest inside
//!   `f` over `x.w` and `y` gets its kernel. The tuple spelling is the same hole: `f((x, y), x)` —
//!   an argument that is an aggregate LITERAL is `Fresh`, so a callee that destructures it with
//!   `let (p, q) = t;` reads `p` and `x` as two buffers. Both measured. The fact that crosses a call
//!   edge is one bit per function, and one bit cannot carry a per-field map.
//!
//!   This bullet used to add that inside the caller `field_views` sees it and "the caller's own
//!   recognizers decline". That sentence is FALSE for exactly the caller that escapes, and it is
//!   deleted rather than softened. The in-body test compares the operands the caller SPELLS, so a
//!   caller that indexes nothing — `fn go(a) { let sa: []f32 = a; let l = Layer { w: sa }; f(l, a); }`
//!   — has an empty operand set, nothing to compare, and declines nothing. A reassurance that holds
//!   everywhere except the escaping case is worse than no reassurance, because it stops the next
//!   reader looking.
//!
//!   What is closed, and was not before this revision, is the same hazard spelled as an ARGUMENT:
//!   `f(l.w, a)`, and `f(v, a)` after `let v: []f32 = l.w;`, both resolve through `field_views` now,
//!   because the argument path and the operand path are one function.
//! * HOLE. A struct local built by a CALL rather than by a literal (`let l: Layer = make(a);`) has
//!   no `field_views` entry, so `l.w` is treated as its own storage even if `make` stored one of its
//!   arguments there. Measured. Same root cause as the bullet above, and the same root cause as the
//!   next one: storage crossing a boundary this pass answers `Fresh` at.
//! * HOLE. Likewise a function returning an aggregate that contains a view: [`origin_of`] calls such
//!   a return `Fresh` (the sret buffer really is fresh), and only a *borrowing* projection out of it
//!   (`make().buf`) is demoted to `Unknown` by [`project`]. Since this revision that demotion also
//!   taints a view BINDING of it (`let v: []f32 = make().buf;` is `Unknown` and is not a call), so
//!   what is left open here is the operand `l.w` after `let l: Layer = make(a);`, which is `Fresh`
//!   and never demoted at all.
//!
//! Those four holes are ONE root, stated plainly because the next reader deserves the shape rather
//! than four anecdotes: **a value that carries borrowed storage inside it is classified by the
//! expression that produced it, and a call or an aggregate literal produces `Fresh`.** Closing them
//! needs something this pass does not have — either a per-field summary crossing the call edge
//! (today it is one bit, deliberately, so it cannot be lost at the `@parallel`/probe seams) or an
//! inter-procedural "which argument does this return borrow from" fact. Neither is a refinement of
//! what is here; both are a different analysis. Until one exists, these four spellings dispatch a
//! disjointness-requiring kernel over overlapping storage, exactly as the whole compiler did for
//! every aliased spelling before this module was written.
//!
//! * PREMISE. "Every caller of every function is visible" (`# What the analysis is`) is true of a
//!   program compiled from ONE root — `--run`, `--emit=exe` — where `load_imports` merges every
//!   module before sema and no call is indirect. It is FALSE the moment a lowered function is
//!   entered from outside the module, and this tree has two such doors. `--emit=obj` writes a
//!   relocatable object whose exported functions anything may link against and call, with no call
//!   site in the `.wk` source at all. And `wukong_xbench` JIT-compiles each peer `.wk` with
//!   `jit_module`, pulls `kbench` out by symbol and calls it through a raw `extern "C"` fn pointer
//!   from Rust — all seven peer sources DECLARE `kbench` and none of them calls it, so `aliasing`
//!   never names it and every one is alias free BY DEFAULT rather than by proof (the C, C++ and
//!   Rust peers beside them are entered the same way, `dlopen`'d through `libloading`).
//!
//!   Nothing is wrong today, and that is measured rather than assumed: `call_block` derives its 27
//!   pointers from 27 distinct `Vec`s, so no two arguments of any peer `kbench` are one buffer. But
//!   "alias free" for such a function means "no call site IN THIS MODULE aliases it", which is a
//!   strictly weaker claim than the one the rest of this file makes, and a reader who takes
//!   `AliasFacts::alias_free`'s doc at face value will not find that out here. Closing it needs
//!   either a declared contract on the entry point or a pessimistic default for functions nothing
//!   in the module calls — and the second would decline all nine of `block_d_struct.wk`'s GEMMs
//!   and every kernel in every `--emit=obj` module in the tree, which is why the boundary is stated
//!   rather than taken.

use wukong_ast::{self as ast, Block, Expr, ExprKind, ForIter, Module, Stmt, StmtKind};
use wukong_sema::{DefKind, SemaResult};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Symbol};
use wukong_types::Ty;

use crate::ty_is_buffer_param;
use republish::ViewMaps;

/// One step of a place projection. `Index` is deliberately index-value-blind: two indices into one
/// array are assumed to meet.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Proj {
    Field(Symbol),
    Tuple(u32),
    Index,
}

/// A syntactic place: a root symbol plus the projections applied to it. `l.blocks[i].wq` is
/// `{ base: l, proj: [Field(blocks), Index, Field(wq)] }`.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Place {
    base: Symbol,
    proj: Vec<Proj>,
}

/// Where a call argument's storage comes from.
#[derive(Clone, Debug)]
enum Origin {
    /// A place in the caller's frame (or reachable from one of its parameters).
    At(Place),
    /// Storage created by the expression itself and held by nobody else: an array/struct/tuple
    /// literal, or a call whose return type is an AGGREGATE (it comes back through the sret ABI
    /// into a buffer the caller just allocated). Cannot alias anything.
    ///
    /// This does NOT include `alloc_f32(n)`: its return type is `[]f32`, which borrows, so
    /// [`origin_of`] classifies it `Unknown`. What keeps the whole `alloc_*` slice corpus
    /// dispatching is that `Unknown` is not a PLACE, so it publishes no second name — not that it
    /// is fresh, which this pass has no way to prove.
    Fresh,
    /// Not classified. Aliases everything, conservatively.
    Unknown,
}

/// Can these two places name overlapping storage? Different roots cannot (a local is its own
/// allocation, and two *parameters* that could be the same array are related through the call-site
/// pass below, not here). Same root: walk the projections in step and look for a pair that proves
/// separation — two different named fields, or two different tuple indices. `Index` never proves
/// anything, and a prefix relation (`l` vs `l.wq`) is containment, which is overlap.
fn places_may_overlap(a: &Place, b: &Place) -> bool {
    if a.base != b.base {
        return false;
    }
    for (pa, pb) in a.proj.iter().zip(b.proj.iter()) {
        match (pa, pb) {
            (Proj::Field(x), Proj::Field(y)) if x != y => return false,
            (Proj::Tuple(x), Proj::Tuple(y)) if x != y => return false,
            _ => {}
        }
    }
    true
}

fn origins_may_overlap(a: &Origin, b: &Origin) -> bool {
    match (a, b) {
        (Origin::Fresh, _) | (_, Origin::Fresh) => false,
        (Origin::Unknown, _) | (_, Origin::Unknown) => true,
        (Origin::At(x), Origin::At(y)) => places_may_overlap(x, y),
    }
}

/// A type that hands back storage its holder did not create — a `[]T` slice, a raw pointer, a
/// reference, or a `Tensor[..]`. Used two ways: a *call* whose return type is one of these can be
/// handing back one of its own arguments, and a *binding* of one of these is a view of whatever it
/// was initialised from. A tensor value IS such a view — a base pointer with a shape, never storage:
/// `let r = a[i];` (a row of `a`), `let h = x as Tensor[..]` (a reshape of `x`) and `let t = a;`
/// all name `a`'s / `x`'s buffer, and a tensor-returning call returns one of its arguments' buffers
/// (it has no sret). An aggregate (`[T; N]`, a struct/tuple) is neither: it returns through the
/// sret ABI into a buffer the caller just allocated, and it deep-copies on `let`.
fn ty_is_borrowing(t: &Ty) -> bool {
    matches!(
        t,
        Ty::Slice(_) | Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Tensor { .. }
    )
}

/// Classify an argument expression. See [`Origin`].
fn origin_of(e: &Expr, sema: &SemaResult) -> Origin {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Origin::At(Place {
            base: p.first().sym,
            proj: vec![],
        }),
        ExprKind::Field { base, name } => project(e, base, Proj::Field(name.sym), sema),
        ExprKind::TupleField { base, index } => project(e, base, Proj::Tuple(*index), sema),
        ExprKind::Index { base, .. } => project(e, base, Proj::Index, sema),
        ExprKind::Unary {
            op: ast::UnOp::Ref | ast::UnOp::RefMut | ast::UnOp::Deref,
            expr,
        } => origin_of(expr, sema),
        ExprKind::Cast { expr, .. } => origin_of(expr, sema),
        // Literal aggregates build a temporary the caller owns outright.
        ExprKind::ArrayLit(_)
        | ExprKind::ArrayRepeat { .. }
        | ExprKind::StructLit { .. }
        | ExprKind::TupleLit(_) => Origin::Fresh,
        // A call's value is fresh unless the callee's return type is a *borrowing* one, in which
        // case it can be one of the callee's own arguments coming back out.
        ExprKind::Call { .. } => match sema.types.get(&e.id) {
            Some(t) if ty_is_borrowing(t) => Origin::Unknown,
            Some(_) => Origin::Fresh,
            None => Origin::Unknown,
        },
        // Scalars cannot alias; everything else is unclassified.
        _ => match sema.types.get(&e.id) {
            Some(t) if !ty_is_buffer_param(t) => Origin::Fresh,
            _ => Origin::Unknown,
        },
    }
}

/// Extend the base's origin by one projection step. `whole` is the projected expression itself, and
/// it is needed for exactly one case: a *borrowing* projection out of a **fresh** aggregate is NOT
/// fresh. `make().buf` allocates its sret temporary here, but the `[]f32` inside it points at
/// storage the callee already had, so propagating `Fresh` through the field would classify it as
/// "storage nobody else holds" — the one answer that can never be walked back.
fn project(whole: &Expr, base: &Expr, step: Proj, sema: &SemaResult) -> Origin {
    match origin_of(base, sema) {
        Origin::At(mut p) => {
            p.proj.push(step);
            Origin::At(p)
        }
        Origin::Fresh if sema.types.get(&whole.id).is_some_and(ty_is_borrowing) => Origin::Unknown,
        other => other,
    }
}

/// Did this expression's [`Origin::Unknown`] come out of [`origin_of`]'s **call** arm -- i.e. is
/// it a call whose return type borrows?
///
/// `Unknown` has two producers, and a REPUBLISHING site needs opposite treatments for them. A call
/// is the producer the `# Known limits` CALL bullet deliberately keeps open, because `alloc_f32(n)`
/// is one and tainting it would taint every `alloc_*` program in the corpus. Every OTHER producer
/// -- an `if`/`match`/`loop`/block value, a borrowing projection out of a `Fresh` aggregate --
/// carries no such cost and must taint. [`republish_of`] is the one caller, and its doc says what
/// it cost not to draw this line.
///
/// [`origin_of`] peels `Cast` and the three borrow/deref `Unary` operators BEFORE it classifies, so
/// this peels the same three; peeling a different set would answer about a different expression
/// than the one that was classified.
fn origin_unknown_is_a_call(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Call { .. } => true,
        ExprKind::Cast { expr, .. }
        | ExprKind::Unary {
            op: ast::UnOp::Ref | ast::UnOp::RefMut | ast::UnOp::Deref,
            expr,
        } => origin_unknown_is_a_call(expr),
        _ => false,
    }
}

/// What a REPUBLISHING site must do with an initializer, as opposed to what a CALL SITE must do
/// with an argument.
///
/// [`Origin`] answers a three-way question, and every republishing consumer in this file used to
/// read one arm of it with an `if let Origin::At(p)`. That silently treats BOTH other arms as
/// "nothing to record" -- and only one of them is. `Fresh` really is nothing: the initializer built
/// the storage itself. `Unknown` is *cannot tell*, and at a call site it aliases everything, which
/// is the conservative direction; at a republishing site the same value meant "record nothing,
/// taint nothing, and never enter the name in `view_names`", so the name silently became its own
/// root. That is the OPPOSITE direction, and it shipped at FIVE separate sites: the `let` view
/// binding, the struct-COPY arm, a struct literal's field value, a destructuring `let`'s
/// initializer, and [`republishing_value`], which gates the `Assign` statement and the four
/// aggregate-literal expression arms. Every one of the five was executed on the binary as a live
/// wrong answer -- a recognized GEMM dispatched over an output aliasing its own A operand, on
/// interp and native alike at -O0 and -O2. The sixth consumer, the for-each iterand, already took
/// the coarse answer and is routed here so that there is one rule rather than five plus a habit.
///
/// This enum exists so that mistake cannot be respelled: a site matches it exhaustively, and the
/// arm it must not forget has a name.
enum Republish {
    /// The initializer names storage reachable from this place. Record the view.
    At(Place),
    /// Storage nobody else holds, or a call. Record nothing, taint nothing.
    Nothing,
    /// Cannot tell. Taint.
    Taint,
}

/// The one classifier every republishing site in this file uses.
///
/// A CALL is the single `Unknown` producer mapped to `Nothing` rather than `Taint`, and it is
/// deliberate: `alloc_f32(n)` returns `[]f32`, so [`origin_of`] classifies it `Unknown` through the
/// call arm, and tainting it would cost every `alloc_*` program in the corpus its kernels. That
/// exemption is the `# Known limits` CALL bullet, and [`origin_unknown_is_a_call`] is where it
/// lives.
fn republish_of(e: &Expr, sema: &SemaResult) -> Republish {
    match origin_of(e, sema) {
        Origin::At(p) => Republish::At(p),
        Origin::Fresh => Republish::Nothing,
        Origin::Unknown if origin_unknown_is_a_call(e) => Republish::Nothing,
        Origin::Unknown => Republish::Taint,
    }
}

/// One resolved call site: which function, and where each argument's storage came from.
struct Site {
    callee: Symbol,
    args: Vec<Origin>,
}

/// Everything the fixpoint needs about one function.
#[derive(Default)]
struct FnInfo {
    /// Indices of the parameters that are passed by base pointer, and their AST names.
    buffer_params: Vec<(usize, Symbol)>,
    /// Calls made from this function's body.
    sites: Vec<Site>,
    /// Two DIFFERENT kernel-operand names in this body can reach one buffer, or the body
    /// republishes storage in a shape this pass cannot attribute to a name — see the module doc's
    /// "Two names for one buffer INSIDE a function". Kept out of the fixpoint deliberately: it says
    /// nothing about whether this function's own *parameters* can be one array, which is the only
    /// thing that forwards to a callee.
    internal_view: bool,
}

/// The whole-program answer. See the module doc.
pub(crate) struct AliasFacts {
    /// Functions for which some call site can bind two DISTINCT buffer parameters to overlapping
    /// storage. Absence is the common case, so the set is normally empty.
    aliasing: HashSet<Symbol>,
    /// Functions whose own body names one buffer twice. A separate concern from `aliasing`, ORed
    /// into [`AliasFacts::alias_free`].
    internal_views: HashSet<Symbol>,
}

impl AliasFacts {
    /// `true` when no two distinct buffer parameters of `f` can name overlapping storage at any call
    /// site in the program, AND `f`'s own body does not give one buffer two names — i.e. a
    /// recognizer inside `f` may rely on two distinct operand symbols being distinct storage.
    /// Conservative: `false` also covers "could not tell".
    ///
    /// "At any call site in the program" means at any call site IN THIS MODULE. A function entered
    /// from outside it -- an `--emit=obj` export, or an `xbench` peer's `kbench` reached through a
    /// raw fn pointer -- has no call site here at all and is therefore answered `true` by DEFAULT
    /// rather than by proof. See the PREMISE bullet in the module doc.
    pub(crate) fn alias_free(&self, f: Symbol) -> bool {
        !self.aliasing.contains(&f) && !self.internal_views.contains(&f)
    }

    /// Run the analysis over a whole (import-merged, normalized) module.
    pub(crate) fn analyze(module: &Module, sema: &SemaResult) -> Self {
        let mut info: HashMap<Symbol, FnInfo> = HashMap::default();

        for item in &module.items {
            let ast::ItemKind::Fn(f) = &item.kind else {
                continue;
            };
            let Some(body) = &f.body else {
                continue;
            };
            let param_tys: Vec<Ty> = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
                Some(DefKind::Fn(sig)) => sig.params.clone(),
                // No resolved signature: fall back to "every parameter could be a buffer", which
                // only ever makes the answer more conservative.
                _ => f
                    .params
                    .iter()
                    .map(|_| Ty::Ptr {
                        mutable: true,
                        pointee: Box::new(Ty::Unknown),
                    })
                    .collect(),
            };
            let buffer_params: Vec<(usize, Symbol)> = f
                .params
                .iter()
                .enumerate()
                .filter(|(i, _)| param_tys.get(*i).is_none_or(ty_is_buffer_param))
                .map(|(i, p)| (i, p.name.sym))
                .collect();

            let mut scan = scan_body(body, sema);
            // Two answers come out of that one walk, both resolved AFTER it rather than during it.
            // The analysis is flow-insensitive either way, so a view declared textually below its
            // use resolves identically — and deferring is what lets ONE traversal do the job the
            // first cut of this module needed two for.
            //
            // First: can two DIFFERENT operand names in this body reach one buffer?
            let internal_view = scan.opaque_republish || body_names_one_buffer_twice(&scan);
            // Second: an argument spelled with a republished name has to resolve back onto the
            // storage it names before the call-site test can see the aliasing. The SAME function as
            // the operand path above -- see the wall's own comment for why that is load-bearing and
            // not tidiness.
            for site in &mut scan.sites {
                for a in &mut site.args {
                    *a = scan
                        .republished
                        .resolve(std::mem::replace(a, Origin::Unknown));
                }
            }

            info.insert(
                f.name.sym,
                FnInfo {
                    buffer_params,
                    sites: scan.sites,
                    internal_view,
                },
            );
        }

        // Fixpoint. Seed: a call site whose two argument origins can overlap makes the callee
        // aliased. Propagate: once a caller is aliased, two of ITS parameters can be one array, so
        // a call that forwards them makes the callee aliased too.
        let mut aliasing: HashSet<Symbol> = HashSet::default();
        loop {
            let mut changed = false;
            for (caller, fi) in &info {
                let caller_aliased = aliasing.contains(caller);
                for site in &fi.sites {
                    let Some(callee) = info.get(&site.callee) else {
                        continue;
                    };
                    if aliasing.contains(&site.callee) {
                        continue;
                    }
                    if site_aliases(site, callee, fi, caller_aliased) {
                        aliasing.insert(site.callee);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let internal_views = info
            .iter()
            .filter(|(_, fi)| fi.internal_view)
            .map(|(f, _)| *f)
            .collect();
        AliasFacts {
            aliasing,
            internal_views,
        }
    }
}

// ---- THE KERNEL-OPERAND CLASSIFIER -------------------------------------------------------------
//
// Everything above answers ONE bit per function. A recognizer emitting a kernel needs a different
// question answered, about the specific operands it is about to hand the kernel, and it needs a
// THREE-way answer, because the three answers lead to three different lowerings:
//
// * `Disjoint`  -> emit the kernel. Nothing to check at run time.
// * `MustAlias` -> the two operands are spelled with the SAME symbol, so they are the same buffer
//                  from the same base. The kernel may run only if its contract permits exact
//                  in-place operation for that pair; otherwise the source loop is the program.
// * `MayAlias`  -> the compiler cannot separate them. Version the dispatch at run time: an overlap
//                  test over the operands' extents picks the kernel when they are disjoint and the
//                  ORIGINAL SOURCE LOOP when they are not (`kernel_guard.rs`). That is LLVM's
//                  `LoopVersioning` shape, and it replaces what this compiler did before: a compile-
//                  time decline, which kept a library-style `fn mm(a, b, mut c)` off the kernel for
//                  EVERY call once ANY call site in the program aliased it.
//
// Every multi-buffer emit site asks through `plan_kernel_alias`, so "two different symbols, so two
// different buffers" -- the AST symbol comparison sixteen matchers used to treat as a proof -- is
// no longer an answer anywhere: a symbol comparison can only ever DECLINE (same symbol), never
// establish disjointness on its own.

/// How two buffer operands a recognizer is about to hand one kernel can relate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum KernelAlias {
    /// Provably distinct storage: different symbols in a function `AliasFacts::alias_free` vouches
    /// for (no call site binds two of its buffer parameters to one array, and its body never gives
    /// one buffer two operand names).
    Disjoint,
    /// The same symbol, so the same buffer from the same base.
    MustAlias,
    /// Different symbols the compiler cannot separate at compile time.
    MayAlias,
}

/// Classify one operand pair. `alias_free` is the enclosing function's whole-program bit
/// (`GemmSyms::alias_free`); it is the only thing that lets two different symbols be `Disjoint`.
pub(crate) fn classify_kernel_operands(alias_free: bool, a: Symbol, b: Symbol) -> KernelAlias {
    if a == b {
        KernelAlias::MustAlias
    } else if alias_free {
        KernelAlias::Disjoint
    } else {
        KernelAlias::MayAlias
    }
}

/// What a kernel's own contract permits for one pair of its operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PairContract {
    /// The two operands must not overlap at all. Every cross-element kernel (GEMM, GEMV, GEVM,
    /// transpose, pool2d, the column and row reductions, the losses) is this for each output
    /// against each input and each other output.
    Disjoint,
    /// Exact in-place operation (same base, same extent) computes what the source loop computes;
    /// any OTHER overlap does not. The row-local scans and rotations whose runtime contract says
    /// "`out` may alias `x`" (cumsum, cummax/cummin, cumprod, lrscan, RoPE) and the backward
    /// passes that document one permitted alias (softmax/RMSNorm backward `dx` over `dy`, the
    /// cross-entropy backward `dx` over `x`). A SHIFTED overlap is not exact: the AVX2 row scan
    /// loads eight lanes before it stores them, while the source loop reads `x[i+1]` after it has
    /// already written `out[i] = x[i+1]`.
    ExactInPlace,
}

/// One operand pair a kernel's contract constrains, by index into the emitter's operand list.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct OperandPair {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) contract: PairContract,
}

impl OperandPair {
    pub(crate) fn disjoint(a: usize, b: usize) -> Self {
        OperandPair {
            a,
            b,
            contract: PairContract::Disjoint,
        }
    }

    pub(crate) fn exact_in_place(a: usize, b: usize) -> Self {
        OperandPair {
            a,
            b,
            contract: PairContract::ExactInPlace,
        }
    }
}

/// The lowering a kernel dispatch gets, given its operands and its contract.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum KernelAliasPlan {
    /// Every constrained pair is compile-time safe: emit the kernel unconditionally.
    Dispatch,
    /// Some pair is the same buffer and the contract forbids that: lower the source loop.
    Decline,
    /// These pairs need a run-time overlap test; the rest are compile-time safe.
    Version(Vec<OperandPair>),
}

/// **The one classifier every multi-buffer recognizer emit site asks.** `syms[i]` is the operand
/// symbol at index `i`; `pairs` are the constraints the kernel's contract places on them.
///
/// A `MustAlias` pair under a `Disjoint` contract declines outright -- a run-time test would only
/// ever pick the fallback, so emitting the kernel arm would be dead code. A `MustAlias` pair under
/// `ExactInPlace` is exact in-place by construction (same symbol, same base, and the emitters that
/// use this contract apply no per-operand offset), so it needs no test. Only `MayAlias` pairs reach
/// the run-time guard.
pub(crate) fn plan_kernel_alias(
    alias_free: bool,
    syms: &[Symbol],
    pairs: &[OperandPair],
) -> KernelAliasPlan {
    let mut runtime = Vec::new();
    for &pair in pairs {
        let (Some(&a), Some(&b)) = (syms.get(pair.a), syms.get(pair.b)) else {
            // An index past the operand list is an emitter bug; refuse the kernel rather than
            // dispatch on an unchecked pair.
            return KernelAliasPlan::Decline;
        };
        match (classify_kernel_operands(alias_free, a, b), pair.contract) {
            (KernelAlias::Disjoint, _) => {}
            (KernelAlias::MustAlias, PairContract::ExactInPlace) => {}
            (KernelAlias::MustAlias, PairContract::Disjoint) => return KernelAliasPlan::Decline,
            (KernelAlias::MayAlias, _) => {
                if !runtime.contains(&pair) {
                    runtime.push(pair);
                }
            }
        }
    }
    if runtime.is_empty() {
        KernelAliasPlan::Dispatch
    } else {
        KernelAliasPlan::Version(runtime)
    }
}

/// A name a recognizer can use as a kernel buffer **operand**. The lowerer turns a nest
/// subexpression into an operand symbol through two functions: `single_path`, which accepts a bare
/// local `x`, and `buffer_base`, which accepts that plus the one-level projection `l.wq`, named
/// with the synthetic `"<local>.<field>"` symbol `intern_field_bases` mints. `buffer_base` is the
/// wider of the two, so its two shapes are the whole vocabulary, and anything else both decline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Operand {
    Name(Symbol),
    Field(Symbol, Symbol),
}

/// The [`Operand`] an index base names, in `buffer_base`'s two shapes (`single_path`, else a
/// `Field` whose own base is a `single_path`). A SUPERSET of what `buffer_base` accepts, never a
/// subset: it declines wherever `buffer_base` declines, and it additionally accepts one case
/// `buffer_base` rejects — `buffer_base` resolves the synthetic `"<local>.<field>"` name with
/// `interner.get`, so an `l.f` whose symbol was never minted comes back `None` there and `Some`
/// here. Wide is the sound direction: an operand recorded here that no recognizer can name costs at
/// most a decline, while one missed here is a kernel emitted over storage it does not own.
impl Operand {
    /// The syntactic place this spelling names, BEFORE any republishing is resolved. Turning it
    /// into the storage it really reaches is [`republish::ViewMaps::resolve`]'s job and nothing
    /// else's.
    fn place(self) -> Place {
        match self {
            Operand::Name(s) => Place {
                base: s,
                proj: Vec::new(),
            },
            Operand::Field(b, f) => Place {
                base: b,
                proj: vec![Proj::Field(f)],
            },
        }
    }
}

fn operand_of(e: &Expr) -> Option<Operand> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(Operand::Name(p.first().sym)),
        ExprKind::Field { base, name } => match &base.kind {
            ExprKind::Path(p) if p.is_single() => Some(Operand::Field(p.first().sym, name.sym)),
            _ => None,
        },
        _ => None,
    }
}

// ---- THE REPUBLISHING WALL -------------------------------------------------------------------
//
// This module holds the two maps that record a second name for one buffer, and the ONE function
// that reads them. It exists to make a specific mistake UNSPELLABLE rather than merely absent.
//
// There are two resolution PATHS through this file -- the in-body operand test
// (`body_names_one_buffer_twice`) and the call-site argument test (`AliasFacts::analyze`) -- and two
// republishing MAPS (`views` for `let v: []T = <place>;`, `field_views` for
// `let l = L { w: <place> };`). That is a 2x2, and four consecutive revisions of this module each
// fixed ONE CELL of it and shipped the other three: the operand path walked `views` once; then
// `views` to a fixpoint; then the ARGUMENT path was found and given `views` alone, which is the
// identical omission one frame up. Every cell left behind was the same observable defect -- a
// recognized GEMM dispatched over an output aliasing its own A operand, bit-identical on interp and
// native at every `-O` level, which no differential, invariance or determinism gate in the tree can
// see.
//
// A 2x2 has four places for a fifth mistake to live, so the 2x2 is deleted rather than patched.
// Both maps are PRIVATE FIELDS of `republish::ViewMaps`; Rust privacy is per-module and a parent
// module is NOT a descendant of its child, so nothing in the rest of `alias.rs` can read either map,
// and `republish::ViewMaps::resolve` is the only way to turn a spelled name into a buffer identity.
// A new consumer cannot resolve one map and forget the other, because it cannot reach a map at all.
// `crate::tests::the_two_republishing_maps_are_read_in_exactly_one_place` additionally reads this
// file back and fails if either map's name appears in code outside this wall, so the wall cannot be
// widened by accident either.
mod republish {
    use super::{Origin, Place, Proj};
    use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Symbol};

    /// How many links [`ViewMaps::resolve`] will rewrite before giving up. Chains in real code are
    /// two or three (`v -> l.wq -> swq -> wq` is the deepest this tree spells); the cap exists so a
    /// `let a = b; let b = a;` cycle the front end would have rejected cannot spin. Exhausting it
    /// answers `Unknown`, which aliases everything, so the cap errs in the safe direction.
    const RESOLVE_STEPS: usize = 16;

    /// The republishing facts of one function body: which names are a second spelling of storage
    /// something else can already reach.
    ///
    /// Both maps are keyed by NAME and this pass is flow-insensitive, so both are last-writer-wins;
    /// `rebound` is what keeps that honest. The fields are private ON PURPOSE -- see the wall
    /// comment above this module.
    #[derive(Default)]
    pub(super) struct ViewMaps {
        /// `let v: []T/&T/*T = <place>;` -- `v` reaches whatever `<place>` reaches.
        views: HashMap<Symbol, Place>,
        /// `let l = Layer { wq: <place>, ... };` -- the operand `l.wq` reaches `<place>`.
        field_views: HashMap<(Symbol, Symbol), Place>,
        /// Every name that has been given a view meaning, whether through `views` or as the local of
        /// a `field_views` entry. Only used to detect the SECOND one -- see [`ViewMaps::register`].
        view_names: HashSet<Symbol>,
        /// Names that acquired a view meaning twice; nothing rooted at one of them resolves soundly.
        rebound: HashSet<Symbol>,
    }

    impl ViewMaps {
        /// **The one function in this file that turns a spelled name into a buffer identity.**
        /// Every consumer goes through it -- the in-body operand test and the call-site argument
        /// test alike -- and any new consumer must too. That is not a convention: the maps it walks
        /// are private to this module precisely so that no caller can walk one of them and forget
        /// the other, which is the mistake four consecutive revisions of this file shipped.
        ///
        /// One flat bounded fixpoint over BOTH maps, not a pass over each. A view can sit on top of
        /// a field view -- `let v: []f32 = l.w;` needs the whole chain `v -> l.w -> sa -> c`, and
        /// stopping short answers `l`, a root no other operand shares -- and a field view can sit on
        /// top of a view, so the two have to alternate until neither applies. Each round rewrites
        /// exactly one link.
        ///
        /// `Fresh` and `Unknown` pass through unchanged: neither is a place, so neither has a name
        /// to resolve.
        pub(super) fn resolve(&self, o: Origin) -> Origin {
            let Origin::At(mut p) = o else {
                return o;
            };
            for _ in 0..RESOLVE_STEPS {
                // A rebound root has no resolution this pass can trust -- the map kept one of the
                // name's two bindings and which one is a flow-sensitive question -- so it answers
                // `Unknown`, which aliases everything. That is what keeps BOTH paths honest:
                // `let s: []f32 = c; mm(s, c); let s: []f32 = d;` would otherwise resolve the
                // argument `s` to `d`, see two distinct roots, and leave `mm` alias free while both
                // of its arguments are one array.
                if self.rebound.contains(&p.base) {
                    return Origin::Unknown;
                }
                if let Some(src) = self.views.get(&p.base) {
                    if src.base != p.base {
                        let mut proj = src.proj.clone();
                        proj.extend(p.proj);
                        p = Place {
                            base: src.base,
                            proj,
                        };
                        continue;
                    }
                }
                // Is what we now hold a field this body republished? `field_views` is keyed by the
                // one-level `(local, field)` pair `buffer_base` can name, so only a LEADING `Field`
                // step can match it; anything below that step rides along unchanged.
                if let Some(Proj::Field(f)) = p.proj.first() {
                    if let Some(src) = self.field_views.get(&(p.base, *f)) {
                        let mut proj = src.proj.clone();
                        proj.extend(p.proj.iter().skip(1).cloned());
                        p = Place {
                            base: src.base,
                            proj,
                        };
                        continue;
                    }
                }
                return Origin::At(p);
            }
            Origin::Unknown
        }

        /// Record `let v: []T/&T/*T = <src>;`. Returns `true` when `v` already had a view meaning,
        /// which the caller turns into a taint -- see [`ViewMaps::register`].
        pub(super) fn note_view(&mut self, name: Symbol, src: Place) -> bool {
            let rebound = self.register(name);
            self.views.insert(name, src);
            rebound
        }

        /// Record one borrowing field of a struct local: the operand `l.wq` reaches `<src>`.
        /// Both spellings use it -- the literal `let l = Layer { wq: <src>, ... };` and the copy
        /// `let l2: Layer = <place>;`, whose borrowing fields keep the source's base pointer.
        /// Registering the LOCAL is [`ViewMaps::note_struct_local`]'s job, once per `let` rather
        /// than once per field.
        pub(super) fn note_field_view(&mut self, local: Symbol, field: Symbol, src: Place) {
            self.field_views.insert((local, field), src);
        }

        /// Register the local of a struct `let` -- literal or copy -- that published at least one
        /// borrowing field. ONCE per `let`, however many it published: counting per field would read
        /// `Layer { wq: swq, wk: swk }` as a rebinding of `l` and cost `block_d_struct.wk` -- ten
        /// weights packed into one `Layer` -- every kernel it has. Returns `true` on a rebinding.
        pub(super) fn note_struct_local(&mut self, local: Symbol) -> bool {
            self.register(local)
        }

        /// Note that `name` now has a view meaning, and report whether it already had one.
        ///
        /// Both maps are keyed by NAME and this pass is flow-insensitive, so a second binding of a
        /// name overwrites the first in `views` (and merges into the same local's entries in
        /// `field_views`); every operand or argument spelled with that name -- including the ones
        /// textually ABOVE the rebinding -- then resolves onto the wrong buffer.
        /// `fn mm(b, mut c, d) { let s: []f32 = c; <nest over s and b writing c>; let s: []f32 = d; }`
        /// resolved the nest's `s` to `d`, found two distinct roots, and dispatched a GEMM writing
        /// its own A operand. There is no sound precise answer for a name-keyed map here, so the
        /// coarse one is the honest answer: the caller taints the function, and `rebound`
        /// additionally poisons the name inside [`ViewMaps::resolve`], so nothing spelled with it
        /// can resolve into a false disjointness on either path.
        fn register(&mut self, name: Symbol) -> bool {
            if self.view_names.insert(name) {
                return false;
            }
            self.rebound.insert(name);
            true
        }
    }
}
// ---- END OF THE REPUBLISHING WALL ------------------------------------------------------------

/// Everything one walk of one function body produces.
#[derive(Default)]
struct BodyScan {
    /// The two republishing maps, behind the wall above: the only way to ask what a name reaches is
    /// `republish::ViewMaps::resolve`, and this field is why no other way exists.
    republished: ViewMaps,
    /// The base of every `Index` in the body, in [`Operand`]'s vocabulary: exactly the names a
    /// recognizer can hand [`crate::FnLowerer::all_provably_disjoint`], and therefore the only
    /// names two of which can be one nest's operands.
    operands: HashSet<Operand>,
    /// The body republishes storage in a shape this pass cannot attribute to a NAME. Taints the
    /// function outright -- see the module doc.
    opaque_republish: bool,
    /// Calls made from this body, arguments unresolved.
    sites: Vec<Site>,
}

/// One walk of one function body. See [`BodyScan`].
fn scan_body(body: &Block, sema: &SemaResult) -> BodyScan {
    let mut out = BodyScan::default();
    // Struct literals that a `let` binds to a name are recorded precisely in `field_views`; the
    // generic expression arm below must not ALSO taint them, or the precision is undone. Safe
    // because `walk_stmt` delivers the statement before it descends into the initializer, so the id
    // is recorded before the literal is reached.
    let mut attributed: HashSet<ast::NodeId> = HashSet::default();
    walk_block(body, &mut |n| match n {
        Node::Stmt(s) => match &s.kind {
            StmtKind::Let {
                pat,
                ty,
                init: Some(e),
                ..
            } => {
                let name = match &pat.kind {
                    ast::PatKind::Ident(n) => n,
                    // `let (p, q) = <place>;` -- a destructuring `let` copies each element, and a
                    // BORROWING element copies as a fat pointer, so `p` is a second name for
                    // `t.0`. Attributable precisely, because `Proj::Tuple` already distinguishes
                    // `t.0` from `t.1`: `let (p, q) = t;` publishes two names that do NOT overlap
                    // each other, while destructuring one tuple TWICE publishes two that do -- and
                    // that second shape dispatched a GEMM over an output aliasing its own A
                    // operand, in a function no call site aliases anything in.
                    ast::PatKind::Tuple(pats) => {
                        note_tuple_pattern(&mut out, pats, e, sema);
                        return;
                    }
                    // Binds nothing, so there is nothing to republish.
                    ast::PatKind::Wildcard | ast::PatKind::Unit => return,
                    // The remaining arms are `Int`, `Char`, `Bool`, `Path`, `Range`, `Or` and
                    // `Variant`. All seven are REFUTABLE, so a `let` should not be spelled with
                    // one -- but two of them (`Or`, `Variant`) do bind names where they are legal,
                    // `wukong_sema::bind_pattern` binds them, and a future binding pattern (struct
                    // destructuring, say) would land here too. This pass must never answer "its own
                    // storage" for a name it has not seen bound, so the arm taints whenever the
                    // initializer is a buffer-shaped place rather than falling through silently the
                    // way the `let else` this replaced did.
                    _ => {
                        out.opaque_republish |=
                            !matches!(republish_of(e, sema), Republish::Nothing)
                                && sema.types.get(&e.id).is_some_and(ty_is_buffer_param);
                        return;
                    }
                };
                // A `let` binds a view when the BINDING borrows: written down (`let s: []f32 = a;`)
                // or inferred from the initializer.
                let annotated_view = ty.as_ref().is_some_and(|t| {
                    matches!(
                        &t.kind,
                        ast::TypeKind::Slice(_)
                            | ast::TypeKind::Pointer { .. }
                            | ast::TypeKind::Ref { .. }
                    )
                });
                let inferred_view = sema.types.get(&e.id).is_some_and(ty_is_borrowing);
                if annotated_view || inferred_view {
                    match republish_of(e, sema) {
                        // The map serves both tests: a CALL SITE spelled with the view's name
                        // resolves back onto the storage it views, and so does an OPERAND spelled
                        // with it.
                        Republish::At(p) => {
                            let rebinding = out.republished.note_view(*name, p);
                            out.opaque_republish |= rebinding;
                        }
                        // Storage the initializer built itself -- an aggregate literal, or an
                        // aggregate return arriving through the sret ABI. Nobody else holds it, so
                        // the binding publishes no second name and there is nothing to record.
                        Republish::Nothing => {}
                        // `Unknown` means the OPPOSITE thing HERE that it means on the argument
                        // path, and that asymmetry was a live wrong answer. At a call site
                        // `Unknown` aliases everything, which is conservative. In a view binding it
                        // used to mean "record nothing and taint nothing": the `if let
                        // Origin::At(p)` this replaced just fell out of the arm, so the name never
                        // entered `view_names` either and silently became its own root.
                        //
                        // `let v: []f32 = if flag { c } else { c };` IS a view. `let_is_slice` is
                        // tested BEFORE the aggregate deep-copy arm, so the binding goes to
                        // `lower_slice_from_source` and is a genuine fat-pointer view of whichever
                        // arm the merge produced. But an `If` is not a place and its type is a
                        // buffer param, so `origin_of` falls to its `_` arm and answers `Unknown` --
                        // and `fn mm(b, mut c, flag) { let v: []f32 = if flag { c } else { c };
                        // <nest over v and b writing c> }` then resolved {v, b, c} onto three
                        // distinct roots, left `mm` alias free and dispatched `wukong_sgemm` with A
                        // and C one buffer. Measured on the binary before this arm existed:
                        // `10 14 5 8 ...` where the scalar nest says `10 14 23 32 ...`, on interp
                        // and native at -O0 and -O2, in all four spellings -- `if`, an UN-annotated
                        // `match` whose arms are the slice (so `inferred_view` fires alone), a bare
                        // `{ c }` block, and `loop { break c; }`.
                        //
                        // A CALL is exempt, and only a call: it is the one `Unknown` producer whose
                        // taint would cost the corpus, because `alloc_f32(n)` is one
                        // (`# Known limits`, the CALL bullet). Everything else taints, and the
                        // four-root census measures that at zero programs changed.
                        Republish::Taint => out.opaque_republish = true,
                    }
                    return;
                }
                // `let l2: Layer = <place>;` -- copying a whole struct is NOT covered by the
                // "value semantics" paragraph in the module doc. `emit_copy` walks the
                // fields and sends a `[]T` one to `copy_slice_fat_ptr`, which moves the 16-byte
                // view and not one element, so every borrowing field of the source gets a second
                // operand name. Measured: `let l: L = L { w: sa }; let l2: L = l;` over a nest
                // spelling `l2.w` dispatched `wukong_sgemm` with A and C one buffer and printed the
                // out-of-place product on both backends at -O0 and -O2. Recorded per borrowing
                // field, exactly like the literal below, so `l2.w -> l.w -> sa -> c` is one chain
                // the resolver already walks.
                if let Some(Ty::Named(sname)) = sema.types.get(&e.id) {
                    match republish_of(e, sema) {
                        Republish::At(p) => {
                            let mut publishes = false;
                            for (fname, fty) in struct_fields(*sname, sema) {
                                if !ty_is_borrowing(fty) {
                                    continue;
                                }
                                let mut q = p.clone();
                                q.proj.push(Proj::Field(*fname));
                                out.republished.note_field_view(*name, *fname, q);
                                publishes = true;
                            }
                            if publishes {
                                let rebinding = out.republished.note_struct_local(*name);
                                out.opaque_republish |= rebinding;
                            }
                        }
                        Republish::Nothing => {}
                        // Only a struct with a BORROWING field can republish anything through a
                        // copy, so a value-only struct copied out of an unclassifiable expression
                        // still costs nothing. Measured before this arm existed:
                        // `let l2: L = if flag { l } else { l };` over a nest spelling `l2.w`
                        // dispatched `wukong_sgemm` with A and C one buffer.
                        Republish::Taint => {
                            out.opaque_republish |= struct_fields(*sname, sema)
                                .iter()
                                .any(|(_, t)| ty_is_borrowing(t));
                        }
                    }
                }
                // An aggregate LITERAL deep-copies every field EXCEPT the borrowing ones: a
                // `[]T`/`*T`/`&T` field keeps the source's base pointer, so `l.wq` becomes a second
                // name for it. A functional update (`..other`) publishes fields this pass cannot
                // enumerate, and a synthesized literal has no id to attribute, so both fall through
                // to the blanket taint in the expression arm.
                let ExprKind::StructLit { fields, rest, .. } = &e.kind else {
                    return;
                };
                if rest.is_some() || e.id == ast::NodeId::DUMMY {
                    return;
                }
                attributed.insert(e.id);
                let lit_struct = match sema.types.get(&e.id) {
                    Some(Ty::Named(s)) => Some(*s),
                    _ => None,
                };
                let mut publishes = false;
                for fi in fields {
                    // The FIELD's declared type is half the question: `L { w: c }` with
                    // `w: []f32` and `c: [f32; 16]` unsizes, so `l.w` views `c` even though `c`'s
                    // own type borrows nothing. See `init_publishes_view`.
                    let fty = struct_field_ty(lit_struct, fi.name.sym, sema);
                    if !init_publishes_view(fty, &fi.value, sema) {
                        continue;
                    }
                    match republish_of(&fi.value, sema) {
                        Republish::At(p) => {
                            out.republished.note_field_view(*name, fi.name.sym, p);
                            publishes = true;
                        }
                        Republish::Nothing => {}
                        // `let l: L = L { w: if flag { c } else { c } };` -- the field really does
                        // view `c`, and `attributed` has already told the generic `StructLit`
                        // expression arm below not to look at this literal, so nothing else would
                        // catch it. Measured: it dispatched over a nest spelling `l.w`.
                        Republish::Taint => out.opaque_republish = true,
                    }
                }
                // Once per `let`, not once per field: one literal names one local, and registering
                // it per field would report the second field of `Layer { wq: swq, wk: swk }` as a
                // rebinding and cost `block_d_struct.wk` every kernel it has.
                if publishes {
                    let rebinding = out.republished.note_struct_local(*name);
                    out.opaque_republish |= rebinding;
                }
            }
            // `for w in ws { ... }` -- the element binding is a VIEW of `ws[...]`, not a copy,
            // for every element type Wukong does not pass by value. See
            // [`note_for_each_pattern`]; `scan_body` matched only `Let` and `Assign` here, so a
            // `for` that binds a name was a republishing spelling this pass had no word for.
            StmtKind::For {
                pat,
                iter: ForIter::Expr(e),
                ..
            } => note_for_each_pattern(&mut out, pat, e, sema),
            // `p = &x;` / `s = a;` / `l.w = c;` — a place stored into a location that already
            // exists, where the store publishes a view. WHICH name now reaches that storage is a
            // flow-sensitive question this pass does not answer, so it taints outright rather than
            // guessing. The TARGET's declared type is what catches `l.w = c;`, whose right-hand
            // side is an array and would otherwise read as a deep copy.
            StmtKind::Assign { target, value, .. } => {
                out.opaque_republish |= republishing_value(sema.types.get(&target.id), value, sema);
            }
            _ => {}
        },
        Node::Expr(e) => match &e.kind {
            // The base of every index is a candidate kernel operand — see [`Operand`].
            ExprKind::Index { base, .. } => {
                if let Some(o) = operand_of(base) {
                    out.operands.insert(o);
                }
            }
            // An aggregate literal that takes a borrowing place as a field value hands that storage
            // a second name. The `let`-bound struct literal is recorded precisely above; every
            // other spelling names its fields in a way no `buffer_base` can reproduce, so it taints.
            ExprKind::StructLit { fields, rest, .. } => {
                if attributed.contains(&e.id) {
                    return;
                }
                let sname = match sema.types.get(&e.id) {
                    Some(Ty::Named(s)) => Some(*s),
                    _ => None,
                };
                out.opaque_republish |= fields.iter().any(|fi| {
                    republishing_value(struct_field_ty(sname, fi.name.sym, sema), &fi.value, sema)
                }) || rest
                    .as_deref()
                    .is_some_and(|r| republishing_value(None, r, sema));
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                let agg = sema.types.get(&e.id);
                out.opaque_republish |= xs
                    .iter()
                    .enumerate()
                    .any(|(i, x)| republishing_value(aggregate_elem_ty(agg, i), x, sema));
            }
            ExprKind::ArrayRepeat { value, .. } => {
                let agg = sema.types.get(&e.id);
                out.opaque_republish |= republishing_value(aggregate_elem_ty(agg, 0), value, sema);
            }
            // A `match` arm PATTERN binds a name onto storage INSIDE the scrutinee -- an enum
            // payload (`E::V(p)`), a tuple element -- and when that storage borrows, the binding is
            // a second name for whatever the payload was built from. `origin_of` calls the variant
            // constructor `E::V(c)` `Fresh` (its type is the enum, which does not borrow), so there
            // is no chain from `p` back to `c` to record and the coarse answer is the only honest
            // one. Measured before this arm existed: `enum E { V([]f32) }` with
            // `let e: E = E::V(c); match e { E::V(p) => <nest over p and b writing c> }` dispatched
            // `wukong_sgemm` with A and C one buffer and printed the out-of-place product on interp
            // and native alike. The module doc used to state this as a hole that "kernel operands do
            // not currently arrive [through]"; they do.
            //
            // Gated on the scrutinee's DECLARED type so the ordinary `match` over a scalar or a
            // C-style enum -- every `match` in the corpus -- costs nothing: only a type that can
            // carry a `[]T`/`*T`/`&T` leaf, destructured by a pattern that actually binds, taints.
            ExprKind::Match { scrutinee, arms } => {
                if arms.iter().any(|a| pattern_binds(&a.pat))
                    && sema
                        .types
                        .get(&scrutinee.id)
                        .is_some_and(|t| ty_contains_borrowing(t, sema, 0))
                {
                    out.opaque_republish = true;
                }
            }
            ExprKind::Call { callee, args, .. } => {
                let ExprKind::Path(p) = &callee.kind else {
                    return;
                };
                if !p.is_single() {
                    return;
                }
                out.sites.push(Site {
                    callee: p.first().sym,
                    args: args.iter().map(|a| origin_of(a, sema)).collect(),
                });
            }
            _ => {}
        },
    });
    out
}

/// Does this pattern bind at least one NAME? A literal, a range, a unit-variant path and `_` test
/// the scrutinee's value and bind nothing, so they republish nothing.
///
/// Wildcard-free over `PatKind`, like the two `match`es in the walk below, so a new pattern form is
/// a compile error here rather than a silent "binds nothing".
fn pattern_binds(p: &ast::Pattern) -> bool {
    match &p.kind {
        ast::PatKind::Ident(_) => true,
        ast::PatKind::Tuple(subs) | ast::PatKind::Or(subs) => subs.iter().any(pattern_binds),
        ast::PatKind::Variant { fields, .. } => match fields {
            ast::VariantPat::Tuple(subs) => subs.iter().any(pattern_binds),
            ast::VariantPat::Struct(fs) => fs.iter().any(|f| pattern_binds(&f.pat)),
        },
        ast::PatKind::Range { lo, hi, .. } => pattern_binds(lo) || pattern_binds(hi),
        ast::PatKind::Struct { fields, .. } => fields.iter().any(|f| pattern_binds(&f.pat)),
        ast::PatKind::Binding { .. } => true,
        ast::PatKind::Slice { prefix, suffix, .. } => {
            prefix.iter().chain(suffix).any(pattern_binds)
        }
        ast::PatKind::Wildcard
        | ast::PatKind::Unit
        | ast::PatKind::Int { .. }
        | ast::PatKind::Char(_)
        | ast::PatKind::Bool(_)
        | ast::PatKind::Path(_) => false,
    }
}

/// Can a value of this type hold a `[]T`/`*T`/`&T` leaf anywhere inside it — i.e. can destructuring
/// it hand one of its parts a second name for storage someone else owns?
///
/// [`ty_is_borrowing`] answers that for the value ITSELF; this walks into arrays, tuples, struct
/// fields and enum payloads. The depth cap answers "cannot tell", which here means "taints", so it
/// errs in the safe direction; Wukong types cannot be recursive by value, so it is insurance rather
/// than a real bound.
fn ty_contains_borrowing(t: &Ty, sema: &SemaResult, depth: u32) -> bool {
    if depth > 8 {
        return true;
    }
    match t {
        Ty::Slice(_) | Ty::Ptr { .. } | Ty::Ref { .. } => true,
        Ty::Tuple(elems) => elems
            .iter()
            .any(|e| ty_contains_borrowing(e, sema, depth + 1)),
        Ty::Array { elem, .. } => ty_contains_borrowing(elem, sema, depth + 1),
        Ty::Named(n) => match sema.defs.lookup(*n).map(|d| &d.kind) {
            Some(DefKind::Struct(fields)) => fields
                .iter()
                .any(|(_, ft)| ty_contains_borrowing(ft, sema, depth + 1)),
            Some(DefKind::Enum(variants)) => variants.iter().any(|v| {
                v.payload
                    .field_tys()
                    .iter()
                    .any(|ft| ty_contains_borrowing(ft, sema, depth + 1))
            }),
            _ => false,
        },
        _ => false,
    }
}

/// The declared fields of a named STRUCT, in declaration order. Empty for an enum, for a name
/// with no resolved definition, and for anything else — all three mean "publishes no field this
/// pass can attribute", which is what the caller does with an empty list.
fn struct_fields(name: Symbol, sema: &SemaResult) -> &[(Symbol, Ty)] {
    match sema.defs.lookup(name).map(|d| &d.kind) {
        Some(DefKind::Struct(fields)) => fields,
        _ => &[],
    }
}

/// `let (p, q) = <init>;` — record each destructured BORROWING element as a view of its tuple
/// field.
///
/// The value elements are skipped on purpose: `emit_copy` deep-copies an array or a scalar, so such
/// a binding owns its storage and recording it would cost a dispatch for nothing. A borrowing
/// element copies as a fat pointer, so it does not.
///
/// `Proj::Tuple` makes this PRECISE rather than a taint: `places_may_overlap` separates `t.0` from
/// `t.1`, so the ordinary `let (p, q) = t;` publishes two names that cannot overlap each other,
/// while `let (p, q) = t; let (r, u) = t;` publishes `p` and `r` onto one place — which is what
/// dispatched a GEMM over an output aliasing its own A operand with no call site aliasing anything.
fn note_tuple_pattern(out: &mut BodyScan, pats: &[ast::Pattern], init: &Expr, sema: &SemaResult) {
    let p = match republish_of(init, sema) {
        Republish::At(p) => p,
        // A tuple LITERAL is `Fresh` and is tainted by the expression arm if it carries a
        // republishing element; a call returning a tuple is `Fresh` too and is a stated hole.
        // Neither is a place, so neither has a field to name.
        Republish::Nothing => return,
        // `let (p, q) = if flag { t } else { t };` -- destructuring something this pass cannot
        // classify. Gated on the type actually being able to carry a borrowing leaf, so a
        // `(i32, i32)` destructured out of an `if` still costs nothing. Measured before this arm
        // existed: it dispatched `wukong_sgemm` over `p` and a second destructuring of the same
        // tuple.
        Republish::Taint => {
            out.opaque_republish |= sema
                .types
                .get(&init.id)
                .is_some_and(|t| ty_contains_borrowing(t, sema, 0));
            return;
        }
    };
    let Some(Ty::Tuple(elems)) = sema.types.get(&init.id) else {
        // A destructuring whose element types this pass cannot see: it cannot tell a fat-pointer
        // copy from a deep one, so it takes the coarse answer.
        out.opaque_republish = true;
        return;
    };
    for (i, sub) in pats.iter().enumerate() {
        let ast::PatKind::Ident(n) = &sub.kind else {
            // A nested or wildcard sub-pattern. A wildcard binds nothing; anything else binds a
            // name this pass cannot attribute to one tuple field, so it taints if that field could
            // be a buffer.
            if !matches!(sub.kind, ast::PatKind::Wildcard) {
                out.opaque_republish |= elems.get(i).is_some_and(ty_is_buffer_param);
            }
            continue;
        };
        if !elems.get(i).is_some_and(ty_is_borrowing) {
            continue;
        }
        let mut q = p.clone();
        q.proj.push(Proj::Tuple(i as u32));
        let rebinding = out.republished.note_view(*n, q);
        out.opaque_republish |= rebinding;
    }
}

/// `for w in ws { … }` — record the element binding as a view of `ws[…]`.
///
/// `lower_for_array` binds an **aggregate** element BY POINTER into the iterated buffer
/// (`bind_slice(name, elem_ptr, …)`, "exactly as the `a[i]` aggregate read arm does"), and a `[]T`
/// element arrives there too, because a slice's `elem_mir` is its fat pointer, whose MIR shape is an
/// `Array`. Only a SCALAR element takes the other branch — "loads into a fresh slot … the faithful
/// `let x = a[i]`" — and owns its storage. [`ty_is_buffer_param`] is the AST-level spelling of that
/// split, and it is the WIDE side of it: it additionally covers a `*T`/`&T` element, which the
/// scalar branch copies by VALUE and which is a second name for the pointee all the same.
///
/// `Proj::Index` is deliberately index-blind ([`Proj`]), and that is what makes this precise rather
/// than a taint: ONE loop over `ws` publishes one name, nothing else resolves onto `ws[…]`, and the
/// kernel stands — `tests/run/for_each_array_of_slices.wk` is exactly that shape. Iterating one
/// array TWICE publishes two names that DO overlap, and that shape dispatched `wukong_sgemm` with A
/// and C one buffer in a function no call site aliases anything in:
/// `fn mm(ws: [[]f32; 1], b) { for w in ws { for u in ws { <nest over w and b writing u> } } }`
/// printed the out-of-place product `10 14 5 8 …` where the scalar nest says `10 14 23 32 …`, on
/// interp and native at -O0 and -O2. It is the destructuring `let` one vocabulary away: `let (p, q)
/// = t; let (r, u) = t;` publishes two names for `t.0`, and two `for` loops over one array publish
/// two names for `ws[…]`.
fn note_for_each_pattern(out: &mut BodyScan, pat: &ast::Pattern, iter: &Expr, sema: &SemaResult) {
    let (Some(Ty::Array { elem, .. }) | Some(Ty::Slice(elem))) = sema.types.get(&iter.id) else {
        // `lower_for_array` accepts only those two and declines everything else to the
        // `unsupported` diagnostic, so no binding is produced today — but this pass must not rest
        // on that, because a widening there would make the hole silent again.
        out.opaque_republish |= pattern_binds(pat);
        return;
    };
    if !ty_is_buffer_param(elem) {
        // A scalar element is a copy: the loop variable is an ordinary mutable local.
        return;
    }
    let ast::PatKind::Ident(name) = &pat.kind else {
        // `_` binds nothing; anything else binds a name this pass cannot attribute to one element.
        out.opaque_republish |= pattern_binds(pat);
        return;
    };
    let mut p = match republish_of(iter, sema) {
        Republish::At(p) => p,
        // Iterating something this pass cannot name. `Nothing` does NOT mean the ELEMENTS are
        // fresh — that is the same "a value carrying borrowed storage is classified by the
        // expression that produced it" root the `# Known limits` holes are named after — so both
        // remaining arms take the coarse answer, which is the honest one for an element type that
        // reached this far.
        Republish::Nothing | Republish::Taint => {
            out.opaque_republish = true;
            return;
        }
    };
    p.proj.push(Proj::Index);
    let rebinding = out.republished.note_view(*name, p);
    out.opaque_republish |= rebinding;
}

/// Can two DIFFERENT kernel-operand names in this body reach the same storage?
///
/// Each operand is resolved through both view maps and then compared with the same
/// [`places_may_overlap`] the call-site test uses. Two operands that land on one place — `sa` and
/// `a` after `let mut sa: []f32 = a;`, or `l.wq` and `wq` after `Layer { wq: swq }` — are the
/// aliased call in disguise: no call site aliases anything, and `main`, having no parameters, is
/// trivially alias free under the call-site fact alone.
///
/// Only names that actually appear as an operand count. A view whose source is never spelled again
/// — the ten `let swq: []f32 = wq;` bindings a struct-held weight block opens with — leaves one
/// name reaching that buffer and nothing to confuse; tainting on the view's mere presence is what
/// silently cost `block_d_struct.wk` its nine GEMMs.
///
/// Grouping by resolved ROOT keeps this linear in practice: different roots cannot overlap, and a
/// function's operands almost always have different roots, so the quadratic term runs only inside a
/// group.
fn body_names_one_buffer_twice(scan: &BodyScan) -> bool {
    let mut by_root: HashMap<Symbol, Vec<Place>> = HashMap::default();
    for op in &scan.operands {
        let Origin::At(p) = scan.republished.resolve(Origin::At(op.place())) else {
            return true; // an unresolvable chain is "cannot tell", which is "may overlap"
        };
        let group = by_root.entry(p.base).or_default();
        if group.iter().any(|q| places_may_overlap(q, &p)) {
            return true;
        }
        group.push(p);
    }
    false
}

/// Is `e` a value that, stored anywhere, gives one buffer a second name? The gate on every coarse
/// taint: the `Assign` statement and the four aggregate-literal expression arms.
///
/// Its type must borrow, and it must be able to reach storage someone else holds. A PLACE (`a`,
/// `l.w`, `t.0`) does. So does a value this pass cannot classify: `Republish::Taint` means *cannot
/// tell*, and this predicate used to read it as "no" — `s = if flag { c } else { c };` and
/// `let ws: [[]f32; 1] = [if flag { c } else { c }];` both dispatched a GEMM over their own output
/// on interp and native at -O0 and -O2, measured.
///
/// `alloc_f32(n)` is the one exemption and it is not the `Fresh` one an earlier version of this
/// comment claimed: its return type is `[]f32`, which borrows, so [`origin_of`] classifies it
/// `Unknown` through the CALL arm (the [`Origin::Fresh`] doc says exactly this). [`republish_of`]
/// maps a call to `Nothing`, which is why the whole `alloc_*` corpus keeps its kernels.
/// `dst` is the DECLARED type of the location being initialized (a struct field, an array or tuple
/// element, an assignment target), or `None` where the caller genuinely has no destination.
fn republishing_value(dst: Option<&Ty>, e: &Expr, sema: &SemaResult) -> bool {
    init_publishes_view(dst, e, sema) && !matches!(republish_of(e, sema), Republish::Nothing)
}

/// **THE test for "this initializer produces a VIEW of its source rather than a copy."** Every site
/// in this pass that asks the question asks it here, and no site may ask it any other way.
///
/// Both types matter, and reading only one of them is a wrong answer:
///
/// * The VALUE's type borrows (`let l = Layer { wq: swq };` with `swq: []f32`) — the fat pointer is
///   copied, so the field is a second name for what `swq` views.
/// * The DESTINATION's type borrows while the value's does not. This is sema's one implicit
///   conversion, `[T; N]` -> `[]T` unsizing (`wukong_sema::compatible`), and after it
///   `FnLowerer::init_field` sends the initializer to `lower_slice_from_source`, which stores the
///   ARRAY's base pointer into the field — "the slice then views the source's storage — no element
///   copy". So `struct L { w: []f32 }` with `let l = L { w: c };`, `c: [f32; 16]`, makes `l.w` a
///   second name for `c` even though `c`'s own type is an array and borrows nothing.
///
/// The second case is new: before `init_field` had a slice arm it reached `emit_copy`, which read a
/// fat pointer out of `c`'s float bytes, so the program died (`expected a pointer` on interp, exit
/// 139 natively) rather than aliasing anything. Making that spelling WORK is what makes its alias
/// real, which is why this predicate had to change in the same commit — a fix that only corrected
/// the lowering would have handed `wukong_sgemm` an output aliasing its own operand, silently and
/// identically on both backends at every `-O` level.
///
/// `None` for `dst` means the caller has no declared destination to offer and the value's type
/// alone decides. Prefer passing the destination: it is what makes the unsizing case visible.
fn init_publishes_view(dst: Option<&Ty>, e: &Expr, sema: &SemaResult) -> bool {
    dst.is_some_and(ty_is_borrowing) || sema.types.get(&e.id).is_some_and(ty_is_borrowing)
}

/// The declared type of field `fname` of struct `sname`, for [`init_publishes_view`]'s `dst`.
/// `None` when the name is not a struct or has no such field, which leaves the value's own type to
/// decide — the conservative direction only in the sense that it is the pre-unsizing answer, so a
/// caller that can resolve the struct must.
fn struct_field_ty(sname: Option<Symbol>, fname: Symbol, sema: &SemaResult) -> Option<&Ty> {
    struct_fields(sname?, sema)
        .iter()
        .find(|(n, _)| *n == fname)
        .map(|(_, t)| t)
}

/// The declared ELEMENT type of an array/tuple literal's slot `i`, for [`init_publishes_view`]'s
/// `dst`. An array yields its one element type for every slot; a tuple yields the `i`-th.
fn aggregate_elem_ty(agg: Option<&Ty>, i: usize) -> Option<&Ty> {
    match agg? {
        Ty::Array { elem, .. } => Some(elem),
        Ty::Tuple(elems) => elems.get(i),
        _ => None,
    }
}

/// Does this call site bind two distinct buffer parameters of the callee to overlapping storage?
fn site_aliases(site: &Site, callee: &FnInfo, caller: &FnInfo, caller_aliased: bool) -> bool {
    for (n, &(i, _)) in callee.buffer_params.iter().enumerate() {
        for &(j, _) in &callee.buffer_params[n + 1..] {
            let (Some(ai), Some(aj)) = (site.args.get(i), site.args.get(j)) else {
                continue;
            };
            if origins_may_overlap(ai, aj) {
                return true;
            }
            // The caller's own parameters may already be one array (established on an earlier
            // round). Two arguments rooted at two of them therefore forward that aliasing.
            if caller_aliased && both_root_at_buffer_params(ai, aj, caller) {
                return true;
            }
        }
    }
    false
}

fn both_root_at_buffer_params(a: &Origin, b: &Origin, caller: &FnInfo) -> bool {
    let root = |o: &Origin| match o {
        Origin::At(p) => Some(p.base),
        _ => None,
    };
    let (Some(ra), Some(rb)) = (root(a), root(b)) else {
        return false;
    };
    let is_param = |s: Symbol| caller.buffer_params.iter().any(|(_, n)| *n == s);
    is_param(ra) && is_param(rb)
}

// ---- AST walk -------------------------------------------------------------------------------
//
// ONE traversal, both node kinds, no wildcard arms over `StmtKind`/`ExprKind`. The predecessor of
// this walk had two mutually recursive halves — a statement walker and an expression walker — and a
// block's `tail` reached only the first of them, so a call written without its trailing semicolon
// was invisible to the site collector. Everything that carries a value out of a block (a `tail`, a
// non-block `match` arm body, a `match` arm GUARD) is an expression position, and every expression
// position has to reach the same callback a statement one does, or the fact this module computes is
// a fact about formatting.

/// One node of the walk. `Stmt` and `Expr` are both delivered because the two consumers want
/// different halves: view bindings are statements, call sites are expressions.
enum Node<'a> {
    Stmt(&'a Stmt),
    Expr(&'a Expr),
}

fn walk_block(b: &Block, f: &mut impl FnMut(Node<'_>)) {
    for s in &b.stmts {
        walk_stmt(s, f);
    }
    if let Some(t) = &b.tail {
        walk_expr(t, f);
    }
}

fn walk_stmt(s: &Stmt, f: &mut impl FnMut(Node<'_>)) {
    f(Node::Stmt(s));
    match &s.kind {
        StmtKind::Let { init: opt, .. } | StmtKind::Return(opt) | StmtKind::Break(_, opt) => {
            if let Some(e) = opt {
                walk_expr(e, f);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            walk_expr(target, f);
            walk_expr(value, f);
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => walk_expr(e, f),
        StmtKind::Continue(_) => {}
        StmtKind::While { cond, body, .. } => {
            walk_expr(cond, f);
            walk_block(body, f);
        }
        StmtKind::For { iter, body, .. } => {
            match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    walk_expr(start, f);
                    if let Some(e) = end {
                        walk_expr(e, f);
                    }
                    if let Some(e) = step {
                        walk_expr(e, f);
                    }
                }
                ForIter::Expr(e) => walk_expr(e, f),
            }
            walk_block(body, f);
        }
    }
}

fn walk_expr(e: &Expr, f: &mut impl FnMut(Node<'_>)) {
    f(Node::Expr(e));
    match &e.kind {
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => walk_expr(expr, f),
        ExprKind::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, f);
            walk_expr(rhs, f);
        }
        ExprKind::Call { callee, args, .. } => {
            walk_expr(callee, f);
            for a in args {
                walk_expr(a, f);
            }
        }
        ExprKind::Index { base, indices } => {
            walk_expr(base, f);
            for i in indices {
                walk_expr(i, f);
            }
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => walk_expr(base, f),
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                walk_expr(x, f);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            walk_expr(value, f);
            walk_expr(count, f);
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for fi in fields {
                walk_expr(&fi.value, f);
            }
            if let Some(r) = rest {
                walk_expr(r, f);
            }
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => walk_block(b, f),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_expr(cond, f);
            walk_block(then_branch, f);
            if let Some(x) = else_branch {
                walk_expr(x, f);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            walk_expr(scrutinee, f);
            for a in arms {
                if let Some(g) = &a.guard {
                    walk_expr(g, f);
                }
                walk_expr(&a.body, f);
            }
        }
        // Leaves. Spelled out rather than `_` so a new `ExprKind` is a compile error here.
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_)
        | ExprKind::Closure { .. } => {}
    }
}
