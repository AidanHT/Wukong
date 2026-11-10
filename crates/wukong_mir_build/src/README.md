# `src/` module index

| file | contents |
|---|---|
| `lib.rs` | Entire crate (~26k lines): `lower_program`, monomorphization, `@parallel` outliners, every `match_*` / `emit_*` recognizer, `FnLowerer`, intrinsics, vectorizer, `const_usize_depth`, inline tests |
| `canon.rs` | Semantics-preserving AST respelling so recognizers see canonical loop nests |
| `tindex.rs` | `a[i, j]` on static tensors → single linear index before recognizers run |
| `builtins.rs` | Lowering of the named kernel builtins: runtime extent checks, output-overlap guard (scratch + copy-out), empty-contraction fill, `sdpa` |
| `methods.rs` | Method-call desugaring (before every other pass), trait-bound dispatch per instance, function-typed parameters as pseudo-generics, the alias-fact taint for indirect calls |
| `noalias.rs` | `@noalias` → MIR `ParamAttrs` (`record_param`); the name-level disjointness proof the recognizers and the `@parallel` region outliner use (`promised_disjoint`, `could_share_storage`) |
| `par_grain.rs` | `@parallel(grain = G)`: chunk-count emission at the `wukong_parallel_for` call and the chunk-span -> iteration mapping in the outlined body |
| `spell.rs` | Natural spellings of the activations (P12.10): the store value read as the tree of operations it lowers to, canonicalized and matched against the table of `wukong_vmath_f32` kernels (codes 36..=58 and the existing ones it IS); `spelling_census_tests.rs` is its census |

No `tests/` directory — tests are `#[cfg(test)] mod tests` at the bottom of `lib.rs`.

**Recognition is switchable, and every recognizer arm must live inside one of the four gated blocks.**
`LowerOptions { recognize }` (from `WUKONG_NO_RECOGNIZE`, or cleared per function by `@generic`) rides
on `GemmSyms` — the struct already threaded to every recognizer and probe — and is read at exactly four
places: `lower_program`'s whole-function interceptions, `FnLowerer::lower_for`'s cascade,
`lower_block`'s statement windows, and the elementwise probes opening
`try_vectorize_for` / `try_vectorize_ranged`. An arm added outside them is a kernel nothing can turn
off. The autovectorizer, `try_fuse_run`, `@parallel` outlining and the explicit builtins
(`alloc_*`, `read_*`, `sdpa`, `now_ns`) are deliberately never gated.

Rough `lib.rs` regions (line numbers drift — grep the symbol):

| lines (approx) | region |
|---|---|
| top → ~680 | `Mono`, `collect_mono`, string interning |
| ~681–1236 | `lower_program`, `GemmSyms`, interceptor cascade |
| ~1237–1870 | `@parallel` whole-function outliner |
| ~2500–7660 | `FnLowerer` core, norm/loss recognizers |
| ~7658–9990 | `kernel_base_ptr`, kernel emitters, reduction recognizers |
| ~10012–10520 | `lower_for` recognizer cascade |
| ~10519–15170 | mid-function `@parallel`, vmath/vectorizer |
| ~15170–19070 | control flow, match, calls, intrinsics |
| ~19580+ | matmul family and free-standing `*_fn` probes |
