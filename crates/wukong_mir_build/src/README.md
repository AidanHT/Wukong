# `src/` module index

| file | contents |
|---|---|
| `lib.rs` | Entire crate (~26k lines): `lower_program`, monomorphization, `@parallel` outliners, every `match_*` / `emit_*` recognizer, `FnLowerer`, intrinsics, vectorizer, `const_usize_depth`, inline tests |
| `canon.rs` | Semantics-preserving AST respelling so recognizers see canonical loop nests |
| `tindex.rs` | `a[i, j]` on static tensors → single linear index before recognizers run |

No `tests/` directory — tests are `#[cfg(test)] mod tests` at the bottom of `lib.rs`.

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
