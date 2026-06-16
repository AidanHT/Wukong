# mercury_runtime

The minimal runtime Mercury programs link against: a bump arena allocator and a (currently sequential) `parallel_for`. The interpreter backend calls these in-process; native (LLVM) builds link the same logic as a static library.

## Layout
- `src/lib.rs` — the entire crate: `Arena` + `parallel_for`, plus unit tests.

## Key types & entry points
- `Arena` (`src/lib.rs`) — bump allocator over an owned `Vec<u8>`. API: `with_capacity`, `alloc(size, align)`, `slice_mut(offset, len)`, `reset`, `used`, `capacity`.
- `Arena::alloc` — rounds `offset` up to `align` (power of two; `debug_assert`ed), bumps, returns the **byte offset** (not a pointer) or `None` if exhausted. `checked_add` guards size overflow. On exhaustion `self.offset` is left unchanged.
- `parallel_for(lo, hi, step, body)` (`src/lib.rs`) — runs `body(i)` for `i = lo, lo+step, ...` while `i < hi`. All params are `i64`. Despite the name it is sequential and deterministic today.

## Connects to
Upstream (depends on): nothing — `Cargo.toml` has an empty `[dependencies]`, no `mercury_` crates. Downstream (consumers): the interpreter backend (in-process) and native LLVM-compiled programs (link as a static lib).

## Gotchas
- `alloc` returns a `usize` offset into the backing buffer, not a raw pointer; pair it with `slice_mut` to read/write bytes.
- `slice_mut(offset, len)` does **not** validate that the region was previously allocated — it indexes `buf[offset..offset+len]` and panics on out-of-bounds. Pass back exactly what `alloc` returned.
- `reset` frees everything at once (offset -> 0); no per-object free, and storage is reused, never grown (`capacity` stays constant across resets).
- `parallel_for` is named for an eventual work-stealing pool but is a plain sequential loop now. `step <= 0` does nothing (early return); a `step` larger than the range still runs once (the `lo` iteration). It is the deterministic reference order — any future parallel impl must preserve observable results for associative reductions.
