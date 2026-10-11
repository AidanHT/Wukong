# `wukong_interp` source layout

| File | Role |
| --- | --- |
| `lib.rs` | `Value`, the public run / kernel-entry functions, `Accelerator`, `CalleeTable`, `SlotReuse`, `Interp` (frame loop, `eval`, `call_intrinsic`), the semantic helpers, `mod tests` |
| `intrinsics.rs` | The one intrinsic name table: `Intrinsic` ids with `name` / `from_name`, and its round-trip tests |
| `marshal.rs` | Kernel-operand marshalling: `LiveBlocks` (the live-allocation table), the `Arena` views, checked extents, the overlap contract, `elementwise_live`, the f32 scratch pool |
| `text_io.rs` / `ml_io.rs` / `decode.rs` | The text and process builtins, the ML I/O and sampling builtins, the decode builtins (KV-cache attention, half / quantized GEMV) |
| `*_tests.rs` | Focused unit-test modules (marshalling, extents, transfers, aliasing contracts, parallel twins, scratch, slot reuse, the SIMD arena) |

## Calls and intrinsics

A callee symbol is resolved once per run by `CalleeTable`: to a Wukong function, an `extern "C"`
declaration, a dense `Intrinsic` id (`intrinsics.rs` lists every name the interpreter implements,
once), or `Unknown`. `call_intrinsic` matches on the id; the text, ML and decode modules dispatch
the ids they own. Adding an intrinsic is one table line plus its arm. The by-name
`Interp::intrinsic` exists for the unit tests only.

Calls are stackless (`run_function` keeps an explicit frame stack). Register files are pooled and
not refilled on entry — verified MIR never reads a register before writing it — and a returning
frame resets its vector-typed registers, so a pooled file never carries a SIMD handle.

## Kernel operands

Every recognized-kernel arm validates each operand before reserving scratch: the extent product is
checked, the range must lie in the arena, and then inside ONE live allocation (`LiveBlocks`, kept by
`SlotReuse` at alloca, heap alloc/free, scratch alloc/release, frame release, statics and
kernel-entry buffers). A violation is `<kernel>: operand <k> extends past its allocation`, `k` the
argument index. A piece of an operand read at a program-computed offset with no whole extent to
check (an embedding weight row) must lie in the allocation holding the operand's own pointer
(`LiveBlocks::check_within`). An empty range is never checked: a zero-trip vector-kernel call
touches no stream, so a stream base past the arena's end is not an error. Helpers take an `arena!(self)` / `arena_mut!(self)` view so the interpreter's other
fields stay borrowable. A hand-built `Interp` (unit tests) keeps an untracked table, bounded by the
arena. Optional pointer params (`bias`, `gamma`/`beta`) arrive as `Value::Int(0)` and are
distinguished by variant; the `dim!` macro makes a non-positive extent a no-op.

`wukong_codegen_gpu` mirrors `apply_cast` and `int_bits` from `lib.rs`; `fusion.rs` mirrors the
`Accelerator` recognizer list. Editing cast or width semantics here without mirroring there
desynchronizes the GPU backend.
