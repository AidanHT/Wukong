# wukong_mir

Wukong's typed, block-structured SSA IR — the currency between lowering and every execution backend
(interp, native Cranelift, textual LLVM, GPU PTX). Wukong is a language for `.wk` tensor kernels;
this crate is the IR those kernels compile into after shape checking, not the shape checker itself.

A `Function` is a `Vec<BasicBlock>`; every block ends in exactly one `Terminator`. SSA merges use
**typed block parameters + branch arguments** (Cranelift/MLIR style), never phi nodes. Data types +
imperative `Builder` + deterministic printer + verifier — no passes, no lowering.

Dependencies: `wukong_span`, `wukong_types::Scalar` only (pinned by
`f16_bits::tests::the_crate_depends_on_wukong_span_and_wukong_types_only`). `half` is a
dev-dependency: the test oracle for `src/f16_bits.rs`, which reaches no consumer.

## Pipeline fit

```
wukong_mir_build → wukong_opt → wukong_interp | wukong_codegen_cranelift | wukong_codegen_gpu | wukong_codegen_llvm
```

`wukong_driver::verify_or_ice` calls `verify_program` before every backend entry. `--emit=mir` /
`--emit=mir-high` print via `print_program` (`src/print.rs`). Note: `--emit=mir-high` means
*pre-optimization* MIR, not `MirLevel::High`. `--emit=mir-text` prints via `print_program_typed`
(`src/text.rs`), and `wukongc <file>.mir` reads that form back with `parse_program`.

## Key files

| file | what |
|---|---|
| `src/lib.rs` | `ValueId`, `BlockId`, `MirType`, `BasicBlock`, `Function`, `Program`, `MirLevel`, `StaticData` |
| `src/inst.rs` | `Op` (20 variants), `BinOp`, `CmpOp`, `CastKind`, `Inst`, `Terminator` |
| `src/builder.rs` | imperative `Builder` for one `Function` |
| `src/print.rs` | `print_program` / `print_function` — grep surface for vectorization |
| `src/text.rs` | `print_program_typed` / `parse_program` — typed textual MIR that reads back (`--emit=mir-text`, `.mir` input); laws: `typed(parse(typed(p))) == typed(p)`, `print_program(parse(typed(p))) == print_program(p)` |
| `src/verify.rs` | `verify_function`, `verify_program` — well-formedness gate |
| `src/vec_kernel.rs` | `VecKernel` — backend-agnostic 256-bit AVX2 loop recipe |
| `src/f16_bits.rs` | `f32_to_f16_bits` / `f16_bits_to_f32` / `round_f16` — the `half` crate's software f16 routine, transcribed (the recipe evaluator's `LoadF16`/`StoreF16`/`RoundF16` rounding) and tested bit for bit against `half` |

| region | file:line | what |
|---|---|---|
| `MirLevel` (inert) | `src/lib.rs:173` | enum exists; nothing branches on `Program::level` |
| program container | `src/lib.rs:197` | `Program { funcs, statics, level }` |
| per-function verify | `src/verify.rs` | `verify_function` — wired into pipeline |
| whole-program verify | `src/verify.rs` | `verify_program` — call-vs-callee cross-check at the driver boundary |
| vec kernel ABI | `src/vec_kernel.rs` | `VecKernel`, `eval_lane`, `VEC_LANES = 8` |

## Core types

- **`MirType`** — signless integers (`I1`…`I64`), floats, `Ptr`, `Vec(elem,n)`, `Array(elem,n)`,
  `Void`. `from_scalar` maps `Char` → `I32`, `Usize`/`Isize` → `I64`.
- **`Function`** — `value_types` arena, `blocks`, `entry`, `params`, `vec_kernels` (function-local
  `VecKernel` recipes indexed by `Op::VecKernelCall`).
- **`Program`** — `funcs`, `statics` (`.rodata` string blobs via `Op::GlobalAddr`), inert `level`.

## Verifier (`verify.rs`)

`verify_function(f) -> Vec<String>` — empty = ok. Failure is an **ICE** (bug in a pass), not a user
diagnostic. Enforces: single-definition SSA, dominance, entry-block param match, block-id indexing,
per-op typing including cast kinds and widths, terminator correctness, `VecKernelCall` index range
and result presence, and every registered recipe's operand indices and reduction shape.

`verify_program` additionally cross-checks each `Call` against callee signatures at the driver
boundary. The optimizer's `--verify-each` checks individual functions after each executed pass and
whole programs at input/output and interprocedural boundaries, in debug and release builds.

## `VecKernel`

Recipe for one vectorizable straight-line loop body from `wukong_mir_build`'s 256-bit path. Consumed
bit-exactly by `wukong_interp` (lane-by-lane oracle) and `wukong_codegen_cranelift::avx2`. Fixed ABI:
`fn(ptrs: *const *mut u8, scalars: *const f32, n: u64)`. `eval_lane` / `eval_all` are the per-lane
truth source.

## Landmines

- **`MirLevel` is NOT an invariant** — do not write `if program.level == MirLevel::High`. Every
  `Program` starts at `Low`; nothing reads `level` in the pipeline.
- **`Round(Nearest)` is ties-to-EVEN** — interpreter must use `round_ties_even()`, not `round()`.
- **Pointer truthiness** — `if p` lowers to `ptrtoint p != 0`. That is sound only because the
  interpreter reserves slot 0 as a never-allocated null (`wukong_interp::NULL_SLOT`); before that
  reservation the first allocation of a run had address 0 and the same lowering was reverted
  (0900352) as a silent interp/native divergence. Any new interpreter entry point must keep the
  reservation.
- **Recipe buffer capacity** — opaque pointer operands do not carry lengths; verification checks
  recipe indices but does not prove that call-site pointer/scalar buffers are large enough.
- **Release verification is opt-in per pass** — use `--verify-each`; default release compilation
  retains the existing backend boundary checks without the cost of checking each optimizer pass.

## Testing

```sh
cargo test -p wukong_mir --lib
```

21 unit tests (verifier rules + printer snapshot). Differential gate: `cargo test -p wukongc --test run`.

This crate owns no `E`-codes. Verifier output is prose prefixed `internal compiler error (MIR verify):`.
