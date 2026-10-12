# `src/` module index

| file | contents |
|---|---|
| `lib.rs` | Core IDs and types: `ValueId`, `BlockId`, `MirType`, `BasicBlock`, `Function`, `Program`, `MirLevel`, `StaticData` |
| `inst.rs` | Instruction vocabulary: `Op`, `BinOp`, `CmpOp`, `CastKind`, `RoundMode`, `Inst`, `Terminator` |
| `builder.rs` | `Builder` — imperative construction of one `Function` |
| `print.rs` | `print_program`, `print_function` — deterministic MIR text (`--emit=mir`) |
| `verify.rs` | `verify_function`, `verify_program` — SSA / dominance / typing rules + unit tests |
| `vec_kernel.rs` | `VecKernel`, `VecOp`, `VecRedOp`, reference evaluator, register-pressure model |
| `f16_bits.rs` | `f32_to_f16_bits`, `f16_bits_to_f32`, `round_f16` — integer f16 conversions bit-identical to the runtime shims (`half`) |

`f16_bits`, `print`, `vec_kernel`, and `verify` are `pub mod`; `builder` and `inst` are private (re-exported types from `lib.rs`).
