# `docs/` — Wukong language and compiler documentation

This folder documents the **Wukong language** (`.wk`) and the **compiler** that implements it:
surface syntax, semantics, the compile pipeline, backends, and how to measure them honestly. It is
not the product roadmap for a datacenter GPU stack.

Start here if you are writing or running Wukong programs.

---

## Language and compiler (read these first)

| Document | What it covers |
|---|---|
| [Language guide](language-guide.md) | The `.wk` surface — types, tensors, control flow, `@parallel`, CLI flags — with an honest maturity legend per feature. |
| [Compiler internals](internals.md) | Crate layering, the lexer→MIR pipeline, optimizer passes, the four execution backends, recognizers, and the test gates. |
| [Roadmap & limitations](roadmap.md) | What runs today, what is checked-only, known sharp edges, and planned work — language and compiler first, GPU as one backend among several. |
| [LLVM setup](llvm-setup.md) | The **optional** textual-LLVM path (`--emit=llvm-ir`). The default native path is Cranelift and needs no LLVM. |

---

## What reaches the GPU from `.wk` source

Wukong is a language and compiler; the GPU is a **backend**, not the thesis of the project.

From `.wk` today, two paths touch the device:

| Path | CLI | What runs on GPU |
|---|---|---|
| Recognizer offload | `--backend=gpu` | Only the five [`Accelerator`](../crates/wukong_interp/src/lib.rs) hooks the interpreter can delegate: `sgemm_nt`, `vmath`, `norm`, `sreduce`, and `sgemm_nt_epi` (fused `act(A·Bᵀ + bias)`). Everything else stays on the CPU oracle path. |
| General MIR lowering | `--backend=gpu-native` | Lowers the whole program's MIR to PTX; supported ops map to `mrt_*` device helpers in `wukong_codegen_gpu`. Coverage is partial — unsupported ops decline with `lower::UNSUPPORTED…`. |

Flash attention, conv2d, quantized GEMM, device-resident training (`train_resident`), and serving
(`serving.rs` / paged KV) are **Rust launch wrappers** inside `wukong_codegen_gpu`. They are gated
by `cargo test -p wukong_codegen_gpu --features gpu` and measured in [`BENCHMARKS.md`](../BENCHMARKS.md);
no `wukongc` flag and no `.wk` syntax reaches them yet.

---

## Measurement and compile-time floor

These documents record **ratios and gates**, not marketing claims. Absolute GFLOP/s are deliberately
absent — this hardware's clocks swing too much for portable absolutes.

| Document | What it covers |
|---|---|
| [BENCHMARKS.md](../BENCHMARKS.md) | Cross-language CPU benchmarks, GPU backend standings, methodology, retractions, and peer-strength corrections. Lives at the repo root because it is the public scoreboard. |
| [metrics.md](metrics.md) | The metrics an ML kernel language must win at, with Wukong's honest standing on each — including losses. North star for optimization work. |
| [compile-floor.md](compile-floor.md) | The principled minimum for code→object compile time and the in-process vs spawned-toolchain tradeoff. |

---

## GPU backend campaign (not the language thesis)

The **datacenter retarget** exists to make `wukong_codegen_gpu` honest on A100/H100 — it does not
define what Wukong the language is.

Round logs and session ledgers live under [`bench/gpu/`](../bench/gpu/), not under `docs/`.
