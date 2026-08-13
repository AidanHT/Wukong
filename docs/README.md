# `docs/` — Wukong language and compiler documentation

This folder documents the **Wukong language** (`.wk`) and the **compiler** that implements it:
surface syntax, semantics, the compile pipeline, backends, and how to measure them honestly. It is
not the product roadmap for a datacenter GPU stack — that campaign lives under [`gpu/`](gpu/) and
[`GPU_RETARGET_PLAN.md`](../GPU_RETARGET_PLAN.md) at the repo root.

Start here if you are writing or running Wukong programs. Start under [`gpu/`](gpu/) only if you
are renting silicon, reading derivation dossiers, or working on `wukong_codegen_gpu`.

---

## Language and compiler (read these first)

| Document | What it covers |
|---|---|
| [Master plan](WUKONG_MASTER_PLAN.md) | North star for Wukong as a **programming language**: mission, success criteria, sequenced phases, and AI-editor / vibe-coding as a first-class user of the kernel language — not a GPU-family campaign. |
| [Language direction](LANGUAGE_DIRECTION.md) | Design for generic gpu-native lowering: consume `@tile` / `@generic` / `shared`, explicit vendor kernels rather than silent PTX rewrites. Review thesis, not a claim those features run today. |
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

[`gpu/`](gpu/) holds working documents for the **datacenter retarget**: rented-GPU runbooks,
derivation dossiers (Wave-0 and Act-2), and the benchmark-instrument prose. That work exists to
make `wukong_codegen_gpu` honest on A100/H100 — it does not define what Wukong the language is.

| Document | What it covers |
|---|---|
| [`gpu/README.md`](gpu/README.md) | Index of this subtree, the benchmark instrument (`bench_instrument.rs`), and how GPU docs relate to the plan. |
| [`gpu/derive/`](gpu/derive/) | Wave-0 derivation dossiers (D1–D7), Act-2 wave dossiers (WAVE3–WAVE6), and predict-before-measure documents — all $0 derivation, tested on silicon later. |
| [`gpu/phase1-runbook.md`](gpu/phase1-runbook.md) | Operational checklist for a rented `sm_89` bring-up session (Modal L4). |
| [GPU_RETARGET_PLAN.md](../GPU_RETARGET_PLAN.md) | The plan itself — mission, phases, benchmark methodology (§6), budget, and success criteria. |

Round logs and session ledgers live under [`bench/gpu/`](../bench/gpu/), not under `docs/`.
