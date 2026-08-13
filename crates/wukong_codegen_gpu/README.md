# `wukong_codegen_gpu`

Wukong (`.wk`) is a systems language for user-written tensor kernels with compile-time shape
safety. **This crate is a backend**, not the product: it emits PTX text, driver-JIT-loads it via
`cudarc` (`cuModuleLoadData` — no `nvcc`/`ptxas`/`nvrtc` on the build path), launches kernels, and
copies results back. The language surface lives in user `.wk` programs; most of what follows is
implementation detail for two driver backends and a separate benchmark kernel zoo.

## Language vs library split

Wukong exposes two GPU execution modes. They share this crate but serve different roles.

### 1. `--backend=gpu` — recognizer offload (the product path today)

The **interpreter still runs the whole `.wk` program**. Control flow, buffer layout, and every
non-kernel op stay on the CPU and remain bit-identical to the interpreter oracle. Only calls that
`wukong_mir_build` recognizes as runtime kernels are offloaded.

The seam is [`wukong_interp::Accelerator`](../../wukong_interp), implemented by
[`wukong_driver::gpu_accel::GpuAccel`](../../wukong_driver/src/gpu_accel.rs). The offload menu is
exactly **five hooks**:

| Hook | Recognized shape |
|------|------------------|
| `sgemm_nt` | `C = A·Bᵀ` (plain GEMM) |
| `sgemm_nt_epi` | fused `act(x·Wᵀ [+ bias])` epilogue |
| `vmath` | elementwise unary/binary f32 ops |
| `norm` | layer/rms norm |
| `sreduce` | parallel reductions |

A GPU error is surfaced as `Some(Err(..))`, never silent CPU fallback. Declining a shape (`None`)
means "run the CPU microkernel instead" — structurally unsupported, not a failed launch.

### 2. `--backend=gpu-native` — whole-program MIR→PTX

[`lower.rs`](src/lower.rs) consumes MIR directly and emits a PTX kernel for the entry function (plus
`.func` bodies for callees), then driver-JIT-loads and launches via the shared [`gpu::Gpu`](src/gpu.rs)
harness. Recognized runtime symbols lower to per-op **`mrt_*` device helpers** — naive single-thread
loops that reproduce the CPU microkernel contract. There are **no tensor cores on this path**; the
`sm_80` scalar floor is intentional.

Anything the lowering tables do not handle returns `Err("lower::UNSUPPORTED…")`. That is a **SKIP**,
never wrong PTX — the coverage gate reports declines honestly.

**Coverage:** as of 2026-08-09, **235 of 357** `tests/run` fixtures lower and match the interpreter
oracle (integer/control-flow bit-exact; float within the CPU↔GPU tolerance). Pinned by
`RUN_CORPUS_COVERAGE_FLOOR` in `lower.rs`; the remaining 122 are honest `UNSUPPORTED` declines,
dominated by not-yet-lowered `wukong_*` runtime symbols. Optionally, eligible programs compile into
one cooperative megakernel via [`megakernel.rs`](src/megakernel.rs) (90 of 103 mega-eligible fixtures
at the same gate).

### 3. `ptx_*` / `gpu.rs` / `serving.rs` / `train_resident.rs` — benchmark kernel zoo

Most of this crate is a **Rust-launched kernel library** used by benches, peers, and internal gates:
hand-tuned PTX families (`ptx_gemm`, `ptx_wmma`, `ptx_wgmma`, `ptx_flash`, `ptx_conv`, …), the
[`gpu.rs`](src/gpu.rs) launch harness, [`serving.rs`](src/serving.rs) (paged KV + continuous
batching decode), and [`train_resident.rs`](src/train_resident.rs) (GPU-resident MLP training step).

**`flash` / `conv` / `serving` are not reached by writing `.wk` today.** They exist to measure
kernels and prove numeric contracts against f64 references — implementation inventory, not the
language thesis. User kernels reach the device through the two paths above.

## Feature gate

All `cudarc` code is behind the `gpu` feature. A plain `cargo test` compiles only the device-free
modules (`ptx_target`, `paged_kv`, `paged_attention`, `bench_instrument`, `tma_host`, `ptx_wgmma`,
`ptx_act`, …) and **does not type-check most of this crate**.

```sh
cargo check --features gpu --all-targets   # required gate; no device needed to compile
cargo test -p wukong_codegen_gpu --features gpu   # runs device gates when a GPU is present
```

`cudarc` uses `dynamic-loading` — building needs no CUDA toolkit; running needs the driver and a
device.

## PTX hygiene

PTX must be **pure ASCII**. A single Unicode character (`×`, `→`, `≥`) pasted into a `format!` is a
`ptxas fatal` at `cuModuleLoadData`. Every generator family has an ASCII gate; add one when adding a
new family.

Module headers (`.version` / `.target` floors) come from [`ptx_target`](src/ptx_target.rs) — never
hardcode `.target sm_89` (or any other floor) at a call site.

## Where to read next

- [`src/README.md`](src/README.md) — one-line module map from `lib.rs`
- [`src/lower.rs`](src/lower.rs) — MIR→PTX lowering design and coverage floors
- [`../../wukong_driver/src/gpu_accel.rs`](../../wukong_driver/src/gpu_accel.rs) — the five-hook offload bridge
