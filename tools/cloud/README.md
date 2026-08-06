# Running Wukong's GPU backend on rented datacenter GPUs (Modal)

Phase 0/1 of [`GPU_RETARGET_PLAN.md`](../../GPU_RETARGET_PLAN.md). Nothing here changes compiler
code — it gets the existing tree onto a real GPU so the ~130 device-executing correctness gates can
run somewhere other than the dev laptop.

## Why Modal

Verified 2026-08-06: **$30/month of recurring free credit, no credit card**, full (non-MIG) devices,
per-second billing, and arbitrary container images. That is ≈12 free A100-hours or ≈7.5 free
H100-hours *every month*. The hyperscaler "$300 free trials" are traps for this purpose — GCP, Azure
and OCI all lock GPU quota at 0 until you convert to paid billing.

Modal is a **container** platform, not an SSH VM. Two consequences the plan depends on:

- You cannot lock clocks (`nvidia-smi -lgc`). Modal is for **bring-up and iteration**; canonical
  published rounds happen later on a root VM (Verda/Hyperstack/Lambda) per plan §6.3.
- The physical host can change between invocations, so **every A/B comparison must live inside one
  invocation**.

## One-time setup

```powershell
# 1. Authenticate (opens a browser; creates ~/.modal.toml). Modal 1.5.3 is already installed.
modal setup
```

That is the only step that needs you rather than me. Sign in with GitHub or Google — no card is
required for the Starter plan's free credit.

## The first session (~2 minutes of GPU time, free)

Run these from the repo root. `WK_GPU` picks the SKU; it is read at import time because Modal binds
a function's GPU at decoration time.

```powershell
# Provenance first — ALWAYS. Verifies you got a full device, not a MIG slice, and that nvcc works.
$env:WK_GPU="L40S"; modal run tools/cloud/modal_app.py::device_info

# Compile on CPU (no GPU attached, ~$0.05/hr instead of $3.95/hr).
modal run tools/cloud/modal_app.py::build

# Run the device correctness suite on the GPU, with skips escalated to failures.
$env:WK_GPU="L40S"; modal run tools/cloud/modal_app.py::test
```

Bash equivalent: `WK_GPU=L40S modal run tools/cloud/modal_app.py::device_info`.

### Start on L40S, not A100

**L40S is `sm_89` — the same architecture as the dev RTX 4050.** Today's PTX hardcodes
`.target sm_89` in 67 places, so it runs there *unmodified*: this validates Linux, `cudarc`, the
driver-JIT and the whole test suite **before** any codegen change. An A100 is `sm_80`, which is
*older*, and PTX is forward-compatible only — **today's build cannot load on an A100 at all.** That
is expected, not a bug; Phase 2 of the plan fixes it.

L40S also has **142 SMs vs the 4050's 20**, so it answers every "does this scale past 20 SMs"
question on the same architecture for pocket change.

## Command reference

| Command | GPU? | What it does |
|---|---|---|
| `::device_info` | yes | §6.1 provenance block: CC, SM count, opt-in SMEM, L2, VRAM, driver, MIG state, peer-library resolution. Checks the device against a spec table and says plainly if it is a slice. |
| `::build` | **no** (8 CPUs) | `cargo check --features gpu --all-targets`, then `cargo test --no-run` for `wukong_codegen_gpu` + `wukong_driver`, then the CPU workspace suite. Writes into the Volume. |
| `::test` | yes | The device gates with `WUKONG_GPU_REQUIRED=1`. `--peers` also requires NVRTC/cuBLAS/cuBLASLt/cuDNN. `--filter <name>` narrows. |
| `::bench` | yes | The `#[ignore]`d perf sweeps, release, single-threaded. `--name gemm_pipe_sweep` selects one. |
| `::interactive` | yes | Target for `modal shell tools/cloud/modal_app.py::interactive`. |

Options are passed as CLI flags, e.g.:

```powershell
modal run tools/cloud/modal_app.py::test --peers --filter gemm
modal run tools/cloud/modal_app.py::bench --name flash_tiled_vs_untiled
modal shell tools/cloud/modal_app.py::interactive
```

## How the cost control works

Two mechanisms, both load-bearing:

1. **Build on CPU, run on GPU.** `build` attaches no GPU. Compiling 21 crates with `--features gpu`
   takes minutes; doing it on an H100 would cost ~80× more per second than a CPU core.
2. **A persistent Volume (`wukong-build`)** holds `CARGO_TARGET_DIR`, the crates.io registry, and the
   JIT'd cubin cache. Rebuilds are incremental across sessions, so you pay the full compile once.

Source is *mounted at runtime*, not baked into the image, so editing Rust code costs a few MB of
upload and never an image rebuild. The upload excludes `target/` (18 GB), `.claude/` (59 GB),
`tools/cuda-redist/` (11 GB) and `data/` (477 MB) — about 12 MB actually ships.

Check spend at <https://modal.com/settings/usage>.

## Expected results on the first run

- `device_info` — should report `sm_89`, 142 SMs for L40S, and resolve all five peer sonames. The
  cuDNN/cuBLAS libraries come from the `nvidia/cuda:*-cudnn-devel` base image, replacing the
  hand-staged `tools/cuda-redist` PATH dance the laptop needs.
- `build` — the first one is slow (full compile + registry download); later ones are incremental.
  The CPU workspace suite is run with `check=False` so a Linux-portability failure is *reported*
  rather than aborting the run. Some CPU results are expected to differ on Linux: the 256-bit AVX2
  vectorizer is Win64-ABI-only (`avx2.rs:53`) and silently drops to 128-bit here, and the thread
  pinning that fixed the CPU timing instrument is `kernel32`-only. **CPU benchmarks are explicitly
  out of scope** for this campaign (plan §9) — these boxes are GPU instruments.
- `test` — this is the real question of Phase 1. Anything that fails is a genuine
  Linux/driver/portability finding, which is exactly what the phase is for.

## Known unknowns this session is meant to answer

Straight from plan §8, in priority order:

1. Does `cudarc 0.16` (`cuda-12060`) work against the cloud driver (r580 / CUDA 13.0 API)?
2. Does the "legacy 8×`.b32`" f16 WMMA fragment spelling (`ptx_wmma.rs:34`) still JIT off Ada?
3. Is the sticky-fault / device-lost policy (`gpu.rs:204`, premised on Windows WDDM) different on
   the Linux driver, and can in-process recovery be restored?
4. Does CUDA-graph capture (`graph.rs:132`, a raw-driver workaround for cudarc 0.16) still behave on
   a newer driver?
5. How long does the full suite take? That number sizes every later metered phase.

## Later: the other providers

Modal covers bring-up and free iteration. When canonical published numbers are needed, plan §4.1
routes to a **root VM** where clock locking is possible — Verda spot H100 $1.63/hr, Hyperstack A100
$1.35/hr, or Lambda as the known-clean control. The image recipe here (CUDA devel + rustup + the
same env) transfers directly to those boxes as a Dockerfile or a plain `apt`/`rustup` script.
