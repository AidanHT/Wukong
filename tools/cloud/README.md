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
required for the Starter plan's free credit. Set the workspace budget cap at
<https://modal.com/settings/usage> **before the first metered run** (plan §11 item 2).

## The first session (~2 minutes of GPU time, free)

Run these from the repo root. `WK_GPU` picks the SKU; it is read at import time because Modal binds
a function's GPU at decoration time, and it is baked into the image env so the container knows what
was asked for (Modal does **not** forward local environment variables into containers).

```powershell
# Provenance first — ALWAYS. Verifies you got a full device, not a MIG slice, and that nvcc works.
$env:WK_GPU="L4"; modal run tools/cloud/modal_app.py::device_info

# Compile on CPU (no GPU attached: ~$0.51/hr for 8 cores + 16 GiB, and no $0.80–$3.95/hr of GPU).
modal run tools/cloud/modal_app.py::build

# Run the device correctness suite on the GPU, with skips escalated to failures.
$env:WK_GPU="L4"; modal run tools/cloud/modal_app.py::test
```

Bash equivalent: `WK_GPU=L4 modal run tools/cloud/modal_app.py::device_info`.

### Start on L40S or L4, not A100

**L4 and L40S are `sm_89` — the same architecture as the dev RTX 4050.** Today's PTX runs there
*unmodified*, which validates Linux, `cudarc`, the driver-JIT and the whole test suite **before** any
codegen change. An A100 is `sm_80`, which is *older*, and PTX is forward-compatible only — a build
whose modules floor at `sm_89` cannot load on an A100 at all. That is expected, not a bug.

L4 (58 SMs, $0.80/hr) is the cheapest correctness box. L40S has **142 SMs vs the 4050's 20**, so it
answers every "does this scale past 20 SMs" question on the same architecture for pocket change.

## Command reference

| Command | GPU? | What it does |
|---|---|---|
| `::device_info` | yes | §6.1 provenance block: CC, SM count, opt-in SMEM, L2, VRAM, driver, MIG state, and a real `dlopen` of every peer library. Checks the device against a spec table keyed on the **device's own name** and says plainly if it is a slice. |
| `::build` | **no** | `cargo check --features gpu --all-targets`, then `cargo test --no-run` for `wukong_codegen_gpu` + `wukong_driver`, then the CPU workspace suite. Writes into the Volume. |
| `::test` | yes | The device gates with `WUKONG_GPU_REQUIRED=1`. `--peers` also requires NVRTC/cuBLAS/cuBLASLt/cuDNN. `--filter <name>` narrows. |
| `::bench` | yes | The `#[ignore]`d perf sweeps, release, single-threaded. `--name gemm_pipe_sweep` selects one. **Needs `::build --release` first.** |
| `::interactive` | yes | Target for `modal shell tools/cloud/modal_app.py::interactive`. |

Options are passed as CLI flags, e.g.:

```powershell
modal run tools/cloud/modal_app.py::test --peers --filter gemm
modal run tools/cloud/modal_app.py::build --release
modal run tools/cloud/modal_app.py::bench --name flash_tiled_vs_untiled
modal shell tools/cloud/modal_app.py::interactive
```

**Never run two of these concurrently.** They share one Volume, and Modal Volumes are last-write-wins
on concurrent modification of the same file — two cargos in one target dir is a corruption you would
pay GPU-minutes to discover.

## How the cost control works

Four mechanisms, all load-bearing:

1. **Build on CPU, run on GPU.** `build` attaches no GPU. Compiling 21 crates with `--features gpu`
   takes minutes; the same container with an H100 attached costs ~8× more per second
   ($3.95 + $0.51 vs $0.51/hr for 8 cores + 16 GiB).
2. **The GPU entry points refuse to compile.** `test` and `bench` abort in seconds if the Volume has
   no prebuilt test binary *for the profile they were asked for*. `bench` hardcodes `--release`
   while `build` defaults to debug, so without this guard the first `::bench` of a session would
   compile the whole workspace in release mode on metered silicon. **Re-run `::build` after every
   source edit**, and `::build --release` before any `::bench` or peer smoke test.
3. **A persistent Volume (`wukong-build`)** holds `CARGO_TARGET_DIR`, the crates.io registry, and the
   JIT'd cubin cache, so you pay the full compile once — see the mtime invariant below, which is what
   makes "incremental" actually true across containers.
4. **Explicit CPU and memory reservations.** Modal's default request is **0.125 cores and 128 MiB**,
   and a container only exceeds that if the worker happens to have capacity spare. The device suite's
   two corpus gates compile 357 `.wk` programs × 2 opt levels **on the CPU** while the GPU idles, so a
   starved container bills GPU-seconds to wait on one core. Every function reserves `WK_CPU`
   (default 8.0) and `WK_MEM` MiB (default 16384); at $0.0472/core/hr + $0.0080/GiB/hr that is
   ~$0.50/hr on top of the GPU, and it pays for itself the moment it saves half an hour of L40S time.

Every metered function prints a `[meter]` line with its $/hr, the cost ceiling implied by the
timeout, and the actual spend on exit — paste those into the round log (plan §6.6).

Source is *mounted at runtime*, not baked into the image, so editing Rust code costs a few MB of
upload and never an image rebuild. The upload excludes `target/` (18 GB), `.claude/` (59 GB),
`tools/cuda-redist/` (11 GB) and `data/` (477 MB) — about 12 MB actually ships.

Check spend at <https://modal.com/settings/usage>.

### The mtime invariant (why `::test` does not recompile the world)

`Image.add_local_dir(..., copy=False)` mounts the source tree at container startup. The wire format
for a mounted file is `MountFile{filename, sha256_hex, size, mode}` — **it carries no mtime**, so the
timestamps your files get inside the container are assigned there and are not guaranteed to be
stable from one container to the next. Cargo's freshness check is mtime-based: a source file newer
than the unit's output is stale. `build` runs in one container and `test` in another, so an unstable
mount would make *every* unit stale and rebuild the workspace **on the GPU**, every run.

`_stamp_sources` closes that: it hashes every mounted file, keeps `{path: [sha256, mtime]}` on the
Volume, restores the previously assigned mtime for unchanged content and stamps changed/new files
with `now`. Unchanged sources stay older than the artifacts built from them (no rebuild); an edited
file is newer (rebuild) — exactly cargo's intent. Watch the `[src] N files stamped; M new/changed`
line: `M` should be 0 on a re-run and small after an edit. If it says "full rebuild expected" on a
GPU function, kill the run and rebuild on CPU.

### Timeouts

`WK_TIMEOUT` defaults to **7200 s** (Modal allows 1 s – 24 h). It is a *cost ceiling*, not a safety
net: at the cap Modal kills the container, every second already spent is still billed, and the run's
results are lost. It was 3600, which plan §8 risk 6 flags as a gamble — the corpus gates' wall time
on this hardware is genuinely unknown, and losing the first `::test` at the one-hour mark would waste
every minute it had already spent. Lower it deliberately on an expensive SKU (2 h on H100 is a $8
ceiling); raise it if the first measured suite run comes close.

The Volume survives a timeout: Modal commits Volumes in the background "every few seconds" and takes
"a final snapshot and commit on container shutdown", so a killed run loses at most the last few
seconds of `target/` — the explicit `commit()` calls are belt-and-braces.

### The loader path (`LD_LIBRARY_PATH`) — the one that fails silently

```
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
```

- **NEVER add `/usr/local/cuda/lib64/stubs`.** It holds a *stub* `libcuda.so` with no driver behind
  it, and `libcuda.so` is literally cudarc's **first** driver candidate — the stub wins over the
  injected real driver and every call fails with a symptom that reads as "no CUDA device". Some peer
  build recipes (CUTLASS, FA2/FA3, vLLM) tell you to add it: add it for the duration of that one
  `cmake`/`pip` command and never to the environment a Wukong process inherits. The functions refuse
  to start if `stubs` appears on the path.
- The devel image legitimately sets `LIBRARY_PATH=/usr/local/cuda/lib64/stubs`. That is **gcc's
  link-time search path**, a different variable; never copy it into `LD_LIBRARY_PATH`.
- The two `/usr/local/nvidia/*` entries are the base image's own value, kept because that is where
  some container runtimes inject the driver.
- **cuDNN can only be found through the unversioned `libcudnn.so`.** cudarc's Linux candidates for
  `cudnn` are `libcudnn.so` and `libcudnn.so.{12,11,10,1}` — the real cuDNN-9 soname `libcudnn.so.9`
  is *never tried* (`cudarc-0.16.6/src/lib.rs:112-147`; the same asymmetry is documented at
  `baselines.rs:67-71`). The `-cudnn-devel` image ships the dev symlink via `libcudnn9-dev-cuda-12`;
  the image also creates it defensively so a `-runtime` tag degrades gracefully instead of skipping
  the conv peer in silence. `::device_info` probes the **exact candidate list, in order, through the
  real `dlopen`** — not `ldconfig -p`, which ignores `LD_LIBRARY_PATH` and would green-light a name
  the code never asks for.

## Expected results on the first run

- `device_info` — should report `sm_89`, 58 SMs for L4 / 142 for L40S, and load all five peer
  libraries. The gate is keyed on the **device's own name**, not on `WK_GPU`, because Modal may serve
  an H200 for an `H100` request (write `H100!` to opt out) and because H100 PCIe (114 SMs) and H100
  SXM5 (132) are different bins of the same SKU name. A MIG slice announces itself in both the name
  and a reduced SM count.
- `build` — the first one is slow (full compile + registry download); later ones are incremental.
  The CPU workspace suite is run with `check=False` so a Linux-portability failure is *reported*
  rather than aborting the run. Some CPU results are expected to differ on Linux: the 256-bit AVX2
  vectorizer is Win64-ABI-only (`avx2.rs:53`) and silently drops to 128-bit here, and the thread
  pinning that fixed the CPU timing instrument is `kernel32`-only. **CPU benchmarks are explicitly
  out of scope** for this campaign (plan §9) — these boxes are GPU instruments.
- `test` — this is the real question of Phase 1. Anything that fails is a genuine
  Linux/driver/portability finding, which is exactly what the phase is for. **Record the wall time**
  from the `[meter]` line: plan §6.6 sizes every later metered phase on it.

## Known unknowns this session is meant to answer

Straight from plan §8, in priority order:

1. Does `cudarc 0.16` (`cuda-12060`) work against the cloud driver (r580 / CUDA 13.0 API)?
2. Does the "legacy 8×`.b32`" f16 WMMA fragment spelling (`ptx_wmma.rs:34`) still JIT off Ada?
3. Is the sticky-fault / device-lost policy (`gpu.rs:204`, premised on Windows WDDM) different on
   the Linux driver, and can in-process recovery be restored?
4. Does CUDA-graph capture (`graph.rs:132`, a raw-driver workaround for cudarc 0.16) still behave on
   a newer driver?
5. How long does the full suite take? That number sizes every later metered phase.
6. Do the mounted sources actually keep stable mtimes (does `[src] … 0 new/changed` hold on a second
   container)? The stamper makes this true by construction; the first two runs confirm it.

## Later: the other providers

Modal covers bring-up and free iteration. When canonical published numbers are needed, plan §4.1
routes to a **root VM** where clock locking is possible — Verda spot H100 $1.63/hr, Hyperstack A100
$1.35/hr, or Lambda as the known-clean control. The image recipe here (CUDA devel + rustup + the
same env) transfers directly to those boxes as a Dockerfile or a plain `apt`/`rustup` script.
