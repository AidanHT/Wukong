# Running Wukong's GPU backend on rented datacenter GPUs (Modal)

Phase 0/1 of [`GPU_RETARGET_PLAN.md`](../../GPU_RETARGET_PLAN.md). Nothing here changes compiler
code — it gets the existing tree onto a real GPU so the ~130 device-executing correctness gates can
run somewhere other than the dev laptop, **and it builds the strong peers** so that "better than
SOTA" is a claim someone can check rather than a claim someone made.

Layout:

| Path | What |
|---|---|
| `modal_app.py` | The app: the image, the pins, and every entry point. |
| `peers/verify_peers.py` | Resolves every strong peer and **exits non-zero** when a declared one is missing. |
| `peers/smoke_inductor.py` | Proves Inductor really emitted a Triton kernel, and warms the autotune cache once. |
| `peers/torch_compile_peer.py` | The `torch.compile` framework bar itself: eager / compiled / max-autotune, fairly configured. |

The research behind every command here is [`docs/gpu/derive/D5_peer_builds.md`](../../docs/gpu/derive/D5_peer_builds.md)
(45 KB, live-verified). **Read that before changing a pin**; nothing here was invented at the keyboard.

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
| `::device_info` | yes | §6.1 provenance block: CC, SM count, opt-in SMEM, L2, VRAM, driver, MIG state, a real `dlopen` of every peer library, the staged-peer manifest and the strong-peer resolution table. Checks the device against a spec table keyed on the **device's own name** and says plainly if it is a slice. |
| `::build` | **no** | `cargo check --features gpu --all-targets`, then `cargo test --no-run` for `wukong_codegen_gpu` + `wukong_driver`, then the CPU workspace suite. Writes into the Volume. |
| `::build_peers` | **no** | Stages the heavy strong peers onto the Volume: the CUTLASS profiler, the vLLM (Marlin/Machete) venv, optionally the FA2/FA3 wheels. Idempotent; `--force` rebuilds. |
| `::peers` | yes | **The strong-peer battery.** Installs any FlashAttention wheel `::build_peers` staged, resolves every peer, proves Inductor emits Triton, runs the in-tree cuBLAS/cuDNN gates with skips escalated, and runs the Rust strong-peer gate. Fails if a `--require`d peer is missing. |
| `::test` | yes | The device gates with `WUKONG_GPU_REQUIRED=1`. `--peers` also requires NVRTC/cuBLAS/cuBLASLt/cuDNN. `--strong-peers <list>` declares the §0 bar. `--filter <name>` narrows. |
| `::bench` | yes | The `#[ignore]`d perf sweeps, release, single-threaded. `--name gemm_pipe_sweep` selects one. `--peers` / `--strong-peers` escalate a missing peer to a failure. **Needs `::build --release` first.** |
| `::framework` | yes | The `torch.compile` bar: eager / compiled / max-autotune over `gemm`, `linear_gelu` or `sdpa`, fastest wins. |
| `::cutlass` | yes | The CUTLASS-profiler GEMM bar, with cuBLAS as a same-binary control column. **Needs `::build_peers`.** |
| `::marlin` | yes | The int4 bar: vLLM's own Marlin (Ampere) / Machete (Hopper) kernel benchmarks. **Needs `::build_peers`.** |
| `::interactive` | yes | Target for `modal shell tools/cloud/modal_app.py::interactive`. |

Options are passed as CLI flags, e.g.:

```powershell
modal run tools/cloud/modal_app.py::test --peers --filter gemm
modal run tools/cloud/modal_app.py::build --release
modal run tools/cloud/modal_app.py::bench --name flash_tiled_vs_untiled --peers
modal run tools/cloud/modal_app.py::framework --op sdpa --causal --shapes 1x16x2048x128
modal shell tools/cloud/modal_app.py::interactive
```

**Never run two of these concurrently.** They share one Volume, and Modal Volumes are last-write-wins
on concurrent modification of the same file — two cargos in one target dir is a corruption you would
pay GPU-minutes to discover.

## The strong peers — what "better than SOTA" has to beat

§0 of the plan sets the bar, and it is not the bar this repo has been publishing against:

| Family | The bar | Where it comes from |
|---|---|---|
| GEMM | cuBLAS/cuBLASLt **with fused epilogues**, and the **CUTLASS profiler** | `::cutlass`, and the in-tree cuBLASLt peers |
| Attention | cuDNN and a **real FlashAttention build** | FA4 in the image (sm_90+), FA2 via torch SDPA's FLASH backend, `::framework --op sdpa` |
| Framework | **`torch.compile` with Inductor+Triton** | `::framework` |
| int4 / int8 | **Marlin / Machete-class** kernels | `::marlin` |

Two of those retire claims this repo currently makes:

- **"Beats PyTorch at every S" is an eager-only number** (BENCHMARKS.md:2182,2230), justified by
  Triton not installing on Windows. On Linux Triton installs, so the excuse expires and the claim has
  to be re-earned. `::framework` prints eager, `compile(default)` and `compile(max-autotune)` in the
  same run and takes the **fastest** as the peer, plus an `eager_over_peer` ratio — so the run itself
  shows how much of the old margin was the peer being weak rather than Wukong being fast.
- **"No library peer exists" for W4A16** is retired by `::marlin`. Pick the right kernel for the
  device: Machete is a Hopper kernel and is the H100 bar; Marlin is an Ampere kernel, documented as
  weak on H100, and is the A100 bar. Reporting either off its own architecture is a strawman, in one
  direction or the other, and `::marlin` says so out loud when you do it.

### Everything is pinned, and the pins are the point

`modal_app.py`'s pin block is the only place a peer version is decided (torch 2.13.0+cu129,
flash-attn-4 4.0.0b25, vLLM 0.26.0, CUTLASS v4.6.1, flash-attn 2.8.3.post1). **A peer whose version
floats is a peer that can change the answer without the benchmark changing** — and this repo has
already been bitten: `wukong_xbench`'s `detect_torch` picks the *newest* torch on the box by
`max_by(version_key)`, so two rounds a month apart can silently race two different peers. Nothing
here works that way: `::build_peers` records what it staged into `/persist/peers.json` and
`::device_info` prints it, so a round log names the peer it actually ran against.

Every pin is also **verified where it is cheap**: the torch/Triton/FA4 install is asserted at
image-build time on a CPU builder, so a wrong pin costs a build log rather than a metered hour.

### A missing peer is a failure, not a shrug

Three mechanisms, deliberately distinct:

| Switch | Means | Enforced by |
|---|---|---|
| `WUKONG_GPU_REQUIRED=1` | the device gates must run, not skip | `gpu.rs`'s `with_gpu` / `diff::skip_or_fail` |
| `WUKONG_PEER_REQUIRED=1` | the **dlopen-able** peers (NVRTC/cuBLAS/cuBLASLt/cuDNN) must load | `gpu.rs`'s `peer_gate` |
| `WUKONG_STRONG_PEERS=<list>` | this round **declares** which §0 bars it is measuring against | `baselines::strong_peer_gate` |

`WUKONG_STRONG_PEERS` is separate from `WUKONG_PEER_REQUIRED` on purpose. The latter already means
"the libraries must load" and every existing round sets it; overloading it so that it *also* demanded
a CUTLASS profiler would break every round that legitimately does not need one. The former is a claim
about what is being measured against, and it is checked: `torch-compile`, `flash-attn`, `cutlass`,
`marlin`, or `all`. **An unknown name is an error** — a typo that quietly meant "require nothing" is
exactly the silent skip the mechanism exists to remove.

Neither variable is baked into the image env. Phase 1 §1 requires the first pass to run *without*
escalation so a missing library is reported rather than failing the whole suite, and a round's peer
claim is a per-round fact, not a property of an image.

### The order to run things (and what each costs)

```powershell
# 1. Provenance. Free-ish, seconds. Also prints the peer manifest and the resolution table.
$env:WK_GPU="L4"; modal run tools/cloud/modal_app.py::device_info

# 2. Compile the workspace on CPU. No GPU attached.
modal run tools/cloud/modal_app.py::build --release

# 3. Stage the heavy peers on CPU. ~1-2 h of $1/hr CPU, ONCE, then never again.
#    --cutlass-arch defaults from WK_GPU; 90a for Hopper, 80 for A100, 89 for L4/L40S.
$env:WK_GPU="H100"; modal run tools/cloud/modal_app.py::build_peers

# 4. Prove the peers on the CHEAPEST device that can do it, and warm the Inductor cache there.
$env:WK_GPU="L4"; modal run tools/cloud/modal_app.py::peers --require all

# 5. Only now, on the target: the suite, then the numbers.
$env:WK_GPU="H100"; modal run tools/cloud/modal_app.py::test --release --peers
$env:WK_GPU="H100"; modal run tools/cloud/modal_app.py::framework --op gemm --shapes 4096x4096x4096
```

Step 3 is where "never pay twice for the same fact" earns its keep: the CUTLASS profiler is a 20-45
minute compile and the FlashAttention wheels are 20-90 minutes, **none of which touches a device** —
nvcc compiles *for* an architecture, it does not need one. On an H100 that same work would cost ~80×
as much and produce a byte-identical artifact. Step 4 is the other half: `max-autotune` compiles for
minutes on the first call for each new shape, so pay it at $0.80/hr on an L4 and let
`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` on the Volume make every later round a cache hit.

The first command after this change **rebuilds the image** (the torch + FA4 venv is a new layer, so
the pull and the ~9 GB install are paid once, on Modal's CPU builder, at $0). Later runs hit the
layer cache; only a changed pin re-runs that one layer, because it is deliberately last.

**The one trap worth checking by hand:** a `cutlass_profiler` built for the wrong arch still runs and
still prints numbers. On Hopper a plain-`90` (or an sm_80) build silently omits the wgmma kernels —
it *understates* the peer and hands Wukong a win it did not earn. `::build_peers` records the arch it
built and `::peers` refuses to proceed when it does not match the device.

**Where the wheels land.** The torch/Triton/FA4 venv is in the *image*, so every container has it.
The FA2/FA3 wheels are Volume artifacts, and a container's image filesystem is per-container — so
they cannot be installed at build time and `::peers` installs them (`--no-deps --no-index`, seconds,
no network) before it probes. Without that step a `--fa2` build would leave an artifact nothing can
import while the manifest reported it staged: the bar would *look* present and not be, which is the
one failure mode this directory exists to prevent.

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
