"""Modal app for running Wukong's GPU backend on datacenter GPUs.

Phase 0/1 of GPU_RETARGET_PLAN.md. This file does NOT change any compiler code — it only gets the
existing tree onto a rented GPU so the ~130 device-executing correctness gates can actually run.

Design decisions (each costs real money if got wrong):

* **Build on CPU, run on GPU.** Compiling 21 crates with `--features gpu` takes minutes; a Modal
  CPU core is ~$0.047/hr while an H100 is $3.95/hr. So `build` runs on CPU and writes into a
  persistent Volume; `test`/`bench` run on the GPU with a warm target dir and only link + execute.
  The GPU entry points **refuse to start** if the Volume has no prebuilt test binary for the
  profile they were asked for (`_require_prebuilt`) — compiling the workspace on metered silicon is
  the single most expensive mistake this harness can make, so it is a hard error, not a comment.
* **Source is mounted at runtime, not baked into the image.** A source edit costs an upload of a few
  MB, never an image rebuild. Combined with the Volume-backed `CARGO_TARGET_DIR`, rebuilds are
  incremental across sessions — *provided* the mounted files' mtimes are stable across containers,
  which they are not by construction (see `_stamp_sources`, which makes them stable).
* **`nvidia/cuda:*-cudnn-devel`** so the honest peers (NVRTC, cuBLAS, cuBLASLt, cuDNN) are all
  present. On the laptop these needed a hand-staged `tools/cuda-redist` on PATH; here they are the
  system libraries and `cudarc`'s `dynamic-loading` finds them by soname.
* **The STRONG peers are staged, pinned and verified OFF the GPU.** GPU_RETARGET_PLAN.md §0 sets a
  harder bar than the dlopen-able libraries: `torch.compile` with Inductor+Triton (eager PyTorch is
  **not** a bar), a real FlashAttention build, CUTLASS's profiler, and Marlin/Machete-class int4 —
  see `docs/gpu/derive/D5_peer_builds.md`, which is where every command below comes from. Every one
  of them installs or compiles on a **CPU** container at ~$0.05/hr, is **version-pinned in the block
  below** (a peer whose version floats can change the answer without the benchmark changing), and is
  **verified at image-build time**, so a wrong pin costs $0 and a build log rather than a metered
  hour. Light, always-needed peers live in the image; the multi-GB / multi-hour ones
  (`cutlass_profiler`, the vLLM venv, the FA2/FA3 wheels) are **Volume artifacts built once** by
  `::build_peers` — §0's "never pay twice for the same fact", which here is worth real money.
* **A missing strong peer is loud, not a shrug.** `::peers` runs the whole battery and fails; the
  Rust suite participates through `WUKONG_STRONG_PEERS` (`baselines::strong_peer_gate`), which is a
  *declaration of what this round is measuring against* and is deliberately separate from
  `WUKONG_PEER_REQUIRED` (that one already means "the dlopen-able libraries must load" and every
  existing round sets it). Neither is baked into the image env: Phase 1 §1 requires the first pass to
  run **without** peer escalation so a missing library is reported rather than failing the suite, and
  a round's peer claim is a per-round fact, not an image fact.
* **Every session starts with `device_info`.** GPU_RETARGET_PLAN.md §6.1 requires a provenance block
  (CC, SM count, SMEM/SM, L2, VRAM, driver, MIG state) verified against spec before any number is
  believed. It is also the cheapest possible smoke test that nvcc and the driver both work.
* **Explicit `cpu=`/`memory=` on every function.** Modal's default request is **0.125 cores and
  128 MiB** and a container only exceeds that "if the worker has available CPU or memory"
  (modal.com/docs/guide/resources, fetched 2026-08-07). The device suite's two corpus gates compile
  357 `.wk` programs x2 opt levels **on the CPU** while the GPU idles, so a starved share multiplies
  metered GPU-minutes; CPU at $0.047/core/hr is ~1.2% of an H100-hour per core.

Usage (see tools/cloud/README.md for the full walkthrough):

    modal setup                                    # once, browser auth
    WK_GPU=L4    modal run tools/cloud/modal_app.py::device_info
                 modal run tools/cloud/modal_app.py::build          # no GPU attached
                 modal run tools/cloud/modal_app.py::build_peers    # no GPU attached
                 modal run tools/cloud/modal_app.py::ptxas          # no GPU attached
    WK_GPU=L4    modal run tools/cloud/modal_app.py::peers
    WK_GPU=L4    modal run tools/cloud/modal_app.py::test
    WK_GPU=H100  modal run tools/cloud/modal_app.py::framework --op gemm     # torch.compile bar
    WK_GPU=H100  modal run tools/cloud/modal_app.py::cutlass                 # CUTLASS GEMM bar
    WK_GPU=H100  modal run tools/cloud/modal_app.py::marlin                  # int4 bar
    WK_GPU=H100  modal shell tools/cloud/modal_app.py::interactive

`WK_GPU` is read at import time because Modal fixes a function's GPU at decoration time; there is no
runtime GPU switch. It is also **baked into the image env**, because Modal does not forward local
environment variables into containers (modal.com/docs/guide/environment_variables) — without that,
in-container code would always read the module default and mis-report which SKU was requested.
Changing it re-runs only the env + mount layers of the image (seconds, on a CPU builder, no GPU).

Modal's menu and what each costs (per-second rates re-verified against modal.com/pricing on
2026-08-07; $/hr = rate x 3600) — note the column that matters here is the ISA, not the speed:

    T4           $0.59   sm_75 Turing
    L4           $0.80   sm_89 Ada         <- cheapest sm_89: 37 free hrs/month
    A10          $1.10   sm_86 Ampere
    L40S         $1.95   sm_89 Ada         <- 142 SMs; the SM-scaling stepping stone
    A100-40GB    $2.10   sm_80 Ampere      <- backward-target validation only
    A100-80GB    $2.50   sm_80 Ampere
    RTX PRO 6000 $3.03   sm_120 Blackwell  <- same ISA as an RTX 5090 (real-user relevance)
    H100         $3.95   sm_90 Hopper      <- THE competitive target (wgmma + TMA)
    H200         $4.54   sm_90 Hopper      <- identical ISA to H100, ~1.4x memory bandwidth
    B200         $6.25   sm_100 Blackwell  <- needs tcgen05: a third codegen family, not yet
    B300         $7.10   sm_103 Blackwell
    (CPU $0.0472/core/hr, memory $0.0080/GiB/hr — both billed on top of the GPU rate)

`gpu="H100"` may be served an **H200** (Modal upgrades silently; `"H100!"` opts out) — which is why
the provenance gate keys off the *probed device name*, never off the requested SKU.
"""

import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import modal

# --------------------------------------------------------------------------------------------
# Configuration (environment-driven, because Modal binds GPU/timeout at decoration time)
# --------------------------------------------------------------------------------------------

WK_GPU = os.environ.get("WK_GPU", "L40S")
"""Which GPU to attach. L40S is the sm_89 stepping stone: SAME architecture as the dev RTX 4050, so
today's PTX runs unmodified (GPU_RETARGET_PLAN.md §3). Start there, not on an A100."""

WK_TIMEOUT = int(os.environ.get("WK_TIMEOUT", "7200"))
"""Wall-clock cap on the metered functions. Modal allows 1 s .. 24 h and **kills the container** at
the cap (`modal.exception.FunctionTimeoutError`); every metered second before the kill is still
billed, so the timeout is a *cost ceiling*, not a safety feature — `_meter` prints what that ceiling
costs on the selected SKU before any work starts.

Raised 3600 -> 7200 because plan §8 risk 6 says the corpus gates' wall time on this hardware is
**unknown**, and the first `::test` losing its whole run at the 1-hour mark would waste every metered
minute it had already spent. The Volume itself survives a timeout: Modal runs background Volume
commits "every few seconds" plus "a final snapshot and commit on container shutdown"
(modal.com/docs/guide/volumes), so a killed run loses at most the last few seconds of `target/`."""

WK_CPU = float(os.environ.get("WK_CPU", "8.0"))
"""CPU cores *reserved* per container. Modal's default is 0.125 cores with bursting only if the
worker happens to be idle; the corpus gates are pure CPU compile work, so this is the difference
between the GPU waiting on one core and on eight. Priced at $0.0472/core/hr = ~$0.38/hr at 8.0,
which pays for itself the moment it saves ~30 minutes of L40S time."""

WK_MEM_MIB = int(os.environ.get("WK_MEM", "16384"))
"""Memory reserved per container (MiB). Default request is 128 MiB; rustc linking
`wukong_codegen_gpu` (a ~20k-line `gpu.rs` plus the PTX generator families) needs GiB, and an
OOM-kill on the metered box throws away the whole run."""

WK_PEER_CPU = float(os.environ.get("WK_PEER_CPU", "16.0"))
WK_PEER_MEM_MIB = int(os.environ.get("WK_PEER_MEM", "32768"))
WK_PEER_TIMEOUT = int(os.environ.get("WK_PEER_TIMEOUT", "14400"))
"""`::build_peers` only: a **CPU-only** container that compiles CUTLASS and, optionally, the
FlashAttention wheels. Wider and longer than the defaults on purpose, and still cheap — 16 cores +
32 GiB is ~$1.01/hr against an H100's $3.95 *plus* the same CPU, and every minute here is a minute
the metered box never spends. The 32 GiB is not slack: nvcc at `MAX_JOBS=16` is the classic FA
build OOM (D5 §3.2), and an OOM-killed 40-minute build has to be paid for twice. The 4 h cap is a
cost ceiling for the worst case (an unfiltered-ish CUTLASS plus FA3); `--force`-free reruns are
seconds because every artifact is skipped if it is already on the Volume."""

WK_CUDA_TAG = os.environ.get("WK_CUDA_TAG", "12.9.2-cudnn-devel-ubuntu22.04")
# 12.9 is the deliberate ceiling, not an oversight: cudarc 0.16.6 has no CUDA-13 bindings (its
# feature list and dlopen candidates stop at the .so.12 line), the nvidia-*-cu12 pip wheels end at
# 12.9.x, and torch 2.13.0+cu129 is the newest CUDA-12 build — so 12.9.2 aligns the image toolkit
# with every peer the harness compiles against it. Re-verified live on the Docker Hub v2 tag API
# 2026-08-07: the tag exists, last pushed 2026-05-20, 5.93 GB, amd64 + arm64.
# A `-runtime` (non-devel) tag would be a silent downgrade: it ships no unversioned `.so` dev
# symlinks, and cuDNN can ONLY be reached through one (see the run_commands note below).

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
REMOTE_SRC = "/wukong"
PERSIST = "/persist"

# --------------------------------------------------------------------------------------------
# PEER PINS — the one place a peer's version is decided
# --------------------------------------------------------------------------------------------
#
# **A peer whose version floats is a peer that can change the answer without the benchmark
# changing.** This repo has already been bitten by exactly that: `wukong_xbench`'s `detect_torch`
# (`crates/wukong_xbench/src/model.rs:1274-1286`) probes several interpreters and picks
# `max_by(version_key)` — the NEWEST torch on the box wins, silently, and two rounds a month apart
# can be measured against two different peers. Nothing here may work that way. Every pin below is an
# exact version, is recorded into the Volume manifest by `::build_peers`, and is printed into the
# provenance block by `::device_info` so a round log names the peer it actually raced.
#
# Live-verified 2026-08-09 against PyPI's JSON API, download.pytorch.org and the GitHub releases API
# (D5 verified the same set on 2026-08-06; these are the re-checks that mattered):
#   * torch 2.13.0+cu129   — `torch-2.13.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl` is in the
#                            cu129 index. NOT the PyPI default, which for >=2.11 is cu130 and would
#                            drag a second CUDA-13 copy of cuBLAS/cuDNN in beside the 12.9 system
#                            libraries the Rust harness dlopens (D5 §2.1).
#   * flash-attn-4 4.0.0b25 (2026-08-05) — pure-`py3-none-any` wheel, JITs through CuTeDSL, so it
#                            needs no CUDA compiler. **Every release is a pre-release**, so `--pre`
#                            (or an exact pin, which is what we do) is mandatory or pip resolves
#                            nothing and it looks unavailable (D5 §9 pitfall 6). SM90/SM100 only.
#   * vLLM 0.26.0          — hard-pins `torch==2.11.0` (a cu130 wheel), so it MUST live in its own
#                            venv or it silently downgrades the torch.compile bar (D5 §9 pitfall 10).
#   * CUTLASS v4.6.1       — the tag D5's build recipe and kernel filters were verified against.
#                            v4.6.2 landed 2026-08-08; bumping is a one-line env override, but a
#                            one-day-old patch release is not what a pinned instrument should adopt
#                            without a reason.
#   * flash-attn 2.8.3.post1 — no prebuilt wheel exists for torch 2.13 (assets stop at torch 2.8/cu12
#                            and 2.9/cu13), so this one is a source build; see `::build_peers --fa2`.
# Triton is deliberately NOT pinned here: it is a hard `install_requires` of torch on Linux, so torch
# pins it (2.13.0 -> 3.7.1) and a second pin could only ever conflict. Its resolved version is
# recorded in the manifest instead.
WK_TORCH = os.environ.get("WK_TORCH", "2.13.0+cu129")
WK_TORCH_INDEX = os.environ.get("WK_TORCH_INDEX", "https://download.pytorch.org/whl/cu129")
WK_FA4 = os.environ.get("WK_FA4", "4.0.0b25")
WK_FA2 = os.environ.get("WK_FA2", "2.8.3.post1")
WK_VLLM = os.environ.get("WK_VLLM", "0.26.0")
WK_CUTLASS_TAG = os.environ.get("WK_CUTLASS_TAG", "v4.6.1")

TORCH_VENV = "/opt/torch-venv"
"""In the IMAGE, not the Volume: the FA2/SDPA peer the existing harness already drives lives here, so
every GPU container needs it, and an image layer is cached per worker while a Volume read is not."""

VLLM_VENV = f"{PERSIST}/vllm-venv"
VLLM_SRC = f"{PERSIST}/vllm-src"
"""On the VOLUME, not in the image: ~9 GB that exactly one entry point uses. Built from the *same
image's* interpreter, so the recorded `home =` in `pyvenv.cfg` resolves identically in every
container -- which is also why `--copies` is no longer the default spelling: it removed a symlink
without removing that image dependency, and on a standalone CPython it removed the interpreter's own
`$ORIGIN/../lib` libpython too (exit 127, measured 2026-08-09). See `_stage_vllm_venv`, which tries
each spelling and *verifies* it before accepting it."""

PEER_MANIFEST = f"{PERSIST}/peers.json"
"""What is actually staged, and at which version — written by `::build_peers`, printed by
`::device_info`, and checked by `::peers`. Provenance §6.1 wants the peer versions in the round log,
and the CUTLASS arch check below needs a place to remember which arch the binary was built for."""

# Local paths never worth uploading. The repo is ~12 MB of source under a ~90 GB working tree, so
# this list is what makes the mount fast rather than impossible.
IGNORE = [
    "**/.git/**", ".git/**", ".git",
    "**/.claude/**", ".claude/**", ".claude",          # 59 GB of agent worktrees
    "**/target/**", "target/**", "target",             # 18 GB
    "target-*/**", "target-*",                         # alternate CARGO_TARGET_DIRs
    "tools/cuda-redist/**", "tools/cuda-redist",       # 11 GB of vendor DLLs; the image has these
    "tools/torch-venv/**", "tools/torch-cuda-venv/**",
    "data/**", "data",                                 # ~477 MB of exported GPT-2 weights
    # Round LOGS, and they must be ignored for a structural reason, not to save bytes. The mount is
    # `copy=False`, and Modal aborts the whole run with "<path> was modified during build process"
    # if any mounted file changes while the image is building. The runbook (docs/gpu/phase1-runbook
    # .md §2.1/§2.2) writes the provenance header into `bench/gpu/<dev>/<date>-s<N>-*.log` and then
    # tees the command's output into that same file -- i.e. it writes inside the mount *during* the
    # run it is logging. Ignoring the directory is what makes the documented logging protocol legal.
    # Only `bench/gpu/**`: `bench/kernels` is a real corpus and must keep shipping.
    "bench/gpu/**", "bench/gpu",
    "**/__pycache__/**", "**/*.pyc",
    "**/*.exe", "**/*.o", "**/*.obj", "**/*.pdb",
    "**/io_test_*.bin",
    ".vscode/**", ".idea/**",
]

app = modal.App("wukong-gpu")

# One Volume holds the cargo target dir AND the crates.io registry, so neither is re-paid per run.
# NEVER run two of these functions against it concurrently: Modal Volumes are last-write-wins on
# concurrent modification of the same file (modal.com/docs/guide/volumes), and two cargos sharing
# one target dir is a corruption you would pay to discover.
build_vol = modal.Volume.from_name("wukong-build", create_if_missing=True)

# The interpreter that builds every peer venv, resolved once in the shell rather than assumed.
# Modal's `add_python=` puts a standalone CPython on the image, and the base Ubuntu 22.04 also has
# one; which name wins depends on PATH order, and a venv built from the wrong one is a wrong-Python
# peer. Resolving explicitly (and printing it into the build log) makes the choice reviewable, and
# `python3-venv` is apt-installed purely so the system-Python fallback is not a dead end.
_PICK_PY = 'PY="$(command -v python3.12 || command -v python3.11 || command -v python3)"; "$PY" -V'

# Install torch, Triton (a hard dep of torch on Linux) and FlashAttention-4 into ONE venv, then
# **prove the pins resolved** — on a CPU builder, where being wrong costs a build log instead of a
# metered GPU hour. flash-attn-4 is checked through `importlib.metadata` rather than by importing it:
# it JITs CuTeDSL kernels and the import path is not something to exercise on a device-less builder.
_TORCH_VENV_CMD = (
    f"set -eu; {_PICK_PY}; "
    f'"$PY" -m venv {TORCH_VENV} && '
    f"{TORCH_VENV}/bin/pip install --no-cache-dir -U pip && "
    f"{TORCH_VENV}/bin/pip install --no-cache-dir --index-url {WK_TORCH_INDEX} torch=={WK_TORCH} && "
    f"{TORCH_VENV}/bin/pip install --no-cache-dir numpy ninja packaging && "
    f"{TORCH_VENV}/bin/pip install --no-cache-dir --pre flash-attn-4=={WK_FA4} && "
    f"{TORCH_VENV}/bin/python -c \"import torch,triton,importlib.metadata as m;"
    f"print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__,"
    f"'fa4',m.version('flash-attn-4'))\""
)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{WK_CUDA_TAG}", add_python="3.11")
    .apt_install(
        "curl", "ca-certificates", "build-essential", "pkg-config", "git",
        # cmake + ninja build the CUTLASS profiler and the FlashAttention wheels; python3-venv and
        # python3-dev are the fallback interpreter's venv support and headers (a source-built FA
        # extension needs Python.h). All apt, all on the CPU builder, all $0.
        "cmake", "ninja-build", "python3-venv", "python3-dev",
    )
    .run_commands(
        # No --component here, deliberately: rustup-init rejects space-separated component lists
        # ("error: unexpected argument 'clippy' found" killed the first image build), and the cloud
        # image only builds and tests — fmt/clippy run in CI, never on metered time.
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable",
        # cuDNN's Linux soname is `libcudnn.so.9`, and that is the ONE name cudarc never asks for:
        # `get_lib_name_candidates` (cudarc-0.16.6/src/lib.rs:112-147) synthesizes, for base "cudnn"
        # on Linux, `libcudnn.so`, `libcudnn64*.so` (Windows shapes) and `libcudnn.so.{12,11,10,1}`.
        # So the unversioned dev symlink is the only reachable name. The `-cudnn-devel` image ships
        # it via `libcudnn9-dev-cuda-12` (`NV_CUDNN_PACKAGE_DEV`, verified in the tag's layer env),
        # making this line a no-op there — it exists so a `-runtime`/`-cudnn-runtime` value of
        # WK_CUDA_TAG degrades to "cuDNN peer works" instead of "cuDNN peer silently skips".
        "if [ ! -e /usr/lib/x86_64-linux-gnu/libcudnn.so ] && "
        "   [ -e /usr/lib/x86_64-linux-gnu/libcudnn.so.9 ]; then "
        "  ln -s libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so; "
        "fi; ldconfig",
        # The strong framework + attention bars. Last, so a re-pin re-runs only this layer, and the
        # rustup/cuDNN layers above stay cached.
        _TORCH_VENV_CMD,
    )
    .env(
        {
            # The base image's own PATH, plus cargo. `/usr/local/nvidia/bin` is where some container
            # runtimes inject `nvidia-smi`, so keep it rather than replacing the image's value.
            "PATH": "/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:"
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            # cudarc dlopens by soname. This is the base image's own LD_LIBRARY_PATH
            # (`/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64`, read off the
            # published layer metadata) plus the multiarch dir where cuDNN and the injected
            # `libcuda.so.1` live. The earlier value dropped the two `/usr/local/nvidia` entries,
            # which is where a container runtime may place the driver.
            #
            # ** NEVER add /usr/local/cuda/lib64/stubs. ** It holds a *stub* `libcuda.so`, and
            # `libcuda.so` is literally cudarc's FIRST driver candidate — the stub would win over the
            # injected real driver and every call would fail with a symptom that reads as "no GPU".
            # The devel image legitimately puts the stubs dir on `LIBRARY_PATH` (gcc's *link*-time
            # search path); that is a different variable and must never be copied into this one.
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:"
                               "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
            # Baked so in-container code knows which SKU the client asked for: Modal does not
            # forward local env vars into containers. Observed, not theorized — the first L4 run
            # had the container re-import fall back to the module default ("L40S"), and the
            # provenance gate then compared a healthy 58-SM L4 against 142 and cried MISMATCH.
            "WK_GPU": WK_GPU,
            "CARGO_TERM_COLOR": "always",
            "RUST_BACKTRACE": "1",
            # --- where the strong peers live, for `baselines.rs`'s resolver ---
            # The Rust crate hardcodes no cloud path on purpose (it would be wrong the first time a
            # round ran on a VM instead of Modal), so the image is what points it at these.
            # WUKONG_TORCH_PYTHON serves the torch.compile bar, the FlashAttention bar AND the
            # existing FA2/SDPA peer: one interpreter, so a round can never time two different
            # torches. WUKONG_FA2_PYTHON is set to the same file for older invocations.
            "WUKONG_TORCH_PYTHON": f"{TORCH_VENV}/bin/python",
            "WUKONG_FA2_PYTHON": f"{TORCH_VENV}/bin/python",
            "WUKONG_FA2_PEER": f"{REMOTE_SRC}/tools/fa2_sdpa_peer.py",
            "WUKONG_CUTLASS_PROFILER": f"{PERSIST}/bin/cutlass_profiler",
            "WUKONG_VLLM_PYTHON": f"{VLLM_VENV}/bin/python",
            "WUKONG_VLLM_BENCH_DIR": f"{VLLM_SRC}/benchmarks/kernels",
            # Inductor's autotune and Triton's kernel cache on the Volume. `mode="max-autotune"`
            # compiles for minutes on the first call for each new shape, and that time is METERED —
            # persisting it is the difference between paying once and paying every round (D5 §2.2).
            "TORCHINDUCTOR_CACHE_DIR": f"{PERSIST}/inductor-cache",
            "TRITON_CACHE_DIR": f"{PERSIST}/triton-cache",
            # Deliberately ABSENT: WUKONG_PEER_REQUIRED and WUKONG_STRONG_PEERS. D5 §7 suggests
            # baking the first one; Phase 1 §1 contradicts it and Phase 1 wins — the first pass must
            # run *without* escalation so a missing library is reported instead of failing the whole
            # suite, and `::test --peers` is exactly that switch. The second is a per-round claim
            # about what is being measured against, which is not a property of an image.
        }
    )
    .add_local_dir(str(REPO_ROOT), REMOTE_SRC, ignore=IGNORE, copy=False)
)


# --------------------------------------------------------------------------------------------
# Cost metering
# --------------------------------------------------------------------------------------------

# $/hr, from the per-second rates on modal.com/pricing (fetched 2026-08-07) x 3600. Used only to
# print what a call can cost before it starts and what it did cost when it ends — plan §6.6 wants
# every round's GPU-hours and dollars in the round log, and §0 wants the operator to see the meter.
_GPU_USD_PER_HR = {
    "T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95,
    "A100": 2.10, "A100-40GB": 2.10, "A100-80GB": 2.50,
    "RTX-PRO-6000": 3.03, "H100": 3.95, "H200": 4.54, "B200": 6.25, "B300": 7.10,
}
_CPU_USD_PER_CORE_HR = 0.0472   # $0.0000131 / core / s
_MEM_USD_PER_GIB_HR = 0.0080    # $0.00000222 / GiB / s


def _sku_rate(spec: str) -> float:
    """$/hr for a `gpu=` string like `H100`, `H100!` or `L4:2` (0.0 for an unpriced/absent SKU)."""
    head, _, count = spec.partition(":")
    n = int(count) if count.isdigit() else 1
    return _GPU_USD_PER_HR.get(head.rstrip("!+").upper().replace("_", "-"), 0.0) * n


def _usd_per_hr(gpu: bool) -> float:
    return (_sku_rate(WK_GPU) if gpu else 0.0) \
        + WK_CPU * _CPU_USD_PER_CORE_HR + (WK_MEM_MIB / 1024.0) * _MEM_USD_PER_GIB_HR


class _meter:
    """Print the cost ceiling before the work and the actual spend after it.

    A rented GPU is an instrument billed by the second (plan §0); the operator should never have to
    reconstruct what a call cost from the dashboard afterwards.
    """

    def __init__(self, name: str, gpu: bool = True):
        self.name, self.gpu = name, gpu

    def __enter__(self):
        self.t0 = time.time()
        rate = _usd_per_hr(self.gpu)
        cap = rate * WK_TIMEOUT / 3600.0
        where = f"{WK_GPU} + {WK_CPU} cpu + {WK_MEM_MIB / 1024:.0f} GiB" if self.gpu \
            else f"CPU-only ({WK_CPU} cpu + {WK_MEM_MIB / 1024:.0f} GiB)"
        print(f"[meter] {self.name}: {where} = ${rate:.3f}/hr; "
              f"timeout {WK_TIMEOUT}s caps this call at ${cap:.2f}", flush=True)
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        print(f"[meter] {self.name}: {dt:.1f}s wall "
              f"= ${_usd_per_hr(self.gpu) * dt / 3600.0:.3f} (record this in the round log)",
              flush=True)
        return False


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------

_STAMP = f"{PERSIST}/src-mtime-stamp.json"


def _stamp_sources(root: str = REMOTE_SRC) -> None:
    """Give every mounted source file a **content-derived, container-stable** mtime.

    THE HAZARD. `add_local_dir(..., copy=False)` mounts the tree at container startup, and the
    client's wire format for a mounted file is `MountFile{filename, sha256_hex, size, mode}`
    (`modal_proto.api_pb2`, inspected 2026-08-07) — **there is no mtime field**, so the local
    modification times are not transmitted and whatever the container assigns is outside our control
    and may differ per container. Cargo's freshness check is mtime-based: a source file newer than
    the unit's output is stale. `build` compiles in one container and `test` runs in another, so if
    the second container's mount stamps the sources "now", *every* unit is stale and cargo rebuilds
    the whole workspace **on the metered GPU box** — the exact cost failure this harness exists to
    prevent, and one that would silently repeat on every run.

    THE FIX, which is cargo-correct in both directions: hash every file, keep the map
    `{path: [sha256, mtime]}` on the Volume, and restore the previously assigned mtime for files
    whose content is unchanged while stamping genuinely changed/new files with `now`. Unchanged
    sources therefore stay older than the artifacts built from them (no rebuild), and an edited file
    is newer than them (rebuild) — which is precisely what cargo means to test.

    The **bootstrap** run (no map yet) deliberately adopts each file's existing mtime minus ten
    seconds instead of `now`: stamping "now" would invalidate an already-warm Volume built before
    this map existed, i.e. it would cause exactly the metered full rebuild it is meant to prevent.
    The ten-second shift is also what makes the write-back verifiable on that first pass (a mount
    that ignores `utime` then shows up as a >2 s discrepancy instead of looking like a success).

    If the mount is not writable the fix cannot be applied; that is reported LOUDLY rather than
    silently accepted, because the consequence is a metered full rebuild.
    """
    if not os.path.isdir(root):
        print(f"!! {root} is not mounted — cannot stamp sources.", flush=True)
        return
    try:
        old = json.loads(pathlib.Path(_STAMP).read_text())
    except (OSError, ValueError):
        old = {}

    new: dict = {}
    now = time.time()
    bootstrap = not old
    fresh = 0
    checked_write = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "target", "__pycache__")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root)
            try:
                with open(p, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                continue
            prev = old.get(rel)
            if prev and prev[0] == digest:
                stamp = prev[1]              # same content -> same mtime as last container saw
            elif bootstrap:
                stamp = os.stat(p).st_mtime - 10.0  # adopt; never invalidate a warm Volume
            else:
                stamp = now                  # genuinely new or edited -> cargo must rebuild it
                fresh += 1
            new[rel] = [digest, stamp]
            try:
                os.utime(p, (stamp, stamp))
            except OSError as e:
                print(f"!! could not set mtimes on the mounted source ({e}). Cargo will treat the "
                      f"whole tree as new and REBUILD IT HERE — abort if this is a GPU box.",
                      flush=True)
                return
            if not checked_write:
                checked_write = True
                if abs(os.stat(p).st_mtime - stamp) > 2:
                    print("!! the mount ignored os.utime — mtimes are not stable across "
                          "containers, so cargo will rebuild on every run. Investigate before "
                          "spending GPU minutes.", flush=True)
                    return

    if not new:
        # Never overwrite a good map with an empty one: that would stamp every file "now" on the
        # next run and rebuild the workspace wherever that run happens to be.
        print("!! no source files found under the mount; leaving the stamp map alone.", flush=True)
        return
    pathlib.Path(_STAMP).write_text(json.dumps(new))
    if bootstrap:
        print(f"[src] {len(new)} files stamped (bootstrap: existing mtimes adopted, so a warm "
              f"Volume stays warm)", flush=True)
    else:
        print(f"[src] {len(new)} files stamped; {fresh} new/changed since the last run "
              f"({'full rebuild expected' if fresh > 50 else 'incremental'})", flush=True)


def _assert_no_stubs(env: dict) -> None:
    """Refuse to run with the CUDA *stub* directory on the loader path.

    `/usr/local/cuda/lib64/stubs/libcuda.so` is a link-time placeholder with no driver behind it,
    and `libcuda.so` is cudarc's first driver candidate — so a stub on `LD_LIBRARY_PATH` shadows the
    injected real driver and every failure looks like broken hardware (D5 §9 pitfall 1). Some peer
    build recipes (CUTLASS, FA2/FA3, vLLM) tell you to add it; add it for the duration of that one
    `cmake`/`pip` command, never to the environment a Wukong process inherits.
    """
    if "stubs" in env.get("LD_LIBRARY_PATH", ""):
        sys.exit("LD_LIBRARY_PATH contains a CUDA stubs directory — a stub libcuda.so would shadow "
                 "the real driver. Refusing to run; fix the environment first.")


def _prepare_cargo() -> dict:
    """Point CARGO_TARGET_DIR at the Volume and persist the crates.io registry across runs.

    rustup's `cargo` is a shim under /root/.cargo/bin, so CARGO_HOME must stay where rustup put it;
    we symlink only the registry/git caches onto the Volume instead of moving CARGO_HOME.
    """
    os.makedirs(f"{PERSIST}/target", exist_ok=True)
    for sub in ("registry", "git"):
        persisted = f"{PERSIST}/cargo-{sub}"
        local = f"/root/.cargo/{sub}"
        os.makedirs(persisted, exist_ok=True)
        if not os.path.islink(local):
            shutil.rmtree(local, ignore_errors=True)
            os.symlink(persisted, local)

    env = dict(os.environ)
    _assert_no_stubs(env)
    env["CARGO_TARGET_DIR"] = f"{PERSIST}/target"
    # Keep the JIT'd SASS cache on the Volume too: cold JIT is ~0.76 ms/module but a corpus sweep
    # loads thousands, and the cache key already includes the driver version (cubin.rs:88).
    env["WUKONG_CUBIN_CACHE"] = f"{PERSIST}/cubin-cache"
    os.makedirs(env["WUKONG_CUBIN_CACHE"], exist_ok=True)
    _stamp_sources()
    return env


def _require_prebuilt(package: str, release: bool) -> None:
    """Abort unless `build` already produced this package's test binary in this profile.

    `bench` hardcodes `--release` while `build` defaults to a debug profile, so the first `::bench`
    of a session would otherwise compile the entire workspace in release mode **on the GPU** — tens
    of minutes at the GPU rate to produce something a CPU container makes for a few cents. Cheap,
    loud failure beats an expensive silent one.
    """
    profile = "release" if release else "debug"
    found = [
        p for p in glob.glob(f"{PERSIST}/target/{profile}/deps/{package}-*")
        if os.path.isfile(p) and not os.path.splitext(p)[1] and os.access(p, os.X_OK)
    ]
    if not found:
        flag = " --release" if release else ""
        sys.exit(
            f"No {profile} test binary for {package} in the Volume. Build it on CPU first:\n"
            f"    modal run tools/cloud/modal_app.py::build{flag}\n"
            f"Refusing to compile the workspace on metered GPU time (GPU_RETARGET_PLAN.md §0)."
        )


def _run(cmd: list[str], env: dict, cwd: str = REMOTE_SRC, check: bool = True) -> int:
    """Run a command with live output, returning its exit code."""
    print(f"\n\033[1m$ {' '.join(cmd)}\033[0m", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    dt = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"\033[1m-> {status} in {dt:.1f}s\033[0m", flush=True)
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc.returncode


# --------------------------------------------------------------------------------------------
# Peer-library resolution (must ask for the EXACT names cudarc will ask for)
# --------------------------------------------------------------------------------------------

def _cudarc_candidates(base: str) -> list[str]:
    """Replicate `cudarc::get_lib_name_candidates` for this build, in order.

    Source of truth: `cudarc-0.16.6/src/lib.rs:112-147`, with `DLL_PREFIX="lib"`, `DLL_SUFFIX=".so"`,
    pointer width 64, and `CUDA_MAJOR/MINOR_VERSION = 12/6` (`build.rs` maps the `cuda-12060` feature
    to the tuple `(12, 6)` — *not* `(12, "06")`). The Windows-shaped entries are in the real list too
    and are kept here so this stays a faithful copy rather than a summary.

    The consequence that matters: for `cudnn` the only Linux-shaped candidates are `libcudnn.so` and
    `libcudnn.so.{12,11,10,1}` — **`libcudnn.so.9`, the actual cuDNN-9 soname, is never tried.** A
    probe that tests `libcudnn.so.9` (as this file used to) green-lights a run whose cuDNN peer then
    fails to load. `baselines.rs:67-71` records the same asymmetry; the D5 dossier's candidate table
    is wrong on this point.
    """
    p, s, w, major, minor = "lib", ".so", "64", "12", "6"
    return [
        f"{p}{base}{s}",
        f"{p}{base}{w}{s}",
        f"{p}{base}{w}_{major}{s}",
        f"{p}{base}{w}_{major}{minor}{s}",
        f"{p}{base}{w}_{major}{minor}_0{s}",
        f"{p}{base}{w}_{major}0_{minor}{s}",
        f"{p}{base}{w}_10{s}",
        f"{p}{base}{w}_{major}0_0{s}",
        f"{p}{base}{w}_9{s}",
        f"{p}{base}{s}.{major}",
        f"{p}{base}{s}.11",
        f"{p}{base}{s}.10",
        f"{p}{base}{s}.1",
    ]


def _mapped_paths(base: str) -> list[str]:
    """Which files this process actually mapped for `lib<base>.*` (post-dlopen ground truth).

    Prefix-anchored on purpose: a loose substring test would report `libcudart.so.12` as the answer
    for the driver probe and `libcublas.so.12` as the answer for cuBLASLt.
    """
    try:
        lines = pathlib.Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        path = line.split(" ", 5)[-1].strip()
        if os.path.basename(path).startswith(f"lib{base}.") and path not in out:
            out.append(path)
    return out


def _probe_peer_libs() -> None:
    """`dlopen` each peer library the way cudarc will, and report the file that actually answers.

    Deliberately not `ldconfig -p`: the loader cache does not include `LD_LIBRARY_PATH`, so it can
    report NOT FOUND for a library that loads fine and can list a file the process would never
    choose. `ctypes.CDLL` goes through the real `dlopen`, in this process, with this environment —
    the same decision cudarc's loader makes.
    """
    import ctypes

    print("\n--- peer libraries (exact cudarc candidate order, resolved by the real loader) ---")
    print(f"  LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
    for base, why in (
        ("cuda", "the driver — everything"),
        ("nvrtc", "Tier-A naive CUDA-C peers"),
        ("cublas", "Tier-B cuBLAS GEMM / IMMA int8"),
        ("cublasLt", "fp8 E4M3 matmul (the only fp8 peer surface)"),
        ("cudnn", "conv2d peer; unversioned .so ONLY — see _cudarc_candidates"),
    ):
        hit = None
        for cand in _cudarc_candidates(base):
            try:
                ctypes.CDLL(cand)
                hit = cand
                break
            except OSError:
                continue
        if hit is None:
            print(f"  {base:<10} NOT LOADABLE — tried {len(_cudarc_candidates(base))} candidates "
                  f"({why})")
            continue
        real = _mapped_paths(base) or ["<not in /proc/self/maps>"]
        print(f"  {base:<10} {hit:<20} -> {real[0]}   ({why})")
        for extra in real[1:]:
            print(f"  {'':<10} {'':<20}    also mapped: {extra}")
        if any("/stubs/" in r for r in real):
            print(f"  !! {base} resolved to a STUB library. Every driver call will fail and it will "
                  f"look like broken hardware. Fix LD_LIBRARY_PATH before spending anything.")


# The provenance probe. Written in CUDA C rather than parsed out of nvidia-smi because SM count,
# opt-in SMEM/SM and MIG state are what the retarget actually keys on, and nvidia-smi reports none
# of the three. Compiling it also proves nvcc works before we depend on it for the peers.
_PROBE_SRC = r"""
#include <cuda_runtime.h>
#include <stdio.h>
int main(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { printf("NO CUDA DEVICE\n"); return 1; }
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("driver_api        : %d.%d\n", drv / 1000, (drv % 1000) / 10);
    printf("runtime_api       : %d.%d\n", rt / 1000, (rt % 1000) / 10);
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        size_t freeB = 0, totB = 0;
        cudaSetDevice(d); cudaMemGetInfo(&freeB, &totB);
        int smem_optin = 0;
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, d);
        printf("--- device %d ---\n", d);
        printf("name              : %s\n", p.name);
        printf("compute_capability: sm_%d%d\n", p.major, p.minor);
        printf("sm_count          : %d\n", p.multiProcessorCount);
        printf("smem_per_block    : %zu KiB (static)\n", p.sharedMemPerBlock / 1024);
        printf("smem_optin        : %d KiB (dynamic opt-in)\n", smem_optin / 1024);
        printf("smem_per_sm       : %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
        printf("regs_per_sm       : %d\n", p.regsPerMultiprocessor);
        printf("max_blocks_per_sm : %d\n", p.maxBlocksPerMultiProcessor);
        printf("warps_per_sm      : %d\n", p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("l2_cache          : %d MiB\n", p.l2CacheSize / (1024 * 1024));
        printf("total_vram        : %zu MiB (free %zu MiB)\n", totB / (1024*1024), freeB / (1024*1024));
        /* clockRate/memoryClockRate are deprecated in CUDA 12 and read 0 on some newer parts;
           a 0 here is a stale-attribute artifact, not a throttled device. */
        printf("mem_clock         : %d MHz\n", p.memoryClockRate / 1000);
        printf("mem_bus_width     : %d bit\n", p.memoryBusWidth);
        printf("peak_bw_GBs       : %.1f\n",
               2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9);
        printf("sm_clock_max      : %d MHz\n", p.clockRate / 1000);
        printf("is_multi_gpu_board: %d\n", p.isMultiGpuBoard);
        printf("cooperative_launch: %d\n", p.cooperativeLaunch);
        printf("integrated        : %d\n", p.integrated);
    }
    return 0;
}
"""

# What each *device* must report, keyed on the name the device reports for itself — NOT on the
# requested SKU. Three reasons the name is the authority:
#   * Modal does not forward local env vars, so a container's `WK_GPU` is whatever was baked into
#     the image, and the client can request something else on the CLI.
#   * `gpu="H100"` may legitimately be served an **H200** (Modal's docs: `"H100!"` opts out of the
#     upgrade). Keying on the request would call that a fraud; keying on the name scopes the numbers
#     to the silicon that actually ran them, which is what §6.1 asks for.
#   * A MIG slice or vGPU announces itself in the name ("... MIG 3g.20gb", "GRID A100D") *and*
#     reports a reduced SM count — the pair is the detector.
#
# The `arch family` column is what actually matters to this project: each distinct sm_XX is a
# separate codegen target, NOT a speed grade.
#   sm_75  Turing      T4
#   sm_80  Ampere      A100                  } mma.sync + cp.async — what the backend emits today
#   sm_86  Ampere      A10                   }
#   sm_89  Ada         L4, L40S, RTX 4090    } SAME as the dev RTX 4050: runs unmodified
#   sm_90  Hopper      H100, H200            <- wgmma + TMA (H200 = H100 ISA, more HBM3e)
#   sm_100 Blackwell   B200                  <- tcgen05 / tensor memory: a THIRD codegen family
#   sm_120 Blackwell   RTX PRO 6000, RTX 5090 <- consumer Blackwell, distinct ISA from sm_100
#
# SM counts are the enabled-per-SKU figures, not full-die: L4 7424 CUDA cores / 128 = 58 (the AD104
# die has 60), L40S 18176 / 128 = 142, H100 SXM5 132 vs **H100 PCIe 114** (a different bin, not a
# fraud), B200 2x74 = 148, RTX PRO 6000 24064 / 128 = 188. Matched longest-prefix-first, so the
# A100/A10 and L40S/L4 substring collisions resolve correctly. A device absent from this table
# simply skips the gate (nothing is asserted about it).
_DEVICE_SPEC = [
    ("L40S", "sm_89", 142),
    ("L4", "sm_89", 58),
    ("A100", "sm_80", 108),
    ("A10G", "sm_86", 80),
    ("A10", "sm_86", 72),
    ("H100 PCIE", "sm_90", 114),
    ("H100", "sm_90", 132),
    ("H200", "sm_90", 132),
    ("B200", "sm_100", 148),
    ("T4", "sm_75", 40),
    ("RTX PRO 6000", "sm_120", 188),
]


def _spec_for(device_name: str):
    """(pattern, cc, sm_count) for a probed device name, or None if we assert nothing about it."""
    up = device_name.upper()
    for pattern, cc, sms in _DEVICE_SPEC:
        if pattern in up:
            return pattern, cc, sms
    return None


# --------------------------------------------------------------------------------------------
# Strong-peer staging: the CUTLASS arch, and the Volume manifest
# --------------------------------------------------------------------------------------------

# `sm_XX` (what a device reports, what a PTX module is tagged with) -> `CUTLASS_NVCC_ARCHS` (what
# CUTLASS's cmake wants). The `a` suffix is NOT cosmetic and NOT optional: on Hopper, `90a` is what
# enables `wgmma`/TMA, so a plain `90` build silently omits the fastest kernels and would understate
# the peer — i.e. it would hand Wukong a rigged win at exactly the shapes Phase 4 is gated on
# (D5 §4). Same story for the Blackwell families.
_CUTLASS_ARCH = {
    "sm_75": "75",
    "sm_80": "80",
    "sm_86": "86",
    "sm_89": "89",
    "sm_90": "90a",
    "sm_100": "100a",
    "sm_120": "120a",
}


def _cc_for_sku(sku: str) -> str:
    """The compute capability the requested Modal SKU should report, from `_DEVICE_SPEC`.

    The CUTLASS profiler is built on a **CPU** container, which has no device to ask, so the arch has
    to come from the SKU string. `-`/`_` become spaces first so `RTX-PRO-6000` matches the table's
    `RTX PRO 6000`. Returns "" for a SKU the table says nothing about — the caller then demands an
    explicit `--cutlass-arch` rather than guessing, because guessing wrong here is a weak peer.
    """
    head = sku.split(":")[0].rstrip("!+").upper().replace("-", " ").replace("_", " ")
    for pattern, cc, _sms in _DEVICE_SPEC:
        if pattern in head:
            return cc
    return ""


def _read_manifest() -> dict:
    try:
        return json.loads(pathlib.Path(PEER_MANIFEST).read_text())
    except (OSError, ValueError):
        return {}


def _write_manifest(update: dict) -> dict:
    """Merge `update` into the Volume's peer manifest. Read-modify-write, because `::build_peers`
    stages one artifact at a time and a later `--fa2` run must not erase the CUTLASS entry."""
    m = _read_manifest()
    m.update(update)
    m["image_cuda_tag"] = WK_CUDA_TAG
    m["torch_pin"] = WK_TORCH
    m["fa4_pin"] = WK_FA4
    os.makedirs(PERSIST, exist_ok=True)
    pathlib.Path(PEER_MANIFEST).write_text(json.dumps(m, indent=2, sort_keys=True))
    return m


def _print_manifest() -> dict:
    """The peer half of the §6.1 provenance block: what is staged, and at exactly which version."""
    m = _read_manifest()
    print("\n--- staged peers (tools/cloud/modal_app.py pins + the Volume manifest) ---")
    print(f"  image           nvidia/cuda:{WK_CUDA_TAG}")
    print(f"  torch           {WK_TORCH} (index {WK_TORCH_INDEX})  -> {TORCH_VENV}")
    print(f"  flash-attn-4    {WK_FA4}")
    if not m:
        print(f"  {PEER_MANIFEST}: absent — no Volume peer artifact has been staged yet.")
        print("  Run `modal run tools/cloud/modal_app.py::build_peers` (CPU-only, $0) first.")
        return m
    for k in sorted(m):
        print(f"  {k:<15} {m[k]}")
    return m


def _torch_env(env: dict) -> dict:
    """A child environment for a peer *subprocess*.

    D5 §9 pitfall 5: a torch venv carries its own `site-packages/nvidia/*/lib` CUDA copy, and letting
    that leak into the loader path of a Wukong process can shadow the system 12.9 libraries. The
    peers are subprocesses precisely so they can carry their own environment; this keeps the split
    honest by handing them the image's env unchanged rather than a merged one.
    """
    child = dict(env)
    child.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{PERSIST}/inductor-cache")
    child.setdefault("TRITON_CACHE_DIR", f"{PERSIST}/triton-cache")
    os.makedirs(child["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
    os.makedirs(child["TRITON_CACHE_DIR"], exist_ok=True)
    return child


def _peer_script(name: str) -> str:
    """A script from `tools/cloud/peers/`, in the mounted source tree."""
    return f"{REMOTE_SRC}/tools/cloud/peers/{name}"


# Volume wheel glob -> the module it provides. `::build_peers --fa2/--fa3` compiles these on a CPU
# container (nvcc compiles *for* an architecture; it does not need one) and leaves the `.whl` on the
# Volume.
_STAGED_WHEELS = (("flash_attn-*.whl", "flash_attn"), ("flash_attn_3-*.whl", "flash_attn_3"))


def _install_staged_wheels(env: dict) -> list:
    """Install any FlashAttention wheel `::build_peers` staged, into the image's torch venv.

    **This step is what stops a staged wheel from being a trap.** The wheel is built on a CPU
    container, but the venv it has to land in lives in the *image*, and an image's filesystem is
    per-container — so a build-time install would vanish. Without an install here, `--fa2`/`--fa3`
    would produce an artifact nothing can ever import while the manifest cheerfully reports it
    staged: the bar looks present and is not, which is the exact failure this whole file exists to
    make impossible.

    Seconds, from a local file, no network. `--no-deps` is deliberate — the dependencies are already
    in the venv, and letting pip resolve them here could pull a *different* torch and silently
    demote the framework bar (D5 §9 pitfall 10). Already-importable wheels are skipped.
    """
    installed = []
    for pattern, module in _STAGED_WHEELS:
        found = sorted(glob.glob(f"{PERSIST}/wheels/{pattern}"))
        if not found:
            continue
        probe = subprocess.run(
            [f"{TORCH_VENV}/bin/python", "-c", f"import {module}"],
            capture_output=True, text=True, env=env,
        )
        if probe.returncode == 0:
            print(f"[peers] {module} already importable; staged wheel not reinstalled")
            continue
        rc = _run([f"{TORCH_VENV}/bin/pip", "install", "--no-deps", "--no-index", found[-1]],
                  env, cwd="/tmp", check=False)
        if rc == 0:
            installed.append(os.path.basename(found[-1]))
        else:
            print(f"!! {found[-1]} did not install. The peer that wheel provides will be reported "
                  f"missing rather than silently skipped, which is the intended behaviour — but if "
                  f"this round declared it, the round will now fail. Rebuild with "
                  f"`::build_peers --fa2 --force`.")
    return installed


# --------------------------------------------------------------------------------------------
# The vLLM venv (the Marlin / Machete int4 bar) -- staged in a VERIFIED fallback chain
# --------------------------------------------------------------------------------------------
#
# THE DEFECT, measured live on 2026-08-09. `::build_peers` staged the CUTLASS sm90a profiler and
# then died here:
#
#     + /usr/local/bin/python3.11 -m venv --copies /persist/vllm-venv
#     Error: Command '['/persist/vllm-venv/bin/python3.11', '-m', 'ensurepip', '--upgrade',
#     '--default-pip']' returned non-zero exit status 127
#
# THE DIAGNOSIS. That is a `CalledProcessError`, not an `OSError`: `venv._call_new_python` passes
# `executable=os.path.realpath(context.env_exec_cmd)` to `subprocess.check_output`, so a
# non-existent or non-executable file would have raised `FileNotFoundError`/`PermissionError`
# instead. The copied interpreter therefore DID start and exited 127 -- and 127 is what the dynamic
# loader exits with when it cannot resolve a NEEDED shared object ("error while loading shared
# libraries: ... cannot open shared object file").
#
# Which library? `/usr/local/bin/python3.11` is Modal's `add_python="3.11"`, i.e. a
# python-build-standalone `install_only` distribution. Those are built `--enable-shared`:
# `bin/python3.11` is a real ELF that NEEDs `libpython3.11.so.1.0` and finds it through an RPATH of
# `$ORIGIN/../lib`. `$ORIGIN` is derived from the realpath of the running image, so:
#
#   * a SYMLINKED venv -- `venv/bin/python3.11 -> /usr/local/bin/python3.11` -- resolves `$ORIGIN`
#     to `/usr/local/bin`, `$ORIGIN/../lib` to `/usr/local/lib`, and loads;
#   * a COPIED venv resolves `$ORIGIN` to `/persist/vllm-venv/bin`, looks in
#     `/persist/vllm-venv/lib`, finds no libpython, and the loader exits 127.
#
# `--copies` is therefore not a neutral hardening choice on this image: it is the direct cause.
#
# WHY THE `--copies` RATIONALE DOES NOT SURVIVE CONTACT. Its comment says "so nothing depends on a
# symlink surviving the Volume". But a copied venv is not image-independent either: `pyvenv.cfg`
# still records `home = /usr/local/bin`, and that is where the interpreter finds its stdlib -- which
# is exactly why the same comment block already requires the venv to be built "from the *same
# image's* interpreter". So `--copies` removes a symlink while leaving the image dependency it was
# meant to stand in for, and buys that with a hard failure. The symlink spelling depends on strictly
# less: one absolute path into the image that a copied venv depends on anyway.
#
# WHY THIS IS A CHAIN AND NOT A ONE-LINE REVERT. The above is a diagnosis, not a measurement -- it
# cannot be tested from Windows, and the orchestrator pays for each attempt. So every spelling is
# tried in order and each one is VERIFIED by actually executing the resulting interpreter before it
# is accepted, which is the step whose absence let the original failure surface three layers away
# from its cause:
#
#   1. `venv` (symlinks)            -- correct for a shared-libpython standalone build.
#   2. `venv --copies` + the libpython copied into `venv/lib` and `venv/bin` -- the `--copies`
#      spelling with its actual defect repaired, for the case where a Volume cannot store symlinks.
#   3. the system interpreter (`/usr/bin/python3`, apt `python3-venv`) with `--copies` -- a distro
#      build links its libpython from an absolute system path, so the whole RPATH class disappears.
#
# Every attempt is created `--without-pip` and has pip bootstrapped explicitly afterwards. That is
# deliberate: the original failure came out of venv's *internal* ensurepip child, which conflated
# "the interpreter cannot run" with "pip could not be installed". Separating them means the next
# round's log names which of the two actually broke.

_VLLM_VENV_MODES = ("symlinks", "copies", "system")


def _python_runs(py: str, env: dict, label: str) -> bool:
    """Does this interpreter actually execute? The check whose absence caused the 127.

    Prints the loader's own error on failure -- `error while loading shared libraries: ...` is the
    difference between "the venv spelling is wrong" and "the package install failed", and those two
    have nothing to do with each other.
    """
    if not os.path.exists(py):
        print(f"[vllm] {label}: {py} does not exist")
        return False
    try:
        p = subprocess.run([py, "-c", "import sys; print(sys.executable); print(sys.version)"],
                           capture_output=True, text=True, env=env)
    except OSError as e:
        # execve itself refused (EACCES on a noexec mount, ENOEXEC on a truncated copy). Distinct
        # from the 127 case below, and worth saying so: 127 means the process STARTED.
        print(f"[vllm] {label}: {py} could not be executed at all: {e}")
        return False
    if p.returncode != 0:
        print(f"[vllm] {label}: {py} exited {p.returncode} -- NOT USABLE")
        for stream in (p.stdout, p.stderr):
            if stream.strip():
                print("       " + _ascii(stream.strip())[:600])
        if p.returncode == 127:
            print("       127 == the dynamic loader could not resolve a NEEDED library "
                  "(classically libpython for a --copies venv over a standalone CPython).")
        return False
    print(f"[vllm] {label}: OK -- {_ascii(' / '.join(p.stdout.split()))[:200]}")
    return True


def _diagnose_interpreter(py: str, env: dict) -> None:
    """Record HOW this interpreter finds its libpython, before choosing a venv spelling.

    Pure evidence, no decisions: the next round's log should be able to confirm or refute the
    diagnosis above without anyone re-deriving it.
    """
    script = (
        "set +e\n"
        f"echo '--- {py} ---'\n"
        f"ls -l {py}\n"
        f"head -c 4 {py} | od -c | head -1\n"
        f"command -v file >/dev/null && file {py}\n"
        f"ldd {py} 2>&1 | head -20\n"
        f"command -v readelf >/dev/null && readelf -d {py} 2>/dev/null | "
        "grep -E 'RPATH|RUNPATH|NEEDED' | head -10\n"
        "true\n"
    )
    _run(["bash", "-c", script], env, cwd="/", check=False)


def _copy_libpython(base_py: str, venv: str, env: dict) -> list:
    """Put the interpreter's shared libpython where a COPIED venv binary will look for it.

    A python-build-standalone `bin/python3.11` carries `RPATH=$ORIGIN/../lib`. Copy it into a venv
    and `$ORIGIN/../lib` becomes `<venv>/lib`, which holds no libpython -- so the loader exits 127
    before `ensurepip` gets a chance to run. Copying the `.so` into BOTH `<venv>/lib` (the RPATH
    target) and `<venv>/bin` (in case the RPATH is a bare `$ORIGIN`) repairs that at its cause and
    needs no `LD_LIBRARY_PATH` at any later call site -- which matters, because the one thing this
    file must never encourage anyone to edit casually is a loader path near CUDA.

    ~20-40 MB on the Volume, paid once, only on the `--copies` route. An empty return means the
    interpreter is statically linked -- in which case `--copies` was never the problem and the log
    should say so rather than imply a fix that did nothing.
    """
    srcs: list = []
    probe = subprocess.run(
        [base_py, "-c",
         "import sysconfig as s\n"
         "print(s.get_config_var('LIBDIR') or '')\n"
         "print(s.get_config_var('INSTSONAME') or '')\n"
         "print(s.get_config_var('LDLIBRARY') or '')\n"],
        capture_output=True, text=True, env=env, check=False)
    if probe.returncode == 0:
        cols = (probe.stdout.splitlines() + ["", "", ""])[:3]
        libdir, soname, ldlibrary = (c.strip() for c in cols)
        for name in (soname, ldlibrary):
            if libdir and name and os.path.isfile(os.path.join(libdir, name)):
                srcs.append(os.path.join(libdir, name))
        if libdir:
            srcs += [p for p in sorted(glob.glob(os.path.join(libdir, "libpython*.so*")))
                     if os.path.isfile(p)]
    # `ldd` is the ground truth for what the loader actually wants; sysconfig is only what the build
    # recorded. Take both, in that order of trust. `ldd` is absent on some minimal images, so its
    # absence must not become the reason the venv could not be staged.
    try:
        ldd = subprocess.run(["ldd", base_py], capture_output=True, text=True, env=env, check=False)
        ldd_out = ldd.stdout
    except OSError as e:
        print(f"[vllm] ldd unavailable ({e}); relying on sysconfig for the libpython location.")
        ldd_out = ""
    for line in ldd_out.splitlines():
        if "libpython" in line and "=>" in line:
            path = line.split("=>", 1)[1].strip().split(" ")[0]
            if os.path.isfile(path):
                srcs.append(path)

    copied = []
    for dst_dir in (f"{venv}/lib", f"{venv}/bin"):
        os.makedirs(dst_dir, exist_ok=True)
        for src in dict.fromkeys(srcs):
            dst = os.path.join(dst_dir, os.path.basename(src))
            if os.path.exists(dst):
                continue
            try:
                shutil.copyfile(src, dst)
                os.chmod(dst, 0o755)
                copied.append(dst)
            except OSError as e:
                print(f"[vllm] could not copy {src} -> {dst}: {e}")
    if copied:
        print(f"[vllm] staged libpython for the --copies venv: {', '.join(copied)}")
    else:
        print("[vllm] no shared libpython found next to this interpreter (statically linked?) -- "
              "if the --copies route still fails, the 127 has a different cause than the RPATH.")
    return copied


def _bootstrap_pip(venv_py: str, env: dict) -> bool:
    """Install pip into a `--without-pip` venv, ensurepip first, network get-pip.py second.

    Kept separate from venv creation on purpose (see the block comment): the original failure was
    reported as an ensurepip error when the interpreter itself was what could not run.
    """
    if _run([venv_py, "-m", "ensurepip", "--upgrade", "--default-pip"], env,
            cwd="/tmp", check=False) != 0:
        print("[vllm] ensurepip failed; falling back to bootstrap.pypa.io/get-pip.py "
              "(this builder has network).")
        if _run(["bash", "-c",
                 "set -eux\n"
                 "curl -sSfL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py\n"
                 f"{venv_py} /tmp/get-pip.py\n"], env, cwd="/tmp", check=False) != 0:
            return False
    _run([venv_py, "-m", "pip", "install", "--no-cache-dir", "-U", "pip"], env,
         cwd="/tmp", check=False)
    return _run([venv_py, "-m", "pip", "--version"], env, cwd="/tmp", check=False) == 0


def _stage_vllm_venv(env: dict, base_py: str) -> tuple:
    """Create `/persist/vllm-venv` by the first spelling that DEMONSTRABLY works.

    Returns (interpreter path, mode).

    Each attempt is: wipe, create `--without-pip`, repair if the spelling needs it, execute the
    interpreter, bootstrap pip, execute pip. Only a spelling that clears all five is accepted, and
    the winner is recorded in the Volume manifest so a round log names how its int4 bar was staged.
    """
    # `/usr/bin/python3` by absolute path, NOT `which("python3")`: Modal's `add_python` puts its
    # standalone CPython on `/usr/local/bin`, which is earlier on PATH, so `which` would hand this
    # route the very interpreter the route exists to avoid. The distro build is the point -- it
    # resolves its libpython through an absolute system path with no `$ORIGIN` in sight. It is also
    # older (3.10 on Ubuntu 22.04), which is why it is last and not first.
    system_py = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else ""
    errors = []
    for mode in _VLLM_VENV_MODES:
        src_py = base_py if mode != "system" else system_py
        if mode == "system" and (not src_py
                                 or os.path.realpath(src_py) == os.path.realpath(base_py)):
            print("[vllm] skipping the 'system' route: no distinct distro interpreter here.")
            continue
        print(f"\n[vllm] === venv attempt: {mode} (from {src_py}) ===")
        shutil.rmtree(VLLM_VENV, ignore_errors=True)
        cmd = [src_py, "-m", "venv"]
        if mode != "symlinks":
            cmd.append("--copies")
        cmd += ["--without-pip", VLLM_VENV]
        if _run(cmd, env, cwd="/tmp", check=False) != 0:
            errors.append(f"{mode}: `python -m venv` itself failed")
            continue
        if mode != "symlinks":
            _copy_libpython(src_py, VLLM_VENV, env)
        venv_py = f"{VLLM_VENV}/bin/python"
        if not _python_runs(venv_py, env, f"venv/{mode}"):
            errors.append(f"{mode}: the created interpreter does not execute")
            continue
        if not _bootstrap_pip(venv_py, env):
            errors.append(f"{mode}: pip could not be bootstrapped")
            continue
        print(f"[vllm] venv spelling '{mode}' works; using it.")
        return venv_py, mode

    shutil.rmtree(VLLM_VENV, ignore_errors=True)
    sys.exit(
        "Could not create a working vLLM venv by any spelling:\n  "
        + "\n  ".join(errors)
        + "\nThe Marlin/Machete int4 bar cannot be staged, so nothing should be published against "
          "it. The diagnostics above (ldd/RPATH of the base interpreter, and each attempt's exit "
          "code) are what the next fix should be derived from -- 127 means the dynamic loader, "
          "not a missing file."
    )


# --------------------------------------------------------------------------------------------
# ptxas: the CPU-only register / SMEM / spill audit
# --------------------------------------------------------------------------------------------
#
# `ptxas` compiles FOR an architecture; it does not need one. The `nvidia/cuda:*-devel` image
# already ships it, and `::build_peers` proved on 2026-08-09 that a CPU container can build CUTLASS
# for `sm90a` for $0.258. So the question D1_h100_gemm.md section 7 item 3 leaves open --
#
#     "Actual registers/thread ptxas allocates for A3/A4. Section 2.3's estimates (~185) are derived
#      from the generator's declarations, not from cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS).
#      If ptxas spills at 128 accumulators, A3 collapses and A2 becomes the top lever."
#
# -- is answerable here for cents rather than on a rented H100. D2_h100_attention.md section 4.2
# ranks the same census FIRST of its five experiments ("$0, NO GPU, runs in CI") and D3_a100.md
# section 7 asks for it on sm_80. It is load-bearing: every register number in this campaign is
# derived from `.reg` declarations, and NVIDIA's own CUTLASS build emitted
# `(C7511) ... wgmma.mma_async instructions are serialized due to insufficient register resources`
# at 256x128x64 -- the tile class the wide-tile work just made expressible.
#
# THE CONTRACT WITH THE CRATE. This entry point does not know, and must not know, how the audit test
# is written. It touches it through exactly two things:
#
#   * `WUKONG_PTXAS` -- absolute path to the resolved ptxas. The knob already exists (`gpu.rs`
#     `gemm_cliff_ptxas_ab`, `ptx_wgmma.rs`'s list) and the test skips when it is unset.
#   * a libtest substring filter, default `ptxas` -- i.e. the audit test's NAME MUST CONTAIN
#     "ptxas". Override with `--filter` if it does not.
#
# `WUKONG_PTXAS_ARCH` / `WUKONG_PTXAS_ARCHS` are also exported, but they are ADVISORY: a test that
# ignores them is not broken, it just sweeps its own arch set, and the coverage report below reads
# the arch out of ptxas's own `Compiling entry function '<e>' for '<arch>'` line rather than out of
# what we asked for. That is why the pass loop is adaptive -- one pass, then a second pass per arch
# that pass did not produce.
#
# Everything else here is independent of the test on purpose: the self-test proves this ptxas
# accepts each `--gpu-name` before the census runs, and validates the output parser against this
# ptxas's actual `-v` format, so an empty table can never be mistaken for "no spills".

_PTXAS_ARCHS_DEFAULT = "sm_89,sm_80,sm_90a"
"""sm_89 = the dev RTX 4050 and L4/L40S; sm_80 = A100; sm_90a = Hopper AND the only target wgmma is
legal on. Register allocation is per-arch, which is the entire reason this is a sweep and not a
single number."""

# Minimum PTX ISA version each target needs. sm_90/sm_90a arrived in ISA 8.0; the Blackwell families
# in 8.7. Too LOW a `.version` for the requested target is a ptxas error, so the self-test picks per
# arch rather than assuming one version is universally accepted.
_PTX_ISA_FOR_ARCH = {
    "sm_75": "7.0", "sm_80": "7.0", "sm_86": "7.0", "sm_87": "7.4", "sm_89": "7.8",
    "sm_90": "8.0", "sm_90a": "8.0", "sm_100": "8.7", "sm_100a": "8.7",
    "sm_103": "8.8", "sm_103a": "8.8", "sm_120": "8.7", "sm_120a": "8.7",
}

# Register file per SM (32-bit registers) and the hardware warp-slot cap, for the occupancy estimate
# next to each row. Both are architecture facts, not device bins: GA100/GH100 give 64 warp slots,
# GA10x/AD10x give 48, and every one of these families has a 64 K-register file per SM.
_REGS_PER_SM = 65536
_MAX_WARPS_PER_SM = {
    "sm_75": 32, "sm_80": 64, "sm_86": 48, "sm_87": 64, "sm_89": 48,
    "sm_90": 64, "sm_90a": 64, "sm_100": 64, "sm_100a": 64,
    "sm_103": 64, "sm_103a": 64, "sm_120": 48, "sm_120a": 48,
}


def _ascii(s: str) -> str:
    """ASCII-fold a line that is about to be printed to the operator's console.

    The console-output sibling of the crate's "PTX must be pure ASCII" law: a Windows console is
    cp1252, Modal streams the container's stdout through the local CLI, and one non-ASCII byte kills
    that CLI while the container keeps billing (README, measured 2026-08-09). The crate's gates
    print `[gate] ...` marks with non-ASCII characters by design, so anything we echo from a cargo
    run gets folded. The round-log FILE on the Volume keeps the bytes unchanged.
    """
    return s.encode("ascii", "backslashreplace").decode("ascii")


def _run_tee(cmd: list, env: dict, cwd: str = REMOTE_SRC) -> tuple:
    """`_run`, but the output is captured as well as echoed. Returns (rc, text).

    Needed because this entry point's whole deliverable IS the text: it has to be parsed into a
    table and written into the round log, while still streaming live so a metered run shows
    progress.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    buf = []
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    for line in proc.stdout:
        buf.append(line)
        print(_ascii(line.rstrip("\n")), flush=True)
    proc.wait()
    dt = time.time() - t0
    status = "ok" if proc.returncode == 0 else "FAILED (%d)" % proc.returncode
    print(f"-> {status} in {dt:.1f}s", flush=True)
    return proc.returncode, "".join(buf)


def _resolve_ptxas(explicit: str = "") -> str:
    """Find `ptxas` in this image, in order of how explicit the instruction was.

    Deliberately NOT a silent skip: the whole point of this entry point is that the image has a
    ptxas, so not finding one is a hard, cheap, CPU-priced failure.
    """
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("WUKONG_PTXAS"):
        cands.append(os.environ["WUKONG_PTXAS"])
    which = shutil.which("ptxas")
    if which:
        cands.append(which)
    cands.append("/usr/local/cuda/bin/ptxas")
    cands += sorted(glob.glob("/usr/local/cuda-*/bin/ptxas"), reverse=True)
    cands += sorted(glob.glob("/opt/nvidia/*/bin/ptxas"), reverse=True)
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.realpath(c)
    sys.exit(
        "No usable `ptxas` in this image. Tried: " + ", ".join(sorted(seen)) + "\n"
        "The image is nvidia/cuda:" + WK_CUDA_TAG + ", which is a `-devel` tag and must ship one; "
        "a `-runtime` value of WK_CUDA_TAG would not. Pass --ptxas-bin <path> or fix WK_CUDA_TAG."
    )


def _selftest_ptx(arch: str) -> str:
    """A minimal, pure-ASCII kernel used to prove this ptxas accepts `--gpu-name <arch>`.

    It also pins the `-v` output format down for the parser below: if the parser cannot read THIS,
    an empty census table means "the parser does not understand this ptxas", not "no spills".
    """
    return (
        ".version " + _PTX_ISA_FOR_ARCH.get(arch, "8.0") + "\n"
        ".target " + arch + "\n"
        ".address_size 64\n"
        "\n"
        ".visible .entry wk_ptxas_selftest(\n"
        "    .param .u64 wk_p\n"
        ")\n"
        "{\n"
        "    .reg .b32   %r<3>;\n"
        "    .reg .f32   %f<4>;\n"
        "    .reg .b64   %rd<4>;\n"
        "\n"
        "    ld.param.u64 %rd1, [wk_p];\n"
        "    cvta.to.global.u64 %rd2, %rd1;\n"
        "    mov.u32 %r1, %tid.x;\n"
        "    mul.wide.u32 %rd3, %r1, 4;\n"
        "    add.s64 %rd2, %rd2, %rd3;\n"
        "    ld.global.f32 %f1, [%rd2];\n"
        "    mov.f32 %f2, 0f3F800000;\n"
        "    fma.rn.f32 %f3, %f1, %f2, %f2;\n"
        "    st.global.f32 [%rd2], %f3;\n"
        "    ret;\n"
        "}\n"
    )


_RE_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_RE_PROPS = re.compile(r"Function properties for (\S+)")
_RE_STACK = re.compile(r"(\d+)\s+bytes stack frame")
_RE_SPILL_ST = re.compile(r"(\d+)\s+bytes spill stores")
_RE_SPILL_LD = re.compile(r"(\d+)\s+bytes spill loads")
_RE_REGS = re.compile(r"Used\s+(\d+)\s+registers")
_RE_BARRIERS = re.compile(r"used\s+(\d+)\s+barriers")
_RE_SMEM = re.compile(r"(\d+)\s+bytes\s+smem")
_RE_TESTRESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored")


def _warps_reg_limited(arch: str, regs):
    """Warps/SM the register allocation alone permits. Estimate, and labelled as one in the table.

    Registers are allocated per warp in units of 8 per thread on every family in the table above, so
    the honest formula is `floor(regfile / (roundup8(R) * 32))`, capped by the hardware warp slots.
    SMEM and the CTA-per-SM cap can only push it lower, never higher -- so this column is an upper
    bound on occupancy, which is exactly the direction the D2 section 2.2 table needed checking in.
    """
    if not regs:
        return None
    alloc = ((int(regs) + 7) // 8) * 8
    if alloc <= 0:
        return None
    return min(_REGS_PER_SM // (alloc * 32), _MAX_WARPS_PER_SM.get(arch, 64))


def _parse_ptxas_v(text: str, source: str) -> list:
    """Rows out of raw `ptxas -v` output.

    The arch on each row comes from ptxas's own `Compiling entry function '<e>' for '<arch>'` line,
    never from what we asked for -- so a test that sweeps its own arch set is reported truthfully
    rather than mislabelled with our hint.

    `Function properties` blocks are emitted for non-inlined `.func`s too, so stack/spill numbers
    are attributed to the entry only when the property block names the entry itself; a spill inside
    a helper is carried separately as `spill_any` rather than silently dropped or silently
    attributed.
    """
    rows: list = []
    cur = None
    prop_name = ""
    for line in text.splitlines():
        m = _RE_ENTRY.search(line)
        if m:
            if cur:
                rows.append(cur)
            cur = {"source": source, "entry": m.group(1), "arch": m.group(2), "regs": None,
                   "smem": 0, "stack": 0, "spill_st": 0, "spill_ld": 0, "spill_any": 0,
                   "barriers": None}
            prop_name = ""
            continue
        if cur is None:
            continue
        mp = _RE_PROPS.search(line)
        if mp:
            prop_name = mp.group(1)
            continue
        own = (prop_name == cur["entry"]) or (prop_name == "")
        ms = _RE_SPILL_ST.search(line)
        if ms:
            val = int(ms.group(1))
            cur["spill_any"] = max(cur["spill_any"], val)
            if own:
                cur["spill_st"] = max(cur["spill_st"], val)
        ml = _RE_SPILL_LD.search(line)
        if ml and own:
            cur["spill_ld"] = max(cur["spill_ld"], int(ml.group(1)))
        mk = _RE_STACK.search(line)
        if mk and own:
            cur["stack"] = max(cur["stack"], int(mk.group(1)))
        mr = _RE_REGS.search(line)
        if mr:
            cur["regs"] = int(mr.group(1))
        mb = _RE_BARRIERS.search(line)
        if mb:
            cur["barriers"] = int(mb.group(1))
        mm = _RE_SMEM.search(line)
        if mm:
            cur["smem"] = max(cur["smem"], int(mm.group(1)))
    if cur:
        rows.append(cur)
    return rows


def _print_ptxas_table(rows: list) -> None:
    """The census, grouped by arch, worst register pressure first."""
    if not rows:
        print("  (no `ptxas -v` records parsed)")
        return
    hdr = ("%-46s %6s %9s %9s %7s %9s %9s %5s"
           % ("entry", "regs", "warps/SM", "smem", "stack", "spill_st", "spill_ld", "barr"))
    for arch in sorted({r["arch"] for r in rows}):
        group = [r for r in rows if r["arch"] == arch]
        group.sort(key=lambda r: (-(r["regs"] or 0), r["entry"]))
        print(f"\n  --- {arch} ({len(group)} entries) ---")
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for r in group:
            warps = _warps_reg_limited(arch, r["regs"])
            if r["spill_st"] or r["spill_ld"]:
                flag = "  <== SPILLS"
            elif r["spill_any"]:
                # Reported, not folded into the entry's own columns: a spill inside a non-inlined
                # helper is a real finding with a different fix from a spilling mainloop.
                flag = "  <== %d B spilled in a helper .func" % r["spill_any"]
            else:
                flag = ""
            name = r["entry"] if len(r["entry"]) <= 46 else r["entry"][:43] + "..."
            print("  %-46s %6s %9s %9s %7s %9s %9s %5s%s" % (
                name,
                "?" if r["regs"] is None else r["regs"],
                "?" if warps is None else warps,
                r["smem"], r["stack"], r["spill_st"], r["spill_ld"],
                "-" if r["barriers"] is None else r["barriers"], flag))
    print("\n  warps/SM is an UPPER BOUND from the register allocation alone "
          "(64K regfile, 8-reg granularity, hardware warp-slot cap); SMEM and the CTA cap can only "
          "lower it.")


# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=600,
              volumes={PERSIST: build_vol})
def device_info():
    """§6.1 provenance block. Run this FIRST in every session, before believing any number.

    Cost: seconds. Verifies the device is what was rented (not a MIG slice), that nvcc works, and
    that every peer library the code will `dlopen` actually resolves under this environment.
    """
    with _meter("device_info"):
        env = dict(os.environ)
        _assert_no_stubs(env)
        print("=" * 78)
        print(f"WUKONG GPU PROVENANCE — image was built for SKU: {WK_GPU}")
        print("=" * 78)

        _run(["nvidia-smi"], env, cwd="/", check=False)
        _run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total,"
             "clocks.max.sm,clocks.max.mem,power.limit,mig.mode.current",
             "--format=csv"],
            env, cwd="/", check=False,
        )

        os.makedirs("/tmp/probe", exist_ok=True)
        pathlib.Path("/tmp/probe/probe.cu").write_text(_PROBE_SRC)
        if _run(["nvcc", "-o", "/tmp/probe/probe", "/tmp/probe/probe.cu"], env,
                cwd="/tmp/probe", check=False) != 0:
            print("\n!! nvcc failed — the CUDA toolkit is not usable in this image.")
            print("   The compiler itself only needs the driver, but the honest peers "
                  "(NVRTC/cuBLAS/cuBLASLt/cuDNN) need this. Fix before Phase 3.")
            return

        print("\n--- CUDA device properties (the numbers the retarget keys on) ---")
        out = subprocess.run(["/tmp/probe/probe"], capture_output=True, text=True, env=env)
        print(out.stdout, flush=True)
        if out.stderr:
            print(out.stderr, file=sys.stderr, flush=True)

        # Provenance gate: refuse to proceed quietly if the silicon is not what its name claims.
        got_name = got_cc = None
        got_sm = None
        for line in out.stdout.splitlines():
            if line.startswith("name "):
                got_name = line.split(":", 1)[1].strip()
            elif line.startswith("compute_capability"):
                got_cc = line.split(":", 1)[1].strip()
            elif line.startswith("sm_count"):
                got_sm = int(line.split(":", 1)[1].strip())

        print("--- provenance gate ---")
        spec = _spec_for(got_name or "")
        if spec is None:
            print(f"device '{got_name}' is not in the spec table — nothing asserted. Add it to "
                  f"_DEVICE_SPEC (with a source for the SM count) before publishing from it.")
        else:
            pattern, want_cc, want_sm = spec
            print(f"device '{got_name}' matched '{pattern}': "
                  f"expected {want_cc}, {want_sm} SMs        got: {got_cc}, {got_sm} SMs")
            if got_cc != want_cc or got_sm != want_sm:
                print("\n!! MISMATCH. This is a MIG slice, a vGPU, or a differently-binned part "
                      "(H100 PCIe reports 114 SMs where SXM5 reports 132).")
                print("!! Occupancy/tile conclusions from this device are NOT transferable. "
                      "Publish nothing from it.")
            else:
                print("OK — full device, matches spec. Safe to tune occupancy against.")
        # "A100-80GB" -> "A100", "RTX-PRO-6000" -> "RTX", "L4:2" -> "L4": the model token that must
        # appear in the device's own name if we were served what we asked for.
        asked = WK_GPU.split(":")[0].rstrip("!+").upper().split("-")[0]
        if got_name and asked not in got_name.upper():
            print(f"note: requested '{WK_GPU}' but the device calls itself '{got_name}'. Modal may "
                  f"serve an H200 for an H100 request (use 'H100!' to opt out) — scope every number "
                  f"to the name above, not to the request.")

        _probe_peer_libs()
        _print_manifest()
        # The strong peers, through the SAME resolver the Rust harness uses — so what this prints is
        # what a round will actually find, not a second opinion that could drift from it.
        print("\n--- strong peers (GPU_RETARGET_PLAN.md §0's bar) ---")
        _run([f"{TORCH_VENV}/bin/python", _peer_script("verify_peers.py"), "--require", "none"],
             _torch_env(env), cwd="/", check=False)


@app.function(image=image, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def build(release: bool = False, driver: bool = True):
    """Compile the GPU feature on CPU, into the persistent Volume. NO GPU is attached.

    This is the cost lever: the same compile on an H100 would burn GPU-hours at ~80x the price.
    `cargo test --no-run` builds the test binaries so the GPU function only links and executes.
    Run this again after ANY source edit — `test`/`bench` refuse to compile on metered hardware.
    """
    with _meter("build", gpu=False):
        env = _prepare_cargo()
        profile = ["--release"] if release else []

        _run(["rustc", "--version"], env)
        _run(["cargo", "--version"], env)

        # The mandatory type-check gate for this crate (plain `cargo test` never builds it at all).
        _run(["cargo", "check", "--features", "gpu", "--all-targets"] + profile, env)

        # Build the device test binaries without running them.
        _run(["cargo", "test", "-p", "wukong_codegen_gpu", "--features", "gpu",
              "--no-run"] + profile, env)
        if driver:
            _run(["cargo", "test", "-p", "wukong_driver", "--features", "gpu",
                  "--no-run"] + profile, env)

        # The CPU-side gate, so a Linux-portability break is caught here and not on metered GPU time.
        #
        # `--test-threads` is MANDATORY here, not tuning. libtest defaults to one thread per core,
        # and this container has WK_CPU=8 against WK_MEM=16 GiB -- a far tighter memory-per-thread
        # ratio than the 2-4-core GitHub runners where this same suite is green. The workspace holds
        # tests that are individually memory-hostile by design: `heap_exhaustion_is_a_diagnostic_
        # not_an_allocator_abort` asks for `alloc_f32(100000000000)` (400 GB) to prove the
        # interpreter *reports* exhaustion instead of aborting, and the deep-recursion oracles run
        # inside `wukong_interp`'s 512 MiB-stack worker threads. Eight of those at once is how a
        # 2026-08-08 S1a run got SIGKILLed with exit 137 -- three times, because Modal retried it,
        # for ~2h and ~$1 without ever reaching S1b.
        threads = os.environ.get("WK_TEST_THREADS", "2")
        _run(["cargo", "test", "--workspace"] + profile + ["--", "--test-threads", threads],
             env, check=False)

        build_vol.commit()
        print("\nBuild artifacts committed to the 'wukong-build' Volume.")


@app.function(image=image, cpu=WK_PEER_CPU, memory=WK_PEER_MEM_MIB, timeout=WK_PEER_TIMEOUT,
              volumes={PERSIST: build_vol})
def build_peers(cutlass: bool = True, cutlass_arch: str = "", cutlass_kernels: str = "",
                cutlass_dir: str = "/tmp/cutlass", vllm: bool = True, fa2: bool = False,
                fa2_archs: str = "", fa3: bool = False, force: bool = False):
    """Stage the multi-GB / multi-hour strong peers onto the Volume. **NO GPU is attached.**

    This is §0's "never pay twice for the same fact" applied to the peers, and here it is worth real
    money: the CUTLASS profiler is a 20-45 minute compile and the FlashAttention wheels are 20-90
    minutes, none of which touches a device — nvcc compiles for an architecture, it does not need one.
    Doing this on an H100 would cost ~80x what it costs here for an identical artifact.

    Everything is **idempotent**: an artifact already on the Volume is reported and skipped unless
    `--force`. The torch/FA4 venv is not here at all — it is baked into the image (§ the pin block).

        modal run tools/cloud/modal_app.py::build_peers                       # cutlass + vllm
        WK_GPU=H100 modal run tools/cloud/modal_app.py::build_peers --fa3     # + the Hopper FA3 wheel
        modal run tools/cloud/modal_app.py::build_peers --cutlass-arch 80 --no-vllm

    `--cutlass-arch` defaults to the arch implied by `WK_GPU`. It is the one setting worth checking
    by hand: `90a` (not `90`) is what enables wgmma/TMA, and a plain-`90` profiler is a *weakened*
    peer that would flatter Wukong at exactly the shapes the wgmma decision rests on.

    `--cutlass-dir` is the scratch tree for a build that is *(est.)* 8-15 GB. It defaults to `/tmp`
    (the container's local disk, which is fast); point it at `/persist/...` if a container turns out
    not to have the room. It is deleted either way — that much scratch must never be left on the
    Volume, which pays for what it stores.
    """
    with _meter("build_peers", gpu=False):
        env = dict(os.environ)
        _assert_no_stubs(env)
        os.makedirs(f"{PERSIST}/bin", exist_ok=True)
        os.makedirs(f"{PERSIST}/wheels", exist_ok=True)
        staged: dict = {}

        if cutlass:
            arch = cutlass_arch or _CUTLASS_ARCH.get(_cc_for_sku(WK_GPU), "")
            if not arch:
                sys.exit(
                    f"WK_GPU={WK_GPU!r} is not in the device-spec table, so the CUTLASS arch cannot "
                    f"be derived. Pass --cutlass-arch explicitly (90a Hopper, 80 A100, 89 Ada, "
                    f"120a consumer Blackwell). Refusing to guess: a profiler built for the wrong "
                    f"arch is a weakened peer, which is worse than no peer."
                )
            out = f"{PERSIST}/bin/cutlass_profiler-sm{arch}"
            if os.path.isfile(out) and not force:
                print(f"[peers] cutlass_profiler sm{arch} already staged at {out} (--force rebuilds)")
            else:
                kernels = cutlass_kernels or (
                    # Hopper's 3.x kernels are named differently from the 2.x ones, and an unfiltered
                    # SM90 build is a multi-hour, >100 GB mistake: NVIDIA's own docs say the full
                    # SM90 instantiation set is "in the order of millions of kernels" and that
                    # generating and filtering them "alone can take hours" (D5 §4 / §9 pitfall 8).
                    "cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f16*,"
                    "cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f32*"
                    if arch.startswith("90")
                    else "cutlass_tensorop_h*gemm*,cutlass_tensorop_s*gemm_f16*"
                )
                src, bld = cutlass_dir, f"{cutlass_dir}/build"
                # D5's verified recipe, unchanged: the default `make` generator, not Ninja. Ninja is
                # installed (the FlashAttention builds genuinely need it) but CUTLASS's quickstart is
                # what was checked, and this is a build we get one paid attempt at getting right.
                script = (
                    f"set -eux\n"
                    f"df -h {os.path.dirname(src) or '/'}\n"
                    f"rm -rf {src}\n"
                    f"git clone --depth 1 --branch {WK_CUTLASS_TAG} "
                    f"  https://github.com/NVIDIA/cutlass {src}\n"
                    f"mkdir -p {bld}\n"
                    f"cd {bld}\n"
                    f"export CUDACXX=/usr/local/cuda/bin/nvcc\n"
                    f"cmake .."
                    f" -DCUTLASS_NVCC_ARCHS={arch}"
                    f" -DCUTLASS_ENABLE_TESTS=OFF"
                    f" -DCUTLASS_UNITY_BUILD_ENABLED=ON"
                    f" -DCUTLASS_LIBRARY_OPERATIONS=gemm"
                    f" -DCUTLASS_LIBRARY_KERNELS='{kernels}'"
                    f" -DCMAKE_BUILD_TYPE=Release\n"
                    f"make cutlass_profiler -j $(nproc)\n"
                    f"cp {bld}/tools/profiler/cutlass_profiler {out}\n"
                    f"chmod +x {out}\n"
                    # The build tree is 8-15 GB of scratch and must NOT be left behind, least of all
                    # on the Volume if the operator pointed --cutlass-dir there.
                    f"rm -rf {src}\n"
                )
                _run(["bash", "-c", script], env, cwd="/tmp")
            # The env var points at one stable name; keep the per-arch copies so switching SKUs back
            # and forth never re-pays a build.
            shutil.copyfile(out, f"{PERSIST}/bin/cutlass_profiler")
            os.chmod(f"{PERSIST}/bin/cutlass_profiler", 0o755)
            staged["cutlass"] = {"tag": WK_CUTLASS_TAG, "arch": arch, "path": out}
            print(f"[peers] cutlass_profiler sm{arch} ({WK_CUTLASS_TAG}) -> {out}")
            # Record it NOW rather than at the end. On 2026-08-09 the sm90a profiler compiled
            # successfully and then the vLLM venv aborted the function, so the manifest never
            # learned about a 45-minute artifact that was sitting on the Volume. The manifest is
            # what `::device_info` prints and what `::peers` checks the arch against, so losing it
            # is losing the peer. `_write_manifest` merges, so calling it per artifact is free.
            _write_manifest(staged)
            build_vol.commit()

        if vllm:
            py = f"{VLLM_VENV}/bin/python"
            mode = (_read_manifest().get("vllm") or {}).get("venv_mode", "unknown")
            # Idempotence keys on "does the staged interpreter RUN", not on "does the file exist".
            # The 2026-08-09 failure left a `/persist/vllm-venv/bin/python3.11` on the Volume that
            # existed and could not execute; `os.path.isfile` would have called that staged.
            if not force and os.path.exists(py) and _python_runs(py, env, "already-staged venv"):
                print(f"[peers] vLLM venv already staged at {VLLM_VENV}, mode={mode} "
                      f"(--force rebuilds)")
            else:
                base_py = shutil.which("python3.12") or shutil.which("python3.11") \
                    or shutil.which("python3") or sys.executable
                print(f"[vllm] base interpreter for the venv: {base_py}")
                _diagnose_interpreter(base_py, env)
                py, mode = _stage_vllm_venv(env, base_py)
                _run([py, "-m", "pip", "install", "--no-cache-dir", f"vllm=={WK_VLLM}"],
                     env, cwd="/tmp")
                _run([py, "-c", "import importlib.metadata as m,torch;"
                                "print('vllm',m.version('vllm'),'torch',torch.__version__)"],
                     env, cwd="/tmp")
            if not os.path.isdir(f"{VLLM_SRC}/benchmarks/kernels") or force:
                # `benchmark_marlin.py` / `benchmark_machete.py` import only the prebuilt wheel's
                # `_custom_ops`, so no source *build* is needed — but the scripts themselves ship in
                # the source tree, not the wheel.
                _run(["bash", "-c",
                      f"set -eux\nrm -rf {VLLM_SRC}\n"
                      f"git clone --depth 1 --branch v{WK_VLLM} "
                      f"  https://github.com/vllm-project/vllm {VLLM_SRC}"], env, cwd="/tmp")
            for s in ("benchmark_marlin.py", "benchmark_machete.py"):
                if not os.path.isfile(f"{VLLM_SRC}/benchmarks/kernels/{s}"):
                    sys.exit(f"vLLM v{WK_VLLM} has no benchmarks/kernels/{s} — the int4 bar would be "
                             f"unmeasurable. Check the tag before spending GPU time.")
            staged["vllm"] = {"version": WK_VLLM, "venv": VLLM_VENV, "src": VLLM_SRC,
                              "venv_mode": mode, "python": py}
            print(f"[peers] vLLM {WK_VLLM} -> {VLLM_VENV} (venv mode: {mode})")
            _write_manifest(staged)
            build_vol.commit()

        if fa2:
            # No prebuilt flash-attn wheel exists for torch 2.13 (assets stop at torch 2.8/cu12 and
            # 2.9/cu13), so this is a source compile. `ninja` is already in the venv: WITHOUT it the
            # build is single-threaded and takes ~2 h (upstream's own number). One arch, not the
            # default four.
            archs = fa2_archs or _cc_for_sku(WK_GPU).replace("sm_", "") or "80"
            have = glob.glob(f"{PERSIST}/wheels/flash_attn-*.whl")
            if have and not force:
                print(f"[peers] flash-attn wheel already staged: {have[0]} (--force rebuilds)")
            else:
                _run(["bash", "-c",
                      f"set -eux\n"
                      f"export MAX_JOBS=$(nproc) NVCC_THREADS=4\n"
                      f"export FLASH_ATTENTION_FORCE_BUILD=TRUE\n"
                      f"export FLASH_ATTN_CUDA_ARCHS={archs}\n"
                      f"{TORCH_VENV}/bin/pip wheel --no-build-isolation --no-deps "
                      f"  flash-attn=={WK_FA2} -w {PERSIST}/wheels"], env, cwd="/tmp")
                have = glob.glob(f"{PERSIST}/wheels/flash_attn-*.whl")
            staged["flash_attn_2"] = {"version": WK_FA2, "archs": archs,
                                      "wheel": have[0] if have else ""}

        if fa3:
            # Hopper-only, and only worth it when the round needs fp8 attention or the backward pass;
            # otherwise FA4 (already in the image) is both cheaper and the newer kernel. The
            # DISABLE_* set is the whole build-time lever list from `hopper/setup.py` — keeping only
            # fp16 forward at hdim 64/128 is what turns hours into tens of minutes.
            have = glob.glob(f"{PERSIST}/wheels/flash_attn_3-*.whl")
            if have and not force:
                print(f"[peers] flash-attn-3 wheel already staged: {have[0]} (--force rebuilds)")
            else:
                disables = " ".join(
                    f"FLASH_ATTENTION_DISABLE_{k}=TRUE"
                    for k in ("BACKWARD", "SPLIT", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP",
                              "PACKGQA", "VARLEN", "FP8", "SM80", "HDIM96", "HDIM192", "HDIM256")
                )
                _run(["bash", "-c",
                      f"set -eux\n"
                      f"rm -rf /tmp/fa3\n"
                      f"git clone --depth 1 https://github.com/Dao-AILab/flash-attention /tmp/fa3\n"
                      f"cd /tmp/fa3/hopper\n"
                      f"export MAX_JOBS=$(nproc) NVCC_THREADS=4 {disables}\n"
                      f"{TORCH_VENV}/bin/python setup.py bdist_wheel\n"
                      f"cp dist/flash_attn_3-*.whl {PERSIST}/wheels/\n"
                      f"rm -rf /tmp/fa3"], env, cwd="/tmp")
                have = glob.glob(f"{PERSIST}/wheels/flash_attn_3-*.whl")
            staged["flash_attn_3"] = {"wheel": have[0] if have else ""}

        m = _write_manifest(staged)
        build_vol.commit()
        print("\n--- peer manifest ---")
        print(json.dumps(m, indent=2, sort_keys=True))
        print(f"\nStaged to the 'wukong-build' Volume. Every later round reads these for free; "
              f"nothing above ever runs on metered silicon again.")


@app.function(image=image, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def ptxas(archs: str = _PTXAS_ARCHS_DEFAULT, filter: str = "", skip: str = "gemm_cliff_ptxas_ab",
          ptxas_bin: str = "", package: str = "wukong_codegen_gpu", release: bool = False,
          prebuilt: bool = False, self_test: bool = True, require_archs: bool = False,
          once: bool = False, tag: str = ""):
    """The register / SMEM / spill census. **NO GPU is attached -- that is the entire point.**

    `ptxas` compiles *for* an architecture; it does not need one. So the campaign's most
    load-bearing unknown -- D1_h100_gemm.md section 7 item 3, "if ptxas spills at 128 accumulators,
    A3 collapses and A2 becomes the top lever" -- costs cents on a CPU container, not an hour. D2's
    section 4.2 ranks the same census first of five experiments; D3 wants it for sm_80.

    What it does, in order:

      1. resolves `ptxas` in the image and records `ptxas --version` in the round log;
      2. self-tests each requested arch on a minimal kernel, which proves this ptxas accepts the
         `--gpu-name` AND validates the output parser against this ptxas's actual `-v` format;
      3. runs the crate's ptxas audit test with `WUKONG_PTXAS` pointed at the resolved binary, once,
         then again for any requested arch that first pass did not produce;
      4. prints the census table and writes every raw byte into `/persist/rounds/ptxas-*.log`.

    The only coupling to the crate is `WUKONG_PTXAS` plus a libtest substring filter (default
    `ptxas`, i.e. the audit test's name must contain that). `WUKONG_PTXAS_ARCH`/`_ARCHS` are also
    exported but advisory; the arch on every reported row is read out of ptxas's own output.

        modal run tools/cloud/modal_app.py::ptxas
        modal run tools/cloud/modal_app.py::ptxas --archs sm_90a --filter wgmma_ptxas_audit
        modal run tools/cloud/modal_app.py::ptxas --once --require-archs

    Cost: CPU only. The first run compiles the gpu feature if `::build` has not; pass `--prebuilt`
    to refuse to compile instead (there is no cost argument for that here the way there is on a GPU
    box -- this is the same ~$0.4/hr container `::build` itself uses -- but it keeps the census
    honest about running the same binary a `::test` round ran).
    """
    with _meter("ptxas", gpu=False):
        env = _prepare_cargo()
        # No device is attached, so escalation would fail every gate that legitimately cannot run
        # here. The audit test is device-free by construction; nothing else in this run should be
        # allowed to pretend otherwise, in either direction.
        env.pop("WUKONG_GPU_REQUIRED", None)
        env.pop("WUKONG_PEER_REQUIRED", None)
        env.pop("WUKONG_STRONG_PEERS", None)

        arch_list = [a.strip() for a in archs.split(",") if a.strip()]
        if not arch_list:
            sys.exit("--archs is empty; nothing to audit.")
        skip_list = [s.strip() for s in skip.split(",") if s.strip()]

        bin_path = _resolve_ptxas(ptxas_bin)
        env["WUKONG_PTXAS"] = bin_path
        ver = subprocess.run([bin_path, "--version"], capture_output=True, text=True, env=env)
        version_text = (ver.stdout or "") + (ver.stderr or "")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(f"{PERSIST}/rounds", exist_ok=True)
        log_path = f"{PERSIST}/rounds/ptxas-{tag or 'audit'}-{stamp}.log"
        json_path = f"{PERSIST}/rounds/ptxas-{tag or 'audit'}-{stamp}.json"
        log = [
            "=" * 78,
            "WUKONG ptxas AUDIT (CPU container, no GPU attached)",
            "=" * 78,
            f"utc            : {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
            f"image          : nvidia/cuda:{WK_CUDA_TAG}",
            f"ptxas          : {bin_path}",
            f"archs          : {', '.join(arch_list)}",
            f"filter         : {filter or '(auto: substring `ptxas`)'}",
            f"skip           : {', '.join(skip_list) or '(none)'}",
            f"package        : {package} ({'release' if release else 'debug'})",
            "note           : the source tree is mounted WITHOUT .git, so identify the tree by the",
            "                 local commit the operator ran this from.",
            "",
            "--- ptxas --version ---",
            version_text.rstrip(),
            "",
        ]
        print("=" * 78)
        print("WUKONG ptxas AUDIT -- CPU container, no GPU attached")
        print("=" * 78)
        print(f"ptxas: {bin_path}")
        print(_ascii(version_text.rstrip()))

        failures: list = []
        rows: list = []
        observed: list = []
        chosen = filter

        def flush():
            """Write the round log before anything else happens, including before an abort.

            The log IS the deliverable. Every abort below happens after this has run, so a failed
            audit still leaves the ptxas version, the arch self-test and whatever raw output it did
            get on the Volume -- otherwise the cheap failure modes would cost the evidence too.
            """
            body = log + ["", "--- parsed census ---"]
            body += [json.dumps(r, sort_keys=True) for r in rows]
            pathlib.Path(log_path).write_text("\n".join(body), encoding="utf-8", errors="replace")
            pathlib.Path(json_path).write_text(json.dumps(
                {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "ptxas": bin_path, "ptxas_version": version_text.strip(),
                 "image_cuda_tag": WK_CUDA_TAG, "archs_requested": arch_list,
                 "archs_observed": observed, "filter": chosen, "rows": rows},
                indent=2, sort_keys=True))
            # One line, the release line if ptxas printed one -- the manifest is read by a human in
            # a provenance block, and "Cuda compilation tools, release 12.9, V12.9.86" is the fact
            # a round log needs to name. The full text is in the log either way.
            vlines = [ln.strip() for ln in version_text.splitlines() if ln.strip()]
            vline = next((ln for ln in vlines if "release" in ln), vlines[-1] if vlines else "")
            _write_manifest({"ptxas": {"path": bin_path, "archs": arch_list,
                                       "log": log_path, "json": json_path, "version": vline}})
            build_vol.commit()
            print(f"\nraw ptxas -v output -> {log_path}")
            print(f"parsed census       -> {json_path}")

        def die(msg: str):
            flush()
            sys.exit(msg)

        # 1. Does THIS ptxas accept each requested target, and can the parser read its output?
        if self_test:
            print("\n--- arch self-test (a minimal kernel per target) ---")
            os.makedirs("/tmp/ptxas-selftest", exist_ok=True)
            rejected, unparsed = [], []
            for arch in arch_list:
                src = f"/tmp/ptxas-selftest/{arch}.ptx"
                pathlib.Path(src).write_text(_selftest_ptx(arch))
                proc = subprocess.run(
                    [bin_path, "-v", "-O3", f"--gpu-name={arch}",
                     "-o", f"/tmp/ptxas-selftest/{arch}.cubin", src],
                    capture_output=True, text=True, env=env, check=False)
                text = (proc.stdout or "") + (proc.stderr or "")
                log += [f"--- self-test {arch} (rc={proc.returncode}) ---",
                        _selftest_ptx(arch), text.rstrip(), ""]
                parsed = _parse_ptxas_v(text, f"selftest:{arch}")
                if proc.returncode != 0:
                    print(f"  {arch:<8} REJECTED by this ptxas (rc={proc.returncode})")
                    print("    " + _ascii(text.strip())[:400])
                    rejected.append(arch)
                    continue
                if not parsed:
                    print(f"  {arch:<8} accepted, but the `-v` output did not parse -- the census "
                          f"table below may be empty for a PARSER reason, not a spill-free one.")
                    unparsed.append(arch)
                    continue
                r = parsed[0]
                print(f"  {arch:<8} accepted; parser reads regs={r['regs']} smem={r['smem']} "
                      f"spill_st={r['spill_st']}")
            # A target this ptxas does not know is a real finding and must stop the round; ALL of
            # them failing identically is far more likely to be the probe kernel than the toolkit,
            # and that must not be reported as "this CUDA 12.9 ptxas cannot target sm_80".
            if rejected and len(rejected) == len(arch_list):
                print("  !! every target was rejected -- suspect the self-test kernel above, not "
                      "the toolkit. The census below is unaffected; re-run with --no-self-test to "
                      "silence this.")
            elif rejected:
                failures.append("self-test rejected: " + ",".join(rejected))
            if unparsed:
                failures.append("parser vs " + ",".join(unparsed))

        # 2. Which test are we actually running? Discover rather than assume a name.
        profile = ["--release"] if release else []
        if prebuilt:
            _require_prebuilt(package, release)
        elif not glob.glob(f"{PERSIST}/target/{'release' if release else 'debug'}/deps/"
                           f"{package}-*"):
            print(f"\nnote: no prebuilt {package} test binary on the Volume, so this run will "
                  f"compile it. That is CPU-priced here and warms the same Volume `::build` uses; "
                  f"pass --prebuilt to refuse instead.")

        base_cargo = ["cargo", "test", "-p", package, "--features", "gpu"] + profile
        if not chosen:
            rc, listing = _run_tee(base_cargo + ["--", "--list"], env)
            log += ["--- cargo test -- --list ---", listing, ""]
            if rc != 0:
                die("`cargo test -- --list` failed; the crate does not build here. Fix that "
                    "first -- this run cannot audit a tree that does not compile.")
            names = [ln.rsplit(":", 1)[0].strip() for ln in listing.splitlines()
                     if ln.strip().endswith(": test")]
            cands = [n for n in names if "ptxas" in n and n not in skip_list]
            if not cands:
                sample = [n for n in names if "wgmma" in n or "flash" in n or "audit" in n][:20]
                die(
                    "No test name contains `ptxas`, so the audit test could not be selected.\n"
                    "The contract is one env var (WUKONG_PTXAS) plus a libtest substring filter;\n"
                    "if the test was named without `ptxas` in it, pass --filter <substring>.\n"
                    "Names that looked related: " + (", ".join(sample) or "(none)") + "\n"
                    f"{len(names)} tests were listed in total."
                )
            chosen = "ptxas"
            print(f"\nAudit tests selected by the substring `ptxas`: {', '.join(cands)}")

        # 3. The census passes. One pass, then one more for any arch it did not produce -- so a test
        #    that sweeps arches itself is not run three redundant times, and a test that honours the
        #    hint still gets every arch.
        def one_pass(hint: str, hint_all: str):
            cmd = base_cargo + ["--", chosen, "--include-ignored", "--nocapture",
                                "--test-threads=1"]
            for s in skip_list:
                cmd += ["--skip", s]
            passenv = dict(env)
            passenv["WUKONG_PTXAS_ARCH"] = hint
            passenv["WUKONG_PTXAS_ARCHS"] = hint_all
            rc, text = _run_tee(cmd, passenv)
            log.append(f"--- census pass: WUKONG_PTXAS_ARCH={hint} "
                       f"WUKONG_PTXAS_ARCHS={hint_all} (rc={rc}) ---")
            log.append(text)
            log.append("")
            ran = sum(int(m.group(1)) + int(m.group(2))
                      for m in _RE_TESTRESULT.finditer(text))
            return rc, text, ran

        all_hint = ",".join(arch_list)
        rc, text, ran = one_pass(arch_list[0], all_hint)
        if rc != 0:
            failures.append(f"census pass {arch_list[0]}")
        rows += _parse_ptxas_v(text, f"pass:{arch_list[0]}")
        if ran == 0:
            die(
                f"The filter {chosen!r} selected no test that actually ran (libtest reported 0 "
                f"passed and 0 failed).\nThat is the silent-skip failure mode this entry point "
                f"exists to remove: an empty audit reported as a green run.\n"
                f"Pass --filter with the real substring, or check that the audit test landed "
                f"in {package}."
            )
        print(f"[ptxas] pass 1 ran {ran} test(s) and produced {len(rows)} ptxas record(s).")

        if not once:
            for arch in arch_list[1:]:
                if any(r["arch"] == arch for r in rows):
                    print(f"[ptxas] {arch} already covered by an earlier pass; not re-running.")
                    continue
                rc, text, ran = one_pass(arch, arch)
                if rc != 0:
                    failures.append(f"census pass {arch}")
                rows += _parse_ptxas_v(text, f"pass:{arch}")

        # Deduplicate: if the test sweeps its own arch set and ignores the hint, the extra passes
        # produce byte-identical records. Keep the first of each (arch, entry).
        seen = set()
        uniq = []
        for r in rows:
            key = (r["arch"], r["entry"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        dropped = len(rows) - len(uniq)
        rows = [r for r in uniq if not r["source"].startswith("selftest:")]

        # 4. The table, the answer, and the log.
        print("\n" + "=" * 78)
        print("REGISTER / SMEM / SPILL CENSUS")
        print("=" * 78)
        if dropped:
            print(f"({dropped} duplicate (arch, entry) records across passes were collapsed.)")
        _print_ptxas_table(rows)

        observed = sorted({r["arch"] for r in rows})
        missing = [a for a in arch_list if a not in observed]
        print("\n--- coverage ---")
        print(f"  requested: {', '.join(arch_list)}")
        print(f"  observed : {', '.join(observed) or '(none)'}")
        if missing:
            print(f"  MISSING  : {', '.join(missing)} -- the audit test did not emit these. "
                  f"Register allocation is per-arch, so a conclusion for one arch does not carry. "
                  f"Re-run with --filter/--archs, or --require-archs to make this an error.")
            if require_archs:
                failures.append("arch coverage: " + ",".join(missing))

        spillers = [r for r in rows if r["spill_st"] or r["spill_ld"] or r["spill_any"]]
        print("\n--- D1 section 7 item 3: does ptxas spill? ---")
        if not rows:
            print("  UNANSWERED -- no raw `ptxas -v` records were parsed out of the audit test's "
                  "output.")
            print(f"  If the test prints its own summary instead of ptxas's `-v` text, its numbers "
                  f"are in the test output above and in {log_path}; this exit code then means only "
                  f"that THIS harness could not verify them, not that the audit found nothing.")
            failures.append("no ptxas records parsed")
        elif spillers:
            print(f"  YES for {len(spillers)} of {len(rows)} entries:")
            for r in sorted(spillers, key=lambda r: -(r["spill_st"] + r["spill_ld"])):
                print(f"    {r['arch']:<8} {r['entry']:<46} regs={r['regs']} "
                      f"spill_st={r['spill_st']} spill_ld={r['spill_ld']} "
                      f"spill_in_helpers={r['spill_any']}")
            print("  A spilling accumulator tile is the case D1 says collapses A3 and promotes A2.")
        else:
            print(f"  NO -- 0 of {len(rows)} entries spill. Highest pressure per arch:")
            for arch in observed:
                worst = max((r for r in rows if r["arch"] == arch),
                            key=lambda r: (r["regs"] or 0))
                print(f"    {arch:<8} {worst['regs']} regs "
                      f"({_warps_reg_limited(arch, worst['regs'])} warps/SM max) in "
                      f"{worst['entry']}")

        flush()
        if failures:
            sys.exit("ptxas audit incomplete: " + "; ".join(failures))


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def test(peers: bool = False, release: bool = False, driver: bool = True, filter: str = "",
         strong_peers: str = ""):
    """Run the ~130 device-executing correctness gates with skips escalated to failures.

    `WUKONG_GPU_REQUIRED=1` is the whole point: without it a device-less run prints `[skip]` and
    reports `ok`, which is how GPU code stays green while doing nothing (crate CLAUDE.md rule 3).
    `peers=True` additionally requires NVRTC/cuBLAS/cuBLASLt/cuDNN to load — leave it off on the
    first run so you learn what is missing instead of failing the whole suite.

    `--strong-peers` is the §0 bar declaration: a comma list of `torch-compile,flash-attn,cutlass,
    marlin` (or `all`). It sets `WUKONG_STRONG_PEERS`, which `baselines::strong_peer_gate` turns into
    a hard failure when a declared peer is not actually on the box — so a round cannot publish
    against a weaker bar than it claims. Unset (the default) declares nothing and changes nothing.

    Two of these gates (`run_corpus_matches_interp_oracle`, `mega_corpus_matches_oracle`) compile
    357 `.wk` programs at two opt levels on the CPU, which is why the container reserves cores.
    """
    with _meter("test"):
        env = _prepare_cargo()
        _require_prebuilt("wukong_codegen_gpu", release)
        if driver:
            _require_prebuilt("wukong_driver", release)
        env["WUKONG_GPU_REQUIRED"] = "1"
        if peers:
            env["WUKONG_PEER_REQUIRED"] = "1"
        if strong_peers:
            env["WUKONG_STRONG_PEERS"] = strong_peers
            print(f"Strong-peer bar declared: {strong_peers}")
        profile = ["--release"] if release else []
        extra = ([filter] if filter else [])

        print(f"Device suite on {WK_GPU} — GPU_REQUIRED=1, PEER_REQUIRED={'1' if peers else '0'}")
        rc1 = _run(["cargo", "test", "-p", "wukong_codegen_gpu", "--features", "gpu"] + profile
                   + ["--"] + extra, env, check=False)
        rc2 = 0
        if driver:
            rc2 = _run(["cargo", "test", "-p", "wukong_driver", "--features", "gpu"] + profile
                       + ["--"] + extra, env, check=False)

        build_vol.commit()
        print(f"\ncodegen_gpu: {'PASS' if rc1 == 0 else 'FAIL'}   "
              f"driver: {'PASS' if rc2 == 0 else 'FAIL' if driver else 'skipped'}")
        if rc1 or rc2:
            sys.exit(1)


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def bench(name: str = "", package: str = "wukong_codegen_gpu", peers: bool = False,
          strong_peers: str = ""):
    """Run the `#[ignore]`d perf sweeps/benches (release, one at a time).

    These are the sweeps that re-tune tiles per architecture (Phase 1 §3 / Phase 3 §2). They need
    `--release` and `--ignored`; pass `name` to select one, e.g. name="gemm_pipe_sweep".
    **Requires `::build --release` first** — see `_require_prebuilt`.

    `--peers` sets `WUKONG_PEER_REQUIRED=1` so a sweep that cannot reach cuBLAS/cuDNN/NVRTC **fails**
    instead of printing `[skip]` and reporting green having measured nothing (`gpu.rs`'s `peer_gate`).
    Any sweep whose number is going to be published should be run with it. `--strong-peers` does the
    same for the §0 out-of-process bars; see `::test`.
    """
    with _meter("bench"):
        env = _prepare_cargo()
        _require_prebuilt(package, release=True)
        env["WUKONG_GPU_REQUIRED"] = "1"
        if peers:
            env["WUKONG_PEER_REQUIRED"] = "1"
        if strong_peers:
            env["WUKONG_STRONG_PEERS"] = strong_peers
        print(f"Sweep on {WK_GPU} — PEER_REQUIRED={'1' if peers else '0'}, "
              f"STRONG_PEERS={strong_peers or '(none declared)'}")
        args = ["cargo", "test", "-p", package, "--features", "gpu", "--release", "--"]
        if name:
            args.append(name)
        args += ["--ignored", "--nocapture", "--test-threads=1"]
        _run(args, env, check=False)
        build_vol.commit()


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def peers(require: str = "torch-compile,flash-attn", release: bool = True, warm: bool = True):
    """**The strong-peer battery. A missing peer fails this, loudly.**

    Runs D5 §8's smoke sequence in cost order, cheapest first, and stops nothing early so one run
    tells you everything that is wrong. Budget: a few minutes of metered time, most of it the first
    Inductor `max-autotune` compile — which is exactly why `--warm` writes it to the Volume, and why
    §0 says to do this run on the **cheapest** SKU (`WK_GPU=L4`) and reuse the cache afterwards.

    `--require` is the list that decides pass/fail: a peer named here and not found is an error, a
    peer not named is reported and tolerated. The default requires the two bars every round needs
    (the framework and attention bars) and leaves `cutlass`/`marlin` opt-in, because those are staged
    per-round by `::build_peers`. Pass `--require all` before publishing a GEMM or int4 number.

        WK_GPU=L4   modal run tools/cloud/modal_app.py::peers
        WK_GPU=H100 modal run tools/cloud/modal_app.py::peers --require all
    """
    with _meter("peers"):
        env = _prepare_cargo()
        _require_prebuilt("wukong_codegen_gpu", release)
        profile = ["--release"] if release else []
        m = _print_manifest()

        # A CUTLASS profiler built for the wrong arch is the subtle way this goes wrong: it runs, it
        # prints numbers, and on Hopper a plain-`90` (or an sm_80) build silently omits the wgmma
        # kernels — i.e. it UNDERSTATES the peer and hands Wukong a win it did not earn. The staged
        # arch is recorded at build time precisely so it can be checked here against the real device.
        want_cutlass = "cutlass" in require or require.strip() == "all"
        staged_arch = (m.get("cutlass") or {}).get("arch", "")
        device_arch = _CUTLASS_ARCH.get(_cc_for_sku(WK_GPU), "")
        if want_cutlass and staged_arch and device_arch and staged_arch != device_arch:
            sys.exit(
                f"The staged cutlass_profiler was built for sm{staged_arch} but {WK_GPU} wants "
                f"sm{device_arch}. That peer would be WEAKER than the real library and any "
                f"%-of-CUTLASS from it would flatter Wukong. Rebuild:\n"
                f"    modal run tools/cloud/modal_app.py::build_peers --cutlass-arch {device_arch}"
            )

        failures = []
        # Before probing anything: make the Volume's staged wheels reachable from the image's venv.
        # Cheap, local, and the difference between "the wheel exists" and "the peer exists".
        staged_wheels = _install_staged_wheels(_torch_env(env))
        if staged_wheels:
            print(f"[peers] installed from the Volume: {', '.join(staged_wheels)}")

        rc = _run([f"{TORCH_VENV}/bin/python", _peer_script("verify_peers.py"),
                   "--require", require, "--json", f"{PERSIST}/peer-verify.json"],
                  _torch_env(env), check=False)
        if rc:
            failures.append("verify_peers")

        if warm:
            # Proves Inductor+Triton really generated a kernel (not an ATen fallback) AND pays the
            # max-autotune compile into the Volume-backed cache exactly once.
            if _run([f"{TORCH_VENV}/bin/python", _peer_script("smoke_inductor.py")],
                    _torch_env(env), check=False):
                failures.append("smoke_inductor")

        # The in-tree library peers, with skips escalated (D5 §1.3). These are `#[ignore]`d benches.
        env["WUKONG_GPU_REQUIRED"] = "1"
        env["WUKONG_PEER_REQUIRED"] = "1"
        for gate in ("reproducibility_vs_cublas", "conv_vs_cudnn"):
            if _run(["cargo", "test", "-p", "wukong_codegen_gpu", "--features", "gpu"] + profile
                    + ["--", gate, "--ignored", "--nocapture", "--test-threads=1"], env,
                    check=False):
                failures.append(gate)

        # The Rust-side strong-peer gate: the same declaration a published round will make.
        env["WUKONG_STRONG_PEERS"] = require
        if _run(["cargo", "test", "-p", "wukong_codegen_gpu", "--features", "gpu"] + profile
                + ["--", "declared_strong_peers_are_actually_present", "--nocapture"], env,
                check=False):
            failures.append("strong_peer_gate")

        build_vol.commit()
        print("\n=== strong-peer battery ===")
        print(f"  required: {require}")
        print(f"  failed:   {', '.join(failures) if failures else 'nothing'}")
        if failures:
            sys.exit(
                "A required peer did not resolve. Publishing a number now would compare Wukong "
                "against a bar it did not actually race (GPU_RETARGET_PLAN.md §0)."
            )


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def cutlass(m: str = "4096", n: str = "4096", k: str = "4096", dtype: str = "f16",
            kernels: str = "", iters: int = 100, warmup: int = 20, tag: str = ""):
    """The CUTLASS-profiler GEMM bar, with cuBLAS as a same-binary control column.

    `--A=f16:row --B=f16:column` is not a style choice: that is the `A*B^T` (`nn.Linear`) contract
    Wukong's GEMM peers use (`baselines.rs`'s `cublas_gemm_nt_f16`), so the layouts match without a
    transpose fudge that would hand either side an advantage.

    Honesty note for whoever writes this up: a profiler number is a **best-of-many-kernels** number
    chosen by exhaustive search, which makes it a *stronger* bar than cuBLAS at some shapes and a
    weaker one at others. It must be labelled as such. m/n/k accept the profiler's comma lists and
    `start:end:step` ranges, so one call can sweep.
    """
    with _meter("cutlass"):
        env = dict(os.environ)
        _assert_no_stubs(env)
        prof = f"{PERSIST}/bin/cutlass_profiler"
        if not os.path.isfile(prof):
            sys.exit(
                f"No cutlass_profiler on the Volume ({prof}). Build it on CPU first — it is a "
                f"20-45 minute compile and must never happen here:\n"
                f"    modal run tools/cloud/modal_app.py::build_peers"
            )
        os.makedirs(f"{PERSIST}/rounds", exist_ok=True)
        out = f"{PERSIST}/rounds/cutlass-{tag or dtype}-{m}x{n}x{k}.csv".replace(":", "_")
        args = [prof, "--operation=Gemm", "--op_class=tensorop",
                f"--m={m}", f"--n={n}", f"--k={k}",
                f"--A={dtype}:row", f"--B={dtype}:column", f"--C={dtype}:column",
                "--accumulator-type=f32", "--providers=cutlass,cublas",
                f"--warmup-iterations={warmup}", f"--profiling-iterations={iters}",
                "--verification-enabled=true", f"--output={out}"]
        if kernels:
            args.append(f"--kernels={kernels}")
        rc = _run(args, env, cwd=f"{PERSIST}/rounds", check=False)
        build_vol.commit()
        print(f"\nCSV -> {out}")
        if rc:
            sys.exit(rc)


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def marlin(kernel: str = "", models: str = "", batch_sizes: str = "1 16 128",
           dtype: str = "float16"):
    """The int4 bar: vLLM's own Marlin / Machete kernel benchmarks.

    §0 retires this repo's "no library peer exists for W4A16" framing — `baselines.rs` measures
    Wukong's int4 decode against the Tier-A naive CUDA-C kernel only, which is not a bar.

    **Pick the right one for the device or the comparison is a strawman in the other direction:**
    Machete is built for Hopper and is the correct H100 bar; Marlin was designed for Ampere and is
    documented as weak on H100, so it is the correct A100 bar. `--kernel` defaults from `WK_GPU`.

    The scripts generate and quantize their own random weights, so **no model download happens** —
    `--models` only selects K/N *shapes* from vLLM's `weight_shapes.py`. Note that comparing to
    Wukong must be done at the operator level (same M/K/N, same group size, same activation dtype,
    both checked against the same f64 dequant-then-GEMM reference): Marlin's B is int4 packed
    8-per-int32 in a permuted tile layout, which Wukong's buffer layout does not match.
    """
    with _meter("marlin"):
        env = dict(os.environ)
        _assert_no_stubs(env)
        py = f"{VLLM_VENV}/bin/python"
        if not os.path.isfile(py):
            sys.exit(
                f"No vLLM venv on the Volume ({VLLM_VENV}). Stage it on CPU first:\n"
                f"    modal run tools/cloud/modal_app.py::build_peers"
            )
        # Existing != usable. The venv is a Volume artifact created in a different container, and
        # its interpreter can be present and unable to load (that is exactly how the 2026-08-09
        # staging failed: exit 127 from the dynamic loader). Checking costs a second HERE and saves
        # the whole metered call, so it happens before anything else touches the device.
        mode = (_read_manifest().get("vllm") or {}).get("venv_mode", "unknown")
        print(f"[marlin] staged venv mode: {mode}")
        if not _python_runs(py, _torch_env(env), "staged vLLM venv"):
            sys.exit(
                f"The staged vLLM interpreter ({py}) does not execute in this container, so the "
                f"int4 bar cannot run. Re-stage it on CPU -- the staging path tries several venv "
                f"spellings and verifies each:\n"
                f"    modal run tools/cloud/modal_app.py::build_peers --force --no-cutlass"
            )
        cc = _cc_for_sku(WK_GPU)
        which = kernel or ("machete" if cc == "sm_90" else "marlin")
        if which not in ("machete", "marlin"):
            sys.exit(f"--kernel must be `machete` or `marlin`, not {which!r}")
        if which == "machete" and cc and cc != "sm_90":
            print(f"!! Machete is a Hopper kernel and {WK_GPU} is {cc}. Marlin is the honest bar "
                  f"here; reporting Machete off Hopper understates the peer.")
        if which == "marlin" and cc == "sm_90":
            print("!! Marlin is an Ampere kernel and is documented as weak on Hopper. Machete is "
                  "the honest H100 bar; reporting Marlin here would be a strawman.")
        default_model = ("meta-llama/Llama-3-8b" if which == "machete"
                         else "meta-llama/Llama-2-7b-hf/TP1")
        args = [py, f"{VLLM_SRC}/benchmarks/kernels/benchmark_{which}.py", "--dtype", dtype]
        if which == "machete":
            args += ["model_bench", "--models", models or default_model,
                     "--batch-sizes"] + batch_sizes.split() + ["--tp-sizes", "1"]
        else:
            args += ["--models", models or default_model, "--batch-sizes"] + batch_sizes.split()
        rc = _run(args, _torch_env(env), cwd=f"{VLLM_SRC}/benchmarks/kernels", check=False)
        build_vol.commit()
        if rc:
            sys.exit(rc)


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def framework(op: str = "gemm", shapes: str = "4096x4096x4096", dtype: str = "fp16",
              causal: bool = False, fa4: bool = False, iters: int = 50, runs: int = 5,
              tag: str = ""):
    """The `torch.compile` framework bar (`tools/cloud/peers/torch_compile_peer.py`).

    This is the peer the published "beats PyTorch at every S" has to be re-earned against: that
    number is **eager-only**, and its stated justification was that Triton does not install on
    Windows. On Linux it does, so the excuse is gone. The harness reports eager, `compile(default)`
    and `compile(max-autotune)` side by side and takes the **fastest** as the bar, so the run itself
    shows how much of the old margin was the peer being weak.

        WK_GPU=H100 modal run tools/cloud/modal_app.py::framework --op gemm \\
            --shapes 4096x4096x4096,8192x8192x8192
        WK_GPU=H100 modal run tools/cloud/modal_app.py::framework --op sdpa --causal --fa4 \\
            --shapes 1x16x2048x128
        WK_GPU=L4   modal run tools/cloud/modal_app.py::framework --op linear_gelu

    Keep every A/B inside ONE invocation: a Modal container can land on a different physical host
    between calls, so a Wukong number from one call and a peer number from another are not comparable
    (plan section 8 risk 7).
    """
    with _meter("framework"):
        env = dict(os.environ)
        _assert_no_stubs(env)
        os.makedirs(f"{PERSIST}/rounds", exist_ok=True)
        out = f"{PERSIST}/rounds/torch-{tag or op}-{dtype}.json"
        args = [f"{TORCH_VENV}/bin/python", _peer_script("torch_compile_peer.py"),
                "--op", op, "--shapes", shapes, "--dtype", dtype,
                "--iters", str(iters), "--runs", str(runs), "--json", out]
        if causal:
            args.append("--causal")
        if fa4:
            args.append("--fa4")
        rc = _run(args, _torch_env(env), check=False)
        build_vol.commit()
        print(f"\nJSON -> {out}")
        if rc:
            sys.exit(rc)


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def interactive():
    """Target for `modal shell tools/cloud/modal_app.py::interactive` — an interactive GPU shell.

    Inside: `cd /wukong`, `export CARGO_TARGET_DIR=/persist/target WUKONG_GPU_REQUIRED=1`.
    Note the container may land on a different physical host between invocations, so keep every
    A/B comparison inside ONE shell session (GPU_RETARGET_PLAN.md §8 risk 7). The meter is running
    the whole time this shell is open — plan §0: think off the GPU, act on it.
    """
    _prepare_cargo()
    print("Interactive placeholder — invoke with `modal shell` rather than `modal run`.")
