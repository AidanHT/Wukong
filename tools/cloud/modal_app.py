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
    WK_GPU=L4    modal run tools/cloud/modal_app.py::test
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

image = (
    modal.Image.from_registry(f"nvidia/cuda:{WK_CUDA_TAG}", add_python="3.11")
    .apt_install("curl", "ca-certificates", "build-essential", "pkg-config", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable --component rustfmt clippy",
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
            # Baked so in-container code knows which SKU the client asked for: Modal does not forward
            # local env vars into containers.
            "WK_GPU": WK_GPU,
            "CARGO_TERM_COLOR": "always",
            "RUST_BACKTRACE": "1",
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
        _run(["cargo", "test", "--workspace"] + profile, env, check=False)

        build_vol.commit()
        print("\nBuild artifacts committed to the 'wukong-build' Volume.")


@app.function(image=image, gpu=WK_GPU, cpu=WK_CPU, memory=WK_MEM_MIB, timeout=WK_TIMEOUT,
              volumes={PERSIST: build_vol})
def test(peers: bool = False, release: bool = False, driver: bool = True, filter: str = ""):
    """Run the ~130 device-executing correctness gates with skips escalated to failures.

    `WUKONG_GPU_REQUIRED=1` is the whole point: without it a device-less run prints `[skip]` and
    reports `ok`, which is how GPU code stays green while doing nothing (crate CLAUDE.md rule 3).
    `peers=True` additionally requires NVRTC/cuBLAS/cuBLASLt/cuDNN to load — leave it off on the
    first run so you learn what is missing instead of failing the whole suite.

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
def bench(name: str = "", package: str = "wukong_codegen_gpu"):
    """Run the `#[ignore]`d perf sweeps/benches (release, one at a time).

    These are the sweeps that re-tune tiles per architecture (Phase 1 §3 / Phase 3 §2). They need
    `--release` and `--ignored`; pass `name` to select one, e.g. name="gemm_pipe_sweep".
    **Requires `::build --release` first** — see `_require_prebuilt`.
    """
    with _meter("bench"):
        env = _prepare_cargo()
        _require_prebuilt(package, release=True)
        env["WUKONG_GPU_REQUIRED"] = "1"
        args = ["cargo", "test", "-p", package, "--features", "gpu", "--release", "--"]
        if name:
            args.append(name)
        args += ["--ignored", "--nocapture", "--test-threads=1"]
        _run(args, env, check=False)
        build_vol.commit()


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
