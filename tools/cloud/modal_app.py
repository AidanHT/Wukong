"""Modal app for running Wukong's GPU backend on datacenter GPUs.

Phase 0/1 of GPU_RETARGET_PLAN.md. This file does NOT change any compiler code — it only gets the
existing tree onto a rented GPU so the ~130 device-executing correctness gates can actually run.

Design decisions (each costs real money if got wrong):

* **Build on CPU, run on GPU.** Compiling 21 crates with `--features gpu` takes minutes; a Modal
  CPU core is ~$0.05/hr while an H100 is $3.95/hr. So `build` runs on CPU and writes into a
  persistent Volume; `test`/`bench` run on the GPU with a warm target dir and only link + execute.
* **Source is mounted at runtime, not baked into the image.** A source edit costs an upload of a few
  MB, never an image rebuild. Combined with the Volume-backed `CARGO_TARGET_DIR`, rebuilds are
  incremental across sessions.
* **`nvidia/cuda:*-cudnn-devel`** so the honest peers (NVRTC, cuBLAS, cuBLASLt, cuDNN) are all
  present. On the laptop these needed a hand-staged `tools/cuda-redist` on PATH; here they are the
  system libraries and `cudarc`'s `dynamic-loading` finds them by soname.
* **Every session starts with `device_info`.** GPU_RETARGET_PLAN.md §6.1 requires a provenance block
  (CC, SM count, SMEM/SM, L2, VRAM, driver, MIG state) verified against spec before any number is
  believed. It is also the cheapest possible smoke test that nvcc and the driver both work.

Usage (see tools/cloud/README.md for the full walkthrough):

    modal setup                                    # once, browser auth
    WK_GPU=L40S  modal run tools/cloud/modal_app.py::device_info
    WK_GPU=L40S  modal run tools/cloud/modal_app.py::build
    WK_GPU=L40S  modal run tools/cloud/modal_app.py::test
    WK_GPU=H100  modal shell tools/cloud/modal_app.py::interactive

`WK_GPU` is read at import time because Modal fixes a function's GPU at decoration time; there is no
runtime GPU switch. Modal's menu and what each costs (Aug 2026, $/hr derived from the per-second
rate) — note the column that matters here is the ISA, not the speed:

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
"""

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

WK_TIMEOUT = int(os.environ.get("WK_TIMEOUT", "3600"))
WK_CUDA_TAG = os.environ.get("WK_CUDA_TAG", "12.9.2-cudnn-devel-ubuntu22.04")
# 12.9 is the deliberate ceiling, not an oversight: cudarc 0.16.6 has no CUDA-13 bindings (its
# feature list and dlopen candidates stop at the .so.12 line), the nvidia-*-cu12 pip wheels end at
# 12.9.x, and torch 2.13.0+cu129 is the newest CUDA-12 build — so 12.9.2 aligns the image toolkit
# with every peer the harness compiles against it. Both this tag and the previous 12.8.1 were
# live-verified on Docker Hub 2026-08-06.

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
build_vol = modal.Volume.from_name("wukong-build", create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{WK_CUDA_TAG}", add_python="3.11")
    .apt_install("curl", "ca-certificates", "build-essential", "pkg-config", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable --component rustfmt clippy"
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
                    "/usr/sbin:/usr/bin:/sbin:/bin",
            # cudarc dlopens by soname; the cudnn-devel image puts them here.
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
            "CARGO_TERM_COLOR": "always",
            "RUST_BACKTRACE": "1",
        }
    )
    .add_local_dir(str(REPO_ROOT), REMOTE_SRC, ignore=IGNORE, copy=False)
)


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------

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
    env["CARGO_TARGET_DIR"] = f"{PERSIST}/target"
    # Keep the JIT'd SASS cache on the Volume too: cold JIT is ~0.76 ms/module but a corpus sweep
    # loads thousands, and the cache key already includes the driver version (cubin.rs:88).
    env["WUKONG_CUBIN_CACHE"] = f"{PERSIST}/cubin-cache"
    os.makedirs(env["WUKONG_CUBIN_CACHE"], exist_ok=True)
    return env


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

# What each SKU must report. A rented "A100" that shows 56 SMs is a MIG slice or a vGPU, which
# silently invalidates every occupancy conclusion — this table is what catches that.
#
# The `arch family` column is what actually matters to this project: each distinct sm_XX is a
# separate codegen target, NOT a speed grade.
#   sm_75  Turing      T4
#   sm_80  Ampere      A100                  } mma.sync + cp.async — what the backend emits today
#   sm_86  Ampere      A10                   }
#   sm_89  Ada         L4, L40S, RTX 4090    } SAME as the dev RTX 4050: runs unmodified
#   sm_90  Hopper      H100, H200            <- wgmma + TMA (H200 = H100 ISA, more HBM3e)
#   sm_100 Blackwell   B200                  <- tcgen05 / tensor memory: a THIRD codegen family
#   sm_103 Blackwell   B300 (Ultra)
#   sm_120 Blackwell   RTX PRO 6000, RTX 5090 <- consumer Blackwell, distinct ISA from sm_100
# A SKU absent from this table simply skips the gate (nothing is asserted about it).
_EXPECTED = {
    "T4":            ("sm_75", 40),
    "A10":           ("sm_86", 72),
    "L4":            ("sm_89", 58),
    "L40S":          ("sm_89", 142),
    "A100-40GB":     ("sm_80", 108),
    "A100-80GB":     ("sm_80", 108),
    "H100":          ("sm_90", 132),
    "H200":          ("sm_90", 132),   # same GH100 die as H100 — identical codegen target
    "B200":          ("sm_100", 148),
    "RTX-PRO-6000":  ("sm_120", 188),  # proxy for the RTX 5090 ISA that real users will have
}


# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

@app.function(image=image, gpu=WK_GPU, timeout=600, volumes={PERSIST: build_vol})
def device_info():
    """§6.1 provenance block. Run this FIRST in every session, before believing any number.

    Cost: seconds. Verifies the device is what was rented (not a MIG slice), that nvcc works, and
    that the driver accepts the PTX ISA levels the backend emits.
    """
    env = _prepare_cargo()
    print("=" * 78)
    print(f"WUKONG GPU PROVENANCE — requested SKU: {WK_GPU}")
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

    # Provenance gate: refuse to proceed quietly if the device is not what was rented.
    exp = _EXPECTED.get(WK_GPU)
    if exp:
        want_cc, want_sm = exp
        got_cc = got_sm = None
        for line in out.stdout.splitlines():
            if line.startswith("compute_capability"):
                got_cc = line.split(":", 1)[1].strip()
            elif line.startswith("sm_count"):
                got_sm = int(line.split(":", 1)[1].strip())
        print("--- provenance gate ---")
        print(f"expected: {want_cc}, {want_sm} SMs        got: {got_cc}, {got_sm} SMs")
        if got_cc != want_cc or got_sm != want_sm:
            print("\n!! MISMATCH. This is a MIG slice, a vGPU, or a different SKU than requested.")
            print("!! Occupancy/tile conclusions from this device are NOT transferable. "
                  "Publish nothing from it.")
        else:
            print("OK — full device, matches spec. Safe to tune occupancy against.")

    # Confirm the libraries cudarc will dlopen are actually resolvable by soname.
    print("\n--- peer libraries (cudarc dlopens these by soname) ---")
    for so in ("libcuda.so.1", "libnvrtc.so.12", "libcublas.so.12", "libcublasLt.so.12",
               "libcudnn.so.9"):
        rc = subprocess.run(["sh", "-c", f"ldconfig -p | grep -m1 {so} || true"],
                            capture_output=True, text=True)
        found = rc.stdout.strip() or "NOT FOUND"
        print(f"  {so:<22} {found}")


@app.function(image=image, cpu=8.0, timeout=WK_TIMEOUT, volumes={PERSIST: build_vol})
def build(release: bool = False, driver: bool = True):
    """Compile the GPU feature on CPU, into the persistent Volume. NO GPU is attached.

    This is the cost lever: the same compile on an H100 would burn GPU-hours at ~80x the price.
    `cargo test --no-run` builds the test binaries so the GPU function only links and executes.
    """
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


@app.function(image=image, gpu=WK_GPU, timeout=WK_TIMEOUT, volumes={PERSIST: build_vol})
def test(peers: bool = False, release: bool = False, driver: bool = True, filter: str = ""):
    """Run the ~130 device-executing correctness gates with skips escalated to failures.

    `WUKONG_GPU_REQUIRED=1` is the whole point: without it a device-less run prints `[skip]` and
    reports `ok`, which is how GPU code stays green while doing nothing (crate CLAUDE.md rule 3).
    `peers=True` additionally requires NVRTC/cuBLAS/cuBLASLt/cuDNN to load — leave it off on the
    first run so you learn what is missing instead of failing the whole suite.
    """
    env = _prepare_cargo()
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


@app.function(image=image, gpu=WK_GPU, timeout=WK_TIMEOUT, volumes={PERSIST: build_vol})
def bench(name: str = "", package: str = "wukong_codegen_gpu"):
    """Run the `#[ignore]`d perf sweeps/benches (release, one at a time).

    These are the sweeps that re-tune tiles per architecture (Phase 1 §3 / Phase 3 §2). They need
    `--release` and `--ignored`; pass `name` to select one, e.g. name="gemm_pipe_sweep".
    """
    env = _prepare_cargo()
    env["WUKONG_GPU_REQUIRED"] = "1"
    args = ["cargo", "test", "-p", package, "--features", "gpu", "--release", "--"]
    if name:
        args.append(name)
    args += ["--ignored", "--nocapture", "--test-threads=1"]
    _run(args, env, check=False)
    build_vol.commit()


@app.function(image=image, gpu=WK_GPU, timeout=WK_TIMEOUT, volumes={PERSIST: build_vol})
def interactive():
    """Target for `modal shell tools/cloud/modal_app.py::interactive` — an interactive GPU shell.

    Inside: `cd /wukong`, `export CARGO_TARGET_DIR=/persist/target WUKONG_GPU_REQUIRED=1`.
    Note the container may land on a different physical host between invocations, so keep every
    A/B comparison inside ONE shell session (GPU_RETARGET_PLAN.md §8 risk 7).
    """
    _prepare_cargo()
    print("Interactive placeholder — invoke with `modal shell` rather than `modal run`.")
