> **COORDINATOR CORRECTION (2026-08-06, before commit):** this dossier claims NVIDIA "no longer
> publishes" the `12.8.1-cudnn-devel-ubuntu22.04` Docker tag. That is FALSE — a direct Docker
> Hub API check the same day returned a valid tag object (last updated 2025-03-13). The image
> bump to `12.9.2` landed anyway, for the honest reason: alignment with the verified pin set
> (cudarc 0.16.6 has no CUDA-13 bindings; the cu12 pip line ends at 12.9.x; torch 2.13.0+cu129
> is the newest CUDA-12 build). Treat the rest of the dossier's live-verified pins as sound;
> treat availability claims as verify-before-relying.

# D5 — Honest-peer build dossier for the cloud Linux instrument

Agent D5, Wave 0, Wukong GPU retarget campaign. **Research only — no repo files modified.**
All version facts below were live-verified **2026-08-06** against PyPI JSON, the GitHub releases API,
Docker Hub's registry API, `download.pytorch.org`, `raw.githubusercontent.com` and NVIDIA docs.
Everything marked *(est.)* is an engineering estimate, not a measurement.

Mission context: `GPU_RETARGET_PLAN.md` §0 ("peers must be strong and fairly configured"), §4.2
(hard requirements checklist), §6 item 4 ("peers get stronger and that is the point").
The rule this dossier exists to serve: **every install/build must be scripted and correct BEFORE
renting**, and no metered GPU-second may be spent compiling a peer.

---

## 0. What the in-tree harness actually demands (read this first)

Ground truth from `crates/wukong_codegen_gpu/src/baselines.rs` (2614 lines) and
`crates/wukong_codegen_gpu/Cargo.toml:16-19`.

**The Rust side never links a CUDA library.** `cudarc` is built with `default-features = false` +
`dynamic-loading`, so every peer library is `dlopen`ed lazily on first call, and a missing library
`panic!`s inside cudarc's loader — which `peers_available()` / `cudnn_available()` /
`fa2_peer_available()` catch with `catch_unwind` and turn into a *skip*
(`baselines.rs:80-108`, `:1986`, `:2305`). A skip becomes a hard failure when `WUKONG_PEER_REQUIRED=1`
is set (`gpu.rs:4885-4893`). **Cloud rounds must always set `WUKONG_PEER_REQUIRED=1`** or a
mis-staged library silently publishes nothing while reporting green.

Enabled cudarc features and the base library name each one dlopens (verified against
`cudarc v0.16.6` sources — `src/{cublas,cublaslt,cudnn,nvrtc,driver}/sys/mod.rs`):

| cudarc feature | base name passed to the loader | used by |
|---|---|---|
| `driver` | `"cuda"`, then `"nvcuda"` | everything |
| `nvrtc` | `"nvrtc"` | Tier-A naive CUDA-C peers (`nvrtc_naive_gemm_nt`, `_attn`, `_conv`, `_int8`, `_w4a16`) |
| `cublas` | `"cublas"` | `cublas_gemm_nt_f16`, `_f16_f32out`, `_int8` (IMMA), `cublas_attn_chain` |
| `cublaslt` | `"cublasLt"` | fp8 E4M3 `cublasLtMatmul` (raw-`sys`, the only fp8 surface) |
| `cudnn` | `"cudnn"` | `cudnn_conv2d_run` / `time_cudnn_conv2d` (v7 heuristic picks the engine) |

**The exact candidate filenames cudarc tries, in order** (`cudarc/src/lib.rs:204-243`,
`get_lib_name_candidates`; on Linux `DLL_PREFIX="lib"`, `DLL_SUFFIX=".so"`, `major`/`minor` come from
the selected `cuda-XXXXX` feature — for this repo's `cuda-12060` that is `12`/`06`):

```
libX.so           libX64.so         libX64_12.so      libX64_1206.so   libX64_1206_0.so
libX640_06.so     libX64_10.so      libX64_11.so      libX64_12.so     libX640_0.so
libX64_9.so       libX.so.12        libX.so.12        libX.so.11       libX.so.10
libX.so.9         libX.so.1
```

Two consequences that decide §1 below:

1. **`libcublas.so` (unversioned) is tried FIRST.** A `-devel` image ships that dev symlink, so the
   system library always wins over anything else on the path. A `-runtime` image does **not** ship it,
   and then only the `libcublas.so.12` candidate can match — which is why a **CUDA-13 runtime image
   would make the harness unable to find cuBLAS at all** (soname `libcublas.so.13` is not in the list).
2. **pip redist wheels ship only versioned sonames.** Verified by reading the wheel central
   directories today:
   - `nvidia_cuda_nvrtc_cu12-12.9.86`: `nvidia/cuda_nvrtc/lib/{libnvrtc.so.12, libnvrtc.alt.so.12,
     libnvrtc-builtins.so.12.9, libnvrtc-builtins.alt.so.12.9}`
   - `nvidia_cudnn_cu12-9.24.0.43`: `nvidia/cudnn/lib/{libcudnn.so.9, libcudnn_graph.so.9,
     libcudnn_ops.so.9, libcudnn_cnn.so.9, libcudnn_adv.so.9, libcudnn_engines_precompiled.so.9,
     libcudnn_engines_runtime_compiled.so.9, libcudnn_engines_tensor_ir.so.9, libcudnn_heuristic.so.9,
     libcudnn_ext.so.9}`
   No unversioned `.so` symlink anywhere — the `.so.12` / `.so.9` candidates are what make the wheel
   route work.

**The out-of-process FA2 peer** (`baselines.rs:2284-2299`): `fa2_peer_paths()` resolves
`WUKONG_FA2_PYTHON` (default `<repo>/tools/torch-cuda-venv/Scripts/python.exe` — **a Windows path;
on Linux this env var is MANDATORY**) and `WUKONG_FA2_PEER` (default `<repo>/tools/fa2_sdpa_peer.py`).
The script needs `numpy` + `torch` and drives `torch.nn.attention.{SDPBackend, sdpa_kernel}` over
FLASH_ATTENTION / EFFICIENT_ATTENTION / CUDNN_ATTENTION / MATH, exchanging raw f16/f32 files through
`std::env::temp_dir()`. It reports `{backend}_sec`, `cudnn_sec`, `{backend}_sdpa_sec`.
Other env vars in play: `WUKONG_PEER_REQUIRED`, `WUKONG_FLASH_WS`, `WUKONG_CUBIN_CACHE`.

---

## 1. cuBLAS / cuBLASLt / cuDNN / NVRTC for the in-tree harness

### 1.1 Route A — the devel image (RECOMMENDED, zero extra install)

`nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04` already contains every library the harness dlopens,
**with the unversioned dev symlinks**, so nothing needs to be installed and nothing needs staging.

Prerequisites: none beyond the image. Install wall time: **0 s** (baked into the image pull).
Disk: image is **5.93 GB compressed** on Docker Hub (verified via `hub.docker.com/v2` API,
`full_size`); *(est.)* ~12 GB unpacked.

Paths and the `LD_LIBRARY_PATH` that must be set (this is the Linux equivalent of the repo's
"nested redist bin dirs" Windows landmine — see `gpu-peer-dll-path.md`):

```sh
# cuBLAS / cuBLASLt / NVRTC / nvJitLink (toolkit, via the /usr/local/cuda symlink)
/usr/local/cuda/lib64            # libcublas.so -> .so.12, libcublasLt.so, libnvrtc.so, libnvJitLink.so
# cuDNN 9 (deb, installed by the -cudnn- image variant)
/usr/lib/x86_64-linux-gnu        # libcudnn.so -> .so.9 + the 8 cuDNN-9 sublibraries
# the driver, injected by the container runtime
/usr/lib/x86_64-linux-gnu/libcuda.so.1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
```

That is **exactly what `tools/cloud/modal_app.py:98-107` already sets** — the existing Modal image is
correct on this point; only the tag needs bumping (§6).

> **LANDMINE (Linux analogue of the nested-bin-dirs bug), the single worst failure mode here:**
> **NEVER put `/usr/local/cuda/lib64/stubs` on `LD_LIBRARY_PATH`.** It contains a *stub*
> `libcuda.so` that exists only to satisfy the linker. cudarc's first driver candidate is literally
> `libcuda.so`, so the stub wins over the real `libcuda.so.1` and every driver call fails or the
> process aborts — with a symptom ("no CUDA device") that looks exactly like a hardware problem.
> Some build recipes (CUTLASS, FA2/FA3, vLLM) tell you to add the stubs dir. Add it **only** for the
> duration of that `cmake`/`pip` command, never in the image `ENV`.

Verification one-liner (run before anything else; ~1 s, no GPU work):

```sh
for l in cublas cublasLt cudnn nvrtc cuda; do
  printf '%-10s ' "$l"; ldconfig -p | grep -m1 "lib$l\.so" || echo MISSING
done
ls -l /usr/local/cuda/lib64/libcublas.so /usr/lib/x86_64-linux-gnu/libcudnn.so
python3 -c "import ctypes;[ctypes.CDLL(n) for n in ('libcublas.so','libcublasLt.so','libcudnn.so','libnvrtc.so','libcuda.so.1')];print('all dlopen OK')"
```

### 1.2 Route B — pip redist wheels (fallback: bare VM without a devel toolkit)

Needed on a Lambda/Verda/Hyperstack VM that has a driver but not the full toolkit. **The cu12 wheel
line terminates at 12.9** — verified today: `nvidia-cublas-cu12` max is `12.9.2.10`,
`nvidia-cuda-nvrtc-cu12` max is `12.9.86`, `nvidia-nvjitlink-cu12` max is `12.9.86`. CUDA 13 moved to
**unsuffixed** names (`nvidia-cublas` = 13.6.1.10, `nvidia-cuda-nvrtc` = 13.3.33), which cudarc 0.16.6
cannot use (§6). So Route B pins the *same* versions the repo already documents in
`peer_env_hint()` — that hint is still correct in 2026-08.

```sh
mkdir -p /opt/cuda-redist
# ONE pip command — pip's --target clobbers the shared nvidia/ namespace if run separately.
pip install --target /opt/cuda-redist --no-deps \
    nvidia-cuda-nvrtc-cu12==12.9.86 \
    nvidia-cublas-cu12==12.9.2.10 \
    nvidia-nvjitlink-cu12==12.9.86 \
    nvidia-cudnn-cu12==9.24.0.43

export LD_LIBRARY_PATH=\
/opt/cuda-redist/nvidia/cuda_nvrtc/lib:\
/opt/cuda-redist/nvidia/cublas/lib:\
/opt/cuda-redist/nvidia/nvjitlink/lib:\
/opt/cuda-redist/nvidia/cudnn/lib:\
/usr/lib/x86_64-linux-gnu
```

- Install wall time: *(est.)* 60–120 s on a fast link (cuBLAS alone is ~600 MB).
- Disk: *(est.)* ~2.6 GB (cublas ~0.6, cudnn ~1.5, nvrtc ~0.09, nvjitlink ~0.03, plus cublasLt inside
  the cublas wheel).
- **Every one of those four `lib` directories must be listed separately** — this is the exact Linux
  restatement of the memory note "cuda-redist nested bin dirs must each be on PATH or peer sweeps
  silently skip". `nvidia/cudnn/lib` in particular is easy to forget because the cuDNN peer is the only
  one that uses it and it skips politely.
- `nvidia-cublas-cu12` ships **both** `libcublas.so.12` and `libcublasLt.so.12` — no separate wheel.
- NVRTC 12.9 hard-imports nvJitLink; omitting the third wheel gives an `undefined symbol: __nvJitLink*`
  at first `compile_ptx`.
- **Do not point `LD_LIBRARY_PATH` at a PyTorch venv's `site-packages/nvidia/*/lib`.** A torch cu130
  install puts `libcublas.so.13` there; a torch cu129 install puts a *different* 12.9 build there.
  Either can shadow the system library inside the Rust process. Keep the peer redist and the torch venv
  in separate trees and never merge their env (the FA2 peer is a subprocess, so it can and should carry
  its own env).

### 1.3 60-second GPU smoke tests

```sh
export WUKONG_PEER_REQUIRED=1
# ~10 s: proves NVRTC + cuBLAS load and produce a correct GEMM against the f64 oracle
cargo test -p wukong_codegen_gpu --features gpu --release reproducibility_vs_cublas -- --ignored --nocapture
# ~20 s: proves cuDNN loads and the v7 heuristic picks an engine (prints the algo name)
cargo test -p wukong_codegen_gpu --features gpu --release conv_vs_cudnn -- --ignored --nocapture
```
A `[skip]` line from either is now a **failure** (`WUKONG_PEER_REQUIRED=1`), which is the point.
Build these binaries on a CPU-only container first (`modal run tools/cloud/modal_app.py::build`);
these commands should only link + run on the metered box.

---

## 2. PyTorch with `torch.compile` (Inductor + Triton) on H100 / L4 / L40S

### 2.1 The pin

| item | pin | verified |
|---|---|---|
| torch | **2.13.0** (released 2026-07-08) | PyPI release history |
| CUDA variant | **`+cu129`** from `download.pytorch.org/whl/cu129` | index lists `torch-2.13.0+cu129-cp{310,311,312,313,314,315}-…manylinux_2_28_x86_64.whl` |
| triton | **3.7.1** (2026-06-17) — a hard `install_requires` of torch on Linux | torch 2.13.0 PyPI metadata |
| python | 3.11 or 3.12 (torch requires ≥3.10) | |

**Why cu129 and not the PyPI default.** For releases ≥2.11 the *default PyPI* torch wheel is **cu130**:
torch 2.13.0's metadata pulls `cuda-toolkit[cublas,…]==13.0.3`, `nvidia-cudnn-cu13==9.20.0.48`,
`nvidia-nccl-cu13==2.29.7`. That works on a 580.95 driver, but it drags a second, CUDA-13 copy of
cuBLAS/cuDNN into the box next to the CUDA-12.9 system libraries the Rust harness uses. Pinning
`+cu129` keeps **one CUDA-12.9 world**. cu128 and cu126 are dead ends for this pin — verified: those
indexes top out at torch 2.11.0 / 2.12.0, no 2.13.

```sh
python3 -m venv /opt/torch-venv
/opt/torch-venv/bin/pip install -U pip
/opt/torch-venv/bin/pip install --index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.13.0+cu129"
/opt/torch-venv/bin/pip install numpy            # required by tools/fa2_sdpa_peer.py
```

Install wall time: *(est.)* 3–6 min (≈4 GB of wheels). Disk: *(est.)* ~7 GB for the venv.
**Do this on a CPU-only container and store it in the Modal Volume** — it is pure download + unzip.

Wire it to the in-tree harness (mandatory on Linux — the default path is a Windows `.exe`):

```sh
export WUKONG_FA2_PYTHON=/opt/torch-venv/bin/python
export WUKONG_FA2_PEER=/wukong/tools/fa2_sdpa_peer.py
```

### 2.2 Making Inductor a *fair* peer (the §0 rule)

```python
import torch
torch.backends.cuda.matmul.fp32_precision = "tf32"   # NEW API; allow_tf32 is deprecated after 2.9
torch.backends.cudnn.conv.fp32_precision  = "tf32"
# Do NOT mix old and new: setting allow_tf32 anywhere alongside these is unsupported and warns.
```

Fairness checklist for any published torch.compile column:

- **dtype parity.** Wukong's GEMM peers are f16-in/f32-accumulate. Feed torch `torch.float16`
  (or `bfloat16`) tensors, not fp32-with-tf32, or the comparison is dtype-rigged in *our* favour.
  If the Wukong column is fp32, then tf32 must be **on** for torch or the peer is strawmanned.
- **`mode="max-autotune"`**, not the default. Default `torch.compile` picks ATen (= cuBLAS) for GEMM
  and only fuses the pointwise tail; max-autotune benchmarks Triton templates against it.
  `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` (default `ATEN,TRITON,CPP`) can be set to
  `ATEN,TRITON,CUTLASS` — the CUTLASS backend additionally needs `TORCHINDUCTOR_CUTLASS_DIR` pointing
  at a CUTLASS checkout (§4) and adds a lot of compile time. Recommendation: publish `ATEN,TRITON`
  as the standard bar, and treat `+CUTLASS` as a stretch column only if §4 is built anyway.
- **Compilation is not measured.** Warm up ≥3 calls after `torch.compile(...)` returns, then time.
  Use `torch.cuda.Event` timing, back-to-back, no `sleep`, trimmed median over ≥100 iters
  (plan §6.3). `mode="max-autotune"` can take minutes on the first call — **that time is metered**,
  so persist the caches (below).
- **Persist the compile caches to the Volume** so the second and later rounds don't re-pay:
  ```sh
  export TORCHINDUCTOR_CACHE_DIR=/persist/inductor-cache   # holds fxgraph/ and aotautograd/ too
  export TRITON_CACHE_DIR=/persist/triton-cache
  ```
- **Shape parity.** Compile with the same static shapes the Wukong column runs
  (`dynamic=False`) — dynamic-shape Inductor emits guarded, slower kernels and would flatter us.
- Record `torch.__version__`, `torch.version.cuda`, `triton.__version__`, the chosen backend, and the
  autotune log in the round's provenance block.

### 2.3 Smoke script — proves Inductor really compiled a matmul+softmax graph (≤30 s GPU)

Write to `/opt/smoke_inductor.py`:

```python
import os, torch, torch._inductor.config as icfg
torch.backends.cuda.matmul.fp32_precision = "tf32"
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print(f"device={p.name} cc={p.major}.{p.minor} sms={p.multi_processor_count} "
      f"torch={torch.__version__} cuda={torch.version.cuda}")
import triton; print("triton", triton.__version__)

def f(a, b):
    return torch.softmax(a @ b, dim=-1)

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
ref = f(a, b)

seen = []
from torch._inductor.codecache import PyCodeCache
g = torch.compile(f, mode="max-autotune", dynamic=False, fullgraph=True)
out = g(a, b)                      # triggers compile
torch.cuda.synchronize()
# PROOF Inductor ran: at least one generated python module is in the code cache, and it
# contains a Triton kernel (not just an ATen fallback).
srcs = [m.__dict__.get("__file__", "") for m in PyCodeCache.modules]
assert srcs, "Inductor generated no code -> it fell back entirely"
body = "".join(open(s).read() for s in srcs if s and s.endswith(".py"))
assert "@triton.jit" in body or "triton_heuristics" in body, "no Triton kernel generated"
print("inductor modules:", len(srcs), "| triton kernels present: True")
print("max abs err vs eager:", (out.float() - ref.float()).abs().max().item())

ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
for _ in range(5): g(a, b)
torch.cuda.synchronize(); ev0.record()
for _ in range(50): g(a, b)
ev1.record(); torch.cuda.synchronize()
print("compiled ms/iter:", ev0.elapsed_time(ev1) / 50)
```

```sh
TORCHINDUCTOR_CACHE_DIR=/persist/inductor-cache TRITON_CACHE_DIR=/persist/triton-cache \
  /opt/torch-venv/bin/python /opt/smoke_inductor.py
```
First run pays max-autotune compile (*(est.)* 60–180 s — run it once on the **cheapest** GPU SKU you
will use, e.g. L4 at $0.80/hr, then reuse the cache); subsequent runs are cache hits.

**Pitfall:** `mode="max-autotune"` on a *container* is where the plan's "no clock lock" bites hardest —
autotuning picks a winner using timings taken during clock ramp. For a canonical published round,
autotune once on the VM with `nvidia-smi -lgc` locked, keep the cache, then measure.

---

## 3. FlashAttention

Three generations, three completely different build stories. **Which one is the honest peer depends
on the rung of the ladder**, and this is the item that most changes the plan's assumptions.

| target | peer | how | build cost |
|---|---|---|---|
| **H100 (sm_90)** | **FlashAttention-4** `flash-attn-4` | **pure-python wheel, JIT via CuTeDSL — no CUDA compiler, no build** | **~30 s install** |
| H100 (sm_90), fp8/backward | FlashAttention-3 (`hopper/`) | source build, nvcc | *(est.)* 15–90 min |
| **A100 (sm_80)**, L40S/L4 (sm_89) | FlashAttention-2 `flash-attn` | source build (no wheel for torch 2.13) **or** torch SDPA's built-in FLASH backend | *(est.)* 20–50 min, or free |

### 3.1 FlashAttention-4 — the H100 answer, and it is nearly free

Verified today: `flash-attn-4` **4.0.0b25**, uploaded **2026-08-05**, wheel is **`py3-none-any`**
(pure Python). Upstream tag `fa4-v4.0.0.beta25`. Dependencies include `nvidia-cutlass-dsl>=4.5.2`
(PyPI `nvidia-cutlass-dsl` is at 4.7.0, 2026-08-05), `torch`, `einops`, `apache-tvm-ffi`,
`quack-kernels>=0.5.0`. Targets **SM90 (Hopper) and SM100/SM110 (Blackwell) only — not sm_80, not sm_89.**

```sh
# NOTE: every published flash-attn-4 release is a pre-release (4.0.0bNN) -> --pre is REQUIRED,
# or pin the exact version. Without it pip resolves nothing and you will think it is unavailable.
/opt/torch-venv/bin/pip install --pre "flash-attn-4==4.0.0b25"
#   CUDA 13 hosts only: /opt/torch-venv/bin/pip install --pre "flash-attn-4[cu13]==4.0.0b25"
```

Install wall time: *(est.)* 30–90 s. Disk: *(est.)* ~1.5 GB (dominated by `nvidia-cutlass-dsl`).
Direct invocation (not through SDPA):

```python
from flash_attn.cute import flash_attn_func      # q,k,v: [B, S, H, D] fp16/bf16 on cuda
out = flash_attn_func(q, k, v, causal=True)
```

**Pitfall:** it JITs at first call, so the first invocation is slow (*(est.)* 10–60 s per new shape
/ head-dim / causal combination). Warm every shape before timing, and if CuTeDSL exposes a disk cache,
put it on the Volume. Also: `flash_attn.cute`'s layout is `[B, S, H, D]`, while the repo's
`fa2_sdpa_peer.py` and Wukong's flash use **head-major `[B, H, S, D]`** — a `.transpose(1, 2)` is
needed and must be done **outside** the timed region (or, better, materialised contiguous once) or the
peer is handicapped by a permute.

### 3.2 FlashAttention-3 (Hopper) — source build, only if fp8 or backward is needed

Repo `Dao-AILab/flash-attention`, `hopper/` subtree. Requirements per its README (verified):
**H100/H800, CUDA ≥ 12.3, CUDA 12.8 recommended.** No prebuilt wheels published on GitHub releases
for the hopper tree.

```sh
git clone --depth 1 https://github.com/Dao-AILab/flash-attention /opt/fa
cd /opt/fa/hopper
# Cut the build from ~1-2h to a fraction by disabling everything the peer does not need.
export MAX_JOBS=$(nproc) NVCC_THREADS=4
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
       FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE  FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
       FLASH_ATTENTION_DISABLE_LOCAL=TRUE    FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
       FLASH_ATTENTION_DISABLE_PACKGQA=TRUE  FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
       FLASH_ATTENTION_DISABLE_FP8=TRUE      FLASH_ATTENTION_DISABLE_SM80=TRUE \
       FLASH_ATTENTION_DISABLE_HDIM96=TRUE   FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
       FLASH_ATTENTION_DISABLE_HDIM256=TRUE
/opt/torch-venv/bin/python setup.py bdist_wheel      # -> dist/flash_attn_3-*.whl ; STASH IT
/opt/torch-venv/bin/pip install dist/flash_attn_3-*.whl
```
```python
from flash_attn_3 import flash_attn_interface
out, *_ = flash_attn_interface.flash_attn_func(q, k, v, causal=True)   # [B, S, H, D]
```

- The `DISABLE_*` list is verified from `hopper/setup.py:51-72` — it is the whole build-time lever set
  (head-dims, fp8, varlen, cluster, sm80, backward, split, pagedkv, appendkv, local, softcap, packgqa).
  Keeping only fp16 forward at hdim 64+128 is *(est.)* a 4–8× reduction in translation units.
- **Build on CPU, not on the GPU box** — nvcc never touches the device. A Modal CPU container is
  ~$0.05/hr against $3.95/hr for an H100. Cache the built `.whl` on the Volume.
- Needs *(est.)* ≥32 GB RAM at `MAX_JOBS≈16`; RAM exhaustion is the classic FA build failure.
- Only build this if the round needs **fp8 attention or the backward pass**; otherwise FA4 (§3.1) is
  strictly cheaper and is the newer kernel.

### 3.3 FlashAttention-2 — the A100 / Ada answer

Verified: `flash-attn` **2.8.3.post1** (2026-06-11). GitHub release assets (50 wheels) cover
**cu12 × torch{2.4…2.8}** and **cu13 × torch2.9** only. **There is no prebuilt wheel for torch 2.12
or 2.13** — `pip install flash-attn --no-build-isolation` against torch 2.13 will fall through to a
**source compile**.

Two honest options:

**(a) Free, already wired: torch SDPA's FLASH_ATTENTION backend.** This *is* FlashAttention-2 compiled
into the torch wheel, forced explicitly by name. `tools/fa2_sdpa_peer.py` already does exactly this and
cross-checks the output against the same f64 reference. On A100/Ada this satisfies §0's "a real
FlashAttention build, not an unfused cuBLAS chain" — it is a genuinely fused FA2 kernel; it is *not*
"eager PyTorch". Cost: $0. Recommended default.

**(b) Standalone `flash_attn` for a direct, SDPA-dispatch-free call:**

```sh
export MAX_JOBS=$(nproc) NVCC_THREADS=4
export FLASH_ATTENTION_FORCE_BUILD=TRUE      # skip the wheel-download attempt outright
export FLASH_ATTN_CUDA_ARCHS="80"            # default is "80;90;100;120" -> ~4x the work
/opt/torch-venv/bin/pip wheel --no-build-isolation --no-deps flash-attn==2.8.3.post1 -w /persist/wheels
/opt/torch-venv/bin/pip install /persist/wheels/flash_attn-*.whl
```
```python
from flash_attn import flash_attn_func        # q,k,v: [B, S, H, D] fp16/bf16
out = flash_attn_func(q, k, v, causal=True)
```
- `FLASH_ATTN_CUDA_ARCHS` is verified from `setup.py:70`; `FLASH_ATTENTION_FORCE_BUILD` from `:61`.
- Upstream claims "3–5 min on a 64-core machine with ninja"; **on a 16-core container budget
  *(est.)* 20–50 min** for a single arch, more for the default four. Needs `ninja` + `packaging`
  (`pip install ninja packaging` first) — **without ninja the build is single-threaded and takes ~2 h**
  (upstream's own number).
- Build on CPU-only, stash the wheel on the Volume, install it on the metered box.
- **The alternative that avoids the compile entirely:** pin the torch venv used *only* for the FA2 peer
  to `torch==2.8.*+cu12x` and install the matching prebuilt
  `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` from the GitHub
  release. This is legitimate — the FA2 peer is an out-of-process subprocess, so it may live in its own
  venv with its own torch, entirely separate from the torch.compile venv (§2). If you take this route,
  match `cxx11abi` to the torch build (`torch._C._GLIBCXX_USE_CXX11_ABI` → TRUE/FALSE).

### 3.4 60-second smoke

```sh
/opt/torch-venv/bin/python - <<'PY'
import torch
from flash_attn.cute import flash_attn_func          # FA4 (H100); or: from flash_attn import ...
q,k,v = (torch.randn(1,1024,8,128,device='cuda',dtype=torch.float16) for _ in range(3))
o = flash_attn_func(q,k,v,causal=True); torch.cuda.synchronize()
ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True).transpose(1,2)
print("max err", (o.float()-ref.float()).abs().max().item())   # expect < 1e-2 in fp16
PY
```

---

## 4. CUTLASS profiler for sm90a (and sm80)

Verified: **CUTLASS 4.6.1**, released **2026-07-15** (GitHub releases API). Prereqs from the
Quickstart: CUDA ≥ 11.4 (12.0+ recommended), **CMake ≥ 3.18**, C++17 host compiler (g++ ≥ 7.5),
Python ≥ 3.6.

> **The release tarballs are NOT the profiler.** I downloaded the first 2 MB of
> `cutlass-install-x86_64-cu12-4.6.1.tar.gz` and listed it: it contains
> `x86_64/cu12/include/CuteDSLRuntime.h`, `include/cutlass_compiler/CompilerCAPI.h`,
> `lib/libCutlassCompiler.so` — i.e. the **CuTe DSL runtime**, not `cutlass_profiler`.
> The C++ profiler must be built from source. (`pip install nvidia-cutlass` / `nvidia-cutlass-dsl`
> likewise gives you the Python DSL, not the profiler binary.)

```sh
git clone --depth 1 --branch v4.6.1 https://github.com/NVIDIA/cutlass /opt/cutlass
cd /opt/cutlass && mkdir build && cd build
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake .. \
  -DCUTLASS_NVCC_ARCHS=90a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f16*,cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f32*" \
  -DCMAKE_BUILD_TYPE=Release
make cutlass_profiler -j"$(nproc)"
```
For A100 swap `-DCUTLASS_NVCC_ARCHS=80` and
`-DCUTLASS_LIBRARY_KERNELS="cutlass_tensorop_h*gemm*,cutlass_tensorop_s*gemm_f16*"`.
`90a` (not `90`) is mandatory for Hopper — the `a` suffix is what enables `wgmma`/TMA; a plain `90`
build silently omits the fastest kernels and would understate the peer.

**Build time / disk:**
- Filtered as above, *(est.)* **20–45 min on 16 cores**, *(est.)* 8–15 GB of build dir.
- **Unfiltered is a trap.** NVIDIA's own docs: on SM90 the full instantiation set "will be in the order
  of millions of kernels", and `CUTLASS_LIBRARY_KERNELS` "must be non-empty, since generating and
  filtering these kernels alone can take hours." A default `cmake .. && make cutlass_profiler` here is
  a multi-hour, >100 GB mistake.
- **Build on a CPU-only container** and put the resulting `build/tools/profiler/cutlass_profiler`
  binary (plus nothing else) on the Volume. It is a single self-contained executable.

Running it (`--providers` limits who competes; `cublas` gives you a same-binary cuBLAS control column):

```sh
/persist/bin/cutlass_profiler \
  --operation=Gemm --op_class=tensorop \
  --m=4096 --n=4096 --k=4096 \
  --A=f16:row --B=f16:column --C=f16:column --accumulator-type=f32 \
  --providers=cutlass,cublas \
  --warmup-iterations=20 --profiling-iterations=100 \
  --output=/persist/rounds/cutlass_4096.csv --verification-enabled=true
```
`--kernels="*f16*"` narrows further; `--m/--n/--k` accept comma lists and `:step` ranges for a sweep.
GPU cost per shape: *(est.)* 5–20 s once the kernel set is small. Note the `--A=f16:row --B=f16:column`
spelling — that is the `A·Bᵀ` (`nn.Linear`) contract the Wukong GEMM peers use
(`baselines.rs:528-545`), so the layouts match without a transpose fudge.

**Fairness note:** a CUTLASS *profiler* number is a best-of-many-kernels number chosen by exhaustive
search. Reporting Wukong against it is honest only if it is labelled as such — it is a *stronger*
bar than cuBLAS at some shapes and weaker at others, and CUTLASS's own docs (4.6.0) ship a
"how to accurately profile GEMM performance" note that should be followed before publishing.

---

## 5. Marlin / Machete-class W4A16 int4

Context: `baselines.rs:1164-1230` currently measures Wukong's W4A16 against the **Tier-A naive
CUDA-C** kernel only, and the docs say "no library peer exists". §0 retires that framing.

### 5.1 The practical route: vLLM's compiled kernels + its own kernel benchmarks

Verified: **vLLM 0.26.0** (2026-07-25), wheel `vllm-0.26.0-cp38-abi3-manylinux_2_28_x86_64.whl`,
**303.7 MB**. `benchmarks/kernels/benchmark_marlin.py` and `benchmark_machete.py` exist at that tag
and import only `vllm._custom_ops` + `vllm.model_executor.layers.quantization.utils.*` — i.e. **the
prebuilt wheel is sufficient; no vLLM source build is required.**

> **Hard constraint: `vllm==0.26.0` pins `torch==2.11.0`** (verified in its metadata, along with
> `flashinfer-python==0.6.14` and `nvidia-cutlass-dsl[cu13]==4.6.0`). It is therefore **incompatible
> with the torch-2.13 venv of §2 and must live in its own venv.** Its torch 2.11.0 comes from PyPI =
> **cu130**, which needs driver ≥ 580.65.06 — fine on the observed Modal 580.95 hosts, but it would
> fail on an older-driver marketplace box. Check `nvidia-smi` before assuming.

```sh
python3 -m venv /opt/vllm-venv
/opt/vllm-venv/bin/pip install -U pip
/opt/vllm-venv/bin/pip install "vllm==0.26.0"          # pulls torch 2.11.0 (cu130) + deps
git clone --depth 1 --branch v0.26.0 https://github.com/vllm-project/vllm /opt/vllm-src
```
Install wall time: *(est.)* 6–12 min. Disk: *(est.)* ~9 GB.

Run the kernels standalone (CLI verified from the scripts' own `--help` epilog):

```sh
cd /opt/vllm-src/benchmarks/kernels
# Machete (Hopper-optimised mixed-input GEMM) — square sweep
/opt/vllm-venv/bin/python benchmark_machete.py --dtype float16 \
    square_bench --dim-start 4096 --dim-end 4096 --dim-increment 1
# Machete — real model shapes (this is the one to publish)
/opt/vllm-venv/bin/python benchmark_machete.py --dtype float16 \
    model_bench --models meta-llama/Llama-3-8b --batch-sizes 1 16 128 --tp-sizes 1
# Marlin (Ampere-optimised) — same weight shapes
/opt/vllm-venv/bin/python benchmark_marlin.py --models meta-llama/Llama-2-7b-hf/TP1 \
    --batch-sizes 1 16 128
```
- `benchmark_marlin.py` sweeps `act_order ∈ {False,True}`, `is_k_full ∈ {False,True}`,
  `group_size ∈ MARLIN_SUPPORTED_GROUP_SIZES` (128 / -1 are the usual), quant types from
  `query_marlin_supported_quant_types` (uint4b8 / uint4 with zero-points, uint8b128, fp8, fp4).
  It generates random weights and quantises them itself — **no model download is required**
  despite the `--models` flag, which only selects K/N *shapes* from `weight_shapes.py`.
- Formats: **A is fp16/bf16 activations [M,K]; B is int4 packed 8-per-int32 in Marlin's permuted
  tile layout**, plus per-group fp16 scales (group 128 or channel-wise -1), optional zero-points
  (AWQ-style `uint4`) and optional `g_idx` act-order permutation. Wukong's W4A16 buffer layout will
  **not** match; the A/B must be done at the *operator* level (same M,K,N, same group size, same
  activation dtype, both verified against the same f64 dequant-then-GEMM reference), not by sharing
  buffers. Use `marlin_quantize` / `awq_marlin_quantize` from
  `vllm.model_executor.layers.quantization.utils.marlin_utils_test` to produce the packed weights, and
  dequantise them with `quantize_weights`'s reference path to build the oracle.
- Machete is the correct **H100** bar (built for Hopper); Marlin is the correct **A100** bar
  (Marlin was designed for Ampere and is documented as weak on H100). Reporting Machete on A100 or
  Marlin on H100 as "the int4 bar" would be a strawman in the other direction.

### 5.2 Alternatives considered

- `IST-DASLab/marlin` (the original standalone repo) — needs its own `pip install -e .` CUDA build and
  is unmaintained relative to vLLM's fork; **not recommended**, vLLM's wheel already contains it.
- `neuralmagic/quant_kernel_benchmarks` — a ready-made multi-library harness
  (`python benchmark_kernels.py --act-type bfloat16 --kernels torch_fp16,machete,fbgemm_i4,marlin,gemlite model_bench`).
  Useful as a cross-check if the vLLM scripts drift; adds `gemlite`/`fbgemm` deps.
- `flashinfer-python` 0.6.16.post2 (2026-08-06) — vLLM pulls it anyway; also worth knowing as an
  alternative FA2/FA3 peer with prebuilt cubins if §3 gets stuck.

---

## 6. Base image decision: 12.x-devel vs 13.0

### The finding

`crates/wukong_codegen_gpu/Cargo.toml:16-19` pins `cudarc = "0.16"` (lockfile: **0.16.6**) with feature
**`cuda-12060`**. I checked `cudarc v0.16.6`'s own `Cargo.toml`: its CUDA feature list **stops at
`cuda-12090`** — *there are no CUDA-13 bindings in 0.16.x at all*. (Latest cudarc on crates.io is
**0.19.8**, 2026-06-19; CUDA-13 features `cuda-13000…13030` appear on the current main branch.)

On a CUDA-13 image the sonames become `libcublas.so.13` / `libnvrtc.so.13`. cudarc's candidate list
contains `libX.so.12/.11/.10/.9/.1` but **not `.so.13`** — so the only thing that would let it load is
the unversioned `libcublas.so` dev symlink that `-devel` images happen to ship. It would then be
running **12.6-shaped bindings against a 13.x library**. That mostly works (I diffed
`cublasComputeType_t` across cudarc's `cuda-12060`, `cuda-12090` and `cuda-13000` variants: the values
Wukong uses are unchanged — `CUBLAS_COMPUTE_32F=68`, `CUBLAS_COMPUTE_32I=72`; 13.x only *adds*
`CUBLAS_COMPUTE_32F_EMULATED_16BFX9=78` and `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT=79`) — but it is an
untested ABI gamble across a **major** library version, taken on metered time, for zero benefit.

### The recommendation

**Pin `nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04`.**

- Verified to exist on Docker Hub (2026-05-20, 5.93 GB) and listed in NVIDIA's `supported-tags.md`.
- **The current default in `tools/cloud/modal_app.py:64` is `12.8.1-cudnn-devel-ubuntu22.04`, which
  NVIDIA's supported-tags list no longer carries a cudnn-devel variant for.** Bumping to 12.9.2 is a
  one-line change and puts the image on a supported tag.
- 12.9 is the **terminus of the CUDA-12 line for pip redist wheels** (`nvidia-cublas-cu12` max
  12.9.2.10), so Route A and Route B (§1) are then the *same* CUDA version — the peer libraries are
  identical whether the box is a container or a bare VM.
- It matches the newest torch that still has a CUDA-12 build: **torch 2.13.0+cu129**.
- It is closest to cudarc's own top binding (`cuda-12090`). **Recommended follow-up (a real code
  change, out of D5's scope): flip the feature `cuda-12060` → `cuda-12090`.** The repo's own
  `peer_env_hint()` already admits "cudarc 0.16's `cuda-12060` bindings actually reference 12.8-era
  NVRTC PCH symbols and 12.9-era cuBLAS emulation symbols" — so the pin is already de facto 12.9.
  The enum diff above shows the bump is additive-only for everything Wukong calls.

**Does any peer need the toolkit to match the driver's CUDA API version? No.**
CUDA is backward compatible: an application built against an older toolkit runs on any newer driver.
Verified driver minimums from the CUDA Toolkit release notes (Table 3):

| toolkit | min Linux driver |
|---|---|
| CUDA 13.0 GA / U1 / **U2** | 580.65.06 / 580.82.07 / **580.95.05** |
| CUDA 12.9 GA / U1 | 575.51.03 / 575.57.08 |
| CUDA 12.x minor-version compatibility | ≥ 525.60.13 |

The Modal hosts observed at **580.95** are exactly CUDA 13.0 Update 2's driver — comfortably above
12.9's 575.x floor. The only peer that *requires* a 580+ driver is anything shipping **cu130 torch**
(vLLM 0.26 → torch 2.11.0 from PyPI; and torch's PyPI default in general). That is satisfied on Modal
but is a live risk on a marketplace box with an r535/r550 driver — **check `nvidia-smi` first**.

If a future rung genuinely forces CUDA 13 (e.g. a Blackwell B200/sm_100 target, where the toolchain
story is CUDA-13-shaped), the prerequisite is a **cudarc 0.16.6 → 0.19.x bump with feature
`cuda-13000`** — three minor versions of API drift across a crate that owns every driver call in
`wukong_codegen_gpu`. Budget that as its own commit series, gated by the full device suite; it is
plan risk #3 and this dossier confirms it is real but **deferrable**.

---

## 7. The consolidated image + the prep budget

Recommended additions to `tools/cloud/modal_app.py`'s image (all `$0`, all on CPU containers):

```python
WK_CUDA_TAG = os.environ.get("WK_CUDA_TAG", "12.9.2-cudnn-devel-ubuntu22.04")   # was 12.8.1

image = (
    modal.Image.from_registry(f"nvidia/cuda:{WK_CUDA_TAG}", add_python="3.12")
    .apt_install("curl", "ca-certificates", "build-essential", "pkg-config", "git",
                 "cmake", "ninja-build")                    # cmake+ninja: CUTLASS / FA builds
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --profile minimal --default-toolchain stable --component rustfmt clippy",
        # peer venv 1: torch.compile / Inductor + FA4 + the fa2_sdpa_peer.py driver
        "python3 -m venv /opt/torch-venv && /opt/torch-venv/bin/pip install -U pip && "
        "/opt/torch-venv/bin/pip install --index-url https://download.pytorch.org/whl/cu129 "
        "  torch==2.13.0+cu129 && "
        "/opt/torch-venv/bin/pip install numpy ninja packaging && "
        "/opt/torch-venv/bin/pip install --pre flash-attn-4==4.0.0b25",
        # peer venv 2: vLLM (Marlin/Machete). Isolated: it hard-pins torch==2.11.0.
        "python3 -m venv /opt/vllm-venv && /opt/vllm-venv/bin/pip install -U pip && "
        "/opt/vllm-venv/bin/pip install vllm==0.26.0 && "
        "git clone --depth 1 --branch v0.26.0 https://github.com/vllm-project/vllm /opt/vllm-src",
    )
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
        # NOTE: /usr/local/cuda/lib64 only. NEVER add .../lib64/stubs.
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
        "WUKONG_FA2_PYTHON": "/opt/torch-venv/bin/python",
        "WUKONG_FA2_PEER": "/wukong/tools/fa2_sdpa_peer.py",
        "WUKONG_PEER_REQUIRED": "1",
        "TORCHINDUCTOR_CACHE_DIR": "/persist/inductor-cache",
        "TRITON_CACHE_DIR": "/persist/triton-cache",
        "CARGO_TERM_COLOR": "always", "RUST_BACKTRACE": "1",
    })
    .add_local_dir(str(REPO_ROOT), REMOTE_SRC, ignore=IGNORE, copy=False)
)
```

CUTLASS profiler, FA3 and (if needed) the FA2 wheel are **not** in the image — they are Volume
artifacts built once by a CPU-only Modal function and reused, so the image stays reproducible and
the multi-hour builds never invalidate the image cache.

| item | wall time | disk | where it runs | GPU $ |
|---|---|---|---|---|
| image pull (12.9.2-cudnn-devel) | *(est.)* 3–6 min first pull, then cached | 5.93 GB pull / *(est.)* ~12 GB unpacked | Modal build | $0 |
| rustup + apt (cmake, ninja) | *(est.)* 2–3 min | *(est.)* ~1.5 GB | Modal build | $0 |
| torch 2.13.0+cu129 venv (+numpy) | *(est.)* 3–6 min | *(est.)* ~7 GB | Modal build | $0 |
| flash-attn-4 4.0.0b25 | *(est.)* 30–90 s | *(est.)* ~1.5 GB | Modal build | $0 |
| vLLM 0.26.0 venv + src | *(est.)* 6–12 min | *(est.)* ~9 GB | Modal build | $0 |
| **cutlass_profiler (sm90a, filtered)** | *(est.)* **20–45 min** | *(est.)* 8–15 GB build → ~200 MB artifact | **CPU-only fn → Volume** | $0 |
| FA3 hopper wheel (only if fp8/bwd) | *(est.)* 15–90 min | *(est.)* ~10 GB build → ~200 MB whl | **CPU-only fn → Volume** | $0 |
| FA2 wheel (only if standalone A100 call) | *(est.)* 20–50 min (1 arch, ninja) | *(est.)* ~8 GB build → ~300 MB whl | **CPU-only fn → Volume** | $0 |
| Inductor max-autotune cache warm | *(est.)* 60–180 s | *(est.)* <1 GB | cheapest GPU (L4 $0.80/hr) | *(est.)* ~$0.05 |
| **all peer smoke tests, GPU** | **≤ 5 min total** | — | target GPU | *(est.)* ≤ $0.35 on H100 |

**Total prep: *(est.)* 1.5–3 h of CPU/build time (≈$0.10 at Modal CPU rates), ~40–55 GB of Volume,
and under $0.50 of metered GPU time.** Every number above is a one-time cost; the Volume makes round 2
onward nearly free.

---

## 8. Smoke-test battery (each ≤60 s of GPU time)

Run in this order; stop at the first failure. Total ≈4–5 min of metered time.

```sh
# 0. Provenance (plan §6.1) — free, and the cheapest proof nvcc+driver work.
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm --format=csv
nvidia-smi -L && nvcc --version | tail -1
python3 -c "import ctypes;[ctypes.CDLL(n) for n in ('libcublas.so','libcublasLt.so','libcudnn.so','libnvrtc.so','libcuda.so.1')];print('peers dlopen OK')"

# 1. In-tree cuBLAS + NVRTC peers (§1.3)              ~10 s
WUKONG_PEER_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu --release \
    reproducibility_vs_cublas -- --ignored --nocapture
# 2. In-tree cuDNN conv peer                          ~20 s
WUKONG_PEER_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu --release \
    conv_vs_cudnn -- --ignored --nocapture
# 3. Torch + Inductor + Triton (§2.3)                 ~30 s warm / ~3 min cold
/opt/torch-venv/bin/python /opt/smoke_inductor.py
# 4. The out-of-process FA2/SDPA peer the harness drives   ~20 s
WUKONG_PEER_REQUIRED=1 WUKONG_FA2_PYTHON=/opt/torch-venv/bin/python \
  cargo test -p wukong_codegen_gpu --features gpu --release \
    attn_vs_fused_peer -- --ignored --nocapture
# 5. FlashAttention-4 direct call (H100 only) (§3.4)  ~40 s incl. first JIT
/opt/torch-venv/bin/python /opt/smoke_fa4.py
# 6. CUTLASS profiler, one shape (§4)                 ~15 s
/persist/bin/cutlass_profiler --operation=Gemm --op_class=tensorop --m=4096 --n=4096 --k=4096 \
  --A=f16:row --B=f16:column --C=f16:column --accumulator-type=f32 \
  --providers=cutlass,cublas --warmup-iterations=10 --profiling-iterations=50
# 7. Marlin/Machete (§5)                              ~40 s
/opt/vllm-venv/bin/python /opt/vllm-src/benchmarks/kernels/benchmark_machete.py \
  --dtype float16 square_bench --dim-start 4096 --dim-end 4096 --dim-increment 1
```

---

## 9. Pitfall index (the things that will actually cost money)

1. **`/usr/local/cuda/lib64/stubs` on `LD_LIBRARY_PATH`** → cudarc loads the stub `libcuda.so` and
   everything looks like "no GPU". The single highest-blast-radius Linux landmine here.
2. **Forgetting `WUKONG_PEER_REQUIRED=1`** → the harness politely skips every peer and publishes a
   green run that measured nothing. Set it in the image `ENV`.
3. **`WUKONG_FA2_PYTHON` unset on Linux** → the default path is `tools/torch-cuda-venv/Scripts/python.exe`
   (Windows). The FA2 peer silently reports unavailable.
4. **pip redist route: only one of the four `nvidia/*/lib` dirs on the path** → cuDNN in particular
   skips quietly. The Linux restatement of the repo's known Windows bug.
5. **Mixing torch's bundled `site-packages/nvidia/*/lib` into the Rust process's `LD_LIBRARY_PATH`** →
   a cu130 torch drops `libcublas.so.13` next to a 12.9 system library. Keep venvs isolated.
6. **`pip install flash-attn-4` without `--pre`** → every release is `4.0.0bNN`, so pip resolves nothing
   and it looks unavailable.
7. **`pip install flash-attn` against torch 2.13** → no prebuilt wheel exists (assets stop at torch 2.8
   cu12 / torch 2.9 cu13); it starts a source compile, and *without `ninja` it is a ~2 h single-threaded
   build*. Install `ninja packaging` first, set `FLASH_ATTN_CUDA_ARCHS` to one arch, and build on CPU.
8. **CUTLASS `cmake` without `CUTLASS_LIBRARY_KERNELS`** on sm90 → NVIDIA's own docs say kernel
   generation alone "can take hours" (millions of instantiations). Always filter, always
   `-DCUTLASS_NVCC_ARCHS=90a` (the `a` matters), always `-DCUTLASS_ENABLE_TESTS=OFF`.
9. **CUTLASS release tarballs are the CuTe DSL runtime, not the profiler** — verified by listing one.
   Do not plan on a prebuilt profiler.
10. **`vllm==0.26.0` hard-pins `torch==2.11.0`** — installing it into the torch-2.13 venv silently
    downgrades torch and breaks §2 and §3.1. Separate venv, always.
11. **vLLM's torch 2.11 is a cu130 wheel** → needs driver ≥ 580.65.06. Fine on Modal (580.95), not
    guaranteed on marketplace hosts.
12. **First-call JIT is not a benchmark.** FA4 (CuTeDSL) and Inductor max-autotune both compile on the
    first call for each new shape. Warm every shape; persist `TORCHINDUCTOR_CACHE_DIR` /
    `TRITON_CACHE_DIR` on the Volume; keep both inside one Modal invocation (plan risk #7).
13. **Old TF32 API + new TF32 API together is unsupported** in torch ≥2.9. Use only
    `torch.backends.cuda.matmul.fp32_precision` / `torch.backends.cudnn.conv.fp32_precision`.
14. **Compile on CPU, run on GPU** — CUTLASS, FA2, FA3 and the 21 Rust crates all build without a
    device. `modal_app.py` already does this for cargo; extend the same discipline to every peer.
