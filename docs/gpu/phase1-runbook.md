# Phase 1 batch runbook — rented `sm_89` bring-up (Modal L4)

**Status: pre-flight. Written before any metered minute, per `GPU_RETARGET_PLAN.md` §0:
*"Write the complete batch script you intend to run — never sit at a GPU shell deciding what to
do next."*** Execute it verbatim. If a step's outcome is not in the table, **stop the meter and
come back to the laptop** — thinking is free at home and costs $0.80/hr on an L4.

Scope: plan §5 Phase 1 steps 1–2 (correctness + Linux behaviour checks) on **Modal L4**
(`sm_89`, 58 SMs, $0.80/hr). Step 3 (L40S SM-scaling) is the **next** session — previewed in §5,
deliberately not scripted here.

Every command below was cross-checked against `tools/cloud/modal_app.py` at the signatures that
exist today (§7 has the flag derivation). **This document does not own `tools/cloud/*`** — where
the harness has a sharp edge, the runbook works around it and §8 records it.

---

## 0. Preconditions — all of them, before the first `modal run`

| # | Precondition | How to verify (all free) | If it fails |
|---|---|---|---|
| 0.1 | **Workspace budget cap set** at <https://modal.com/settings/usage> | Read the page. Plan §11 item 2 lists this as **the only blocker for Phase 1**. Set $30 (or $25). If no payment method is on file, leave it that way | **Do not run anything.** An uncapped workspace is how a stuck container becomes a bill |
| 0.2 | Modal client authenticated | `modal --version` → `1.5.3`; `modal volume list` returns without an auth error | `modal setup` (browser auth, one time) |
| 0.3 | You are in the **main checkout**, not an agent worktree | `git rev-parse --show-toplevel` must print `…/Mercury` | `cd` there. `modal_app.py:71` derives `REPO_ROOT` from its own path, so a worktree ships *that* branch's tree |
| 0.4 | **Clean tree** on the intended commit | `git status --porcelain` prints **nothing**; record `git rev-parse HEAD` | Commit or stash first. The mount ships the **working tree, not a commit** (§2.3) — a dirty metered run is a provenance violation |
| 0.5 | Local two-part gate green at that commit | `cargo test` then `cargo check --features gpu --all-targets`, unpiped, as two separate steps | Fix at home. Never discover a build break on the meter |
| 0.6 | `bench/gpu/l4/` exists for the logs | `mkdir` per §2.1 | — |
| 0.7 | No source edits between `::build` and `::test` | Discipline. Any edit re-uploads the mount and re-triggers a **GPU-side compile** (§4.5) | Re-run `::build` (CPU, ~free) before the metered step |

**Profile law:** `::bench` hardcodes `--release` (`modal_app.py:369`) while `::build` defaults to
**debug** (`modal_app.py:300`). If the profile of the metered step does not match a profile already
built into the Volume, **the GPU box compiles 21 crates at GPU prices.** This runbook therefore
builds *both* profiles on CPU (S1a/S1b) and runs every metered step `--release`.

---

## 1. The batch — L4 bring-up, in order

Set the session env once. `WK_GPU` is read at **import time** (`modal_app.py:59`), so it must be
exported in the same shell as every `modal run`. `WK_CUDA_TAG` must be **identical across all
steps** or a second image is built and the Volume's warm target dir is used against different
system libraries.

**PowerShell (primary)**

```powershell
cd C:\Users\Quant\Documents\Programming\Projects\Compiler\Mercury
$env:WK_GPU       = "L4"
$env:WK_TIMEOUT   = "5400"                                    # 90 min hard cap; see S2
$env:WK_CUDA_TAG  = "12.9.2-cudnn-devel-ubuntu22.04"          # = modal_app.py:64 default
$D = Get-Date -Format "yyyy-MM-dd"
New-Item -ItemType Directory -Force bench\gpu\l4 | Out-Null
```

**bash**

```sh
cd /c/Users/Quant/Documents/Programming/Projects/Compiler/Mercury
export WK_GPU=L4 WK_TIMEOUT=5400 WK_CUDA_TAG=12.9.2-cudnn-devel-ubuntu22.04
D=$(date +%F); mkdir -p bench/gpu/l4; set -o pipefail
```

Each step below is: **command → expected wall → expected $ → MUST record → ABORT if.**
Write the §2.2 provenance header into the log file *first*, then append the tee'd output.

---

### S0 — Provenance (`::device_info`) — GPU, seconds

```powershell
$log = "bench\gpu\l4\$D-s0-device-info.log"
# write the §2.2 header into $log first, then:
modal run tools/cloud/modal_app.py::device_info 2>&1 | Tee-Object -FilePath $log -Append
$rc = $LASTEXITCODE; "exit_code : $rc" | Add-Content $log
```

```sh
log=bench/gpu/l4/$D-s0-device-info.log
modal run tools/cloud/modal_app.py::device_info 2>&1 | tee -a "$log"; rc=${PIPESTATUS[0]}
echo "exit_code : $rc" >> "$log"
```

- **Wall:** 1–3 min the first time (image pull + build of the rustup layer), <60 s after.
- **$:** ≈$0.02–0.05 (L4 seconds). First-ever run also pays the image build — CPU-side, ~$0.
- **MUST record:** the whole `nvidia-smi` block; every line of the CUDA-properties probe
  (`name`, `compute_capability`, `sm_count`, `smem_optin`, `smem_per_sm`, `regs_per_sm`,
  `max_blocks_per_sm`, `warps_per_sm`, `l2_cache`, `total_vram`, `mem_clock`, `peak_bw_GBs`,
  `sm_clock_max`, `cooperative_launch`); the `--- provenance gate ---` verdict; **all five
  peer-library `ldconfig` lines**; `driver_api` / `runtime_api`.
- **Expected:** `sm_89`, **58 SMs** (`_EXPECTED["L4"]`, `modal_app.py:220`), `mig.mode.current`
  = `N/A` or `Disabled`, driver ≥ 575.x, all five sonames resolved from
  `/usr/local/cuda/lib64` + `/usr/lib/x86_64-linux-gnu`.
- **ABORT if:**
  - The gate prints `!! MISMATCH` (wrong CC or SM count) → **abort the session and publish
    nothing** from this device (plan §6.1). ⚠ `device_info` **still exits 0** on a mismatch
    (`modal_app.py:283-285`) — a human must read the line; never chain the next step on `$rc`.
  - `!! nvcc failed` → **stop.** The toolkit is unusable; the peers cannot be honest and the image
    is wrong. Fix the image locally, re-run S0 only (~$0.03). Same exit-0 caveat
    (`modal_app.py:259-262`).
  - Any soname `NOT FOUND` → do **not** install anything on the meter (§3 branch B).
  - `l2_cache` disagrees with the spec sheet → record the **probed** number and use it; the 4050
    taught this exact lesson (probe says 24 MiB, spec says 12 — `docs/gpu/derive/README.md`).

---

### S1a — CPU build, debug (`::build`) — **NO GPU**

```powershell
modal run tools/cloud/modal_app.py::build 2>&1 |
  Tee-Object -FilePath "bench\gpu\l4\$D-s1a-build-debug.log" -Append
```
```sh
modal run tools/cloud/modal_app.py::build 2>&1 | tee -a bench/gpu/l4/$D-s1a-build-debug.log
```

- **Wall:** 20–45 min cold (full compile + crates.io download), 2–5 min warm.
- **$:** CPU only — `cpu=8.0` at ≈$0.05/core-hr (`modal_app.py:9`) ⇒ **≈$0.15–0.30**. No GPU.
- **MUST record:** `rustc --version`, `cargo --version`, the `cargo check --features gpu
  --all-targets` result (**this is the gate `cargo test` never runs**), and the **CPU workspace
  suite result on Linux** — this is Phase 0's exit criterion measured on the target OS.
- **Expected:** check clean; `cargo test --workspace` green *except* CPU-perf-shaped differences —
  the 256-bit AVX2 vectorizer is Win64-ABI-only and drops to 128-bit on Linux, thread pinning is
  `kernel32`-only (plan §2.6). CPU benchmarks are **out of scope** (plan §9); a CPU *correctness*
  failure is a real finding and blocks S2.
- **ABORT if:** `cargo check --features gpu --all-targets` fails → a Linux-only type error. Fix at
  home; this costs $0 to rediscover locally and would have cost GPU minutes to discover in S2.

### S1b — CPU build, release (`::build --release`) — **NO GPU**

```powershell
modal run tools/cloud/modal_app.py::build --release 2>&1 |
  Tee-Object -FilePath "bench\gpu\l4\$D-s1b-build-release.log" -Append
```
```sh
modal run tools/cloud/modal_app.py::build --release 2>&1 | tee -a bench/gpu/l4/$D-s1b-build-release.log
```

- **Wall:** 25–55 min cold; **$:** ≈$0.20–0.40, CPU only.
- **MUST record:** same fields; note that `cargo test --workspace --release` also ran.
- **Why both profiles:** S1a gives the Linux CPU gate in the profile the repo's law uses; S1b gives
  the artifacts every metered step needs. Both are CPU; together they cost less than **one** minute
  of H100 time. **Never** let a metered step be the first to build a profile.
- **ABORT if:** release-only failure (overflow-check-dependent behaviour, `debug_assertions`
  divergence) → a genuine finding; fix at home.

---

### S2 — Device suite, pass 1, peers OFF (`::test --release`) — GPU, the main event

```powershell
$log = "bench\gpu\l4\$D-s2-test-pass1-release.log"
modal run tools/cloud/modal_app.py::test --release --filter=--nocapture 2>&1 |
  Tee-Object -FilePath $log -Append
$rc = $LASTEXITCODE; "exit_code : $rc" | Add-Content $log
```
```sh
log=bench/gpu/l4/$D-s2-test-pass1-release.log
modal run tools/cloud/modal_app.py::test --release --filter=--nocapture 2>&1 | tee -a "$log"
echo "exit_code : ${PIPESTATUS[0]}" >> "$log"
```

> **`--filter=--nocapture` is deliberate, not a typo.** `::test` appends `filter` verbatim after
> `--` (`modal_app.py:343,346-352`), so this lands in libtest's argument position and turns
> `--nocapture` on. Without it libtest **captures the output of passing tests**, which silently
> discards every `[skip] …` line — and "learn what is missing instead of failing the whole suite"
> is the entire stated purpose of pass 1 (`modal_app.py:336-337`, plan §5 Phase 1 step 1).
> Use the `=` form; `--filter --nocapture` would be parsed as a missing value by click.
> If `modal_app.py` later grows a real `nocapture` flag, use that instead.

- **Wall: UNKNOWN — and measuring it is the deliverable** (plan §5 Phase 1 step 1, §6.6, risk #6).
  `WK_TIMEOUT=5400` caps the exposure at 90 min ⇒ **≤ $1.20** worst case. Working expectation
  10–45 min; the two corpus gates dominate (357 `tests/run` programs × 2 opt levels through
  `lower.rs` + 103 eligible through `megakernel.rs`).
- **$:** ≈$0.15–0.60 expected, ≤$1.20 capped.
- **MUST record (first-class deliverables):**
  1. **Total wall time per package** — the harness prints `-> ok in NNN.Ns` per command
     (`modal_app.py:152`). **This number sizes Phases 3/3b** and goes in the §6 ledger and in the
     round summary.
  2. `test result:` lines for `wukong_codegen_gpu` **and** `wukong_driver` (pass/fail/ignored counts).
  3. **Every `[skip]`, `[skip:capability]` and `peer` line** — that is what `--nocapture` bought.
  4. The two corpus gates by name and their coverage counts:
     `run_corpus_matches_interp_oracle` (floor `RUN_CORPUS_COVERAGE_FLOOR = 217`, `lower.rs:4260`)
     and `mega_corpus_matches_oracle` (floor `MEGA_CORPUS_COVERAGE_FLOOR = 87`,
     `megakernel.rs:229`). A *higher* count on Linux is a finding; a lower one trips the ratchet.
  5. Any `cudarc` / driver diagnostics — plan risk #3 (`cuda-12060` bindings vs an r5xx/CUDA-13 host)
     is answered here, for free, as a side effect.
- **Expected:** `WUKONG_GPU_REQUIRED=1` is set by `::test` (`modal_app.py:339`), so **no device gate
  may skip**: a `[skip] …: GPU unavailable` becomes an assertion failure (`gpu.rs:6352-6356`).
  fp8 gates **must run** on L4 (cc 8.9 ≥ the fp8 floor) — a `[skip:capability]` for fp8 here is a
  **gate bug**, and `with_cap` escalates it (`gpu.rs:6374-6392`).
- **ABORT if:**
  - `CUDA device lost after an earlier in-process kernel fault` / `device_lost` (`megakernel.rs:238-244`,
    `gpu.rs:647`) → **kill the run now.** Every later test in that process is a cascade of false
    failures; the remaining minutes buy nothing. Capture the log, fix at home.
  - The log shows `Compiling wukong_*` → **you are compiling at GPU prices.** Ctrl-C (the
    non-detached `modal run` stops the app), re-run the matching `::build` on CPU, then retry (§4.5).
  - Any failure at all → **do not iterate on the meter.** Capture the log, end the session, fix
    LOCALLY, re-run `::build`, and re-run S2. That loop is the §0 law; an interactive shell here is
    exactly the "workbench" the plan forbids.
  - No output for >10 min after the container starts → suspect the timeout path; stop and check
    `modal app list` / `modal app logs`.

---

### S3 — Pass 2: the peers decision (§3 decides *which* of these you run)

Pass 2 is **not** "the same suite again with `--peers`". Evidence: the crate has **38**
`peer_gate("…")` sites, and exactly **one** of them is in a non-`#[ignore]`d test —
`cublaslt_fp8_matches_reference_within_tol` (`gpu.rs:14997-15005`). The other 37 are `#[ignore]`d
benches, which `::test` never runs (it passes no `--ignored`). So a whole-suite `--peers` re-run
costs a **second full suite** to add **one** assertion. Run these instead, in order, and stop at the
branch §3 sends you to.

**S3a — the one peer test in the default suite (seconds, ≈$0.01):**

```powershell
modal run tools/cloud/modal_app.py::test --release --peers --no-driver --filter cublaslt_fp8 2>&1 |
  Tee-Object -FilePath "bench\gpu\l4\$D-s3a-peers-cublaslt-fp8.log" -Append
```
```sh
modal run tools/cloud/modal_app.py::test --release --peers --no-driver --filter cublaslt_fp8 2>&1 |
  tee -a bench/gpu/l4/$D-s3a-peers-cublaslt-fp8.log
```

(`--no-driver` skips the `wukong_driver` binary, which has no test matching this filter.)

**S3b — the in-tree cuBLAS/NVRTC peer smoke (D5 §1.3), ≈20 s, ≈$0.01:**

```powershell
modal run tools/cloud/modal_app.py::bench --name reproducibility_vs_cublas 2>&1 |
  Tee-Object -FilePath "bench\gpu\l4\$D-s3b-peer-smoke-cublas.log" -Append
```

**S3c — the cuDNN peer smoke (D5 §1.3), ≈30 s, ≈$0.01:**

```powershell
modal run tools/cloud/modal_app.py::bench --name conv_vs_cudnn 2>&1 |
  Tee-Object -FilePath "bench\gpu\l4\$D-s3c-peer-smoke-cudnn.log" -Append
```

bash form for both (`::bench` is release-only by construction — no `--release` flag exists):

```sh
modal run tools/cloud/modal_app.py::bench --name reproducibility_vs_cublas 2>&1 |
  tee -a bench/gpu/l4/$D-s3b-peer-smoke-cublas.log
modal run tools/cloud/modal_app.py::bench --name conv_vs_cudnn 2>&1 |
  tee -a bench/gpu/l4/$D-s3c-peer-smoke-cudnn.log
```

- **MUST record:** for each, whether the peer **resolved** or the log shows a `peer_gate` skip, plus
  the library/algorithm names the peer prints (`cudnn_fwd_algo_name`).
- **`::bench` always passes `--nocapture --test-threads=1`** (`modal_app.py:372`), so skips are
  visible — but `::bench` does **not** set `WUKONG_PEER_REQUIRED` (§8 finding 4). Read the lines;
  do not infer availability from a green exit.
- **ABORT/NEVER:** never `pip install` a peer on the metered box. Peer staging is a **CPU/image**
  job (D5 §7, "compile on CPU, run on GPU"). A 3–6 min torch install on an L4 is $0.05 of waste and
  a precedent that costs $0.40 on an H100.

> ⚠ **Do not run `::bench` with no `--name`.** The default `name=""` runs **all 87** `#[ignore]`d
> sweeps single-threaded (`modal_app.py:361,369-372`). That is an unbounded metered run.

---

### S4 — Linux behaviour checks (plan §5 Phase 1 step 2) — read them out of S2/S3

These need **no extra invocation**; they are observations from the S2/S3 logs. Record each as
ANSWERED / UNANSWERED in the round summary:

| Question (plan §8 / tools/cloud README) | Where the answer is |
|---|---|
| Does `cudarc 0.16` (`cuda-12060`) work against the cloud driver? | S2 ran ~130 device gates — a green suite *is* the answer |
| Does the "legacy 8×`.b32`" f16 WMMA fragment spelling still JIT? | S2's wmma gates; a JIT rejection names the module |
| Sticky-fault / device-lost policy under the Linux driver (`gpu.rs` recovery premise is WDDM) | Only observable **if** a fault occurred. If S2 was clean, mark **UNANSWERED** — do not provoke a fault on the meter |
| CUDA-graph capture on r5xx (`graph.rs`) | S2's graph/pool gates |
| Cubin cache on the mounted Volume | `WUKONG_CUBIN_CACHE=/persist/cubin-cache` (`modal_app.py:140`); confirm a warm second run is faster, and that the cache is arch+driver-keyed (`cubin.rs:97-112`) so L4 and L40S entries cannot collide |
| **Suite wall time** | S2, the deliverable |

---

### S5 — Teardown and ledger (local, free)

```powershell
modal app list              # nothing of ours should still be running
modal volume list           # 'wukong-build' should exist and be growing
```

Then: strip ANSI (§2.4), fill the §6 ledger from <https://modal.com/settings/usage>, write the round
summary, commit (§2.5).

---

## 2. Recording protocol

### 2.1 Where the logs live

```
bench/gpu/<device>/<YYYY-MM-DD>-s<N>-<step>.log
```

For this session, `<device>` = `l4`:

```
bench/gpu/l4/2026-08-07-s0-device-info.log
bench/gpu/l4/2026-08-07-s1a-build-debug.log
bench/gpu/l4/2026-08-07-s1b-build-release.log
bench/gpu/l4/2026-08-07-s2-test-pass1-release.log
bench/gpu/l4/2026-08-07-s3a-peers-cublaslt-fp8.log
bench/gpu/l4/2026-08-07-s3b-peer-smoke-cublas.log
bench/gpu/l4/2026-08-07-s3c-peer-smoke-cudnn.log
bench/gpu/l4/2026-08-07-session.md          # provenance + ledger + findings + verdict
```

Plan §6.5 requires raw round logs under `bench/gpu/<device>/`; `bench/gpu/README.md` holds the
layout rules. Re-running a step the same day: suffix `-r2`, never overwrite.

### 2.2 The §6.1 provenance block — at the top of **every** log

Write it into the file *before* the command appends its output. Copy-paste template:

```
# ---- WUKONG ROUND PROVENANCE (GPU_RETARGET_PLAN.md §6.1) ----
round_id         : 2026-08-07-l4-s2-test-pass1
local_time_start : 2026-08-07T14:03:11+10:00
operator         : <who>
checkout         : C:\...\Mercury            (MAIN checkout, not a worktree)
git_branch       : main
git_head         : <40-char sha>  <subject>
git_dirty        : NO            # `git status --porcelain` was EMPTY — see 2.3
provider / SKU   : Modal / L4    (WK_GPU=L4)
image tag        : nvidia/cuda:12.9.2-cudnn-devel-ubuntu22.04   (WK_CUDA_TAG)
modal client     : 1.5.3
command          : modal run tools/cloud/modal_app.py::test --release --filter=--nocapture
cargo profile    : release
env              : WK_TIMEOUT=5400  WUKONG_GPU_REQUIRED=1 (set by ::test)  WUKONG_PEER_REQUIRED=unset
device           : <name> / <cc> / <sm_count> SMs / smem_optin <KiB> / L2 <MiB> / VRAM <MiB>
driver / runtime : <driver_api> / <runtime_api>          (from S0)
MIG              : <mig.mode.current>                    (from S0)
provenance gate  : OK | MISMATCH                         (from S0 — MISMATCH ⇒ publish nothing)
clock lock       : NONE — Modal is a container (plan §6.3). ITERATION data, NOT publication data.
wall_time        : <filled in after>
exit_code        : <filled in after>
cost_estimate    : <seconds> x $0.80/hr = $<x.xx>   (dashboard actual: $<y.yy>)
# --------------------------------------------------------------
```

Rules: **a log without this block is not evidence.** Fields are filled from S0 for every later step
in the same session (the device does not change inside a session, but it *may* change between
invocations — plan risk #7 — so each log states which S0 it inherits).

### 2.3 Git HEAD and dirty state — mandatory, and here is why

`modal_app.py:113` mounts `add_local_dir(REPO_ROOT, …, copy=False)`. **The mount ships the WORKING
TREE, not a commit.** `.git` is excluded (`modal_app.py:78`), so *the container cannot know what it
is running* — nothing on the remote side can reconstruct the source's identity. Therefore:

- **Record `git rev-parse HEAD` and `git status --porcelain` LOCALLY, immediately before each
  metered command**, into that log's provenance block.
- **A dirty tree on a metered run is a provenance violation.** The numbers cannot be tied to any
  commit, cannot be reproduced, and must not be published. If `git status --porcelain` prints
  anything, commit or stash *before* spending the money — not after.
- Run from the **main checkout**: `REPO_ROOT` is derived from `modal_app.py`'s own path
  (`modal_app.py:71`), so invoking it from an agent worktree silently ships that worktree's branch.
- Mount identity is content-addressed (filename + sha256 + mode — `modal/mount.py:556-561` in the
  installed client 1.5.3, `MountFile` carries no timestamp) — no mtime
  travels with it. So cargo's incremental fingerprints on the Volume are only trustworthy while the
  mount is **byte-identical** between `::build` and `::test`. One edit in between ⇒ re-upload ⇒
  GPU-side recompile (§4.5).

### 2.4 Log hygiene

The image sets `CARGO_TERM_COLOR=always` (`modal_app.py:109`), so raw logs contain ANSI escapes.
Strip before committing:

```powershell
$t = Get-Content $log -Raw; ($t -replace "`e\[[0-9;]*[a-zA-Z]", '') | Set-Content $log -NoNewline
```
```sh
sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$log"
```

Logs are **append-only records**. Never edit a log to look better; a correction is a new commit that
says what was wrong (repo law, and the four retractions this project has already published).

### 2.5 Committing

- `git add bench/gpu/l4/<files>` — **explicitly, never `git add -A`.**
- One commit for the session's raw logs + summary. Conventional subject, prose body naming what was
  run, the defect/finding if any, the reproducer, and a `Gate:`/`Verified:` line quoting real output.
  **No `Co-Authored-By` trailer.**
- Suite wall time goes in the commit body — it is the number Phases 3/3b are budgeted from.
- `BENCHMARKS.md` / `docs/metrics.md` / `docs/compile-floor.md` are **measurement documents**: a
  container round is iteration data (plan §6.3) and **does not** go in them.

---

## 3. Pass-2 peers decision tree

Enter with S0's five `ldconfig` lines and S3a–S3c's output.

```
S0: all five sonames resolved?  (libcuda.so.1, libnvrtc.so.12, libcublas.so.12,
                                 libcublasLt.so.12, libcudnn.so.9)
├─ NO ────────────────────────────────────────────────────────────► BRANCH B
└─ YES → S3a (cublaslt_fp8, --peers) 
         ├─ PASS → S3b/S3c
         │         ├─ both report a real peer timing ─────────────► BRANCH A  (done)
         │         └─ either logs a peer_gate skip ───────────────► BRANCH B
         └─ FAIL → read the reason in the log
                   ├─ "no fp8 algo for the shape on this device" ─► BRANCH C
                   └─ library not loadable / wrong soname ────────► BRANCH B
torch / FA2 peer (any *_vs_fused_peer bench) ────────────────────► BRANCH D
```

**BRANCH A — peers resolve.** Record in the session summary: peer tier = cuBLAS + cuBLASLt + NVRTC +
cuDNN, resolved from the devel image, `WUKONG_PEER_REQUIRED` honoured. Phase 1's peer question is
answered; **stop.** Do not run more peer benches — Phase 1 is a correctness phase and L40S/L4
numbers are not headline numbers (plan §5 Phase 1 step 4).

**BRANCH B — a library does not resolve.** Do **not** install anything on the meter.
1. End the GPU session now.
2. Diagnose at home from the log against D5 §1.1: the devel image ships the unversioned dev symlinks
   (`libcublas.so` → `.so.12`), and `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu`
   is already correct at `modal_app.py:108`. A miss therefore means the image tag changed or a
   `-runtime` variant crept in — an image fix, owned by `tools/cloud/*`.
3. **Never add `/usr/local/cuda/lib64/stubs` to `LD_LIBRARY_PATH`** (D5 §1.1 LANDMINE): cudarc's
   first driver candidate is literally `libcuda.so`, the stub wins, and every driver call fails with
   a symptom that looks exactly like "no GPU".
4. Re-run **S0 only** (~$0.03) to confirm the fix, then resume at S3a.

**BRANCH C — cuBLASLt has no fp8 algo for those shapes on L4.** This is a *device/library*
capability outcome, not a staging error — and under `WUKONG_PEER_REQUIRED=1` it is reported as a
hard FAIL because `peer_gate` asserts (`gpu.rs:7466-7470`). Record it as a **finding, not a
regression**: name it in the session summary, note that the same test is expected to behave
differently on H100, and *do not* set `--peers` on the whole suite in this session (that is exactly
how a non-defect turns a bring-up red). File it against Phase 3, where cuBLASLt fp8 is the peer that
matters.

**BRANCH D — torch / FlashAttention peers are unresolvable in the image. This is EXPECTED today.**
The image installs no torch (`modal_app.py:96-114` — rustup only), and `fa2_peer_available()`
probes `WUKONG_FA2_PYTHON` or falls back to `tools/torch-cuda-venv/bin/python`
(`baselines.rs:2583-2594`), which is not in the mount. So every `*_vs_fused_peer` bench skips by
construction.
- **Correct action: record it and move on.** Phase 1's exit criterion (plan §5) is the *device
  suite*, not the peer scoreboard. Nothing in Phase 1 is published, so a missing torch peer costs
  nothing here.
- **Do not** install torch on the meter (3–6 min of pip = pure waste, D5 §2.1 says build it on CPU
  into the Volume).
- **Open the follow-up now, off the meter:** the image needs D5 §7's additions
  (`/opt/torch-venv` with `torch==2.13.0+cu129`, `numpy`, `WUKONG_FA2_PYTHON`,
  `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` on the Volume). That is a **Phase 3 prerequisite** —
  `torch.compile`, not eager, is the framework bar (plan §0) — and it is owned by `tools/cloud/*`.
- **Publication law (Phase 3, not Phase 1):** any *published* round must run with
  `WUKONG_PEER_REQUIRED=1` (`baselines.rs:30-34`, D5 §0) or a mis-staged library silently publishes
  nothing while reporting green.

---

## 4. Failure playbook (what "stop" means in each case)

| Symptom | Meaning | Action |
|---|---|---|
| 4.1 `!! MISMATCH` in S0 | MIG slice, vGPU, or a different SKU | **Abort the session. Publish nothing** from this device (plan §6.1, risk #4). Retry once — a new invocation may land on a different host — then escalate |
| 4.2 `!! nvcc failed` | Toolkit unusable in the image | **Stop.** Peers cannot be honest. Fix the image locally, re-run S0 only |
| 4.3 Any suite failure in S2 | A genuine Linux/driver/portability/retarget finding — *the point of the phase* | Capture the full log → **end the session** → fix LOCALLY → `::build` (CPU) → re-run S2. **Never** `modal shell` to poke at it: that is the §0 "GPU as workbench" violation |
| 4.4 `device lost` / sticky fault | Every later test is a false failure | Kill the run immediately (Ctrl-C stops a non-detached `modal run`; `modal app stop <id>` as backup). The rest of the invocation buys nothing |
| 4.5 `Compiling wukong_*` in a GPU step | Profile mismatch or an edited mount — you are compiling at $0.80/hr | Ctrl-C. Re-run the matching `::build` on CPU. Re-run the step. See §0's profile law |
| 4.6 Step hits `WK_TIMEOUT` | The suite is longer than the cap | That **is** a result: record it, and it triggers plan risk #6 (a device-gate subset tag for cloud). Raise `WK_TIMEOUT` deliberately, once, with the new cap written into the log — never "just retry" |
| 4.7 Local terminal dies mid-run | Non-detached ⇒ the app stops with it (meter stops; work lost). Detached ⇒ it keeps running (work kept; meter runs to `WK_TIMEOUT`) | Runbook default: **not** detached for GPU steps (a broken pipe should stop the meter). Use `-d` only for the CPU builds. Either way, verify with `modal app list` and stop strays with `modal app stop` |

---

## 5. NEXT session preview — L40S SM-scaling (plan §5 Phase 1 step 3)

**Not scripted here on purpose.** Script it after the L4 session, when the suite wall time is known.

**What it answers.** L40S is `sm_89` — the *same ISA and the same ~100 KiB SMEM budget* as the dev
4050 — with **142 SMs vs 20**. It is the only rung on the ladder that changes **SM count alone**, so
every "does this scale past 20 SMs" verdict gets an answer with **no codegen variable** confounding
it. Caveat to state in the round: L2 also changes (plan §3 lists L40S ≈96 MB vs the 4050's probed
24 MiB), so any conclusion that involves an L2-resident regime is **not** SM-isolated — probe L2 in
S0 and say which side of that line each row falls on.

**What it does NOT answer:** anything keyed on *per-SM* occupancy limits. Ada's caps (24 blocks/SM,
`FLASH_WARPS=2` derived in-comment from them, `ptx_flash.rs:68`) are identical on L40S, so D2's
Hopper occupancy predictions (32 blocks/SM, 64 warps/SM) and D1/D3's 164/228 KB SMEM predictions are
untestable here. Do not let an L40S result be quoted as evidence for them.

**Predicted-vs-measured skeleton** (fill on the metal; a failed prediction is the useful outcome):

| # | Question | Prediction + source | Bench (`::bench --name …`) | Measured | Verdict |
|---|---|---|---|---|---|
| 1 | Do 256×128 CTA tiles still *lose* to the shipped 64×64 warp tile? | **Still lose.** D3 §(a): the "1 CTA/SM starves latency hiding" cause (`ptx_int8.rs:1259-1260`) is a **register-file** limit (64K regs/SM on cc 8.0 *and* 8.9) — architecture- and SM-count-invariant. More SMs buys no second CTA | `quant_int8_bigtile_sweep` | | |
| 2 | Does the 3-stage int8 ring flip from NEGATIVE to positive? | **Still negative on L40S.** D3 §(b) shows the 4050 verdict (`ptx_int8.rs:1286-1298`) is **100% an SMEM-capacity artifact**: at 16 KiB/stage, s2 = 32 KiB → 3 CTAs/SM but s3 = 48 KiB → 2 CTAs/SM on a ~100 KB Ada budget. L40S has that same budget, so s3 must still cost a third of the occupancy. It flips on **A100's 164 KB**, not here. A flip *here* would mean the cause was SM-count after all — surprising, and valuable | `quant_int8_w64_s3_sweep` | | |
| 3 | Do the f16 regime thresholds (L2-keyed) still pick the right kernel? | Thresholds derived from a 12/24 MiB L2 will mis-classify on ~96 MB: predict the large-shape regime engages too early | `gemm_pipe_sweep`, `gemm_cliff_ab` | | |
| 4 | Megakernel on 142 SMs | Grid `(1,1,1)` uses **1/142** of the machine (plan §2.3); predict the 4050's mega-vs-single wins shrink or invert | `mega_vs_single_gemm`, `mega_vs_single_reduce`, `mega_vs_chain_reduce` | | |
| 5 | **Wave quantization / split-K — the headline row.** At what square size does the grid stop filling 142 SMs? | D3 §(a)'s method, recomputed for **142 SMs**: one 128×128 CTA per SM needs `M·N ≥ 142·128² = 2.33 M` (**≈1525² square**); 3 CTAs/SM needs `≥ 6.98 M` (**≈2642²**). So predict **1024³ leaves ~2/3 of the machine idle** and split-K/stream-K — not bigger tiles — is the lever. This is the inverse of the plan §2.4 intuition, and the L40S is where it becomes cheap to test | `int8_splitk_occupancy`, `int4_splitk_occupancy`, `conv_splitk_vs_cudnn`, plus the 1024³ column of `gemm_pipe_sweep` | | |
| 6 | ws-flash S≥4096 crossover | Occupancy-per-SM is unchanged (Ada), so predict the crossover **moves with SM count only** — i.e. small-S flash wins shrink because 20 SMs were easy to fill and 142 are not (plan §2.7 item 3) | `flash_ws_vs_mp`, `flash_vs_gemm_scaling`, `flash_mp4_vs_mp` | | |
| 7 | `grid_stride_cfg` = 256 threads × 32·`sm_count` — the one correctly-scaling pattern (`ptx_optim.rs:217`) | Should scale cleanly to 142 SMs; if it does **not**, the SM probe or the launch math is wrong and every other row is suspect | `hbm_bandwidth` (also gives % of device peak) | | |
| 8 | Norm-kernel underfill: one warp per row, `grid=rows` (`ptx_norm.rs:36`) | Row counts < 142 leave SMs idle; predict a large relative regression vs the 4050 at small row counts | **No `#[ignore]`d norm sweep exists** (verified: 87 ignored benches, none norm/softmax). **Write it locally, for $0, BEFORE renting** | | |

Row 8 is the point of writing this preview early: it is a **$0 fix at home** and a wasted session if
discovered on the meter.

**Two things to fix at home BEFORE the L40S session:**
- Rows 1, 2, 3 and part of 5 are **peer-gated** sweeps (`quant_int8_bigtile_sweep`,
  `quant_int8_w64_s3_sweep`, `gemm_pipe_sweep`, `gemm_cliff_ab` and `conv_splitk_vs_cudnn` all call
  `peer_gate`; the `*_splitk_occupancy` pair does not). `::bench` cannot
  set `WUKONG_PEER_REQUIRED` (§8 finding 4), so any of them can *politely skip* and report green —
  on a $1.95/hr box, measuring nothing. Fix the harness, or read every `[skip]` line before believing
  a row.
- Write the norm-underfill bench (row 8). $0 at home; a whole rung of the ladder without it.

**Session shape (to script later):** S0 provenance (`WK_GPU=L40S`, expect `sm_89` / **142 SMs**,
`modal_app.py:221`) → `::build --release` on CPU → `::test --release` (confirm the suite is still
green at 142 SMs — cheap, and it validates the SM-derived launch math) → the named `::bench` rows
above, **one `--name` per invocation**, every A/B inside a single invocation (plan risk #7) →
teardown + ledger. Budget: L40S is $1.95/hr (`modal_app.py:36`); at ~1 GPU-hour that is ≈$2.
RunPod L40S at $0.99/hr is the cheaper alternative if the free credit is tight (plan §4.1).

---

## 6. Cost ledger (plan §6.6 — fill this in, every session)

Rates: L4 **$0.80/hr**, L40S **$1.95/hr**, A100-40GB $2.10, H100 $3.95 (`modal_app.py:33-43`,
plan §7). CPU ≈$0.05/core-hr (`modal_app.py:9`); `::build` requests `cpu=8.0` ⇒ ≈$0.40/hr.
Modal free credit: **$30/month, recurring** (plan §4.1) ⇒ ≈37 free L4-hours per cycle.

| Step | SKU | Wall (s) | GPU-hrs | Est. $ | Dashboard $ | Notes / what it bought |
|---|---|---|---|---|---|---|
| S0 device_info | L4 | | | | | provenance gate verdict |
| S1a build (debug) | CPU×8 | | — | | | Linux CPU gate + `check --features gpu` |
| S1b build (release) | CPU×8 | | — | | | artifacts for every metered step |
| S2 test pass 1 | L4 | | | | | **suite wall time — the deliverable** |
| S3a peers (fp8) | L4 | | | | | |
| S3b peer smoke cuBLAS | L4 | | | | | |
| S3c peer smoke cuDNN | L4 | | | | | |
| **Session total** | | | | | | vs. the $30 credit: |

Rules:
- **The dashboard is authoritative** (<https://modal.com/settings/usage>). Record the estimate *and*
  the actual, and note any discrepancy rather than silently trusting the arithmetic.
- Record the **running campaign total** against the plan's ~$54–108 envelope and the **$150 hard cap**
  (plan §7) at the bottom of every session summary.
- Phase-1 expected total: **≈$0.5–2.0** on L4 — comfortably inside one month's free credit.
  If a session's actual exceeds **$5**, stop and write down why before running anything else.

---

## 7. The CLI surface this runbook was written against

From `tools/cloud/modal_app.py` (do not edit it from here — another owner). Modal generates flags
from the Python signature: a `bool` becomes `--x/--no-x`, everything else `--x VALUE`
(`modal/cli/run.py:158-189`).

| Function | Signature (`modal_app.py`) | GPU? | Flags |
|---|---|---|---|
| `device_info` | `()` — `:236` | yes | none |
| `build` | `(release: bool = False, driver: bool = True)` — `:300` | **no**, `cpu=8.0` | `--release/--no-release`, `--driver/--no-driver` |
| `test` | `(peers: bool = False, release: bool = False, driver: bool = True, filter: str = "")` — `:330` | yes | `--peers/--no-peers`, `--release/--no-release`, `--driver/--no-driver`, `--filter VALUE` |
| `bench` | `(name: str = "", package: str = "wukong_codegen_gpu")` — `:361` | yes | `--name VALUE`, `--package VALUE` |
| `interactive` | `()` — `:378` | yes | `modal shell` target — **not used in this runbook** (§0 law) |

Environment read at import time: `WK_GPU` (`:59`, default `L40S` — **always set it explicitly**),
`WK_TIMEOUT` (`:63`, default 3600 s), `WK_CUDA_TAG` (`:64`).
Set by the harness inside the container: `CARGO_TARGET_DIR=/persist/target`,
`WUKONG_CUBIN_CACHE=/persist/cubin-cache` (`:137-141`), `WUKONG_GPU_REQUIRED=1` in `test`/`bench`
(`:339`, `:368`), `WUKONG_PEER_REQUIRED=1` in `test` only when `--peers` (`:341`).

Allowed control-plane commands (free): `modal --version`, `modal config …`, `modal volume list`,
`modal app list`, `modal app logs`, `modal app stop`. **Metered:** `modal run`, `modal shell`,
`modal deploy` — only the steps above, only in this order.

---

## 8. Known harness sharp edges (worked around above; fixes belong to `tools/cloud/*`)

1. **`device_info` exits 0 on a provenance MISMATCH** (`modal_app.py:280-287`) **and on an nvcc
   failure** (`:257-262`) — it prints and returns. A scripted `&&` chain would proceed to spend the
   metered suite on a MIG slice. *Workaround:* a human reads S0 before S2. *Fix:* `sys.exit(1)`.
2. **`::test` never passes `--nocapture`**, so libtest discards the output of passing tests — which
   silently defeats pass 1's stated purpose of *reporting* missing peers. *Workaround:*
   `--filter=--nocapture` (§S2). *Fix:* a `nocapture: bool = True` parameter.
3. **`lower.rs:4264-4267`** — `run_corpus_matches_interp_oracle` skips with a bare `eprintln!` +
   `return` when no device is reachable. Unlike `with_gpu` (`gpu.rs:6347-6357`) and its megakernel
   twin (`megakernel.rs:246-252`, which uses `diff::skip_or_fail`), **it does not escalate under
   `WUKONG_GPU_REQUIRED=1`** — and because libtest captures passing output, a *filtered* re-run of
   that gate on a device-less container reports a silent green. One-line fix: route it through
   `crate::diff::skip_or_fail`.
4. **`::bench` cannot set `WUKONG_PEER_REQUIRED`** (`modal_app.py:367-373`), so every peer sweep can
   politely skip. Tolerable in Phase 1 because `--nocapture` makes the skip visible; **must be fixed
   before any published round** (plan §6.4, D5 §0).
5. **`::bench` with no `--name` runs all 87 `#[ignore]`d sweeps** single-threaded — an unbounded
   metered run. Always pass `--name`.
6. **Phase 1's premise has moved.** The plan describes Phase 1 as testing the *pre-retarget* tree,
   but Phase 0+2 are **landed on main**: module headers now come from `ptx_target` at per-family
   floors (e.g. `ptx_norm.rs:28-33` emits the `sm_80` floor, not `sm_89`). Everything still loads on
   an `sm_89` L4, so the batch is unchanged — but a S2 failure may now be a **retarget** regression
   rather than a portability one, and this session is the **first silicon test of the floors**.
   Triage accordingly, and say so in the round summary.
