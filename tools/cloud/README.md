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

### ⚠ On Windows, `PYTHONIOENCODING=utf-8` is REQUIRED on every `modal run`

Not cosmetic — it decides whether a metered round produces a log at all.

Modal streams the container's stdout to your local terminal, and a Windows console is **cp1252**.
One non-ASCII byte in that stream kills the CLI:

```
'charmap' codec can't encode character '✓' in position 0: character maps to <undefined>
```

**The container keeps running and the meter keeps billing** — you have simply lost the output, the
provenance block and the results. On a CPU staging run that costs cents; measured live on
2026-08-09 it killed `::build_peers` at exit 1. On an H100 at $3.95/hr it costs the round.

This is reachable from several directions and will stay reachable: `modal_app.py` alone carries 112
non-ASCII lines, and the crate's own device gates print `✓` in their `[gate]` messages by design.
So the fix belongs at the call site, not in a promise to keep output ASCII:

```powershell
$env:PYTHONIOENCODING = "utf-8"        # once per shell
```

```sh
PYTHONIOENCODING=utf-8 modal run tools/cloud/modal_app.py::device_info    # or per-command
```

Treat it as the console-output sibling of the crate's **"PTX must be pure ASCII"** law: same class
of defect (one character, a hard failure far from its cause), different pipe.

### ⚠ Killing a `modal run` does NOT stop the container, and it keeps billing

The clause above — *"the container keeps running and the meter keeps billing"* — is not specific to
the encoding crash. It is true of **every** way the local CLI can die: Ctrl-C, a closed terminal, a
killed background job, a dropped network. `modal run` is a client watching a remote container; the
container's life is bounded by `WK_TIMEOUT`, not by yours.

Measured on 2026-08-10, in this repo's own session: two background `modal run` invocations were
killed from the driving session, and `modal app list` then showed an `ephemeral` app with a **live
task still charging at $0.506/hr** against the 7200 s ceiling — i.e. up to **$1.01 of nothing**, per
orphan, and there is no console message telling you it is there.

**So: after any interruption, check.**

```powershell
modal app list                                     # look for `ephemeral` apps with live tasks
modal app stop <app-id>                            # only after reading the paragraph below
```

**Stopping is not automatically the right move**, and this is the part worth internalising: *know
where the Volume commit sits in the entry point you interrupted.* Kill a run just before its commit
and you throw away work you have already paid for.

| Entry point | Where it commits | What an interruption costs |
|---|---|---|
| `::build` | **once, at the very end**, after the CPU workspace suite | kill it mid-suite and the *whole* `--features gpu` compile it already paid for is discarded. Let it finish. |
| `::build_peers` | after **each** artifact (CUTLASS, then vLLM, then at the end) | safe-ish: whatever already printed `[peers] ... ->` is on the Volume and a re-run skips it. |
| `::ptxas` | inside `flush()`, which runs **before every abort path** | safe: the round log, the JSON and the PTX archive survive any failure the function itself detects. |
| `::test` / `::bench` / `::peers` / `::cutlass` / `::marlin` / `::framework` | at the end of the function | the round's results are the stdout you are watching; the commit only persists caches. |

Modal's own background commits ("every few seconds", plus a final snapshot on container shutdown)
soften this — a *killed* container still flushes `target/`. The explicit `commit()` calls are what
make a **named artifact** (a profiler binary, a round log, a manifest entry) atomic and findable, and
those are the ones whose position above is worth knowing before you press Ctrl-C.

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
| `::ptxas` | **no** | The register / SMEM / spill census, **at every arch**. Resolves `ptxas`, self-tests each target, runs the crate's audit test behind a capture shim, then **recompiles every captured module for each requested arch**, prints the per-arch matrix and writes the raw `ptxas -v` into `/persist/rounds/`. |
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
modal run tools/cloud/modal_app.py::ptxas --archs sm_80,sm_90,sm_90a
modal shell tools/cloud/modal_app.py::interactive
```

## `::ptxas` — the one measurement that needs no GPU at all

`ptxas` compiles **for** an architecture; it does not need one. The `nvidia/cuda:*-devel` image
already ships it, and `::build_peers` proved the point on 2026-08-09 by building the CUTLASS
profiler for `sm90a` on a **CPU container** for $0.258.

That makes the campaign's most load-bearing unknown answerable for cents.
[`docs/gpu/derive/D1_h100_gemm.md`](../../docs/gpu/derive/D1_h100_gemm.md) §7 item 3:

> **Actual registers/thread ptxas allocates for A3/A4.** §2.3's estimates (~185) are derived from the
> generator's declarations, not from `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)`. If ptxas
> spills at 128 accumulators, **A3 collapses and A2 becomes the top lever.**

[`D2_h100_attention.md`](../../docs/gpu/derive/D2_h100_attention.md) §4.2 ranks the same census
**first of its five experiments** ("$0, NO GPU, runs in CI"), and
[`D3_a100.md`](../../docs/gpu/derive/D3_a100.md) §7 asks for it on `sm_80`. Every register number in
this campaign is currently derived from `.reg` declarations rather than measured — and NVIDIA's own
CUTLASS build emits `(C7511) ... wgmma.mma_async instructions are serialized due to insufficient
register resources` at 256x128x64, which is exactly the tile class the wide-tile work made
expressible.

```powershell
modal run tools/cloud/modal_app.py::ptxas                        # sm_80, sm_89, sm_90, sm_90a
modal run tools/cloud/modal_app.py::ptxas --archs sm_80,sm_90,sm_100
modal run tools/cloud/modal_app.py::ptxas --sweep-from /persist/ptx-archive/20260810-041627
modal run tools/cloud/modal_app.py::ptxas --require-archs
```

Five steps, in this order, and every one of them is designed so an empty answer cannot look like a
clean one:

1. **Resolve `ptxas` and record `--version`** in the round log. Not finding one is a hard failure —
   a `-devel` image must have it, and a `-runtime` value of `WK_CUDA_TAG` would not.
2. **Write and self-test a capture shim** (below). It must be byte-for-byte transparent *and* must
   demonstrably record; both are checked on a throwaway kernel before the census runs.
3. **Self-test every requested arch** on a minimal generated kernel, through the shim. This proves
   *this* ptxas accepts the `--gpu-name` **and** validates the output parser against *this* ptxas's
   `-v` format — so an empty census table can never be misread as "nothing spills". If *every* arch
   is rejected the probe kernel is blamed, not the toolkit; if only some are, that is a real finding
   and the run fails.
4. **Run the crate's audit test once**, `WUKONG_PTXAS` pointing at the shim, `--include-ignored
   --nocapture --test-threads=1`, then **recompile every module the shim captured at every requested
   arch**. Step 4b is the measurement the census was missing.
5. **Print the census, the per-arch matrix and the coverage report, and write everything raw** to
   `/persist/rounds/ptxas-*.log` (+ a parsed `.json`, + the captured PTX under
   `/persist/ptx-archive/<stamp>/`). The log is written **before** every abort path, so a failed
   audit still leaves its evidence on the Volume.

### What the 2026-08-10 round exposed, and what changed

The first census ran on a CPU container for **$0.025** and answered D1 §7.3 — 31 kernels, zero spill
stores, zero spill loads, zero stack, no `(C7511)`, including the four 254-register wide tiles and
all three wgmma configs. It also exited 1, with `no ptxas records parsed`, and both of its gaps came
from the same under-specified contract:

| Gap | Cause | Fix |
|---|---|---|
| The harness could not verify its own numbers | The contract pinned the *invocation* (`WUKONG_PTXAS` + a name filter) and left the *output format* open. The audit test prints its own already-parsed table, so there was no raw `ptxas -v` to parse. | `WUKONG_PTXAS` now points at a **capture shim**, so raw `-v` reaches the round log regardless of what the test prints. |
| Every number was measured at `sm_80` | The test compiles each module at the module's **own** declared `.target`, and treats `WUKONG_PTXAS_ARCH` as advisory. Asking for three arches changed nothing. | A **per-arch sweep** recompiles the captured PTX with `--gpu-name`, so `sm_90` is measured rather than assumed. |

The second gap is the one that mattered. **An H100 driver JITs `sm_80`-tagged PTX for `sm_90`**, and
register allocation is per-arch — so the H100 register cost of the 128x256 / 256x128 wide tiles, the
tiles whose entire value rests on fitting, had never been measured. Now it is a cell in a table.

### The contract with the crate, and why it is a shim

`::ptxas` still knows nothing about how the audit test is written. It touches it through:

| | |
|---|---|
| `WUKONG_PTXAS` | absolute path to an executable — which is why a **transparent wrapper** can go there. The knob already exists (`gpu.rs`'s `gemm_cliff_ptxas_ab`, `ptx_wgmma.rs`'s verification list). |
| `WUKONG_PTXAS_REQUIRED=1` | the crate's own knob turning "no ptxas, so I measured nothing and reported ok" into a failure. On a container whose whole purpose is that it *has* a ptxas, a skip is never the answer. |
| a libtest substring filter | default `ptxas`, i.e. **the audit test's name must contain `ptxas`**. Override with `--filter <substring>`. |

The shim runs the real `ptxas` with argv untouched, returns stdout/stderr/exit-status byte-for-byte
(the crate's own parse is unaffected and stays the crate's), and on the side writes **every raw `-v`
byte** and **the exact PTX the crate handed it**, keyed by content hash. It never writes to stdout or
stderr itself; capture failures go to `capture-errors.log` and are reported by the harness. Recording
must never change what is being recorded.

Two independent readings of the same run therefore exist — this harness's parse of raw `ptxas -v`,
and the crate's own printed table read back column-by-column — and where they overlap they are
**cross-checked**. A disagreement is a finding, not a tie-break: one of the two numbers this campaign
is about to size tiles with would be wrong.

`WUKONG_PTXAS_ARCH` / `WUKONG_PTXAS_ARCHS` are still exported and still **advisory**. The first round
proved this crate ignores them (three passes, three byte-identical tables, three times the money for
one fact), so the extra hint passes are now opt-in behind `--hint-passes`. Every reported row's arch
is read out of ptxas's own `Compiling entry function '<e>' for '<arch>'` line, never out of what was
asked for.

### The per-arch sweep

Four targets by default, because **register allocation is per-arch and that is the whole question**:

| | |
|---|---|
| `sm_80` | A100, and the floor every Ampere-legal module is tagged at |
| `sm_89` | the dev RTX 4050, L4, L40S |
| `sm_90` | **what an H100 driver actually JITs that `sm_80` PTX for** |
| `sm_90a` | Hopper's arch-specific target, and the only one `wgmma` is legal on |

Every (module, arch) pair is **attempted** — the harness predicts which ones should be refused
(an older-than-declared target; an `sm_90a`-only module at anything else) and prints the prediction
beside what ptxas actually did, but it never uses the prediction as a filter. "I did not try it
because I thought it would fail" is how a harness ends up reporting a conclusion it never measured.

The output is a matrix: one row per kernel, one column per arch, `-` where ptxas declined and `!`
where a cell spills, plus an explicit list of the entries whose register cost **moves** between
targets. That list is the reason the sweep exists.

`--sweep-from <dir>` re-runs the sweep alone over PTX a previous census archived under
`/persist/ptx-archive/<stamp>/`: adding `sm_100` to the question later costs a CPU minute, no cargo
build, and re-measures the *same bytes* a named round measured rather than whatever the tree says
today.

Failure modes made loud on purpose, because all of them would otherwise be green:

- **a filter that selects nothing** — libtest reports `0 passed; 0 failed` and exits 0. The run
  aborts instead, and names the test names it did find.
- **a shim that does not record** — caught by its own self-test, before the census.
- **an arch with nothing measured** — split into two lines that are *different facts*: `DECLINED`
  (every module refused it for a documented reason, e.g. `sm_90a`-only modules at `sm_80` — not a
  gap in the harness) and `MISSING` (nothing was measured and nothing explains why — always a
  failure). `--require-archs` escalates `DECLINED` too.
- **the two parsers disagreeing** — a failure, with the offending fields named.

Cost: CPU only, the same ~$0.51/hr container `::build` uses. It compiles the gpu feature if the
Volume has none (pass `--prebuilt` to refuse instead — there is no cost argument for refusing here
the way there is on a GPU box, only a provenance one). The sweep itself is 17 modules × 4 targets of
`ptxas`, i.e. a minute or two of the same CPU.

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

# 3b. The register/SMEM/spill census, at sm_80/sm_89/sm_90/sm_90a. Also CPU, also cents, and it
#     decides tile shape BEFORE anything is rented -- ptxas compiles for an arch, it does not
#     need one, and it will compile an sm_80-tagged module for sm_90 exactly as an H100 driver does.
modal run tools/cloud/modal_app.py::ptxas

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

### The vLLM venv (the Marlin/Machete int4 bar) — why `--copies` was the bug

The 2026-08-09 `::build_peers` run staged CUTLASS `sm90a` and then died staging vLLM:

```
+ /usr/local/bin/python3.11 -m venv --copies /persist/vllm-venv
Error: Command '['/persist/vllm-venv/bin/python3.11', '-m', 'ensurepip', '--upgrade',
'--default-pip']' returned non-zero exit status 127
```

That is a `CalledProcessError`, not an `OSError`: `venv._call_new_python` passes
`executable=os.path.realpath(...)`, so a missing or non-executable file would have raised
`FileNotFoundError`/`PermissionError`. The copied interpreter **started** and exited 127 — and 127 is
what the dynamic loader exits with when it cannot resolve a NEEDED shared object.

`/usr/local/bin/python3.11` is Modal's `add_python="3.11"`, a python-build-standalone
`install_only` CPython. Those are built `--enable-shared`: the binary NEEDs `libpython3.11.so.1.0`
and finds it through `RPATH=$ORIGIN/../lib`, where `$ORIGIN` is the realpath of the running image.
So a **symlinked** venv resolves to `/usr/local/lib` and loads; a **copied** one resolves to
`/persist/vllm-venv/lib`, finds nothing, and the loader exits 127 before `ensurepip` ever runs.

The `--copies` rationale — *"so nothing depends on a symlink surviving the Volume"* — does not
survive contact: a copied venv still records `home = /usr/local/bin` in `pyvenv.cfg` and still finds
its stdlib there, which is why the same comment already required the venv to be built from *this
image's* interpreter. `--copies` removed a symlink while leaving the image dependency it stood in
for, and paid for it with a hard failure. The symlink spelling depends on strictly less.

Since that is a diagnosis and not a measurement — it cannot be tested from Windows, and each attempt
costs the orchestrator — the staging path is a **verified fallback chain**. Each venv spelling is
created `--without-pip` and has pip bootstrapped separately afterwards; in every route, including
the venv-free one, the resulting interpreter is actually **executed** before it is accepted:

| # | spelling | why it is there |
|---|---|---|
| 1 | `venv` (symlinks) | correct for a shared-libpython standalone build; what every standard tool does |
| 2 | `venv --copies` + libpython copied into `venv/lib` and `venv/bin` | the `--copies` spelling with its actual defect repaired, for the case where a Volume cannot store symlinks |
| 3 | `/usr/bin/python3` (distro) with `--copies` | a distro build links its libpython from an absolute system path, so the whole `$ORIGIN` class disappears. Older (3.10 on 22.04), which is why it is not first. |
| 4 | **no venv at all**: `pip install --target /persist/vllm-pkgs` behind a two-line `/bin/sh` launcher that exports `PYTHONPATH` + `PIP_TARGET` and `exec`s the image interpreter | routes 1–3 are three *spellings of one mechanism* and share its failure modes. Route 4 removes the mechanism: no `pyvenv.cfg`, no relocated interpreter, no RPATH question, no symlink for the Volume to lose. Everything downstream still sees a working `<venv>/bin/python`, because the launcher **is** one. |

Splitting venv creation from pip installation is the other half of the fix: the original failure was
*reported* as an ensurepip error when the interpreter itself was what could not run. Now the round
log names which of the two broke. (Route 4 needs no bootstrap — the image interpreter already has
pip, and `PIP_TARGET` is what redirects it into the prefix.)

What route 4 costs is isolation — the launcher *adds* the prefix to the base interpreter's import
path instead of replacing it. That is acceptable here and only here: nothing installs into the
image's `python3.11` site-packages (torch lives in its own venv at `/opt/torch-venv`, which the
launcher does not touch), and vLLM's pinned torch cannot reach the `torch.compile` bar because that
bar is a different interpreter entirely.

`--vllm-mode <symlinks|copies|system|prefix>` pins one spelling instead of walking the chain. Use it
only once a round log has named the winner: skipping the verified chain to save a minute is exactly
how the `--copies` failure happened.

Every attempt's outcome — not just the winner's — is recorded in the Volume manifest
(`vllm.attempts`, or `vllm_staging_failed.attempts` when all four lose), and `_venv_evidence` dumps
the accepted spelling's `pyvenv.cfg` and `bin/` listing into the build log. None of this is testable
off Modal and every attempt costs the orchestrator, so the next fix has to be derivable from a round
log without re-deriving any of the above.

The winning spelling is recorded as `venv_mode`, `::device_info` prints it, and **both
`::build_peers` and `::marlin` verify the staged interpreter executes before trusting it** — "the
file exists" is exactly the check that would have called that broken venv staged. `::build_peers`
additionally `import vllm`s rather than only reading its metadata (a `--target` prefix can record a
distribution the launcher's `PYTHONPATH` does not actually reach; "installed" is not the claim, "the
int4 bar will run" is). `::marlin` re-checks in the first second of a metered call, so a bad venv
costs seconds, not the round.

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
