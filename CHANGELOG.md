# Changelog

All notable changes to Wukong are documented here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows semantic versioning.

## [Unreleased]

### GPU — the Hopper `wgmma` GEMM family gets a product call site
Until now the warpgroup-MMA + TMA kernel that the whole Act-2 campaign measures had **no caller
outside its own test module**: no `.wk` program could reach it, on any device. `--backend=gpu` now
routes a recognized `C = A·Bᵀ` through it on a Hopper part, and through the pre-Hopper launcher
everywhere else.

- The decision is a pure function of the *probed* `GpuTarget` (`gpu_accel::gemm_route_for`), and it
  is `cc_major == 9` rather than `>= (9,0)` because `sm_90a` is architecture-**locked** — a test pins
  the rule to `ptx_wgmma::Sm90aLicense` over the whole capability grid, so the two cannot drift.
- Config selection goes through **one** seam function that delegates to the generator's shipped
  regime rule; wave 3's per-shape dispatcher replaces exactly that body.
- **Decline, never wrong.** Architecture, shape, an unencodable tensor map (`K % 8 != 0`), an output
  past the epilogue's `u32` index, a ring larger than the device's opt-in shared memory, or an
  `UNSUPPORTED` from the generator all fall back to the existing path with unchanged results; only a
  real driver failure is an error. Ragged M/N/K are not declines. The launcher's `assert!`
  preconditions are pre-decided driver-side so an offload can never abort a user's program — and an
  **odd `N`**, which the launcher does *not* check, is declined here: both shipped rows use the
  `st.global.v2.f32` epilogue, whose pair is 8-byte aligned only when `N` is even, and a misaligned
  store leaves the CUDA context stickily errored for the rest of the process.
- **f16 is four decades narrower than f32, not only three digits shorter.** The route converts both
  operands with `half::f16::from_f32`, whose largest finite value is 65504, so a `.wk` GEMM over
  values past that would return `inf` where the pre-Hopper f32 launcher returns a number — a
  different *answer*, which no tolerance band covers and which no gate would have caught (every band
  gate seeds `U(-1,1)`). An operand outside the seam dtype's finite range is therefore a **decline**
  like every other, at the cost of one read-only pass over both operands, the same order as the
  host-side conversion that follows it. The underflow end is documented rather than declined: it is
  the arithmetic class every f16 tensor-core path here has carried since Act 1, and a rule strict
  enough to catch it would refuse nearly every real GEMM.
- On Hopper the route converts both operands to f16 on the host, so the plain-GEMM offload changes
  arithmetic class there — inside the existing CPU↔GPU *tolerance* contract, and the driver's device
  gate sizes its band from the route that **actually ran**. That is the per-route offload counters
  (`GpuAccel::gemm_wgmma_calls` / `gemm_existing_calls`, read by `gemm_route_taken`), not the
  device's capability: a Hopper part is eligible for wgmma at every shape while declining a real
  subset of them to the f32 launcher, and those results must keep the f32 band or a genuine
  regression hides under a band ~30× looser. `WUKONG_GPU_NO_WGMMA=1` forces the pre-Hopper path in
  the same binary, and `WUKONG_GPU_ROUTE_LOG=1` prints the route and decline reason per dispatch.
- The fused-epilogue hook (`sgemm_nt_epi`) is deliberately unchanged: no wgmma module computes
  `act(A·Bᵀ + bias)` yet, and splicing a second pointwise launch onto the plain GEMM would be the
  unfused chain wave 4 exists to delete. The seam and the exact kernel-side gap are named in place.
- **Everything up to the launch is checked on a laptop that cannot launch it.** The route, the
  selected config, every decline and the resulting `LaunchPlan` are pure functions of the shape and
  the probed capability, so the end-to-end Hopper gate is `#[ignore]`d while its *shape list* is
  asserted device-free: each shape must not decline under the H100 opt-in SMEM budget, its grid must
  be a multiple of its cluster on every axis, and exactly one must cross the generator's own
  `W1_CLUSTER_MIN_OUTPUT_ELEMS` — because the clustered row compiles `.reqnctapercluster` into the
  entry, and a gate whose shapes all fell on one side of a re-measured threshold would silently stop
  covering the arm with the extra launch mechanism while still passing.

### Documentation — every published claim is scoped to the device and the peer it was measured against
`GPU_RETARGET_PLAN.md` §10 asks for "no published claim anywhere in the repo that silently
generalizes a 4050 number to other hardware". This closes that for the doc set, and adds the CPU
side, which is currently the larger debt. **No measured figure was added, altered or removed** — the
measurement docs change only with a fresh same-run measurement behind them, and there is none here.
The only numeric edits anywhere in this change are two stale `tests/run` fixture counts, which are
tree facts rather than measurements and are verifiable with `ls tests/run/*.wk | wc -l`.

- `BENCHMARKS.md` opens with a **standing index**: for every family, whether its numbers *stand*,
  were *corrected* (with the struck value kept beside the correction, as this document's style
  requires), or are *not currently load-bearing*. It covers the CPU debts explicitly — the
  general-code tables **superseded with no replacement**, ~40 rows whose **Rust** column needs
  re-measuring after the 2026-08-06 `noalias` finding, a **C++** column that only ever existed in
  three sections, and the end-to-end model ratios as an **upper bound** pending re-measurement — as
  well as the GPU device scope.
- The **three GPU families that need more than a banner** are flagged where a reader meets them, not
  only in the index: *"beats PyTorch at every S"* is against **eager** and is **not re-earned**
  against `torch.compile`, which is native on Linux and now buildable and *verifiable*
  (`tools/cloud/peers/`); the **CUDA-graph** launch-overhead multiples are **Windows/WDDM** numbers
  that should shrink on the Linux driver before the GPU changes at all; and the long-context flash
  loss diagnosed as *"structural on 20 SMs"* rests on a premise that dissolves at 108/132 SMs, with
  the direction unknown — while the *short*-context occupancy **wins** rest on 20 SMs being easy to
  fill and may **invert**.
- *"No CUDA toolkit"* is reframed throughout as **toolkit-optional**: it says what Wukong needs to
  build and run, never that no stronger peer exists. The "no honest peer is buildable" framings
  (fused FA2-class CUDA-C, a library int4 W4A16 decode GEMM, vLLM/TRT-LLM) are marked as true of a
  toolkit-free Windows laptop and **expiring** on a cloud box.
- `README.md`, `docs/roadmap.md`, `docs/metrics.md` and `docs/internals.md` carry matching scope, and
  `docs/internals.md` gains the structural facts that moved with the retarget: per-family PTX header
  floors from one authority, probed device identity, the un-gated-module rule and `CRATE_SOURCES`,
  the megakernel's cooperative multi-CTA launch layer, and the host-vectorizer-invariant corpus
  floors. Two stale `tests/run` fixture counts (`docs/internals.md`, `docs/roadmap.md`) are corrected
  against `ls tests/run/*.wk | wc -l` and now say to count rather than trust the line.
- Repairs a heading this file lost: `d10d868` overwrote
  `### Benchmark instrument — a byte-identical control column, a bias fix, and one retraction`
  while inserting the section above it, leaving that section's body running on out of the GPU
  retarget's bullet list with no title.

### GPU — the datacenter codegen wave: wgmma + TMA, wide CTA tiles, a cooperative megakernel, GQA, SM-derived norms
Five independent generator/launch capabilities, each landed on its own branch behind the two-part
gate plus the device suite. None changes a language or CLI surface, and none is a performance claim
in this file — the measured evidence for each is in its commit body.

- **A Hopper `wgmma` + TMA GEMM family** (`ptx_wgmma.rs`, `tma_host.rs`) — `wgmma.mma_async` in the
  SS form, TMA-filled shared memory behind `mbarrier` transaction barriers, warp-specialised
  cooperative with a producer warpgroup at `setmaxnreg 32` and two consumers at 232. `sm_90a` is
  architecture-**locked**, not a floor: such a module loads on Hopper and nowhere else, in either
  direction, so the gate is a **type** — `wgmma_module` demands an `Sm90aLicense` that only a probed
  `GpuTarget` can mint, and a launcher that skips it fails to compile rather than merely failing a
  textual scan. This is also the crate's first `cuTensorMap` surface. **Nothing dispatches to it and
  nothing here is proven to run**: there is no Hopper part on this machine and `sm_90a` cannot be
  JITed, emulated or `ptxas`-checked locally. What *is* proven device-free is enumerated in the
  module's own `WGMMA_DEVICE_VALIDATION`, in the order it should be re-checked on first contact with
  silicon — starting with the shared-memory matrix descriptor's field naming, whose two offset
  fields were read from a figure rather than confirmed, and whose swizzle enum is numbered in the
  **opposite direction** from `CUtensorMapSwizzle`, so the obvious cast between them silently swaps
  the widest and narrowest modes.
- **The Act-1 CTA tile can finally be wider than 128** (`ptx_wmma.rs`): six `PIPE_DEEP_VARIANTS` rows
  at 128×256 and 256×128, at every depth an Ada opt-in carveout can hold, plus four
  `PIPE_WIDE_VARIANTS` rows budgeted against A100's 163 KiB so each loads on both datacenter
  targets. The generator was always tile-generic; what blocked it was that 128×256 costs 24 KiB per
  stage and every useful depth exceeds the PTX ISA's 48 KiB **static** `.shared` cap — so this is the
  dynamic-SMEM window from Phase 2 paying off. On this card all six new tiles are **bit-identical**
  to the shipped 2-stage kernel (`max_abs = 0`), not merely within tolerance, because CTA tiling
  changes neither the mma sequence nor the f32 accumulation order per output element. The four
  `WIDE` rows exceed this card's carveout and are proven textually only; they need an A100 or H100.
  Dispatch is deliberately unchanged — being able to express a tile is a separate decision from
  selecting one.
- **The megakernel has a cooperative multi-CTA launch path.** It launched a `(1,1,1)` grid through a
  plain launch, so "the whole program in one kernel" described one SM. Three new pieces:
  `grid_barrier_ptx` (a sense-reversing grid-wide counter with device-scope fences, its state a
  launch *parameter* so a kernel that faults mid-barrier cannot hang the next program), `plan_grid`
  (grid derived from `cuOccupancyMaxActiveBlocksPerMultiprocessor` × the probed SM count, declining
  when the request would not be co-resident — a cooperative grid that is not co-resident does not
  run slowly, it hangs), and `launch_mega` through `cuLaunchCooperativeKernel`. A kernel opts in via
  its **ABI**, read back out of the emitted PTX; half of either shape is a hard error, because the
  two halves fail in opposite and equally silent ways. `lower::emit_mega_ptx` still emits the
  block-scoped form, so every corpus program launches exactly as before.
- **Grouped-query geometry in the paged KV cache.** `KvConfig` had one head count and assumed
  `kv_heads == q_heads`, so for every Llama-class model the stack allocated `g` copies of a cache the
  model needs one of — which silently made every capacity, paging and throughput figure a
  measurement of a workload `g` times too large, and meant a vLLM/TRT-LLM-class comparison was not
  comparing the same thing. `KvConfig.heads` is now *defined* as the KV-head count (what every offset
  helper already meant by it) and the query count lives in a `GqaConfig` that enforces
  `q_heads % kv_heads == 0` behind private fields. The load-bearing gate is exact rather than
  tolerant: a grouped cache must give **bit-for-bit** what an ungrouped cache gives over the same K/V
  with each head replicated `g` times — which is what GQA *means*, and which a round-robin
  `h % kv_heads` mapping would fail while still passing a tolerance gate on random data. Every
  existing entry point keeps its signature and delegates at `GqaConfig::mha`, bit-for-bit the
  pre-GQA behaviour.
- **The norms take their launch geometry from the SM count.** `ptx_norm.rs` shipped one warp per row
  at `grid = (rows,1,1)`, and since the kernel reads `%ctaid.x` as *the* row index and never advances
  it, that was the only legal geometry: the row count alone decided both the CTA count and the
  per-row parallelism. `{softmax,layernorm,rmsnorm}_w{1,2,4,8}` add W-warp cooperation plus a
  grid-stride over rows, and `norm_launch` sizes both from a probed `sm_count`. Three findings the
  design did not predict are recorded at the site: baking W into the entry beats reading it from
  `%ntid.x` (a register induction step defeats ptxas's unroller and its immediate-offset folding, so
  the "more general" spelling started the whole family in the hole); W == 1 returns *exactly* today's
  launch uncapped, so wiring it in is provably a no-op for shapes it does not intend to change; and
  the covered regime is nearly tapped out on Ada — the occupancy headroom is real and the bandwidth
  headroom is not, against a written prediction that said otherwise. LayerNorm's variance stays the
  two-pass form, with a structural gate: widening a reduction is exactly the edit that tempts the
  one-pass `E[x²] − mean²`, which cancels in f32 and NaNs whole rows.

### GPU — flash and conv join the dynamic-SMEM window, and a 44 KiB gate that was declining nothing
Phase 2 landed the dynamic-SMEM path and the int8/fp8 families already routed through it; these two
did not.

- **`ptx_flash`** declared its K/V ring as a static `.shared` array with no budget parameter, so the
  whole family sat under the PTX ISA's 48 KiB cap on *statically declared* shared memory — a rule
  about the declaration, not a device fact, and identically 48 KiB on this Ada card's carveout, on
  A100's 164 and on H100's 228. The generators now take a `stages` count and an SMEM budget and
  select their emission form through `smem_mode_for`; at depth ≥ 3 the two-buffer swap becomes an
  add-and-wrap ring, with the `cp.async` commit/`wait_group` bookkeeping stated at the site as the
  correctness argument. Over budget is a loud panic at generation naming the kernel and the ceiling;
  a dispatcher's decline point is `FlashStageCfg::fits`, checked before generation, never a clamped
  launch. This is what unlocks the deeper rings, wider K/V staging and D=128 long-S configurations
  the datacenter derivations rank.
- **`ptx_conv`**'s shared-memory footprint was written twice — once in a `tiled_applies` gate a
  caller had to remember to call, once in the generator — and a desync between them was not a build
  error but an entry that declares more shared memory than the gate cleared, surfacing as an opaque
  driver error naming neither the family, the shape nor a budget. The implicit-GEMM and Winograd
  bgemm paths had no ceiling check at all. There is now one closed form per family, a real budget
  seam, and a generation-time panic that names both byte counts. The retired bound was a hardcoded
  `44 * 1024` whose comment read "no opt-in to the larger Ada banks" — untrue since the dynamic
  window landed, and never backed by a mechanism: the static boundary is the ISA's 48 KiB and the
  device boundary is the opt-in carveout, and the old gate conflated the two. **It was declining
  nothing**, and the commit says so rather than claiming a win.

### Measurement — the GPU instrument is code now, and the strong peers are buildable off the meter
This project has retracted four published claims and every one traced to an instrument that could not
resolve what it claimed. Before this, the GPU side's control column lived in a human's memory and its
provenance block lived in a PowerShell runbook.

- **`bench_instrument.rs`**: a `PtxTwin` control (one PTX text, two module handles, timed as two
  columns — it *refuses* equal module keys, because module caching would make two arms one
  `CUfunction` and every row would tie having compared nothing); a `prime` step that loads and
  warm-launches both handles before anything is timed, because byte-identical PTX means both arms
  share one on-disk cubin entry and cold-vs-warm is not a twin; a rotating multi-round driver that
  discards at least one warm-up; ratios over **medians**, with the round's floor taken as the largest
  control deviation over its live fields; and `Round`/`Published` as a **gate function rather than a
  convention** — a MIG slice, a vGPU profile, an SM count off spec or an unknown part cannot mint a
  number at all, and a bench whose floor exceeds the pre-registered bar yields `Unresolved`, never a
  tie. The provenance header is emitted by the harness, and a declared clock lock the readings
  contradict is printed as a contradiction rather than believed. Two findings the work produced
  rather than assumed are gated: **module-load latency has no in-process twin** (byte-identical PTX,
  a discarded warm-up, and the second arm still read tens of percent faster across two days and two
  drivers, because the driver keeps a JIT cache neither of ours controls — so `time_module_load` now
  refuses a second call per process), and a signed field with a non-positive median is `Degenerate`,
  not `Constant`, so the log names the right cause. A follow-up put the file under the crate's
  source-scanning laws, which it had been invisible to.
- **The strong peers are buildable, pinned, and verified — from a CPU container.** `nvcc` compiles
  *for* an architecture; it does not need one, so the multi-GB, multi-hour artifacts stage onto a
  volume off the meter: the CUTLASS profiler (filtered kernel set — an unfiltered SM90 build is a
  multi-hour, >100 GB mistake by NVIDIA's own documentation), a vLLM venv carrying Marlin and
  Machete, optional FlashAttention wheels, and a torch venv with Triton pinned and *asserted at image
  build time*. `torch_compile_peer.py` reports eager, `compile(default)` and `compile(max-autotune)`
  side by side and takes the **fastest** as the peer, with an `eager_over_peer` ratio so the round
  itself measures how much of the retracted margin was the peer being weak. A missing declared peer
  is a **failure, not a shrug**: `verify_peers.py` exits non-zero, an unknown peer name is an error
  rather than a token that quietly requires nothing, and `smoke_inductor.py` asserts a Triton kernel
  was actually generated — a `torch.compile` that fell back to ATen is cuBLAS wearing the framework
  bar's name. Every version is pinned in one block and recorded to a manifest, because this repo has
  already been bitten by a floating peer.

### Fixed — two gates that were measuring the host instead of the thing under test
- **`wukong_runtime` is now verified in release, the profile its contract ships in.** No gate in this
  repo had ever run `cargo test --release`, so the crate whose first invariant is that every AVX2
  kernel agrees **bit-for-bit** with its scalar twin was only ever checked in the profile where LLVM
  does the least — while every performance number the project publishes comes from a release binary.
  That is backwards, and it was not hypothetical: a `colreduce` NaN-payload divergence reproduced
  byte-identically on Windows and Linux in release and passed in debug, because it needs the
  optimizer to have auto-vectorized the *scalar twin* before it can exist at all. The assertion
  itself was the bug — it pinned a NaN payload IEEE-754 never promised, and the `-O3` asm shows the
  scalar twin's auto-vectorized body loading `x[i][j]` into the destination register (i.e. commuting
  the `fadd`, which is legal), with the divergence boundary landing exactly where the vector body
  ends and the scalar epilogue begins. The gate is scoped to this one crate deliberately: it is the
  only one that asserts raw-bit agreement between two independent implementations of the same
  arithmetic.
- **The gpu-native corpus floors measure the backend, not the host CPU.** Both floors were Windows
  numbers, and the same two sweeps read higher on a Linux L4 from the same commit — not because the
  backend lowered more, but because the general 256-bit AVX2 vectorizer is Win64-only
  (`host_supports_vec_kernels`), `lower.rs` declines the op it produces, and fewer such ops means
  more programs lower. So one hardcoded floor could only ever ratchet one of two readings, and on
  Linux it passed with enough slack to never catch a regression — on precisely the fleet the
  datacenter round is paid to run on. The fix normalizes the *input*: both sweeps build through the
  documented, result-identical 128-bit CLIF fallback, so one constant is live on both operating
  systems and Windows gains real device coverage rather than the floor being relaxed. A device-free
  gate proves the normalization is effective (non-vacuously — the host-native re-sweep must find a
  carrier) and complete (it reproduces the old numbers exactly, every gained program is a carrier,
  none are lost), so a lowering regression is now catchable on a box with no card.
- **The interpreter's allocation arena is capped explicitly.** `try_reserve` guards the *reserve*,
  not the *commit*: under Linux overcommit an absurd request succeeds and the subsequent resize gets
  the process OOM-killed. Cap before asking the allocator.

### Infrastructure — the Phase-1 cloud harness, and a clippy sweep that turned CI green on all three jobs
- **`tools/cloud/`** (a Modal app plus a `nvidia/cuda:*-devel` image) is the entry point for every
  rented-GPU session: build, device suite, corpus gates, benches and peer staging, each a named
  command so a metered session never pays for a decision. Written and reviewed **before** any GPU was
  rented — with the round's expectations pre-registered in `docs/gpu/derive/` so a result cannot be
  rationalized after the fact — and hardened against five ways the harness itself would have burned
  metered time (including one where the runbook's own logging aborted every run). Session ledgers and
  raw round logs are committed under `bench/gpu/` as artifacts, so no fact is paid for twice.
- **A workspace-wide clippy sweep under `-D warnings`** closed a month of unread red on the ubuntu
  job. The red was hiding four real Linux defects, not lint noise: `--backend=native` was broken for
  any program with a vectorizable f32 loop, the `--emit=exe` gate was dark, the `cc` fallback could
  not link, and one test was vacuous — the first three are the two sections below and the fourth is
  fixed with them. Two process notes worth keeping: CI installs a **newer** stable rustc than this
  box, so a locally clean sweep can still turn all three jobs red; and `clippy --fix` broke a
  bit-exactness contract in one hunk, so every hunk of an automated fix needs reading.

### Fixed — `--emit=exe` takes its preferred link path on any cargo-built compiler, and the `cc` fallback links libm
Two independent link-path defects, both of which made `--emit=exe` quietly worse than it looks. They
were found together because the AOT gate (`crates/wukongc/tests/exe.rs`) reported them as a single
red on ubuntu CI. Neither changes what any program *computes*.

- **The runtime rlib is now located wherever cargo actually put it.** `rustc_link` looked only for
  `libwukong_runtime.rlib` next to the compiler binary — and cargo *uplifts* a library there only
  when it is a root unit of a **`build`**. Under `cargo test --workspace` the package's root unit is
  its test binary and the plain rlib stays hash-suffixed in `target/<profile>/deps/`; `cargo run -p
  wukongc` does not uplift it either, because `-p` makes `wukong_runtime` a dependency rather than a
  selected member. So on any tree that had not *also* had a plain `cargo build` run in it — every
  clean checkout, including CI's — `--emit=exe` silently degraded to the `cc` fallback, whose output
  is knowingly *not* byte-identical to the oracle (no `wukong_rt_print_str`, no `wukong_*` kernels,
  under-precise `printf("%g")` floats). This was invisible locally purely because the working tree
  had a leftover uplifted rlib. The lookup is now the single function
  `wukong_driver::runtime_rlib_in` (uplifted copy first, then the newest
  `deps/libwukong_runtime-*.rlib`), which `tests/exe.rs` calls too, so the gate's probe cannot drift
  from the driver's decision — it had drifted exactly that way, leaving the AOT gate dark: on ubuntu
  every fixture skipped and the "compared ZERO linked executables" assertion fired. Reproducing that
  artifact layout on Windows (a scratch `CARGO_TARGET_DIR`) shows the same darkness passing quietly
  there: three of four fixtures skip and only the scalar `heap_alloc.wk` links, through MinGW `cc`,
  which is enough to keep the count non-zero while proving almost nothing. Both layouts now link all
  four through the preferred path.
- **The `cc` fallback passes `-lm`.** The generated C runtime unconditionally defines
  `wukong_rt_fmod_f64`/`_f32` over `fmod`/`fmodf`, which glibc keeps in `libm` and `cc` does not link
  by default — so the fallback could not link a *single* program on Linux, regardless of whether it
  used `%` on floats (`undefined reference to 'fmod'`). The flag goes last, after the objects that
  reference it, and is a no-op where libm is folded into the C library (macOS, MinGW).

### Fixed — `--backend=native` builds again on every non-Windows host
The general vectorizer built a 256-bit raw-AVX2 `VecKernel` recipe whenever the body qualified, with
**no host check** — but `avx2::assemble_kernel` refuses on any host that is not x86-64 Windows with
runtime AVX2 + FMA3, because the emitted bytes hardcode the Win64 argument registers (rcx/rdx/r8) and
carry no in-kernel dispatch. `populate_module` propagates that `Err`, so on Linux/macOS *the whole
compile died* — `avx2 assemble vec kernel 0.0: avx2: host lacks AVX2+FMA3 under the Win64 ABI these
kernels encode` — for any program with a vectorizable f32 loop. The GitHub Actions ubuntu job had
been red on ~20 `wukong_codegen_cranelift` tests since 2026-07-07.

- The host predicate is now **one function**, `wukong_mir::host_supports_vec_kernels`, placed below
  both users. `avx2::host_supports_kernels` is a thin wrapper over it (name and behaviour unchanged),
  and both `wukong_mir_build` recipe sites — elementwise (`emit_vectorized_for`) and reduction
  (`emit_reduction`) — test it in the same condition as `WUKONG_P4_NO_256`. Off-gate the loop keeps
  the 128-bit CLIF strips: the documented, result-identical fallback the kill-switch already forces
  (`p4_kill_switch_is_result_identical`). The Win64/AVX2 requirement itself is **not** weakened, and
  `assemble_kernel`'s refusal stays as a backstop.
- The tests that assert a recipe was actually built are host-gated to match, since off-gate they are
  false by construction rather than failing meaningfully: `p4_vec256_general_matches_interp`,
  `p4_kill_switch_is_result_identical`, `p4_reduction256_gate_and_differential`, the non-vacuity
  anchor inside `p4_counting_while_normalizes_and_matches_interp` (its differential half still runs
  everywhere, since while-normalization is width-independent), and the two `#[ignore]`d 256-vs-128
  benches, which would otherwise print a 128-vs-128 ratio as if widening bought nothing. The pure
  equivalence sweeps (`p4_vec256_coverage_sweep`, `p4_reduction_f64_reference`) stay host-agnostic
  and sweep the 128-bit strips there instead.
- `wukongc`'s `transformer_block_tensor_dispatches_the_same_kernels` counted `veckernel #` in
  `--emit=mir` output — which never reaches codegen, so it had been passing on Linux over a recipe
  that could not have been assembled. It now accepts the 128-bit `<4 x f32>` form off-gate only.

### GPU retarget, Phase 2 — the backend stops being fused to one laptop card
The datacenter retarget's local phase (`GPU_RETARGET_PLAN.md`), landed as ~20 gated merges. No
language or CLI surface changed; every measured claim in `BENCHMARKS.md` is untouched (the perf
identity leg of the closing regression is deferred until the machine is on AC power — no number
here is a performance claim).

- **PTX module headers come from one authority** (`wukong_codegen_gpu::ptx_target`): 67 hardcoded
  `.target sm_89` sites across 18 files now emit per-family floors — `sm_80` for the Ampere-legal
  majority, `sm_89` only where `e4m3`/`e5m2` genuinely exist (fp8). `.version` obeys the same
  lowest-legal rule (7.8 everywhere but fp8's 8.4) so no module demands a newer driver than its
  instructions need. Every family carries a floor gate; an adversarial pass mutation-tested them.
- **Device identity is probed, not assumed**: `Gpu::target()` (cc, SMs, opt-in SMEM, L2, VRAM,
  driver) replaces the silent `.unwrap_or(20)` era. The probe settled a two-year comment
  contradiction: this 4050's L2 reads 24 MiB, not 12. The f16 GEMM regime thresholds now derive
  from the probed L2 with a unit test pinning byte-identical 4050 dispatch.
- **fp8 is capability-gated before any module load** (`cc >= 8.9`, a loud decline distinct from
  no-device/no-peer), enforced by a crate-wide, PTX-symbol-keyed law test that fails on any new
  ungated fp8 load path. NVRTC peers now compile for the probed device; PTX modules tag the floor —
  opposite directions, both documented.
- **Caches are device-keyed**: cubin SASS by `sm_XX` (cross-arch cubin loads can succeed silently
  under forward-minor compatibility, so the key closes what the loader cannot), autotune by
  arch+SM-count. Pre-change caches read as clean misses.
- **Dynamic shared memory works end-to-end** (`.extern .shared` window + `cuFuncSetAttribute`),
  unlocking the >48 KiB budgets every non-`a` target forbids statically: deep-stage pipeline
  variants now exist for int8 (s2–s5, plus 256×128/128×256 at s3), f16 (s2–s5, 128×256 s3) and
  fp8 (to 90 KiB), each proven bit-identical to its shipped 2-stage kernel on device and declined
  automatically where a card's budget cannot hold it.
- **`Op::Iota` lowers to PTX** — a silent hole that had cost `--backend=gpu-native` ~10% of the
  run corpus (183→217/357 oracle-matched; megakernel 69→87/103); coverage floors are now ratcheted
  in both corpus gates so the next silent decline goes red, and megakernel declines print their
  reason under `WUKONG_GPU_DUMP_PTX`.
- Linux portability arms (per-OS driver-library names, peer discovery, venv paths, exe suffix), a
  CI job that finally type-checks the `gpu` feature, and six derivation dossiers committed under
  `docs/gpu/derive/` for the rented-silicon phases. `docs/internals.md` and this changelog were
  de-duplicated from concatenated copies (with content-loss proofs).

### Benchmark instrument — a byte-identical control column, a bias fix, and one retraction
`wukong_xbench`'s opt-in `general` suite only. No kernel, no peer source and no compiler crate is
touched — `wukong_xbench` is a leaf binary crate nothing depends on, so `wukongc` is byte-identical
and no program's dispatch set can move. That makes the dispatch census a **non-regression argument
and nothing more**; it is not independent evidence for anything below.

- **A `C(twin)` CONTROL COLUMN — every run now measures its own noise floor.** The identical C
  source, through the identical compiler, at the identical flags, into a second DLL. Its ratio
  against `C` has expected value exactly 1.00 and contains no language content, so what it reads away
  from 1.00 is the *instrument*. In one validation run it read C **75.20 ms** against C(twin)
  **57.60 ms**: the previous estimator would have published **a byte-identical binary as "1.31×
  faster than C"**. Every cross-language cell is now classified against it — a per-round range that
  overlaps what the control did prints `BELOW FLOOR` and carries **no number**, however tightly its
  rounds agreed, because a tight spread around a tiny effect is exactly what the control exists to
  refuse. The floor is printed as a figure (`NOISE FLOOR, MEASURED THIS RUN: 1.03x`), so a run that
  resolves nothing has still reported its own resolution rather than leaving blanks. When the control
  fails its own spread gate the section says so and claims no size.
- **A real positional bias, fixed.** `bench_structure_tax` timed all five Wukong spellings to
  completion and only then built and timed the peers, one language after another, each exactly once,
  in a fixed order — so C always held the first and coolest peer slot and Rust always held the last,
  and the Wukong columns always got the earliest slots, in Wukong's own benchmark. (The loss and scan
  sections had been given a forward/reverse two-pass at `17ab8d0`; this one had not.) Every column of
  every section is now **built first**, then timed **once per round** in a rotating,
  direction-alternating order whose position balance is *exact* over each pair of rounds, and each
  ratio is formed **inside** a round and reported as the median with the `[min-max]` the rounds
  spanned — never as the quotient of two minima drawn from different instants of the machine's clock.
- **The timing thread is pinned to one core, and that is the largest confound this suite ever had.**
  This laptop is a hybrid Core Ultra 7 155H (6 P-cores, 8 E-cores, 2 LP E-cores, three performance
  classes) and every program in the section is single-threaded, so Windows walks a long timing thread
  down the classes mid-run. Measured, same binary, kill switch only (`XBENCH_PIN=off`): unpinned, the
  C column degraded **81 → 174 ms monotonically inside one run** while the 4–5 ms Wukong columns in
  the same rounds did not move, and the zero-difference control read a **2.12× spread**; pinned, the
  same section held 75 → 71 ms and the control read ~1.05×. Applied identically to every column, so
  it carries no language content — and the control is what validates it rather than an assertion. A
  `PinGuard` restores the previous affinity on the way out, because a leaked one-core mask would cap
  every later multicore row and read as a performance finding. A pinned figure is single-core only.
- **RETRACTED: the "run-to-run variance fell from median 1.451× to 1.142×" claim.** It was made for
  the round protocol on branch `bench/noise-resistant-ratios` and **it does not replicate.** An
  adversarial re-run with both binaries interleaved measured the old protocol at median 1.195× /
  worst 2.876× against the new at **1.270× / 1.869×** — the median got *worse*; only the tail
  improved. What this work establishes is a **bias fix** and a **noise-floor control**, plus the
  scheduler finding above. It is not a median variance reduction, and no doc or comment states one.
- **The per-visit budget is back to 2 warm-ups + best-of-7** (it had been cut to 1 + best-of-3), and
  the default round count is cut 6 → 4 to pay for it. The reason is measured and is *not* the one the
  cut was blamed for: on a pinned machine the budget does not move the published ratio's spread at
  all (1.049×/1.094× restored against 1.041×/1.092× cut), but it does tighten **the run's own noise
  floor** — the control's per-round spread landed inside the 1.10 limit in 7 of 9 sections against 2
  of 9 — and the floor is the gate that decides what may be printed at all.
- **The `ms` column's LEVEL is protocol-dependent, and the table says so where the numbers are.**
  Pinned, same machine, same day, the block section's C peer read 92–103 ms under the old protocol
  and 105–155 ms under this one. It is not the extra calls (a run cut to 8 calls per column — *fewer*
  than the old protocol's 9 — sits in the same band); it is that this protocol builds every column
  before timing any, where the old one compiled and timed each peer in turn and the core re-boosted
  across each `gcc` pause. The two sections that already built first show no offset (1.01×). Read
  `ms` only against other `ms` from the same run.
- **The power-change detector separates STATE from DRAIN instead of losing one of them.** A state
  change (battery / AC+charging / AC+full — three different machines) replaces every ratio in the
  section with `NON-REPORTABLE` **in the cell**, not with a warning line a reader can scroll past.
  Battery drain while the state holds no longer voids the section, but it is still sampled every
  round and **printed** (`! BATTERY MOVED 71% -> 70% inside this section`), so stripping the
  percentage from the invalidation key is not a loss of sensitivity.
- **The per-round AVX2-FMA clock probe is off by default** (`XBENCH_CLOCK_PROBE=1`); it used to run
  every round unconditionally. It was measured to perturb what it observes — a saturating burn loop
  immediately before a timed round pre-boosts the core (block C 120/115 ms with it on against
  145/149 ms off) and *widens* the zero-difference control (1.112×/1.416× on against 1.095×/1.152×
  off). Using it as a *normalizer* had already been evaluated and rejected on
  `bench/noise-resistant-ratios` (carried over, not re-measured here): on the published ratio it
  cancels exactly, and on the `ms` column it makes the spread strictly worse. It survives only as an
  opt-in printed drift trace; the honest always-on one is the raw per-round table.
- **`BENCHMARKS.md`'s general-code tables are flagged as superseded** — they predate both the
  recognizer widenings above (which change the *census* column outright) and this instrument work.
  No replacement figures are published: every run available was battery or AC+charging.

What it adds up to, measured in the state the adversarial verifier used. Six `general` runs,
2026-08-06, AC+charging 68%→72%, the predecessor binary and this one **interleaved** three each
(ship/pred/pred/ship/ship/pred), power checked before and after every run. Cells resolved to a size:
**0 of 57 → 19 of 57**. Sections whose control failed: **9 of 9 → 3 of 9**. Control per-round spread:
median **1.702× → 1.049×**. Run-to-run spread of the published `vs C` ratio: median **1.276× →
1.056×** — and the 1.276× replicates the verifier's independently measured 1.270× for that protocol,
which is the best evidence available that the comparison itself is sound. It is a three-change A/B
(pin, per-visit budget, round count); the battery A/Bs above attribute the ratio spread mostly to the
pin and the control spread mostly to the budget. **No cross-language figure is published from any of
it** — every number here is about the instrument.

### Loop vectorizer: a lane ramp for a counter read as a value
- **`wukong_opt::vectorize` no longer declines a loop whose body reads its induction variable as a
  *value*.** It used to say `the induction variable is used for something other than addressing` and
  leave the loop scalar, which is not an exotic shape — it is what a classification loss is
  (`if c == target { qt } else { qo }`). The decline was correct rather than merely conservative: the
  counter is uniform across a group only as a `gep` index, and splatting it writes the group's FIRST
  counter into all `W` lanes, a silent wrong answer on both backends.
  The widened body now reads it through the **lane ramp** `splat(i) + iota`, so lane `k` sees the
  counter scalar iteration `i + k` had, and an integer compare against it becomes a per-lane **mask**
  the existing `select` path blends. Over `tests/run` + `examples`, `--emit=mir -O2` goes from 531 to
  619 `WIDEN` decisions and that decline drops from 221 (file, loop) pairs to 64.
  Deliberately narrow: only the counter itself gets a ramp (a derived `i + 1` or `sext i` read as a
  value is still declined), the ramp's lane count must equal the group width the loop's data runs at
  (an `i64` counter over `f32` data is declined), a store's pointer stays an address while its value
  may be the ramp (`o[i] = i` widens), and the two `select`s if-conversion synthesizes keep their
  flat rejection.
- **New MIR op `Op::Iota(MirType)`** — the lane-index ramp `<0, 1, .., n-1>`, no operands, pure and
  constant. It exists because no expression over splatted scalars is lane-varying, so a non-uniform
  vector needs one non-uniform source and MIR had none. The interpreter materializes the lanes, the
  Cranelift backend emits one `vconst`, the textual-LLVM emitter writes the vector literal, and the
  GPU MIR→PTX path declines it as `UNSUPPORTED` exactly as it already declines `Splat`/`ExtractLane`.
  LICM hoists it to the preheader and CSE numbers it by type, so it costs nothing per iteration.
- **`WUKONG_NO_IV_RAMP=1`** restores the old decline, so the "what did the ramp buy" A/B is
  same-binary and same-run (the reason `WUKONG_NO_VECTORIZE` exists). It is strictly a narrowing
  switch: every loop it turns off is one the pass declined before the ramp existed.
- **New gate** `tests/run/vectorized_iv_as_value.wk` prints every lane of six kernels — one-hot
  select with the target in the first group, the last full group, the scalar epilogue and nowhere;
  `sitofp` of the ramp; `o[i] = i`; a lane mask in front of a float reduction; a counter starting at
  3; and the same one-hot at `<2 x f64>`/`i64` lanes — with expected values derived by hand from the
  scalar semantics.
### Recognizers: the natural spelling of a transformer block now dispatches
Four widenings, all in `wukong_mir_build`. Each closes a case where a program that computes exactly
what a recognized one computes lost its kernel to *spelling* alone. Nothing was narrowed: the
path-keyed dispatch census over `tests/run` + `examples` is 0 LOST throughout, 835 → 848 sites.

- **An activation written INTO the matmul store.** `c[i*N+j] = silu(bias[j] + s)` — the way an FFN is
  actually written — used to decline the whole GEMM, because the store value is a `Call` and neither
  the α peel (which wants a `Mul`) nor the bias peel (which wants an `Add`) matched. `peel_epi_act`
  now peels `fmax(_,0)` / `gelu` / `silu` off the store first and `MatmulNest` carries the code, so it
  fuses into one `wukong_sgemm_nt_epi` — **one** kernel call and one pass over C, where the
  separate-loop spelling needs a GEMM plus a `wukong_vmath_f32`.
- **A residual read from another buffer, folded into the store.** `c[i*N+j] = x[i*N+j] + s` with
  `x != c` matched neither the plain matmul (`c = s`) nor the in-place residual (`c = c + s`). It now
  lowers to `wukong_sgemm_nt` + a flat `wukong_velem_f32` sweep — the same pair, and the same bits, as
  the factored `for i { c[i] = x[i] + c[i]; }` spelling.
- **A dual-store projection.** `c[i*N+j] = s; d[i*N+j] = d[i*N+j] * s` in one body — the SwiGLU
  up-projection multiplied into its gate — made the inner body four statements where every matmul
  matcher wants three. It now lowers to `wukong_sgemm_nt` + `wukong_velem_f32` (Hadamard).
- **An accumulator seed declared early** (`wukong_mir_build::canon`). Several recognizers match a
  window of consecutive statements by position; putting `let mut s = 0.0;` one statement above where
  the canonical softmax puts it split the window and dispatched nothing. A `let mut x = <literal>;` is
  now sunk to the first statement that *assigns* it. Restricted to `mut`-and-assigned deliberately:
  sinking an immutable constant lands it between two statements and cost `i8_linear_dequant.wk` its
  fused `wukong_i8gemm_nt_deq`, which only the census saw.

Every one of the four declines rather than guesses on any shape its kernel cannot represent (an α
alongside a fused activation, a non-NT or transposed-A nest, a batch offset, or a second output
buffer overlapping a GEMM operand), and the two-kernel forms resolve every pointer and dimension
before emitting the GEMM so a late bail can never leave a half-lowered nest.

Effect on `wukong-xbench general`'s structure-tax benchmark — one transformer block written five
arithmetically identical ways. The monolithic natural spelling went from **7 dispatched call sites to
15**: `wukong_norm_affine_f32` ×2, `wukong_norm_f32`, `wukong_sgemm_nt` ×7, `wukong_sgemm_nt_alpha`,
`wukong_sgemm_nt_epi`, `wukong_velem_f32` ×3 — the recognizer-dialect spelling's 16 minus the one
`wukong_vmath_f32` it no longer needs, since its activation rides the GEMM epilogue. No scalar GEMM
remains in it, and it is now bit-identical to the other four spellings. Program 0's two fragility
probes, `epilogue: in the store` and `residual: in the store`, went from dispatching NOTHING to
dispatching the same kernels as their own-loop twins. Timings from that suite are **not** reported
here: the machine was on battery and changed power state mid-run.

### Performance — the vmath exp/log loss vs oneMKL VML was the dispatch, not the polynomial (2026-08-06)
The last documented single-core loss to an industry library ("exp 1.23–1.45× and log ~1.25× slower
than VML; residual is algorithmic — VML has a cheaper ~0.5-ULP core") was **not** algorithmic. The
AVX2 elementwise dispatch selected its 8-lane kernel once with `vmath8_for(op)` and then called that
**function pointer** every 8 lanes. That pointer comes from a runtime 36-arm `match`, so LLVM cannot
fold it, and the crate is built without crate-wide AVX — so a 256-bit vector cannot cross a real call
boundary in a register and goes through memory. The emitted inner loop (reproduced standalone with
`rustc -O --emit asm` on a faithful 4-arm copy of the dispatch shape; a 1-arm copy devirtualizes and
proves nothing) was: load 8 lanes, `vmovaps %ymm0, 64(%rsp)` to spill the argument, load the hidden
return-slot and argument pointers into `rcx`/`rdx`, `vzeroupper`, `callq *%r12`, `vmovaps 32(%rsp),
%ymm0` to reload — and, because the callee owns its registers, the polynomial's `_mm256_set1_ps`
constants had to be rematerialized on every vector instead of living in ymm across the loop.
Four *arithmetic* attempts had already been spent on this gap (8-bucket in-register LUTs, Estrin
scheduling, the ×4/×6 ILP unroll, and a bit-identical `ldexp` exp tail that measured a 10–15% loss
and was reverted); none of them could see it, because none of them was in the loop that was slow.
- **The op dispatch is now monomorphic** in both the one-input (`wukong_vmath_f32`) and two-input
  (`wukong_vmath2_f32`) elementwise kernels: one `match` outside the loop, each arm instantiating the
  shared loop body with its kernel inlined, and both the arm list and the `vmath8_for` /
  `vmath2_8_for` tables generated from a single `with_vmath8_table!` / `with_vmath2_8_table!` so they
  cannot desync. `WUKONG_VMATH_FNPTR=1` restores the old spelling in the same binary as the A/B
  kill-switch.
- **Nothing about the arithmetic changed**, and that is proven rather than asserted:
  `vmath_exp_log_exhaustive_ulp` runs **all 2³² f32 bit patterns** through exp and log and requires
  monomorphized lane == function-pointer lane == scalar twin, bit-for-bit — NaNs of every payload,
  ±0, ±∞, subnormals and the negative-garbage bands included. The two per-kernel gates
  (`vmath{,2}_monomorphized_dispatch_matches_the_function_pointer_path`) cover every declared op code
  at every length class and in both store regimes.
- **Measured** (`vmath_exp_log_vs_vml`, all arms in one process, ABBA-interleaved, best-of-30 and
  best-of-40 minima over three rounds, MKL pinned to 1 thread, n = 2¹²…2²³, in **both** of this
  laptop's power states — battery at 32–35%, AC at 14% and AC at 28%): the internal
  monomorphized-vs-function-pointer ratio — power-independent, same code, same bits — is
  **1.12–1.75× faster** (typically ~1.4×; the 1.12 is exp at 2²³, where the pass is memory-bound).
  A fourth arm times the real exported `wukong_vmath_f32` and tracks whichever spelling is selected,
  so the twins are not flattering models. The two-input kernel measures **1.09–1.48×**
  (`vmath2_mono_vs_fnptr`, internal A/B only — VML has no `silu'`/`gelu'`/SwiGLU-gate entry to peer
  against). The vs-VML figures first published here — log 1.17–1.46× faster, exp 1.08–1.25× faster
  at 2¹²/2²⁰ — **are retracted by the entry below**, which was measured the same day against an
  accuracy-matched peer.
- **Accuracy re-swept exhaustively**, not sampled: exp max 1.625e-7 relative / ≤2 ULP over all
  2,237,579,431 f32 in [−87, 88.376]; log max 6.924e-7 / ≤12 ULP over all 2,130,706,432 positive
  normal f32 (12 ULP and 6.9e-7 are the same figure — the peak sits at x ≈ 1.0157 where ln x ≈ 0.0156).
  Both bars are now asserted at the measured value. The whole exp-composed family (tanh, sigmoid,
  silu, gelu, elu, softplus, mish, selu, erf, the hyperbolics, fused softmax and the GEMM activation
  epilogue) inherits the speedup through `exp8`/`log8` unchanged.
- Cost: `libwukong_runtime.rlib` grows 2.82 → 3.03 MiB (+7.4%) for the one-input change and its
  release build 18.1 → 20.5 s. Dispatch census over all 351 `tests/run/*.wk` + 17 `examples/*.wk` is
  byte-identical — this is a runtime-library change with no compile-time surface. **Not** changed:
  the bf16/f16-input dispatchers still go through the function-pointer table; they are bandwidth-bound
  and 36 more monomorphizations each is a code-size trade that was not measured.

### Benchmark honesty — the vmath "faster than VML" headline is retracted; at matched accuracy it is a tie (2026-08-06)
No kernel changed and no accuracy bar moved. The entry above kept its engineering and lost its
headline: the ratios it published were measured against **oneMKL VML in its default HA mode**, which
runs at ~0.5 ULP, while Wukong ships a ≤2 ULP exp and a ≤12 ULP log. A faster-but-less-accurate
kernel is not straightforwardly faster, that mismatch existed only in a source comment, and it is
what turned this row from a loss into a win.
- **The probe now runs an accuracy-matched peer.** `vmath_exp_log_vs_vml` resolves `vmlSetMode` /
  `vmlGetMode` out of `mkl_rt` and times VML in **LA** and **EP** as well as HA, preserving the mode
  word's non-accuracy bits so the arms differ in accuracy alone, checking the readback so a mode that
  did not take cannot be printed under its label, and measuring **every arm's max relative error and
  max ULP against a f64 reference on the very data it timed**. On the timed band VML-LA measures 2
  ULP for exp and 4 ULP for log — exactly where Wukong measures 2 and 4. The mode word this box loads
  at, `0x1a02` = `VML_ERRMODE_DEFAULT | VML_HA`, independently confirms the accuracy constants.
- **Against that matched peer, exp and log are a TIE.** Five same-run best-of-40 ABBA rounds in one
  process, AC+charging 68–76%, nothing else running: exp **1.09–1.20× slower** at 2¹⁶ and 2²⁰, log
  between **1.09× slower and 1.07× faster**. Only at n = 2²³ does either come near the floor, and
  even there only log clears it (1.48–1.66×) while exp straddles it (1.37–1.54×) — and that is
  Wukong's non-temporal store regime, not its core.
  **RETRACTED**: "exp 1.08–1.25× faster at 2¹²/2²⁰" and "log 1.17–1.46× faster at every size".
- **tanh keeps its win and gains a disclosure**: **2.85–3.24× vs VML's fastest mode** (3.93–4.91× vs
  HA), well clear of any noise floor — but Wukong's tanh measures **44 ULP / 3.5e-6 relative against
  VML's ≤1 ULP in every mode**, so it buys speed with accuracy.
- **The n = 2¹² row is withdrawn as an artefact of the clock.** The probe now *measures* the smallest
  non-zero interval `Instant` can report (100 ns here) and prints `TIMER-QUANTIZED` when the fastest
  arm is under 100 ticks. At 16 KiB/array every arm's best-of-40 minimum is 0.5–1.4 µs — 5 to 14
  ticks — which is why that row's ratios came out as exact small fractions (1.000, 8/7, 3/2, 8/5).
- **Two control columns whose true ratio is exactly 1.000** now print every run: the shipped entry
  against the `mono` twin it dispatches to, and VML-HA against a second, identical VML-HA arm. They
  read 1.06–1.23× at 2²⁰ and, in one round at 2¹⁶ **with no power-state change and nothing else
  running, 2.303×** — the same function timed at 27.4 µs and 11.9 µs, in a round whose `mono vs
  fnptr` inverted to 1.60× *slower*. That round is excluded from every range above, and it is why
  nothing under ~1.4× is published from this instrument. It is the in-crate twin of `wukong_xbench`'s
  C(twin) column, which once read a byte-identical binary as 1.31× faster than itself. Ratios below
  the wider of 1.4× and the run's own control spread are now printed as `TIE`.
- **Two documentation errors fixed.** `docs/metrics.md` called the superseded "the residual is
  algorithmic" reading *four-year-old*; per the commit bodies it dates from 2026-07-09 — about a
  month. The changelog entry above described best-of-30 in both power states, while the round that
  produced its numbers was three rounds at best-of-30 **and** best-of-40 across two states.
- **Still open, and now measured rather than asserted**: VML's LA exp/log core is a shade ahead of
  ours at n ≤ 2²⁰. `wukong-xbench`'s own VML peer (which prints much larger ratios) runs
  in default HA mode as well and has **not** been re-measured against LA.

### Benchmark honesty — three defects an adversarial verifier found and proved
No compiler behaviour changes; all three are in `wukong_xbench` (plus the measurement docs). Two of
them make Wukong look **worse**, which is the point.
- **The C/C++ pairing guard was vacuous.** `every_c_column_is_paired_with_a_cpp_column` exists to
  make it impossible for a family to print a "vs C" ratio with no C++ column behind it. Its scan
  needle was the CONTIGUOUS string `bench_external` + `("c",`, which matches only when the ext
  literal is on the same LINE as the opening paren — one occurrence in 8,733 lines, and that one on
  the exemption list, because rustfmt puts `"c",` on its own line everywhere else. It could not match
  `bench_external4(`/`_i8(`/`_bf16(`/`_halfout(` at all. The `unpaired` vector was unconditionally
  empty and the `_fast`/`_omp` exemption branch had never once executed. Falsified directly: a
  `bench_c_cpp` call reverted to a C-only `bench_external` plus `let cm_cpp = None;` compiled and the
  test still said `ok`. The scan is now whitespace-insensitive, suffix-aware, covers `build_peer` as
  well, and parses the real top-level argument list instead of a fixed text window; four floors stop
  it passing by finding nothing. Falsified again after the rewrite, three ways, each confirmed to
  FAIL and then reverted.
- **The transpose benchmark measured only the regime it wins in.** The sweep was `ns in [1024, 2048]`
  — two power-of-two squares. A transpose writes column-major, so the peer's write stream steps by
  the destination row stride, and when that stride in bytes is a large power of two the writes land
  on a few L1 sets and the blocked C peer thrashes. Same run, adjacent columns: the peer moves
  2.4 GB/s at 1024² and 18.9 GB/s at 1000² — an **8× swing in the PEER for a 2.4% change in shape**,
  collapsing the published ratio from 7.43× to 1.51×. The sweep now also runs 1000², 1031² (prime)
  and the rectangular 1100×950, labels each row's regime in its header, and closes with a REGIME
  SUMMARY that geomeans the two apart and marks the general-stride line as the result (6.56× vs
  **1.46×** single-core on the battery-state round used to develop the change). The power-of-two
  shapes are kept and reported beside it, never as it. The peers' new tile edge handling is a
  full-tile fast path plus an else-branch — never a clamped bound — so gcc still compiles the
  constant-trip-count tile body at the aligned sizes (verified standalone, ABAB best-of-20:
  new/old = 1.004–1.011, output bit-identical).
- **The C++ column was always timed second.** Every one of the ~44 `bench_c_cpp*` pair sites read
  `let c = …; let cpp = …;`, so every within-pair drift — clock ramp, thermal, a hybrid-scheduler
  P↔E migration — was charged to C++ in every family. That is a systematic bias in the published
  "vs C++" ratio, not noise: the order was the same on every repetition, so repeating the suite could
  not cancel it. Probed by putting the identical C source through the identical compiler in the
  second slot, the second slot was slower on **6 of 7 rows, by up to +17%**. `bench_external*` is
  split into `build_peer` (compile + load) and `time_peer*` (zero, time, snapshot); both peers are
  now built before either is timed — so no compiler spawn sits between two timed regions about to be
  compared — and then timed **A B B A**, each column keeping its own fastest sample. Each column gets
  one early and one late slot, so a monotone drift cancels in the ratio, and the per-column minimum
  is the rule `time_ns` already applies across its 14 blocks.
- `BENCHMARKS.md`: **withdrew** "a first smoke run already shows rows where the C and C++ columns
  differ by well more than a few percent" — an empirical claim published with no numbers, from a run
  its own commit declared non-reportable, and taken before the ordering bias above was known. The
  transpose rows there and in `docs/roadmap.md` are flagged **power-of-two-only and unverified**:
  they no longer reproduce on the current tree, in the same direction both before and after this
  change (so the sweep change is not the cause), and no corrected figure is published because every
  run available was battery-state.
- Dispatch census over all 367 `tests/run` + `examples` programs, base vs branch: **367 identical,
  0 lost, 0 gained** — as expected for a change that touches no compiler crate.

### Recognizers — a weight held in a struct field dispatches like any other buffer
Every kernel recognizer named its buffer operands by a bare `Symbol`, extracted with `single_path`.
A weight reached through a struct field — `l.wq[j*D + p]`, how every real model groups its
parameters — is an `ExprKind::Field`, so the extraction returned nothing and the *whole nest*
declined: no kernel at all, on both backends, at every `-O` level. In `wukong_xbench`'s structure-tax
benchmark, one pre-norm transformer block written with its ten weights in a `struct Layer`
dispatched **7** kernels where the arithmetically identical block over flat parameters dispatched
**16** — six projection GEMMs, the fused-bias SwiGLU gate and both affine RMSNorms lost purely to the
spelling of their weight operand.
- **A struct-field buffer base now resolves through `kernel_base_ptr`, exactly like a bare local.**
  `lower_program` interns a synthetic `"<local>.<field>"` name for every `local.field[…]` in the
  module (a text no identifier can equal — `.` is not an identifier character), and `FnLowerer`
  records each struct local/parameter's *buffer* fields (slice / array / tensor / pointer) as it
  binds them. A `[]T` field is recorded as a slice, so the kernel receives its **data** pointer and
  not the 16-byte fat pointer; a fixed `[T; N]` field's address *is* its storage and is passed
  as-is.
- **Fail-closed by construction.** The synthetic names live outside the scope chain, so any path
  that did not go through `kernel_base_ptr` simply does not resolve them and declines. Five operand
  extraction points were widened (the two GEMM operand matchers, the fused-bias base, the affine
  norm's gamma/beta) and every consumer of each routes through the shared resolver. A record is
  revalidated against the current binding of its base name, so an inner `let l: *mut Layer` that
  shadows the struct can never resolve to the outer one's field.
- Result: the struct spelling's dispatch census is now **byte-identical to the flat spelling's** —
  the same 16 call sites, symbol for symbol — and its output joins the flat, helper-factored and
  shape-typed spellings in a single bit-identical class (it was previously the only spelling outside
  one). Guarded by `tests/run/struct_field_kernel_base.wk` (every output lane printed, values
  hand-derived from the scalar loops) and `tests/run/struct_field_base_shadowed.wk`. Dispatch census
  over all 363 existing `tests/run` + `examples` programs: identical, 0 lost, 0 gained.
- Still declining, unchanged and scalar: a nested projection `l.inner.w[…]`, a tuple field `t.0[…]`,
  a field of a struct reached through a pointer, and a struct captured into a mid-function
  `@parallel` region.

### Optimizer — inlining across the call graph, and loop unrolling (2026-08-04)
- **The inliner now works bottom-up over the call graph** instead of only on leaves. Callees are
  spliced before their callers, in reverse topological order taken from Tarjan SCCs (computed
  iteratively, so a deep call graph cannot blow the compiler's own stack), so a chain of helpers
  collapses in one sweep instead of one layer per never-taken round; recursion is refused by the SCC
  itself rather than by forbidding all calls. `should_inline` now *scores* a site — a sole call site
  gets a much larger budget, a site inside a loop doubles it, a callee carrying a loop nest earns a
  bonus because inlining hands that nest to LICM/CSE — starting from the old blanket 40-instruction
  cap, so it can only ever approve more sites than before. Growth is bounded three ways so compile
  time cannot run away.
- **Partial loop unrolling** (`crates/wukong_opt/src/unroll.rs`, `-O2` and above): the counted
  two-block loop the front end emits for `for i in a..b` becomes a four-at-a-time main loop plus a
  one-at-a-time remainder loop. **It never reassociates** — float addition is not associative, so
  splitting a reduction across accumulators would compute a different number and break both the
  interp-vs-native bit-exactness gate and `-O0` ≡ `-O{1,2,3}`; the unrolled body is the original body
  four times, in order, computing the original values, and what it recovers is loop overhead and ILP
  between successive iterations. The main loop's guard is wrap-safe: `bound - 3` can wrap when the
  bound sits near the bottom of its type, so the preheader computes
  `select (bound-3 < bound), bound-3, INT_MIN` and a wrapped limit sends every iteration to the
  remainder loop instead of running the body past the end of the range.
  Measured on non-recognizer kernels (release, `--backend=native -O2`, n = 2²⁰, interleaved
  best-of-8 against `WUKONG_NO_UNROLL=1`, with an un-unrollable loop as the control for the noise
  floor): integer reduction **28%** faster, elementwise store loop **10%**, float reduction **4%** —
  the ordering a non-reassociating unroller should produce. Cost: the unrolling stage is 22% of
  optimizer time and optimizer time rises 27% at `-O2`, which end-to-end stays inside `compile-vs`'s
  run-to-run spread (−4.3% to +6.2%) with the gcc/g++/rustc ratios unchanged at 7.3–11.9×. The
  default `-O0` compile is untouched. `WUKONG_NO_UNROLL=1` disables the pass.
### Performance — kernel recognition no longer depends on spelling (2026-08-04)
Every kernel recognizer matches a *syntactic* loop-nest shape on the raw AST, inside `lower_program`
and therefore before the optimizer has run mem2reg or CSE. A semantics-preserving respelling that
changed no result could silently cost the kernel — and the two that cost the most are things every
programmer and every optimizer does. A new pre-lowering AST normalization
(`wukong_mir_build::canon`) rewrites them back to the canonical form the matchers already
understand, so natural, readable code dispatches the same kernels the hand-tuned dialect does.
- **A hoisted row base is no longer fatal.** `let ib = i*256;` followed by `a[ib + p]` is plain
  common-subexpression elimination, but it moved the index arithmetic out of the index expression:
  `match_row_col` stopped seeing `i*N + p` and the whole GEMM fell back to a scalar nest. The same
  edit on an affine RMSNorm lost `wukong_norm_affine_f32`. Such a binding is now forward-substituted
  into its index uses and the dead `let` removed. Measured: a pre-norm FFN block (RMSNorm → Linear
  → +bias → gelu → Linear → +bias → residual) written with per-loop row bases and `let`-bound dims
  dispatched **1** kernel before and dispatches the same **4** its flat twin does now
  (1 `norm_affine_f32`, 2 `sgemm_nt_epi`, 1 `velem_f32`), printing byte-identical values.
- **A module `const` used directly as a dimension now resolves.** `const N: i64 = 768;` matched every
  stride/bound comparison symbolically and then declined at `dim_value`, which resolves a `Dim::Var`
  in the function's locals — the sharp edge `examples/gpt2_forward_bench.wk` documents and works
  around by re-binding every dimension to a local `let`. Integer consts with a literal initializer
  are now folded into their uses, which is exactly what lowering does with them anyway. As a result
  `examples/gpt2_infer.wk` (the real GPT-2 124M forward) gained 1 `sgemm_nt` + 6 `sgemm_nt_epi` and
  its reduced twin `gpt2_infer_small.wk` gained 1 + 4, with byte-identical logits and argmax on both
  backends at `-O0` and `-O2`.

Both rewrites are value-identical at every node and are proved legal over the region a binding
dominates; a reassigned binding, a reassigned operand, a narrowing `let` annotation, a `/` or `%`, a
call, an index, a `defer` in the function, or a local shadowing a const all decline. Because a
recognizer is invisible to the differential and opt-invariance gates (both backends call the same
runtime symbol at every `-O` level), the change was gated by a **dispatch census** — the multiset of
`wukong_*` symbols in `--emit=mir -O2` over all 332 `tests/run` + 17 `examples` programs — which
reports 345 files identical, 2 gained, **0 lost**, 742 → 754 dispatch sites. Compile time is
unaffected: measured with a kill-switch build, front-end 30.0 ms with the pass vs 30.3 ms without
(best of 3 adjacent rounds over the 342-file corpus), and total optimizer time *falls*, since the
deleted `let`s and folded consts hand `wukong_opt` less MIR.

### Measurement — a benchmark for general code, outside the recognizer dialect (2026-08-04)
`cargo run -p wukong_xbench --release -- general` — a new opt-in mode measuring Wukong written the way
an ML engineer actually writes it, deliberately NOT in the shapes `wukong_mir_build`'s kernel
recognizers match. Every other row in `wukong_xbench` is written in that dialect, so the suite could
not see what missing it costs; and recognized kernels are **gate-blind** (a recognizer fires
pre-optimization and in both backends, so the differential and opt-invariance gates agree on the same
answer whether one fired or not), so no correctness gate could see it either. The new mode therefore
**prints the dispatch census** for every program it times and writes the exact source it compiled to
the xbench temp directory, so `wukongc --emit=mir -O2 <file> | grep -oE 'wukong_[a-z0-9_]+'`
reproduces the census against the real binary.
- **Recognizer fragility** (census only, nothing timed): ten probes in five pairs that compute
  bit-identical results and differ by one edit. Hoisting a row base out of a loop (`let ib = i*256;`
  then `a[ib+p]`), writing an activation into the store instead of a follow-up loop, writing a
  residual into the store, or naming a dimension `let n: i64 = 256` — each **on its own** makes the
  GEMM, epilogue or RMSNorm recognizer decline, and none of them changes a floating-point result.
- **Structure tax**: one pre-norm transformer block (RMSNorm → RoPE → causal MHA → RMSNorm → SwiGLU
  MLP) written five ways with the same arithmetic in the same order — recognizer dialect, natural
  spelling, factored into `[]f32` helpers, weights in a `struct`, and shape-typed
  `Tensor[f32, S, D]` — against one C peer written once. Measured spread **41–50×**; see
  `BENCHMARKS.md` for the table and the disclosed asymmetries.
- Two programs with no pattern to match: a fused focal loss with label smoothing, per-class weights
  and a hand-written backward, and a Mamba/S6 selective scan with a vector state.
- Each program is checked against an **f64 scalar reference on every lane** (never a reduction — a
  partially-correct buffer passes a sum; and never the interpreter, which cannot hold these buffers),
  and a `C(fast)` `-ffast-math` column runs wherever Wukong's dispatched kernels reassociate.

### Benchmark honesty — peer-strength audit (2026-08-04)
An adversarial audit of every C/C++/Rust peer kernel in `crates/wukong_xbench` found four systematic
defects, all of which inflated Wukong's published numbers, and corrected them. **No compiler, kernel
or `.wk` source changed; only the benchmark peers and the documents that quote them.**
- **`__restrict__` on every generated C/C++ peer kernel.** `grep -c restrict` over
  `crates/wukong_xbench/src/main.rs` previously returned **0**: not one of the 43 C kernels told gcc
  its output buffer does not overlap its inputs, so gcc could not vectorize or reorder any nested
  kernel, while Wukong's tensor parameters carry non-overlap in the type system. Isolated with the
  kernels in their own translation unit — the harness compiles each peer to a shared library, so a
  one-TU probe over `static` arrays lets gcc's IPA prove non-overlap by itself and makes `restrict`
  look like a no-op — `restrict` alone is worth **8.7× on `matmul_tn` 512³ (176.6 → 20.2 ms)**,
  **5.0× on the direct convolution (1.99 → 0.40 ms)**, **4.2× on the column-outer `colsum`**, 1.33×
  on `saxpy`, and nothing at all on `relu`/`biasadd` (0.96×/1.01×). The conv peer went from **4.9 to
  21.1 GFLOP/s** in the suite, taking the published conv multiple from 5.90× to **1.55×**. Two latent
  aliasing hazards were removed first so the qualifier is true rather than merely fast
  (`fused_linear_relu`'s const-cast-away write, and a dequant bench that passed its own output buffer
  as the unused middle pointer).
- **Column reductions / column argmax rewritten row-outer.** `c_colsum`, `c_colmax`/`min`/`absmax`,
  `c_colstat` (mean/sum-sq/L2/RMS) and `c_colarg` all scanned `for j { for i { … x[i*N+j] } }` — the
  worst loop order for a row-major axis-0 reduction, and the only order measured. Isolated at `-O3
  -march=native`, 4096×1024: colsum **21.33 → 0.58 ms (36.7×)**, colmax **34.82 → 0.86 ms (40.6×)**,
  colmean **33.03 → 1.24 ms (26.7×)**, colargmax **9.87 → 1.57 ms (6.3×)**, output identical.
  **The published ~29–50× column-reduction win becomes a tie at best and a 1.8× LOSS at worst**
  (15 of 16 measured rows are losses; the >L3 shape is a consistent ~1.65× loss across two
  independent A/B rounds), and the column argmax's
  ~2.7–4.1× becomes a 1.5–2.5× loss.
- **`matmul_tn` peer given the natural `kij` order.** It was `ijk` with **both** operands read
  column-strided. Isolated 512³: **176.6 → 9.70 ms (18.2× total: 8.7× `restrict` × 2.1× loop
  order)**. The weight-gradient GEMM's ~128× single / ~445× parallel becomes **2.6–9.2× /
  3.1–13.9×**.
- **Transpose peer cache-blocked 32×32**, matching Wukong's own kernel. Isolated 2048²: **29.51 →
  14.09 ms (2.09×)**. Single-core transpose becomes a **tie**.
- **Ten reduction-bearing benches gained the `C(fast)` [-ffast-math] column** their own fairness rule
  required (gemv, scaled_gemm, linear_bf16, conv, colmax/min/absmax, rowarg, colarg, bf16 dot/sum,
  cumsum, xent_bwd). Four of them turn out to be ties or losses once it exists: the bf16 reductions
  (1.1–1.8× slower), gemv (a tie), conv (1.12× slower), cumsum (1.03–1.42×).
- **The `@parallel` standing lines are direction-aware.** Twenty sites printed a bare ratio, so a
  0.73× — a 27% loss — rendered as "Wukong @parallel is 0.73x idiomatic single-threaded C".
- **Four regression tests** (`every_generated_c_kernel_declares_restrict`,
  `column_family_peers_stay_row_outer`, `matmul_tn_peer_stays_kij`,
  `transpose_peer_stays_cache_blocked`) plus a written peer-strength checklist now pin all of it,
  because a weakened peer breaks nothing observable — the suite still runs, the cross-language check
  still passes, and the only symptom is a bigger multiple.
- `BENCHMARKS.md`, `README.md` and `docs/roadmap.md` republish the corrected figures with the old
  ones struck through, and every "why Wukong wins" explanation that was in fact describing a peer
  defect is rewritten. `BENCHMARKS.md` gains a full before/after board and an explicit list of what
  was **not** re-measured (the end-to-end model peers in `model.rs`, the whole-suite geomean, the GPU
  section).

### Optimizer — alias analysis (`wukong_opt::alias`)
The optimizer can now tell one buffer from another. `mir_build` erases every pointer-ish source type
(`Ty::Ptr`, `Ty::Ref`, `Ty::Tensor`, `Ty::Slice` all become a bare `MirType::Ptr`), so the memory
transforms had been giving up wholesale: CSE forwarded a load only when its pointer was *literally* an
`alloca` result and cleared its whole table on any other store, DSE had the mirror rule, and LICM
refused `Op::Load` outright. A new provenance analysis answers `may_alias(a, b)` from the SSA chain
that produced each address, and all three passes plus the Cranelift backend query it.
- **What it proves** (and only this): two distinct `alloca`s are distinct storage; an `alloca` of this
  function is never a *parameter* of it; an `alloca` is never a `.rodata` blob and two blobs are never
  each other; a **non-escaping** `alloca` is unreachable through any untracked pointer and by anything
  a `call` writes; and disjoint constant byte intervals off one base do not overlap.
- **What it refuses to prove**: two distinct pointer parameters **may alias**. Wukong makes no promise
  otherwise — nothing rejects `f(a, a)` — so adopting the informal assumption `docs/roadmap.md`
  recorded for the AST loop vectorizer would be an unsound noalias, i.e. a miscompile. Closing that
  gap is a language decision, not an analysis one. `tests/run/alias_slice_params.wk` pins a program
  whose printed answer depends on it (and on a zero-trip loop over a caller-owned buffer, which is why
  a parameter's loop-invariant load is never speculated into a preheader).
- **CSE** now keys load forwarding on `(address value, accessed type)` and invalidates only what a
  store may alias; the type in the key also closes a latent hazard where a `store f32` forwarded raw
  bits into a `load i64`, and `bf16`/`f16` are excluded so the narrowing round is never skipped.
- **LICM** hoists a loop-invariant load when nothing in the body may write it *and* the access is
  dereferenceable — a known in-bounds offset into a stack slot of this function. Every `[]T` / struct /
  tuple local is such a slot, so the fat-pointer header's `data` and `len` fields stop being re-read on
  every subscript.
- **Cranelift** tags every access to a never-escaping stack slot with `MemFlags::alias_region`, so a
  store elsewhere no longer invalidates it in Cranelift's own redundant-load elimination.
- **Measured** over the whole `tests/run` corpus (333 programs, branch-point vs branch-tip compilers
  in one session). MIR counted as 4-space-indented lines of `--emit=mir -O2`, machine code as
  `objdump -d` mnemonic lines of `--emit=obj -O2`: MIR loads **3100 → 2341 (−24.5%)**, MIR
  instructions 41943 → 41056, machine instructions 67936 → 67480 (82 programs' MIR improved and
  **none regressed**; of 60 whose machine code changed, 8 grew, the largest by 18 instructions).
- **Compile time is NOT at parity — the analysis costs ~15–19% of optimizer time**, and the entry
  previously published here claiming parity was wrong. In-process `wukong_bench compile-time`,
  three-way ABBA over 12 samples per arm (branch point / branch tip before the provenance fixpoint /
  branch tip), all three measuring the same 332 programs, normalized by the untouched front-end
  column: the branch costs **1.15× (minima) / 1.19× (medians)** of the branch point's optimizer time,
  of which the provenance fixpoint is only 1.02× / 1.01×. The optimizer's share of a compile moves
  61.4% → 64.6%; on a ~70 ms compile that is roughly +11% wall clock. The cause is structural, not a
  hot spot: `Cse`, `Dse` and `Licm` each call `AliasInfo::analyze` on every invocation of every
  fixpoint iteration, so the function is walked three times per round. Caching one analysis across
  the three passes is the obvious lever and is *not* done here, because every one of those passes
  mutates instructions and a stale alias fact is an unsound no-alias answer.
- Wall clock on four hand-written *general* (non-recognized) kernels over distinct slice buffers,
  P-core-pinned ABBA rounds, best-of-10 minima, is a **wash**: 0.990 / 1.008 / 0.987, and one kernel
  at **1.066 (a 6.6% loss)** traced to Cranelift rematerializing `f32const` inside the loop once the
  freed register changed its allocation. The header loads that were removed were L1-hot and not on the
  critical path of an out-of-order core with two load ports, so the IR win does not show up as speed.
  The gain is in the IR — which is what the interpreter executes, and what a MIR vectorizer will have
  to pattern-match.

### Performance — the shape-typed surface stops costing anything
`Tensor[f32, M, N]` was the language's slowest data type. Every kernel recognizer and the whole
autovectorizer match a **single-index** `ExprKind::Index`, so the idiomatic multi-index access
`a[i, j]` fell off the end of all of them and lowered to a scalar GEP nest — and a tensor *parameter*
was spilled to a stack slot and reloaded on every element, where the `[f32; M*N]` spelling of the
identical buffer is bound directly. Measured on a 64×64 elementwise map (release, `--backend=native
-O2`, adjacent same-run A/B, best-of-5 minima): the shape-typed spelling ran **5.69× slower** than the
hand-flattened one. The tell was in the corpus — only 11 of 332 `tests/run` programs mentioned
`Tensor[`, and all 11 were tests *of* the feature; not one model program used it.
- **`a[i, j]` is normalized to `a[i*N + j]` before lowering** (`wukong_mir_build::tindex`) — the same
  row-major offset the lowering computed anyway, now spelled as an index expression every matcher
  already understands. Applies to a contiguous tensor whose every extent is a compile-time constant,
  whose element count is ≤ `i32::MAX`, and whose indices share one ≥32-bit integer type; those bounds
  are what make it value-identical to the old sign-extend-to-`i64` offset for every in-bounds access.
  A symbolic `Tensor[f32, M, N]` is untouched and keeps the hidden runtime-dim path.
- **A statically-shaped tensor parameter binds as the buffer it is**, `[T; d0·…·dn]`, not an opaque
  pointer — so it is not reloaded per element. The call ABI is unchanged (still one base pointer).
- **The autovectorizer streams tensors**, and the whole-function GEMM/int8-GEMM wrappers no longer
  emit a dead spill slot per tensor parameter.
- Result: the shape-typed and hand-flattened spellings of the same program now compile to
  **byte-identical MIR** — pinned by `crates/wukongc/tests/tensor_parity.rs` for 2-D elementwise, a
  matmul and a non-square rank-3 nest — and the measured ratio is **0.999** (same-run A/B, from 5.69).
  `tests/run/transformer_block_tensor.wk` is a full pre-norm transformer block written in tensor
  notation: it dispatches the same four GEMMs and three fused norms as the flat fixture, where before
  it would have dispatched none. Programs with no statically-shaped tensor are byte-identical to
  before (the pass returns early on a type-table scan).
### Optimizer — `mem2reg` promotes pointer slots (2026-08-04)
- **Pointer-typed stack slots are now promoted to SSA.** `mem2reg` accepted only integer and float
  slots, so every pointer-typed *parameter* — `*T`, `&T`, and every `Tensor[…]`, all of which lower to
  one MIR `ptr` — stayed in its `alloca` for the life of the function and had its base pointer
  re-loaded from memory at *every* element access. No other pass could remove it: `licm` will not
  hoist a load out of a loop and `cse` only forwards a load within one block. A three-operand
  `Tensor[f32, N]` loop with a data-dependent branch went from 17 MIR instructions and 5 loads per
  element to 14 and 2, and from **23 x86 instructions and 12 loads per element to 17 and 7** on an
  eight-pointer probe. Corpus-wide (`--emit=mir -O2` over all 333 `tests/run/*.wk`): **86 → 0**
  `alloca ptr`; by `--emit=obj`, x86 instructions −0.35%, memory-referencing instructions −1.03%,
  **0 programs got larger** (the corpus average is small because most fixtures never take a pointer
  or `Tensor` parameter; the affected *functions* lose 30–39% of their instructions and 51–62% of
  their memory operations). The win compounds — deleting `store <slot>, <ptr slot>` can un-escape
  the pointee, so a later fixpoint round promotes that too.
- **Only the definitely-initialized subset is promoted.** A read-before-write slot is promoted to a
  zero constant of its type; there is no sound one for a pointer, because `inttoptr 0` is a genuine
  null on the native backend but a *valid, addressable* slot in the interpreter. A pointer slot
  therefore qualifies only when the first access inside its own `alloca`'s block is a store and that
  block is the entry block or has an empty dominance frontier — which covers how `mir_build` lowers a
  pointer parameter, and, via the frontier arm, an inlined callee's pointer parameter whose reload
  sits inside the callee's own loop. A pointer local whose `alloca` and initializing store are
  separated by a branch keeps its slot. A `[]T` **slice** parameter is unaffected: its base comes
  from `load ptr (gep <param>, 0)`, caller-owned memory rather than a local slot, and removing that
  needs loop-invariant load motion.
- **Hardening**: `mem2reg` now also refuses a slot whose `load`/`store` type disagrees with the slot's
  own — a type-punned access whose promoted value would carry the wrong MIR type. Verified to change
  no emitted MIR on its own (332/332 corpus programs byte-identical).
- **`--emit=grad` now works on a loss whose buffers are raw `*T` / `*mut T` parameters.** It used to
  refuse outright — the parameter's base pointer reached autodiff as `load ptr <slot>` and
  `Vjp::canon` will not route a gradient through a load whose result is a pointer (*"cannot route
  gradient for load pointer … (not a parameter or a one-level gep of a parameter)"*). With the slot
  promoted the base pointer *is* the parameter value, so every access is a one-level gep off a
  parameter. `--train` still declines on such a loss: a raw pointer carries no extent, so the trainer
  cannot size its buffers.
- New gates: `tests/run/ptr_slot_promotion.wk` (every lane printed, values derived from the scalar
  semantics of each loop), four `wukong_opt` unit tests covering promotion, the inlined-callee case,
  the late-initialization exclusion, and the aliasing case where the pointer is promoted to a block
  parameter merging two addresses whose pointees must stay in memory, and a finite-difference-gated
  `raw_pointer_parameter_grad` in `wukong_driver`.
- **No wall-clock claim.** The instruction and load counts above are static and exactly reproducible;
  the run-time consequence is *not* established. The best round obtained (interleaved, core-pinned,
  high priority, best-of-14 minima) put the A/B at 1.014–1.020× against an **A/A control on the
  identical binary of 0.965–0.991×** — the effect is inside the instrument's own noise band. Earlier
  rounds were worse still: the same binary on the same program varied 2.3× across adjacent rounds
  while ~20 other build processes shared the machine. A real number needs a quiet box.

### Correctness + robustness — code-map-hardening campaign (2026-07-29)
A codebase-wide correctness pass over every crate (read-only audit groups → fix branches over disjoint
write-sets → adversarial re-verification), plus a harness-hardening wave. Almost nothing here is a new
feature: the dominant shape is a construct that used to be silently accepted and miscompiled now being
**rejected with a catalogued diagnostic**, and a kernel recognizer that used to fire on a shape its
kernel cannot represent now **declining to the (correct) scalar path**. Several uncatchable process
aborts and ICEs became ordinary diagnostics.
- **Diagnostics — a closed, two-directionally gated error-code catalogue**: `E0001` ("malformed type
  syntax") is **retired** — no stage ever emitted it, the parser reports malformed types as
  `E0203`/`E0204`, and `--explain E0001` now reports it as an unknown error code (a usage error, exit
  2) instead of printing an explanation. The live catalogue is exactly the
  29 codes `E0101`–`E0104`, `E0200`–`E0209`, `E0300`–`E0305`, `E0401`–`E0403`, `E0405`,
  `E0501`–`E0504`, `C0001` (plus the internal `E9999` fallback). Codes are stable: never renumbered,
  and a retired number is never reused. Two gates keep it honest — every code any crate emits must
  have a catalogue entry *and* every entry must have an emitter
  (`crates/wukong_diag/tests/catalog_coverage.rs`; the scanner skips comment lines, so prose may cite
  codes freely), and every catalogued code must be reached by a `tests/fail` fixture or an inline
  lexer/parser probe (`crates/wukongc/tests/fail.rs`, with a floor on catalogue size so it cannot pass
  vacuously). Adding a diagnostic now means adding a catalogue entry **and** a fixture.
- **Three `--explain` bodies corrected to what the checks actually do**: `E0304` now states that
  mutating through a non-`mut` **aggregate parameter** (`p.f = …`, `p[i] = …`) is rejected — such a
  parameter is passed by reference, so the write would land in the caller's value — while the same
  writes through a `let` binding, and `*p = …` through a pointer parameter, stay legal; `E0402` is
  retitled *recursive struct or enum has infinite size* (a data-carrying enum is reported too); and
  `E0405` drops the value-position framing — a non-exhaustive `match` is rejected in **every**
  position, including statement position and an empty `match`.
- **Diagnostic rendering**: the human renderer expands tabs to a fixed width of 4 on **both** the
  echoed source line and the caret padding, so carets land under the construct in tab-indented source
  (reported line/col — and therefore `--error-format=json` — are unchanged), and the JSON writer
  escapes any control character with no short escape as `\u00xx` so the line stays parseable JSON.
- **Lexer**: an empty char literal `''` is now `E0104` instead of silently decoding to `0` (i.e. being
  indistinguishable from `'\0'`), and a multi-codepoint `'ab'` reports one `E0104` and resyncs past its
  own closing quote instead of re-lexing that quote as an opening one and swallowing the next token.
- **Parser**: the recursion-depth budget is charged for every `.field` / `[i]` / `(args)` / `::name`
  postfix fold, so a long postfix chain reports `E0209` rather than overflowing the compile thread's
  stack with no diagnostic at all; integer literals in **type** positions (tensor dimension, SIMD lane
  count, `.tiled(…)` extent, tuple index) are decoded with full radix / `_`-separator / suffix support,
  and an undecodable or out-of-range one is a catalogued error instead of a silent `0`; attribute
  parsing no longer consumes tokens it does not own (a stray `@` no longer eats the following item and
  silently deletes it, `extern` is the only keyword spelling accepted as an attribute name, and a
  missing attribute value reports `E0207` at the missing value); and struct **functional update**
  `S { x: 1, ..base }` — which has an AST slot but no lowering — is a single clean rejection naming the
  unsupported feature, with `rest` deliberately left `None` so it cannot silently leave fields
  uninitialized (it previously cascaded five errors and fell out of the function body into module
  scope). Struct functional update is **not** a language feature.
- **Sema — new rules and closed holes**: a function name beginning with `wukong_` is **rejected**
  (`E0300`) — the prefix is reserved for the symbols the kernel recognizers emit, and such a function
  used to silently hijack kernel dispatch; a `\u{…}` escape above `10FFFF`, or in the surrogate range
  `D800..=DFFF`, is rejected, so the promise that a `char` holds a Unicode scalar value is now
  enforced; an enum discriminant the constant folder cannot read (typically a reference to a top-level
  `const`) is `E0401` instead of being silently replaced by the auto-increment value — a discriminant
  must be an integer literal or arithmetic on integer literals; integer-literal **patterns**, and both
  bounds of a range pattern, are range-checked against the scrutinee type instead of being truncated
  to its width; an array-repeat initializer's count must equal its annotation's length
  (`let a: [i32; 4] = [7; 2];` is `E0401`); an unresolvable name used as an **array length** is `E0301`
  instead of typing the array as length `0` (parity with tensor dims' `E0504`); struct field, enum
  payload and `extern`-block types are re-lowered in the body pass purely for their diagnostics, so an
  unknown tensor dim reports `E0504`/`E0301` there exactly as in a function signature; and const array
  lengths and turbofish dim values decode radix prefixes, `_` separators and integer suffixes like the
  rest of the language.
- **Call unification tightened**: a `[]T` slice, struct or tuple argument against a `Tensor` parameter
  is rejected instead of falling through a lenient wildcard arm (a slice is not a tensor base); a
  const-dim parameter against a caller's variable dim is rejected in inference mode as it already was
  in rigid mode (`?` stays the escape hatch); and tensor **layout** is part of unification, so a
  `.col_major` tensor can no longer route silently through a contiguous-typed callee.
- **A documented language limit — `MAX_CONST_ARRAY_LEN` = `u32::MAX`**: the const folder shared by
  sema's index bounds check and `mir_build`'s slot sizing declines — folding to the invalid-length
  sentinel `0`, which sema then rejects at the first index — when an **operand** or the result exceeds
  it, and uses checked arithmetic so a wrapping `+`/`-`/`*` declines instead of producing a colossal
  length. No in-range length changes value. The *operand* bound is load-bearing, because sema folds in
  `u64` while `mir_build` narrows to `u32`.
- **Layout invariant restored**: the alignment of a vector type is always a power of two
  (`next_power_of_two(elem_size * lanes)`), re-establishing the mirror with `mir_build`'s bitmask
  round-up so the two aggregate-layout authorities agree. Identity for every power-of-two lane count,
  so no existing program's layout moves.
- **MIR: the two-level story was false, and is now marked so.** `MirLevel::High` is constructed
  nowhere, `Program::new` starts at `Low`, and nothing reads `Program::level` — there is **no**
  High→Low lowering invariant, and `--emit=mir-high` means *pre-optimization* MIR, not a different IR
  level. The enum and field remain in place, documented as unused.
- **MIR verifier, materially stronger**: it now enforces true SSA — a value is defined exactly once,
  inside the value arena, and its definition **dominates** every use (dominator-tree preorder walk) —
  rejects an entry block whose parameters disagree with `Function::params` and any predecessor of the
  entry block, range-checks `Op::VecKernelCall.kernel` against the function's `vec_kernels` table and
  checks the call's result against the recipe kind, and, given a whole `Program`, cross-checks every
  call against its **callee's** declared signature (arity, argument types, result type) — available
  through `verify_program` only, which no compile path calls: `verify_or_ice` and the optimizer's
  debug-only per-pass check both go through `verify_function`, so that one check is test-only today.
  Callees with no MIR body — `print` and the `wukong_*` runtime kernels — are external symbols and stay unchecked;
  blocks unreachable from entry keep their type checks but skip the dominance rule. These are the
  invariants `cse`/`licm`/`mem2reg`/`simplify-phis` argue from. Known gap: operand *lane counts* are
  not yet checked, so a mis-shaped vector `Cast` still reaches the backends (where it is now a
  diagnostic rather than a panic).
- **Optimizer — three `-O0`-vs-`-On` divergences closed**: constant folding no longer folds bf16/f16
  **comparisons** (two distinct f32 literals landing on one narrow-precision grid point folded to "not
  equal" while the runtime compare says equal, flipping an `if`; `fold_bin` already refused narrow
  arithmetic, so the comparison path now matches — bf16/f16 constants are never folded, arithmetic or
  comparison); the self-operand folds `x - x` / `x ^ x` / `x <cmp> x` decline when the instruction's
  result type is a **vector**, where they used to materialize a scalar constant the verifier rejects;
  and inlining now merges the callee's `vec_kernels` recipe table into the caller and rebases every
  copied `VecKernelCall.kernel` index — a vectorized leaf inlined at `-O2` previously ran one of the
  caller's unrelated recipes or indexed an empty table. Contract for anyone writing a MIR pass:
  `VecKernelCall.kernel` is a **function-local index** and must be rebased when instructions move
  between functions.
- **Slices `[]T` are legal operands to the whole recognized-kernel family** (a genuine capability
  expansion): every kernel buffer operand now resolves through one `kernel_base_ptr` helper, and every
  emitter declines to the scalar nest when an operand is unbound. Around thirty emitters previously
  took a slot address directly, handing the kernel the address of the 16-byte **fat pointer** instead
  of its data. Slice-ness comes from a `slice_slots` set recorded from the *sema* type at every binding
  site — parameters, `let`, tuple fields, `@parallel` region captures, the whole-scrutinee `match`
  binding, tuple sub-patterns, data-carrying-variant payloads, and the element binding of a `for`-each
  over an array of slices — never from the MIR slot shape, which cannot distinguish `[]T` from a
  16-byte struct, a tuple or `[u8; 16]`.
- **The recognized-kernel family is f32-only (`i32` labels)**: element-type gates were added to the
  fused norms, LayerNorm backward, the row-loss heads (KL divergence, entropy), RoPE, per-row
  argmax/argmin, the four scan bodies (cumsum, cumprod, gated carry, lrscan), the shared reduction
  index base, and the cross-entropy label gather. `f64`, bf16/f16 data and `i64` labels **decline to
  the scalar nest**. This class was a hard interp/native divergence: the interpreter marshals
  slot-per-scalar and converted, while native reinterpreted the bytes. The explicitly cast spelling
  `(x[k] as f32)` still routes to the low-precision kernels, so the mixed-precision suite above is
  unaffected.
- **Recognizer preconditions that used to be assumed are now checked**, each declining to the correct
  (unaccelerated) scalar path: only half-open unit-step `for` ranges dispatch — a `step` or `..=` range
  used to have its trip count read as `end - start` and dispatch a kernel that walks each index once;
  `pool2d` requires literal dims and the nest's own output extent to equal the full no-padding extent
  the kernel recomputes, so a sub-region pool declines instead of writing the full extent at the wrong
  row stride into a smaller buffer; the fused matmul epilogue declines a peeled alpha, a peeled store
  bias and a transposed A (the alpha folds into the scaled GEMM and the bias into its own epilogue
  call, so declining costs nothing); a **whole-function** matmul whose shape the GEMM emitter declines
  now lowers as the ordinary scalar nest instead of leaving the wrapper as dead constants plus a bare
  `ret`; the batched-norm matcher requires a per-row width independent of the row; the cumulative-max
  seed must be an actual infinity; recognizer coefficients must be loop-invariant **and**
  side-effect-free (a body-local capture silently changed the polynomial, and an arbitrary call ran
  once instead of per element); the activation peelers respect shadowing through one shared predicate,
  so a user `fn gelu` / `fn silu` / `fn fmax` is no longer fused away in favour of the runtime's
  builtin curve; the arg-reduce nest guards its `x[ki]` load under `ki >= 0` (the kernel returns `-1`
  for an empty span or an all-NaN row) and uses the loop's own strict compare for the tie-break, so a
  runtime-zero trip count leaves the seed untouched; and the 256-bit AVX2 recipe path declines when a
  stream base is not a scope binding — a module-level `const` array — reporting the catalogued,
  spanned `C0001` instead of panicking the compiler.
- **The 256-bit recipe contracts a float `x + y*z` into one `Fma`**, matching both the scalar tail and
  the 128-bit CLIF path. The vector part previously rounded twice where the tail rounded once, so one
  loop returned two different answers across its own lane/tail boundary and `WUKONG_P4_NO_256=1`
  changed a program's *numbers* rather than only its instruction selection. That kill-switch is now a
  result-identical A/B knob, gated by a test.
- **Run-time disjointness for the parallel gather/scatter**: `wukong_embedding_f32_parallel` and
  `wukong_scatter_add_f32_parallel` are selected under a run-time byte-range non-overlap test (through
  `PtrToInt`, the one pointer cast both backends implement). An aliased call — two *distinct* array
  parameters bound to the same array at the call site, which the old symbol-equality guard could not
  see — lands on the serial kernel, and a weight with no compile-time extent (a slice) keeps the serial
  kernel outright. Aliased buffers are therefore correct, just serial.
- **`sdpa`'s operand contract is enforced in lowering** (sema has no signature for `sdpa` at all): the
  four buffer arguments must be f32 arrays, slices or tensor views, and when `s` and `d` are
  compile-time constants every buffer whose type pins an extent must hold `s*d` elements. A violation
  declines the lowering so the call reports `C0001` at its own span rather than segfaulting; slice
  arguments now pass their **data** pointer. Pinned by `tests/fail/sdpa_operand_{type,extent}.wk`.
- **Lowering-core fixes**: a binary operator reads each operand's width off the value the builder
  produced and widens an integer operation to hold *both* operands instead of truncating the loop
  counter back to sema's narrower type — closing a whole class of ICEs on `for i in 1..n` with
  `n: i64`; ordered-comparison signedness is derived from both operands by one shared width-aware rule
  faithful to C's usual arithmetic conversions (only an equal-rank mixed-sign pair goes unsigned), and
  the range-`for` counter uses the same rule, so `for i in 0..big` with `big: u32` no longer runs zero
  iterations; the counting-`while` normalization's hoisted bound is restricted to a **pure**
  loop-invariant bound, so a bound with a side effect or one the body mutates re-evaluates per
  iteration; every literal `match` pattern is materialized through one checked helper that declines a
  non-integer scrutinee, a `true`/`false` pattern against a non-bool scrutinee, an out-of-range literal
  (including the i64/u64 widths sema's own check cannot cover) and a non-literal range bound — MIR
  integers are signless, so the accepted range is the union of the signed and unsigned ranges of the
  scrutinee's width; array-initializer length is checked on **every** path that reaches lowering,
  including an assignment RHS and a data-enum tuple payload, with a mismatch reported as `E0401`
  instead of writing outside the buffer or leaving the tail unwritten; the monomorphization key
  separator is `$`, which no user identifier can contain, so a structural-type instantiation can no
  longer share a namespace with a user type name and mint one instance for two ABIs (the remaining
  catch-all that collapses every vector/tensor/fn instantiation onto one key is a documented gap); and
  `parse_float` strips the same suffix list sema accepts, including the bare C-style `f`, so
  `let a: f32 = 5f;` no longer emits `const.f32 0` at every opt level on both backends.
- **`@parallel` declines four shapes and lowers serially**: a labelled loop, a plain `break` out of the
  parallelized loop, a `return` out of it, and a modelled body-local reassigned inside an inner loop. A
  `break`/`continue` inside an **inner** loop remains legal. Each of these used to change what the loop
  means — unreachable code, a chunk-dependent answer, or a race. The failure mode is a silent serial
  fallback (correct), not an error.
- **Interpreter: three uncatchable process aborts became diagnostics.** Unbounded or very deep
  recursion reports `interpreter call-depth limit exceeded (likely unbounded recursion)` through a call
  depth counter; all **five** public entry points now run on the 512 MiB big-stack worker (four
  previously skipped it); and the runtime allocation intrinsic uses `try_reserve` and returns an error
  instead of reaching Rust's allocation-error handler. The interpreter reports resource exhaustion as a
  diagnostic with exit 1, never a process abort. It cannot mirror the runtime's null-return on OOM,
  because interpreter pointers are slot indices and index `0` is a live slot.
- **Interpreter: a non-positive kernel extent means zero iterations on both backends.** All 81
  output-buffer shape reads go through one `dim!` macro that bails out of the arm with the kernel's
  documented no-op when the extent is `<= 0`, mirroring every runtime kernel's own `<= 0` guard, and
  the eight value-returning reduction reads clamp with `.max(0)` so the kernel's empty-input identity
  applies. Such an extent previously wrapped to `~usize::MAX` and aborted with a raw `capacity
  overflow` out of the backend that is supposed to be the semantic oracle. Zero extents were corrected
  the same way.
- **Interpreter: three semantic-oracle repairs.** Integer SIMD lanes are truncated to their declared
  lane width in the vector `Bin`, `Neg` and `Not` arms — a lane used to keep its full-width product and
  was observably scattered to memory, disagreeing with Cranelift; `wukong_embedding_f32` gathers
  row-at-a-time straight from live interpreter memory in ascending order, so an in-place gather
  (`out == weight`) matches the source nest and native, and the real vocabulary argument drives the
  out-of-range zeroing rule instead of a snapshot heuristic; and a mis-shaped vector operand produces a
  diagnostic instead of an index-out-of-bounds panic.
- **Cranelift: the native backend no longer promises 16-byte alignment.** Generated loads and stores
  use `notrap` memory flags rather than `trusted` (`notrap|aligned`). Cranelift's `aligned` is a
  *caller* promise this backend cannot make — the 128-bit CLIF vectorizer emits vector loads at
  addresses it knows are not 16-byte aligned, and on an x86-64 host without AVX the old flag let the
  x64 lowering embed the misaligned operand in a legacy SSE instruction and fault on the first
  vectorized iteration. Strictly weaker: it can only make Cranelift materialize a load it would
  otherwise have folded.
- **Cranelift: the raw 256-bit AVX2 emitter is gated on host ISA *and* ABI.** It requires AVX2 + FMA
  **and** `x86_64` + Windows, because the emitted body hardcodes the Win64 `rcx`/`rdx`/`r8` argument
  mapping and there is no in-kernel fallback; an unsupported host now gets a compile diagnostic rather
  than SIGILL or silent memory corruption. The module's earlier claim that Windows and SysV both pass
  the first arguments in registers it normalizes to was false and is corrected. Its lane/register
  constants alias the vectorizer's so the two bounds cannot drift, and the register free list releases
  a repeated vector operand exactly once (`a*a`, or an `Fma` sharing an operand, used to push the
  register twice and compute the wrong expression).
- **Cranelift: a narrow-float entry point returns through its own ABI.** `f32`/`f16`/`bf16` `main`
  transmutes to `extern "C" fn() -> f32` instead of sharing the `f64` arm — all three are computed in
  f32 registers, so reading 64 bits back reinterpreted a live float as a denormal and
  `fn main() -> f32 { return 42.9; }` disagreed with the interpreter oracle.
- **Cranelift: recognizer/backend drift is a loud compile failure.** A `wukong_*` kernel reaching
  `lower_call` with an arity no arm handles fails the compile naming the symbol and the arity, instead
  of lowering to nothing — which silently deleted a void kernel's call and left its destination buffer
  stale, a native-only wrong answer since the interpreter dispatches on the symbol name with no arity
  check. This is the check that catches a missed arm when adding a runtime kernel. Parallel object
  codegen reports the **first failing function by index**, matching the `WUKONG_PAR_CODEGEN=0` serial
  path, so which diagnostic is reported is deterministic. And the JIT's contract is now stated:
  `JitProgram::run` executes the entry on the 512 MiB worker, while `JitProgram::call` invokes it on
  the caller's stack.
- **Runtime: scalar/AVX2 twin agreement extended to non-finite data and past the f32 mantissa limit.**
  The row-max folds in the norm and log-softmax paths make the **scalar twin** adopt MAXPS semantics
  (`(a > b) ? a : b`) so it mirrors the unblended AVX2 chain, while the cross-entropy path resolves the
  same hazard the other way — its scalar twin keeps `f32::max` (= `maxNum`) and its AVX2 fold blends the
  accumulator back over the NaN lanes; each file is internally twin-exact, and the two deliberately pick
  **different** row maxima on a row containing a NaN; the cummax/cummin vector scan folds NaN-bearing
  blocks with the scalar recurrence; the vector arg-reduce delegates any span leaving a sentinel lane
  to its scalar twin, so an all-identity span argmaxes to index `0` rather than `-1`; the row and
  column arg-reductions dispatch to AVX2 only while the running index — which rides in an f32 lane —
  is exactly representable, handing wider rows to the exact-integer scalar twin; and an unknown vmath
  op code runs the named scalar twin instead of leaving the output untouched. The module headers'
  unconditional bit-identity claims are now true.
- **Runtime: kernel contracts stated rather than hoped for.** Cross-entropy **bounds the label** at the
  one place each kernel uses it (a negative `i32` sign-extends above the row width, so one `<` covers
  both directions): an out-of-range `target[r]` yields `NaN` loss forward and a plain-softmax gradient
  backward, never a read or write past the row, and the unenforceable in-range precondition is dropped
  from the safety blocks. `wukong_embedding_f32[_parallel]` takes its three extents as `i64` — matching
  the compiler's own import declaration — and returns early on a non-positive `t` or `h` (a non-positive
  `v` leaves no valid weight row, so every output row is zeroed instead). The scaled NT GEMM's
  doc states what its writeback implements (alpha applied **after** the beta accumulate, not the BLAS
  ordering), and only the `beta = 0` combination is reachable from `.wk` source, now pinned by a debug
  assertion at the single dispatch choke point. The file-I/O read intrinsics attempt the **open before**
  the zero-length short-circuit, so a missing or unopenable path is `-1` even when `len <= 0`, matching
  the interpreter — ordering is part of that frozen contract.
- **Runtime: thread-pool provisioning is now an invariant.** Every `_parallel` kernel that can be the
  process's first rayon touch calls `ensure_global_pool()` before forking; otherwise rayon builds its
  default small-stack registry and the runtime's later 16 MiB `build_global()` silently loses the race
  for the whole process. As a backstop the pool helper records whether `build_global` actually
  installed the pool and otherwise falls back to a lazily built private 16 MiB pool. Provisioning and
  scheduling only: no chunk boundary or fold order moves, so serial == parallel stays bit-exact. Known
  gap, documented not fixed: at pool width 1 the helper runs the body **inline on the caller**, so an
  outlined `@parallel` region body is not on a 16 MiB stack there — the gate that proves a fork lands
  on a runtime-configured pool returns early at that width for exactly this reason.
- **Runtime: env knobs and reference surface.** Two GEMM instrument knobs validate their input in pure
  helpers instead of trapping inside an `extern "C"` kernel (a `0` divisor is treated as unset; a
  KiB→byte scale saturates), so a sweep script walking either range falls back rather than aborting,
  and the bump `Arena` rejects a non-power-of-two alignment (`align - 1` wrapped and the round-up mask
  became `0`, aliasing every live region).
- **Autodiff: every silent-zero or silently-wrong gradient in reach became a refusal.** The elementwise
  VJP uses a **whitelist** of the two `velem` spellings the affine rule is valid for, so a Hadamard or
  quotient kernel is refused by name instead of differentiated with the linear rule; an `Op::Store`
  into a buffer that already has a live adjoint, whose stored value is not a constant, is refused
  (constant stores — zero-init and the loss sink — stay skippable); `Op::VecKernelCall` is covered by
  the same loud guard as an unrecognized `Op::Call`; and a call to a recognized kernel **name** with a
  non-ABI arity — reachable by declaring the symbol in an `extern "C"` block — is a clean diagnostic
  instead of an index-out-of-bounds panic. Consequence: a mixed scalar+kernel loss that previously
  "succeeded" with a clobbered or zero gradient now fails to compile under `--emit=grad` / `--train`.
  The supported envelope is narrower than the old behaviour suggested, and `--emit=grad` never emits a
  silently-zero gradient — a missing rule is a hard error.
- **Autodiff: two rules added, and the coupling contract reinforced.** A self-dot reduction —
  `loss = Σ x[i]²`, lowered with both reduction operands the same buffer — now differentiates to one
  elementwise scaling (it used to fail with a misleading multiple-contributions error), so a
  sum-of-squares loss / L2 regularizer works. And `wukong_norm_f32_parallel` is mirrored into the tape
  (`Syms` / `is_kernel` / `diff_kernel_call`), so a `@parallel` batched norm — the shipped transformer
  shape — differentiates through the same rule as its serial twin. This is the standing rule: every new
  `_parallel` recognizer arm in `mir_build` **must** get a tape counterpart, or every `@parallel`
  backward through it breaks. Diagnostics now name the offending callee, and the reduction fallback
  message lists its supported set correctly.
- **Autodiff: known limitation, documented not fixed.** When a loss reads a kernel-written buffer with
  a scalar `load` rather than through another recognized kernel, the kernel's output adjoint is never
  seeded and the gradient is silently zero. The scalar-load path does now register its buffer as
  having contributed, so a later kernel-path overwrite cannot clobber it, and a matmul adjoint into a
  scalar-loaded buffer accumulates rather than overwrites.
- **GPU (`--backend=gpu` offload)**: the LayerNorm PTX computes variance in the numerically stable
  two-pass form `mean((x - mean)²)` instead of the one-pass `E[x²] - mean²`, which catastrophically
  cancels in f32 and made whole rows NaN; both siblings — the CPU runtime kernel it replaces and the
  gpu-native lowering — already used the stable form, so this was a CPU-vs-GPU divergence, and
  LayerNorm now agrees across interp / native / gpu / gpu-native. The norm offload also **declines** op
  codes with no PTX entry (log-softmax, L2-norm) and falls back to the CPU kernel instead of aborting
  the process on ordinary user input, joining the existing vmath and reduction gates.
- **GPU (gpu-native MIR→PTX)**: an `i1` in memory is accessed as **one byte** — it fell through to the
  i32 arm, and a 4-byte access at an odd address is `CUDA_ERROR_MISALIGNED_ADDRESS`, reachable from
  `struct F { a: bool, b: bool }` or `[bool; 8]`; a narrow unsigned float→int cast stays sign-extended
  to its MIR width, so it compares equal to the same number materialized as a constant; bf16/f16 values
  round to their grid at a **const** and at an `FpExt` widen, not only at a store, so gpu-native's own
  `-O0` and `-O3` agree (interp and Cranelift already rounded at all three points); and an aggregate or
  void load/store declines with `UNSUPPORTED:` rather than emitting a plausible 4-byte access. Coverage
  grew as well: the seven `_parallel` runtime symbols `mir_build` emits — the f32 activation kernel and
  the six low-precision `{dot,sum,reduce}_{bf16,f16}` symbols — are classified as their serial kinds, so a program whose
  `@parallel` function is recognized as an activation or a low-precision reduction is no longer refused
  outright by gpu-native nor declined by the megakernel, and the remaining `expect` sites became
  `UNSUPPORTED:` declines.
- **GPU: an out-of-contract launch is a recoverable decline, never a panic or a sticky driver fault.**
  The reduction launch derives its argument list from the op (a mismatched pair read one slot past the
  argument vector and latched a process-sticky illegal-address error); the autotune cache honours a
  cached token only when this build has that candidate and it is applicable at the shape, otherwise
  counting it as a miss and re-tuning; the resident-backward, AdamW and paged-KV launchers check every
  buffer length against the geometry the kernel recomputes; the flash planner owns the (entry, config)
  pairing so a shape no kernel covers cannot be built; the tile generators assert their shared-memory
  staging and warp-grid preconditions; and an op with no kernel declines with
  `CUDA_ERROR_NOT_SUPPORTED` instead of panicking inside the compiler process. In the serving stack a
  request's generation length is floored at 1 (a zero-length request underflowed its remaining counter,
  so its slot never retired and never returned its KV blocks) and the host-side decode advance is
  all-or-nothing behind a feasibility pass. The int4 dequant reference discriminates on the field every
  launcher dispatches on (`zeros`) rather than the descriptive `signed` flag, and the host fp8 E5M2
  encoder handles subnormals and signed zero like the hardware instruction — the E4M3 twin has the
  identical defect and is a known open item.
- **CLI: two flag combinations are now rejected instead of silently doing half the work.** `--run`
  cannot be combined with `--emit=<stage>` — the driver dispatches exactly one of them and dropped the
  other while still exiting 0, and *which* one won depended on where the stage sat relative to the run
  block. And `-o` is accepted only with `--emit=obj` or `--emit=exe`, and never with `--run`; `USAGE`
  now states the real semantics: **only `--emit=obj` and `--emit=exe` write a file; every other stage
  prints its artifact on stdout.**
- **Driver: MIR is verified before every backend exit.** `--emit=llvm-ir`, `--emit=obj` and
  `--emit=exe` verify through the same helper `--run` uses. `--emit=mir` / `--emit=mir-high` stay
  ungated at that point on purpose — they verify *after* dumping, so a broken program still prints the
  MIR that shows why. Note the optimizer's per-pass verify-each is `#[cfg(debug_assertions)]`, so
  before this a release compiler verified nothing at all on the AOT path.
- **Driver: the compiler no longer aborts when its output pipe closes.** Artifact writes (tokens, ast,
  llvm-ir, mir) and the diagnostic and verifier-ICE lines go through fallible writes, so
  `wukongc --emit=mir p.wk | head -1` exits with the compile status instead of crashing — the
  workspace's `panic = "abort"` had turned a failed `print!` into a process abort. Exit codes are
  meaningful under truncation.
- **Driver: `--emit=exe` generates its link inputs in a pid-keyed scratch directory** that is removed
  when the link finishes, instead of writing `{stem}_shim.rs` / `{stem}_rt.c` into the process's CWD —
  which overwrote and then deleted a user file of the same name and raced between concurrent same-stem
  links. The object still lands beside the source. The rustc-driven link also passes `-C panic=abort` to
  match the workspace's release panic strategy, without which a release-built compiler's sibling rlib
  rejects the shim.
- **Driver: three `--grad`/`--train` defects fixed.** The default `--grad-wrt` has a *training* flavour
  that excludes the loss-output parameter, so `--train --grad-of=<fn>` works with the documented
  default instead of always failing; a repeated index (`--grad-wrt=0,0`) is rejected rather than
  emitting a backward whose extra gradient buffer is never written; `--train-steps` no longer
  pre-reserves its trajectory vector, so a huge value is not an allocator abort; and `--emit=grad`
  propagates verification failure instead of printing an ICE and exiting 0.
- **Test gates now pin documented contracts.** Object emission must be byte-identical across three
  fresh `--emit=obj -O2` processes per program, and the `WUKONG_PAR_CODEGEN=0` serial path must be
  byte-equal to the parallel one on the 24 largest programs (the ones with enough functions for the
  parallel path to engage); diagnostic emission must be identical across three fresh
  `--error-format=json` processes per compile-fail fixture; all three determinism tests count
  comparisons and require corpus coverage so they cannot pass vacuously. The limit is stated: the
  seeds only perturb std `HashMap`/`HashSet`, and the optimizer's `FxHash` maps have a fixed seed, so
  these gates say nothing about insertion-order dependence there. The AOT-exe gate probes **once**
  which of the driver's two link paths is available, fails on a genuine link failure with the captured
  stderr, refuses to pass having linked zero fixtures, and uses a per-process scratch directory.
- **The examples corpus carries declared classes.** `examples/*.wk` are automatically interp-vs-native
  differential and opt-invariance fixtures — and an example that *fails to compile* satisfied both
  agreement checks, because both backends failed identically. Each example now declares what running it
  must produce: `Runs` (exit 0, non-empty stdout) is the default, so a new example must execute or be
  declared; `NotLowered` (exit 1 + `C0001`) covers `matmul.wk`, `softmax.wk` and `vadd.wk`; `NoEntry`
  covers `gpt2_config.wk`; and `DataGuarded` covers the three GPT-2 programs that take a data-absent
  early return and are excluded from the agreement checks, with a printed reason, when the weight blob
  *is* reachable. A separate gate re-derives the whole GPT-2 offset table from the exporter's own
  tensor order, needing neither torch nor the blob. Five example headers were rewritten to the dispatch
  the emitted MIR actually shows — `--emit=mir -O2` is the only authority, because recognized kernels
  are gate-blind — and `examples/gpt2_forward_bench_small.wk` was added to carry the same dispatch
  surface at tiny dims with no file I/O, so it is fit to be a differential fixture. The GPT-2 export
  tool resolves its output directory from `$GPT2_DATA_DIR`, then `argv[1]`, then `<repo>/data/gpt2`
  instead of hard-coding one machine's path, and the verifier makes the logits artifact's **age** part
  of its verdict so a verify cannot re-assert a result for a run that never happened.
- **Corpus shape and placement rule.** `tests/run` holds 332 `.wk` fixtures and `tests/fail` 110, plus
  eleven inline lexer/parser probes — together covering all 29 catalogued codes. A program that must be
  **rejected** belongs in `tests/fail`, because every `tests/run` program is required to reach
  `--emit={mir-high,mir,llvm-ir}` successfully (two `sdpa`-decline fixtures moved for exactly that
  reason). `tests/run/*.wk` supports a non-default `// RUN:` directive, and eleven element-type and
  aliasing fixtures carry `// RUN: --run --backend=native` so their expected output pins the backend
  that was wrong.
- **GPU test knobs.** `WUKONG_GPU_REQUIRED=1` turns the GPU suite's `[skip]` lines into assertions and
  `WUKONG_PEER_REQUIRED=1` does the same for every peer/toolchain skip — both no-ops by default, so a
  genuinely CPU-only or peer-less box still passes. Skip lines now carry the driver's actual error, so
  an exclusive-mode device or an empty `CUDA_VISIBLE_DEVICES` is not misreported as a machine with no
  GPU, and the paged-attention module's device-free parts (PTX generators, int8 quantizer, f64
  reference) run under a plain `cargo test`.
- **Reverted: the pointer arm of `lower_bool_cond`.** `PtrToInt(p) != 0` is **not** a null test in the
  interpreter, whose addresses are slot indices starting at `0`, so the first pointer taken in a
  function reads as null (`if p` printed 0 under interp and 1 natively) — trading a loud error for a
  silent interp/native divergence, the worse failure. A **pointer** condition (`if p`, `while p`,
  `if p && …`, `assert(p)`) therefore still reaches `cond_br` unchanged and both backends reject it
  with raw verifier text and no span. Do **not** document C-like truthiness for pointers: a correct
  lowering needs a null test the two backends genuinely share — a dedicated `IsNull` op, or the
  interpreter reserving address `0` as never-allocated. The same commit's coercion of integer operands
  to the six activation-**backward** intrinsics was kept and is pinned by a fixture.
- **Refuted, then raised: the interpreter's release call-depth cap.** Each limit is the largest value
  that rejects **no** depth known to complete, so an earlier, lower pair was refuted for rejecting
  depths that run fine — the pre-guard interpreter simply ran until the stack gave out, and a cap
  chosen for "safety margin" is a regression over that band, not a precaution. The debug limit
  additionally has to clear the depth the Cranelift differential gate exercises, or the *interpreter* —
  the semantic oracle — refuses a depth native completes, reintroducing the very divergence that gate
  exists to catch.

### Capability — GPT-2 124M end-to-end inference + typed file-I/O intrinsics (2026-07-13)
- **File-I/O intrinsics — headerless raw little-endian typed blobs**: `read_<T>` / `write_<T>` for `T`
  in `{f32, f64, i32, i64, i8, u8}` read and write a flat file of that element type
  into / out of a `[]T` buffer, with no header. `read_<T>(path, buf)` returns
  `min(buf.len, file_bytes / sizeof T)` on success, `-1` if the file cannot be opened, and `-2` on a
  mid-read I/O error (`0` when the buffer length is `≤ 0` **on an openable path** — the open is
  attempted first, so a missing file is `-1` regardless of the buffer length; that ordering is part of
  the frozen contract); `write_<T>(path, buf)` creates/truncates the
  file and returns the element count written (`-1` if the file cannot be created, `-2` on a write
  error). Both are differentially tested **interp == native** (round-trip, partial-read, and
  missing-file run tests), with compile-fail tests for path / buffer / arity misuse.
- **GPT-2 124M inference end-to-end, matching HuggingFace**: `examples/gpt2_infer.wk` compiles and runs
  the **real pretrained OpenAI GPT-2 124M** (124,439,808 parameters) as an ordinary Wukong program on
  the native Cranelift-JIT backend. It loads the weights from a flat little-endian f32 blob (exported
  from HuggingFace `transformers` by `tools/export_gpt2.py`) via `read_f32`, and the prompt token ids
  via `read_i32`, then runs the full forward — token + learned positional embedding, 12 pre-LayerNorm
  blocks (biased QKV, 12-head causal attention, tanh-GELU MLP, residuals), final LayerNorm, tied LM head
  → logits `[5, 50257]`. **The next-token logits match HuggingFace's reference to a relative max error
  of 1.87×10⁻⁶** (`max|Δ| = 2.44×10⁻⁴`, `max|ref| = 130.28`), and the argmax next token is
  **1757 (" John")** for the prompt *"Hello, my name is"* — matching HuggingFace exactly
  (`tools/verify_gpt2.py`). This is a **numerical-correctness / capability** result — **inference, not
  training; no speed comparison was measured or is claimed**. **Tokenization is external** (the `.wk`
  consumes integer token ids produced by the HuggingFace tokenizer). The full run is **native-only**
  (the interpreter cannot hold 124M parameters); the reduced-config twin `examples/gpt2_infer_small.wk`
  runs the same forward pass **bit-identically interp == native** as the CI examples differential +
  opt-invariance gate.

### Performance + honest measurement — perf-perfect session (2026-07-10/11)
Three-target directive round (A: parallel GEMM ≈ oneMKL + near-linear model scaling; B: beat fused
`torch.compile`; C: compile-time floor). Same-run/adjacent instruments; full ledger
`prompts/results/perf-perfect-session1.md`.
- **Beats fully-optimized `torch.compile`, both batch sizes (Target B MET)**: vs TorchInductor
  max-autotune, fullgraph, warmed — ATEN-pinned because the Windows Inductor CPP FP32 GEMM template
  (`cpp_CppMicroGemmFP32Vec`) is broken, so ATEN/MKL is torch's strongest *working* CPU GEMM
  (disclosed) — all-threads, the 12-layer GPT-2-class stack runs **S=512 1.39–1.81× / S=128
  1.07–1.43× FASTER** across three independent same-day rounds (isolated per-side probes, torch at
  its best; ≤1e-6 cross-check, serial==parallel bit-exact). Wukong-1c also beats compiled-torch-1T
  at both shapes, and beats torch's strongest *eager* config too (S=512 1.07–1.37×). Beating eager
  does not count — this is the compiled bar.
- **Dynamic block-claiming is the parallel-GEMM default (`WUKONG_GEMM_DYN`)**: an atomic block-claim
  queue replaces the static split; root-caused the mid-size variance to OS-preemption straggler
  episodes on the worker pool (per-call min stays fast, p50/p90 widen) and halved it. Standing vs
  MKL-all, same-run: **512³ ~98–99%, 1024³ ~104%, 2048³ 91%, 4096³ 93%, 256³ ~94–110%** (was 70–83%
  @512–1024³); skinny NT shapes **75–112%** (4/6 at/above parity). A `(96,96)` 2D block candidate
  fixed wide-M/narrow-N mis-shaping (512×768·768ᵀ 75→89%).
- **Near-linear-as-physics-allows model scaling (Target A(b) MET)**: pool unification + a **width-1
  serial fast-path** (T=1 now at serial parity, was 0.62–0.80×). S=512 curve
  **1.00/1.70/2.13/3.00/3.68/4.70× (T=1..16)**, default 4.44×, best 5.22× — tracking the ~5.0–5.4×
  same-run hybrid MKL ceiling with no Wukong-attributable serial shortfall.
- **Compile-time floor characterized (Target C)**: central AC run — 296-file corpus **168.1 ms
  front→object** (0.57 ms/file); backend 73.4% (Cranelift codegen 91.7% / object-write 8.3%),
  optimize 16.3%. compile-vs **7.4–11.7×** vs gcc/g++/rustc (both subprocess, obj, -O2); in-process
  vs spawned-toolchain ~306× JIT; `--emit=exe` ~95% rustc-link. Release verifier-off + parallel
  per-function codegen (byte-identical 296/296). See `docs/compile-floor.md`.
- **Documentation reconciled to these records**: a five-way doc audit refreshed every stale perf
  claim (`README.md`, `docs/metrics.md`, `BENCHMARKS.md`, `docs/internals.md`, `docs/compile-floor.md`,
  `next-steps.md`) to the 2026-07-11 numbers, corrected the model-vs-torch bar from *eager* to
  *`torch.compile`* everywhere, and replaced bare-absolute-GFLOP/s headlines with %-of-MKL /
  %-roofline / same-run ratios.

### Performance + honest measurement — perf-sota session 3 (2026-07-09/10)
Ten-target directive round; every baseline re-proven before optimizing, every win from the real
pipeline via same-run/adjacent instruments (full ledger: `prompts/results/perf-sota-session3.md`).
- **`@parallel` head-loop regions — the model now beats all-threads PyTorch at S=512**: an
  independent-iteration `for` loop with body-local scratch inside an `@parallel` fn outlines into
  its own MIR fn + one `wukong_parallel_for` region (the whole-fn outliner's exact contract —
  zero backend changes; 16 MiB worker stacks for the privatized frames). Legality is a
  conservative affine-disjointness proof (one `hh*C` term per written array, outer strides erased
  mod C·N, inner terms bounded below C; declines to serial on anything unproven — captured-scalar
  writes, opaque indices, calls, `print`, slices, cross-iteration deps), each iteration runs the
  SERIAL kernels (identical per-iteration op order ⇒ serial == parallel bit-exact), and autodiff
  declines loudly (no silent zero gradients). The model bench spells its attention head loop that
  way naturally; e2e fixtures pin interp == native == @parallel at every opt level, even + ragged
  head counts, and the decline case. Result (two roofline-validated rounds, all cross-checks
  green): S=512 `@parallel` **440 → ~312 ms** ⇒ **1.10–1.24× FASTER than all-threads eager
  torch** (was 1.01–1.63× behind); S=128 parity (1.19× faster / 1.02× behind at round noise; was
  1.5–1.9× behind); model `@parallel` scaling **3.1–3.6×** (campaign start: ~1.5–2.1×); the
  multicore stack ~61–69× idiomatic single-thread C.
- **Size-keyed 2D parallel GEMM dispatch**: the cooperative shared-pack redesign
  (`sgemm_2d_shared` — panels packed once per K-block) was built, gated bit-exact, and then
  **refuted at mid/large shapes by adjacent ABBA in both orderings** (per-block packing wins
  1024³ ~462 vs ~385 GF/s; its "redundant" packing is each worker warming its own L2, while a
  shared pack hands consumers panels another core packed, plus a per-K-block barrier). Shipped
  as a size-keyed default: per-block ≥2²⁶ MACs, shared-pack + 1 Mi-MAC work-scaled tasks in the
  small band, parallel gate 2²⁶→2²³ (256³ engages at **1.7–2× over serial = 102–123% of
  adjacent MKL-all**; was ~39–52% deliberately-serial). Mid-size standing: **70–83% of MKL-all
  @512–1024³, 88–93% @2048³ vs a healthy peer**. New skinny-M/row-rebalance block-shape policy
  (+16–25% at the model's skinny FFN shapes, MKL-anchored vs the old binary).
- **C-tile prefetch in the GEMM microkernel prologue**: 2048³ single-core
  **86–88% → 96–99% of MKL-1c** (MKL-anchored ABBA, both orderings; `WUKONG_GEMM_PF_C=0`
  reproduces the old tail). Hint-only — bits unchanged everywhere.
- **`@parallel` streaming maps go multicore**: `wukong_velem_f32_parallel` (fixed-chunk,
  serial==parallel bit-for-bit) + recognizer/interp/cranelift/autodiff-tape wiring, so a mixed
  `@parallel` function's residual-add/saxpy loops (the transformer block's hot elementwise ops)
  no longer run serial; e2e fixture pins interp==native at every opt level.
- **Model bench (3 valid rounds)**: `@parallel` scaling **S=128 2.19→2.96×**; vs all-threads
  torch **S=128 1.08–1.20× behind (was 1.5–1.9×)**, S=512 an honest **1.01–1.63× range — the
  width is the peer's power-state swing** (torch-Tn 443→271 ms across rounds; Wukong held
  ~440 ms in all of them — the parallel path does not yet ride clock upside; head-loop
  parallelism is the named open lever). Single-thread: 1.10–1.16× faster than torch-1T @S=512.
- **exp ldexp restructure reverted on measurement**: the bit-identical exponent-field-add tail
  (exhaustively verified over 2³²) measured a consistent ~10–15% throughput **loss in BOTH
  thermal states** (old-vs-new binary interleaved ABBA; tanh, which composes exp8, confirmed
  independently) — port rebalancing cannot beat a clock throttle that slows every port.
  Honest vmath standing vs VML across states: **tanh 2.7–2.9× faster; exp 1.23–1.45× and log
  ~1.25× slower** (the earlier single-session "exp 1.05× faster / log 1.14×" did not reproduce).
- **GPU: warp-specialized flash shipped where it wins**: 2-warp named-barrier anti-phase
  (`flash_d64_ws`) + 3-stage-ring (`flash_d128_ws3_lm`) kernels, tolerance-gated vs the f64
  oracle; the clock-cancelled A/B wins **only 4–6% @S=4096** (tie @2048, 10–52% loss @≤1024),
  so the default route is exactly S≥4096 with pins keeping the losing regimes unroutable
  (`WUKONG_FLASH_WS=0` kill-switch). The long-S cuDNN gap is structural (SFU-bound) and now
  measured shut as a scheduling problem.
- **GPU 4096³ GEMM: v2cs streaming epilogue** (+2.7% round-robin over the swz base →
  **76.8% of cuBLAS-f16 / 80.4% of the honest f32-out peer**) dispatched at A+B ≥ 48 MB; the
  new `f32-out cuBLAS peer column` makes the C-write dtype asymmetry visible (~10pp of the old
  "gap" was peer flattery). 3-stage pipe / raster re-tunes / launch-bounds all measured losses.
- **Serving goodput ceiling doubled**: Bcap parameterized end-to-end with KV-budget guards,
  graph-driven scheduler (one cached `cuGraphLaunch` per step, **bit-identical to eager** over a
  96-request drain), bounded-look-ahead first-fit admission, an honest static-batching peer, and
  opt-in int8-KV (1.88× smaller cache; exact round-trip gate). **Bcap=256: 85.6× goodput vs
  fill=1** (33.1k tok/s; old ceiling 38–39× @Bcap=64, reproduced), scheduler drain **1.14–1.27×
  vs static batching** with the amortization-vs-scheduling decomposition disclosed.

### Performance + correctness — perf-sota session 2 (2026-07-08)
- **vmath exp/log to (near-)VML parity**: the transcendental dispatch loop gained a ×4 ILP unroll +
  an NT-store streaming regime, then both cores were rewritten as 8-bucket in-register-LUT
  (`vpermps`) reductions — exp `2^(j/8)` table + degree-3 residual poly (~1.3 ULP, exhaustively
  swept), log reciprocal/ln tables + degree-5 poly (≤6.9e-7 rel, exhaustive over [0.25,4)) — each
  mirrored value-exactly in the scalar twins and the inlined-MIR emitters. Same-run vs oneMKL VML:
  **tanh 2.4–3× faster, log 1.14×, exp 1.05×-faster(cool)–~1.3×(throttled)**, from the former
  ~1.7–2× loss; log2/log1p now 9.9×/7.7× vs scalar C.
- **2D block-parallel GEMM shipped as the default parallel path**: BLIS/MKL-style per-thread
  L2-resident C-block ownership (`sgemm_2d_blocks` — MR/NR-aligned block grid, per-worker pack
  scratch, one task per block, no barriers) replaces the row-panel/shared-B decomposition —
  adjacent-run ABBA **1.26–1.65×** at 512–1024³ (512³ ~182→~305 GF/s, 1024³ ~365→~460), lifting
  mid-size `@parallel` from ~46–72% to a power-state-stable **66–69% of all-threads oneMKL**
  (87% @2048³ same-run). Bit-exact vs the serial kernel by construction (one owner per C block,
  ascending K-blocks, same kc grouping; pinned by `sgemm_2d_blocks_matches_serial`);
  `WUKONG_GEMM_2D=0` opts back to the previous path as an adjacent-run instrument. Downstream,
  model-bench `@parallel` scaling rose from ~1.5–2.1× to **1.9–3.8×** and the all-threads-torch
  gap at S=512 stabilized at **~1.1×** (1.07/1.10× across two rounds; was 1.05–2.5×
  thermal-dependent).
- **Parallel GEMM (precursor work)**: A+B packing fused into one parallel region per K-block
  (+7.5% @512³ same-run, `WUKONG_PACK_SPLIT_REGIONS=1` kill-switch) and a persistent-broadcast-
  region variant (one pool wake per call). Honest negatives recorded in-code: the persistent
  region A/Bs as a wash, a smaller mid-size pool and hard worker pinning
  (`WUKONG_GEMM_AFFINITY=1`) both measure slower — scheduling was never the mid-size gap; the
  2D decomposition above was.
- **gpu-native (MIR→PTX) correctness, 8 programs repaired**: the `mrt_velem` device kernel ignored
  the Hadamard/Div mode bits (computed `x+y` for `x*y`/`x÷y`); `mrt_norm` sent log-softmax and
  L2-norm to the RMSNorm branch; both sreduce kernels lacked the |x|-sum/|x−y|-sum ops (silent 0);
  float→narrow-int casts truncated mod 2^w instead of saturating; and megakernel mode null-deref'd
  through tid0-guarded pointer slots (`CUDA_ERROR_ILLEGAL_ADDRESS`). Corpus gate: 193/274 matching
  the interp oracle, zero mismatches/faults (81 honest UNSUPPORTED skips); a device fault can no
  longer cascade — the harness records the root fault and reports later programs as NOT RUN.
- **D=128 flash attention productionized**: `wmma_flash_applies/entry` now dispatch the
  `flash_d128_mp_lm` ldmatrix kernel (1.11–1.20× cutlass mem-efficient @S≤1024) — D=128 previously
  fell back to the f32 flash. A single-buffer occupancy probe pins the D=64 long-S plateau as
  SFU/serial-softmax-bound (2× occupancy = 1.03× wash).
- **End-to-end model bench hardened**: two full-peer rounds (C, `-ffast-math` C, PyTorch eager
  1-thread/all-threads), all cross-checks <2e-6, interp gate + serial==@parallel bit-exact:
  ~19–21× C(gcc), parity vs torch-1T (up to 1.49× faster @S=512), behind all-threads torch
  multicore — post-2D-GEMM a stable ~1.1× @S=512 and 1.5–1.9× @S=128 (see the 2D bullet above).

### Performance — close-the-NVIDIA-gap campaign (vs the vendor libraries)
- **CPU GEMM to oneMKL parity** (`perf/cpu-library-grade`): size-adaptive `select_kc(k)` + a private
  physical-core rayon pool put single-core GEMM at 102–104% of MKL-1-thread (256/512³) and 84–95%
  (1024–4096³); `@parallel` reaches 86–129% of MKL-all-threads at ≥2048³. MKL cblas/VML peers added to
  `wukong_xbench`. (vmath's then-open ~1.7–2× VML loss was closed in the 2026-07-08 session above.)
- **fp16/bf16 large-GEMM cliff vs cuBLAS** (`perf/gpu-gemm-cliff-2`): the no-pad ldmatrix+XOR-swizzle
  workhorse takes 2048³ to ~87–90% and 4096³ to ~83% of cuBLAS (past the prior 77% `mma.sync` ceiling).
- **int8/fp8 GEMM vs cuBLAS IMMA / cuBLASLt** (`perf/gpu-quant-2`): int8 beats IMMA at 2048³
  (~96–105%); the fused int8 GEMM+dequant beats the cuBLAS GEMM+dequant chain 1.1–2.2×; fp8 is 82–151%
  of cuBLASLt. w64 warp-tile + autotuner candidates.
- **Fused attention vs a genuinely-fused FA2-class peer** (`perf/gpu-attention-2`): vs PyTorch SDPA's
  cuDNN / cutlass mem-efficient backends, the fused flash wins fused-RoPE 1.8–5.7× and causal D=64
  1.03–1.16× (beats both) for S≤512; D=128 ldmatrix beats cutlass for S≤1024.
- **Conv vs cuDNN** (`perf/gpu-conv-2`): a real cuDNN-9 peer plus implicit-GEMM + Winograd
  F(2×2,3×3)/F(4×4,3×3) + a fused epilogue + `conv2d_best` per-shape dispatch reach cuDNN
  parity-to-win (1×1 3.5–5.6×, deep-channel 0.93–1.21×).
- **GPU serving stack** (`perf/gpu-serving`): vLLM-style paged KV-cache + Orca continuous batching +
  whole-model decode CUDA graph + int8 KV (3.88× footprint) + a TP partition sim — 39.1× batching
  goodput at fill=64.

All measured same-run (clock-invariant) on a mobile RTX 4050; the documented gaps and measured negative
results are recorded in `prompts/results/`, and every kernel stays gated against the interpreter oracle.

### Added
- **Front-end**: lexer (with `@attributes` and error recovery), recursive-descent + Pratt parser,
  AST with a pretty-printer, and `--emit=tokens|ast`.
- **Types & semantics**: the shared type vocabulary (`wukong_types`), name resolution, type
  checking, and **compile-time shape checking** for tensors (rank/dimension unification, symbolic
  dims), with errors `E0501`/`E0502`.
- **Middle-end**: block-parameter SSA MIR, a builder, a pretty-printer, and an SSA/dominance
  verifier; AST → MIR lowering (alloca-per-local). There is **no** `MirLevel` invariant —
  `MirLevel::High` is constructed nowhere, `Program::new` starts at `Low`, nothing reads
  `Program::level`, and `--emit=mir-high` means *pre-optimization* MIR, not a different IR level. See
  the 2026-07-29 entry for the verifier's actual rule list.
- **Optimizer**: a fixpoint pass manager backed by CFG and dominator analyses (Cooper–Harvey–Kennedy
  immediate dominators + dominance frontiers), with whole-program leaf-function `inlining`, `mem2reg`
  (promote scalar slots to block-parameter SSA), `simplify` (constant folding + algebraic identities
  + self-comparison folding), `simplify-cfg` (constant-branch folding + straight-line block merging +
  unreachable-block pruning), `simplify-phis` (dead/trivial block-parameter elimination), `dce`,
  `cse` (dominator-tree value numbering with load forwarding), `dse` (dead-store elimination), and
  `licm` (loop-invariant code motion), wired across `-O0..-O3`. In debug builds the pass manager
  verifies the MIR after every pass. Across the run suite and kernels, `-O3` removes ~42% of IR ops
  (~48–54% on the heavy transformer/GEMM kernels) and runs ~1.5–2.5x faster than `-O0` under the interpreter.
- **Back-ends**: a zero-dependency MIR interpreter (`--run`), a from-scratch **native Cranelift
  backend** (JIT + `--emit=obj` host object + `--emit=exe`, the latter linked via a rustc-driven link
  that falls back to the system `cc`/`$CC` — **no LLVM toolchain**), and a textual LLVM-IR emitter
  (`--emit=llvm-ir`, text only — emitting it needs no LLVM installed). The native backend is
  differentially tested against the interpreter bit-for-bit.
- **Matmul → tuned GEMM dispatch**: the compiler recognizes a matmul loop nest — the `ikj` accumulate
  and `ijk` dot-product forms, including the `nn.Linear` `C = A·Bᵀ` spelling — and lowers the whole
  nest to a register-blocked (6×16), cache-tiled, packed **AVX2/FMA** GEMM microkernel in the runtime
  (`wukong_sgemm` / `_nt` / `_parallel`). On a Meteor Lake laptop this beats `gcc -O3 -march=native`
  on the naive nest by **~2.4–3.5× single-thread and ~18× parallel** on `C = A·B` (and **~20–75× on
  `nn.Linear`**, where naive C stays latency-bound), the lead growing with matrix size. The serial
  kernel holds ~90–105 GFLOP/s (~80% of one P-core's AVX2-FMA peak; ~105 pinned at 512³); the parallel
  one packs panels across cores — reusing one pack-scratch allocation across all cache blocks instead
  of re-allocating per K-block (≈+45% at 1024³, ~395–434 GFLOP/s) — and skips threading below a work
  threshold. The 6×16 microkernel stores full tiles straight to C and unrolls the K loop ×4. The interpreter calls the identical
  kernel (marshalling its memory) so the oracle stays exact.
- **Auto-vectorization**: straight-line elementwise loops (incl. branchy ones via if-conversion) and
  float **reductions** (reassociated to vector-lane accumulators) lower to SIMD automatically;
  `x + y*z` contracts to a hardware FMA; adjacent same-range loops fuse. Reductions (`dot`, L2 loss)
  run ~2.6–2.8× faster than serial C. The general vectorizer emits 128-bit CLIF by default (Cranelift's
  x64 vector ISA still caps there — a 256-bit `f32x8` SSA value is rejected, per the
  `cranelift_still_rejects_f32x8` tripwire) and dispatches to a **raw 256-bit AVX2 machine-code path**
  (VEX-encoded via `iced-x86`) for large trip counts (trip-gated; kill-switch `WUKONG_P4_NO_256`;
  **x86-64 Windows hosts with AVX2 + FMA only** — the emitted body hardcodes the Win64 `rcx`/`rdx`/`r8`
  argument mapping, and any other host gets a compile diagnostic rather than a fallback).
- **Transcendental → 256-bit AVX2 dispatch**: a pure `out[i] = f(x[i])` loop for **35** functions —
  `exp`/`log`/`expm1`/`log1p`/`tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`mish`/`selu`/`tanhshrink`/
  `hardsigmoid`/`hardswish` plus **`softsign`** (bounded poly activation) and **`logsigmoid`** (the stable
  log-sigmoid behind binary-cross-entropy-with-logits / contrastive losses), **`sin`/`cos`/`tan`/`atan`/`asin`/`acos`**
  (RoPE rotary embeddings and the geometry/3D-vision/graphics-ML angle ops), **`erf`** (exact
  BERT/GPT-2 GELU), **`exp2`/`log2`/`exp10`/`log10`** (FlashAttention base-2 softmax, quantization, and
  base-10 decibel/log-scale features), **`cbrt`** (the all-real cube root — LAB color, variance-stabilizing
  transforms — completing the `sqrt`/`rsqrt`/`cbrt` root family), and the full
  **hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh`** (`atanh` = the Fisher z-transform; the
  inverse trio powers hyperbolic/Poincaré embeddings and normalizing flows) — lowers to a tuned **256-bit AVX2/FMA runtime
  kernel** (`wukong_vmath_f32`) — the width Cranelift's general vectorizer can't emit (it caps at
  128-bit SSE). `silu` (Llama/SwiGLU) and `gelu` (BERT/GPT-2/ViT) are first-class intrinsics; the
  kernel's per-element op sequence mirrors the inlined Cephes/A&S polynomial, and the interpreter marshals
  through the identical kernel, so the differential oracle stays exact and dispatched/composed forms
  agree. A multi-statement (fusion-merged) body dispatches one kernel call per activation, and an
  `@parallel` activation dispatches each thread's chunk — so it runs multicore × 256-bit. Versus C's
  scalar `libm` (which can't vectorize a loop with a call), the family runs **~4–11.5×
  faster** single-thread (`asinh`/`acosh` and `sin`/`cos` win most — `libm`'s `asinhf`/`sinf`/`cosf`
  are heavier than `expf`), ~28× `@parallel`. The in-process `vmath_throughput` probe (kernel vs the
  scalar `libm`-call loop gcc/rustc are forced to emit) confirms it locally: exp ~5.9×, gelu ~5.5×,
  tan ~3.6×, asin ~5.4×, exp10 ~13×, logsigmoid ~4.6×.
- **Two-arg transcendentals → 256-bit AVX2 dispatch (`pow`/`atan2`/`hypot`)**: `pow(x,y) = exp(y·log(x))`,
  `atan2(y,x)` (the full-circle angle — geometry, robotics, complex argument), and `hypot(a,b)` (the
  overflow-safe 2-norm) get a two-input kernel (`wukong_vmath2_f32`): a `for j { out[j] = f(x[j], y[j]) }`
  loop lowers to the **256-bit** AVX2 kernel (the same width jump that ~doubled the single-arg
  activations), and the three kernels mirror the inlined `emit_pow`/`emit_atan2`/`emit_hypot` op-for-op
  (via the shared exp8/log8/atan8/√), so dispatched == composed and the interpreter marshals through the
  identical kernel — interp == native, -O0 == -O3. A composed/scalar use (or a body that fuses with an
  adjacent store loop) still lowers to the inlined ≈1-ULP poly (128-bit auto-vec). Either way a win over
  C's scalar `powf`/`atan2f`/`hypotf` (a loop with the call won't vectorize).
- **Streaming elementwise → 256-bit AVX2 dispatch**: a recognized streaming map
  (`out[i] = act(a·x[i] (+ b·y[i]) + c)`, incl. ReLU/ReLU6) lowers to `wukong_velem_f32` and a Horner
  polynomial to `wukong_vhorner_f32` — both true 256-bit AVX2/FMA, unrolled, emitting **non-temporal
  stores** once the working set spills L3 (the read-for-ownership-skipping store gcc/rustc won't emit).
  This turns the former memory-bound *ties* into wins: saxpy ~1.3×, poly ~1.2× at `N=2²⁰`, widening to
  ~1.3–1.6× at realistic >L3 tensor sizes. A velem **identity-affine fast path** (skip the wasted
  `fma(1·x+0)` for a bare `relu`/copy) plus gating the software prefetch on the DRAM/non-temporal
  regime removed a ~1.2× `relu` regression at L3-resident sizes (now a clean tie there, ~1.4× at >L3);
  `vhorner` runs **six** independent Horner chains (was four — a 5-deep dependent-FMA chain needs ~8 in
  flight to fill both FMA ports), turning the degree-4 poly tie into a consistent win over gcc's own
  256-bit autovec. The interpreter marshals through the identical kernel, so the oracle stays exact.
- **`@parallel` reduction → multicore reduction kernel**: a reduction loop in a `@parallel` function
  (`s += x[k]*y[k]`, `(x[k]-y[k])²`, or `x[k]`) lowers to a **deterministic multicore reduction
  kernel** (`wukong_sreduce_f32_parallel`: dot/ssd/sum/sumsq) instead of a sequential per-thread
  accumulation. The parallel result is bit-identical to the serial one regardless of core count
  (fixed-size chunks, ascending partial combine), and the interpreter calls the serial form, so the
  differential oracle stays exact. Spreads the stream across cores to aggregate memory bandwidth:
  **dot ~7.9×, ssd ~8.6× faster** than single-threaded C (which stays serial & latency-bound).
- **`@parallel`**: loops execute across CPU cores via a rayon runtime, each per-core chunk itself
  vectorized — ~2.2–8× faster than idiomatic single-threaded C on the (memory-bound) elementwise and
  reduction kernels.
- **Mixed-precision (bf16 *and* f16) → SIMD dispatch**: both `bf16` and `f16` are real 2-byte storage
  (round-to-nearest-even on store/cast, f32 compute; bf16 via inline bit-math, f16 via shared
  `half`-crate shims since Cranelift x64 lacks f16 convert lowering). A full, symmetric op suite over
  `[bf16]`/`[f16]` arrays read through an `as f32` widening cast (lossless: `<<16` for bf16, F16C
  `vcvtph2ps` for f16) with f32 accumulate/compute dispatches to half-precision runtime kernels:
  **`dot`/`sum`** (`wukong_{dot,sum}_{bf16,f16}`, ~3–8× vs C), **`max`/`min`/`absmax`**
  (`wukong_reduce_{bf16,f16}` — the per-tensor absmax is the symmetric int8-quant scale), **streaming
  `axpby`** (`wukong_axpby_{bf16,f16}`, half-in/f32-out), and the **36-op activation set**
  (`wukong_vmath_{bf16,f16}`). Precision-generic recognizers; the interpreter marshals through the
  identical kernel, so native == interp bit-for-bit. C/Rust can vectorize neither a `libm` call nor the
  half→f32 widen, so the gap is structural.
- **Arrays**: fixed-size `[T; N]` run end to end — literal/repeat initializers, indexed load/store
  with a runtime index, and array parameters passed by base pointer (out-params work). Real kernels
  (dot product, SAXPY, a flat GEMM) run on the interpreter.
- **Tuples, structs, pointers, `loop`, and constant-shape tensors execute**: tuples (`(a, b)`, field
  access/assign `t.0`, heterogeneous padded fields), structs (`struct S { … }`, literals with fields
  in any order, field access/assign), **nested structs** (struct-in-struct to any depth, an aggregate
  field deep-copied from a variable, arrays of structs, tuple-of-struct), pointers/references
  (`&mut x`, `*p` load/store, a pointer threaded through a call — address-taken locals stay in memory,
  so `-O0` == `-O3`), and `loop { … }` with `break`/`continue` all run end-to-end on both the
  interpreter and the native Cranelift backend. Aggregates lower to a flat padded byte buffer with no
  dedicated aggregate MIR type (the local's value *is* its base pointer, like an array; nested fields
  recurse, a non-literal aggregate field is a leaf-precise deep copy). **Constant-shape tensors** also
  run: a `Tensor[f32, R, C]` parameter passes by base pointer and a multi-dimensional index `a[i, j]`
  flattens to a row-major GEP — the shape-typed surface executing, not just shape-checking. A matmul
  written in that tensor notation (`c[i,j] = Σ a[i,k]·b[k,j]`, both the dot-product and accumulate
  spellings, plus the `b[j,k]` `nn.Linear` `A·Bᵀ` form) **dispatches to the same tuned `wukong_sgemm`
  microkernel** as the flat `a[i*K+k]` spelling — a 2-index operand access supplies its row stride
  from the tensor's inner dimension (gated by `tensor_matmul_is_correct`/`tensor_matmul_accumulate_form`).
  Fixtures `tests/run/{tuple,struct,struct_nested,pointer,loop,tensor_add,tensor_matmul}.wk`; the
  aggregate path is differentially gated by `differential_{tuple,struct,nested_struct}` and pointers by
  `differential_pointer` (native vs interpreter, bit-for-bit). By-value aggregate parameters/returns
  (an sret ABI) now lower too (see the language-surface additions below); **symbolic-generic tensor
  dimensions now execute too** (`fn f<M, N>(t: Tensor[f32, M, N])`, via hidden dim params — see below).
- **Intrinsics**: `print`/`println` (captured stdout) and `assert` (traps on false).
- **Runtime**: a bump `Arena` allocator and a deterministic `parallel_for` — both *reference* surface
  with no caller in the workspace (kernels use thread-local `Vec` scratch, and the native `@parallel`
  lowering target is `wukong_parallel_for`). The runtime's real surface is the dispatched kernel family
  (GEMM/GEMV, vmath, reductions, norms, …). `Arena::alloc` rejects a non-power-of-two alignment rather
  than aliasing a live region.
- **Diagnostics**: rustc-style renderer, a stable error-code catalog with `--explain <CODE>`, and
  `--error-format=json` (JSON Lines).
- **Tooling & tests**: end-to-end run-suite with `// EXPECT-*` directives, an opt-level differential
  test (`-O0` vs `-O1/-O2/-O3`), per-stage `--emit` smoke tests, the `wukong_bench` harness (IR-op
  reduction + `-O0`-vs-`-O3` interpreter speedup, doubling as an optimizer-equivalence gate over
  heavy kernels in `bench/kernels`), a GitHub Actions CI (fmt + clippy + test on Linux & Windows),
  and the language guide and internals docs.
- **Performance**: the interpreter pools per-call register files and passes block-parameter arguments
  through a reused buffer, roughly halving its wall-clock; large `[v; n]` array initializers lower to
  a fill loop instead of unrolled stores.
- **Embedding-lookup dispatch**: the LLM token-id row gather `out[t,:] = weight[ids[t],:]` (over an
  `i32` index array — the first recognized dispatch with an integer *index input*) is recognized and
  lowered to `wukong_embedding_f32[_parallel]` (a 256-bit row copy; bit-exact data movement, mapped
  across the independent output rows under `@parallel` — since 2026-07-29 the parallel kernel is
  selected under a **run-time** byte-range non-overlap test on the two buffers, so an aliased call runs
  the serial kernel, and a slice weight with no compile-time extent keeps it outright).
- **2D pooling dispatch**: the idiomatic 5-deep max/avg-pool nest lowers to
  `wukong_{max,avg}pool2d_f32[_parallel]` (the CNN spatial downsampler; `@parallel` across channels;
  bit-exact — max is idempotent, the avg `(dy,dx)` sum order is fixed). Honest sharp edge: gcc
  auto-vectorizes regular-stride (e.g. 2×2/s2) pooling, so single-core is a tie there, not a win.
  **Superseded (2026-07-29):** dispatch additionally requires all eight dims to be literal and the
  nest's own `OH`/`OW` bounds to equal the full no-padding extent the kernel recomputes; a
  partial-window / sub-region pool declines to the (correct, unaccelerated) scalar nest.
- **xbench**: a broadcast bias-add (`out[r,c] = x[r,c] + bias[c]`) cross-language row, plus a
  **fused FFN** row (`C = silu(A·Bᵀ)` — the real Dense/SwiGLU layer, matmul + activation folded into
  one `wukong_sgemm_nt_epi` C-write; **~24–26× single-core, ~48–95× `@parallel`** vs C, where C pays
  an un-tiled serial-reduction GEMM + a separate scalar-`expf` silu pass) and an **`argmax@parallel`**
  row (**~17× vs single-threaded C** — the multicore global argmax).
- **`match` expressions** (`tests/run/{match_expr,match_patterns,match_tuple}.wk`): integer/bool
  literal, identifier-binding, and wildcard `_` patterns, **or-patterns** `1 | 2 | 3`, half-open
  `0..10` / inclusive `0..=10` **range** patterns, **enum-variant** patterns `Color::Red` (matched by
  discriminant), and **tuple** patterns `(0, _)` (per-field tests + bindings, nesting, and composition
  like `(0 | 1, y)`), each with an optional `if` guard. The scrutinee is evaluated once and the whole
  `match` lowers to an if-else chain; it works in value and statement position. Gated by
  `differential_match`/`differential_match_patterns`/`differential_tuple_match` (native==interp, -O0..3).
- **C-style enums** (`tests/run/enum_cstyle.wk`): `enum Code { Ok = 10, Err = 20 }`, auto-incrementing
  `enum Color { Red, Green, Blue }` (0,1,2), and continue-after-explicit `enum Step { A = 5, B, C }`
  (5,6,7). A variant *is* its integer discriminant — usable in `let`, `==`, `as i32`, and as a `match`
  pattern. **Data-carrying (tagged-union) variants with tuple/struct payloads and payload `match`**
  (with bindings, `if` guards, and literal sub-patterns) **now run too**
  (`tests/run/enum_payload_{tuple,struct}.wk`). Gated by `differential_enum`.
- **Top-level `const` usable as a value** (`tests/run/top_level_const.wk`): sema type-checks each
  initializer against its annotation (an unsuffixed literal adapts) and records it; mir_build inlines
  it at every use site — a bare value, in arithmetic, as an array index, as a loop bound, and when one
  `const` references another (recursive inlining). Gated by `differential_top_level_const`.
- **Tuple destructuring in `let`** (`tests/run/let_destructure.wk`): `let (a, b) = …`, nested
  `let ((m, n), o) = …`, a wildcard `let (keep, _) = …`, and destructuring a tuple-returning call
  result — each sub-pattern binds a view into the initialized tuple buffer (value semantics). Gated by
  `differential_let_destructure`.
- **Nested tuple-field access** (`tests/run/nested_tuple_field.wk`): `t.0.1`, `t.0.0.0`, as reads and
  as assignment targets — the parser splits a lexer-glued `N.M` float in field position into two
  consecutive tuple-field accesses. Gated by `differential_nested_tuple_field`.
- **By-value aggregate parameters and returns** (`tests/run/{struct_fn,struct_return}.wk`): a
  tuple/struct passed by value and a `fn … -> Struct` return are modeled with a MIR-level **sret** ABI
  (a hidden leading pointer, a deep-copy into it, a void return; the call site allocates the
  destination and passes it as the hidden first argument), so no aggregate ever rides in a register and
  both backends agree. A struct param resolves to its registry-aware byte-buffer type (passed by base
  pointer), and `p.x` through a `&`/`*mut` reference auto-derefs. Whole-aggregate **assignment**
  (`s = other;`, `*p = Struct{..}`) now deep-copies leaf by leaf (`tests/run/struct_assign.wk`). Gated
  by `differential_struct_across_fns`/`differential_struct_return`/`differential_struct_assign`.
- **Radix integer literals** (`tests/run/radix_literals.wk`): hex `0xFF`, octal `0o17`, binary
  `0b1010` — with `_` digit separators and an optional type suffix — parse to their real value (they
  previously all evaluated to `0`, since `parse_int` kept only the leading run of decimal digits). Gated
  by `differential_radix_literals`.
- **Char literals** (`tests/run/char_literals.wk`): `'A'` lowers to its `u32` Unicode scalar value,
  with the one-character escapes (`\n` `\t` `\\` `\'` `\0`), `\xHH` hex, and `\u{…}` Unicode escapes
  (the lexer's escape scanner now consumes the multi-byte forms). Gated by `differential_char_literals`.
- **String literals** (`tests/run/string_literal.wk`): `"hello"` materializes its UTF-8 bytes plus a
  NUL terminator into a stack byte buffer (the same by-pointer convention as arrays), typed `*u8`; the
  escapes `\n` `\r` `\t` `\\` `\"` `\'` `\0` `\xHH` `\u{…}` decode, each code point re-encoded as UTF-8.
  `print`/`println` of a `*u8` (a literal, a `let`-bound string, or one returned from a function) is
  routed (type-directed) to a `print_str`/`println_str` path that renders the bytes — the interpreter
  walks its slot memory, native reads the buffer via `rt_print_str` — while numeric `print` still
  prints numbers. No string *type* beyond `*u8` yet (no concatenation/indexing/length, no general
  static-data section). Gated by `differential_string_literals`.
- **Labeled loops** (`tests/run/labeled_loop.wk`): `'label: loop/while/for { … break 'label;
  continue 'label; }` — a labeled `break`/`continue` targets the named enclosing loop, not just the
  innermost. The lexer has a `Label` token disambiguated from a char literal exactly as Rust (`'a'`
  closed by a `'` is a char; `'outer:` is a label); sema tracks an enclosing-label stack, so a
  `break`/`continue` outside any loop **or** one naming an undeclared label is `E0303`
  (`tests/fail/break_unknown_label.wk`); mir_build's loop stack carries each loop's label and resolves
  the branch target. Loop-as-expression / break-with-value (`let x = loop { break 5; };`) **now works
  too**: `loop` is a value-producing expression and `break <v>` carries a value, merged on a typed
  exit-block param like `if`/`match` (`tests/run/loop_break_value.wk`). Gated by
  `differential_labeled_loops`.
- **Slices `[]T`** (`tests/run/slice_basics.wk`): a `[]T` fat pointer `{ data: *T @ 0, len: i64 @ 8 }`
  viewing existing array storage — `s.len()`, indexed read/write `s[i]`, iteration `for x in s`, passing
  to a function (including an array **unsized** to a slice at the call site), and write-through aliasing
  of the backing array. A 16-byte by-pointer aggregate addressed identically by both backends
  (interpreter slot memory == native byte offsets), so interp == native bit-for-bit and `-O0` == `-O3`.
- **Symbolic-generic tensor shapes** (`tests/run/generic_shape*.wk`, `matmul_dynamic.wk`): a
  `fn f<M, N>(t: Tensor[f32, M, N])` runs via hidden per-dimension `i64` params bound under their symbol
  names (stride / matmul-dim / dim-as-value lookups resolve through them), so the shape-typed surface
  executes on **runtime** dimensions with zero backend change; an undeclared dimension is `E0504`.
- **Autodiff reachable from the CLI**: `wukong_autodiff` (a reverse-mode VJP MIR→MIR transform plus a
  fused AdamW kernel, finite-difference-gated) is now wired into `wukongc` — `--emit=grad` prints a
  loss function's backward MIR (`--grad-of=<fn>`, `--grad-wrt=<i,..>`), and `--train` runs a
  forward→backward→optimizer loop (`--train-steps`, `--train-lr`, `--train-opt=sgd|adamw`,
  `--train-seed`) printing the loss trajectory (driver `build_grad` / `run_train`).
- **Correctness fixes** (interpreter↔native divergences and ICEs removed; each gated bit-for-bit):
  - **`break`/`continue` outside any loop** is now a clean **`E0303`** instead of a backend divergence
    — the lowerer left a fallback `unreachable` the interpreter trapped (exit 1) but the native backend
    turned into a SIGILL (exit 132). Registered in the diagnostic catalog (`--explain E0303`);
    `tests/fail/break_outside_loop.wk`.
  - **`&&` / `||` now short-circuit** — they lowered to a bitwise `And`/`Or` of both eagerly-evaluated
    operands (so a side-effecting or guarded RHS always ran, e.g. `n != 0 && 100/n > 0` divided by
    zero); now lowered to control flow. Gated by `differential_short_circuit`.
  - **`continue` in a range `for`** now runs the loop step (it had branched straight back to the header
    without advancing the index — an infinite loop); a dedicated latch block holds the step, on both the
    sequential and `@parallel` per-thread paths. `tests/run/for_continue.wk` (`differential_for_continue`).
  - **float → narrow-int casts** (`1e30 as i8`) no longer ICE the native backend and now **saturate**:
    Cranelift's `fcvt_to_*_sat` can't target a sub-32-bit result, so the backend converts to `i32`
    saturating then clamps/reduces to the narrow range — Rust `as` semantics, matching the oracle.
    `tests/run/float_cast_narrow.wk`.
  - **Deep recursion** in the interpreter no longer aborts the process: `run_with_output` runs on a
    scoped 512 MiB-stack worker thread, so a deeply recursive program returns instead of overflowing the
    ~8 MiB main stack (which had taken the whole differential gate down). Gated by
    `deep_recursion_does_not_overflow_oracle`. **Superseded (2026-07-29):** all *five* public entry
    points run on that worker (four previously skipped it), and *unbounded* recursion is now a
    diagnostic — `interpreter call-depth limit exceeded (likely unbounded recursion)`, exit 1 — rather
    than an eventual overflow of the 512 MiB reservation.
  - **int → f32 casts above 2⁵³** round in one step in the interpreter oracle — it had double-rounded
    `int → f64 → f32`, disagreeing with native's single `fcvt_from_*` by a full ULP. Gated by
    `differential_int_to_f32_rounding`.
  - **Math intrinsics on integer operands** no longer emit float-op-on-int MIR: `abs`/`round`/`floor`/
    `ceil`/`trunc` are now type-preserving (integer `abs` = `select(x<0, −x, x)`, rounding an integer is
    the identity) and `sqrt`/the transcendentals promote an integer operand to `f32` — native had
    rejected the old MIR while the interpreter ran it lossily. `tests/run/int_math.wk`
    (`differential_int_math_intrinsics`).

### Changed
- **Cast precedence fixed**: `*p as T` now parses as `(*p) as T`, not `*(p as T)` (which had
  mis-typed the deref as a `ptrtoint` then a load). `as` binds looser than `*`/unary, tighter than the
  binary operators.
- A construct lowering cannot handle (e.g. explicit SIMD `f32x8` load/store intrinsics) is a hard
  `error[C0001]` instead of a warning, and the driver refuses to optimize, run, or codegen a module
  whose lowering failed — so the compiler never emits or executes invalid MIR.
- **Global argmax/argmin** (`wukong_argreduce_f32`) gained a 256-bit AVX2 path (4 accumulators × 8
  `f32` value + `i32` index lanes, strict-compare + `blendv`, collapsed through the scalar tie-break).
  It was the one memory-bound reduction lacking one, so it had *lost* to gcc's branch-predicted scalar
  loop by 1.30× — now **~5–9× faster**, bit-identical to the scalar lowest-index result.
- **Column argmax/argmin** (`wukong_colarg{max,min}_i32`) rewritten to a single row-major pass with
  the column range's running best kept L1-resident, instead of re-reading the matrix `cols/8` times
  per 8-column band — a former 0.9–1.6× tie/loss became **2.7–5.3× faster** (bit-exactness unchanged).
- **Fused elementwise chains**: the vectorizer now forwards a just-stored intermediate's register value
  to its consumer in the same fused body, dropping the per-element store+reload round-trip (one fewer
  memory stream per chained op; the intermediate need not touch memory between producer and consumer).
- **`@parallel` global argmax/argmin** now dispatches to the multicore `wukong_argreduce_f32_parallel`.
  It previously fell through to the *serial* kernel (the parallel reduction recognizer handles only
  plain `+`/`fmax`/`fmin` reductions, not the argmax `(value, index)` bookkeeping), so global argmax
  never scaled across cores. The parallel kernel folds the same fixed `RCHUNK` decomposition in
  ascending order, so the index is bit-identical to the serial kernel the interpreter calls — **~17×
  vs single-threaded C** (the bandwidth-limited ~2× scaling over the 8.9× single-core fold).
- **Optimizer compile time ~31% faster** (in-process, 400-function `-O2`: 18.0 ms → 12.5 ms), from two
  output-preserving changes — the MIR is bit-identical to before, verified by `optimization_preserves_
  results`, the native-vs-interpreter differential gate, and the `-O0`-vs-`-O{1,2,3}` invariance gate:
  - **CSE value-numbering key** is now a packed, allocation-free `enum` instead of a `format!` string
    built per pure instruction on every pass (CSE is the costliest pass; the key induces the identical
    equality relation, so value numbering is unchanged).
  - **Fixpoint loop** skips passes already at fixpoint: a pass that ran with no change is not re-run
    until another pass mutates the function, which drops the optimizer's final all-passes no-op
    *confirmation* sweep without changing the sequence of mutations.

### Notes
- **Constant-shape** *and* **symbolic-generic** tensors, **C-style `enum`s** (including **data-carrying
  tagged-union** variants), **by-value aggregate parameters/returns**, **slices `[]T`**, and
  **loop-as-value** (`break <v>`) now all execute end-to-end (above). What still only parses and
  type/shape-checks without lowering is **SIMD vector *values*** (explicit `f32x8` load/store
  intrinsics). Native code generation is **Cranelift** (`--emit=obj|exe`, no LLVM); the LLVM path is
  the textual `--emit=llvm-ir` emitter only (see `docs/llvm-setup.md`).
