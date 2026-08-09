# `docs/gpu/` — the datacenter retarget's working documents

The plan itself is [`GPU_RETARGET_PLAN.md`](../../GPU_RETARGET_PLAN.md) at the repo root. This
directory holds what the plan consumes and produces.

| | What |
|---|---|
| [`derive/`](derive/) | The Wave-0 derivation dossiers (D1–D6) and the predict-before-measure documents (P1, P2). §0's *derive before you rent* rule lives here: everything in it cost $0 and touched no rented silicon. |
| [`phase1-runbook.md`](phase1-runbook.md) | The operational checklist for a rented-GPU session, step by step, with the commands. |
| §"The benchmark instrument" below | What the code does so a number is allowed to be published at all. |

Raw round logs and session ledgers live under [`bench/gpu/<device>/`](../../bench/gpu/), not here.

---

## The benchmark instrument

`crates/wukong_codegen_gpu/src/bench_instrument.rs` implements `GPU_RETARGET_PLAN.md` §6.1–§6.2.
Before it existed, the provenance block lived in a PowerShell runbook and the control column lived in
a human's memory. Both are now code, because this project has retracted four published claims and
**every one traced to an instrument that could not resolve what it claimed**.

### What it refuses, and where the refusal lives

| Refusal | Enforced by |
|---|---|
| The device is a MIG slice, a vGPU profile, an SM count off spec, or a part the table does not know | `Round::open` returns `Err` — **there is no `Round`, so there is no number**. Marketplaces have sold virtualized "A100"s. |
| The round was never closed, or has no `nvidia-smi` reading at one end | `Round::gate` |
| The SM clock moved more than ±5% between the two readings | `Round::gate` — that was not one machine throughout |
| Arms `A` and `C` produced identical sample vectors | `analyze` → `Refusal::AliasedArms`; two real timings are never bit-identical, so this means one handle was used twice |
| An arm skipped a round | `Refusal::MalformedField` — and it blocks the bench's *healthy* fields too, because the instrument failed |
| The measured control floor exceeds the pre-registered bar | `Refusal::FloorAboveBar` — a wide tie is **not** a tie |
| The cell's effect does not clear the floor | `Cell::Tie`, published as a tie |

`Published` has no public constructor. The only way to obtain one is `Round::publish`, which runs
every check above. `Round::cell` is the always-safe form: it returns the number, or `--` **and the
reason there is no number**.

`BenchVerdict`'s numeric fields stay public because the round log has to print them — an unresolved
or degenerate cell is a finding, not a gap. They are not the sanctioned path to a number. The three
outputs that carry their own verdict are `BenchVerdict::table()`, `Round::cell()` and
`Round::publish()`; reading `field(..).effect` and printing it bare is how a gate turns back into a
convention.

### The arithmetic (pre-registered, and identical to the 4050 identity leg)

- Three arms per round: `A` baseline, `C` the twin of `A`, `B` the contender. The leading arm
  **rotates** every round, and at least one warm-up round is discarded.
- Summary across rounds is the **median**, never the minimum — a minimum picks the worst sample of a
  rate field and is meaningless for a printed ratio field.
- `floor = max |C/A − 1|` over the round's *live* fields. A field that never varies is excluded: left
  in, it would pin the floor at zero and let any blip "clear" it.
- A bench publishes only if `floor ≤ bar` (default ±5%, `DEFAULT_CONTROL_BAR`).

### The twin, and the two ways it can be faked

1. **One module key for both arms.** `Gpu::function` caches on the `&'static str` key alone, so two
   arms sharing a key are one `CUmodule` and one `CUfunction`: every row ties perfectly having
   compared nothing. `PtxTwin::new` refuses equal keys — the analogue of `tools/perf/gpu_ab.ps1`
   sha256-ing its two arm binaries, which it does because the two arms legitimately share a cargo
   hash in their *filename*.
2. **A cache that makes one arm warm.** Byte-identical PTX means both arms share one on-disk cubin
   entry, so arm `A` compiles and arm `C` reads the file back. `PtxTwin::prime` moves that asymmetry
   entirely outside the timed region by loading *and warming* both handles before anything is timed;
   `time_arm` then measures steady state only. A cold device-keyed cubin cache once read a **27x**
   false regression here (`derive/p2-expected-4050-identity.md` §5.1).

**Module-load latency has no in-process twin, and that is measured.** The first draft of this module
claimed a direct PTX JIT was symmetric because it bypasses both Wukong caches. Its own device gate
disproved that on the 4050, and re-running the experiment on a second day and a second driver
reproduced it — three consecutive runs each time, byte-identical PTX, a discarded warm-up load:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| driver 5xx, first sitting | `A 0.211 / C 0.160 ms` −24.2% | −24.0% | −27.3% |
| driver 592.82, re-run | `A 0.409 / C 0.294 ms` −28.3% | −31.6% | −20.5% |

Arm `C` reads a **20–32% "improvement"** on work byte-identical to arm `A`'s, and the effect does not
saturate after one warm-up: every load of the text in a process is faster than the one before it. The
*driver* keeps a JIT cache of its own (`~/.nv/ComputeCache`, `CUDA_CACHE_DISABLE=1`) — the first load
of a never-before-seen text measured 8.8 ms here against ~0.3 ms once warm — plus per-context state
that the second load does not re-pay. `PtxTwin::time_module_load` refuses the second call in a
process; a load-latency claim is one measurement per process, alternated across processes.

The experiment is kept as an `#[ignore]`d diagnostic gate so the refusal is checkable, not merely
asserted:

```sh
cargo test -p wukong_codegen_gpu --features gpu --lib \
    the_in_process_load_twin_is_asymmetric -- --ignored --nocapture
```

### The peer twin

§6.2's "cuBLAS called twice" needs no new machinery: `run_twin` hands its `reference` closure to
**both** `A` and `C`, so a peer round gets the peer's own noise floor (`C/A`) and our ratio against it
(`B/A`) in the same round.

### The device spec table

`DEVICE_SPECS` carries the published SM count and compute capability of every part in the plan's
ladder, each row recording the arithmetic it came from (`AD102, 18176 cores / 128`). Matching is over
name *tokens* with longest-match-wins, so `L4` cannot match an `L40S` and the **114-SM H100 PCIe** is
not mistaken for the 132-SM SXM part. A part not in the table refuses: adding one is a code change,
reviewable, on purpose — an env var that let an operator declare the expected SM count would hand the
check back to the human it exists to replace.

### Adopting it in a bench

```rust
use crate::bench_instrument as bi;

let mut round = match bi::open_round("gemm_throughput", g) {
    Ok(r) => r,
    Err(why) => { eprintln!("[skip:provenance] {why}"); return; }   // publishes nothing
};
eprintln!("{}", round.provenance().header());

let plan = bi::TwinPlan::default();
let s = bi::run_twin("gemm_throughput", &plan,
    || vec![bi::read("gemm_ms", peer_gemm_ms())],     // A and C: the same work, twice
    || vec![bi::read("gemm_ms", ours_gemm_ms())]);

round.close(bi::query_smi());
let v = bi::analyze(&s, plan.bar);
eprintln!("{}\n{}", v.table(), round.header());
eprintln!("gemm_ms -> {}", round.cell(&v, "gemm_ms"));
```

`bi::machine_floor(g, &plan)` is the cheap triage: it runs the built-in probe twin and reports what
the *machine* can resolve today, in a couple of seconds. If that already exceeds the bar, no bench in
the round can publish, and it is far cheaper to learn it now than after forty minutes of sweeps.

**One borrow trap, with a one-line fix.** `run_twin` takes two closures, so if both arms need
`&mut Gpu` inside the timed region the borrow checker refuses them (E0499). The fix is `run_rotated`,
which takes a *single* closure that matches on the arm — the borrow then happens once. Reach for it
whenever an arm calls `Gpu::function` or another `&mut self` method while being timed; it is how
`machine_floor` drives the built-in twin. `run_twin` fits the shape every existing `gpu.rs` bench
already has: resolve the `CudaFunction` handles first, then time under a shared borrow.

### Environment

| Variable | Effect |
|---|---|
| `WUKONG_GPU_PROVIDER` / `WUKONG_GPU_SKU` | Recorded in the provenance header |
| `WUKONG_GPU_USD_PER_HR` | Turns the round's wall time into the §6.6 cost line |
| `WUKONG_GPU_CLOCK_LOCK` | `locked:1410` / `locked` / `unlocked`. **Declared metadata**: the before/after clocks are the evidence, and a declared lock the readings contradict is reported as a contradiction |
| `CUDA_CACHE_DISABLE=1` | Disables the *driver's* JIT cache; only relevant to a load-latency measurement |

### What the header prints, and one honest blank

Device name, compute capability, SM count, the spec-table verdict, opt-in SMEM, L2, VRAM, the decoded
driver CUDA API version, the kernel-mode driver string from `nvidia-smi`, provider/SKU, clock-lock
status, clocks/temp/power **before and after**, the drift against the bar, wall time, an estimated
cost, and the round-level gate verdict.

The **CUDA runtime** line reads `none linked`. That is not a gap: this backend is driver-API only —
it JITs PTX through `cuModuleLoadData` and links no CUDA runtime, which is exactly why it needs no
toolkit. §6.1 asks for the field, so the field is printed with the true answer rather than omitted.
