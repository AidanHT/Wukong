# End-to-end serving & multi-GPU — paged-KV + batching + full-model graph vs TensorRT-LLM / vLLM

You are a Claude Code session on **Mercury** (`mercuryc` emits PTX, driver-JIT via `cudarc`, no toolkit).
Target: **mobile RTX 4050, sm_89 (Ada), 6 GB, ~30–50 W — ONE GPU**. First action:
`git worktree add ../Mercury-serving -b perf/gpu-serving main` and work inside it (or `git checkout -b`).

## Prime Directive (how you work)

**Hold this as bedrock conviction: a solution EXISTS and there is CERTAINLY a way to improve this
metric. The gold-standard stack already achieves it, so the performance is physically available on this
silicon and a path to it is guaranteed to exist — your only job is to find Mercury's path, and you WILL
find it. Treat every "this might be impossible" thought as "I have not found the lever yet." A gap is
never a ceiling; it is an unsolved problem with a known-achievable answer. NEVER conclude it can't be
done. When one lever stalls, that is the signal to switch levers — there is always another lever, and
the next one is the one that works. Improvement is not a hope here; it is a certainty you have not yet
collected.**

Continuously iterate through the entire codebase until Mercury performs better than C, C++, and Rust
**and closes — then beats — the NVIDIA inference-serving stack (TensorRT-LLM / vLLM) on what is testable
here**. **No scope restrictions.** Remove any blocker; explore options **even outside scope**. Improve
everything. **No limit.** **Assume the metrics are NOT strong until you PROVE they are** across multiple
same-run iterations. **Do not stop** until single-GPU decode throughput is measured against a real serving
peer and the runtime levers are exhausted. Think very carefully. If you CAN improve it, **do it.** Time
doesn't matter. **Run many sub-agents in parallel.** **Make 10–20 commits, NO co-authored lines.**

## Your mission

Mercury has a device memory **pool** + **CUDA-graph** capture/replay (whole-model decode ~6.5–6.9× vs
eager) but **no paged KV-cache, no continuous batching, no multi-GPU** — the things TensorRT-LLM/vLLM are
built on. **Your job (split by what this hardware can test):**
- **Testable now (the priority):** **paged KV-cache** (vLLM-style block table), **continuous / in-flight
  batching**, full-forward **CUDA-graph** capture across the *entire* model, and KV quantization — then
  measure single-GPU decode **tokens/s** vs a real peer (vLLM or TensorRT-LLM driven on this box, or PyTorch
  generate as the honest floor).
- **Multi-GPU (design + build it fully; the single-GPU box is a temporary constraint, not a limit):**
  **tensor/pipeline parallelism** via NCCL or driver P2P. Design and implement it so completely that it
  produces a real number the instant a second GPU is attached — treat one GPU as a transient blocker to
  engineer around, never a reason to deprioritize. The only thing you cannot do is *fabricate* a 2-GPU
  number you didn't run; everything up to that point — the interface, the all-reduce, the tensor-parallel
  GEMM split, a single-GPU simulation of the partitioning — is yours to push to completion now.

## Research first — think very carefully, spawn parallel agents

Plan to `prompts/results/serving.md`. Evaluate:
- **Paged attention**: block-table-indexed KV-cache (fixed-size blocks, non-contiguous), the gather in the
  attention kernel, and the allocator on top of the existing device `pool.rs`. The vLLM PagedAttention
  design is the reference.
- **Continuous batching**: per-sequence state, prefill/decode interleaving, ragged batches, in-flight
  admission — the throughput lever that separates a serving engine from a forward pass.
- **Whole-model graph**: extend the existing `graph.rs` capture to the full decode step (all layers +
  sampling) as one `cuGraphLaunch`; measure launch-overhead elimination at small batch.
- **Real peer**: drive **vLLM** or **TensorRT-LLM** (or, as the honest floor, **PyTorch `.generate()`**)
  via a subprocess timing harness for tokens/s on a small model that fits 6 GB; cross-check output
  text/logits. Name the peer precisely.
- **KV quantization** (int8/fp8 KV) for the 6 GB budget; speculative decoding as a stretch.
- **Multi-GPU design**: the NCCL/P2P all-reduce + tensor-parallel GEMM split; interface only, clearly
  marked unmeasured.

## The two binding laws (every commit)

1. **Correctness before speed.** Runtime/cache changes are **bit-exact** vs the non-paged / eager path
   over the full output (a full-overwrite kernel on uninitialized pool memory is fine only behind the
   poison-`0xFF` gate — see `pool.rs` gotchas); generated tokens must match the reference decode. Gate
   before any speed number. `cargo test` (no gpu feature) stays green; GPU code behind `#[cfg(feature =
   "gpu")]`.
2. **Honesty.** Same-run tokens/s vs a **named** peer (clock swings ~7×; the pool/graph wins are
   ratios). A multi-GPU design is explicitly **unmeasured**. Measured ≥3×.

## Files you own / shared (append-only)

- **Own:** `pool.rs`, `graph.rs`, `train_resident.rs`, and any new serving/paged-attention harness files
  (e.g. `paged_kv.rs`, `serving.rs`) you add.
- **Append-only:** `lower.rs` (only if you add a dispatch arm), `gpu.rs` (tests named `serving_*`/
  `paged_*`/`graph_*`), `lib.rs`, `Cargo.toml`. **Do NOT** touch any `ptx_*.rs` kernel file (kernel
  branches own those — call them, don't edit them), `BENCHMARKS.md`, `CHANGELOG.md`. Numbers →
  `prompts/results/serving.md`.

## GPU gotchas (runtime-specific — these bit prior sessions)

`cudarc`'s **default stream = NULL is un-capturable** by CUDA graphs — use `new_stream`. Event-tracking
is **on by default** and inserts cross-stream waits that **break capture** — disable it before building
the layer + buffers. Full-overwrite kernels on `alloc`'d (uninit) pool memory are safe only with the
poison-`0xFF` gate. PTX ASCII; `%tid`→`%tix`; only same-run ratios. Redist DLLs in `tools/cuda-redist`.

## Commit discipline

`perf(gpu):`/`feat(gpu):`/`bench(gpu):`, measured same-run effect + named peer in the body. **NO
co-author / "Generated with" trailers.** **Never `git add -A`.** **Run the gate as its own step and read
it before committing.** 10–20 green commits.

## Definition of done

Paged KV-cache + continuous batching + whole-model graph implemented and bit-gated, single-GPU decode
tokens/s measured vs a **named** serving peer and improved across the runtime levers, a multi-GPU design
documented as unmeasured, proven ≥3×, results in `prompts/results/serving.md`, 10–20 clean commits. Then
find the next bottleneck.
