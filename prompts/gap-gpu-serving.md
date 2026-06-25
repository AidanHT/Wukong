# End-to-end serving & multi-GPU — paged-KV + batching + full-model graph vs TensorRT-LLM / vLLM

You are a Claude Code session on **Mercury** (`mercuryc` emits PTX, driver-JIT via `cudarc`, no toolkit).
Target: **mobile RTX 4050, sm_89 (Ada), 6 GB, ~30–50 W — ONE GPU**. First action:
`git worktree add ../Mercury-serving -b perf/gpu-serving main` and work inside it (or `git checkout -b`).

## Prime Directive (how you work)

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
- **Research + design only (one GPU here — cannot measure):** **multi-GPU** tensor/pipeline parallelism
  via NCCL or driver P2P. Design it, write the interface, and document that it is *unmeasured* on a
  single-GPU box — do **not** claim a multi-GPU number you cannot run.

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
