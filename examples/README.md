# Wukong examples

Run any example through the interpreter (no LLVM required):

```sh
cargo run -p wukongc -- --run examples/<name>.wk
```

Inspect any stdout-printing stage with `--emit=tokens|ast|mir-high|mir|grad|llvm-ir` (only
`--emit=obj` and `--emit=exe` write a file, via `-o`), e.g.:

```sh
cargo run -p wukongc -- --emit=mir -O3 examples/dot.wk
```

`--run` and an explicit `--emit=<stage>` are mutually exclusive: the compiler honours exactly one of
them, so the combination is rejected rather than silently discarding half the work.

## Runnable today (interpreter)

| File                | What it shows                                   | Output            |
|---------------------|-------------------------------------------------|-------------------|
| `hello.wk`         | smallest program with output                    | `42` `42`         |
| `fib.wk`           | recursion, branches, arithmetic                 | `55`              |
| `dot.wk`           | dot product over a fixed-size array             | `120`             |
| `saxpy_array.wk`   | SAXPY `out = a*x + y` over arrays                | `12 24 36 48`     |
| `relu.wk`          | ReLU over an array with an out-parameter        | `0 5 0 0 8 0`     |
| `gemm.wk`          | f32 matmul recognized + dispatched to the tuned AVX2 GEMM | `4 5 10 11` |
| `gpt2.wk`          | GPT-2 decoder block: LayerNorm + multi-head causal attention + GELU MLP + residuals. Readable *shape* spec; measured, it dispatches only the 2 LayerNorms, the 6 non-attention projections and the 2 residual adds — see `gpt2_forward_bench_small.wk` for the full recognized surface | `-1113 1099 677 -264` |
| `llama_block.wk`   | Llama block: RMSNorm + RoPE + multi-head attention + SwiGLU FFN + residuals | `-1227 1204 593 -404` |

The end-to-end test programs under [`../tests/run`](../tests/run) are also runnable and cover
arrays, a flat GEMM, transpose, bubble sort, gcd, casts, floats, and more. Heavier programs for
the benchmark harness live in [`../bench/kernels`](../bench/kernels).

## GPT-2 124M end-to-end inference

The real pretrained OpenAI GPT-2 124M, run as a Wukong program and verified numerically against
HuggingFace. This is a **correctness / capability** result — inference only, external tokenization, no
speed claim (see [../README.md](../README.md) and [../BENCHMARKS.md](../BENCHMARKS.md)).

| File | What it shows | How to run |
|------|---------------|------------|
| `gpt2_infer.wk` | Full GPT-2 124M forward: loads the **real pretrained weights** (124,439,808 params) + prompt token ids from disk via the `read_f32` / `read_i32` file-I/O intrinsics, runs embed → 12 pre-LayerNorm blocks (biased QKV, 12-head causal attention, tanh-GELU MLP, residuals) → final LayerNorm → tied LM head → logits `[5, 50257]`, writes them back to `data/gpt2/gpt2_wuk_logits.bin`, and prints four integers — the argmax of the last position (**1757 " John"** for *"Hello, my name is"*), then three signature values: the top logit ×1000, `logits[0,0]` ×1000, and the element count written. **Native-only** — the interpreter cannot hold 124M params. | export the weights (below), then `wukongc --run --backend=native examples/gpt2_infer.wk` from the repo root; check with `python tools/verify_gpt2.py` (rel 2.05e-6) |
| `gpt2_infer_small.wk` | The **same** forward pass at a reduced, interpreter-runnable config (D=64, H=4, DFF=256, SEQ=8, LAYERS=2) with synthetic in-loop weights and no file I/O — the differential gate that runs **bit-identically on interpreter and native**. Like `gpt2_infer.wk` it uses bare const dims. That used to dispatch **no** recognized kernel at all; since `wukong_mir_build::canon` folds an integer const into its uses before recognition, it now dispatches its projections (measured: 1 `sgemm_nt`, 4 `sgemm_nt_epi`; `gpt2_infer.wk` gains 1 + 6 alongside its 2 `velem_f32`) and gates the scalar *pipeline around them* — the norm/softmax/gelu/LM-head forms stay scalar here and are gated by `gpt2_forward_bench_small.wk`. Its printed logits and argmax are unchanged by that gain. | `cargo run -p wukongc -- --run examples/gpt2_infer_small.wk` |
| `gpt2_forward_bench_small.wk` | The same config again, but spelled in the **recognized dispatch forms** (local `let` dims, per-head repack, `gelu()`, gathered γ/β, bare-index LM-head GEMV) — measured: 6 `sgemm_nt_epi`, 3 `norm_affine_f32`, 2 `velem_f32`, 1 each of `sgemm_nt_alpha` / `sgemm_nt` / `norm_f32` / `vmath_f32` / `sgemv`, the identical set `gpt2_forward_bench.wk` emits. This is the differential fixture for the **kernel-call lowering seam**; its output cross-checks against the all-scalar `gpt2_infer_small.wk` (argmax 15, −14352, 9521). | `cargo run -p wukongc -- --run examples/gpt2_forward_bench_small.wk` |
| `gpt2_forward_bench.wk` | The same 124M forward as `gpt2_infer.wk`, at `s_len = 512` with deterministic in-program token ids and a warmup + min-of-K timing loop, respelled in the **recognized dispatch forms** (model dims bound to local `let`s, per-head repack, `gelu()`, gathered γ/β, bare-index LM-head GEMV) — measured at `-O2`: 6 `sgemm_nt_epi`, 3 `norm_affine_f32`, 2 `velem_f32`, 1 each of `sgemm_nt_alpha` / `sgemm_nt` / `norm_f32` / `vmath_f32` / `sgemv`, the identical set `gpt2_forward_bench_small.wk` emits. Writes the last-position logit row to `data/gpt2/gpt2_bench_lastrow.bin` and prints three integers: the argmax of that row, the min forward time in microseconds, and the repetition count. **Native-only**; it probes the blob with a one-element read first, so with `data/gpt2/` unreachable it prints `-1` and exits 1 identically on both backends. | export the weights (below), then `wukongc --run --backend=native examples/gpt2_forward_bench.wk` from the repo root |
| `gpt2_forward_bench_par.wk` | The all-core companion to `gpt2_forward_bench.wk`: the per-layer block is a `@parallel fn`, so measured at `-O2` the whole-`[S, D]` ops dispatch their multicore twins (6 `sgemm_nt_epi_parallel`, 2 `norm_affine_f32_parallel`, 2 `velem_f32_parallel`, 1 `vmath_f32_parallel`). The final LayerNorm and the last-position LM head live in the non-`@parallel` `main()`, so they stay serial (`norm_affine_f32`, `sgemv`), and the 12-head attention loop is a **serial** chain that fans cores *within* each head (`sgemm_nt_alpha_parallel`, one outlined `wukong_parallel_for` region for the causal-mask write, `norm_f32_parallel`, `sgemm_nt_parallel`) — not across heads. Same prints, same `-1` data guard, same native-only restriction as the serial bench. | `wukongc --run --backend=native examples/gpt2_forward_bench_par.wk` from the repo root (`RAYON_NUM_THREADS=1` forces a single worker for the same-program single-core cross-check) |
| `gpt2_config.wk` | GPT-2 124M layout constants (dims + flat-blob element offsets) imported by `gpt2_infer.wk`, `gpt2_forward_bench.wk` and `gpt2_forward_bench_par.wk`; **generated** by `tools/export_gpt2.py` (the single source of truth for the offset table) — not run directly. `crates/wukongc/tests/gpt2_config.rs` re-derives every constant from the exporter's own `LAYER_TENSORS` order, so a hand edit or an un-regenerated layout change fails `cargo test` without needing torch or the blob. | imported, not run |

The three data-guarded programs (`gpt2_infer.wk` and the two benches) are excluded from the
interpreter-vs-native and `-O` invariance agreement checks *while the blob is reachable*: they are
native-only and two of them print a wall clock, so their stdout is only a valid differential fixture
when the data is absent and both backends take the `-1` early return.

**Obtaining the weight blob.** The weights are large and not committed to the repo. Export them from
HuggingFace `transformers` (needs `torch` + `transformers` + `numpy`):

```sh
python tools/export_gpt2.py
```

This downloads `GPT2LMHeadModel.from_pretrained("gpt2")`, writes the flat little-endian f32 weight blob,
the prompt token ids, the authoritative HuggingFace reference logits and a `MANIFEST.md` (the small
*committed* layout doc — dtypes, byte sizes, and every top-level and within-layer element offset;
force-added past the `/data/` gitignore) under `<repo>/data/gpt2/` (override with `$GPT2_DATA_DIR` or a
positional argument), and regenerates `examples/gpt2_config.wk`. Before reporting success the exporter
runs a mandatory self-check — an independent numpy forward that slices the *just-written* blob at the
*computed* offsets, so it validates the Conv1D transposes and the offset table at once — and either
prints `EXPORTER SELF-CHECK PASS …` or exits 1 with `SELF-CHECK FAILED …`.
Then run `gpt2_infer.wk` on the native backend and verify with `tools/verify_gpt2.py`. Tokenization is
performed by the HuggingFace tokenizer inside the exporter; the `.wk` program only ever consumes the
integer token ids it produces.

`verify_gpt2.py` refuses a `gpt2_wuk_logits.bin` older than an hour (`--max-age-seconds`, or
`--no-age-check` to re-check an archived artifact): `gpt2_infer.wk` exits 1 *without writing* when the
data is unreachable, so a stale artifact would otherwise let the verifier pass a run that never happened.

## Shape-checked, not yet executed

These parse and pass compile-time **shape checking** (the headline feature) but use tensor/SIMD/
parallel constructs that lowering does not handle yet, so compiling or running them reports
`error[C0001]: ... is not yet supported by codegen`. Inspect them with `--emit=ast`:

| File          | What it shows                                         |
|---------------|-------------------------------------------------------|
| `vadd.wk`    | SIMD vector add over shape-typed tensors              |
| `matmul.wk`  | tiled, parallel matmul with `Tensor[f32, M, K]` shapes |
| `softmax.wk` | numerically-stable softmax                            |

Try the shape checker on a mismatch:

```sh
cargo run -p wukongc -- --emit=mir tests/fail/matmul_dim.wk   # error[E0502]
cargo run -p wukongc -- --explain E0502
```
