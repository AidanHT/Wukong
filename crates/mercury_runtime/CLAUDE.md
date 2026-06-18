# mercury_runtime

The runtime Mercury programs call into: a bump arena allocator, a multicore `parallel_for`, and the
tuned **GEMM microkernels** (`mercury_sgemm*`) the compiler lowers a matmul nest to. The native
backend binds these as JIT symbols; the interpreter calls the *same* functions (marshalling its
abstract memory through real buffers) so the differential oracle stays bit-exact.

## Layout
- `src/lib.rs` — `Arena`, `parallel_for` / `mercury_parallel_for` (rayon), the `bf16` helpers, plus
  unit tests.
- `src/gemm.rs` — f32 GEMM: `mercury_sgemm[_nt][_parallel]`, the 6×16 packed AVX2/FMA microkernel
  (full tiles store straight to C; K loop unrolled ×4), cache-block packing (parallel path reuses one
  pack-scratch allocation across all blocks), and a scalar fallback. Plus `mercury_sgemm_nt_epi` — the
  fused-epilogue `nn.Linear` (`C = act(A·Bᵀ + bias)`): bias-add + activation (identity / ReLU) folded
  into the C-tile writeback on the final K-block, so C is written once (serial-only).
- `src/vmath.rs` — `mercury_vmath_f32(x, out, n, op)`: the **256-bit AVX2/FMA elementwise
  transcendental** kernel (exp/log/tanh/sigmoid/silu/gelu/relu, by `VM_*` op code) — the width
  Cranelift can't emit. 8 lanes/step + a scalar tail; the per-element op sequence mirrors the inlined
  Cephes polys in `mercury_mir_build`, and the scalar twins (`exp1`/`log1`/…) back the tail and the
  no-AVX2 fallback, so every lane of every path agrees.
- `src/reduce.rs` — `mercury_sreduce_f32[_parallel](x, y, n, op) -> f32`: **deterministic f32
  reductions** (dot / ssd / sum / sumsq, by `RED_*` op code). A `@parallel` reduction loop lowers to
  the `_parallel` one. The parallel result is **bit-identical** to the serial one: the array is cut
  into fixed-size `RCHUNK` chunks (count independent of thread count), each reduced by the identical
  per-chunk function, and partials summed in ascending chunk order (rayon's indexed `collect`). So
  serial == parallel == interpreter on any machine. AVX2 single accumulator (memory-bound at N=2^20,
  so one is enough) + a scalar tail/twin that matches lane-for-lane (`mul_add` == `fmadd`).
- `src/norm.rs` — `mercury_norm_f32[_parallel](x, out, rows, cols, eps_bits, op)`: **fused
  single-pass row-wise normalizations** (softmax / LayerNorm / RMSNorm, by `NORM_*` op code) over the
  last axis of a `[rows, cols]` matrix. Memory-bound, so the win is fusing the 2–3 passes (each row
  loaded once) + 256-bit AVX2 — and softmax reuses `vmath`'s `exp8`/`exp1`. `eps` rides in as
  `f32::to_bits()` in an i64 to keep the all-integer dispatch ABI. Rows are independent so
  `_parallel` just maps the per-row routine across rows: serial == parallel bit-for-bit, no
  cross-row combine. Within a row the reductions use the same fixed 8-lane accumulator + horizontal
  combine in the AVX2 and scalar paths, so they agree bit-for-bit (twin test across tail sizes). `x`
  and `out` may alias (in-place).
- `src/i8gemm.rs` — `mercury_i8gemm_nt[_parallel](a, b, c, m, k, n)`: **int8 quantized `nn.Linear`**
  `C = A·Bᵀ` (`u8` activations × `i8` weights → `i32` accumulator), the QNNPACK/oneDNN layout. A's row
  and B's row are both contiguous over `K`, so each `C[i,j]` is a dot. Three tiers, detected **once
  per call** (not per element): **AVX-VNNI** `vpdpbusd` (`_mm256_dpbusd_avx_epi32` — one instruction
  folds 32 `u8×i8` products into the 8 `i32` lanes *and* accumulates; the path gcc `-march=native`
  takes, so it's what beats it), else **AVX2** widen+`vpmaddwd` (`_mm256_madd_epi16`, 16/step), else
  scalar. Both SIMD tiers are **register-blocked four B-rows at a time** (`dot4_i8_{vnni,avx2}`): the
  A-row chunk is loaded once per step and reused across the four dots (4× less A traffic) with four
  independent accumulator chains for ILP and an in-register horizontal sum (no per-`(i,j)` stack
  round-trip). **No reassociation exception**: `i32` add is associative mod 2³² (wrapping) and
  `vpdpbusd`/`vpmaddwd` are non-saturating, so every order gives the *same bits* — the fused kernel
  equals the naive `s += a[k]*b[k]` loop exactly (twin tests: scalar==avx2==vnni across K boundaries,
  plus an overflow case). Rows independent → `_parallel` maps per-row across cores, serial ==
  parallel. Measured **~1.3–1.9× faster than gcc single-core** (`-O3 -march=native`, which also uses
  `vpdpbusd`) — the lead widens with size as C spills cache — and **~6.7–11.5× with `@parallel`**.

## Key types & entry points
- `Arena` (`src/lib.rs`) — bump allocator over an owned `Vec<u8>`. API: `with_capacity`, `alloc(size, align)`, `slice_mut(offset, len)`, `reset`, `used`, `capacity`.
- `Arena::alloc` — rounds `offset` up to `align` (power of two; `debug_assert`ed), bumps, returns the **byte offset** (not a pointer) or `None` if exhausted. `checked_add` guards size overflow. On exhaustion `self.offset` is left unchanged.
- `parallel_for(lo, hi, step, body)` — sequential reference order (the interpreter uses it).
- `mercury_parallel_for(n, body, env)` (`#[no_mangle] extern "C"`) — the real multicore dispatch the
  native `@parallel` lowering calls; splits `[0,n)` into one chunk per CPU via rayon.
- `mercury_sgemm(a,b,c,m,k,n,beta)` and `mercury_sgemm_nt` (`C = A·Bᵀ`, nn.Linear), each with a
  `_parallel` variant — `#[no_mangle] extern "C"`. `beta`: 0 overwrites C, 1 accumulates. AVX2/FMA
  (runtime-detected) with a scalar fallback; the parallel ones pack A/B once per K block and run the
  C tile grid across cores. Per-(i,j) accumulation order matches the serial kernel, so serial,
  parallel, and the interpreter all agree bit-for-bit.

## Connects to
Upstream (depends on): `rayon` only (no `mercury_` crates). Downstream (consumers): the interpreter
backend and `mercury_codegen_cranelift` (binds `mercury_*` as JIT symbols), and native-compiled
programs (link as a static lib).

## Gotchas
- `alloc` returns a `usize` offset into the backing buffer, not a raw pointer; pair it with `slice_mut` to read/write bytes.
- `slice_mut(offset, len)` does **not** validate that the region was previously allocated — it indexes `buf[offset..offset+len]` and panics on out-of-bounds. Pass back exactly what `alloc` returned.
- `reset` frees everything at once (offset -> 0); no per-object free, and storage is reused, never grown (`capacity` stays constant across resets).
- `parallel_for` (the closure form) is the **sequential** reference order; `mercury_parallel_for`
  (the C-ABI form) is the **rayon** multicore dispatch. They must agree on observable results for
  data-parallel bodies (the interpreter runs the sequential one, native the parallel one).
- **GEMM block sizes** (`MR=6, NR=16, MC=144, KC=256, NC=4080`) are tuned for AVX2 + a typical
  L1/L2/L3 hierarchy. The microkernel keeps 12 `__m256` accumulators (of 16 ymm). `MC=144` (a ~144 KB
  A-block) measured the sweet spot — bigger re-streams the B-panel from L3 fewer times, smaller leaves
  L2 headroom for the streaming B-block; 216/288 both regressed. Re-measure if you change `KC`/`NR`. AVX-512 is **not**
  used (this CPU lacks it; Cranelift can't emit f32x8 either — `gemm.rs` and `vmath.rs` are the two
  hand-written AVX2 paths that give the compute-bound kernels their 256-bit width).
- **`mercury_vmath_f32` is also a differential contract.** Like the GEMM, the interpreter marshals its
  memory through this exact kernel, so any change to its math changes the oracle too. The AVX2 lanes
  and the scalar twins must stay bit-identical (a test pins this at a non-multiple-of-8 length).
- **`mercury_sreduce_f32` parallel must equal serial bit-for-bit.** The interpreter calls the *serial*
  form; native `@parallel` calls the *parallel* form — the differential gate compares them, so they
  must agree exactly. That rests on the fixed `RCHUNK` decomposition and ascending partial combine
  being independent of thread count: don't make chunk size depend on core count, don't reduce partials
  in completion order, and keep the AVX2 path's store-then-tail identical to the scalar twin. A test
  pins serial==parallel and scalar==avx2 across sizes with partial chunks and non-mult-of-8 tails.
- **Packing reads must be contiguous.** `pack_a`/`pack_b_trans` read each source row contiguously
  over the contraction and scatter into the L1-resident packed panel; the transposed (strided) order
  thrashes cache and dominates runtime. Don't "simplify" them back to row-inner loops.
- **`mercury_sgemm` is the differential contract.** The interpreter marshals its `Value` memory into
  real f32 buffers and calls this exact function, so any change to accumulation order changes the
  oracle too — keep serial and parallel numerically identical (per-(i,j) k-order unchanged).
