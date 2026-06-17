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
- **GEMM block sizes** (`MR=6, NR=16, MC=72, KC=256, NC=4080`) are tuned for AVX2 + a typical
  L1/L2/L3 hierarchy. The microkernel keeps 12 `__m256` accumulators (of 16 ymm). AVX-512 is **not**
  used (this CPU lacks it; Cranelift can't emit f32x8 either — `gemm.rs` and `vmath.rs` are the two
  hand-written AVX2 paths that give the compute-bound kernels their 256-bit width).
- **`mercury_vmath_f32` is also a differential contract.** Like the GEMM, the interpreter marshals its
  memory through this exact kernel, so any change to its math changes the oracle too. The AVX2 lanes
  and the scalar twins must stay bit-identical (a test pins this at a non-multiple-of-8 length).
- **Packing reads must be contiguous.** `pack_a`/`pack_b_trans` read each source row contiguously
  over the contraction and scatter into the L1-resident packed panel; the transposed (strided) order
  thrashes cache and dominates runtime. Don't "simplify" them back to row-inner loops.
- **`mercury_sgemm` is the differential contract.** The interpreter marshals its `Value` memory into
  real f32 buffers and calls this exact function, so any change to accumulation order changes the
  oracle too — keep serial and parallel numerically identical (per-(i,j) k-order unchanged).
