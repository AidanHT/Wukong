# mercury_xbench

Standalone binary `mercury-xbench`: an **honest** cross-language benchmark. For each kernel it builds
the *same* computation three ways — Mercury (Cranelift-JIT native), C (`gcc -O3 -march=native`), and
Rust (`rustc -O -C target-cpu=native`) — and times all three through one identical Rust harness over
the same buffers. C/Rust are compiled to shared libraries and called via their C ABI; Mercury is
JIT-compiled in-process. Results and methodology live in `BENCHMARKS.md`.

## Layout
- `src/main.rs` — entire crate: kernel sources (Mercury/C/Rust string builders), `bench_mercury` /
  `bench_external`, `time_ns` (best-of-many batches), the elementwise kernel table, `bench_matmul` /
  `bench_linear` (GFLOP/s + roofline %), `bench_conv` (im2col+GEMM vs direct), `bench_norm`
  (fused softmax/LayerNorm/RMSNorm vs per-row C/Rust), and `bench_i8gemm` (int8 `u8×i8→i32`
  `nn.Linear` GOP/s, with the int8 `bench_mercury_i8` / `bench_external_i8` over the `MeasureI8`
  struct and the `(u8, i8, i32)` `I8KernelFn` ABI).

## Key types & entry points
- `main` — runs the elementwise kernel table (saxpy/dot/relu/poly + `@parallel` variants incl.
  relu6), prints per-kernel compile/runtime/GB-per-s and a geomean, then `bench_matmul`,
  `bench_linear`, `bench_conv`, `bench_norm`, and `bench_i8gemm`.
- `bench_i8gemm` (+ `mer_i8gemm`/`c_i8gemm`/`rust_i8gemm`) — int8 quantized `nn.Linear` (`C = A·Bᵀ`,
  `u8`×`i8`→`i32`) at 512²/1024². The Mercury source is the `ijk` dot-product the `mir_build`
  recognizer folds to one `mercury_i8gemm_nt[_parallel]` call (AVX-VNNI `vpdpbusd`); C/Rust are the
  idiomatic naive int8 GEMM at `-O3 -march=native` (gcc auto-vectorizes to `vpdpbusd` too — the
  honest baseline). **Integer arithmetic, so the cross-language check is bit-exact** (a stronger bar
  than the f32 kernels' tolerance); inputs stay within `i32` (no overflow at these sizes). Reported as
  int8 GOP/s (2 ops per MAC) + compile ms.
- `bench_norm` (+ `mer_norm`/`c_norm`/`rust_norm`) — fused row normalizations over a `1×cols` feature
  row (cols ∈ {768, 4096}). The Mercury source is the copy-then-in-place form the `mir_build`
  recognizer folds to one `mercury_norm_f32` call; C/Rust are the strong cache-friendly per-row
  baselines at honest default flags (no `-ffast-math`, so their float reductions stay sequential —
  same basis as the `dot` kernel). All three copy `x`→`out` then normalize in place (identical work),
  so the full-buffer cross-check is valid. softmax also pits Mercury's vectorized `exp` vs scalar
  `expf`.
- `KernelFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32)` — the shared `(x, y, out)` ABI;
  matmul reuses it as `(a, b, c)`. `N = 1<<20` elements; matmul is 512×512.
- `bench_mercury` parse→sema→lower→`optimize(_,3)`→`jit_module`, times `kbench`; `bench_external`
  shells out to the compiler, loads the `.dll` via `libloading`, times the symbol. Both return a
  `Measure { compile, ns_per_call, checksum }`.
- A per-kernel **checksum** is cross-checked across languages — a miscompiled kernel is caught.

## Connects to
Upstream: `mercury_span`, `mercury_parser`, `mercury_sema`, `mercury_mir_build`, `mercury_opt`,
`mercury_codegen_cranelift` (JIT), plus `libloading` (load C/Rust shared libs). Leaf binary; nothing
depends on it.

## Gotchas
- **Fairness:** Mercury now contracts `x + y*z` to a fused multiply-add, so gcc gets its *default*
  `-ffp-contract=fast` (the old `-ffp-contract=off` suppressed C's natural FMA). Both Mercury and C
  fuse; idiomatic Rust does not contract unless written with `f32::mul_add`, so the Rust column shows
  rustc's default — a real defaults difference, not a handicap. The `@parallel` rows compare
  Mercury's automatic SIMD+multicore lowering against *idiomatic single-threaded* C/Rust.
- **Variance:** the all-core kernels (and matmul) vary a lot on a busy box. `time_ns` takes the min
  of many batches (least-interfered estimate); run the built exe directly (`target/release/
  mercury-xbench.exe`) and several times. Report ranges, not single numbers.
- Needs `gcc` and `rustc` on PATH; `CC` overrides the C compiler. A missing compiler prints a notice
  and that column shows `n/a` rather than failing the run.
- The `__declspec(dllexport)` on the C kernel is what exports `kbench` for `libloading` on Windows.
