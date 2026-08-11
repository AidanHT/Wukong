# WAVE 4 DERIVATION -- the epilogue as a product surface

**Scope.** `ACT2_WAVE_PLAN.md` wave 4: fused GEMM epilogues on the wgmma kernel, scored against the
cuBLASLt fused-epilogue peer landed in W2 (`baselines.rs` Tier B). This dossier derives what may be
built, what may be *claimed*, and the exact laws that keep the second from outrunning the first.

**The one sentence.** Fusion beats the peer only where the peer *cannot* fuse; for the six epilogues
cuBLASLt does fuse, a Wukong row is a GEMM-parity fight we currently lose at 54.8-82.2% of cuBLAS,
and the wave plan's refusal clause is exactly right to forbid publishing it as a fusion win.

**Provenance of every number below.**

| source | what it gives |
|---|---|
| `bench/gpu/h100/2026-08-10-h100-act2-wgmma-vs-cublas.log` | round 1: all seven shapes, peer ms and ours ms, un-clustered W1 |
| `bench/gpu/h100/2026-08-10-h100-act2-r3-bmulticast.log` | round 3: sq2048/sq4096/sq8192 at 12 configs; the B-multicast primary |
| `crates/wukong_codegen_gpu/src/baselines.rs:3194-3955` | the cuBLASLt fused-epilogue peer (READ-ONLY here) |
| `crates/wukong_codegen_gpu/src/ptx_wgmma.rs:3657-3712` | the current wgmma epilogue (READ-ONLY here) |
| `crates/wukong_codegen_gpu/src/ptx_wmma.rs:1159-1210` | `Act::{None,Relu,Silu,Gelu}` -- shipped epilogue PTX to port |

**Standing caution carried into every section.** The campaign has never measured HBM bandwidth on
H100: `gpu::tests::hbm_bandwidth` is `#[ignore]`d and shows `ignored` in both
`2026-08-10-h100-s2b-device-suite.log:387` and `-s2d-full-suite.log:394`. Every break-even here
divides by a bandwidth taken from the SXM5 spec sheet (3.35 TB/s HBM3). Section 4 makes measuring it
a mandatory row of the one H100 visit, because the whole wave's publishable margin is a function of
that denominator.

---

## 1. EPILOGUE ENUMERATION

`CUBLASLT_FUSABLE_EPILOGUES` (`baselines.rs:3251`) is the complete fusable set of `cublasLtMatmul`:
sixteen values, pinned to `cudarc`'s `cublasLtEpilogue_t` value-by-value by
`the_cublaslt_fusable_set_is_exactly_these_sixteen` (`baselines.rs:5284`). Semantics below are the
`cublasLt.h` doxygen text, quoted from the header, not paraphrased from a blog.

### 1.0 The two facts that decide every row

**Fact A -- the bias axis coincides.** `CUBLASLT_EPILOGUE_BIAS`: *"Bias vector length must match
matrix D rows, it must be packed. Bias vector is broadcasted to all columns and added before applying
final postprocessing."* Under Wukong's `C[MxN] = A[MxK] * B[NxK]^T` -> `C^T[NxM] = B^T * A` mapping,
the rows of cuBLASLt's D are Wukong's **N**. So a Wukong bias -- one value per output column -- is
cuBLASLt's per-row bias of length N with no reshape and no transpose. `baselines.rs:3221-3228` states
this and `the_fused_reference_adds_bias_along_n_not_m` (`baselines.rs:5366`) is its gate. The wgmma
epilogue's own column index `%colb = ((lane & 3) << 1) + %ctan` (`ptx_wgmma.rs:3675`) is the same
axis, so a fused wgmma bias indexes by `%col`, never by `%row0/%row1`.

**Fact B -- the wgmma D-fragment layout is NOT opaque, and that is worth more than it sounds.** The
Act-1 wmma bias epilogue (`ptx_wmma.rs:1490-1528`, `:1913-1952`) has to *round-trip the whole
accumulator through shared memory* -- `wmma.store.d` into a per-warp 16x16 f32 scratch, then re-read
each element by explicit `(row, col)` -- purely because `wmma`'s fragment-to-(row,col) map is
architecture-defined and opaque. `wgmma`'s is documented and the epilogue already computes it in
closed form (`ptx_wgmma.rs:3661-3676`): warp `w` of the warpgroup holds rows `w*16 + lane/4` and
`+ 8`, register group `j` covers columns `8j + 2*(lane&3)` and `+1`. **A wgmma bias epilogue is a
pure register operation.** The port of Act-1's epilogue to wgmma therefore *drops* its most
expensive component; do not port the SMEM scratch with it.

### 1.1 The table

Columns: **(a)** expressible in `.wk` today; **(b)** the wgmma-epilogue PTX; **(c)** cost on top of
the current epilogue (SMEM bytes / live registers per consumer thread / extra HBM bytes per output
element). `nacc = bn/8 * 4 = 128` accumulators per consumer thread at `bn = 256`.

| # | cuBLASLt epilogue (value) | (a) `.wk` today | (b) wgmma-epilogue PTX | (c) SMEM / regs / bytes |
|---|---|---|---|---|
| 1 | `DEFAULT` (1) -- plain GEMM | yes; any matmul nest -> `wukong_sgemm_nt` | today's epilogue, unchanged | 0 / 0 / 0 |
| 2 | `RELU` (2) -- `max(x,0)` | yes; `out[i*N+j] = fmax(s, 0.0)` -> `nt_epi` act=1, bias=None | `max.f32 %accX,%accX,0f00000000;` x nacc. Shipped verbatim: `ptx_wmma.rs:1183` | 0 / **0** / 0 |
| 3 | `BIAS` (4) -- `x + bias[col]` | yes; `nt_epi` act=0 with bias. `tests/run/gemm_fused_bias_slice.wk` | stage `bias[bn]` in SMEM once per CTA (256 threads, one `f32` each, one `bar.sync`), then per j: `ld.shared.v2.f32 {%b0,%b1},[..]` + 4 `add.f32` | `bn*4` = **1024 B** / 2 f32 + 1 addr / 0 |
| 4 | `RELU_BIAS` (6) | yes; `tests/run/linear_bias_relu.wk`, `linear_bias_relu_store.wk` | rows 3 then 2, in that order (bias first -- `apply_f64` at `baselines.rs:3380` pins it) | 1024 B / 2 / 0 |
| 5 | `GELU` (32) -- **tanh** form | yes; `gelu(s)` with no bias -> `nt_epi` act=2 | 8 instructions, 1 of them MUFU: `mul,mul,fma,mul,tanh.approx.f32,add,mul,mul`. Shipped verbatim with identical constants: `ptx_wmma.rs:1194-1203` | 0 / **2** (`%act0,%act1`) / 0 |
| 6 | `GELU_BIAS` (36) | yes; `tests/run/linear_bias_gelu.wk` | rows 3 then 5 | 1024 B / **4** / 0 |
| 7 | `RELU_AUX` (130) -- relu + a **bit-mask** matrix | **NONE** | `setp.gt.f32` per acc, then a pack. **The trap:** the D-fragment gives one lane the *non-adjacent* column pairs `8j+2t, 8j+2t+1` for two rows, so `vote.sync.ballot.b32` across a warp does **not** produce a packed row-major mask. A correct pack needs an SMEM bit-transpose or per-lane `st.global.b8` of 2-bit fragments with read-modify-write. See 1.2 | `bm*bn/8` = **4096 B** / 2 / **+0.125 write** |
| 8 | `RELU_AUX_BIAS` (134) | **NONE** | rows 3 + 7 | 5120 B / 4 / +0.125 write |
| 9 | `DRELU` (136) -- relu gradient | **NONE** | `ld` mask bit, `selp.f32 %accX,%accX,0f00000000,%p` | 4096 B / 2 / **+0.125 read** |
| 10 | `DRELU_BGRAD` (152) | **NONE** | row 9 plus a **column reduction over M**, which no single CTA owns -- needs `red.global.add.f32` or a second pass | +`bn*4` / 2 / +0.125 read, +atomics |
| 11 | `GELU_AUX` (160) -- gelu + the **pre-activation matrix** | **NONE** | a second `st.global` of the *pre*-activation accumulator, at f32 (`AUX_LD` must be divisible by 8) | 0 / 0 / **+4 write** (f32) or +2 (f16) |
| 12 | `GELU_AUX_BIAS` (164) | **NONE** | rows 3 + 11 | 1024 B / 4 / +4 write |
| 13 | `DGELU` (192) -- gelu gradient | **NONE** | reads the aux pre-activation `x`, computes `dy * gelu'(x)`. Tanh-form derivative: `u = c0*(x + c1*x^3)`, `t = tanh(u)`, `gelu' = 0.5*(1+t) + 0.5*x*(1-t*t)*c0*(1 + 3*c1*x*x)` -- ~12 instructions, 1 MUFU | 0 / **3** / +4 read |
| 14 | `DGELU_BGRAD` (208) | **NONE** | rows 13 + the column reduction of row 10 | +`bn*4` / 3 / +4 read, +atomics |
| 15 | `BGRADA` (256) -- bias grad from A, reduced over K | **NONE**, and **structurally out of reach**: it is a reduction of the *A operand* over K, not a function of the C tile the epilogue holds | n/a |
| 16 | `BGRADB` (512) -- ditto from B | **NONE**, same reason | n/a |

### 1.2 The RELU_AUX bit-mask is the sharpest item in the table

cuBLASLt's own constraint discloses the shape: `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD` -- *"Leading
dimension for epilogue auxiliary buffer. ReLu bit-mask must be divisible by 128; GELU input must be
divisible by 8."* 128 **bits**, i.e. a 16-byte granule per mask row. Our fragment hands each lane two
columns of one row-pair; a 32-lane ballot therefore interleaves 8 different rows into one 32-bit
word. Two honest options:

* **Do not implement it.** `DRELU` is the only consumer and `DRELU` is in cuBLASLt's fusable set, so
  the whole RELU_AUX/DRELU pair is a parity fight even if built.
* **Use GELU_AUX's shape instead** -- a full second tensor of pre-activation values, f16, at
  `+2 bytes/element`. It costs 16x the bytes of a bitmask and zero cleverness, and it is what a
  Wukong backward pass would want anyway, because `dgelu` needs the *value*, not a predicate.

### 1.3 The five epilogues cuBLASLt has NO member for -- the actual Wave-4 product surface

`the_cublaslt_fusable_set_is_exactly_these_sixteen` asserts the *absence* half explicitly
(`baselines.rs:5297`): no `SILU`, `SWISH`, `RESIDUAL`, `F16`, `BF16`, `FP8` anywhere in a member
name.

| form | `.wk` today | wgmma PTX | SMEM / regs / bytes |
|---|---|---|---|
| **SiLU / swish** `x*sigmoid(x)` | **yes** -- `tests/run/linear_silu.wk` (null bias), `linear_silu_store.wk`; `nt_epi` act=3 | 5 instructions, **2** MUFU: `mul,ex2.approx.f32,add,rcp.approx.f32,mul`. Shipped verbatim: `ptx_wmma.rs:1187-1192` | 0 / **1** / 0 |
| **SiLU + bias** | yes -- the recognizer composes bias with any act code | rows 3 + SiLU | 1024 B / **3** / 0 |
| **residual + act** `act(A*B^T + b) + R` | yes at the source level (`tests/run/linear_residual_relu.wk`) but it **never reaches a GPU kernel** -- see section 5 | a second base pointer, `ld.global.v2.f32` of `R[row, col..col+1]`, `add.f32` after the activation | 0 / **4** / **+4 read** |
| **gated FFN** SwiGLU/GeGLU | **NONE** | the accumulator already holds adjacent columns in adjacent registers (`%acc{4j}`, `%acc{4j+1}` are columns `8j+2t`, `+1` of row0). With a merged weight pre-shuffled so gate column `c` and up column `c` are **adjacent**, SwiGLU is `mul.f32 %out, silu(%acc{4j}), %acc{4j+1}` -- a pure register op with no cross-lane traffic at all. Output N halves | 0 / 1 / **-4 write** (the output halves) |
| **low-precision out + act** | partially (`linear_f16_ffn.wk` is f16 in *and* out, but has no GPU seam) | `cvt.rn.f16x2.f32 %h, %accB, %accA;` then `st.global.b32` -- this **is** the W2A v2-store vectorization, obtained for free | 0 / **1** / **-2 write** |

The gated row is the one worth reading twice. It is target #1 of the plan's list, it has **no library
peer on H100 at all** (cuDNN's GEMM+SwiGLU is SM100-only), and on wgmma -- unlike on wmma -- it costs
one multiply and one register, because the fragment layout puts the two operands of the gate in
adjacent registers of the same thread. The entire cost is a host-side weight interleave, which is
free and offline.

### 1.4 SMEM and register arithmetic, exactly

**SMEM.** `WGMMA_W1.smem_bytes()` = `4*49152 + 2*4*8` = **196,672 B** (`ptx_wgmma.rs:4258`), against
the H100 opt-in ceiling of **232,448 B** (probed and printed in the round log: `opt-in SMEM: 232448
B`). Free: **35,776 B**.

* A whole f32 C tile is `bm*bn*4` = `128*256*4` = **131,072 B**. Does not fit -- this is G10's
  premise, and it is why the plan says a TMA-store epilogue must alias the mainloop ring.
* **A row-blocked C stage does fit, and the aliasing question then disappears entirely.**
  `32 rows * 256 cols * 4 B` = **32,768 B**, leaving 3,008 B. Double-buffered as `2 x 16 rows` it is
  the same 32,768 B. Section 4 turns this into a law.
* A f32 bias stage is `bn*4` = **1,024 B**. Bias + a 32,768 B C block = **33,792 of 35,776** -- fits,
  1,984 B of margin.
* Under `WGMMA_W3C` (128x128x64 s6): `smem_bytes()` = `6*32768 + 96` = 196,704 B, free 35,744 B; the
  C tile is 65,536 B and a **64-row** block is 32,768 B. Same conclusion.

**Registers.** `regs_after_split()` = `128*32 + 256*232` = **63,488 of 65,536** per SM. Headroom
2,048 registers = **8 per consumer thread**, and `setmaxnreg` moves in steps of 8, so exactly one
step is available: `consumer_regs` 232 -> **240** (`128*32 + 256*240 = 65,536`, on the nose).
Dropping `producer_regs` to its ISA floor of 24 does not buy a second step (`(65536 - 128*24)/256 =
244`, which rounds down to 240). **+8 registers per consumer thread, once, and no more.**

Against that budget, from column (c):

| epilogue | live scratch regs | fits in 8? |
|---|---|---|
| relu | 0 | yes |
| silu | 1 | yes |
| gelu | 2 | yes |
| bias | 2 (+1 addr) | yes |
| f16-out | 1 | yes |
| bias + relu | 3 | yes |
| bias + silu | 4 | yes |
| bias + gelu | 5 | yes |
| bias + gelu + f16-out | 6 | yes |
| residual | 4 | yes alone |
| **bias + act + residual** | **9** | **NO** -- the plan's claim, itemized |

One nuance the static count misses and the census must settle: after the epilogue stores register
group `j`, `%acc{4j..4j+3}` are **dead** and ptxas may reuse them as scratch. Only `j = 0` has zero
slack. So "9 > 8" is a worst-case static bound, not a proof of a spill -- which is precisely why
G14's ptxas census must run on *every* new variant rather than a sample, and why `C7511` (a silent
2-4x, not a failure) is the thing to grep for.

### 1.5 Cost of the transcendentals, relative to the store they ride on

`tanh.approx.f32`, `ex2.approx.f32` and `rcp.approx.f32` all issue on the SM's multi-function units:
16 results per SM per clock on cc 9.0, against 128 for f32 FMA. At 132 SMs and the round log's
measured 1,980 MHz SM clock:

* MUFU throughput = `132 * 16 * 1.98e9` = **4.18e12 elements/s**
* f32 FMA throughput = `132 * 128 * 1.98e9` = **3.34e13 elements/s**
* HBM3 f32 stores = `3.35e12 / 4` = **8.4e11 elements/s**

**GELU's single MUFU op is 5.0x faster than HBM can store the result it produces; SiLU's two are
2.5x faster; GELU's seven ALU ops are 4.8x faster.** The activation is not the cost. The store is the
cost, and it is the store W2A's v2/v4 change and 1.3's f16 narrowing attack. Concretely at sq4096
(`M*N = 16.78e6`): a fused GELU adds `16.78e6 / 4.18e12` = **4.0 us** of MUFU issue to a kernel
measured at **222.7 us** (r3, `w1_s4_mcb2`) -- **1.8%**, and it is overlapped with the store, not
serialized with it. Publish the fused-GELU row without an activation-cost asterisk.

Sources for section 1:
[cublasLt.h `cublasLtEpilogue_t`](https://www.math.cmu.edu/users/aullrich/myenv/lib/python3.8/site-packages/nvidia/cublas/include/cublasLt.h),
[cuBLASLt notes (corsix)](https://www.corsix.org/content/cublaslt-notes),
[CUTLASS epilogue fusion / EVT (Colfax)](https://research.colfax-intl.com/epilogue_visitor_tree/),
[CUTLASS EVT reference](https://deepwiki.com/NVIDIA/cutlass/5.3-epilogue-fusion-and-activation-functions).
