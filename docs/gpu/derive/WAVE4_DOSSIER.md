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

---

## 2. BREAK-EVEN ARITHMETIC

### 2.0 The model, and the one identity everything follows from

Write `T_p` for the peer's measured GEMM ms, `T_w` for ours, and `r = T_p / T_w` for our fraction of
cuBLAS. The **unfused chain** the peer must run when it cannot fuse is: its GEMM (which already
writes `C`), then a separate pointwise kernel that reads `C` and writes `D`. That kernel's traffic is

```
  read C (M*N*4) + write D (M*N*4)  =  8*M*N bytes        (f32 out, in place)
```

which is the plan's *"8*M*N bytes of chain traffic deleted"* (target #3). Call its time
`E = 8*M*N / BW`. Then

```
  break-even fraction      r* = T_p / (T_p + E)
  fused-vs-chain speedup   S  = (T_p + E) / T_w = r / r*
```

**`S = r / r*`.** The whole section is that ratio. It also means the *shape* of a fusion win is
entirely determined by how memory-bound the output is relative to the GEMM's own FLOPs -- a short-K
GEMM with a full-size C is the best fusion target and a big cube is the worst.

**Two bandwidth columns, on purpose.** `3.35 TB/s` is the H100 SXM5 HBM3 spec peak: using it makes
the peer's extra kernel as cheap as physically possible, so `r*@3.35` is a strict **upper bound** on
what we must reach -- a conservative gate. `3.0 TB/s` (89.6% of peak) is what an achieved streaming
kernel gets, and it is the denominator `ACT2_WAVE_PLAN.md:59` used: all five of its published
break-evens (56.5 / 70.5 / 78.6 / 79.0 / 87.7) reproduce to the tenth of a point at 3.0 and at no
other value, which is how this dossier confirms it is reading the plan's arithmetic and not a
lookalike. **Neither number is measured on this H100** -- see section 4's sweep rows.

### 2.1 The seven shapes

`T_p` and `T_w` are the round-1 DIAGNOSTIC absolutes
(`2026-08-10-h100-act2-wgmma-vs-cublas.log:261-267`); `r (r3)` is the best round-3 config for the
three shapes round 3 covered (`...-r3-bmulticast.log`, published table). Round 3 never re-measured
the four `gpt_*`/`sq1024` shapes, so their `r` is still round 1's un-clustered W1.

| shape | M x N x K | M*N | 8*M*N | T_p ms | T_w ms | r today | E@3.35 us | E@3.0 us | **r\*@3.35** | **r\*@3.0** |
|---|---|---|---|---|---|---|---|---|---|---|
| sq1024 | 1024x1024x1024 | 1.049e6 | 8.39 MB | 0.0070 | 0.0215 | 32.3% | 2.50 | 2.80 | 73.7% | 71.5% |
| sq2048 | 2048x2048x2048 | 4.194e6 | 33.55 MB | 0.0267 | 0.0391 | 68.2% | 10.02 | 11.19 | 72.7% | 70.5% |
| sq4096 | 4096x4096x4096 | 16.78e6 | 134.22 MB | 0.1639 | 0.2429 | 73.5% (r3) | 40.07 | 44.74 | 80.4% | 78.6% |
| sq8192 | 8192x8192x8192 | 67.11e6 | 536.87 MB | 1.2722 | 2.1627 | 82.2% (r3) | 160.26 | 178.96 | 88.8% | 87.7% |
| **gpt_d1024_up** | 4096x4096x1024 | 16.78e6 | 134.22 MB | 0.0581 | 0.1060 | 54.8% | 40.07 | 44.74 | **59.2%** | **56.5%** |
| gpt_d1024_down | 4096x1024x4096 | 4.194e6 | 33.55 MB | 0.0448 | 0.0575 | 78.0% | 10.02 | 11.19 | 81.7% | 80.0% |
| gpt_d4096_up | 4096x16384x4096 | 67.11e6 | 536.87 MB | 0.6734 | 1.5722 | 42.8% | 160.26 | 178.96 | 80.8% | 79.0% |

**`gpt_d1024_up` is the break-even champion by 14 points**, and the mechanism is visible in a second
statistic -- the fraction of our own kernel that is the C store (`M*N*4 / BW` over `T_w`, at 3.0
TB/s):

| shape | C store | T_w | store share of our kernel |
|---|---|---|---|
| **gpt_d1024_up** | 22.4 us | 106.0 us | **21.1%** |
| sq2048 | 5.6 us | 39.1 us | 14.3% |
| sq4096 | 22.4 us | 222.7 us (r3) | 10.0% |
| gpt_d1024_down | 5.6 us | 57.5 us | 9.7% |
| sq1024 | 1.4 us | 21.5 us | 6.5% |
| sq8192 | 89.5 us | 1546.5 us (r3) | 5.8% |
| gpt_d4096_up | 89.5 us | 1572.2 us | 5.7% |

A fifth of `gpt_d1024_up` is the store. `K = 1024` makes the GEMM short while `M*N` stays full-size,
so both the fusion saving and the f16-narrowing saving are worth more there than anywhere else in
the suite.

### 2.2 The margin table: `S = r / r*` at 3.0 TB/s

| shape | r\* | **today** | at r=0.75 | at r=0.82 | at r=0.90 | at r=1.00 |
|---|---|---|---|---|---|---|
| sq1024 | 71.5% | 0.452x | 1.050x | 1.147x | 1.259x | 1.399x |
| sq2048 | 70.5% | **0.968x** | 1.064x | 1.163x | 1.277x | 1.419x |
| sq4096 | 78.6% | 0.936x | 0.955x | 1.044x | 1.146x | 1.273x |
| sq8192 | 87.7% | 0.938x | 0.855x | 0.935x | 1.027x | 1.141x |
| **gpt_d1024_up** | **56.5%** | **0.970x** | **1.328x** | **1.451x** | **1.593x** | **1.770x** |
| gpt_d1024_down | 80.0% | **0.975x** | 0.937x | 1.025x | 1.125x | 1.250x |
| gpt_d4096_up | 79.0% | 0.542x | 0.949x | 1.038x | 1.139x | 1.266x |

At 3.35 TB/s every entry shrinks by 2-4%; the ordering is unchanged and `gpt_d1024_up` still leads by
13+ points. **The three shapes at 0.968 / 0.970 / 0.975 are exactly the plan's *"three shapes are
already at 0.97x of the chain at today's un-improved GEMM speed"*** -- reproduced independently,
which validates both this model and the plan's.

**And that is also the sentence the wave must not misread: 0.97x is a LOSS.** At today's measured
speeds **not one of the seven shapes clears 1.0x** on fused-vs-chain. Wave 4 cannot publish a fusion
win on its own; it needs C1's or W3's gain to have landed *at the shape being published*, and C1's
+6.2 points are measured only at sq4096/sq8192 -- the four remaining shapes, including the champion,
have not been re-measured since round 1. **Re-measuring `gpt_d1024_up` under `w1_s4_mcb2` is a
prerequisite row of Wave 4's visit, not a nice-to-have.**

### 2.3 Fused-vs-FUSED: where it is winnable, and by how much

This is the comparison the refusal clause is about. Split the 16 by whether the peer's *best
available implementation* is a fused kernel or a chain.

**Group A -- the peer fuses it (DEFAULT, RELU, BIAS, RELU_BIAS, GELU, GELU_BIAS, and the AUX/BGRAD
forms).** Both sides are register-resident, both write the same bytes, both pay the same store. The
peer's own epilogue cost is directly measurable as `epilogue_sec / default_sec` from
`time_cublaslt_gemm_nt_f16_epilogue` with `LtEpilogue::None` as the control (`baselines.rs:3276`
exists for exactly this), and section 1.5 says our side of it is ~1.8% at sq4096. So

```
  S(group A)  =  r * (1 + eps_peer) / (1 + eps_us)   ~=  r
```

**No shape in the suite is winnable in group A**, at any measured `r` from 32.3% to 82.2%. Publish
these as parity rows with the peak-fraction column beside them, never as fusion wins. That is the
refusal clause, derived rather than asserted.

**Group B -- the peer has no member (SiLU, residual+act, gated, and conditionally lowp-out+act).**
The peer's floor is "fuse what it can, then run a kernel", which is `T_p + E`, so `S = r / r*` and
section 2.2 is the answer. Three sub-derivations, because the traffic is not identical:

*SiLU / SiLU+bias.* The peer fuses BIAS (free) and runs a SiLU kernel: `T_p + 8*M*N/BW`. Our side
pays nothing extra. `S = r / r*` exactly as tabulated. **This is the cleanest publishable form in
the wave** and its PTX is already shipped (`ptx_wmma.rs:1187-1192`).

*Residual + act, `act(A*B^T + b) + R`.* The peer fuses `GELU_BIAS` and runs a residual-add kernel
that reads D, reads R and writes D: `T_p + 12*M*N/BW`. We must read R: `T_w + 4*M*N/BW`. The
break-even is **identical** (`T_p/r + 4MN/BW <= T_p + 12MN/BW` reduces to `r >= r*`), but the
speedup is smaller because our numerator grows:

```
  S(residual)  =  (T_p + 12*M*N/BW) / (T_p/r + 4*M*N/BW)
```

At `gpt_d1024_up`, r=0.82: `(58.1 + 67.1) / (70.9 + 22.4)` = **1.343x**, against 1.451x for
SiLU+bias at the same shape and speed. Residual costs 0.11x of margin *and* 4 registers *and* a
kernel parameter. Rank it behind.

*Gated FFN (SwiGLU/GeGLU).* Let `P = M * dff` be the **final** gated output; the GEMM's own output is
`2P`. The peer has no epilogue at all and runs `T_p + (8P read + 4P write)/BW = T_p + 12P/BW`. We
write `4P` instead of `8P`, so our kernel is *cheaper* than the plain GEMM by `4P/BW`. Break-even
again reduces to the same `r*` (because `16P = 8*(2P) = 8*M*N`), and

```
  S(gated)  =  (T_p + 12P/BW) / (T_p/r - 4P/BW)
```

At `gpt_d4096_up` (N=16384 = 2 x 8192, so P = 33.55e6), r=0.82: `(673.4 + 134.2) / (821.2 - 44.7)` =
**1.040x** -- within 0.2% of the plain-act 1.038x. **The gated form's value is not a bigger ratio;
it is that no library peer exists on H100 at all**, so the row is a fusion claim rather than a parity
claim, and it costs one multiply and one register (section 1.3).

### 2.4 Low-precision output: the claim hinges on one cheap measurement

Plan target #2 asserts *"cuBLASLt has no low-precision-output-plus-activation epilogue"*. Read
carefully, the enum argument does **not** support that: `cublasLtEpilogue_t` is orthogonal to D's
data type, which is a matrix-layout attribute (`FusedLtPlan::new` sets it via
`create_matrix_layout(out.as_sys(), ...)` at `baselines.rs:3682`). The absence test at
`baselines.rs:5297` bans the *strings* `F16`/`BF16`/`FP8` from member names, which is true and is not
the same claim. `baselines.rs:3216-3219` already concedes the point -- *"'the enum has no such name'
is an argument, not a measurement"* -- which is why `cublaslt_epilogue_support_matrix`
(`baselines.rs:3918`) exists.

Two scenarios, and the whole target lives or dies on which one the matrix reports:

**Scenario A -- the f16 column DECLINES `GELU_BIAS`.** The peer must run `T_p` (f32 out, 4MN
written) plus a cast/activate kernel (read 4MN + write 2MN = 6MN). We write 2MN instead of 4MN.
Break-even reduces to the same `r*` once more, and

```
  S(f16-out)  =  (T_p + 6*M*N/BW) / (T_p/r - 2*M*N/BW)
```

At `gpt_d4096_up`: **0.529x today** (r=0.428), **1.040x at r=0.82**, **1.285x at GEMM parity**.

> **Correction to carry forward.** The plan's headline for this target -- *"0.179 ms on
> gpt_d4096_up = 27% of the peer's GEMM"* -- is `8*M*N/BW = 178.96 us` at 3.0 TB/s, i.e. the total
> *pipeline traffic-time deleted*, expressed against `T_p = 673.4 us`. That is the **GEMM-parity**
> figure (S = 1.285x, +28.5%). At the 82% the rest of the plan uses it is **+4.0%**. Publish it as
> "at parity this fusion is worth 27% of the peer's GEMM; at today's 42.8% it is a 0.53x loss",
> never as a present-tense 27%.

**Scenario B -- the f16 column SUPPORTS it.** The peer writes 2MN directly, there is no cast pass,
and the comparison collapses to pure GEMM parity: `S = r`, a loss. **Target #2 evaporates.** Prior
evidence favours Scenario B: `CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` is documented as *"Generally same
as output matrix type"*, which presumes an f16 D with a bias is ordinary, and TransformerEngine
drives cuBLASLt with fp16/bf16 D and GELU epilogues in production.

`cublaslt_epilogue_support_matrix(g, m, k, n)` is a descriptor build plus a heuristic query -- no
launch, no timing, **milliseconds for all twelve cells**. It is the highest value-per-dollar
measurement in the wave and it must run *before* any engineering is spent on the f16-out arm.

### 2.5 What section 2 concludes

1. `S = r / r*`, and `r*` is a property of the shape alone. Publish `r*` beside every fused row.
2. **Group A (the six cuBLASLt fuses) is unwinnable at any measured `r`.** Parity rows only.
3. **Group B's highest-margin target is `gpt_d1024_up` at `r* = 56.5%`** -- 14 points below the next
   shape, because 21.1% of that kernel is the C store.
4. **Nothing publishes today.** Three shapes sit at 0.97x; the rest are worse. The champion has not
   been re-measured since round 1 and must be, under `w1_s4_mcb2`, in Wave 4's visit.
5. The f16-out target is **unproven**, gated on one millisecond-cost support probe, and its
   headline number is a parity-case figure.

---

## 3. RANKED IMPLEMENTATION ORDER

Scored as (product relevance for the ML surface) x (derived margin from section 2) x (1 / risk).
"Margin" is `S` at `gpt_d1024_up` with `r = 0.82` unless the row names another shape, because that is
the cell section 2 identified as the champion. "Group A/B" is section 2.3's split -- **a group-A row
has margin zero by construction and can only ever be published as a parity row.**

| # | epilogue | product surface | group | margin | risk | why here |
|---|---|---|---|---|---|---|
| **R1** | **bias (+ relu)** | every `Linear`, every projection | A (0) | **0** | **MED-LOW, structural** | the only item that adds a kernel PARAMETER |
| **R2** | **SiLU + bias** | Llama / Qwen / Mistral FFN gate | **B** | **1.451x** | **LOW** | shipped PTX, 1 register, no new param beyond R1 |
| **R3** | GELU + bias | GPT-2 / BERT FFN | A (0) | 0 | LOW | the control that makes R2 legible |
| **R4** | f16 narrowing out (+ act) | every inference FFN | **B?** | **1.040x** (1.285x at parity) | LOW-MED | **conditional on the support probe** |
| **R5** | gated SwiGLU / GeGLU | every modern FFN | **B** | 1.040x @ gpt_d4096_up | MED-HIGH | no library peer on H100 at all |
| **R6** | residual + act | down-proj, attn out-proj | **B** | **1.343x** | MED | 9 registers > the 8 available |
| R7 | GELU_AUX (pre-act tensor) | training forward | A (0) | 0 | LOW | build when a Wukong backward needs it |
| R8 | DGELU / DRELU | training backward | A (0) | 0 | MED | ditto, and it needs R7's tensor first |
| R9 | RELU_AUX bitmask | training forward | A (0) | 0 | **HIGH** | **do not build** -- see 3.4 |
| R10 | BGRADA / BGRADB | training backward | A (0) | 0 | n/a | **not an epilogue task** -- see 3.4 |
| R11 | RoPE in the QKV epilogue | prefill / decode | B | out of scope | HIGH | target #4, its own derivation |

### 3.1 R1 first, and it is not because it is easy

Bias is the only Wave-4 epilogue that changes the kernel's **signature**. `PARAM_ORDER`
(`ptx_wgmma.rs:1517`) is a single `&[ParamKind]` of six entries, and `gpu.rs:17896-17945` asserts
device-free that *every* wgmma entry declares exactly it -- the round log prints the result as a gate
line (`[gate] 13 wgmma entries declare exactly PARAM_ORDER (6 params, kinds in order)`). Add a bias
pointer and that law is false for half the corpus. `gpu.rs:7135` already names the failure mode:
pushing a short argument array is not an error the driver reports; it reads whatever follows on the
host stack as the bias pointer.

So R1 is sequenced first because it is the **riskiest structural change and the cheapest numerical
one**. Land the parameter, the derived name and both laws while the only thing that can be wrong is a
bias -- whose correctness is checkable against a two-line host reference (`fused_epilogue_reference`,
`baselines.rs:3495`, already written and already gated) -- and everything after it is purely
additive. Relu rides along for free (one instruction, zero registers) and gives the epilogue-variant
axis its first non-trivial member.

**R1 publishes nothing.** `BIAS` and `RELU_BIAS` are cuBLASLt members; the row is a parity row.

### 3.2 R2 is the wave's headline, and the arithmetic says so twice

`SiLU + bias` on `gpt_d1024_up` is the highest (product x margin) / risk cell in the entire wave:

* **Product**: it is the gate half of SwiGLU, i.e. the activation of essentially every model shipped
  since Llama 2. `tests/run/linear_silu.wk` already spells the bias-free form and the recognizer
  composes bias with any act code, so no language work is needed.
* **Margin**: `r* = 56.5%` (3.0 TB/s) or `59.2%` (3.35). At W3's own predicted 59-69% for this shape
  it publishes at **1.04x-1.22x**; at C1's already-measured 82.2% it is **1.45x**; at parity,
  **1.77x**. cuBLASLt has no SiLU member -- `LtEpilogue::parse("silu")` is an explicit error
  (`baselines.rs:3342`) and the absence is pinned by a device-free test -- so the peer's floor is
  `BIAS` fused plus a separate SiLU kernel, and there is no strawman to accuse us of.
* **Risk**: five PTX instructions and one scratch register, and the exact instruction sequence is
  already shipped and already gated against the standalone `vmath` kernels
  (`ptx_wmma.rs:1187-1192`). No new parameter beyond R1's. No SMEM beyond R1's 1 KB.

### 3.3 R3-R6, and the ordering constraints between them

**R3 (GELU + bias) is a control, not a product.** Build it because (a) it is the only arm that can be
compared to `cublaslt_gemm_nt_f16_epilogue(GELU_BIAS)` **arm-for-arm** and therefore the only way to
measure `eps_peer`, and (b) it differs from R2 by exactly one MUFU op and one register, so
`S(R2) / S(R3)` isolates the activation's own cost on a single kernel instead of on two shapes.
Publish it labelled parity.

**R4 (f16 out) must not start before the support probe.** Section 2.4: if
`cublaslt_epilogue_support_matrix`'s f16 column supports `GELU_BIAS`, the entire target collapses to
GEMM parity and the engineering is wasted. If it declines, R4 is worth 1.04x now and 1.285x at
parity, *and* it delivers W2A's rung-1 vectorization for free on the f16 path -- `cvt.rn.f16x2.f32`
packs the adjacent column pair into one b32, so the pair of scalar `st.global.f32` at `+0/+4` becomes
one `st.global.b32`, which is the same 2:1 sector consolidation the v2 change buys on the f32 path.
Sequence R4 **after** W2A so the two are not confounded in one A/B.

**R5 (gated) is the highest product relevance and the highest structural risk.** In the kernel it is
one `mul.f32` and one register (section 1.3). Everything else about it is plumbing: the output N
halves, so the grid, the epilogue's `mad.lo.s32 %tmp,%row,%N,%colb` index and the store addresses all
change; the merged weight must be pre-shuffled host-side so gate column `c` and up column `c` land
adjacent; and **there is no `.wk` spelling for a gated FFN today** (section 5). Its margin at
`gpt_d4096_up` is only 1.040x at r=0.82 because that shape is already store-light (5.7%) -- the win
is not the ratio, it is that no library peer exists, so it is a *fusion* claim rather than a *parity*
claim. Do not sell R5 on a ratio.

**R6 (residual + act) is the register cliff.** 4 scratch registers on top of bias's 2 and gelu's 2 is
9 against the 8 available (section 1.4), and lowering `producer_regs` to the ISA floor of 24 does not
buy a second `setmaxnreg` step. Two ways out, in order of preference: (a) rely on ptxas reusing the
dead `%acc{4j..4j+3}` after each store -- true for every `j > 0` and settled only by the census, not
by a static count; (b) drop the activation, since `residual + bias` with no activation is exactly the
transformer down-projection and attention output-projection sublayer, and it is the form
`ptx_wmma.rs:1922` already ships on the wmma path. Its margin (1.343x) is genuinely behind R2's
(1.451x) *because* we must read `R` -- 4 bytes per output element that the fused kernel pays and the
unfused GEMM does not.

### 3.4 The two that should not be built, and why that is a finding

**R9, the RELU_AUX bitmask: do not build it.** It is a group-A epilogue -- cuBLASLt fuses
`RELU_AUX`, `RELU_AUX_BIAS`, `DRELU` and `DRELU_BGRAD` -- so the maximum publishable margin is zero,
and the implementation is the hardest item in the table: cuBLASLt's own `AUX_LD` constraint is 128
**bits**, our D-fragment's lane-to-column map is non-contiguous, and a warp ballot therefore
interleaves eight rows into one 32-bit word (section 1.2). Zero margin at maximum risk is the
definition of a row to skip. If a Wukong backward ever needs the forward mask, spend the
`+2 bytes/element` on GELU_AUX's shape (a plain f16 pre-activation tensor) and keep the epilogue
trivial.

**R10, BGRADA/BGRADB: not an epilogue task at all.** *"Bias gradient based on the input matrix A.
Reduction occurs over the GEMM's k dimension."* The wgmma epilogue holds the C tile; it never sees
the A or B operand outside the mainloop's shared-memory ring, and the reduction axis is K, which is
the axis the mainloop consumes and discards. Implementing these means a separate reduction kernel,
which is a `wukong_sreduce` job, not a wgmma job. Recording this stops a future wave from budgeting
for it as "two more epilogues".

### 3.5 The dependency order, as a single line

```
  W2A (v2 store)  ->  R1 (bias param + PARAM_ORDER law + derived-name law)
                          |
                          +->  R2 (SiLU+bias)   [PUBLISHABLE, headline]
                          +->  R3 (GELU+bias)   [control, parity]
                          +->  R6 (residual)    [PUBLISHABLE, after the register census]
                          |
     support probe  ------+->  R4 (f16 out)     [PUBLISHABLE only under Scenario A]
                          |
     weight interleave ---+->  R5 (gated)       [PUBLISHABLE, no peer exists]
```

The support probe and the `gpt_d1024_up` re-measurement under `w1_s4_mcb2` are both **prerequisites of
the round, not results of it**: the first decides whether R4 is built at all, the second decides
whether any Wave-4 row can be published as a win rather than a 0.97x tie.

---

## 4. GUARD / LAW SET

Eight laws. Three are extensions of existing guards (G3, G9, G10), two are new (G21, G22), and three
are the correctness floor Wave 4 must add because no existing gate covers a fused epilogue. Law text
is written to be pasted into a doc comment; the rationale under each is what a reader needs when the
law fires.

### L1 -- the ASCII gate, and the enumeration it rides on (crate rule 1, standing rule 3)

> **L1.** Every epilogue variant this wave emits appears in `wgmma_device_free_modules()`. That one
> enumeration is what `wgmma_ptx_is_pure_ascii`
> (`ptx_wgmma.rs:4358`), `every_module_opens_at_the_architecture_locked_hopper_floor`, the crate-wide
> `.version` law and the ptxas census all scan, so a module cannot be inside three of them and
> outside the fourth. A variant reachable from a launcher but absent from the corpus is a law
> violation in itself.

*Rationale.* The ASCII gate is not theoretical here. The activation constants are emitted through
`format!("0f{:08X}", x.to_bits())` and are ASCII by construction, but the prose that will accompany
them is not: this dossier's own source for the GELU formula reads `0.5*x*(1 + tanh(sqrt(2/pi)*(x +
0.044715*x^3)))` precisely because the natural spelling uses a square-root sign and a middle dot, and
one of those copied into a `format!` is a `ptxas fatal` at `cuModuleLoadData`. `ptx_wgmma.rs:4354`
already says this file's prose "is full of arrows and multiplication signs waiting to be copied into
a `format!`" -- Wave 4 adds a family whose reference formulas are the worst offenders in the repo.

### L2 -- G3 extended: the derived name must carry the epilogue

> **G3-W4.** `WgmmaCfg::derived_name()` (`ptx_wgmma.rs:1230`) is a **total function of the emitting
> geometry**, and Wave 4 adds three arguments to that geometry: the activation, the bias flag and the
> output dtype. The name becomes
>
> ```
>   wgmma_nt_{dtype}_{bm}x{bn}x{bk}_s{stages}{multicast_tag}{epilogue_tag}{out_tag}
> ```
>
> with `epilogue_tag` empty for the plain GEMM and otherwise `_b`? + one of `relu|gelu|silu|gated`
> + `_r`? for the residual, and `out_tag` empty for f32 and `_f16o` for the narrowing store.
> `validate()` refuses to emit any config whose `name` **or** `key` differs from `derived_name()`.

*Rationale, and why this is the wave's #1 hazard rather than a tidiness rule.* `Gpu::function` /
`raw_function_dyn` cache on the key string **alone** and never re-examine the PTX on a hit
(`ptx_wgmma.rs:1219`). A row written `WgmmaCfg { act: Silu, ..WGMMA_W1 }` that forgets `key` loads
the **plain-GEMM** module, launches it a thousand times, and publishes its time under the `SILU`
heading. Note carefully what does and does not catch that: the *correctness* arm builds its own
reference and would fail, so the defect is loud **if the correctness arm runs on the same cached
module** -- but the timing loop is a separate call, and a round that skipped or reordered the
correctness arm would report a **fabricated fusion win with no symptom at all**. C1's sweep already
carries this hazard on one axis; Wave 4 adds three more, and `epilogue_tag` is the axis whose
mistaken value is *fastest* (the plain GEMM is always the quickest arm in the table), which is the
worst possible failure direction.

### L3 -- G21, NEW: the parameter list becomes a function of the variant

> **G21.** `PARAM_ORDER` (`ptx_wgmma.rs:1517`) stops being a constant and becomes
> `WgmmaCfg::param_order() -> &'static [ParamKind]`: the six existing entries, then `GlobalPtr` for a
> bias, then `GlobalPtr` for a residual, then `GlobalPtr` for an aux output -- **in that order and no
> other**. `LaunchPlan::params` carries the per-variant slice. The device-free law at
> `gpu.rs:17896-17945` re-derives the expected list **per entry, from the config that emitted it**,
> and still prints one gate line with the total count.

*Rationale.* The launcher pushes a fixed-length argument array. Pushing short is **not an error the
driver reports** -- `gpu.rs:7135` already spells this out -- it reads whatever follows on the host
stack as the bias pointer, and the kernel dereferences it. Today the law holds because there is
exactly one list and 13 entries declare it; the round log prints
`[gate] 13 wgmma entries declare exactly PARAM_ORDER (6 params, kinds in order)`. The instant a bias
pointer exists that sentence is false for part of the corpus, and the *easy* fix -- relaxing the
assert to "at least six" -- deletes the property entirely. **Land G21 in the same commit as the
parameter, never after.**

### L4 -- G9 restated as a transport law over the store CLASS

> **G9-W4.** For every emitted variant, collect every global-store instruction -- `st.global.f32`,
> `st.global.v2.f32`, `st.global.v4.f32`, `st.global.b32/b64`, and any
> `cp.async.bulk.tensor.*.global.shared::cta` descriptor store -- and assert:
>
> 1. the multiset of **accumulator registers appearing as store source operands**, after any
>    in-place activation rewrite, is exactly `{%acc0 .. %acc{nacc-1}}`, each exactly once;
> 2. every such store is predicated on a conjunction containing **both** a row bound and a column
>    bound;
> 3. the emitted store count equals `nacc / lanes_per_store` for the variant's declared vector width.
>
> Device gates at `N % 8 != 0` -- specifically `N = bn + 1` and `N = bn - 3` -- accompany it.

*Rationale.* The law as it stands
(`the_epilogue_stores_every_accumulator_exactly_once_and_bounded`, `ptx_wgmma.rs:4711`) asserts
`ptx.matches("st.global.f32").count() == nacc` and `ptx.matches("],%acc{i};").count() == 1`. Both
break **loudly** under every Wave-4 change: a v2 store spells the source `{%acc0,%acc1}`, an f16 store
sources `%h`, and a TMA store emits zero `st.global.f32`. That is the good news. The hazard is the
**repair**: the one-line fix is to relax `assert_eq!` to `assert!(count <= nacc)`, which passes
vacuously at zero and deletes both real properties -- exactly what the plan means by "a TMA-store
epilogue ... would otherwise delete the law along with both real properties". Restating it as a
positive property over source operands makes the vacuous relaxation unavailable. The `N % 8 != 0`
gates matter because a vector store cannot satisfy a ragged column edge and the predication must
therefore fall back per-lane there; a variant that is only ever run at `N % 8 == 0` never exercises
the fallback.

### L5 -- G10 restated, with the row-blocked finding that removes the aliasing question

> **G10-W4.** `smem_bytes()` is the **one** authority for `dyn_smem_bytes`; any epilogue region is
> part of it or `dyn_smem_bytes` silently under-requests. `the_smem_map_is_disjoint` asserts every
> region's `[off, off+len)` is pairwise disjoint and that the maximum end equals `smem_bytes()`.
> **An epilogue region may not alias the mainloop ring.**

*Rationale, and the derived alternative.* W1 leaves `232448 - 196672 = 35776` free bytes; a whole f32
C tile is `128*256*4 = 131072` and does not fit, which is why the plan concludes a TMA-store epilogue
"physically must alias the mainloop ring". **It does not have to.** A *row-blocked* stage fits with
room to spare: `32 rows x 256 cols x 4 B = 32768 B`, or `2 x 16 rows` double-buffered for the same
32768, leaving 3008 B; add the 1024 B f32 bias stage and it is 33792 of 35776, with 1984 B of margin
(section 1.4). Under W3C the same arithmetic gives a 64-row block at 32768 of 35744 free. Taking the
row-blocked region makes G10 a *statement about a map* rather than a *race to reason about*, and that
matters most exactly when W3's persistence lands -- because then the producer refills stage 0 for
tile `t+1` while a consumer would be staging C out of it, and an aliasing epilogue that is correct in
a one-tile kernel becomes a live race with no compile-time symptom.

### L6 -- G22, NEW: the register-budget law

> **G22.** `regs_after_split() <= 65536` and every `setmaxnreg` target is a multiple of 8 in
> `[24, 256]` (both already asserted). Wave 4 adds: **the epilogue's declared scratch registers are
> part of the config**, and `validate()` rejects any variant whose
> `accum_regs() + epilogue_scratch()` exceeds `consumer_regs`. The derived headroom at W1 is exactly
> **+8 per consumer thread, once**: `128*32 + 256*232 = 63488` of 65536, so `consumer_regs` may rise
> to 240 (`128*32 + 256*240 = 65536` exactly) and no further -- lowering `producer_regs` to the ISA
> floor of 24 gives `(65536 - 3072)/256 = 244`, which rounds down to the same 240.

*Rationale.* Section 1.4's itemization: relu 0, silu 1, gelu 2, bias 2 (+1 address), f16-out 1,
residual 4. `bias + gelu = 5` fits; `bias + gelu + residual = 9` does not. The static count is
conservative -- `%acc{4j..4j+3}` are dead after group `j` stores and ptxas may reuse them for every
`j > 0` -- so **G22 is a design gate, not a verdict**, and the verdict is G14's ptxas census. Run the
census on **every** new variant, not a sample, and grep for `C7511`: it is a *silent 2-4x*, not a
failure, and a spilled epilogue that still produces correct numbers is exactly the shape of result
that gets published as "fusion did not help".

### L7 -- THE EXACTNESS LAW. Bit-exact at f32 out; a monotonicity law at f16 out

This is the law the wave does not currently have, and it is stronger than the plan assumes.

> **L7a (bit-exactness, f32 out).** For every `(activation, bias)` variant with `LtOut::F32`
> semantics,
>
> ```
>   wgmma_fused_epilogue(A, B, bias)  ==  vmath_act( gemm_nt_wgmma(A, B) + bias )
> ```
>
> **element-wise `==`, not a tolerance**, where both sides are Wukong kernels on the same device.
>
> **L7b (f16 out).** Not bit-exact, and the law is a *monotonicity* statement instead: against an
> independent f64 reference `R`,
>
> ```
>   max_i | fused_f16[i] - R[i] |   <=   max_i | chain_f16[i] - R[i] |
> ```
>
> i.e. **the fused arm is never worse than the chain it replaces**.
>
> **L7c (vs cuBLASLt).** A tolerance, `c * sqrt(K) * eps`, plus a separately derived absolute band
> for the activation -- never bit-exactness, because the peer's reduction order is its own.

*Why L7a is bit-exact and not a tolerance -- the derivation.* The accumulator is f32. In the unfused
chain, the GEMM stores that f32 to HBM and a second kernel loads it back: **an f32 store followed by
an f32 load is the identity**, no rounding occurs. The bias add is the same `add.f32` on the same two
f32 values in both arms. The activation is the *same PTX instruction sequence with the same
constants* in both arms -- `ptx_wmma.rs:1180` already states this contract for the wmma path ("the
exact same formulas + constants as the standalone `ptx::vmath_ptx` kernels, so a fused `silu(A*B^T)`
equals the unfused `silu(gemm)`"), and Wave 4 must preserve it while porting. `tanh.approx.f32`,
`ex2.approx.f32` and `rcp.approx.f32` are approximate but **deterministic**: a fixed function of the
input bit pattern on a given architecture. Therefore every intermediate is bit-identical and so is the
result. Anything less than `==` here is hiding a real difference.

Three edges L7a must be written to survive, all of which agree *because both arms use the same
instruction* and would diverge under any "equivalent" rewrite:

* **Signed zero.** `max.f32 x, 0f00000000` on `x = -0.0` is not `f32::max`'s answer. Both arms issue
  the identical `max.f32`, so they agree; a scalar rewrite of one arm would not.
* **NaN.** Same argument, same instruction, same tie-break.
* **The bias-then-activate order.** `apply_f64` (`baselines.rs:3380`) pins bias-first; the fused
  epilogue must too, and a fused arm that activated first would still pass a *tolerance* gate on
  smooth data.

*Why L7b is a monotonicity law and not a tolerance.* At f16 out the fused arm rounds **once**
(activate in f32, then `cvt.rn.f16x2.f32`); the chain rounds **twice** (GEMM stores f16, reload,
widen, activate, store f16). They cannot be bit-equal, and the fused arm is *structurally the more
accurate one*. A plain "within tolerance of the chain" gate would therefore be satisfied by a fused
arm that had silently become worse, which is the failure the wave most needs to see. Compare both to
`R` and assert the inequality.

### L8 -- the GELU convention, and the trap under it

> **L8.** Two separate gates, never one. (i) The **convention** -- tanh vs erf -- is pinned
> device-free by `the_two_gelu_conventions_are_distinguishable_and_pinned` (`baselines.rs:5397`) and
> by the shipped constants. (ii) The **implementation** band for `tanh.approx.f32` is derived from a
> measured `max |tanh.approx.f32(x) - tanh(x)|` over the exact ramp the correctness gate uses, and
> that measurement is reported in the round log. If the derived band exceeds `1e-3`, say so in the
> log, because the gate is then blind to the convention question and (i) is carrying it alone.

*Rationale.* `baselines.rs:3416-3423` pins the tanh/erf gap at up to `~1e-3` around `|x| ~ 2`, three
orders above the `c*sqrt(K)*eps` band an f16 GEMM gate uses, and warns that a fused-GELU gate written
against the wrong convention "would fail on a perfectly good peer". Wave 4 adds a second source of
GELU error on top: `tanh.approx.f32` is an approximate MUFU instruction, not a correctly-rounded
tanh. **This dossier does not establish its bound and will not assert one** -- but the structural
point holds whatever the number turns out to be: *any tolerance band widened to admit
`tanh.approx`'s error may also be wide enough to admit the erf/tanh confusion the pinning test exists
to catch.* Two questions, two gates, and the second one's band must be a measured number in the log
rather than a constant someone chose to make a row pass.

### 4.9 The sweep rows for the one H100 visit

One container, one log, in this order. Standing rule 1 puts bring-up E/F/G before any perf row;
standing rule 3 puts the $0.02 CPU ptxas census before the visit.

**Preflight -- host-only, free, and it decides what gets built (run BEFORE the round, not in it).**

| row | what | why it is first |
|---|---|---|
| P0 | `cublaslt_epilogue_support_matrix(g, m, k, n)`, both `LtOut`, all six `LtEpilogue::ALL`, at two shapes | 12 cells x 2, **milliseconds**, no launch. **Decides whether R4 exists at all** (section 2.4). A claim of absence must be something a round log shows |
| P1 | ptxas census over `wgmma_device_free_modules()` including every new variant: spill count, `C7511`, `sm_90a`, ASCII | G14. `C7511` is a silent 2-4x, not a failure -- section 1.4's register cliff is settled here and nowhere else |

**Correctness -- every row before any timing row.**

| row | arm | law |
|---|---|---|
| C0 | bring-up E/F/G on W2's corrected guard shape | standing rule 1 |
| C1 | exact-integer `==` per variant on the ragged set, **including `N = bn+1` and `N = bn-3`** | G9-W4's device gates |
| C2 | **`fused(A,B,bias) == vmath_act(gemm_nt_wgmma(A,B) + bias)`, element-wise `==`**, per `(act, bias)` | **L7a -- the wave's new floor** |
| C3 | pseudorandom f16 vs an independent f64 reference at `c*sqrt(K)*eps`, per variant | G2 |
| C4 | f16-out: `max|fused - R| <= max|chain - R|` | L7b |
| C5 | two-run bit-identity on the C3 arm | G8 |
| C6 | `tanh.approx.f32` vs f64 `tanh` over the C3 ramp; report the max | L8(ii) |

**Measurement.**

| row | arm | why |
|---|---|---|
| **M0** | **`hbm_bandwidth`** -- currently `#[ignore]`d and never run on H100 | **every `r*` in section 2 divides by it.** Re-derive the whole break-even table in-round from the measured number and publish that table, not this one |
| **M1** | **our own pointwise activation kernel over `M*N` f32, per shape** | converts `E` from a derivation into a **measurement**. This is the single highest-value row in the round: it turns every fused-vs-chain claim from "8*M*N/BW says" into "we timed it" |
| M2 | per shape x per epilogue: **A** = `time_cublaslt_gemm_nt_f16_epilogue(.., None, F32)`, **B** = our fused variant, **C** = `time_cublaslt_gemm_nt_f16_epilogue(.., epi, F32)` | `C/A - 1` = `eps_peer`, the library's own epilogue cost; `C/B` = fused-vs-fused; `(A + M1)/B` = fused-vs-chain. A and C are **both cuBLASLt**, so `C/A - 1` doubles as the peer's dispersion floor (G16), exactly as rounds 1-3 used the twin |
| **M3** | **`gpt_d1024_up` and `gpt_d4096_up` under `w1_s4_mcb2`** | section 2.2: C1's +6.2 points are measured only at sq4096/sq8192, and the champion shape has not been re-measured since round 1. **Without M3 no Wave-4 row can be published as a win rather than a 0.97x tie** |
| M4 | the `LtOut::F16` column, for whatever P0 said is supported | R4's arm; skipped **honestly**, per `cublaslt_epilogue_available` |
| M5 | an epilogue-elided arm at one shape | splits our own epilogue cost from the mainloop on ONE kernel instead of two shapes (W2's lever, reused) |
| M6 | shapes at **real model dims** -- Llama-3-8B, GPT-2, Qwen -- never `d=64/dff=256` | the plan's own instruction for this wave |

Shapes for M2: `gpt_d1024_up` (the champion), `gpt_d1024_down`, `sq2048`, `sq4096`, `sq8192`,
`gpt_d4096_up`. `sq1024` may be dropped -- at `r = 32.3%` against `r* = 71.5%` it cannot clear
break-even under any Wave-4 change, and its 32-CTA wave quantization is W3's problem, not this
wave's.

### 4.10 Refusals

1. **Any bias / relu / gelu row published as a fusion win** -- the plan's own clause, now derived:
   section 2.3 group A is unwinnable at any measured `r`.
2. **Any fused row published without its `r*` and its peak-fraction beside it.** The ratio alone
   misleads in both directions (standing rule 4), and `r*` is what tells a reader whether a 1.05x is
   a triumph or a rounding error.
3. **Any row whose apparent gain is inside the `C/A` dispersion** (G16).
4. **Any low-precision-output claim without P0's f16 column in the same log.** Section 2.4: the
   enum-absence argument does not support it and the likely answer is that the peer supports it.
5. **Any break-even quoted from 3.35 or 3.0 TB/s once M0 has measured the real number.** This
   dossier's tables are provisional by construction.
6. **Any variant reachable from a launcher but absent from `wgmma_device_free_modules()`** -- it
   would sit outside the ASCII, `.target`, `.version` and census laws simultaneously (L1).
7. **Any fused row published from a round whose C2 arm did not run.** L7a is `==`; a round that
   skipped it and reported a timing has no evidence the timed kernel computed the epilogue at all --
   which is precisely the G3-W4 cache hazard's payload.
