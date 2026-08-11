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
