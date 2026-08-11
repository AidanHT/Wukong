# WAVE 5 DERIVATION -- 8-bit wgmma on sm_90a (e4m3 / e5m2 / s8 / u8)

**Status: DERIVATION. Cost so far: $0.** Nothing here has executed. What it does is turn the
Wave-5 line of `ACT2_WAVE_PLAN.md` into text an implementer can write PTX from without a device in
the room, by re-deriving -- not assuming -- how much of the hardware-settled 16-bit `wgmma` model
transfers to 1-byte operands.

**Why the derivation is worth doing carefully.** The 16-bit shared-memory descriptor cost one paid
H100 visit to settle (`bench/gpu/h100/2026-08-10-h100-s2c-desc-sweep.log`, 19 candidates x 2 K
passes through one module). The 8-bit descriptor must be right *device-free*, because a second
sweep is a second visit. Section 2 is the whole point of this file.

**Provenance rules, inherited from this directory's README.** Every claim below is one of:

* **FACT(repo)** -- read out of this tree at a cited `file:line`, or measured in a cited round log.
  These outrank everything else.
* **FACT(ext)** -- an external document with a URL. Used only where the tree is silent.
* **DERIVED** -- arithmetic over the two above, reproducible by hand.
* **PREDICTION** -- falsifiable, and named as the thing a round exists to test.

Where an external source and a campaign measurement disagree, the measurement wins and the
disagreement is stated rather than smoothed.

---

## 0. The one-paragraph answer

At 1 byte per element the `wgmma` shared-memory descriptor is **byte-identical** to the 16-bit one
that the 2026-08-10 round crowned -- same `Swizzle128`, same `SBO = 1024`, same ignored LBO, same
`base_offset = 0`, same 32-byte K-step -- **provided `BK` goes 64 -> 128 so the shared row stays
exactly 128 bytes.** Every quantity in the descriptor is measured in *bytes*; element width enters
only through the element-count-to-byte conversions (`BK`, the TMA `box_dim[0]`, the shape's K
token). Hold the byte geometry fixed and the paid answer transfers unchanged. Let `BK` stay 64 and
the shared row becomes 64 bytes, the correct mode becomes `Swizzle64` with `SBO = 512` and a
512-byte start alignment -- **a descriptor reading nothing has measured**, and the H100 hour is
spent again. `BK = 128` is not a tuning choice. It is the condition under which the answer we
already bought still applies.

---

## 1. INSTRUCTION SURFACE

### 1.1 The shape menu

**FACT(repo)**, `docs/gpu/derive/D1_h100_gemm.md:161-164`, from a verbatim text extraction of
[PTX ISA section 9.7.16](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
(the section is numbered 9.7.14 in the 12.4/PTX-8.4 archive and 9.7.16 in the current one -- cite
by name as well as number):

> Shape menu, f16 and bf16 dense: `.m64n{8,16,24,...,256}k16` -- M is always 64, N is every multiple
> of 8 from 8 to 256, K is always 16. tf32: same N menu, **K = 8**. fp8 (`.e4m3`/`.e5m2`) and int8:
> same N menu, **K = 32**.

**FACT(ext), corroborating**, NVIDIA's own CUTLASS Warpgroup-MMA programming guide
(<https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/wgmma_programming.html>):
`MmaF16BF16Op` has instruction K = 16; `MmaF8Op` (e4m3/e5m2) and `MmaI8Op` both have instruction
**K = 32**; "WGMMA requires `M = 64` and `8 <= N <= 256` in steps of 8", stated uniformly across
data types.

**DERIVED.** `WgmmaShape::K` (`ptx_wgmma.rs:351`) is currently the constant `16`. It becomes a
function of `WgmmaDtype`: 16 for `{F16, Bf16}`, 32 for `{E4M3, E5M2, S8, U8}`. `WgmmaShape::N_MIN`,
`N_MAX` and `N_STEP` do not move, so `WgmmaShape::new`'s rejection stays exactly as written and
`accum_regs() = n / 2` stays exactly as written (see 1.4).

> **The N menu is the one row of section 1 that is settleable for $0.02 and should be.** `ptxas
> -arch=sm_90a` is a host cross-compiler (`bench/gpu/h100/2026-08-10-ptxas-census.log:7-9`): it
> assembles *for* Hopper without one present, and it rejects a shape token that is not on the menu.
> Emit `m64n{8,64,128,192,256}k32` at each 8-bit type into the census corpus and let ptxas answer.
> Treat a rejection as the ISA speaking, not as a bug.

### 1.2 The operand tail -- three different instructions, not one parameterised one

**FACT(ext)**, PTX ISA section 9.7.16 syntax blocks, corroborated by the Colfax CUTLASS WGMMA
tutorial (<https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/>) and by CUTLASS's own
`SM90_MxNxK_XYZ_SS` atom set:

| operand types | emitted tail after `d, a-desc, b-desc` |
|---|---|
| `.f16` / `.bf16` | `scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b` |
| `.e4m3` / `.e5m2` (and `.tf32`) | `scale-d, imm-scale-a, imm-scale-b` |
| `.s8` / `.u8` | `scale-d` |

**FACT(repo)**, the line this replaces, `ptx_wgmma.rs:3006-3007`:

```
wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 {accs}, %descA, %descB, %pfirst, 1, 1, 0, 0;
```

**DERIVED.** The three 8-bit forms are:

```
wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 {accs}, %descA, %descB, %pfirst, 1, 1;
wgmma.mma_async.sync.aligned.m64n256k32.f32.e5m2.e4m3 {accs}, %descA, %descB, %pfirst, 1, 1;
wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8     {accs}, %descA, %descB, %pfirst;
```

Three things about that table are load-bearing:

1. **`imm-scale-a` / `imm-scale-b` are SIGN FLIPS, not scale factors.** They take `1` or `-1` and
   negate the operand. They are *not* where an fp8 quantization scale goes, and an implementer who
   reads the token "scale" and writes a float there produces text ptxas rejects -- loudly, which is
   the good case. Section 3 is where the real scales live.
2. **The transpose immediates do not merely default to zero for 8-bit; they do not exist.** The ISA
   restricts the transpose operation to the `.f16`/`.bf16` variants, i.e. **8-bit operands are
   K-major only** (Colfax, quoting the ISA: "for non 16-bit operand datatypes, the layout must
   always be K-major"; NVIDIA's CUTLASS guide: "FP8 and INT8 variants are K-major only"). This
   costs us nothing: `SmemDesc::for_layout`'s doc comment at `ptx_wgmma.rs:643-647` already records
   that this backend's NT product stores A as `M x K` K-contiguous and B as `N x K` K-contiguous,
   "so the NT layout this backend already stores needs no transpose flag on either operand". The
   16-bit path emits `0, 0` only because the field is there. The 8-bit path emits nothing, and the
   layout it wants is the only layout the instruction has.
3. **Mixed fp8 input types are legal** (`.e5m2.e4m3` above). `ptx_fp8_train.rs:8` already relies on
   the `mma.sync` equivalent -- "the backward GEMMs therefore multiply an E5M2 gradient by an E4M3
   weight/activation and accumulate in f32". The wgmma retype of the backward pass is one token
   pair, not a second mainloop.

### 1.3 `.satfinite`, and why the right move is to decline it

**FACT(ext).** `.satfinite` exists as a modifier in the integer `mma` family and CUTLASS carries
`..._SATURATE` twins of its SM90 integer atoms, so the integer `wgmma` form very likely accepts it.

**Not settled here, and it does not need to be:** the $0.02 ptxas census answers "does
`wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8.satfinite` assemble" for free.

**DERIVED -- the reason to decline it regardless of the answer.** With `.s8` operands,
`|a| <= 128` and `|b| <= 128`, so `|a*b| <= 16384` and a K-term dot product is bounded by
`16384 * K`. It fits `s32` exactly iff `16384 * K < 2^31`, i.e. **`K < 131072`**. Every shape in
the campaign grid (`K <= 8192`) is more than an order of magnitude inside that bound, so the
saturating and the wrapping instruction **compute the identical value on every input this project
will ever hand them**. Adding `.satfinite` would therefore be an untested difference that can only
manifest on data no gate generates -- the exact shape of a defect that ships. Emit the plain form,
and record the `K < 131072` bound as the reason, so the next reader knows the decline was derived
rather than forgotten.

### 1.4 What K = 32 does to the loop, and what it does not do to the registers

**DERIVED**, all of it from `WgmmaCfg`'s own accessors (`ptx_wgmma.rs:1014-1025, 1145-1160`):

| quantity | 16-bit, `BK = 64` | 8-bit, `BK = 128` | moves? |
|---|---|---|---|
| `row_bytes = bk * size` | 64 * 2 = **128 B** | 128 * 1 = **128 B** | no |
| `tile_a_bytes = bm * bk * size` (bm=128) | **16384 B** | **16384 B** | no |
| `tile_b_bytes = bn * bk * size` (bn=256) | **32768 B** | **32768 B** | no |
| stage bytes | **49152 B** | **49152 B** | no |
| `smem_bytes` at 4 stages | **196672 B** | **196672 B** | no |
| `stage_tx_bytes` | **49152 B** | **49152 B** | no |
| `wgmma_per_stage = bk / K` | 64 / 16 = **4** | 128 / 32 = **4** | no |
| `accum_regs = n / 2` at n=256 | **128** | **128** | no |
| K elements per stage | 64 | **128** | **x2** |
| K tiles for K=4096 | 64 | **32** | **/2** |
| wgmma issues for K=4096 | 256 | **128** | **/2** |
| ragged-K waste, worst case | 63 elements | **127 elements** | x2 |

The `196672 B` row is not a re-derivation: it is what the ptxas census actually measured for
`wgmma_nt_f16_128x256x64_s4` (`bench/gpu/h100/2026-08-10-ptxas-census.log:687`, `smem(gen)
196672`, `384 thr`, `168 regs`, `0 spill`). The 8-bit retype reproduces it exactly, which means
the stage lattice, the barrier offsets, the ring arithmetic, the `expect_tx` count and the
occupancy verdict (1 CTA/SM) are all carried over untouched. **That is the payoff of choosing
`BK = 128`, and it is why the plan calls the geometry "byte-identical at 49,152 B".**

Registers are also untouched, and this is worth stating because it is the thing that usually
breaks on a retype. The accumulator is `f32` for fp8 and `s32` for int8 -- both 32-bit, both
`N / 2` per thread, both 128 registers at `N = 256`. The warp-specialised budget
`128 * 32 + 256 * 232 = 63488 <= 65536` (`ptx_wgmma.rs:188`, `REGS_PER_CTA`) is unchanged, so
`WGMMA_W1`'s `producer_regs: 32 / consumer_regs: 232` split transfers verbatim.

**PREDICTION.** The ptxas census row for `wgmma_nt_e4m3_128x256x128_s4` will read `regs 168,
spill 0, smem(gen) 196672, 384 thr` -- identical to the f16 row -- and will carry **no C7511**
("wgmma instructions are serialized due to insufficient register resources"). A C7511 on the 8-bit
row would mean the retype changed the register pressure, which nothing above predicts; it is a
silent 2-4x and must be treated as a finding, not a warning (G14).

### 1.5 TMA box shapes

**DERIVED**, against `tma_host.rs::TensorMapArgs::validate` (`tma_host.rs:246-335`):

| rule (`tma_host.rs` line) | 16-bit at BK=64 | 8-bit at BK=128 | verdict |
|---|---|---|---|
| `box_dim[0] <= 256` (`:268`) | 64 | **128** | ok |
| `box_dim[0] * elem` multiple of 16 (`:321`) | 128 B | **128 B** | ok |
| `box_dim[0] * elem <= swizzle atom` (`:326`) | 128 <= 128 | **128 <= 128** | ok, exactly at the atom |
| `box_dim[1] <= 256` (`:268`) | 128 (A) / 256 (B) | same | ok, B exactly at the cap |
| `global_strides[d]` multiple of **16 bytes** (`:292`) | row stride `K*2` -> **K % 8 == 0** | row stride `K*1` -> **K % 16 == 0** | **TIGHTER** |

The last row is the only new *constraint* the retype introduces, and it is easy to miss because it
is a property of the operand tensor rather than of the tile: a 1-byte row-major matrix whose row
stride in elements is not a multiple of 16 cannot be described by a tensor map at all. `K = 4096`,
`K = 4104` and every campaign shape clear it; `K = 4100` does not. `TmaDataType` already carries
`U8 = 0` (`tma_host.rs:67`, `CU_TENSOR_MAP_DATA_TYPE_UINT8` at `:470`), and it is the correct
choice for **all four** 8-bit types -- there is no fp8 or s8 tensor-map data type, because TMA
moves bytes and never interprets them. The interpretation is entirely in the `wgmma` operand-type
token.

**One consequence worth writing down:** because `box_dim[0] * elem` sits exactly at the 128-byte
atom in both the 16-bit and the 8-bit configuration, the swizzled TMA copy writes the *same byte
pattern* in both -- which is the mechanical reason section 2's conclusion holds.

---

## 2. DESCRIPTOR / LAYOUT DELTA -- the section this dossier exists for

### 2.1 What the H100 actually settled, restated as bytes

**FACT(repo)**, `bench/gpu/h100/2026-08-10-h100-s2c-desc-sweep.log:93-182`. Nineteen candidates,
each run at K=16 and K=64, scored out of 4096 output lanes exact:

| candidate | exact @K16 | exact @K64 | what it retires |
|---|---|---|---|
| `sw128/lbo16` | **4096** | **4096** | the production reading |
| `sw128/lbo128` | **4096** | **4096** | LBO = one core matrix |
| `sw128/lbo1024` | **4096** | **4096** | LBO = SBO |
| `sw128/swapped` | 64 | 64 | SBO is on the **MN** axis, not the K axis |
| `sw128/sbo128` | 64 | 64 | SBO counts **8-row groups** (1024), not rows (128) |
| `sw128/lbo16-bo1` | 1024 | 1536 | `base_offset` must be **0** |
| `sw128/lbo16-bo2` | 0 | 2048 | same, at the other phase |
| `ctl/rowmajor-k-leading` | 64 | 64 | the sweep reproducing the old round: self-validation |

Four settled facts fall out, and every one of them is stated in **bytes**:

1. **The B128 mode IGNORES the leading-dimension field.** Three different spellings of LBO -- 16,
   128 and 1024 -- are all exactly 4096/4096. The log says so in its own words
   (`:179`): "More than one field spelling is exact in that arm, which is itself a finding: the
   mode IGNORES the field they differ in."
2. **SBO is the MN-direction distance and equals `8 * row_bytes` = 1024 B.** `swapped` and
   `sbo128` both collapse to 64/4096, which is the `r % 8 == 0` signature.
3. **The swizzle pattern anchors at the enclosing 1024-byte boundary, not at the descriptor's start
   address.** The `bo1`/`bo2` rows are the probe for the alternative and they fail; the `bo=0` rows
   pass at *both* K passes, which is precisely the discriminating evidence
   (`:129`: a start-address anchor "needs a phase and every bo=0 row of this arm fails at K=64
   while passing at K=16" -- they did not).
4. **The core matrix is `8 rows x 16 bytes`** (`ptx_wgmma.rs:50`, `:486-489`), and the within-core-
   matrix row stride is a fixed **16 bytes** the descriptor cannot change (`:723`). This is the
   fact that made a wide row-major tile unrepresentable and cost the first round.

**FACT(ext), the independent statement of the same geometry.** Colfax: "Each core matrix has a
_strided_ direction and a _contiguous_ direction, such that its length is 8 in the strided
direction and **16 bytes** in the contiguous direction", and, for K-major, "128-byte swizzle:
SBO = 128 x 8 = 1024 bytes". CUTLASS's own descriptor struct
(`include/cute/arch/mma_sm90_desc.hpp`) matches `SmemDesc` field for field: `start_address_`
[0,14), `leading_byte_offset_` [16,30), `stride_byte_offset_` [32,46), `base_offset_` [49,52) --
with the comment that `base_offset_` is "valid only for B128 and B64 swizzle modes" -- and
`layout_type_` [62,64) with `INTERLEAVE=0, B128=1, B64=2, B32=3`, which is exactly the reversed-
from-TMA encoding `SmemSwizzle` pins at `ptx_wgmma.rs:394-401`.

### 2.2 The core matrix at 1 byte: derive it, do not port it

The ISA defines a core matrix by its **byte** extent, not its element extent. So at 1 byte per
element:

```
core matrix            = 8 rows x 16 bytes            (unchanged -- it is a byte quantity)
elements per row of it = 16 bytes / 1 byte  = 16      (16-bit: 16 / 2 = 8)
core matrix in elements= 8 x 16                       (16-bit: 8 x 8)
```

The K-direction core-matrix index `j` therefore spans **K elements 16j .. 16j+15** for a 1-byte
type, where the module docs' formula (`ptx_wgmma.rs:722`) spans `8j .. 8j+7` for a 2-byte one. The
hardware's address formula is untouched, because every term in it is a byte count:

```
addr(i, j, r, c) = start + i*SBO + j*LBO + r*16 + c*elem
```

Only `c*elem` knows the element width, and `c` is bounded by the core matrix's *element* extent
(16 instead of 8), so `c*elem` still spans exactly 16 bytes. **The core matrix occupies the same
16 bytes per row at every width. That is the invariant, and it is why nothing else moves.**

Now the K step. One `wgmma` K step is `K` elements, and `K` is dtype-dependent:

```
16-bit: 16 elements x 2 bytes = 32 bytes = 2 core matrices along K (16 / 8  = 2)
 8-bit: 32 elements x 1 byte  = 32 bytes = 2 core matrices along K (32 / 16 = 2)
```

**DERIVED, and this is the load-bearing coincidence:** `DescFields::k_step_bytes` is **32 for both**
(`ptx_wgmma.rs:543-545, 599`), and the number of core matrices consumed per K step is **2 for
both**. The ISA fixed K at 32 *bytes*, not at a count of elements -- Colfax states it exactly that
way ("more generally, `K` is fixed to be 32 bytes"). So the mainloop's per-K-step descriptor
advance, `mul.lo.s32 %tmp,%kt,%kstep` followed by the `shr/and/or` address fold
(`ptx_wgmma.rs:2997-3002`), is **byte-identical** between the 16-bit and the 8-bit kernel.

### 2.3 `SmemLayout::Swizzle128` at 1-byte granularity

**DERIVED.** `desc_fields(Swizzle128 { lbo_bytes, swapped: false }, row_bytes, rows_per_desc)`
(`ptx_wgmma.rs:588-601`) reads `row_bytes` and nothing else. At `BK = 128` for a 1-byte type,
`row_bytes = 128` -- the same value the 16-bit kernel passes it. Therefore, **field for field:**

| field | 16-bit W1 | 8-bit retype at BK=128 | source |
|---|---|---|---|
| `lbo` | 16 (ignored by B128) | **16 (ignored by B128)** | `SHIPPED_LAYOUT`, `ptx_wgmma.rs:1660` |
| `sbo` | `8 * 128` = **1024** | `8 * 128` = **1024** | `desc_fields`, `:592` |
| `base_offset` | **0** | **0** | `:597`, and the sweep's `bo1/bo2` failures |
| `swizzle` | `SmemSwizzle::B128` | **`SmemSwizzle::B128`** | `:598` |
| `k_step_bytes` | **32** | **32** | `:599`, and 2.2 |
| TMA swizzle | `TmaSwizzle::B128` | **`TmaSwizzle::B128`** | `SmemLayout::tma_swizzle`, `:626-631` |
| start alignment | **1024 B** | **1024 B** | `SmemSwizzle::required_alignment`, `:426-433` |

Every one of `{1024, 16, 32}` is a multiple of 16 and far inside the 14-bit field
(`SmemDesc::pack`, `:673-716`), so the packer's two rejections cannot fire. The start addresses are
unchanged too, and they still clear the 1024-byte rule: a stage base is a multiple of
`49152 = 48 * 1024`; B's tile sits at `+16384 = 16 * 1024`; a consumer's `m64` A slab sits at
`+ w * 64 * 128 = w * 8192`. All multiples of 1024.

**The canonical-TMA claim, re-derived rather than assumed.** `CU_TENSOR_MAP_SWIZZLE_128B` applies a
permutation of **16-byte chunks within a 128-byte row over an 8-row atom** -- CUTLASS spells it
`Swizzle<3,4,3>`: `B=3` (8 patterns), `M=4` (a 16-byte base chunk), `S=3` (8 shifts), and the
permutation is `chunk_index XOR row_index`. Every parameter of that is a byte count and none of
them is an element count, so **the byte pattern a 128-B-swizzled TMA copy writes is a function of
the box's byte geometry alone**. Two copies with the same `box_dim[0] * elem_size` (both 128 B, per
1.5) and the same box row count write the same permutation. The 16-bit reading therefore transfers
*because* `BK` was doubled, not in spite of it. This is corroborated by CUTLASS's own layout atoms,
where `Layout_K_SW128_Atom` is `(8, 64):(64, 1)` composed with `Swizzle<3,4,3>` for a 2-byte
element and `(8, 128):(128, 1)` composed with **the same** `Swizzle<3,4,3>` for a 1-byte element --
same swizzle, different element extent, identical 128-byte row.

### 2.4 THE TRAP: what happens if `BK` stays 64

**DERIVED, and this is the derivation the implementer must not get wrong.**

At `BK = 64` with a 1-byte type, `row_bytes = 64`. Then:

* `desc_fields` computes `sbo = 8 * 64 = 512`, not 1024.
* The correct descriptor swizzle mode becomes `SmemSwizzle::B64`, whose required start alignment is
  **512** (`ptx_wgmma.rs:430`), and the correct TMA swizzle becomes `CU_TENSOR_MAP_SWIZZLE_64B`.
* **Nothing in the tree rejects it.** `TensorMapArgs::validate` only refuses `box_dim[0] * elem`
  *greater* than the atom (`tma_host.rs:326`); 64 B under a B128 declaration passes validation and
  produces a copy whose byte pattern is not the one the B128 descriptor expects to undo.
* And crucially: **the `Swizzle64` reading is not what the H100 round settled.** The sweep carried
  seven `sw128/*` rows and zero `sw64/*` rows. Shipping a 64-byte row would put the family back
  where it was on 2026-08-10 -- a plausible descriptor, no evidence, and a partial-exactness log to
  interpret.
* The stage geometry would also break the byte-identity: `tile_a = 128*64*1 = 8192`,
  `tile_b = 256*64*1 = 16384`, stage `= 24576 B`. At 4 stages that is 98368 B, so the SMEM budget,
  the occupancy verdict and the whole stage lattice would need re-derivation as well.

**Rule, stated so it can be a test:** for any `WgmmaCfg`, `bk * dtype.size()` must equal **128**.
That single predicate is what makes `SHIPPED_LAYOUT` correct, and it is checkable in
`WgmmaCfg::validate` with no device. It is the 8-bit family's equivalent of the `>> 4` truncation
check: cheap, total, and the only thing standing between a retype and a second paid sweep.

### 2.5 What a hypothetical 8-bit descriptor sweep would have to carry -- and why it should not run

`desc_sweep_candidates()` (`ptx_wgmma.rs:2584`) is still the right machinery if a *new* question
appears. But section 2.3 leaves no free parameter: the LBO is measured-ignored, the SBO is
measured-on-MN, the base offset is measured-zero, the anchor is measured-at-1024, and every one of
those verdicts is a statement about bytes that 1-byte elements do not touch. **Spending an H100 arm
on an 8-bit descriptor sweep would be re-buying an answer.** The honest device-free residual is
narrower and belongs in the correctness arm, not the sweep arm:

* **PREDICTION (the only one section 2 makes):** an 8-bit `wgmma` kernel emitted with
  `SHIPPED_LAYOUT`, `BK = 128`, `box_dim[0] = 128`, `CU_TENSOR_MAP_SWIZZLE_128B` scores
  **4096/4096 exact** on the first launch, at both a short-K and a long-K pass, with no descriptor
  arm behind it.
* **If it does not**, the diagnosis is *not* "try another LBO spelling" -- the mode ignores LBO.
  The diagnosis is that some byte-level assumption above is element-width-dependent after all, and
  the cheapest discriminator is a single extra arm at `Swizzle64 / BK=64`, which distinguishes
  "the swizzle is byte-defined" (our claim) from "the swizzle is element-defined" (the only
  alternative that would explain a failure) in one launch.

That is the whole 8-bit descriptor risk: one prediction, one named fallback, zero speculative arms.

---

## 3. DEQUANT / SCALE EPILOGUE

### 3.1 Where a thread's outputs actually are

**FACT(repo)**, `ptx_wgmma.rs:3658-3661` (the generator's own comment) and `:3670-3677` (the code
that computes it). With `grp = lane/4` and `tg = lane%4`, warp `w` of a consumer warpgroup holds

```
row0 = ctam + cwg*64 + w*16 + grp      row1 = row0 + 8
col(j) = ctan + 2*tg + 8*j             for j in 0 .. bn/8
```

and the four registers of group `j` are `(row0,col) (row0,col+1) (row1,col) (row1,col+1)`.

**DERIVED.** At `bn = 256` a thread owns **2 rows x 64 columns = 128 outputs**, in 32 register
quads. So a per-channel dequant needs, per thread, **2 row scales and 64 column scales**, and the
64 column scales are drawn from a set of only 256 that the whole warpgroup shares. That asymmetry
decides the implementation: row scales are 2 scalar `ld.global` (or `ld.global.nc`) per thread;
column scales are staged **once per CTA tile** into shared memory and read back as
`ld.shared.v2.f32`, because `col(j)` and `col(j)+1` are adjacent and `col(j)*4` is a multiple of 8
(`2*tg` is even), so the pair is a legal aligned `v2`.

### 3.2 int8: where the scales enter, and where the exactness gate must NOT

**DERIVED.** The instruction produces an exact `s32`:

```
D_s32[m,n] = sum_k A_s8[m,k] * B_s8[n,k]          exact for K < 131072  (section 1.3)
C_f32[m,n] = D_s32[m,n] * sa[m] * sb[n]           per-channel dequant
C_f32[m,n] = D_s32[m,n] * (sa * sb)               per-tensor dequant
```

Per output that is `cvt.rn.f32.s32` + two `mul.f32` (or one `mul` if the per-tensor product is
folded host-side), i.e. ~3 ALU ops against one 4-or-8-byte store. The epilogue stays store-bound.

**The guard that matters more than the arithmetic:** the dequant multiply is `f32` and rounds. The
`==` exactness claim belongs to `D_s32`, not to `C_f32`. So Wave 5 must ship **two arms, never
one**:

* an **s32-out arm** compared `==` against an `i32` scalar model -- this is the claim no library
  makes, and it is the surviving differentiator named in the plan's target table row 5;
* a **dequant arm** compared against an f64 reference at a tolerance.

The repo already has exactly this split at `mma.sync` (`gpu::tests::int8_gemm_matches_reference`
and `int8_dequant_matches_reference` are separate tests,
`bench/gpu/h100/2026-08-10-ptxas-census.log:306-307`), and the wgmma family must inherit the split
rather than collapse it. A single fused test would silently convert a bit-exactness claim into a
tolerance claim.

**Scale-vector SMEM cost.** `bn` f32 = **1024 B** at `bn = 256`. Free shared memory after the
mainloop ring is `227*1024 - 196672 = 35,776 B` (`HOPPER_SMEM_PER_CTA`, `ptx_wgmma.rs:184`, minus
the census-measured `smem(gen)`). 1 KiB fits with three orders of magnitude to spare -- but see
3.5, because it is not the only claimant on those bytes.

### 3.3 fp8: two different scaling regimes with two different costs

**DERIVED.** `wgmma` itself needs no scale beyond the +-1 sign immediates (section 1.2), so all fp8
scaling is ours to place, and *where* we place it decides what it costs:

**(a) Per-tensor scaling -- an epilogue multiply, free.** `C = D_f32 * (sA * sB)` where `sA`, `sB`
are the reciprocal-amax scalars the quantizer already produces (`ptx_fp8_train.rs`'s delayed-scaling
path; `gpu::tests::amax_matches_reference`, `fp8_device_quantize_within_ulp`). One `mul.f32` per
output, zero extra memory traffic, and it composes with the existing `alpha` fold.

**(b) Block scaling (DeepSeek-V3 style) -- a MAINLOOP change, and it does not fit on W1.** With
`1x128` activation blocks and `128x128` weight blocks, the scale varies along K, so it must be
applied at the boundary where the tensor-core partial is promoted to a CUDA-core f32 accumulator.
`BK = 128` makes that boundary **exactly one stage**, which is the good news. The bad news is
registers:

```
wgmma accumulators at bn=256      = n/2 = 128 regs/thread
promotion accumulators (f32)      = another 128 regs/thread
                                    ------------------------
                                    256 > consumer_regs = 232      DOES NOT FIT
```

At `bn = 128` (the `WGMMA_W3C` geometry, `ptx_wgmma.rs:1685-1692`) the same arithmetic reads
`64 + 64 = 128` accumulator registers inside `consumer_regs: 168`, which fits with headroom.

> **PREDICTION / design constraint the plan does not state: the two-level-accumulation arm belongs
> on the 128x128 tile, not on W1's 128x256.** Putting it on W1 either spills (a `ptxas` C7511 and a
> silent 2-4x) or forces `setmaxnreg` past the file. Emit it as its own row, at its own tile, and
> A/B it against the un-promoted W1 for *accuracy*, not for speed.

**Block-scale traffic, derived** (the plan's "+3.1% of operand bytes"): the activation side is one
f32 per 128 bytes of A = `4/128` = **3.125%**; the weight side is one f32 per `128*128` bytes of
B = `4/16384` = **0.024%**. The activation side is the whole cost, and the honest published number
is the scaled one.

### 3.4 Output dtype and store width

**DERIVED**, over the current epilogue (`ptx_wgmma.rs:3678-3696`: four predicated `st.global.f32`
per register quad, 128 scalar stores per thread at `bn = 256`):

| out dtype | per-quad emission | bytes/thread at bn=256 | C write, 128x256 tile |
|---|---|---|---|
| f32 (today) | 4 x `st.global.f32` | 512 | 131,072 B |
| f32 (W4 rung 1) | 2 x `st.global.v2.f32` | 512 | 131,072 B, half the sectors |
| f16 | 2 x (`cvt.rn.f16x2.f32` + `st.global.b32`) | 256 | **65,536 B** |
| s32 (int8 exact arm) | 2 x `st.global.v2.u32` | 512 | 131,072 B |
| s8 (requantised out) | needs a cross-lane gather | -- | **not free, do not promise it** |

The `f16` row is the one with a number attached: it halves the C write *and* deletes the separate
`f32 -> f16` cast pass, which is the mechanism behind the plan's `0.179 ms = 27% of the peer's
GEMM` at `gpt_d4096_up`. The `s8` row is called out because it looks symmetric and is not: a thread
holds columns `c` and `c+1` on **two different rows**, so `cvt.pack.sat.s8.s32.b32` (which wants
four column-adjacent values) cannot be fed from one thread's registers without a shuffle or an SMEM
transpose. An int8-out epilogue is a separate piece of work with its own cost; Wave 5 should ship
f32-out and f16-out and say so.

### 3.5 Interaction with the Wave-4 fused epilogue surface

Wave 4 turns the epilogue into a product surface (bias, relu/silu/gelu, `beta*C` residual,
`cvt.rn.f16x2` low-precision store, RoPE) and its **G10** is the sharp guard: W1's SMEM map leaves
**35,776 free bytes** against a **131,072-byte** f32 C tile, so a TMA-store epilogue physically
must alias the mainloop ring. Three notes where Wave 5 touches that surface:

1. **An f16-out epilogue does not rescue G10.** 65,536 B is still 1.8x the free budget. It halves
   the aliasing pressure and no more; the disjoint-region requirement stands.
2. **The dequant scale vectors are a second, much smaller claimant on the same 35,776 B** (1 KiB
   for `sb` at `bn=256`). They fit trivially, but they must be *in* the SMEM map, not carved out
   ad hoc, or `smem_bytes()` under-reports and `dyn_smem_bytes` under-requests -- the exact defect
   G10 already names for the epilogue region.
3. **Order matters and must be fixed once:** `dequant -> bias -> activation -> beta*C residual ->
   cast -> store`. Applying bias before dequant, or activation before the residual, are both
   plausible and both wrong, and neither is visible in a tolerance gate that only checks
   magnitudes. Write the order into the generator's doc comment and into the reference.

**Register budget, restated for Wave 5.** The plan's W4 note -- `128*32 + 256*232 = 63,488` of
`65,536` leaves exactly 8 registers per consumer thread, and `setmaxnreg` moves in steps of 8, so
"bias + act + residual together do not fit" -- is a **16-bit** statement, and section 1.4 shows the
8-bit retype does not change a single term of it. It transfers verbatim. Wave 5 inherits the
squeeze; it does not create or relieve it.

---

## 4. PEER BAR

### 4.1 The three-family CUTLASS profiler IS staged; the 8-bit MEASUREMENT artifacts do not exist yet.

**CORRECTION (orchestrator, 2026-08-11).** This section was first derived against
`bench/gpu/h100/2026-08-09-preflight-build-peers.log:915`, where the staged binary was f16-only.
That state is obsolete: Wave 2B's rebuild ran on 2026-08-10 (`--cutlass-max-kernels 1500`,
$0.187 CPU-only), and `peers.json` fetched from the wukong-build Volume on 2026-08-11 records

```
"cutlass": { "arch": "90a", "tag": "v4.6.1",
             "dtypes": ["f16", "fp8", "int8"],
             "counts": { "_selected": 1458, "f16": 368, "fp8": 782, "int8": 308 },
             "path": "/persist/bin/cutlass_profiler-sm90a-f16+fp8+int8" }
```

So the CPU-only rebuild step this section originally prescribed is **already done** -- do NOT
re-run `::build_peers --cutlass-dtypes ... --force`. The census denominators for "we beat the
best of N" are 368 f16 / 782 fp8 / 308 int8 SM90 kernels. What still does not exist is any
*measurement*: there is no `cutlass-e4m3-*.csv` or `cutlass-s8-*.csv` anywhere under
`bench/gpu/` -- the Wave-5 peer sequence therefore starts directly at the H100 `::cutlass`
profiler run (Section 4.2's configs), not at a build.

**FACT(repo), the unlock machinery** (Wave 2B's work, now both coded and executed):
`tools/cloud/modal_app.py:852-866` carries `_CUTLASS_KERNELS_3X` with
`"fp8": ("cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_*",)` and
`"int8": ("cutlass3x_sm90_tensorop_gemm_s8_s8_s32_*", "cutlass3x_sm90_tensorop_gemm_u8_u8_s32_*")`,
and `build_peers`'s `cutlass_dtypes` **defaults to `"f16,fp8,int8"`** (`:2739`). Two guards
already stand behind it and should be trusted rather than re-implemented:

* `_cutlass_census` (`:1005-1053`) counts selected kernels **per family before the compile** and
  exits if a requested family selected zero -- "the build would succeed, the binary would run, and
  the missing column would read as 'the library has no such kernel'".
* `::cutlass` refuses at run time to profile a dtype the manifest says the staged binary was not
  built with (`:3873-3880`), and prints a loud "unrecorded, assume f16 only" warning when the
  manifest predates 2026-08-10 (`:3870-3871`, `:3881-3884`) -- the currently staged binary's
  manifest is from the 2026-08-10 rebuild, so neither refusal fires for f16/fp8/int8.

**Therefore the Wave-5 peer sequence starts directly at the H100 `::cutlass` profiler run** over
Section 4.2's configs -- the CPU-only rebuild this dossier originally prescribed here already ran
on 2026-08-10 (see the CORRECTION at the top of 4.1), and its census (368 f16 / 782 fp8 /
308 int8 SM90 kernels in CUTLASS 4.6.1) is the denominator for "we beat the best of N".

### 4.2 Exactly which peer configs are the honest bar

`_CUTLASS_PROFILER_DTYPE` (`modal_app.py:983-992`) fixes the C type and accumulator per dtype, and
the accumulator is not cosmetic -- asking for `f32` on an `s8` kernel selects nothing and reads as
a missing library kernel.

| # | peer | invocation | why it is on the bar |
|---|---|---|---|
| 1 | CUTLASS fp8, f16 out | `::cutlass --dtype e4m3` (defaults `--c-dtype f16 --acc f32`) | the library's own best-of-N at the dtype we claim |
| 2 | CUTLASS fp8, f32 out | `::cutlass --dtype e4m3 --c-dtype f32` | matches **our** store width; without it the C-traffic term differs and the ratio is not about the mainloop |
| 3 | CUTLASS int8 | `::cutlass --dtype s8` (defaults `--c-dtype s32 --acc s32`) | the s32-out arm, same output type as our exactness arm |
| 4 | cuBLAS control | the `--providers=cutlass,cublas` column of 1-3 | same binary, same shapes: this is the dispersion control (G16) |
| 5 | cuBLASLt fp8 | `baselines.rs`'s existing raw-sys plan; `gpu::tests::cublaslt_fp8_matches_reference_within_tol`, `fp8_vs_cublaslt_pct` | already implemented; the vendor bar rather than the best-of-N bar |
| 6 | cuBLAS IMMA | the s8 `cublasGemmEx` path | the vendor int8 bar |
| 7 | vLLM `cutlass_scaled_mm` | `benchmark_int8_gemm.py`, already on the Volume | the **fused-dequant** int8 bar; the plan's target-table row 5 says the repo's "libraries do not offer fused dequant" framing is FALSE on Hopper and must be retired, and this is the kernel that retires it |

**The published headline must carry both peer columns.** The plan's own reason: "a library fp8 bar
at 58.7% of peak versus a cuBLAS f16 bar at 87.3% means the choice of peer, not the kernel, decides
whether the headline reads 97% or 73%." Report the peak-fraction column beside every ratio
(standing rule 4).

### 4.3 cuBLASLt FP8 -- the API constraints that decide whether the comparison is honest

**FACT(ext)**, cuBLAS documentation and the `CUDALibrarySamples/cuBLASLt/LtFp8Matmul` sample:

* **TN only on Hopper.** The FP8 matmul path does not support non-TN layouts on Hopper -- A must be
  transposed (row-major `K`-contiguous), B column-major. This is the *same* NT/K-major contract our
  kernel already uses (section 1.2), so the layouts match without a transpose fudge that would
  advantage either side. Note it explicitly in the round log, because "we and the peer are both TN"
  is a fairness fact, not a coincidence.
* **Leading dimensions must be a multiple of 16** for FP8 tensor scaling -- the same 16-byte rule
  section 1.5 derives for our TMA global strides. Both sides refuse the same shapes.
* **Scale pointers:** `CUBLASLT_MATMUL_DESC_{A,B,C,D}_SCALE_POINTER`, plus
  `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER`. Setting a scale pointer on an unsupported
  data/scale/compute-type combination returns `CUBLAS_INVALID_VALUE` rather than falling back --
  a hard error, which is the good kind.
* **`CUBLASLT_MATMUL_DESC_FAST_ACCUM` is the fairness switch, and it is the same switch as section
  3.3(b).** Fast-accum on is the un-promoted mode: the tensor core's 14-bit accumulation, straight
  through. Fast-accum off is the peer's version of DeepSeek promotion.

> **REFUSAL (Wave 5's own, added here): do not compare our promoted kernel against a
> `FAST_ACCUM=true` peer, or our un-promoted kernel against a `FAST_ACCUM=false` peer.** Either
> direction publishes an accuracy/speed trade as a speed result. Match the mode, say which mode, or
> publish both columns. The plan's existing refusal -- "any fp8 headline against a per-tensor-scaled
> peer" -- is the same defect on the scaling axis; this is its accumulation-axis twin.

### 4.4 Machete / Marlin -- trigger status: NOT TRIGGERED by Wave 5, with one caveat

**FACT(repo)**, `ACT2_WAVE_PLAN.md:102` holds "Beating Machete on W4A16" out of the plan
(measure-only, a `::marlin` round at M=1/16/128), and `:77` states the Wave-5 refusal: "an int4
headline without the Machete column".

**DERIVED.** Wave 5's levers are `{E4M3, E5M2, S8, U8}`. There is no int4 `wgmma` -- the ISA's
8-bit menu is where `wgmma` stops -- so **Wave 5 cannot produce an int4 headline and the Machete
trigger does not fire.** The caveat is section 5: the baseline round *measures* the existing
`ptx_int4.rs` family on Hopper to get a Hopper denominator. That is a baseline, not a headline, and
it stays inside the refusal as long as no int4 **ratio against a peer** is published from it.

`::marlin` also picks the right kernel by device (`modal_app.py:3948-3957`): Machete on `sm_90`,
Marlin on Ampere, with a loud warning in each wrong direction. When the int4 trigger does fire, use
that dispatcher rather than naming a kernel by hand.

---

## 5. BASELINE ROUND SPEC -- the Hopper denominator, before any Wave-5 code exists

The plan asks for this round *before* implementation, and the reason is one sentence: **every
Wave-5 projection is a ratio whose denominator is currently a 4050 memory.** `2.89x` and `1.48x`
are instruction-issue ratios from a published H800 table; what they multiply is our own measured
8-bit throughput, and we have never measured it on Hopper.

### 5.1 What to run

Existing families, existing kernels, no new code. All three groups run through the prebuilt Volume
binary (`_require_prebuilt` refuses to compile on metered silicon, `modal_app.py:8-13`), in the
same container, in this order:

| group | filter | tests it selects (from the 2026-08-10 census test list) |
|---|---|---|
| fp8 | `fp8_` | `fp8_gemm_matches_reference_within_tol`, `fp8_pipe_matches_reference_within_tol`, `fp8_pipe_regime_matches_reference`, `fp8_pipe_vs_peers`, `fp8_vs_cublaslt_pct`, `fp8_tensorcore_tile_matches_reference`, `cublaslt_fp8_matches_reference_within_tol` |
| int8 | `int8_` + `quant_int8_` | `int8_gemm_matches_reference`, `int8_gemm_vs_peers`, `int8_dequant_matches_reference`, `quant_int8_fused_dequant_vs_chain`, `quant_int8_w64_confirm`, `quant_int8_w64_s3_sweep`, `quant_int8_bigtile_sweep`, `quant_int8_raster_sweep` |
| int4 | `int4_` | `int4_gemm_matches_reference`, `int4_gemm_vs_peers`, `int4_splitk_occupancy` |
| denominator | `tensorcore_roofline_pct`, `gpu_target_is_probed_and_sane` | the peak-fraction column (standing rule 4) and the device identity |

Invocation shape, copied from `WGMMA_SWEEP_INVOCATION` (`ptx_wgmma.rs:2086-2088`) because the same
two hazards apply -- a run without `--nocapture` throws away everything the rented minutes produced,
and `--test-threads=1` is what keeps the timings readable:

```
WUKONG_GPU_REQUIRED=1 WUKONG_PEER_REQUIRED=1 cargo test -p wukong_codegen_gpu --features gpu \
    --release -- --ignored --nocapture --test-threads=1 <filter>
```

### 5.2 What each number decides

| number | decides |
|---|---|
| `int8_gemm_vs_peers` on H100 | **Whether Ada-tuned int8 is already competitive on Hopper without wgmma.** If it lands high against cuBLAS IMMA, the 1.48x issue headroom is the entire remaining story and W5-int8 is close to a pure retype. If it lands low, the Ada tuning (w64 warp tile, XOR swizzle, stage count) does not transfer and W5-int8 needs its own tile search -- which is exactly what the f16 Act-1 -> Act-2 transition taught, and it would move int8 from "cheap" to "a wave of its own". |
| `fp8_pipe_vs_peers` and `fp8_vs_cublaslt_pct` on H100 | **Whether the 2.89x claim's denominator is real.** The plan asserts `ptx_fp8.rs` runs at fp16 issue rate on Hopper. Run the fp8 kernel and the f16 kernel of the *same* tile back to back in the same container: if the absolutes coincide, the assertion is confirmed on this device and 2.89x stands. If fp8 is measurably faster, the denominator (490.7) is wrong for this part and the projected headroom shrinks -- a finding worth more than the round costs. |
| `int4_gemm_vs_peers` | **Whether D1's "INT4 `mma` on Hopper degenerates to `IMAD` on the CUDA cores" reproduces.** This is a $0-marginal sanity check on the whole Luo-et-al table that Wave 5's projections rest on. Expect a catastrophic Hopper number and publish it as a *baseline*, never as a ratio (section 4.4). |
| `quant_int8_fused_dequant_vs_chain` | The **current** fused-dequant advantage, which is the thing vLLM's `cutlass_scaled_mm` will be measured against in the Wave-5 round proper. Measuring it now separates "wgmma helped" from "fusion helped". |
| `tensorcore_roofline_pct` | The peak-fraction denominator for every ratio the wave later publishes. |

### 5.3 The cost line, checked rather than repeated

**DERIVED.** `modal_app.py:373-379` prices H100 at **$3.95/hr** = $0.001097/s, so the plan's
**$0.06** buys **~55 seconds** of H100 wall time, plus the CPU/memory adder. That is achievable
only with a warm Volume target dir and a tight filter; a full `--include-ignored` sweep of the
three families will not fit in it. State the intended budget in the round preamble and let the
meter (`_meter`, which prints the ceiling before the call and the actual after) settle it -- the
honest options are "narrow the filter to fit $0.06" or "declare a $0.25 round", not "hope".

### 5.4 The census rows to add to `wgmma_all_emittable`

**FACT(repo).** `wgmma_all_emittable()` (`ptx_wgmma.rs:2098-2106`) is `WGMMA_VARIANTS` plus every
generatable `WGMMA_SWEEP_GRID` row, deduplicated **by module key**;
`wgmma_device_free_modules()` (`:3728-3747`) turns it into the corpus that the ASCII rule, the
`.target` floor, the `.version` law and the CPU ptxas census all scan; and `gpu.rs:8336` asserts
`EXPECTED_MODULES = 109` over the crate-wide union.

**DERIVED.** Proposed Wave-5 rows, all at `BK = 128` so `bk * dtype.size() == 128` (section 2.4):

| row | tile | why |
|---|---|---|
| `wgmma_nt_e4m3_128x256x128_s4` | W1 | the fp8 forward centerpiece |
| `wgmma_nt_e5m2_128x256x128_s4` | W1 | the backward-gradient type; one token pair from the above |
| `wgmma_nt_s8_128x256x128_s4` | W1 | the int8 centerpiece, s32 out |
| `wgmma_nt_e4m3_128x128x128_s6` | W3c | **the two-level-accumulation arm** -- section 3.3(b) shows it does not fit on W1 |
| `wgmma_nt_s8_128x128x128_s6` | W3c | the int8 moderate-size twin |

Five rows takes `EXPECTED_MODULES` 109 -> 114, and the constant must move **in the same commit** or
the crate-wide law fails -- which is the intended behaviour: it is what forces the count to be
thought about. If the B-multicast arm has landed from Wave C1 by then, each `_mcb2` twin is a
further distinct module because `WgmmaCfg::derived_name` encodes the whole geometry including the
multicast axis.

**The census answers four things for $0.02, and all four are otherwise H100 questions:**

1. **The N menu at K=32.** ptxas rejects an off-menu shape token (section 1.1).
2. **Whether `.satfinite` assembles on the integer form** (section 1.3) -- ask, then decline it
   anyway, with the answer recorded.
3. **Registers and spills.** The prediction is `regs 168, spill 0, smem(gen) 196672` on every 8-bit
   row, identical to the f16 row at `bench/gpu/h100/2026-08-10-ptxas-census.log:687`.
4. **C7511.** "wgmma instructions are serialized due to insufficient register resources" is a
   **silent 2-4x, not a failure** (G14). The 2026-08-09 CUTLASS build produced 112 of them from
   NVIDIA's own kernels; ours must produce zero, and a nonzero count is a finding that blocks the
   H100 visit rather than a warning to scroll past.

> **LANDMINE, from the census log itself.** The 2026-08-10 run exited **1** with "ptxas audit
> incomplete: no ptxas records parsed" (`:842-869`) -- the audit test printed its own table and the
> harness's `ptxas -v` parser read nothing out of it, so the per-arch coverage section reported
> `MISSING: sm_80, sm_89, sm_90a` even though the table above it has the sm_90a rows. **The
> register numbers in that log are real; the harness's verdict on them is not.** Wave 5 must not
> read that exit code as "the census failed", and should fix the parse or narrow the claim before
> relying on the census as a gate.

---

## 6. GUARDS -- law text

Each item below is written so it can become a test name and an assertion message, not a paragraph
someone has to re-derive.

### G-W5-1. The byte-geometry law (the one that protects the paid answer)

> **`WgmmaCfg::validate` must reject any configuration for which `bk * dtype.size() != 128`.**
>
> `SHIPPED_LAYOUT` is `Swizzle128 { lbo_bytes: 16, swapped: false }`, and every field of it --
> `SBO = 8 * row_bytes = 1024`, `base_offset = 0`, `k_step_bytes = 32`, `TmaSwizzle::B128`, the
> 1024-byte start alignment -- is correct **only** at `row_bytes == 128`. That reading was settled
> on hardware for 16-bit at `BK = 64` (`2026-08-10-h100-s2c-desc-sweep.log:178-182`) and is
> byte-invariant, not element-invariant. A 1-byte row at `BK = 64` needs `Swizzle64`, `SBO = 512`
> and a 512-byte alignment -- a reading nothing has measured -- and `TensorMapArgs::validate` will
> **not** catch it, because it only refuses a contiguous extent *greater* than the swizzle atom
> (`tma_host.rs:326`).

Device-free, total, one line of arithmetic. It is the highest-value assertion in the wave.

### G-W5-2. Capability: `require_fp8` is necessary and NOT sufficient on sm_90a

**FACT(repo).** `FP8_MIN_CC = (8, 9)` (`gpu.rs:150`) and `Gpu::supports_fp8` is
`target.supports((8,9))` (`:560-563`). Hopper is cc `9.0`, so **`require_fp8` can never decline on
an H100** -- it is satisfied by construction and therefore gates nothing about this family.

The binding gate is the type: `wgmma_module` cannot be called without an `Sm90aLicense`
(`ptx_wgmma.rs:135-139`), which only `require_sm90a` mints from a probed `9.x`. Law text:

> **Every 8-bit `wgmma` launcher must call `require_sm90a` AND `require_fp8` (fp8 rows only).**
> The license is the real gate; the `require_fp8` call exists to satisfy the textual scan
> `every_fp8_module_load_is_capability_gated`, which is a source scan and cannot see a type. Do not
> "simplify" it away: the scan's corpus is files, and dropping the call would make an fp8-naming
> launcher invisible to it. Document at the call site that on `sm_90a` it is subsumed, so the next
> reader does not conclude the gate is doing work it is not.

The failure this prevents is precise and would otherwise be confusing: without the license the fp8
textual law **passes** (the module names `require_fp8`) while `cuModuleLoadData` refuses the module
on any non-Hopper part, because the header is `HDR_SM90A_V80` and `sm_90a` is
architecture-*specific* in both directions.

### G-W5-3. `.version`: do not let a quantizer drag the mainloop to the r550 floor

**FACT(repo).** `gpu.rs`'s durable law: "a module may declare an r550+ driver floor only if it
emits an instruction that needs one". The wgmma family is licensed at `.version 8.0` by `wgmma` +
`cp.async.bulk` (`gpu.rs:8318-8319`). `ptx_fp8_train.rs` declares `HDR_SM89_V84` because it emits
`cvt.rn.satfinite.e4m3x2.f32` (`:22`, `:317`, `:322`).

> **The 8-bit `wgmma` mainloop module must contain no `cvt.rn.satfinite.e{4m3,5m2}x2.f32`.**
> Quantization stays in its own module at its own floor. Folding a quantize prologue into the
> mainloop would drag an `sm_90a`/`8.0` module to an `8.4` driver floor for no reason, and
> `only_the_ada_floor_declares_the_r550_driver_version` plus
> `the_family_declares_no_floor_it_does_not_need` would both have to be weakened to allow it.

### G-W5-4. ASCII, and where an 8-bit generator will break it

`wgmma_ptx_is_pure_ascii` scans `wgmma_device_free_modules()`, so a new 8-bit generator is covered
**iff** its config reaches `wgmma_all_emittable()`. Law text:

> **A new 8-bit row is added to `WGMMA_VARIANTS` or `WGMMA_SWEEP_GRID` in the same commit as its
> generator, and `EXPECTED_MODULES` moves with it.** A module outside that enumeration is outside
> four laws at once -- ASCII, `.target`, `.version` and the ptxas census (G14).

The specific 8-bit hazard is prose: an emitted comment describing "e5m2 x e4m3" or "K = 32 -> 2
core matrices" with a Unicode multiplication sign or arrow is a `ptxas fatal` at
`cuModuleLoadData`, at the worst possible moment, from a comment.

### G-W5-5. G3 naming: the dtype AND the `bk` must be in the derived name

> **`WgmmaCfg::validate` must reject any row whose `name`/`key` is not derivable from its own
> geometry, and for 8-bit that geometry now includes `bk`.**

`wgmma_nt_e4m3_128x256x64_s4` and `wgmma_nt_e4m3_128x256x128_s4` are different modules with
different descriptors. `Gpu::function` keys on the module key alone and never re-examines the text
(`ptx_wgmma.rs:2094-2097`), so a `WgmmaCfg { dtype: E4M3, bk: 128, ..WGMMA_W1 }` that forgets to
override `name` and `key` gets **W1's cached f16 module** back and bills its launches to a
configuration that never ran. This is the plan's named "#1 sweep hazard", and the 8-bit retype adds
two new fields it can hide in.

### G-W5-6. G15 dtype-aware `ramp_radix` -- and the finding that the fp8 exact arm barely exists

**FACT(repo).** `ramp_code(w, d0, d1, d2) = 1 + d0%w + w*(d1%w) + w^2*(d2%w)`
(`ptx_wgmma.rs:2136-2138`), so the largest operand value is `1 + (w-1)(1 + w + w^2)`.
`ramp_radix(k)` (`:2125-2130`) picks the largest `w` in `{8,4,2}` with `k * w^6 <= 2^24`, i.e. it
bounds only the **accumulation**, against a hardcoded f32 limit.

**DERIVED, both bounds, per dtype:**

| dtype | significand bits | exact-integer limit | max `w` by **value** (`1+(w-1)(1+w+w^2) <= limit`) | accumulation limit |
|---|---|---|---|---|
| f16 | 11 | 2048 | 8 (gives 512) | 2^24 |
| bf16 | 8 | 256 | 4 (gives 64; w=8 gives 512 -- **fails**) | 2^24 |
| e4m3 | 4 | **16** | **2** (gives 8; w=4 gives 64 -- fails) | **2^14** |
| e5m2 | 3 | **8** | **2** (gives 8, exactly at the limit) | **2^14** |
| s8 | integer | 127 | **4** (gives 64; w=8 gives 512 -- fails) | 2^31, exact |

The **2^14** accumulation limit for fp8 is not f32's 2^24: Hopper's fp8 tensor core right-shift-
aligns the 32 mantissa products to the maximum exponent and keeps only the **top 14 bits** of each,
truncating the rest (DeepSeek-V3 technical report; corroborated by Colfax's write-up,
<https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/>). So the fp8
ramp must satisfy `k * w^6 <= 2^14` as well as `w <= 2`:

```
w = 2  =>  k * 64 <= 16384  =>  k <= 256
```

> **FINDING: the exact-integer bring-up arm is only available for fp8 at `K <= 256`, and only at
> radix 2.** Above that, no radix in `{8,4,2}` satisfies the bound and the arm cannot be
> constructed at all. This is not a tuning inconvenience -- it means **G2's pseudorandom f64-
> reference arm is the primary correctness instrument for the fp8 family, not the secondary one**,
> and it is the derivation behind the plan's "G2 is a hard prerequisite, not a nicety". The
> int8 family keeps its `==` arm at every K (radix 4, accumulation exact in s32), which is precisely
> why int8 is the wave's low-risk half and fp8 is not.

Law text:

> **`ramp_radix` takes the dtype.** It enforces `1 + (w-1)(1 + w + w^2) <= dtype.exact_integer_limit()`
> **and** `k * w^6 <= dtype.exact_accumulation_limit()`, and it **returns an error rather than a
> radix** when no `w` satisfies both -- so a caller that cannot have an exact arm is told, instead
> of silently getting a ramp that rounds and an unexplained near-miss.

### G-W5-7. Exactness gates: two arms for int8, a derived tolerance for fp8

> **int8: `==` against an `i32` scalar model, on the s32 output, at every gate shape.** The bound
> that makes it legitimate is `|a*b| <= 16384` and `16384 * K < 2^31`, i.e. **`K < 131072`**
> (section 1.3), which every campaign shape clears by an order of magnitude. The dequantised f32
> output gets a *separate* tolerance arm (section 3.2). A single fused test is a downgrade of the
> claim.

> **fp8: `c * sqrt(K) * eps` against an f64 reference, with `eps = 2^-14 = 6.104e-5` -- not f32's
> `2^-24`.** The 14-bit figure is the hardware's product-alignment width, so the tolerance is
> *derived from the mechanism*, which is the only kind that may be widened. Two further rules:
>
> * **The f64 reference is computed over the exactly-decoded fp8 operand values**, never over the
>   pre-quantization f32. Otherwise the input rounding (half-ulp `2^-4` for e4m3, `2^-3` for e5m2)
>   swamps everything and the gate measures the quantizer instead of the kernel.
> * **Publish which accumulation mode produced the number.** With DeepSeek promotion every `K = 128`
>   (== one `BK` stage), the tensor-core accumulator is reset per stage, so the alignment maximum is
>   taken over 128 products rather than over the whole running sum and the dominant term loses its
>   K-dependence. Un-promoted and promoted are two different tolerances and two different speeds;
>   reporting one number from a mixed set is the section-4.3 fairness defect turned inward.
>
> **Never widen `c` to make a run pass.** If the measured error exceeds the derived bound, the
> derivation is wrong or the kernel is -- and either is a finding.

### G-W5-8. `TmaOobFill::Zero` still means zero at 8 bits (G20, re-checked rather than assumed)

The out-of-bounds law works because a zeroed byte pattern is a zero *value*, so a ragged M/N/K edge
contributes exactly nothing to the accumulator and only the epilogue needs a bounds check
(`tma_host.rs:137-153`). **Re-derived for the new types:** `0x00` is `+0.0` in e4m3 and in e5m2
(both use sign-magnitude with an all-zero exponent and mantissa), and `0` in s8 and u8. All four
are true zeros, so the law transfers.

It is worth one line because it is *not* automatic: it is a property of these four encodings, and a
future 1-byte format whose `0x00` is not zero (a biased or offset encoding) would break the whole
ragged-shape story silently.

### G-W5-9. G19 `scale-d` is unchanged, and still the thing a tile loop breaks

`scale-d = %pfirst = (kt != 0)` (`ptx_wgmma.rs:3003-3004`) has the same meaning in all three 8-bit
forms -- it is the first positional operand of every tail in section 1.2. The Wave-3 hazard
restated for Wave 5: **in a persistent/tile-loop kernel `kt != 0` has no meaning across tiles**, and
tile 2 accumulating into tile 1 is an *exact integer* in the int8 arm, hence invisible to the `==`
oracle that is otherwise the wave's strongest gate. Assert the `mov.u32 %kt,0` count equals the
tile-loop entry count, exactly as G19 specifies.

### G-W5-10. Round ordering (standing rules 1 and 3)

> **The $0.02 CPU ptxas census runs before the H100 is rented, over
> `wgmma_device_free_modules()`, and the 8-bit descriptor sweep and correctness arms run before the
> peer round in the same container.** The plan states the order; section 5.4 states the four
> questions the census answers for free, and section 2.5 states that an 8-bit *descriptor* sweep
> should not be one of the arms at all -- the descriptor question is closed, and re-opening it as a
> sweep would be re-buying an answer this dossier already derives.

---

## 7. THE SHORT LIST -- what an implementer must not get wrong

1. **`bk * dtype.size() == 128`.** Everything in section 2 hangs off it. `BK` goes 64 -> 128 for
   1-byte types, and the whole hardware-settled descriptor transfers unchanged. Leave `BK` at 64
   and the family silently needs a `Swizzle64` reading nobody has measured. **This is the one.**
2. Three different operand tails; the transpose immediates are *absent*, not zero; `imm-scale-a/b`
   are sign flips, not quantization scales.
3. The int8 `==` claim lives on the **s32** output. Dequant is a separate arm.
4. Two-level fp8 accumulation does not fit on the 128x256 tile. Put it on 128x128.
5. The fp8 tolerance is `2^-14`, derived from the hardware's 14-bit product alignment, and the f64
   reference must be taken over the *decoded fp8* operands.
6. The CUTLASS fp8/int8 profiler does not exist yet. Build it on CPU first, or the peer column is
   an artifact of our own build.
7. Name and key every new row from its own geometry, `bk` included, or `Gpu::function` hands back
   the f16 module and the round bills launches to a config that never ran.

---

## 8. SOURCES

### In-tree (FACT(repo)) -- these outrank everything below

| what | where |
|---|---|
| The crowned 16-bit descriptor, 19 candidates x 2 K passes | `bench/gpu/h100/2026-08-10-h100-s2c-desc-sweep.log:93-182` |
| The first round that found the row-major reading wrong (64/4096) | `bench/gpu/h100/2026-08-10-h100-s2a-bringup.log` |
| ptxas census: `wgmma_nt_f16_128x256x64_s4` regs/smem/threads | `bench/gpu/h100/2026-08-10-ptxas-census.log:687-690` |
| ptxas census harness landmine (exit 1, "no records parsed") | same log, `:842-869` |
| CUTLASS profiler staged f16-only (OBSOLETE 2026-08-10: rebuilt f16+fp8+int8, see 4.1 CORRECTION) | `bench/gpu/h100/2026-08-09-preflight-build-peers.log:915`; `peers.json` on the Volume |
| 112 C7511 warnings from NVIDIA's own SM90 f16 kernels | same log, `:1074-1185`; ledger entry 4 in this directory's README |
| The wgmma model: core matrix, LBO/SBO, `SmemLayout`, `SHIPPED_LAYOUT`, `desc_fields`, `SmemDesc::pack` | `crates/wukong_codegen_gpu/src/ptx_wgmma.rs:44-78, 436-716, 1660-1663` |
| Tile/stage/SMEM/register accessors and the shipped rows | same file, `:983-1160`, `:1612-1774` |
| The emitted `wgmma` line and the epilogue | same file, `:3003-3007`, `:3657-3696` |
| The module corpus the four laws scan | same file, `:2098-2106`, `:3728-3747`; `gpu.rs:8243`, `:8336` |
| TMA data types, validation rules, OOB fill | `crates/wukong_codegen_gpu/src/tma_host.rs:54-153, 246-335` |
| fp8 capability constant and `require_cap` | `crates/wukong_codegen_gpu/src/gpu.rs:147-150, 560-575` |
| fp8 mixed-type `mma.sync` precedent and the `.version 8.4` split | `crates/wukong_codegen_gpu/src/ptx_fp8_train.rs:5-25, 190, 317-323` |
| PTX ISA shape menu, extracted verbatim by the campaign | `docs/gpu/derive/D1_h100_gemm.md:154-173` |
| H800 tensor-core instruction throughput table (Luo et al.) | `docs/gpu/derive/D1_h100_gemm.md:114-134` |
| CUTLASS kernel filters, census guard, profiler dtype table, meter rates | `tools/cloud/modal_app.py:373-379, 819-1053, 2736-2768, 3830-3900, 3948-3957` |
| Wave-5 levers, targets, refusals, standing rules | `docs/gpu/derive/ACT2_WAVE_PLAN.md:67-77, 102, 114, 122-128` |

### External (FACT(ext)) -- used only where the tree is silent

- PTX ISA, "Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions"
  (section 9.7.16 in the current doc, 9.7.14 in the PTX-8.4 archive):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma>
  (archive: <https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/contents.html>)
- NVIDIA CUTLASS, Warpgroup MMA Programming Guide -- per-type instruction K (16 for f16/bf16, **32**
  for `MmaF8Op` and `MmaI8Op`), `M = 64` and `8 <= N <= 256` in steps of 8, and "FP8 and INT8
  variants are K-major only":
  <https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/mma_docs/wgmma_programming.html>
- Colfax Research, "CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs"
  -- core matrix "8 in the strided direction and 16 bytes in the contiguous direction", "K is fixed
  to be 32 bytes", 128-byte swizzle `SBO = 128 x 8 = 1024`, and the transpose restriction to
  f16/bf16: <https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/>
- CUTLASS `include/cute/arch/mma_sm90_desc.hpp` -- the GMMA descriptor bitfields and the
  `INTERLEAVE=0, B128=1, B64=2, B32=3` encoding, with `base_offset_` "valid only for B128 and B64":
  <https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_desc.hpp>
- DeepSeek-V3 on Hopper's fp8 accumulation -- 32 mantissa products right-shift-aligned to the
  maximum exponent, only the top **14 bits** kept, and promotion to CUDA-core f32 every 128 elements
  (4 wgmma): <https://github.com/deepseek-ai/DeepSeek-V3/issues/197>, and Colfax's write-up
  <https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/>
- Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture",
  <https://arxiv.org/abs/2402.13499> -- the source of D1's instruction-throughput table and of the
  "INT4 `mma` degenerates to IMAD on Hopper" fact. **The plan's `1417.2 / 490.7 = 2.89x` and
  `1448.7 / 977.9 = 1.48x` are not yet transcribed into D1**; Wave 5 should copy those two rows into
  D1 section 1.2 with their table numbers, so the wave's headline ratio has an in-tree provenance
  line rather than a plan sentence.
- cuBLASLt FP8 matmul -- TN-only on Hopper, leading dimensions a multiple of 16, the
  `CUBLASLT_MATMUL_DESC_{A,B,C,D}_SCALE_POINTER` / `AMAX_D_POINTER` attributes returning
  `CUBLAS_INVALID_VALUE` on an unsupported combination, and `CUBLASLT_MATMUL_DESC_FAST_ACCUM`:
  <https://docs.nvidia.com/cuda/cublas/> and the reference sample
  <https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtFp8Matmul/sample_cublasLt_LtFp8Matmul.cu>

### Open, and deliberately left open

* Whether `.satfinite` assembles on the integer `wgmma` form (section 1.3) -- a $0.02 ptxas
  question, and the answer does not change the decision to decline it.
* The exact N menu at `K = 32`, if it is narrower than the f16 menu (section 1.1) -- same $0.02
  instrument. D1's verbatim ISA extraction says "same N menu"; ptxas is the cheap confirmation.
* The 2.89x / 1.48x issue-rate rows' provenance line in D1 (above).
