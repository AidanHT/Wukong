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
