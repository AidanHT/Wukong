//! `tma_host` — the **host half of TMA**: building the 128-byte `CUtensorMap` that every
//! `cp.async.bulk.tensor` instruction dereferences, and stating the encoding rules as pure,
//! device-free data so they can be proven without a Hopper part.
//!
//! # Why this module has to exist
//!
//! TMA (the Tensor Memory Accelerator, `cp.async.bulk.tensor.{1d..5d}`) is the only global->shared
//! copy engine on Hopper that a `wgmma` mainloop can be fed from at full rate, and unlike every
//! other instruction this crate emits it is **not self-describing**. The instruction takes a
//! *tensor map* operand: a 128-byte opaque object that encodes the tensor's rank, its global
//! dimensions and byte strides, the shape of the tile ("box") to copy, the shared-memory swizzle the
//! copy applies, and what to write for out-of-bounds elements. That object is built **on the host**
//! by `cuTensorMapEncodeTiled` and passed to the kernel as a by-value grid-constant parameter. It is
//! not a PTX construct and there is no way to synthesize it from device code.
//!
//! Before this module, `wukong_codegen_gpu` made no `cuTensorMap*` driver call at all (D1 §6
//! finding 4). This is that surface, and nothing else: it builds and validates descriptors. It
//! launches nothing, allocates nothing, and knows nothing about GEMM.
//!
//! # The shape of the proof
//!
//! Every argument `cuTensorMapEncodeTiled` takes is **pure data** derived from a tensor's geometry,
//! and every documented precondition on those arguments is a pure predicate over that data. So the
//! whole encoding is split in two:
//!
//! * [`TensorMapArgs`] — the argument bundle plus [`TensorMapArgs::validate`], which enforces every
//!   driver-documented precondition. Un-gated (no `cudarc`), so it is built and checked in a plain,
//!   toolchain-free, device-free `cargo test`. **This is where the real proof lives.**
//! * [`TensorMap`] — a thin `gpu`-gated wrapper that calls the driver once. Its single `unsafe`
//!   block states and checks its preconditions ahead of the call (crate rule #2), and it refuses to
//!   call the driver at all with arguments [`TensorMapArgs::validate`] rejects, because
//!   `CUDA_ERROR_INVALID_VALUE` from inside the encoder names none of them.
//!
//! # The one fact worth memorising
//!
//! Descriptor dimension **0 is the CONTIGUOUS dimension** — the fastest-varying axis of the tensor
//! in memory, and the one whose extent in *bytes* the swizzle mode constrains. For a row-major
//! `rows x cols` matrix that means `global_dim = [cols, rows]`, not `[rows, cols]`. Getting this
//! backwards produces a descriptor the driver accepts and a kernel that reads a transposed,
//! garbage tile — silently. [`TensorMapArgs::tiled_2d_row_major`] exists so no caller has to
//! remember it.

/// Bytes in a `CUtensorMap`. Fixed by the driver ABI (`CUtensorMap_st` is `[u64; 16]`), and the
/// number a PTX kernel parameter must declare — see [`ptx_param_decl`].
pub const TENSOR_MAP_BYTES: usize = 128;

/// Required alignment of a `CUtensorMap`, in bytes (`#[repr(align(64))]` on the driver struct, and
/// the `.align` a PTX `.param` declaration of it must carry).
pub const TENSOR_MAP_ALIGN: usize = 64;

/// The maximum tensor rank `cuTensorMapEncodeTiled` accepts.
pub const TMA_MAX_RANK: usize = 5;

/// Largest extent, in elements, of one box (tile) dimension. Driver-documented limit.
pub const TMA_MAX_BOX_DIM: u32 = 256;

/// Largest `elementStrides[i]` the driver accepts.
pub const TMA_MAX_ELEMENT_STRIDE: u32 = 8;

/// The **element type** the tensor map describes. Values are the driver's own
/// `CUtensorMapDataType` discriminants, restated here so the whole argument bundle is device-free
/// data (the `gpu`-gated [`TensorMap::encode`] converts them back). Only the types this backend
/// actually moves are listed — a missing one is a deliberate decline, not an oversight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TmaDataType {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    F16 = 6,
    F32 = 7,
    Bf16 = 9,
}

impl TmaDataType {
    /// Bytes per element. The swizzle and 16-byte-multiple rules are stated in *bytes*, so every
    /// validation multiplies a box extent by this.
    pub const fn size(self) -> usize {
        match self {
            TmaDataType::U8 => 1,
            TmaDataType::U16 | TmaDataType::F16 | TmaDataType::Bf16 => 2,
            TmaDataType::U32 | TmaDataType::F32 => 4,
        }
    }
}

/// The shared-memory swizzle the TMA copy applies as it writes the tile.
///
/// **This must agree with the `wgmma` shared-memory descriptor's own swizzle field**
/// (`ptx_wgmma::SmemSwizzle`): TMA writes the pattern, `wgmma` reads it back, and each has an
/// independent 2-bit encoding of the same choice. Two encodings of one fact is how an oracle and a
/// kernel diverge, so `ptx_wgmma` derives its value from this enum rather than from a second table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TmaSwizzle {
    None = 0,
    B32 = 1,
    B64 = 2,
    B128 = 3,
}

impl TmaSwizzle {
    /// The swizzle atom width in bytes, and therefore the **upper bound on `boxDim[0] * elem_size`**
    /// when the mode is not `None`. A tile whose contiguous extent exceeds its swizzle atom cannot
    /// be described at all — the driver rejects it, and there is no partial form.
    pub const fn atom_bytes(self) -> usize {
        match self {
            TmaSwizzle::None => 0,
            TmaSwizzle::B32 => 32,
            TmaSwizzle::B64 => 64,
            TmaSwizzle::B128 => 128,
        }
    }
}

/// Interleave mode. Only `None` is supported here: the interleaved modes exist for image formats,
/// carry their own alignment rules, and no GEMM/attention operand uses them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TmaInterleave {
    None = 0,
    B16 = 1,
    B32 = 2,
}

/// L2 promotion hint — how wide a footprint the copy asks L2 to fetch. A hint only; it changes no
/// result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TmaL2Promotion {
    None = 0,
    B64 = 1,
    B128 = 2,
    B256 = 3,
}

/// **What TMA writes for an element the box covers but the tensor does not.**
///
/// This is the single most useful property of a TMA mainloop and the reason a `wgmma` GEMM needs no
/// bounds predication on its loads: `Zero` makes every out-of-range element a hard zero, so a ragged
/// `M`, `N` or `K` contributes exactly nothing to the accumulator and the *only* place a shape check
/// is still needed is the epilogue store.
///
/// `NanRequestZeroFma` fills NaN instead, which the tensor cores then treat as zero for FMA — useful
/// for distinguishing "padded" from "genuinely zero" data. Not used by this backend; `Zero` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TmaOobFill {
    /// `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` — out-of-bounds elements read as zero.
    Zero = 0,
    /// `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA`.
    NanRequestZeroFma = 1,
}

/// **The complete argument list of `cuTensorMapEncodeTiled`, as plain data.**
///
/// Fixed-size arrays rather than `Vec`s: the FFI takes pointers to exactly `rank` (or `rank - 1`)
/// entries, the maximum is 5, and a fixed array makes the struct `Copy` and trivially constructible
/// in a `const` test table. Entries at index `>= rank` are ignored and must be left at their
/// defaults — [`TensorMapArgs::validate`] enforces that so a stale entry cannot be read by a later
/// rank change.
///
/// `global_address` is deliberately **not** part of this struct. The device pointer is the one piece
/// that is not geometry, it is the only piece that changes between two otherwise identical
/// descriptors, and keeping it out means the whole bundle is `Copy`, comparable and testable with no
/// device present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorMapArgs {
    pub data_type: TmaDataType,
    /// Number of tensor dimensions, `1..=5`.
    pub rank: u32,
    /// Extent of each dimension **in elements**, dimension 0 being the contiguous one.
    pub global_dim: [u64; TMA_MAX_RANK],
    /// `global_strides[i]` is the distance **in bytes** between consecutive elements along dimension
    /// `i + 1`. There is deliberately no entry for dimension 0: it is contiguous by definition, so
    /// its stride is `data_type.size()` and the driver's array has `rank - 1` entries.
    pub global_strides: [u64; TMA_MAX_RANK - 1],
    /// Extent of the copied tile in each dimension, **in elements**.
    pub box_dim: [u32; TMA_MAX_RANK],
    /// Element-level stride (subsampling) per dimension; `1` means dense.
    pub element_strides: [u32; TMA_MAX_RANK],
    pub interleave: TmaInterleave,
    pub swizzle: TmaSwizzle,
    pub l2_promotion: TmaL2Promotion,
    pub oob_fill: TmaOobFill,
}

impl TensorMapArgs {
    /// A **2-D row-major matrix**, tiled — the only shape a NT GEMM operand needs, and the shape
    /// that gets dimension order wrong if built by hand.
    ///
    /// `rows x cols` elements, row stride `row_stride_elems` (>= `cols`, so a sub-matrix view works).
    /// The tile is `box_rows x box_cols`. Because dimension 0 is the contiguous axis:
    ///
    /// ```text
    /// global_dim     = [cols, rows]                     // elements
    /// global_strides = [row_stride_elems * elem_size]   // bytes, one entry (for dimension 1)
    /// box_dim        = [box_cols, box_rows]             // elements
    /// ```
    ///
    /// and the tensor coordinates a `cp.async.bulk.tensor.2d` passes are `{col, row}`, in that
    /// order, in elements.
    pub const fn tiled_2d_row_major(
        data_type: TmaDataType,
        rows: u64,
        cols: u64,
        row_stride_elems: u64,
        box_rows: u32,
        box_cols: u32,
        swizzle: TmaSwizzle,
    ) -> Self {
        Self {
            data_type,
            rank: 2,
            global_dim: [cols, rows, 0, 0, 0],
            global_strides: [row_stride_elems * data_type.size() as u64, 0, 0, 0],
            box_dim: [box_cols, box_rows, 0, 0, 0],
            element_strides: [1, 1, 0, 0, 0],
            interleave: TmaInterleave::None,
            swizzle,
            l2_promotion: TmaL2Promotion::B128,
            oob_fill: TmaOobFill::Zero,
        }
    }

    /// Bytes one copy of this box moves — `prod(box_dim) * elem_size`.
    ///
    /// **This is the number an `mbarrier.arrive.expect_tx` must declare**, and it is the reason the
    /// value is computed here rather than at the launch site: the transaction count and the box are
    /// two views of the same fact, and a mismatch does not fail, it hangs (the barrier never reaches
    /// its expected byte count and every consumer waits forever).
    pub fn transaction_bytes(&self) -> usize {
        let mut n: usize = self.data_type.size();
        for d in 0..self.rank as usize {
            n *= self.box_dim[d] as usize;
        }
        n
    }

    /// **Every documented precondition of `cuTensorMapEncodeTiled`, as one predicate.**
    ///
    /// `global_address` is checked separately by [`TensorMap::encode`], since it is not part of this
    /// bundle. Each rule below is a driver-documented requirement, and each has a test; the point of
    /// stating them here rather than letting the driver reject is that `CUDA_ERROR_INVALID_VALUE`
    /// names none of them, so a wrong descriptor otherwise costs an H100 hour to localise.
    pub fn validate(&self) -> Result<(), String> {
        let elem = self.data_type.size();
        let rank = self.rank as usize;
        if rank == 0 || rank > TMA_MAX_RANK {
            return Err(format!("tma: rank {rank} outside 1..={TMA_MAX_RANK}"));
        }
        if self.interleave != TmaInterleave::None {
            return Err(format!(
                "tma: interleave {:?} is not supported by this backend (only INTERLEAVE_NONE)",
                self.interleave
            ));
        }
        for d in 0..rank {
            if self.global_dim[d] == 0 {
                return Err(format!("tma: global_dim[{d}] is zero"));
            }
            if self.global_dim[d] > u32::MAX as u64 {
                return Err(format!(
                    "tma: global_dim[{d}] = {} exceeds the 2^32 element limit",
                    self.global_dim[d]
                ));
            }
            if self.box_dim[d] == 0 || self.box_dim[d] > TMA_MAX_BOX_DIM {
                return Err(format!(
                    "tma: box_dim[{d}] = {} outside 1..={TMA_MAX_BOX_DIM}",
                    self.box_dim[d]
                ));
            }
            if self.element_strides[d] == 0 || self.element_strides[d] > TMA_MAX_ELEMENT_STRIDE {
                return Err(format!(
                    "tma: element_strides[{d}] = {} outside 1..={TMA_MAX_ELEMENT_STRIDE}",
                    self.element_strides[d]
                ));
            }
        }
        // Entries past the rank are read by nobody, so a stale one is invisible until the rank
        // changes and it silently becomes live. Pin them at their defaults instead.
        for d in rank..TMA_MAX_RANK {
            if self.global_dim[d] != 0 || self.box_dim[d] != 0 || self.element_strides[d] != 0 {
                return Err(format!(
                    "tma: dimension {d} is past rank {rank} but is not zeroed"
                ));
            }
        }
        for d in 0..rank.saturating_sub(1) {
            let s = self.global_strides[d];
            if !s.is_multiple_of(16) {
                return Err(format!(
                    "tma: global_strides[{d}] = {s} B is not a multiple of 16"
                ));
            }
            if s == 0 {
                return Err(format!("tma: global_strides[{d}] is zero"));
            }
            if s >= 1u64 << 40 {
                return Err(format!("tma: global_strides[{d}] = {s} B exceeds 2^40"));
            }
            // The stride along dimension d+1 must clear a whole dimension-d row, else consecutive
            // rows overlap and the "tensor" is not one.
            let row = self.global_dim[d] * elem as u64;
            if s < row {
                return Err(format!(
                    "tma: global_strides[{d}] = {s} B is shorter than dimension {d} ({row} B)"
                ));
            }
        }
        for d in rank.saturating_sub(1)..TMA_MAX_RANK - 1 {
            if self.global_strides[d] != 0 {
                return Err(format!(
                    "tma: global_strides[{d}] is past rank {rank} but is not zeroed"
                ));
            }
        }
        // The contiguous extent of the box, in bytes: the quantity every remaining rule is about.
        let inner = self.box_dim[0] as usize * elem;
        if !inner.is_multiple_of(16) {
            return Err(format!(
                "tma: box_dim[0] * elem = {inner} B is not a multiple of 16"
            ));
        }
        if self.swizzle != TmaSwizzle::None && inner > self.swizzle.atom_bytes() {
            return Err(format!(
                "tma: box_dim[0] * elem = {inner} B exceeds the {:?} swizzle atom ({} B) -- a tile \
                 wider than its swizzle atom cannot be described",
                self.swizzle,
                self.swizzle.atom_bytes()
            ));
        }
        Ok(())
    }
}

/// The PTX declaration a kernel parameter holding a tensor map must carry:
/// `.param .align 64 .b8 <name>[128]`.
///
/// A tensor map is passed **by value** as a grid-constant parameter, not by pointer, so the kernel's
/// parameter is 128 opaque bytes with 64-byte alignment. The address the
/// `cp.async.bulk.tensor` instruction wants is then the *generic* address of that parameter, which
/// PTX reaches with the two-instruction idiom `mov.b64 %rd, <name>; cvta.param.u64 %rd, %rd;` —
/// see [`ptx_param_address`].
pub fn ptx_param_decl(name: &str) -> String {
    format!(".param .align {TENSOR_MAP_ALIGN} .b8 {name}[{TENSOR_MAP_BYTES}]")
}

/// The two instructions that turn the by-value parameter `name` into the generic address a
/// `cp.async.bulk.tensor` takes, leaving it in `reg`.
///
/// `cvta.param` is the piece that is easy to omit: the parameter's own address lives in the `.param`
/// state space, and the bulk-tensor instruction requires a generic (or `.const`/`.global`) address.
/// Without the conversion the driver JIT accepts the module and the copy reads from nowhere.
pub fn ptx_param_address(reg: &str, name: &str) -> String {
    format!("    mov.b64 {reg},{name};\n    cvta.param.u64 {reg},{reg};\n")
}

// --- the driver call -----------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod device {
    use super::*;
    use cudarc::driver::sys;

    /// A built 128-byte `CUtensorMap`, ready to be pushed as a by-value kernel argument.
    ///
    /// `#[repr(transparent)]` over the driver's own struct, so pushing `&TensorMap` pushes exactly
    /// the 128 bytes the kernel's `.param .align 64 .b8 [128]` expects and nothing else.
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct TensorMap(pub sys::CUtensorMap);

    // SAFETY: `TensorMap` is `#[repr(transparent)]` over `CUtensorMap_st`, which is `#[repr(C)]`
    // `[u64; 16]` with 64-byte alignment: 128 bytes of plain data, no padding, no pointers into host
    // memory, no `Drop`. `DeviceRepr`'s contract is exactly that the value may be memcpy'd into a
    // kernel parameter buffer, which is how the CUDA C++ `__grid_constant__ const CUtensorMap`
    // parameter is passed too.
    unsafe impl cudarc::driver::DeviceRepr for TensorMap {}

    impl std::fmt::Debug for TensorMap {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TensorMap({} opaque bytes)", TENSOR_MAP_BYTES)
        }
    }

    impl TensorMap {
        /// Build the descriptor for `args` over the device allocation starting at `global_address`.
        ///
        /// # Preconditions, all checked before the call
        ///
        /// 1. `args.validate()` passes — every geometry rule the driver documents. Checked first, so
        ///    a bad descriptor is named by *this* function rather than by an anonymous
        ///    `CUDA_ERROR_INVALID_VALUE` from inside the encoder.
        /// 2. `global_address` is 16-byte aligned. The driver requires it; an unaligned base is
        ///    otherwise a silently wrong copy.
        /// 3. `global_address` is non-null.
        /// 4. The caller guarantees the allocation is at least
        ///    `global_dim[rank-1] * global_strides[rank-2]` bytes long and outlives every launch
        ///    that uses this map. That one cannot be checked here — a `CUdeviceptr` carries no
        ///    length — so it is the caller's documented obligation and the reason this function is
        ///    `unsafe`.
        ///
        /// # Safety
        ///
        /// `global_address` must point to a live device allocation of at least the size the tensor
        /// geometry implies, and must stay live for as long as any kernel launched with the returned
        /// map can run.
        pub unsafe fn encode(
            args: &TensorMapArgs,
            global_address: sys::CUdeviceptr,
        ) -> Result<Self, String> {
            args.validate()?;
            if global_address == 0 {
                return Err("tma: global address is null".to_string());
            }
            if !global_address.is_multiple_of(16) {
                return Err(format!(
                    "tma: global address {global_address:#x} is not 16-byte aligned"
                ));
            }

            let mut map = sys::CUtensorMap::default();
            // Local copies so the pointers handed to the driver point at `rank`/`rank-1` live
            // entries of the right integer width, independent of this struct's storage type.
            let global_dim: [u64; TMA_MAX_RANK] = args.global_dim;
            let global_strides: [u64; TMA_MAX_RANK - 1] = args.global_strides;
            let box_dim: [u32; TMA_MAX_RANK] = args.box_dim;
            let elem_strides: [u32; TMA_MAX_RANK] = args.element_strides;

            // SAFETY: preconditions 1-3 are checked above and precondition 4 is the caller's, hoisted
            // into this function's own `unsafe`. `map` is a live, zeroed, 64-byte-aligned
            // `CUtensorMap`. The four array pointers address `rank`, `rank - 1`, `rank` and `rank`
            // initialised entries respectively -- `rank <= TMA_MAX_RANK` is checked by `validate`, so
            // none of them can be read past its storage. The enums are `#[repr(u32)]` with the
            // driver's own discriminants (asserted by `enum_discriminants_match_the_driver`).
            let r = unsafe {
                sys::cuTensorMapEncodeTiled(
                    &mut map as *mut sys::CUtensorMap,
                    driver_data_type(args.data_type),
                    args.rank,
                    global_address as *mut std::ffi::c_void,
                    global_dim.as_ptr(),
                    // rank - 1 entries; for rank 1 the driver reads none, and `as_ptr` on a live
                    // array is a valid non-null pointer regardless.
                    global_strides.as_ptr(),
                    box_dim.as_ptr(),
                    elem_strides.as_ptr(),
                    driver_interleave(args.interleave),
                    driver_swizzle(args.swizzle),
                    driver_l2(args.l2_promotion),
                    driver_oob(args.oob_fill),
                )
            };
            r.result()
                .map_err(|e| format!("cuTensorMapEncodeTiled failed: {e:?} (args {args:?})"))?;
            Ok(TensorMap(map))
        }

        /// The raw driver object, for a launch argument list.
        pub fn raw(&self) -> &sys::CUtensorMap {
            &self.0
        }
    }

    pub(super) fn driver_data_type(t: TmaDataType) -> sys::CUtensorMapDataType {
        use sys::CUtensorMapDataType_enum as D;
        match t {
            TmaDataType::U8 => D::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            TmaDataType::U16 => D::CU_TENSOR_MAP_DATA_TYPE_UINT16,
            TmaDataType::U32 => D::CU_TENSOR_MAP_DATA_TYPE_UINT32,
            TmaDataType::F16 => D::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            TmaDataType::F32 => D::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            TmaDataType::Bf16 => D::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        }
    }

    pub(super) fn driver_swizzle(s: TmaSwizzle) -> sys::CUtensorMapSwizzle {
        use sys::CUtensorMapSwizzle_enum as S;
        match s {
            TmaSwizzle::None => S::CU_TENSOR_MAP_SWIZZLE_NONE,
            TmaSwizzle::B32 => S::CU_TENSOR_MAP_SWIZZLE_32B,
            TmaSwizzle::B64 => S::CU_TENSOR_MAP_SWIZZLE_64B,
            TmaSwizzle::B128 => S::CU_TENSOR_MAP_SWIZZLE_128B,
        }
    }

    pub(super) fn driver_interleave(i: TmaInterleave) -> sys::CUtensorMapInterleave {
        use sys::CUtensorMapInterleave_enum as I;
        match i {
            TmaInterleave::None => I::CU_TENSOR_MAP_INTERLEAVE_NONE,
            TmaInterleave::B16 => I::CU_TENSOR_MAP_INTERLEAVE_16B,
            TmaInterleave::B32 => I::CU_TENSOR_MAP_INTERLEAVE_32B,
        }
    }

    pub(super) fn driver_l2(p: TmaL2Promotion) -> sys::CUtensorMapL2promotion {
        use sys::CUtensorMapL2promotion_enum as P;
        match p {
            TmaL2Promotion::None => P::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            TmaL2Promotion::B64 => P::CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
            TmaL2Promotion::B128 => P::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            TmaL2Promotion::B256 => P::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        }
    }

    pub(super) fn driver_oob(f: TmaOobFill) -> sys::CUtensorMapFloatOOBfill {
        use sys::CUtensorMapFloatOOBfill_enum as F;
        match f {
            TmaOobFill::Zero => F::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            TmaOobFill::NanRequestZeroFma => F::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
        }
    }
}

#[cfg(feature = "gpu")]
pub use device::TensorMap;

#[cfg(test)]
mod tests {
    use super::*;

    /// The f16 A operand of the shipped 128x256x64 wgmma GEMM: `M x K` row-major, tile `128 x 64`.
    fn a_operand(m: u64, k: u64) -> TensorMapArgs {
        TensorMapArgs::tiled_2d_row_major(TmaDataType::F16, m, k, k, 128, 64, TmaSwizzle::None)
    }

    #[test]
    fn dimension_zero_is_the_contiguous_axis() {
        let a = a_operand(4096, 4096);
        // Row-major M x K: the contiguous axis is K, so it is dimension 0. The single stride entry
        // is the byte distance between rows.
        assert_eq!(a.rank, 2);
        assert_eq!(
            a.global_dim[0], 4096,
            "dim 0 must be the K (contiguous) axis"
        );
        assert_eq!(a.global_dim[1], 4096, "dim 1 must be the M (row) axis");
        assert_eq!(a.global_strides[0], 4096 * 2, "row stride, in BYTES");
        assert_eq!(a.box_dim, [64, 128, 0, 0, 0], "box is [box_cols, box_rows]");
        assert_eq!(a.element_strides, [1, 1, 0, 0, 0]);
        a.validate().expect("the shipped A operand must validate");
    }

    /// A sub-matrix view: `row_stride_elems > cols`. The stride must describe the *parent*, the
    /// dimension the *view*. Getting these confused reads the wrong rows with no error anywhere.
    #[test]
    fn a_strided_view_keeps_the_parent_stride_and_the_view_extent() {
        let v = TensorMapArgs::tiled_2d_row_major(
            TmaDataType::F16,
            256,
            128,
            4096,
            128,
            64,
            TmaSwizzle::None,
        );
        assert_eq!(v.global_dim[0], 128);
        assert_eq!(v.global_strides[0], 4096 * 2);
        v.validate().unwrap();
    }

    #[test]
    fn transaction_bytes_is_the_product_of_the_box() {
        // 128 x 64 f16 = 16384 B; 256 x 64 f16 = 32768 B. These two are the per-stage byte counts
        // the shipped 128x256x64 mainloop declares in `mbarrier.arrive.expect_tx`.
        assert_eq!(a_operand(4096, 4096).transaction_bytes(), 128 * 64 * 2);
        let b = TensorMapArgs::tiled_2d_row_major(
            TmaDataType::F16,
            4096,
            4096,
            4096,
            256,
            64,
            TmaSwizzle::None,
        );
        assert_eq!(b.transaction_bytes(), 256 * 64 * 2);
        assert_eq!(
            a_operand(4096, 4096).transaction_bytes() + b.transaction_bytes(),
            49152,
            "one 128x256x64 stage: D1 section 3.1's 49168 B minus CUTLASS's 16 B of pipeline state"
        );
    }

    #[test]
    fn element_size_is_per_type() {
        assert_eq!(TmaDataType::U8.size(), 1);
        assert_eq!(TmaDataType::F16.size(), 2);
        assert_eq!(TmaDataType::Bf16.size(), 2);
        assert_eq!(TmaDataType::F32.size(), 4);
    }

    /// Every rejection path, each with the geometry that triggers it. A validator whose negative
    /// arms are untested is a validator that passes everything.
    #[test]
    fn validate_rejects_every_documented_violation() {
        let ok = a_operand(4096, 4096);
        ok.validate().unwrap();

        let mut bad = ok;
        bad.rank = 0;
        assert!(bad.validate().unwrap_err().contains("rank 0"));
        bad = ok;
        bad.rank = 6;
        assert!(bad.validate().unwrap_err().contains("rank 6"));

        bad = ok;
        bad.global_dim[0] = 0;
        assert!(bad
            .validate()
            .unwrap_err()
            .contains("global_dim[0] is zero"));

        bad = ok;
        bad.global_dim[1] = 1u64 << 33;
        assert!(bad.validate().unwrap_err().contains("2^32 element limit"));

        bad = ok;
        bad.box_dim[1] = 257;
        assert!(bad.validate().unwrap_err().contains("box_dim[1] = 257"));
        bad.box_dim[1] = 0;
        assert!(bad.validate().unwrap_err().contains("box_dim[1] = 0"));

        bad = ok;
        bad.element_strides[0] = 9;
        assert!(bad
            .validate()
            .unwrap_err()
            .contains("element_strides[0] = 9"));

        // Past-rank entries must be zero, or a later rank bump silently reads stale geometry.
        bad = ok;
        bad.global_dim[2] = 7;
        assert!(bad.validate().unwrap_err().contains("past rank 2"));
        bad = ok;
        bad.global_strides[1] = 16;
        assert!(bad
            .validate()
            .unwrap_err()
            .contains("global_strides[1] is past rank 2"));

        // Strides: multiple of 16, non-zero, under 2^40, and at least one row wide.
        bad = ok;
        bad.global_strides[0] = 24;
        assert!(bad.validate().unwrap_err().contains("multiple of 16"));
        bad = ok;
        bad.global_strides[0] = 0;
        assert!(bad.validate().unwrap_err().contains("is zero"));
        bad = ok;
        bad.global_strides[0] = 1u64 << 41;
        assert!(bad.validate().unwrap_err().contains("exceeds 2^40"));
        bad = ok;
        bad.global_strides[0] = 16; // 4096 f16 = 8192 B of row, described by a 16 B stride
        assert!(bad
            .validate()
            .unwrap_err()
            .contains("shorter than dimension"));

        // The contiguous box extent must be a multiple of 16 B.
        bad = ok;
        bad.box_dim[0] = 4; // 4 f16 = 8 B
        assert!(bad.validate().unwrap_err().contains("multiple of 16"));

        bad = ok;
        bad.interleave = TmaInterleave::B32;
        assert!(bad.validate().unwrap_err().contains("interleave"));
    }

    /// **The swizzle-atom rule, at every mode.** `box_dim[0] * elem` is the contiguous tile width in
    /// bytes and must fit the atom: a `BK = 64` f16 tile is exactly 128 B, so it is legal under the
    /// 128-B swizzle and illegal under 64-B and 32-B. This is the constraint that pins `BK = 64` for
    /// 16-bit operands on Hopper -- the same `BK = 64` CUTLASS ships (D1 section 3.1) -- and it is a
    /// consequence of the descriptor format, not a tuning choice.
    #[test]
    fn the_swizzle_atom_bounds_the_contiguous_box_extent() {
        assert_eq!(TmaSwizzle::None.atom_bytes(), 0);
        assert_eq!(TmaSwizzle::B32.atom_bytes(), 32);
        assert_eq!(TmaSwizzle::B64.atom_bytes(), 64);
        assert_eq!(TmaSwizzle::B128.atom_bytes(), 128);

        // BK = 64 f16 = 128 B.
        let bk64 =
            |sw| TensorMapArgs::tiled_2d_row_major(TmaDataType::F16, 4096, 4096, 4096, 128, 64, sw);
        bk64(TmaSwizzle::None).validate().unwrap();
        bk64(TmaSwizzle::B128).validate().unwrap();
        for sw in [TmaSwizzle::B64, TmaSwizzle::B32] {
            let e = bk64(sw).validate().unwrap_err();
            assert!(e.contains("swizzle atom"), "{e}");
        }
        // BK = 32 f16 = 64 B fits the 64-B atom, and BK = 16 f16 = 32 B fits the 32-B atom.
        let bk = |cols, sw| {
            TensorMapArgs::tiled_2d_row_major(TmaDataType::F16, 4096, 4096, 4096, 128, cols, sw)
                .validate()
        };
        bk(32, TmaSwizzle::B64).unwrap();
        bk(16, TmaSwizzle::B32).unwrap();
        assert!(bk(32, TmaSwizzle::B32).is_err());
    }

    #[test]
    fn the_ptx_parameter_declaration_matches_the_driver_abi() {
        assert_eq!(TENSOR_MAP_BYTES, 128);
        assert_eq!(TENSOR_MAP_ALIGN, 64);
        assert_eq!(ptx_param_decl("tmap_a"), ".param .align 64 .b8 tmap_a[128]");
        let addr = ptx_param_address("%rdA", "tmap_a");
        assert!(addr.contains("mov.b64 %rdA,tmap_a;"));
        assert!(
            addr.contains("cvta.param.u64 %rdA,%rdA;"),
            "without cvta.param the bulk-tensor copy reads a .param address as generic"
        );
        assert!(addr.is_ascii());
    }

    /// The enum discriminants ARE the driver's ABI values -- restating them device-free is only safe
    /// while they agree. Under `--features gpu` this is checked against `cudarc`'s own enums; without
    /// it, the numbers are pinned literally so a careless edit is still loud.
    #[test]
    fn enum_discriminants_match_the_driver() {
        assert_eq!(TmaSwizzle::None as u32, 0);
        assert_eq!(TmaSwizzle::B32 as u32, 1);
        assert_eq!(TmaSwizzle::B64 as u32, 2);
        assert_eq!(TmaSwizzle::B128 as u32, 3);
        assert_eq!(TmaDataType::F16 as u32, 6);
        assert_eq!(TmaDataType::F32 as u32, 7);
        assert_eq!(TmaDataType::Bf16 as u32, 9);
        assert_eq!(TmaOobFill::Zero as u32, 0);
        assert_eq!(TmaInterleave::None as u32, 0);
        assert_eq!(TmaL2Promotion::B128 as u32, 2);

        #[cfg(feature = "gpu")]
        {
            for s in [
                TmaSwizzle::None,
                TmaSwizzle::B32,
                TmaSwizzle::B64,
                TmaSwizzle::B128,
            ] {
                assert_eq!(s as u32, super::device::driver_swizzle(s) as u32);
            }
            for t in [
                TmaDataType::U8,
                TmaDataType::U16,
                TmaDataType::U32,
                TmaDataType::F16,
                TmaDataType::F32,
                TmaDataType::Bf16,
            ] {
                assert_eq!(t as u32, super::device::driver_data_type(t) as u32);
            }
            for f in [TmaOobFill::Zero, TmaOobFill::NanRequestZeroFma] {
                assert_eq!(f as u32, super::device::driver_oob(f) as u32);
            }
            for i in [TmaInterleave::None, TmaInterleave::B16, TmaInterleave::B32] {
                assert_eq!(i as u32, super::device::driver_interleave(i) as u32);
            }
            for p in [
                TmaL2Promotion::None,
                TmaL2Promotion::B64,
                TmaL2Promotion::B128,
                TmaL2Promotion::B256,
            ] {
                assert_eq!(p as u32, super::device::driver_l2(p) as u32);
            }
        }
    }

    /// The descriptor is 128 bytes with 64-byte alignment, and the wrapper must not have grown a
    /// byte -- it is pushed straight into a kernel parameter buffer whose layout the PTX declares.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_wrapper_is_exactly_the_driver_object() {
        assert_eq!(std::mem::size_of::<TensorMap>(), TENSOR_MAP_BYTES);
        assert_eq!(std::mem::align_of::<TensorMap>(), TENSOR_MAP_ALIGN);
    }
}
