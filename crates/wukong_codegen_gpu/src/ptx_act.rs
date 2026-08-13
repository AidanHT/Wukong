//! **The crate's one activation-epilogue text**, device-free.
//!
//! Every fused activation this backend emits -- the Act-1 `wmma` epilogue
//! ([`crate::ptx_wmma`]), the fp8 `mma` pipeline ([`crate::ptx_fp8`]), the conv epilogue and, since
//! wave 4, the Hopper `wgmma` epilogue ([`crate::ptx_wgmma`]) -- generates its instructions from
//! [`Act::epilogue`] and from nowhere else.
//!
//! # Why one function instead of four copies of five instructions
//!
//! Because the sharing is what makes the fused-vs-unfused exactness law TRUE, rather than a claim
//! somebody has to re-check. The law (WAVE4_DOSSIER L7a) is
//!
//! ```text
//!   fused_act(A*B^T + bias)  ==  vmath_act( gemm(A*B^T) + bias )
//! ```
//!
//! element-wise `==`, **not a tolerance**. It holds because an f32 store followed by an f32 load is
//! the identity, the bias add is the same `add.f32` on the same two f32 values in both arms, and
//! the activation is the same instruction sequence with the same constants in both arms.
//! `tanh.approx.f32`, `ex2.approx.f32` and `rcp.approx.f32` are approximate but **deterministic** --
//! a fixed function of the input bit pattern on a given architecture -- so every intermediate is
//! bit-identical and so is the result. A second copy of the sequence, however carefully
//! transcribed, turns that into a property of two texts staying equal.
//!
//! This module is deliberately **not** behind the `gpu` feature: `ptx_wgmma` is not either (its
//! arithmetic is the wgmma family's only device-free proof, so its gates must run in a plain
//! `cargo test`), and a shared text that only existed under a feature would force the wgmma
//! epilogue to carry a copy.

/// Fused activation applied to the f32 accumulator **before** the C store -- the thing cuBLAS
/// structurally cannot do (it only computes `A*B^T`; an activation needs a second kernel that
/// round-trips C through HBM), and, for [`Act::Silu`], the thing cuBLAS**Lt** cannot do either:
/// its `cublasLtEpilogue_t` has sixteen members and none of them is SiLU or swish
/// (`baselines::the_cublaslt_fusable_set_is_exactly_these_sixteen` pins the absence).
///
/// The activation is elementwise on each accumulator register, so it needs no knowledge of any
/// fragment's row/column layout -- which is why one generator serves `wmma`, `mma` and `wgmma`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    None,
    Relu,
    Silu,
    Gelu,
}

impl Act {
    /// PTX applying the activation in place to one f32 accumulator register `reg`. Transcendental
    /// activations use the SFU fast paths (`ex2.approx`/`tanh.approx`/`rcp.approx`) and the **exact
    /// same formulas + constants** as the standalone `ptx::vmath_ptx` kernels, so a fused
    /// `silu(A*B^T)` equals the unfused `silu(gemm)` and inherits its tolerance gate.
    ///
    /// Scratch lives in `%act0`/`%act1`, which the calling entry must declare; each accumulator is
    /// processed sequentially so the scratch reuses.
    pub fn epilogue(self, reg: &str) -> String {
        let hexf = |x: f32| format!("0f{:08X}", x.to_bits());
        match self {
            Act::None => String::new(),
            Act::Relu => format!("    max.f32 {reg},{reg},0f00000000;\n"),
            // silu(x) = x*sigmoid(x) = x / (1 + exp(-x)), exp via 2^(x*log2e).
            Act::Silu => {
                let (nlog2e, one) = (hexf(-std::f32::consts::LOG2_E), hexf(1.0));
                format!(
                    "    mul.f32 %act0,{reg},{nlog2e};\n    ex2.approx.f32 %act0,%act0;\n    \
                     add.f32 %act0,%act0,{one};\n    rcp.approx.f32 %act0,%act0;\n    \
                     mul.f32 {reg},{reg},%act0;\n"
                )
            }
            // gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) -- the TANH convention, which
            // is cuBLASLt's and CUTLASS's, and which differs from the erf form by up to ~1e-3 around
            // |x| ~ 2 (`baselines::the_two_gelu_conventions_are_distinguishable_and_pinned`).
            Act::Gelu => {
                let c0 = hexf((2.0f32 / std::f32::consts::PI).sqrt());
                let (c1, one, half) = (hexf(0.044715), hexf(1.0), hexf(0.5));
                format!(
                    "    mul.f32 %act0,{reg},{reg};\n    mul.f32 %act0,%act0,{reg};\n    \
                     fma.rn.f32 %act0,%act0,{c1},{reg};\n    mul.f32 %act0,%act0,{c0};\n    \
                     tanh.approx.f32 %act0,%act0;\n    add.f32 %act0,%act0,{one};\n    \
                     mul.f32 %act1,{reg},{half};\n    mul.f32 {reg},%act0,%act1;\n"
                )
            }
        }
    }

    /// **Scratch registers this activation holds live**, in 32-bit units -- part of the wgmma
    /// family's register budget law (G22). `%act0` alone for SiLU, both for GELU, none for ReLU
    /// (`max.f32` is in place).
    pub const fn scratch_regs(self) -> u32 {
        match self {
            Act::None | Act::Relu => 0,
            Act::Silu => 1,
            Act::Gelu => 2,
        }
    }

    /// The derived-name tag this activation contributes (guard G3-W4). Empty for [`Act::None`], so
    /// a plain row's key never moves.
    pub const fn key_tag(self) -> &'static str {
        match self {
            Act::None => "",
            Act::Relu => "_relu",
            Act::Silu => "_silu",
            Act::Gelu => "_gelu",
        }
    }

    /// Is this activation one cuBLASLt can fuse itself? `RELU` and `GELU` are members of
    /// `cublasLtEpilogue_t`; **SiLU is not**, and that absence is the whole of wave 4's publishable
    /// margin (WAVE4_DOSSIER 2.3 group B). A row whose activation the peer fuses is a PARITY row --
    /// `S(group A) ~= r` -- and may never be published as a fusion win.
    pub const fn peer_fuses_it(self) -> bool {
        match self {
            Act::None | Act::Relu | Act::Gelu => true,
            Act::Silu => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Act;

    /// **L1 for the activation family.** One non-ASCII byte in a `format!` is a `ptxas fatal` at
    /// `cuModuleLoadData`. This family's reference formulas are the worst offenders in the repo
    /// (square-root signs, middle dots, `A*B^T` spelled with a real transpose), so the gate lives
    /// on the generator rather than waiting for a wgmma module to carry a copy of the text.
    #[test]
    fn every_fused_activation_snippet_is_pure_ascii() {
        for act in [Act::None, Act::Relu, Act::Silu, Act::Gelu] {
            let ptx = act.epilogue("%acc0");
            if let Some(c) = ptx.chars().find(|c| !c.is_ascii()) {
                panic!("{act:?}: activation PTX must be pure ASCII, found {c:?} in:\n{ptx}");
            }
        }
    }

    /// The hex literals are the **vmath** constants, pinned by value so a "cleanup" that swapped
    /// the tanh-form GELU for the erf form -- or recomputed `0.044715` -- fails here rather than
    /// inside a tolerance band three orders wider than the convention gap (WAVE4_DOSSIER L8).
    #[test]
    fn the_activation_hex_literals_are_the_vmath_constants() {
        let hexf = |x: f32| format!("0f{:08X}", x.to_bits());
        let silu = Act::Silu.epilogue("%r");
        assert!(silu.contains(&hexf(-std::f32::consts::LOG2_E)), "{silu}");
        assert!(silu.contains(&hexf(1.0)), "{silu}");
        assert!(silu.contains("ex2.approx.f32"), "{silu}");
        assert!(silu.contains("rcp.approx.f32"), "{silu}");
        let gelu = Act::Gelu.epilogue("%r");
        assert!(
            gelu.contains(&hexf((2.0f32 / std::f32::consts::PI).sqrt())),
            "{gelu}"
        );
        assert!(gelu.contains(&hexf(0.044715)), "{gelu}");
        assert!(gelu.contains(&hexf(0.5)), "{gelu}");
        assert!(gelu.contains("tanh.approx.f32"), "{gelu}");
        assert_eq!(Act::Relu.epilogue("%r"), "    max.f32 %r,%r,0f00000000;\n");
        assert!(Act::None.epilogue("%r").is_empty());
    }

    #[test]
    fn the_derived_name_and_budget_tags_are_the_geometry() {
        assert_eq!(Act::None.key_tag(), "");
        assert_eq!(Act::Relu.key_tag(), "_relu");
        assert_eq!(Act::Silu.key_tag(), "_silu");
        assert_eq!(Act::Gelu.key_tag(), "_gelu");
        assert_eq!(Act::None.scratch_regs(), 0);
        assert_eq!(Act::Relu.scratch_regs(), 0);
        assert_eq!(Act::Silu.scratch_regs(), 1);
        assert_eq!(Act::Gelu.scratch_regs(), 2);
        // The publishable split, as data: SiLU is the only group-B activation in this enum.
        assert!(!Act::Silu.peer_fuses_it());
        assert!(Act::Relu.peer_fuses_it() && Act::Gelu.peer_fuses_it());
        assert!(Act::None.peer_fuses_it());
    }
}
