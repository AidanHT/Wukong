//! `wukong_backend` — the seam between MIR and the things that consume it.
//!
//! Four types implement [`Backend`], all outside this crate: `wukong_interp::Interpreter` (reference
//! execution + differential-testing oracle), `wukong_codegen_cranelift::CraneliftBackend` and
//! `wukong_codegen_gpu::GpuLowerBackend` (feature `gpu`) all produce [`Artifact::Executed`];
//! `wukong_codegen_llvm::LlvmBackend` produces [`Artifact::Emitted`]. Each `compile` is a thin
//! adapter over the crate's real free function.
//!
//! LANDMINE: this trait is a *declared* seam, not the live dispatch path. Nothing in the workspace
//! calls [`Backend::compile`] — `wukong_driver::compile` matches on its own `BackendKind` and calls
//! `wukong_interp::run_with_output` / `wukong_codegen_cranelift::jit_run` / `run_on_gpu` /
//! `run_on_gpu_lower` (and `emit_llvm_ir` / `emit_object` for the AOT exits) directly, and
//! `--backend=gpu`, the offloading interpreter, has no `Backend` impl at all. Adding an [`Artifact`]
//! variant or a trait method therefore does not reach the CLI: a new target needs the `BackendKind`
//! variant, its dispatch arm in `wukong_driver`, and the `--backend=` string parse in `wukongc`.

use std::path::PathBuf;

use wukong_mir::Program;
use wukong_span::{Interner, Symbol};

/// What a backend produced.
#[derive(Debug)]
pub enum Artifact {
    /// The program was run directly (interpreter, Cranelift JIT, or gpu-native): an exit code and
    /// captured stdout. The interpreter and the Cranelift JIT must agree bit-for-bit on both fields —
    /// that identity is what the differential gate compares; gpu-native is held to the CPU↔GPU
    /// *tolerance* gate instead (bit-exact only for integer / control-flow output).
    Executed { exit_code: i64, stdout: Vec<u8> },
    /// Native artifacts were emitted (LLVM): textual IR and/or an object/executable path.
    Emitted {
        llvm_ir: Option<String>,
        object: Option<PathBuf>,
    },
}

/// A consumer of fully-lowered MIR.
///
/// Nothing here enforces that: `wukong_mir::MirLevel` is inert, so the guarantee comes from
/// `wukong_driver::verify_or_ice`, which runs `wukong_mir::verify::verify_function` over every
/// function before each backend entry. Errors are plain `String`s, not `wukong_diag` diagnostics — a
/// backend failure is ICE-class or an unsupported construct (`wukong_codegen_gpu` prefixes the latter
/// with its `UNSUPPORTED` marker, which that crate's own megakernel path and coverage gate use to
/// decline — the driver does not inspect it and reports the error verbatim).
pub trait Backend {
    fn name(&self) -> &'static str;

    /// Compile (or execute) `program`, using `entry` as the program entry point where relevant.
    fn compile(
        &self,
        program: &Program,
        entry: Symbol,
        interner: &Interner,
    ) -> Result<Artifact, String>;
}
