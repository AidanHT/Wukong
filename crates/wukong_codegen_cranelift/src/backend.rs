//! The [`Backend`] implementation: JIT-compile with Cranelift and run in process.

use wukong_backend::{Artifact, Backend};
use wukong_mir::Program;
use wukong_span::{Interner, Symbol};

/// Native backend. Lowers Low MIR to machine code via Cranelift and executes `entry` directly,
/// capturing stdout and the exit code into [`Artifact::Executed`] — the same shape the interpreter
/// produces, so anything written against the `Backend` seam can treat the two interchangeably.
///
/// LANDMINE: this impl is the *declared* seam, not the live dispatch path. Nothing in the workspace
/// constructs `CraneliftBackend` or calls [`Backend::compile`] — `wukong_driver::compile` matches on
/// its own `BackendKind` and calls `crate::jit_run` directly, and the differential gate calls
/// `jit_run` / `wukong_interp::run_with_output` itself. Editing this file does not change what
/// `--backend=native` does.
pub struct CraneliftBackend;

impl Backend for CraneliftBackend {
    fn name(&self) -> &'static str {
        "cranelift"
    }

    fn compile(
        &self,
        program: &Program,
        entry: Symbol,
        interner: &Interner,
    ) -> Result<Artifact, String> {
        let (exit_code, stdout) = crate::jit_run(program, entry, interner)?;
        Ok(Artifact::Executed { exit_code, stdout })
    }
}
