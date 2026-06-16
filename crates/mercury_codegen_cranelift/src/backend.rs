//! The [`Backend`] implementation: JIT-compile with Cranelift and run in process.

use mercury_backend::{Artifact, Backend};
use mercury_mir::Program;
use mercury_span::{Interner, Symbol};

/// Native backend. Lowers Low MIR to machine code via Cranelift and executes `entry` directly,
/// capturing stdout and the exit code into [`Artifact::Executed`] — the same shape the interpreter
/// produces, so the driver and the differential gate treat the two interchangeably.
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
