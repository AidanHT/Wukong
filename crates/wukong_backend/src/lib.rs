//! `wukong_backend` — the seam between MIR and the things that consume it.
//!
//! Both the interpreter (reference execution + differential-testing oracle) and the LLVM backend
//! implement [`Backend`], so the driver is backend-agnostic. The interpreter produces
//! [`Artifact::Executed`]; the LLVM backend produces [`Artifact::Emitted`].

use std::path::PathBuf;

use wukong_mir::Program;
use wukong_span::{Interner, Symbol};

/// What a backend produced.
#[derive(Debug)]
pub enum Artifact {
    /// The program was run directly (interpreter): an exit code and captured stdout.
    Executed { exit_code: i64, stdout: Vec<u8> },
    /// Native artifacts were emitted (LLVM): textual IR and/or an object/executable path.
    Emitted {
        llvm_ir: Option<String>,
        object: Option<PathBuf>,
    },
}

/// A consumer of fully-lowered (Low) MIR.
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
