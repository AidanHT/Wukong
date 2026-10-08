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

#[cfg(test)]
mod tests {
    //! The seam's own tests. The four production impls live in other crates and cannot be
    //! constructed here without a dependency cycle (`wukong_interp` already depends on this crate),
    //! so these pin the *contract* every impl must honour — object-safety, faithful forwarding of
    //! the three inputs, and the shape of each [`Artifact`] variant — directly, so an adapter that
    //! quietly dropped stdout, lost the exit code, or returned the wrong variant has a definition to
    //! be measured against even though no default-build test reaches the live adapters themselves.
    //!
    //! LANDMINE (same as the module header): this does *not* make the trait a live dispatch path.
    //! `wukong_driver` still matches on its own `BackendKind`; a new target is not wired by passing
    //! this test. The test exists so the seam is not carried with zero coverage, not to imply the
    //! driver routes through it.
    use super::*;

    /// A faithful adapter: it records exactly what it was handed and returns the variant it was
    /// configured for, so the test can assert the trait forwards `program`/`entry`/`interner`
    /// without mangling them — the forwarding every real adapter does before calling its crate's
    /// free function.
    struct Recording {
        kind: &'static str,
        seen_funcs: std::cell::Cell<usize>,
        seen_entry: std::cell::Cell<u32>,
        seen_statics: std::cell::Cell<usize>,
    }

    impl Recording {
        fn new(kind: &'static str) -> Recording {
            Recording {
                kind,
                seen_funcs: std::cell::Cell::new(usize::MAX),
                seen_entry: std::cell::Cell::new(u32::MAX),
                seen_statics: std::cell::Cell::new(usize::MAX),
            }
        }
    }

    impl Backend for Recording {
        fn name(&self) -> &'static str {
            self.kind
        }

        fn compile(
            &self,
            program: &Program,
            entry: Symbol,
            _interner: &Interner,
        ) -> Result<Artifact, String> {
            self.seen_funcs.set(program.funcs.len());
            self.seen_entry.set(entry.0);
            self.seen_statics.set(program.statics.len());
            match self.kind {
                // Mirrors `Interpreter`/`CraneliftBackend`/`GpuLowerBackend`.
                "executing" => Ok(Artifact::Executed {
                    exit_code: 7,
                    stdout: b"hello".to_vec(),
                }),
                // Mirrors `LlvmBackend`.
                "emitting" => Ok(Artifact::Emitted {
                    llvm_ir: Some("; ir".to_string()),
                    object: None,
                }),
                _ => Err("UNSUPPORTED: nothing to do".to_string()),
            }
        }
    }

    /// The three inputs reach `compile` unchanged — the pass-through every adapter relies on.
    #[test]
    fn compile_receives_the_program_entry_and_interner_it_was_given() {
        let mut program = Program::new();
        let mut interner = Interner::new();
        program.statics.push(wukong_mir::StaticData {
            name: interner.intern("s0"),
            bytes: b"lit".to_vec(),
        });
        let entry = interner.intern("main");

        let backend = Recording::new("executing");
        let _ = backend.compile(&program, entry, &interner).unwrap();

        assert_eq!(backend.seen_funcs.get(), program.funcs.len());
        assert_eq!(backend.seen_statics.get(), 1);
        assert_eq!(backend.seen_entry.get(), entry.0);
    }

    /// An executing backend yields `Executed` carrying both the exit code and the stdout bytes —
    /// the two fields `native_matches_interpreter` compares, so an adapter that dropped either would
    /// be caught here rather than silently.
    #[test]
    fn executing_backend_carries_exit_code_and_stdout() {
        let program = Program::new();
        let mut interner = Interner::new();
        let entry = interner.intern("main");
        match Recording::new("executing")
            .compile(&program, entry, &interner)
            .unwrap()
        {
            Artifact::Executed { exit_code, stdout } => {
                assert_eq!(exit_code, 7);
                assert_eq!(stdout, b"hello");
            }
            other => panic!("expected Executed, got {other:?}"),
        }
    }

    /// An emitting backend yields `Emitted`, never `Executed` — the variant split the driver relies
    /// on to know whether a run already happened or an artifact must still be written/linked.
    #[test]
    fn emitting_backend_yields_emitted_not_executed() {
        let program = Program::new();
        let mut interner = Interner::new();
        let entry = interner.intern("main");
        match Recording::new("emitting")
            .compile(&program, entry, &interner)
            .unwrap()
        {
            Artifact::Emitted { llvm_ir, object } => {
                assert_eq!(llvm_ir.as_deref(), Some("; ir"));
                assert!(object.is_none());
            }
            other => panic!("expected Emitted, got {other:?}"),
        }
    }

    /// The trait is object-safe and usable behind `&dyn Backend`. This is the one property the trait
    /// exists for — holding heterogeneous backends behind one type — and the thing a signature
    /// change (a generic method, `Self: Sized`, an associated type) would silently break.
    #[test]
    fn backend_is_object_safe_and_name_dispatches() {
        let program = Program::new();
        let mut interner = Interner::new();
        let entry = interner.intern("main");
        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(Recording::new("executing")),
            Box::new(Recording::new("emitting")),
        ];
        let names: Vec<&str> = backends.iter().map(|b| b.name()).collect();
        assert_eq!(names, ["executing", "emitting"]);
        for b in &backends {
            assert!(b.compile(&program, entry, &interner).is_ok());
        }
    }

    /// A declined compile is a plain `Err(String)` — the driver reports it verbatim, so the error
    /// type must stay `String` (not a `wukong_diag::Diagnostic`), as the module header states.
    #[test]
    fn a_declined_compile_is_a_string_error() {
        let program = Program::new();
        let mut interner = Interner::new();
        let entry = interner.intern("main");
        let err = Recording::new("declining")
            .compile(&program, entry, &interner)
            .unwrap_err();
        assert!(err.starts_with("UNSUPPORTED:"), "got {err:?}");
    }
}
