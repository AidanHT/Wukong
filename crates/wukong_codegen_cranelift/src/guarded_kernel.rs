//! Guarded calls to a fixed buffer-table ABI, retaining ownership of executable memory.
use super::*;

/// Compiled dispatcher with MIR signature `(Ptr, I64) -> Void`.
/// Its executable memory stays alive until this handle is dropped.
pub struct JitBufferKernel {
    program: JitProgram,
}

/// Compile a dispatcher once. Calls must obey [`JitBufferKernel::run`]'s safety contract.
pub fn jit_compile_buffer_kernel(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<JitBufferKernel, String> {
    let errors = wukong_mir::verify::verify_program(program);
    if !errors.is_empty() {
        return Err(format!("invalid buffer dispatcher MIR: {errors:?}"));
    }
    Ok(JitBufferKernel {
        program: jit_compile_entry(
            program,
            entry,
            interner,
            true,
            TargetCpu::Native,
            BackendOpt::default(),
            default_parallel(),
        )?,
    })
}

impl JitBufferKernel {
    /// Invoke the dispatcher, capturing output and reporting assertion failure. Calls use the
    /// same serialized runtime state and 512 MiB worker stack as [`JitProgram::run`].
    ///
    /// # Safety
    /// `entry` must select a valid dispatcher branch. The table must contain every pointer that
    /// branch reads, and each pointer must address initialized storage of the type, alignment and
    /// extent required by all code it calls. Storage written by that code must be writable and
    /// exclusively accessible for the call; aliases must satisfy the compiled code's contract.
    /// Neither table nor buffers may be retained by the compiled code after it returns. The
    /// worker joins before this method returns, so the table, buffers and this handle must remain
    /// valid until then. Signature validation alone does not prove these memory requirements.
    pub unsafe fn run(&self, buffers: &[*mut f32], entry: u64) -> Result<Vec<u8>, String> {
        let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clear();
        reset_failure();
        let code = self.program.code as usize;
        let table = buffers.as_ptr() as usize;
        // SAFETY: the constructor validates the fixed ABI. The caller guarantees the table and
        // pointees remain live; the scoped worker joins while self still owns executable memory.
        run_on_big_stack(move || unsafe {
            let call: extern "C" fn(*const *mut f32, u64) = std::mem::transmute(code);
            call(table as *const *mut f32, entry);
        });
        if let Some(failure) = take_failure() {
            return Err(failure);
        }
        Ok(OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }
}

#[cfg(test)]
mod tests;
