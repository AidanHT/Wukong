//! Training execution selected once, with one owning JIT handle for all native steps — and the
//! kernel dispatcher that both a native `--train` run and a `--train --emit=exe|obj` artifact call.
//!
//! The loop itself is `wukong_runtime::train::run_training`; it reaches every kernel through a
//! selector (`wukong_runtime::train`'s ABI). In process, [`TrainingExecutor::run_selector`] answers
//! it on the interpreter or through the JIT-compiled dispatcher; in an artifact the runtime calls the
//! object's compiled copy of the same dispatcher, which [`add_dispatcher`] generates for both.
use super::{BackendKind, Interner};
use wukong_mir::{Builder, CmpOp, MirType, Op, Program};
use wukong_span::Symbol;

/// Every entry must be a function of `program` taking exactly its declared dense f32 buffers.
fn check_entries(program: &Program, entries: &[(Symbol, Vec<usize>)]) -> Result<(), String> {
    for (name, lens) in entries {
        let f = program.function(*name).ok_or("missing training kernel")?;
        if f.params.len() != lens.len()
            || f.params.iter().any(|&p| f.value_type(p) != &MirType::Ptr)
            || lens
                .iter()
                .any(|&n| !wukong_runtime::train::valid_extent(n))
        {
            return Err("training kernel must take the declared f32 buffers".into());
        }
    }
    Ok(())
}

/// Append the kernel dispatcher `(table: Ptr, selector: I64) -> void` to `program` and return its
/// name: selector `i` loads entry `i`'s buffer pointers from the table, in order, and calls it. The
/// name is `__wukong_training_dispatch_<n>` for the first `n` that no function or static of the
/// program already uses. The whole program is verified before it is returned.
pub(super) fn add_dispatcher(
    program: &mut Program,
    interner: &mut Interner,
    entries: &[(Symbol, Vec<usize>)],
) -> Result<Symbol, String> {
    check_entries(program, entries)?;
    let mut suffix = 0;
    let name = loop {
        let candidate = interner.intern(&format!("__wukong_training_dispatch_{suffix}"));
        if program.function(candidate).is_none()
            && !program.statics.iter().any(|s| s.name == candidate)
        {
            break candidate;
        }
        suffix += 1;
    };
    let mut b = Builder::new(name, MirType::Void);
    let table = b.add_param(MirType::Ptr);
    let selector = b.add_param(MirType::I64);
    for (index, (entry, lens)) in entries.iter().enumerate() {
        let selected = b.build(MirType::I64, Op::ConstInt(index as i128, MirType::I64));
        let condition = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, selector, selected));
        let call = b.new_block();
        let next = b.new_block();
        b.cond_br(condition, call, vec![], next, vec![]);
        b.switch_to(call);
        let mut args = Vec::with_capacity(lens.len());
        for i in 0..lens.len() {
            let offset = b.build(MirType::I64, Op::ConstInt(i as i128, MirType::I64));
            let slot = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: table,
                    index: offset,
                    elem: MirType::Ptr,
                },
            );
            args.push(b.build(MirType::Ptr, Op::Load(slot, MirType::Ptr)));
        }
        let ret = program
            .function(*entry)
            .ok_or("missing training kernel")?
            .ret
            .clone();
        b.push(
            (ret != MirType::Void).then_some(ret),
            Op::Call { func: *entry, args },
        );
        b.ret(None);
        b.switch_to(next);
    }
    b.ret(None);
    program.funcs.push(b.finish());
    wukong_opt::verify_program_stage(program, "native training dispatcher")
        .map_err(|e| e.to_string())?;
    Ok(name)
}

pub(super) struct TrainingExecutor {
    kernel: Option<wukong_codegen_cranelift::JitBufferKernel>,
    entries: Vec<(Symbol, Vec<usize>)>,
}

impl TrainingExecutor {
    /// `entries` in selector order. The native backend appends the dispatcher to `program` and
    /// JIT-compiles it once; the interpreter calls each entry by name.
    pub(super) fn new(
        program: &mut Program,
        interner: &mut Interner,
        backend: BackendKind,
        entries: Vec<(Symbol, Vec<usize>)>,
    ) -> Result<Self, String> {
        check_entries(program, &entries)?;
        let kernel = match backend {
            BackendKind::Interp => None,
            BackendKind::Native => {
                let name = add_dispatcher(program, interner, &entries)?;
                Some(wukong_codegen_cranelift::jit_compile_buffer_kernel(
                    program, name, interner,
                )?)
            }
            _ => return Err("--train supports only --backend=interp or --backend=native".into()),
        };
        Ok(Self { kernel, entries })
    }

    #[cfg(test)]
    pub(super) fn is_native(&self) -> bool {
        self.kernel.is_some()
    }

    /// Run the kernel named `entry` (see [`TrainingExecutor::run_selector`]).
    #[cfg(test)]
    pub(super) fn run(
        &self,
        program: &Program,
        entry: Symbol,
        buffers: &mut [Vec<f32>],
        interner: &Interner,
    ) -> Result<(), String> {
        let selector = self
            .entries
            .iter()
            .position(|(name, _)| *name == entry)
            .ok_or("unknown training kernel")?;
        self.run_selector(program, selector, buffers, interner)
    }

    /// Run the kernel `selector` names (its index in `entries`) over `buffers`, which must have
    /// exactly that kernel's declared lengths, and return what it printed.
    pub(super) fn run_selector_output(
        &self,
        program: &Program,
        selector: usize,
        buffers: &mut [Vec<f32>],
        interner: &Interner,
    ) -> Result<Vec<u8>, String> {
        let (entry, lens) = self
            .entries
            .get(selector)
            .ok_or("unknown training kernel")?;
        if buffers.len() != lens.len() || buffers.iter().zip(lens).any(|(b, &n)| b.len() != n) {
            return Err("training buffer lengths do not match the kernel signature".into());
        }
        match &self.kernel {
            Some(kernel) => {
                // Refresh the table each time: AdamW replaces Vec allocations between calls.
                let pointers: Vec<_> = buffers.iter_mut().map(|v| v.as_mut_ptr()).collect();
                // SAFETY: training's semantic gate permits only dense f32 buffers with checked
                // extents; generated kernels and this dispatcher are verified above. Each entry
                // has the validated argument count/lengths. Each Vec owns disjoint writable
                // storage and remains exclusively borrowed until the worker joins. These kernels
                // do not retain pointers. The owning kernel handle lives across every call.
                unsafe { kernel.run(&pointers, selector as u64) }
            }
            None => {
                let mut views: Vec<_> = buffers.iter_mut().map(|v| v.as_mut_slice()).collect();
                wukong_interp::run_kernel_f32_with_output(program, *entry, &mut views, interner)
            }
        }
    }

    /// [`TrainingExecutor::run_selector_output`], with what the kernel printed written to stdout
    /// at once — exactly what the same kernel prints, at the same point of the run, inside a
    /// `--train --emit=exe` executable (whose kernels write to stdout directly).
    pub(super) fn run_selector(
        &self,
        program: &Program,
        selector: usize,
        buffers: &mut [Vec<f32>],
        interner: &Interner,
    ) -> Result<(), String> {
        let printed = self.run_selector_output(program, selector, buffers, interner)?;
        if !printed.is_empty() {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(&printed);
            let _ = out.flush();
        }
        Ok(())
    }
}
