//! Training execution selected once, with one owning JIT handle for all native steps.
use super::{BackendKind, Interner};
use wukong_mir::{Builder, CmpOp, MirType, Op, Program};
use wukong_span::Symbol;

pub(super) struct TrainingExecutor {
    kernel: Option<wukong_codegen_cranelift::JitBufferKernel>,
    entries: Vec<(Symbol, Vec<usize>)>,
}

impl TrainingExecutor {
    pub(super) fn new(
        program: &mut Program,
        interner: &mut Interner,
        backend: BackendKind,
        entries: Vec<(Symbol, Vec<usize>)>,
    ) -> Result<Self, String> {
        for (name, lens) in &entries {
            let f = program.function(*name).ok_or("missing training kernel")?;
            if f.params.len() != lens.len()
                || f.params.iter().any(|&p| f.value_type(p) != &MirType::Ptr)
                || lens.iter().any(|&n| !super::valid_train_extent(n))
            {
                return Err("training kernel must take the declared f32 buffers".into());
            }
        }
        let kernel = match backend {
            BackendKind::Interp => None,
            BackendKind::Native => {
                let mut suffix = 0;
                let name = loop {
                    let candidate =
                        interner.intern(&format!("__wukong_training_dispatch_{suffix}"));
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
                    let ret = program.function(*entry).unwrap().ret.clone();
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

    pub(super) fn run(
        &self,
        program: &Program,
        entry: Symbol,
        buffers: &mut [Vec<f32>],
        interner: &Interner,
    ) -> Result<(), String> {
        let (selector, (_, lens)) = self
            .entries
            .iter()
            .enumerate()
            .find(|(_, (name, _))| *name == entry)
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
                unsafe {
                    kernel.run(&pointers, selector as u64)?;
                }
                Ok(())
            }
            None => {
                let mut views: Vec<_> = buffers.iter_mut().map(|v| v.as_mut_slice()).collect();
                wukong_interp::run_kernel_f32(program, entry, &mut views, interner)
            }
        }
    }
}
