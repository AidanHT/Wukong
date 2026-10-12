//! The AOT entry contract and the object's symbol policy.
//!
//! Two defects lived where `emit_object` declared every user function `Linkage::Export` under its
//! bare source name:
//!
//! * **The user's `main` was the C entry point as written.** A `fn main() -> f32` or a `fn main()`
//!   left `eax`/`w0` undefined, so the executable exited with whatever was in the register, while
//!   `--run` exits with the converted value. An `i32` main exited with its full 32-bit value on
//!   Windows and the low 8 bits elsewhere, so `return 300` was 300 on one OS and 44 on another and
//!   under `--run`. And the user's `main` ran on the OS main-thread stack (1 MiB on Windows), while
//!   both `--run` backends give the program a 512 MiB worker, so recursion depth depended on the
//!   artifact.
//! * **Every user function was an exported global symbol under its source name**, so a `.wk`
//!   function called `write`, `fmod`, `free` or `memcpy` collided with, or interposed on, the C
//!   library at link time, differently on each OS, while the JIT resolved it module-locally.
//!
//! The fix is the shape C and Rust toolchains use. The user's `main` becomes an internal function,
//! called by a synthesized exported `main` through [`RT_RUN_MAIN`], which runs it on a 512 MiB
//! worker thread (matching `run_on_big_stack`) and returns its code. A thunk converts the user's
//! return value exactly as `invoke_code` does for the JIT (a float saturates to `i64`, void is 0)
//! and keeps the low 8 bits, the code `wukongc --run` reports. In an executable object every user
//! function is also internal (`Linkage::Local`) and mangled `_WK<len><name>`, so it can neither
//! collide with nor interpose on any library symbol. A library object (`--emit=obj`) keeps its
//! non-entry functions exported under their source names: that is its C interface.

use cranelift_codegen::ir::{types, AbiParam, InstBuilder, Signature, Type, Value};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use wukong_mir::{Function, MirType, Program};
use wukong_span::Interner;

/// The runtime entry that runs the program's entry thunk on a big-stack worker thread and returns
/// its exit code: `int wukong_rt_run_main(int (*entry)(void))`. Provided by both link runtimes
/// (`wukong_driver`'s Rust shim and its C fallback).
pub(crate) const RT_RUN_MAIN: &str = "wukong_rt_run_main";

/// The internal thunk the runtime calls: `() -> i32`, the user's `main` plus the exit-code
/// conversion. The `wukong_` prefix is reserved by sema (E0300), so no user function can take it.
const ENTRY_THUNK: &str = "wukong_entry_thunk";

/// What kind of object [`crate::emit_object`] and its siblings produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ObjectKind {
    /// `--emit=obj`: a relocatable object other code links against. Its functions keep their source
    /// names as exported symbols (its C interface); only an entry `main` is internalized behind the
    /// synthesized C `main`.
    #[default]
    Library,
    /// `--emit=exe`: the object is linked into an executable and nothing outside it calls in, so
    /// every user function is internal and mangled.
    Executable,
}

/// How [`crate::populate_module`] names and links the user's functions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SymbolMode {
    /// The JIT: source names, exported; functions are reached by `FuncId`, never by name.
    Jit,
    /// An object file of the given kind.
    Object(ObjectKind),
}

/// The symbol and linkage of the user function `name`. `is_entry` marks the function the
/// synthesized C `main` calls.
pub(crate) fn user_symbol(name: &str, mode: SymbolMode, is_entry: bool) -> (String, Linkage) {
    match mode {
        SymbolMode::Jit => (name.to_string(), Linkage::Export),
        SymbolMode::Object(ObjectKind::Library) if !is_entry => (name.to_string(), Linkage::Export),
        SymbolMode::Object(_) => (mangle(name), Linkage::Local),
    }
}

/// `_WK<byte length><name>`: a reserved-looking, length-prefixed spelling (in the style of the
/// Itanium and Rust v0 manglings) that no C library defines and that cannot be confused with
/// another mangled name, since the length fixes where the name ends.
pub(crate) fn mangle(name: &str) -> String {
    format!("_WK{}{}", name.len(), name)
}

/// The function the synthesized C `main` calls, if this object gets one: the program's `main`
/// when building an object (never for the JIT). An executable whose `main` cannot be an entry
/// (parameters, or an aggregate or vector return) is an error rather than an object without a C
/// `main`; sema's E0331 rejects such a program first, so this is a backstop.
pub(crate) fn entry_function<'p>(
    program: &'p Program,
    interner: &Interner,
    mode: SymbolMode,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
) -> Result<Option<&'p Function>, String> {
    let SymbolMode::Object(kind) = mode else {
        return Ok(None);
    };
    let Some(main) = program
        .funcs
        .iter()
        .find(|f| interner.resolve(f.name) == "main")
    else {
        return Ok(None);
    };
    let sig = crate::signature_of(main, ptr_ty, call_conv);
    let scalar_ret = match &main.ret {
        MirType::Void => true,
        ret => crate::cl_type(ret, ptr_ty).is_some_and(|t| t.is_int() || t.is_float()),
    };
    if sig.params.is_empty() && scalar_ret {
        return Ok(Some(main));
    }
    match kind {
        ObjectKind::Library => Ok(None),
        ObjectKind::Executable => Err(
            "native entry point `main` must take no parameters and return an integer, a float \
             or nothing to be linked into an executable"
                .to_string(),
        ),
    }
}

/// Define the entry thunk and the exported C `main` that hands it to [`RT_RUN_MAIN`]. `user_id` is
/// the (already defined) user `main`. Both are recorded in `log` when one is kept (`-g`).
pub(crate) fn define_entry<M: Module>(
    module: &mut M,
    user_id: FuncId,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
    mut log: Option<&mut crate::debuginfo::CodeLog>,
) -> Result<(), String> {
    use crate::debuginfo::CodeOrigin;
    let mut code_sig = Signature::new(call_conv);
    code_sig.returns.push(AbiParam::new(types::I32));
    let mut run_sig = Signature::new(call_conv);
    run_sig.params.push(AbiParam::new(ptr_ty));
    run_sig.returns.push(AbiParam::new(types::I32));

    let thunk_id = module
        .declare_function(ENTRY_THUNK, Linkage::Local, &code_sig)
        .map_err(|e| e.to_string())?;
    let run_id = module
        .declare_function(RT_RUN_MAIN, Linkage::Import, &run_sig)
        .map_err(|e| e.to_string())?;
    let main_id = module
        .declare_function("main", Linkage::Export, &code_sig)
        .map_err(|e| e.to_string())?;

    let mut ctx = module.make_context();
    let mut fbctx = FunctionBuilderContext::new();

    // The thunk: call the user's `main`, convert its result to the 8-bit exit code.
    ctx.func.signature = code_sig.clone();
    let user_ref = module.declare_func_in_func(user_id, &mut ctx.func);
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let block = b.create_block();
        b.switch_to_block(block);
        b.seal_block(block);
        let call = b.ins().call(user_ref, &[]);
        let ret = b.inst_results(call).first().copied();
        let code = exit_code(&mut b, ret);
        b.ins().return_(&[code]);
        b.finalize();
    }
    module
        .define_function(thunk_id, &mut ctx)
        .map_err(|e| format!("cranelift define entry thunk: {e:?}"))?;
    if let (Some(log), Some(code)) = (log.as_deref_mut(), ctx.compiled_code()) {
        log.record_compiled(thunk_id, CodeOrigin::Entry(ENTRY_THUNK), code, module.isa());
    }
    module.clear_context(&mut ctx);

    // The C `main`: run the thunk on the runtime's big-stack worker and return its code.
    ctx.func.signature = code_sig;
    let run_ref = module.declare_func_in_func(run_id, &mut ctx.func);
    let thunk_ref = module.declare_func_in_func(thunk_id, &mut ctx.func);
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let block = b.create_block();
        b.switch_to_block(block);
        b.seal_block(block);
        let entry = b.ins().func_addr(ptr_ty, thunk_ref);
        let call = b.ins().call(run_ref, &[entry]);
        let code = b.inst_results(call)[0];
        b.ins().return_(&[code]);
        b.finalize();
    }
    module
        .define_function(main_id, &mut ctx)
        .map_err(|e| format!("cranelift define C main: {e:?}"))?;
    if let (Some(log), Some(code)) = (log, ctx.compiled_code()) {
        log.record_compiled(main_id, CodeOrigin::Entry("main"), code, module.isa());
    }
    module.clear_context(&mut ctx);
    Ok(())
}

/// The process exit code of an entry that returned `ret` (`None`: void). Mirrors the `--run` chain
/// (`invoke_code`'s `as i64`, then the driver's `as i32` and `main.rs`'s `as u8`): a float converts
/// with saturation (NaN is 0), an integer keeps its low bits, void is 0, and only the low 8 bits
/// survive, so every OS reports the same code.
fn exit_code(b: &mut FunctionBuilder, ret: Option<Value>) -> Value {
    let Some(v) = ret else {
        return b.ins().iconst(types::I32, 0);
    };
    let ty = b.func.dfg.value_type(v);
    let code = if ty.is_float() {
        let wide = b.ins().fcvt_to_sint_sat(types::I64, v);
        b.ins().ireduce(types::I32, wide)
    } else if ty.bits() > 32 {
        b.ins().ireduce(types::I32, v)
    } else if ty.bits() < 32 {
        b.ins().uextend(types::I32, v)
    } else {
        v
    };
    b.ins().band_imm(code, 0xFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mangling_is_length_prefixed() {
        assert_eq!(mangle("main"), "_WK4main");
        assert_eq!(mangle("write"), "_WK5write");
        assert_eq!(mangle("a.b.fmod"), "_WK8a.b.fmod");
    }

    #[test]
    fn executable_functions_are_internal_and_mangled() {
        let exe = SymbolMode::Object(ObjectKind::Executable);
        assert_eq!(
            user_symbol("write", exe, false),
            ("_WK5write".to_string(), Linkage::Local)
        );
        assert_eq!(
            user_symbol("main", exe, true),
            ("_WK4main".to_string(), Linkage::Local)
        );
    }

    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
        (program, interner)
    }

    fn contains(object: &[u8], symbol: &str) -> bool {
        object.windows(symbol.len()).any(|w| w == symbol.as_bytes())
    }

    /// The symbol policy reaches the object file: an executable object names the user's `write`
    /// `_WK5write`, never `write`, a library object keeps `write` as its C interface, and both
    /// hide the user's `main` behind a synthesized C `main` that imports `wukong_rt_run_main`.
    #[test]
    fn objects_carry_the_symbol_policy_and_the_synthesized_entry() {
        let (program, interner) = lower(
            "fn write(x: i64) -> i64 { return x * 2; } fn main() -> i32 { print(write(21)); return 0; }",
        );
        let exe = crate::emit_object_kind_for_with(
            &program,
            &interner,
            crate::TargetCpu::Native,
            crate::BackendOpt::default(),
            ObjectKind::Executable,
        )
        .unwrap();
        assert!(contains(&exe, "_WK5write"));
        assert!(contains(&exe, "_WK4main"));
        assert!(contains(&exe, RT_RUN_MAIN));
        assert!(contains(&exe, ENTRY_THUNK));
        let lib = crate::emit_object(&program, &interner).unwrap();
        assert!(!contains(&lib, "_WK5write"));
        assert!(contains(&lib, "_WK4main"));
        assert!(contains(&lib, RT_RUN_MAIN));
    }

    /// A program without a `main` (a kernel library) gets no C `main` and imports no entry runtime.
    #[test]
    fn a_library_without_main_gets_no_entry() {
        let (program, interner) = lower("fn twice(x: i64) -> i64 { return x * 2; }");
        let lib = crate::emit_object(&program, &interner).unwrap();
        assert!(!contains(&lib, RT_RUN_MAIN));
        assert!(!contains(&lib, ENTRY_THUNK));
    }

    #[test]
    fn library_objects_keep_their_c_interface_but_not_main() {
        let lib = SymbolMode::Object(ObjectKind::Library);
        assert_eq!(
            user_symbol("saxpy", lib, false),
            ("saxpy".to_string(), Linkage::Export)
        );
        assert_eq!(
            user_symbol("main", lib, true),
            ("_WK4main".to_string(), Linkage::Local)
        );
        assert_eq!(
            user_symbol("main", SymbolMode::Jit, false),
            ("main".to_string(), Linkage::Export)
        );
    }
}
