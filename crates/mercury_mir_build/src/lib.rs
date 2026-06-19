//! `mercury_mir_build` — lowers the type-checked AST into MIR.
//!
//! Strategy: every local (and every parameter) gets a stack slot via `alloca`; reads `load` and
//! writes `store`. This keeps lowering simple and non-recursive in its SSA reasoning — a later
//! `mem2reg` pass promotes the slots to SSA registers. Control flow is lowered directly to a CFG
//! of basic blocks with `br`/`cond_br`.
//!
//! The lowerer covers the scalar + pointer + control-flow + direct-call core end to end. Tensor,
//! SIMD-method, and parallel-loop constructs are not yet lowered; encountering one records a
//! diagnostic and substitutes a placeholder so the rest of the function still lowers.

use std::collections::{HashMap, HashSet};

use mercury_ast::{
    self as ast, Block, Expr, ExprKind, FnDecl, ForIter, Module, Pattern, Stmt, StmtKind,
};
use mercury_diag::Diagnostic;
use mercury_mir::{BinOp, Builder, CastKind, CmpOp, Function, MirType, Op, Program, ValueId};
use mercury_sema::{DefKind, SemaResult};
use mercury_span::{Interner, Span, Symbol};
use mercury_types::Ty;

/// Array-repeat initializers (`[v; n]`) with at most this many elements are unrolled to
/// straight-line stores; larger ones lower to a fill loop to keep the IR compact.
const REPEAT_UNROLL_LIMIT: u32 = 8;

/// Lower a whole module to a MIR [`Program`]. Only functions with bodies are lowered.
pub fn lower_program(
    module: &Module,
    sema: &SemaResult,
    interner: &mut Interner,
) -> (Program, Vec<Diagnostic>) {
    let mut diags = Vec::new();
    let mut program = Program::new();
    // Runtime symbols the matmul recognizer lowers a GEMM nest to (interned once, threaded down).
    let gemm = GemmSyms {
        mm: interner.intern("mercury_sgemm"),
        mm_par: interner.intern("mercury_sgemm_parallel"),
        nt: interner.intern("mercury_sgemm_nt"),
        nt_par: interner.intern("mercury_sgemm_nt_parallel"),
        nt_epi: interner.intern("mercury_sgemm_nt_epi"),
        vmath: interner.intern("mercury_vmath_f32"),
        velem: interner.intern("mercury_velem_f32"),
        sred_par: interner.intern("mercury_sreduce_f32_parallel"),
        norm: interner.intern("mercury_norm_f32"),
        norm_affine: interner.intern("mercury_norm_affine_f32"),
        i8nt: interner.intern("mercury_i8gemm_nt"),
        i8nt_par: interner.intern("mercury_i8gemm_nt_parallel"),
    };
    for item in &module.items {
        if let ast::ItemKind::Fn(f) = &item.kind {
            if let Some(body) = &f.body {
                // Whole-function matmul: lower the entire nest to a single (optionally parallel)
                // `mercury_sgemm` call — the tuned 256-bit AVX2/FMA microkernel.
                if let Some(nest) = matmul_fn(body, sema, interner) {
                    let parallel = has_parallel_attr(item, interner);
                    let func =
                        lower_matmul_fn(f, &nest, parallel, sema, interner, gemm, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // Whole-function int8 quantized matmul (`u8×i8→i32` `C = A·Bᵀ`) → the int8 GEMM
                // microkernel. Checked *before* the `@parallel` outliner below so a `@parallel` int8
                // kernel dispatches to the multicore `mercury_i8gemm_nt_parallel` instead of being
                // outlined to a scalar loop (rows are independent, so it stays deterministic).
                // Integer math, so the kernel equals the scalar nest bit-for-bit (no reassoc).
                if let Some(nest) = i8matmul_fn(body, sema, interner) {
                    let parallel = has_parallel_attr(item, interner);
                    let func =
                        lower_i8matmul_fn(f, &nest, parallel, sema, interner, gemm, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                // `@parallel` on a function whose whole body is `for i in 0..n { … }` over array
                // (pointer) parameters is lowered to a multi-threaded runtime dispatch: the loop
                // body becomes a separate ranged function, and the original becomes a thin wrapper
                // that hands chunks to `mercury_parallel_for`.
                if has_parallel_attr(item, interner) {
                    if let Some((idx, hi, loop_body)) = parallel_spec(f, body, sema, interner) {
                        let base = interner.resolve(f.name.sym).to_string();
                        let par_sym = interner.intern(&format!("{base}$par"));
                        let pfor_sym = interner.intern("mercury_parallel_for");
                        let (outlined, wrapper) = lower_parallel(
                            f, par_sym, pfor_sym, idx, hi, loop_body, sema, interner, gemm,
                            &mut diags,
                        );
                        program.funcs.push(outlined);
                        program.funcs.push(wrapper);
                        continue;
                    }
                    // A `@parallel` function that is not a single elementwise loop — e.g. a reduction
                    // (`let mut s = 0; for k { s += x[k]*y[k] }; …`). Lower it normally, but with any
                    // recognized reduction loop dispatched to the multicore reduction kernel.
                    let func = lower_fn(f, body, sema, interner, gemm, true, &mut diags);
                    program.funcs.push(func);
                    continue;
                }
                let func = lower_fn(f, body, sema, interner, gemm, false, &mut diags);
                program.funcs.push(func);
            }
        }
    }
    (program, diags)
}

/// Does this item carry a `@parallel` attribute?
fn has_parallel_attr(item: &ast::Item, interner: &Interner) -> bool {
    item.attrs
        .iter()
        .any(|a| interner.resolve(a.name.sym) == "parallel")
}

/// Recognize a parallelizable function: its entire body is a single `for idx in 0..hi { … }` over
/// pointer (array) parameters. Returns the index name, the upper-bound expression, and the loop
/// body. Anything else falls back to ordinary sequential lowering.
fn parallel_spec<'a>(
    f: &'a FnDecl,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<(Symbol, &'a Expr, &'a Block)> {
    // Captures are exactly the parameters, so they must all be arrays (passed by pointer).
    let param_tys: Vec<Ty> = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => sig.params.clone(),
        _ => return None,
    };
    if param_tys.is_empty()
        || !param_tys
            .iter()
            .all(|t| matches!(mir_ty(t), MirType::Array(..)))
    {
        return None;
    }
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    let ForIter::Range {
        start,
        end: Some(end),
        inclusive: false,
        step: None,
    } = iter
    else {
        return None;
    };
    // Require the range to start at literal 0 (the runtime iterates [0, hi)).
    let ExprKind::Int(s) = &start.kind else {
        return None;
    };
    if parse_int(interner.resolve(*s)) != 0 {
        return None;
    }
    let ast::PatKind::Ident(name) = &pat.kind else {
        return None;
    };
    Some((*name, end, lb))
}

#[allow(clippy::too_many_arguments)]
fn lower_fn(
    f: &FnDecl,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    parallel_fn: bool,
    diags: &mut Vec<Diagnostic>,
) -> Function {
    // Recover the resolved signature for parameter/return types.
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (f.params.iter().map(|_| Ty::Unknown).collect(), Ty::Unit),
    };
    let ret_mir = mir_ty(&ret_ty);

    let mut fl = FnLowerer {
        builder: Builder::new(f.name.sym, ret_mir.clone()),
        sema,
        interner,
        diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn,
        vec_loads: HashMap::new(),
    };

    // Declare all parameters first (so their value ids are contiguous), then materialize each.
    // Arrays are passed by base pointer (ABI type `Ptr`); scalars by value.
    let param_vals: Vec<ValueId> = param_tys
        .iter()
        .map(|pty| fl.builder.add_param(param_abi_ty(pty)))
        .collect();
    for ((p, pty), val) in f.params.iter().zip(&param_tys).zip(param_vals) {
        let mty = mir_ty(pty);
        if matches!(mty, MirType::Array(..)) {
            // The parameter value *is* the array's base pointer; bind it directly so indexing
            // geps off it (no copy into a local slot).
            fl.bind(p.name.sym, val, mty);
        } else {
            let slot = fl.builder.alloca(mty.clone());
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: val,
            });
            fl.bind(p.name.sym, slot, mty);
        }
    }

    let tail = fl.lower_block(body);
    if !fl.terminated {
        match (&ret_mir, tail) {
            (MirType::Void, _) => fl.builder.ret(None),
            (_, Some(v)) => fl.builder.ret(Some(v)),
            (_, None) => fl.builder.set_term(mercury_mir::Terminator::Unreachable),
        }
    }
    fl.builder.finish()
}

/// Lower a `@parallel for idx in 0..hi { body }` function into two MIR functions:
///   * `par_sym(start: i64, end: i64, env: *ptr)` — the loop body over `[start, end)`, reading the
///     array base pointers back from `env`;
///   * the original `f.name(params…)` — a wrapper that packs the parameter pointers into a stack
///     `env`, then calls `mercury_parallel_for(hi, &par, env)`.
#[allow(clippy::too_many_arguments)]
fn lower_parallel(
    f: &FnDecl,
    par_sym: Symbol,
    pfor_sym: Symbol,
    idx: Symbol,
    hi: &Expr,
    loop_body: &Block,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    diags: &mut Vec<Diagnostic>,
) -> (Function, Function) {
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (
            f.params.iter().map(|_| Ty::Unknown).collect::<Vec<_>>(),
            Ty::Unit,
        ),
    };
    let ret_mir = mir_ty(&ret_ty);
    let k = f.params.len();

    // ---- outlined body: par_sym(start, end, env) ----
    let outlined = {
        let mut fl = FnLowerer {
            builder: Builder::new(par_sym, MirType::Void),
            sema,
            interner,
            diags: &mut *diags,
            scopes: vec![HashMap::new()],
            terminated: false,
            loops: Vec::new(),
            gemm,
            parallel_fn: false,
            vec_loads: HashMap::new(),
        };
        let start = fl.builder.add_param(MirType::I64);
        let end = fl.builder.add_param(MirType::I64);
        let env = fl.builder.add_param(MirType::Ptr);
        // Recover each array base pointer from env[k] and bind it to the parameter name.
        for (idx_k, (p, pty)) in f.params.iter().zip(&param_tys).enumerate() {
            let mty = mir_ty(pty);
            let kidx = fl
                .builder
                .build(MirType::I64, Op::ConstInt(idx_k as i128, MirType::I64));
            let slot = fl.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: env,
                    index: kidx,
                    elem: MirType::Ptr,
                },
            );
            let base = fl.builder.build(MirType::Ptr, Op::Load(slot, MirType::Ptr));
            fl.bind(p.name.sym, base, mty);
        }
        // The index variable's source type (the range's element type) drives the loop so the
        // body's index arithmetic matches sema; the i64 runtime bounds are coerced into it.
        let ity = fl.expr_mir(hi);
        let ity = if ity.is_int() { ity } else { MirType::I64 };
        fl.lower_ranged_loop(idx, start, end, ity, loop_body);
        if !fl.terminated {
            fl.builder.ret(None);
        }
        fl.builder.finish()
    };

    // ---- wrapper: f.name(params…) ----
    let wrapper = {
        let mut fl = FnLowerer {
            builder: Builder::new(f.name.sym, ret_mir.clone()),
            sema,
            interner,
            diags: &mut *diags,
            scopes: vec![HashMap::new()],
            terminated: false,
            loops: Vec::new(),
            gemm,
            parallel_fn: false,
            vec_loads: HashMap::new(),
        };
        let param_vals: Vec<ValueId> = param_tys
            .iter()
            .map(|pty| fl.builder.add_param(param_abi_ty(pty)))
            .collect();
        let env = fl
            .builder
            .alloca(MirType::Array(Box::new(MirType::Ptr), k as u32));
        for (idx_k, pv) in param_vals.iter().enumerate() {
            let kidx = fl
                .builder
                .build(MirType::I64, Op::ConstInt(idx_k as i128, MirType::I64));
            let slot = fl.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: env,
                    index: kidx,
                    elem: MirType::Ptr,
                },
            );
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: *pv,
            });
        }
        let n_raw = fl.lower_expr(hi);
        let n_ty = fl.expr_mir(hi);
        let n = fl.coerce_to(n_raw, &n_ty, &MirType::I64, true);
        let bodyaddr = fl.builder.build(MirType::Ptr, Op::FuncAddr(par_sym));
        fl.builder.build_void(Op::Call {
            func: pfor_sym,
            args: vec![n, bodyaddr, env],
        });
        if !fl.terminated {
            match ret_mir {
                MirType::Void => fl.builder.ret(None),
                _ => {
                    let z = fl.const_zero(ret_mir.clone());
                    fl.builder.ret(Some(z));
                }
            }
        }
        fl.builder.finish()
    };

    (outlined, wrapper)
}

/// The runtime entry points the recognizers dispatch to: the four GEMM kernels (`C = A·B` and the
/// `nn.Linear` form `C = A·Bᵀ`, each serial or `@parallel`), the fused-epilogue Linear, and the
/// vectorized elementwise-math kernel.
#[derive(Clone, Copy)]
struct GemmSyms {
    mm: Symbol,
    mm_par: Symbol,
    nt: Symbol,
    nt_par: Symbol,
    /// The fused-epilogue `nn.Linear` kernel (`mercury_sgemm_nt_epi`): `C = act(A·Bᵀ + bias)`. A
    /// matmul immediately followed by a bias-add / ReLU loop over its output lowers to this.
    nt_epi: Symbol,
    /// The 256-bit AVX2 elementwise-math kernel (`mercury_vmath_f32(x, out, n, op)`): an
    /// `out[i] = f(x[i])` transcendental loop lowers to this (the width Cranelift can't emit).
    vmath: Symbol,
    /// The streaming affine+activation kernel (`mercury_velem_f32(x, y, out, n, a, b, c, op)`): a
    /// recognized `out[i] = act(a·x[i] (+ b·y[i]) + c)` map loop (saxpy / scale / residual-add /
    /// bias / ReLU / ReLU6) lowers to this — 256-bit AVX2 + non-temporal stores for a large output.
    velem: Symbol,
    /// The multicore deterministic f32 reduction kernel (`mercury_sreduce_f32_parallel(x, y, n, op)
    /// -> f32`): a reduction loop in a `@parallel` function lowers to this. It is bit-equal to the
    /// serial `mercury_sreduce_f32` the interpreter calls, so native and interp stay bit-exact.
    sred_par: Symbol,
    /// The fused single-pass row-wise normalization kernel (`mercury_norm_f32(x, out, rows, cols,
    /// eps_bits, op)`): an idiomatic multi-pass softmax / LayerNorm / RMSNorm written in plain loops
    /// lowers to this one call. The interpreter marshals through the identical kernel.
    norm: Symbol,
    /// The affine fused norm kernel (`mercury_norm_affine_f32(x, out, gamma, beta, rows, cols,
    /// eps_bits, op)`): a LayerNorm/RMSNorm whose normalize step also applies a per-column scale
    /// `gamma` (and, for LayerNorm, a shift `beta`) lowers here instead — the real transformer form.
    norm_affine: Symbol,
    /// The int8 quantized `nn.Linear` kernel (`mercury_i8gemm_nt[_parallel](a, b, c, m, k, n)`): a
    /// `u8×i8→i32` `C = A·Bᵀ` nest lowers to this. Integer arithmetic, so the fused kernel equals the
    /// naive loop bit-for-bit (no reassociation exception).
    i8nt: Symbol,
    i8nt_par: Symbol,
}

// Elementwise-math op codes — must match `mercury_runtime::vmath`'s `VM_*` (mir_build does not depend
// on the runtime crate; same arrangement as the EPI_ACT_* codes mirroring the runtime's).
const VMATH_EXP: u32 = 0;
const VMATH_LOG: u32 = 1;
const VMATH_TANH: u32 = 2;
const VMATH_SIGMOID: u32 = 3;
const VMATH_SILU: u32 = 5;
const VMATH_GELU: u32 = 6;

// Streaming affine+activation op codes — must match `mercury_runtime::velem`'s `VE_*`. The low byte
// is the activation; `VE_USE_Y` (bit 8) flags that the kernel reads `y`.
const VE_ID: i64 = 0; // out = a·x (+ b·y) + c
const VE_RELU: i64 = 1; // out = max(.., 0)
const VE_RELU6: i64 = 2; // out = min(max(.., 0), 6)
const VE_USE_Y: i64 = 256;

/// One additive term of a recognized streaming affine body. `Scaled(arr, s)` is `arr[j]` (`s = None`,
/// coefficient 1) or `s·arr[j]` / `arr[j]·s` for a loop-invariant f32 scalar `s`; `Const(s)` is a
/// loop-invariant f32 scalar added in (the bias). Borrows the coefficient exprs from the body.
enum VTerm<'b> {
    Scaled(Symbol, Option<&'b Expr>),
    Const(&'b Expr),
}

/// A recognized `out[j] = act(a·x[j] (+ b·y[j]) + c)` streaming map: resolved array base pointers plus
/// the (loop-invariant) coefficient exprs, lowered to ValueIds at emit time so a runtime scale such as
/// saxpy's `a` works. `op` is the activation byte; `VE_USE_Y` is set iff `y` is present.
struct VElemPlan<'b> {
    out: ValueId,
    x: ValueId,
    y: Option<ValueId>,
    a: Option<&'b Expr>,
    b: Option<&'b Expr>,
    c: Option<&'b Expr>,
    op: i64,
}

// Reduction op codes — must match `mercury_runtime::reduce`'s `RED_*`. `x[k]*x[k]` recognizes as
// `RED_DOT` with both bases equal (≡ sum-of-squares), so the recognizer needs only these three.
const RED_DOT: i64 = 0; // sum(x[k] * y[k])
const RED_SSD: i64 = 1; // sum((x[k] - y[k])^2)
const RED_SUM: i64 = 2; // sum(x[k])

// Fused-normalization op codes — must match `mercury_runtime::norm`'s `NORM_*`.
const NORM_SOFTMAX: i64 = 0; // out = softmax(x) over the row
const NORM_LAYERNORM: i64 = 1; // out = (x - mean) / sqrt(var + eps)
const NORM_RMSNORM: i64 = 2; // out = x / sqrt(mean(x^2) + eps)

struct FnLowerer<'a> {
    builder: Builder,
    sema: &'a SemaResult,
    interner: &'a Interner,
    diags: &'a mut Vec<Diagnostic>,
    scopes: Vec<HashMap<Symbol, (ValueId, MirType)>>,
    terminated: bool,
    /// (continue target, break target) for the innermost loops.
    loops: Vec<(mercury_mir::BlockId, mercury_mir::BlockId)>,
    /// Pre-interned runtime symbols the matmul recognizer lowers a GEMM nest to.
    gemm: GemmSyms,
    /// True while lowering the body of a `@parallel` function: a recognized reduction loop dispatches
    /// to the multicore `mercury_sreduce_f32_parallel` instead of the sequential vectorizer.
    parallel_fn: bool,
    /// Within one vectorized loop-body copy, the vector already loaded for an index expression
    /// (keyed by its canonical text), so `x[i]` read twice (e.g. relu's `if x[i]>0 {x[i]}`) loads
    /// once. Cleared between unroll copies (addresses differ) and after any store (avoid staleness).
    vec_loads: HashMap<String, ValueId>,
}

impl FnLowerer<'_> {
    fn unsupported(&mut self, span: Span, what: &str) {
        // A construct codegen cannot lower yields incomplete/invalid MIR, so this is a hard error:
        // the compiler refuses to emit a broken program rather than silently producing one. The
        // front end (parsing, type and shape checking) still accepts these constructs — only
        // lowering to runnable code is unsupported so far.
        self.diags.push(
            Diagnostic::error(format!("`{what}` is not yet supported by codegen"))
                .with_code("C0001")
                .primary(span, "this construct cannot be lowered yet"),
        );
    }

    // ---- scopes ----

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: Symbol, slot: ValueId, ty: MirType) {
        self.scopes.last_mut().unwrap().insert(name, (slot, ty));
    }

    fn lookup(&self, name: Symbol) -> Option<(ValueId, MirType)> {
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(&name) {
                return Some(v.clone());
            }
        }
        None
    }

    // ---- type helpers ----

    fn expr_ty(&self, e: &Expr) -> Ty {
        self.sema.types.get(&e.id).cloned().unwrap_or(Ty::Unknown)
    }

    fn expr_mir(&self, e: &Expr) -> MirType {
        mir_ty(&self.expr_ty(e))
    }

    fn signed(&self, e: &Expr) -> bool {
        matches!(self.expr_ty(e), Ty::Scalar(s) if s.is_signed())
    }

    fn const_zero(&mut self, ty: MirType) -> ValueId {
        if ty.is_float() {
            self.builder.build(ty.clone(), Op::ConstFloat(0.0, ty))
        } else if ty.is_int() {
            self.builder.build(ty.clone(), Op::ConstInt(0, ty))
        } else {
            self.builder
                .build(MirType::I32, Op::ConstInt(0, MirType::I32))
        }
    }

    // ---- blocks & statements ----

    fn lower_block(&mut self, b: &Block) -> Option<ValueId> {
        self.push_scope();
        let mut i = 0;
        while i < b.stmts.len() {
            if self.terminated {
                break;
            }
            // Epilogue fusion: a `nn.Linear` matmul immediately followed by its bias-add / ReLU loop
            // folds into one GEMM call with the epilogue applied in the C writeback (no separate pass
            // over C). Checked before the elementwise fusion below — they match disjoint shapes.
            if let Some(n) = self.try_fuse_matmul_epilogue(&b.stmts[i..]) {
                i += n;
                continue;
            }
            // Fused normalization: an idiomatic multi-pass softmax / LayerNorm / RMSNorm over a flat
            // f32 array folds into one `mercury_norm_f32` call (each row loaded once + 256-bit AVX2).
            // Checked before the elementwise-fusion run so it sees the raw loop sequence, not a
            // pre-fused one. The three windows are structurally disjoint (max+exp vs mean+var+shift vs
            // sum-of-squares+scale), so probe order is immaterial.
            if let Some((n, arr, n_expr)) = self.match_softmax(b, i) {
                if self.emit_norm(arr, &n_expr, 0, NORM_SOFTMAX, None, None) {
                    i += n;
                    continue;
                }
            }
            if let Some((n, arr, n_expr, eps, gamma, beta)) = self.match_layernorm(b, i) {
                if self.emit_norm(arr, &n_expr, eps, NORM_LAYERNORM, gamma, beta) {
                    i += n;
                    continue;
                }
            }
            if let Some((n, arr, n_expr, eps, gamma, beta)) = self.match_rmsnorm(b, i) {
                if self.emit_norm(arr, &n_expr, eps, NORM_RMSNORM, gamma, beta) {
                    i += n;
                    continue;
                }
            }
            // Operator fusion: a run of adjacent same-range elementwise `for` loops whose *fused*
            // body the vectorizer accepts is lowered as one loop (CSE/DSE then forward any
            // intermediate array through registers, cutting its memory traffic).
            match self.try_fuse_run(&b.stmts[i..]) {
                Some(n) => i += n,
                None => {
                    self.lower_stmt(&b.stmts[i]);
                    i += 1;
                }
            }
        }
        let tail = match &b.tail {
            Some(e) if !self.terminated => Some(self.lower_expr(e)),
            _ => None,
        };
        self.pop_scope();
        tail
    }

    /// If `stmts` begins with two or more adjacent `for` loops over the *identical* range whose
    /// concatenated body the vectorizer accepts, lower them as a single fused loop and return how
    /// many statements were consumed. The vectorizer's "no written array touched at a second index"
    /// rule, applied to the fused body, is exactly the condition that makes fusion dependence-safe,
    /// so a successful check both authorizes and SIMD-accelerates the fusion. Returns `None`
    /// otherwise (and the caller lowers the first statement normally).
    fn try_fuse_run(&mut self, stmts: &[Stmt]) -> Option<usize> {
        let (pat0, iter0, _) = fusable_for(&stmts[0])?;
        let var = match &pat0.kind {
            ast::PatKind::Ident(s) => *s,
            _ => return None,
        };
        let (start0, end0) = range_bounds(iter0)?;
        // Maximal run of for-loops over the identical (var, start, end).
        let mut run = 1;
        while run < stmts.len() {
            match fusable_for(&stmts[run]) {
                Some((p, it, _)) => {
                    let same = matches!(&p.kind, ast::PatKind::Ident(s) if *s == var)
                        && range_bounds(it).is_some_and(|(s, e)| {
                            exprs_struct_eq(s, start0) && exprs_struct_eq(e, end0)
                        });
                    if same {
                        run += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
        if run < 2 {
            return None;
        }
        // Fuse the longest dependence-safe (vectorizable) prefix of length >= 2.
        for m in (2..=run).rev() {
            let fused = fuse_for_bodies(&stmts[..m]);
            if self.vectorizable(&fused, var).is_some() {
                self.lower_for(pat0, iter0, &fused);
                return Some(m);
            }
        }
        None
    }

    // ---- fused normalization recognition (block-level look-ahead, like the matmul epilogue) ----

    /// Match `for v in 0..N { body }` (exclusive, literal-0 start, no step), returning the loop
    /// variable, the upper-bound expression, and the body. Pure (no MIR).
    fn as_range0_for<'b>(&self, stmt: &'b Stmt) -> Option<(Symbol, &'b Expr, &'b Block)> {
        let StmtKind::For {
            pat, iter, body, ..
        } = &stmt.kind
        else {
            return None;
        };
        let ast::PatKind::Ident(v) = &pat.kind else {
            return None;
        };
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return None;
        };
        if const_usize_expr(start, self.interner) != Some(0) {
            return None;
        }
        Some((*v, end, body))
    }

    /// `let [mut] name [: ty] = init` → `(name, init)`. Pure.
    fn let_init(stmt: &Stmt) -> Option<(Symbol, &Expr)> {
        let StmtKind::Let {
            pat,
            init: Some(init),
            ..
        } = &stmt.kind
        else {
            return None;
        };
        let ast::PatKind::Ident(name) = &pat.kind else {
            return None;
        };
        Some((*name, init))
    }

    /// Body `m = fmax(m, x[v])` (running max into scalar `m`, indexing `x` by `v`) → return `x`. Pure.
    fn match_max_reduce_body(&self, body: &Block, v: Symbol, m: Symbol) -> Option<Symbol> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if single_path(target) != Some(m) {
            return None;
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 2
            || !matches!(
                self.vectorizable_intrinsic(callee),
                Some(MathIntrinsic::Fmax)
            )
        {
            return None;
        }
        // One arg must be `m`; the other must be `x[v]`.
        let xside = if single_path(&args[0]) == Some(m) {
            1
        } else if single_path(&args[1]) == Some(m) {
            0
        } else {
            return None;
        };
        self.index_by_loopvar(&args[xside], v)
    }

    /// Body `x[v] = exp(x[v] - m)` over f32, in place. Pure.
    fn match_exp_sub_body(&self, body: &Block, v: Symbol, x: Symbol, m: Symbol) -> Option<()> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if self.index_by_loopvar(target, v) != Some(x) || self.expr_mir(value) != MirType::F32 {
            return None;
        }
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 1
            || !matches!(
                self.vectorizable_intrinsic(callee),
                Some(MathIntrinsic::Exp)
            )
        {
            return None;
        }
        let ExprKind::Binary {
            op: ast::BinOp::Sub,
            lhs,
            rhs,
        } = &args[0].kind
        else {
            return None;
        };
        if self.index_by_loopvar(lhs, v) == Some(x) && single_path(rhs) == Some(m) {
            Some(())
        } else {
            None
        }
    }

    /// Body `s += x[v]` / `s = s + x[v]` (sum into scalar `s`), verifying the summed array is `x`. Pure.
    fn match_sum_body(&self, body: &Block, v: Symbol, x: Symbol, s: Symbol) -> Option<()> {
        if self.sum_body_array(body, v, s)? == x {
            Some(())
        } else {
            None
        }
    }

    /// Body `s += x[v]` / `s = s + x[v]` (sum into scalar `s`) → the summed array `x` (whichever it
    /// is). The array-discovering form of [`Self::match_sum_body`] (LayerNorm's leading sum loop is
    /// what first names the row array). Pure.
    fn sum_body_array(&self, body: &Block, v: Symbol, s: Symbol) -> Option<Symbol> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(s) {
            return None;
        }
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        self.index_by_loopvar(addend, v)
    }

    /// Body `s += x[v]*x[v]` (sum of squares into scalar `s`) → the squared array `x` (RMSNorm's lead
    /// loop, and the mean-square reduction). Pure.
    fn match_sumsq_body(&self, body: &Block, v: Symbol, s: Symbol) -> Option<Symbol> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(s) {
            return None;
        }
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        // addend must be `x[v] * x[v]` — same array, same index, both factors.
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &addend.kind
        else {
            return None;
        };
        let xl = self.index_by_loopvar(lhs, v)?;
        let xr = self.index_by_loopvar(rhs, v)?;
        if xl == xr {
            Some(xl)
        } else {
            None
        }
    }

    /// `arr[v]` where `arr != x` (the data) → `arr`: a per-column affine parameter array (gamma/beta)
    /// indexed by the normalize loop variable. Pure.
    fn affine_index(&self, e: &Expr, v: Symbol, x: Symbol) -> Option<Symbol> {
        match self.index_by_loopvar(e, v) {
            Some(a) if a != x => Some(a),
            _ => None,
        }
    }

    /// Peel an optional affine wrapper `core * gamma[v] (+ beta[v])` off a normalize-loop RHS, in the
    /// canonical `(... ) * gamma[i] + beta[i]` spelling (gamma on either side of its `*`, beta on
    /// either side of its `+`). Returns the remaining `core` expression plus the gamma/beta arrays
    /// (each `None` when absent). `gamma`/`beta` must be indexed by the loop var `v` and be arrays
    /// other than the data `x`. Pure — leaves `core` == `value` when there is no affine wrapper.
    fn peel_affine<'e>(
        &self,
        value: &'e Expr,
        v: Symbol,
        x: Symbol,
    ) -> (&'e Expr, Option<Symbol>, Option<Symbol>) {
        // outermost `+ beta[v]`
        let (after_beta, beta) = match &value.kind {
            ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } => {
                if let Some(b) = self.affine_index(rhs, v, x) {
                    (lhs.as_ref(), Some(b))
                } else if let Some(b) = self.affine_index(lhs, v, x) {
                    (rhs.as_ref(), Some(b))
                } else {
                    (value, None)
                }
            }
            _ => (value, None),
        };
        // then `* gamma[v]`
        let (core, gamma) = match &after_beta.kind {
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } => {
                if let Some(g) = self.affine_index(rhs, v, x) {
                    (lhs.as_ref(), Some(g))
                } else if let Some(g) = self.affine_index(lhs, v, x) {
                    (rhs.as_ref(), Some(g))
                } else {
                    (after_beta, None)
                }
            }
            _ => (after_beta, None),
        };
        (core, gamma, beta)
    }

    /// Body `x[v] = x[v] * inv [* gamma[v] [+ beta[v]]]` (scale by an invariant scalar, either operand
    /// order, with an optional affine wrapper for RMSNorm). Returns the captured `(gamma, beta)`
    /// arrays (both `None` for the plain form). Pure.
    fn match_scale_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        inv: Symbol,
    ) -> Option<(Option<Symbol>, Option<Symbol>)> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if self.index_by_loopvar(target, v) != Some(x) {
            return None;
        }
        let (core, gamma, beta) = self.peel_affine(value, v, x);
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &core.kind
        else {
            return None;
        };
        let ok = (self.index_by_loopvar(lhs, v) == Some(x) && single_path(rhs) == Some(inv))
            || (self.index_by_loopvar(rhs, v) == Some(x) && single_path(lhs) == Some(inv));
        if ok {
            Some((gamma, beta))
        } else {
            None
        }
    }

    /// Is `init` a sound running-max seed for softmax over `x` — `x[0]`, or a large-negative literal
    /// (`<= -1e30`)? Either is `<= max(x)`, so the user's `fmax(seed, …)` equals the true max the
    /// kernel computes; anything else might change behavior, so we decline. Pure.
    fn is_max_seed(&self, init: &Expr, x: Symbol) -> bool {
        if let ExprKind::Index { base, indices } = &init.kind {
            return single_path(base) == Some(x)
                && indices.len() == 1
                && matches!(&indices[0].kind, ExprKind::Int(t) if parse_int(self.interner.resolve(*t)) == 0);
        }
        let v = match &init.kind {
            ExprKind::Float(t) => parse_float(self.interner.resolve(*t)),
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => match &expr.kind {
                ExprKind::Float(t) => -parse_float(self.interner.resolve(*t)),
                _ => return false,
            },
            _ => return false,
        };
        v <= -1e30
    }

    /// `let name = 1.0 / s` → `name`. Pure.
    fn match_recip(&self, stmt: &Stmt, s: Symbol) -> Option<Symbol> {
        let (name, init) = Self::let_init(stmt)?;
        let ExprKind::Binary {
            op: ast::BinOp::Div,
            lhs,
            rhs,
        } = &init.kind
        else {
            return None;
        };
        let one = matches!(&lhs.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 1.0);
        if one && single_path(rhs) == Some(s) {
            Some(name)
        } else {
            None
        }
    }

    /// Recognize the canonical in-place flat softmax window (7 statements) at `b.stmts[at..]`:
    ///
    /// ```text
    /// let mut m = x[0];                       // seed: x[0] or a large-negative literal
    /// for i in 0..N { m = fmax(m, x[i]); }    // row max
    /// for i in 0..N { x[i] = exp(x[i] - m); } // exp(x - max), in place
    /// let mut s = 0.0;
    /// for i in 0..N { s += x[i]; }            // sum
    /// let inv = 1.0 / s;
    /// for i in 0..N { x[i] = x[i] * inv; }    // normalize
    /// ```
    ///
    /// Every loop must range over the identical `0..N` and index the *same* array `x` exactly by its
    /// loop variable; the scalars must chain (max → exp, sum → reciprocal, reciprocal → scale) and the
    /// internal scalars must not be read after the window (the kernel hides them). Pure: returns
    /// `(consumed, array, N)` with `N` cloned so the caller can emit freely. `None` on any deviation
    /// (the generic vectorizer + vmath path then lowers the loops correctly).
    fn match_softmax(&self, b: &Block, at: usize) -> Option<(usize, Symbol, Expr)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 7 {
            return None;
        }
        let (m, seed) = Self::let_init(&stmts[0])?;
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        let x = self.match_max_reduce_body(body1, v1, m)?;
        if !self.is_max_seed(seed, x) {
            return None;
        }
        let (v2, n2, body2) = self.as_range0_for(&stmts[2])?;
        if !exprs_struct_eq(n2, n_expr) {
            return None;
        }
        self.match_exp_sub_body(body2, v2, x, m)?;
        let (s, s_init) = Self::let_init(&stmts[3])?;
        if !matches!(&s_init.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0)
        {
            return None;
        }
        let (v4, n4, body4) = self.as_range0_for(&stmts[4])?;
        if !exprs_struct_eq(n4, n_expr) {
            return None;
        }
        self.match_sum_body(body4, v4, x, s)?;
        let inv = self.match_recip(&stmts[5], s)?;
        let (v6, n6, body6) = self.as_range0_for(&stmts[6])?;
        if !exprs_struct_eq(n6, n_expr) {
            return None;
        }
        // softmax's normalize is a plain `x[i] *= inv`; reject any affine wrapper (softmax has no
        // gamma/beta) so it falls back to the generic vectorizer rather than silently dropping it.
        if self.match_scale_body(body6, v6, x, inv)? != (None, None) {
            return None;
        }
        // The three internal scalars must not be read after the window — the kernel hides them.
        let rest = &b.stmts[at + 7..];
        let tail = b.tail.as_deref();
        for sc in [m, s, inv] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((7, x, n_expr.clone()))
    }

    /// Emit one in-place recognized norm: `mercury_norm_f32(x, x, 1, N, eps_bits, op)` for a plain
    /// (gamma=1, beta=0) norm, or `mercury_norm_affine_f32(x, x, gamma, beta, 1, N, eps_bits, op)` when
    /// the normalize step carried a per-column scale `gamma` (and optional shift `beta`) — the real
    /// transformer form. Bails (false) if the data array or a captured affine array is somehow unbound,
    /// so the caller lowers the loops normally.
    fn emit_norm(
        &mut self,
        arr: Symbol,
        n: &Expr,
        eps_bits: i64,
        op: i64,
        gamma: Option<Symbol>,
        beta: Option<Symbol>,
    ) -> bool {
        let Some((xv, _)) = self.lookup(arr) else {
            return false;
        };
        let n_ty = self.expr_mir(n);
        let nval = self.lower_expr(n);
        let nval = self.coerce_to(nval, &n_ty, &MirType::I64, true);
        let rows = self
            .builder
            .build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let epsv = self
            .builder
            .build(MirType::I64, Op::ConstInt(eps_bits as i128, MirType::I64));
        let opv = self
            .builder
            .build(MirType::I64, Op::ConstInt(op as i128, MirType::I64));
        if gamma.is_none() && beta.is_none() {
            self.builder.build_void(Op::Call {
                func: self.gemm.norm,
                args: vec![xv, xv, rows, nval, epsv, opv],
            });
            return true;
        }
        // Affine: resolve gamma/beta to their array base pointers. An absent param is a null pointer,
        // built as an integer `0` (a `Ptr`-typed `ConstInt` is invalid MIR; the verifier requires
        // integer-typed int consts). On the native side ptr_ty == i64, so the i64 zero is passed as the
        // null pointer the kernel checks; the interpreter sees a `Value::Int(0)` (distinct from any
        // real array's `Value::Ptr` by variant) and marshals it as absent → scale-1 / shift-0.
        let ptr_or_null = |me: &mut Self, sym: Option<Symbol>| -> Option<ValueId> {
            match sym {
                Some(s) => me.lookup(s).map(|(v, _)| v),
                None => Some(
                    me.builder
                        .build(MirType::I64, Op::ConstInt(0, MirType::I64)),
                ),
            }
        };
        let (Some(gptr), Some(bptr)) = (ptr_or_null(self, gamma), ptr_or_null(self, beta)) else {
            return false;
        };
        self.builder.build_void(Op::Call {
            func: self.gemm.norm_affine,
            args: vec![xv, xv, gptr, bptr, rows, nval, epsv, opv],
        });
        true
    }

    // ---- LayerNorm / RMSNorm recognition (the per-token transformer normalizations) ----

    /// Is `e` a float literal (or its negation)? → its `f32` bit pattern (the kernel's `eps` ABI). Pure.
    fn float_lit_bits(&self, e: &Expr) -> Option<i64> {
        let v: f64 = match &e.kind {
            ExprKind::Float(t) => parse_float(self.interner.resolve(*t)),
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => match &expr.kind {
                ExprKind::Float(t) => -parse_float(self.interner.resolve(*t)),
                _ => return None,
            },
            _ => return None,
        };
        Some((v as f32).to_bits() as i64)
    }

    /// Does `d` denote the row length `n` as an `f32` divisor — a float literal equal to a literal
    /// count, an `(n as f32)` cast of the exact bound, or the bound expression itself? (The mean /
    /// mean-square divides the row sum by the element count; this pins that divisor to the loop trip
    /// count, so we only fuse a genuine per-row mean.) Pure.
    fn count_as_f32(&self, d: &Expr, n: &Expr) -> bool {
        if let (ExprKind::Float(f), ExprKind::Int(k)) = (&d.kind, &n.kind) {
            return parse_float(self.interner.resolve(*f))
                == parse_int(self.interner.resolve(*k)) as f64;
        }
        if let ExprKind::Cast { expr, .. } = &d.kind {
            return self.expr_mir(d) == MirType::F32 && exprs_struct_eq(expr, n);
        }
        exprs_struct_eq(d, n)
    }

    /// Is `f` the float literal `1/count` (the multiply-by-reciprocal mean, e.g. `* 0.0625` for 16)?
    /// This is exactly what the kernel computes (`* invn`), so it is the *most* faithful spelling. Pure.
    fn recip_of_count(&self, f: &Expr, n: &Expr) -> bool {
        if let (ExprKind::Float(t), ExprKind::Int(k)) = (&f.kind, &n.kind) {
            let cnt = parse_int(self.interner.resolve(*k)) as f64;
            return cnt != 0.0 && parse_float(self.interner.resolve(*t)) == 1.0 / cnt;
        }
        false
    }

    /// Does `e` scale the row sum `s` by `1/count` — `s / count` or `s * (1/count)` (either factor
    /// order)? Both the mean and the mean-square divide a row sum by the element count; this accepts
    /// the divide and the (kernel-faithful) multiply-by-reciprocal spellings. Pure.
    fn scaled_by_inv_count(&self, e: &Expr, s: Symbol, n: &Expr) -> bool {
        match &e.kind {
            ExprKind::Binary {
                op: ast::BinOp::Div,
                lhs,
                rhs,
            } => single_path(lhs) == Some(s) && self.count_as_f32(rhs, n),
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } => {
                (single_path(lhs) == Some(s) && self.recip_of_count(rhs, n))
                    || (single_path(rhs) == Some(s) && self.recip_of_count(lhs, n))
            }
            _ => false,
        }
    }

    /// `let name = s / count` or `s * (1/count)` (row sum → mean), the scale pinned to the trip count
    /// `n`. Pure.
    fn match_mean(&self, stmt: &Stmt, s: Symbol, n: &Expr) -> Option<Symbol> {
        let (name, init) = Self::let_init(stmt)?;
        if self.scaled_by_inv_count(init, s, n) {
            Some(name)
        } else {
            None
        }
    }

    /// Is `e` the centered value `x[v] - mean`? Pure.
    fn is_centered(&self, e: &Expr, v: Symbol, x: Symbol, mean: Symbol) -> bool {
        matches!(
            &e.kind,
            ExprKind::Binary { op: ast::BinOp::Sub, lhs, rhs }
                if self.index_by_loopvar(lhs, v) == Some(x) && single_path(rhs) == Some(mean)
        )
    }

    /// Body `acc += (x[v]-mean)*(x[v]-mean)` (sum of squared deviations). Pure.
    fn match_var_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        mean: Symbol,
        acc: Symbol,
    ) -> Option<()> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign { target, op, value } = &stmt.kind else {
            return None;
        };
        if single_path(target) != Some(acc) {
            return None;
        }
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(acc) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(acc) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &addend.kind
        else {
            return None;
        };
        if self.is_centered(lhs, v, x, mean) && self.is_centered(rhs, v, x, mean) {
            Some(())
        } else {
            None
        }
    }

    /// Body `x[v] = (x[v]-mean) * inv` (center then scale, in place; either operand order). Pure.
    fn match_shift_scale_body(
        &self,
        body: &Block,
        v: Symbol,
        x: Symbol,
        mean: Symbol,
        inv: Symbol,
    ) -> Option<(Option<Symbol>, Option<Symbol>)> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if self.index_by_loopvar(target, v) != Some(x) {
            return None;
        }
        let (core, gamma, beta) = self.peel_affine(value, v, x);
        let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &core.kind
        else {
            return None;
        };
        let ok = (self.is_centered(lhs, v, x, mean) && single_path(rhs) == Some(inv))
            || (self.is_centered(rhs, v, x, mean) && single_path(lhs) == Some(inv));
        if ok {
            Some((gamma, beta))
        } else {
            None
        }
    }

    /// The argument `X` of `1.0 / sqrt(X)` or `rsqrt(X)`. Pure.
    fn as_rsqrt_arg<'b>(&self, e: &'b Expr) -> Option<&'b Expr> {
        match &e.kind {
            ExprKind::Binary {
                op: ast::BinOp::Div,
                lhs,
                rhs,
            } if matches!(&lhs.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 1.0) =>
            {
                let ExprKind::Call { callee, args, .. } = &rhs.kind else {
                    return None;
                };
                if args.len() == 1
                    && matches!(
                        self.vectorizable_intrinsic(callee),
                        Some(MathIntrinsic::Sqrt)
                    )
                {
                    Some(&args[0])
                } else {
                    None
                }
            }
            ExprKind::Call { callee, args, .. }
                if args.len() == 1
                    && matches!(
                        self.vectorizable_intrinsic(callee),
                        Some(MathIntrinsic::Rsqrt)
                    ) =>
            {
                Some(&args[0])
            }
            _ => None,
        }
    }

    /// `let inv = 1.0 / sqrt(sum/count + eps)` (or `rsqrt(...)`, `eps` either side) → `(inv, eps_bits)`,
    /// the reciprocal-standard-deviation binding shared by LayerNorm (variance sum) and RMSNorm
    /// (mean-square sum). Pure.
    fn match_inv_rstd(&self, stmt: &Stmt, sum: Symbol, n: &Expr) -> Option<(Symbol, i64)> {
        let (name, init) = Self::let_init(stmt)?;
        let arg = self.as_rsqrt_arg(init)?;
        let ExprKind::Binary {
            op: ast::BinOp::Add,
            lhs,
            rhs,
        } = &arg.kind
        else {
            return None;
        };
        // One operand is the eps literal; the other is `sum / count`.
        let (ms, eps_bits) = if let Some(b) = self.float_lit_bits(rhs) {
            (lhs.as_ref(), b)
        } else if let Some(b) = self.float_lit_bits(lhs) {
            (rhs.as_ref(), b)
        } else {
            return None;
        };
        if self.scaled_by_inv_count(ms, sum, n) {
            Some((name, eps_bits))
        } else {
            None
        }
    }

    /// Recognize the canonical in-place flat LayerNorm window (7 statements) at `b.stmts[at..]`:
    ///
    /// ```text
    /// let mut s = 0.0;
    /// for i in 0..N { s += x[i]; }                          // row sum
    /// let mean = s / (N as f32);                            // mean
    /// let mut v = 0.0;
    /// for i in 0..N { v += (x[i]-mean)*(x[i]-mean); }       // variance sum
    /// let inv = 1.0 / sqrt(v / (N as f32) + eps);           // 1/sqrt(var+eps)
    /// for i in 0..N { x[i] = (x[i]-mean) * inv * g[i] + b[i]; }  // normalize (affine g/b optional)
    /// ```
    ///
    /// Same discipline as [`Self::match_softmax`]: every loop ranges the identical `0..N` over the same
    /// array `x`; the divisors are pinned to the trip count; the scalars chain; and the internal
    /// scalars must not be read after the window (the kernel hides them). The normalize step may carry
    /// an optional per-column affine `* gamma[i] (+ beta[i])` (the real transformer form) — those
    /// arrays are captured and returned. Returns `(consumed, array, N, eps_bits, gamma, beta)`; `None`
    /// on any deviation (the generic vectorizer then lowers the loops).
    fn match_layernorm(
        &self,
        b: &Block,
        at: usize,
    ) -> Option<(usize, Symbol, Expr, i64, Option<Symbol>, Option<Symbol>)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 7 {
            return None;
        }
        let (s, s_init) = Self::let_init(&stmts[0])?;
        if !self.is_zero_lit(s_init) {
            return None;
        }
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        let x = self.sum_body_array(body1, v1, s)?;
        let mean = self.match_mean(&stmts[2], s, n_expr)?;
        let (vv, vv_init) = Self::let_init(&stmts[3])?;
        if !self.is_zero_lit(vv_init) {
            return None;
        }
        let (v4, n4, body4) = self.as_range0_for(&stmts[4])?;
        if !exprs_struct_eq(n4, n_expr) {
            return None;
        }
        self.match_var_body(body4, v4, x, mean, vv)?;
        let (inv, eps_bits) = self.match_inv_rstd(&stmts[5], vv, n_expr)?;
        let (v6, n6, body6) = self.as_range0_for(&stmts[6])?;
        if !exprs_struct_eq(n6, n_expr) {
            return None;
        }
        let (gamma, beta) = self.match_shift_scale_body(body6, v6, x, mean, inv)?;
        let rest = &b.stmts[at + 7..];
        let tail = b.tail.as_deref();
        for sc in [s, mean, vv, inv] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((7, x, n_expr.clone(), eps_bits, gamma, beta))
    }

    /// Recognize the canonical in-place flat RMSNorm window (4 statements) at `b.stmts[at..]`:
    ///
    /// ```text
    /// let mut s = 0.0;
    /// for i in 0..N { s += x[i]*x[i]; }              // mean-square sum
    /// let inv = 1.0 / sqrt(s / (N as f32) + eps);    // 1/sqrt(ms+eps)
    /// for i in 0..N { x[i] = x[i] * inv * g[i]; }     // normalize (affine scale g[i] optional)
    /// ```
    ///
    /// The normalize step may carry an optional per-column scale `* gamma[i]` (the real transformer
    /// form; RMSNorm has no shift, but a `+ beta[i]` is also accepted). Returns `(consumed, array, N,
    /// eps_bits, gamma, beta)`; `None` on any deviation.
    fn match_rmsnorm(
        &self,
        b: &Block,
        at: usize,
    ) -> Option<(usize, Symbol, Expr, i64, Option<Symbol>, Option<Symbol>)> {
        let stmts = &b.stmts[at..];
        if stmts.len() < 4 {
            return None;
        }
        let (s, s_init) = Self::let_init(&stmts[0])?;
        if !self.is_zero_lit(s_init) {
            return None;
        }
        let (v1, n_expr, body1) = self.as_range0_for(&stmts[1])?;
        let x = self.match_sumsq_body(body1, v1, s)?;
        let (inv, eps_bits) = self.match_inv_rstd(&stmts[2], s, n_expr)?;
        let (v3, n3, body3) = self.as_range0_for(&stmts[3])?;
        if !exprs_struct_eq(n3, n_expr) {
            return None;
        }
        let (gamma, beta) = self.match_scale_body(body3, v3, x, inv)?;
        let rest = &b.stmts[at + 4..];
        let tail = b.tail.as_deref();
        for sc in [s, inv] {
            if block_mentions(rest, tail, sc) {
                return None;
            }
        }
        Some((4, x, n_expr.clone(), eps_bits, gamma, beta))
    }

    /// Is `e` the float literal `0.0`? Pure.
    fn is_zero_lit(&self, e: &Expr) -> bool {
        matches!(&e.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) == 0.0)
    }

    fn lower_stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Let { pat, ty, init, .. } => {
                let mty = match ty {
                    Some(t) => mir_ty_of_ast(t, self.interner),
                    None => init
                        .as_ref()
                        .map(|e| self.expr_mir(e))
                        .unwrap_or(MirType::I32),
                };
                let slot = self.builder.alloca(mty.clone());
                if let Some(e) = init {
                    if let MirType::Array(elem, n) = &mty {
                        self.lower_array_init(slot, elem, *n, e);
                    } else {
                        let v = self.lower_expr(e);
                        self.builder.build_void(Op::Store {
                            ptr: slot,
                            value: v,
                        });
                    }
                }
                if let Pattern {
                    kind: ast::PatKind::Ident(name),
                    ..
                } = pat
                {
                    self.bind(*name, slot, mty);
                }
            }
            StmtKind::Assign { target, op, value } => {
                let rhs0 = self.lower_expr(value);
                let rhs_ty = self.expr_mir(value);
                let (ptr, elem) = self.lower_place(target);
                // Coerce the value to the place's type before storing, so a narrowing store (e.g.
                // an `f32` value into a `[bf16; N]` slot) carries the element type — the backends
                // then store the right width and round to bf16. A no-op when the types match.
                let rhs = self.coerce_to(rhs0, &rhs_ty, &elem, self.signed(value));
                let store_val = match op {
                    ast::AssignOp::Assign => rhs,
                    _ => {
                        let cur = self
                            .builder
                            .build(elem.clone(), Op::Load(ptr, elem.clone()));
                        let bin = compound_binop(*op, elem.is_float(), self.signed(target));
                        self.builder.build(elem.clone(), Op::Bin(bin, cur, rhs))
                    }
                };
                self.builder.build_void(Op::Store {
                    ptr,
                    value: store_val,
                });
            }
            StmtKind::Expr(e) => {
                self.lower_expr_stmt(e);
            }
            StmtKind::Return(opt) => {
                let v = opt.as_ref().map(|e| self.lower_expr(e));
                self.builder.ret(v);
                self.terminated = true;
            }
            StmtKind::While { cond, body, .. } => self.lower_while(cond, body),
            StmtKind::For {
                pat, iter, body, ..
            } => self.lower_for(pat, iter, body),
            StmtKind::Loop { body, .. } => self.lower_loop(body),
            StmtKind::Break(_) => {
                if let Some((_, brk)) = self.loops.last().copied() {
                    self.builder.br(brk, vec![]);
                }
                self.terminated = true;
            }
            StmtKind::Continue(_) => {
                if let Some((cont, _)) = self.loops.last().copied() {
                    self.builder.br(cont, vec![]);
                }
                self.terminated = true;
            }
            StmtKind::Defer(e) => {
                // Defer semantics (run at scope exit) are not modeled yet; lower for effects.
                self.unsupported(e.span, "defer");
                self.lower_expr_stmt(e);
            }
        }
    }

    /// Lower an expression used in statement position (control-flow expressions handled here).
    fn lower_expr_stmt(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.lower_if(cond, then_branch, else_branch.as_deref());
            }
            ExprKind::Block(b) => {
                self.lower_block(b);
            }
            _ => {
                self.lower_expr(e);
            }
        }
    }

    fn lower_while(&mut self, cond: &Expr, body: &Block) {
        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        self.terminated = false;
        let c = self.lower_expr(cond);
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        self.builder.switch_to(body_bb);
        self.terminated = false;
        self.loops.push((header, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(header, vec![]);
        }

        self.builder.switch_to(exit);
        self.terminated = false;
    }

    fn lower_loop(&mut self, body: &Block) {
        let header = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);
        self.builder.switch_to(header);
        self.terminated = false;
        self.loops.push((header, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(header, vec![]);
        }
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Emit a call to the GEMM microkernel for a recognized nest. Returns `false` (and emits
    /// nothing) if any operand array is not a pointer in scope, so the caller lowers it normally.
    fn emit_sgemm(&mut self, nest: &MatmulNest<'_>, parallel: bool) -> bool {
        let (Some((a, _)), Some((b, _)), Some((c, _))) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(nest.c),
        ) else {
            return false;
        };
        // Apply any per-operand base offset (the batch/head index of a batched matmul) as a pointer
        // GEP; a plain 2-D matmul has empty offsets and passes the array base straight through. The
        // inner matmul is identical under a constant base shift, so both backends stay bit-exact.
        let a = self.offset_base(a, &nest.a_off);
        let b = self.offset_base(b, &nest.b_off);
        let c = self.offset_base(c, &nest.c_off);
        // Materialize each dimension as an i64 value — a constant for a literal dim, or a load of
        // the runtime dimension variable (a function param/local). Bails to the scalar nest if a
        // variable dim is somehow out of scope at the call site.
        let (Some(m), Some(k), Some(n)) = (
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        let beta = self
            .builder
            .build(MirType::I64, Op::ConstInt(nest.beta as i128, MirType::I64));
        let func = match (parallel, nest.transposed) {
            (false, false) => self.gemm.mm,
            (true, false) => self.gemm.mm_par,
            (false, true) => self.gemm.nt,
            (true, true) => self.gemm.nt_par,
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, c, m, k, n, beta],
        });
        true
    }

    /// Emit one `mercury_i8gemm_nt[_parallel](a, b, c, m, k, n)` call for a recognized int8 quantized
    /// `nn.Linear` (`C = A·Bᵀ`, `u8`×`i8`→`i32`). Bails (false) if an operand/dim is unbound at the
    /// call site, so the caller lowers the scalar nest. `parallel` selects the multicore kernel (rows
    /// are independent, so it is bit-identical to the serial one the interpreter runs).
    fn emit_i8gemm(&mut self, nest: &I8MatmulNest, parallel: bool) -> bool {
        let (Some((a, _)), Some((b, _)), Some((c, _))) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(nest.c),
        ) else {
            return false;
        };
        let (Some(m), Some(k), Some(n)) = (
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        ) else {
            return false;
        };
        let func = if parallel {
            self.gemm.i8nt_par
        } else {
            self.gemm.i8nt
        };
        self.builder.build_void(Op::Call {
            func,
            args: vec![a, b, c, m, k, n],
        });
        true
    }

    /// Fuse a `nn.Linear` matmul immediately followed by its bias-add / activation epilogue into one
    /// `mercury_sgemm_nt_epi` call (`C = act(A·Bᵀ + bias)`), folding the epilogue into the GEMM's C
    /// writeback so C is written once instead of paying a separate read-modify-write pass. Fires only
    /// for the plain 2-D `C = A·Bᵀ` form (no batch offsets) immediately followed by the recognized
    /// epilogue loop over the same C; returns the number of statements consumed (always 2), else
    /// `None`. The match is strict (dims/strides/output/column all verified) so it never misfires.
    fn try_fuse_matmul_epilogue(&mut self, stmts: &[Stmt]) -> Option<usize> {
        if stmts.len() < 2 {
            return None;
        }
        let StmtKind::For {
            pat, iter, body, ..
        } = &stmts[0].kind
        else {
            return None;
        };
        let nest = recognize_matmul(pat, iter, body, self.sema, self.interner)?;
        // Only the plain 2-D nn.Linear form `C = A·Bᵀ` — no batch/head offsets (the epilogue kernel
        // is serial nt-only).
        if !nest.transposed
            || !nest.a_off.is_empty()
            || !nest.b_off.is_empty()
            || !nest.c_off.is_empty()
        {
            return None;
        }
        let (bias, act) = match_bias_act_epilogue(&stmts[1], &nest, self.interner)?;
        if self.emit_sgemm_epi(&nest, bias, act) {
            Some(2)
        } else {
            None
        }
    }

    /// Emit the fused `mercury_sgemm_nt_epi(a, b, c, m, k, n, beta, bias, act)` call for a recognized
    /// Linear+epilogue. Bails (false) if any operand/dim is somehow unbound at the call site, so the
    /// caller falls back to lowering the matmul and the epilogue loop separately.
    fn emit_sgemm_epi(&mut self, nest: &MatmulNest<'_>, bias: Symbol, act: u32) -> bool {
        let (
            Some((a, _)),
            Some((b, _)),
            Some((c, _)),
            Some((bias_ptr, _)),
            Some(m),
            Some(k),
            Some(n),
        ) = (
            self.lookup(nest.a),
            self.lookup(nest.b),
            self.lookup(nest.c),
            self.lookup(bias),
            self.dim_value(nest.m),
            self.dim_value(nest.k),
            self.dim_value(nest.n),
        )
        else {
            return false;
        };
        let beta = self
            .builder
            .build(MirType::I64, Op::ConstInt(nest.beta as i128, MirType::I64));
        let act_v = self
            .builder
            .build(MirType::I64, Op::ConstInt(act as i128, MirType::I64));
        self.builder.build_void(Op::Call {
            func: self.gemm.nt_epi,
            args: vec![a, b, c, m, k, n, beta, bias_ptr, act_v],
        });
        true
    }

    /// GEP `base` by the sum of the `offset` terms (element indices) — the batch/head base shift of
    /// a batched matmul. Returns `base` unchanged when there is no offset (a plain 2-D matmul). Each
    /// term is lowered in the current scope (the enclosing batch-loop variable and the dimensions are
    /// all live) and coerced to `i64` before summing.
    fn offset_base(&mut self, base: ValueId, offset: &[&Expr]) -> ValueId {
        if offset.is_empty() {
            return base;
        }
        let mut acc: Option<ValueId> = None;
        for &t in offset {
            let ty = self.expr_mir(t);
            let v = self.lower_expr(t);
            let v = self.coerce_to(v, &ty, &MirType::I64, true);
            acc = Some(match acc {
                None => v,
                Some(prev) => self
                    .builder
                    .build(MirType::I64, Op::Bin(BinOp::Add, prev, v)),
            });
        }
        let idx = acc.unwrap();
        self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: MirType::F32,
            },
        )
    }

    /// Materialize a matmul dimension as an `i64` MIR value: a constant for a literal, or a load
    /// (coerced to i64) of the runtime dimension variable.
    fn dim_value(&mut self, dim: Dim) -> Option<ValueId> {
        match dim {
            Dim::Lit(v) => Some(
                self.builder
                    .build(MirType::I64, Op::ConstInt(v as i128, MirType::I64)),
            ),
            Dim::Var(sym) => {
                let (slot, ty) = self.lookup(sym)?;
                let v = self.builder.build(ty.clone(), Op::Load(slot, ty.clone()));
                Some(self.coerce_to(v, &ty, &MirType::I64, true))
            }
        }
    }

    /// Recognize a kernel-dispatchable reduction body `s = s + f(x[k], y[k])` (or `s += …`) over the
    /// loop variable `k`: dot `x[k]*y[k]`, ssd `(x[k]-y[k])*(x[k]-y[k])`, or sum `x[k]`. Returns the
    /// accumulator symbol, the `RED_*` op code, and the two array bases (`y == x` for the unary sum).
    /// Strict pure-AST match — single statement, index exactly `k`.
    fn match_reduction_kernel(
        &self,
        body: &Block,
        k: Symbol,
    ) -> Option<(Symbol, i64, Symbol, Symbol)> {
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        let StmtKind::Assign { target, op, value } = &body.stmts[0].kind else {
            return None;
        };
        let s = single_path(target)?;
        if s == k {
            return None;
        }
        // `s += addend`, or `s = s + addend` / `s = addend + s`.
        let addend: &Expr = match op {
            ast::AssignOp::Add => value,
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(lhs) == Some(s) => rhs,
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if single_path(rhs) == Some(s) => lhs,
                _ => return None,
            },
            _ => return None,
        };
        if expr_uses_sym(addend, s) {
            return None;
        }
        // `base[k]` with the index exactly the loop variable → the base array symbol.
        let idx_base = |e: &Expr| -> Option<Symbol> {
            let ExprKind::Index { base, indices } = &e.kind else {
                return None;
            };
            if indices.len() != 1 || single_path(&indices[0]) != Some(k) {
                return None;
            }
            single_path(base)
        };
        match &addend.kind {
            // dot `a[k]*b[k]` (a==b ≡ sum-of-squares), or ssd `(a[k]-b[k])*(a[k]-b[k])`.
            ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } => {
                if let (
                    ExprKind::Binary {
                        op: ast::BinOp::Sub,
                        lhs: l1,
                        rhs: r1,
                    },
                    ExprKind::Binary {
                        op: ast::BinOp::Sub,
                        lhs: l2,
                        rhs: r2,
                    },
                ) = (&lhs.kind, &rhs.kind)
                {
                    let (a1, b1, a2, b2) =
                        (idx_base(l1)?, idx_base(r1)?, idx_base(l2)?, idx_base(r2)?);
                    return if a1 == a2 && b1 == b2 {
                        Some((s, RED_SSD, a1, b1))
                    } else {
                        None
                    };
                }
                Some((s, RED_DOT, idx_base(lhs)?, idx_base(rhs)?))
            }
            // sum `a[k]`.
            ExprKind::Index { .. } => {
                let a = idx_base(addend)?;
                Some((s, RED_SUM, a, a))
            }
            _ => None,
        }
    }

    /// Lower a recognized `@parallel` reduction `for k in 0..n { s += f(x[k], y[k]) }` to one
    /// `s = s + mercury_sreduce_f32_parallel(x, y, n, op)`. The kernel returns the same value the loop
    /// would (a reassociation of the same terms), and the interpreter calls the identical *serial*
    /// kernel — which is bit-equal to the parallel one — so native and interp agree. Returns false
    /// (fall back to the scalar/vector loop) unless the range is `0..n`, the accumulator is an
    /// in-scope f32 scalar, and both arrays are in scope.
    fn try_emit_parallel_reduction(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) -> bool {
        let ForIter::Range {
            start,
            end: Some(end),
            inclusive: false,
            step: None,
        } = iter
        else {
            return false;
        };
        // Only `0..n`; a non-zero start would need a pointer/length shift the call does not do.
        if const_usize_expr(start, self.interner) != Some(0) {
            return false;
        }
        let Pattern {
            kind: ast::PatKind::Ident(k),
            ..
        } = pat
        else {
            return false;
        };
        let Some((s, op, xb, yb)) = self.match_reduction_kernel(body, *k) else {
            return false;
        };
        // Accumulator must be an in-scope f32 scalar; both arrays must be in scope (base pointers).
        let Some((s_slot, MirType::F32)) = self.lookup(s) else {
            return false;
        };
        let (Some((xv, _)), Some((yv, _))) = (self.lookup(xb), self.lookup(yb)) else {
            return false;
        };
        let n_ty = self.expr_mir(end);
        let n = self.lower_expr(end);
        let n = self.coerce_to(n, &n_ty, &MirType::I64, true);
        let opv = self
            .builder
            .build(MirType::I64, Op::ConstInt(op as i128, MirType::I64));
        let result = self.builder.build(
            MirType::F32,
            Op::Call {
                func: self.gemm.sred_par,
                args: vec![xv, yv, n, opv],
            },
        );
        // s = s + result — matches the loop's `s_final = s_init + Σ` (reassociated inside the kernel).
        let cur = self
            .builder
            .build(MirType::F32, Op::Load(s_slot, MirType::F32));
        let new_s = self
            .builder
            .build(MirType::F32, Op::Bin(BinOp::FAdd, cur, result));
        self.builder.build_void(Op::Store {
            ptr: s_slot,
            value: new_s,
        });
        true
    }

    fn lower_for(&mut self, pat: &Pattern, iter: &ForIter, body: &Block) {
        // A matmul nest lowers to the tuned microkernel (single-threaded on this statement path; the
        // whole-function `@parallel` form is handled earlier in `lower_program`).
        if let Some(nest) = recognize_matmul(pat, iter, body, self.sema, self.interner) {
            if self.emit_sgemm(&nest, false) {
                return;
            }
        }
        // int8 quantized `nn.Linear` (`u8×i8→i32` `C = A·Bᵀ`) → the int8 GEMM microkernel. Integer
        // arithmetic, so the kernel equals the scalar nest bit-for-bit; in a `@parallel` function the
        // multicore kernel is used (rows independent → deterministic, so the gate stays exact).
        if let Some(nest) = match_matmul_i8_nt(pat, iter, body, self.sema, self.interner) {
            if self.emit_i8gemm(&nest, self.parallel_fn) {
                return;
            }
        }
        // Inside a `@parallel` function, a recognized reduction loop (`s += x[k]*y[k]`, etc.) lowers
        // to one multicore `mercury_sreduce_f32_parallel` call instead of the sequential vectorizer.
        if self.parallel_fn && self.try_emit_parallel_reduction(pat, iter, body) {
            return;
        }
        let (start, end, inclusive, step) = match iter {
            ForIter::Range {
                start,
                end: Some(end),
                inclusive,
                step,
            } => (start, end, *inclusive, step),
            _ => {
                self.unsupported(body.span, "for over a non-range iterator");
                return;
            }
        };

        let ity = self.expr_mir(start);
        let signed = self.signed(start);

        // Straight-line elementwise loops lower to SIMD (vector main loop + scalar remainder); this
        // is purely an optimization, so on any doubt it returns false and we lower scalar below.
        if !inclusive && step.is_none() && self.try_vectorize_for(pat, start, end, body) {
            return;
        }

        // i = start
        let slot = self.builder.alloca(ity.clone());
        let s0 = self.lower_expr(start);
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: s0,
        });

        self.push_scope();
        if let Pattern {
            kind: ast::PatKind::Ident(name),
            ..
        } = pat
        {
            self.bind(*name, slot, ity.clone());
        }

        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        // header: i < end (or <=)
        self.builder.switch_to(header);
        self.terminated = false;
        let i_val = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let end_val = self.lower_expr(end);
        let pred = match (inclusive, signed) {
            (false, true) => CmpOp::Slt,
            (true, true) => CmpOp::Sle,
            (false, false) => CmpOp::Ult,
            (true, false) => CmpOp::Ule,
        };
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(pred, i_val, end_val));
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        // body; i += step
        self.builder.switch_to(body_bb);
        self.terminated = false;
        self.loops.push((header, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            let cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
            let step_val = match step {
                Some(st) => self.lower_expr(st),
                None => self
                    .builder
                    .build(ity.clone(), Op::ConstInt(1, ity.clone())),
            };
            let next = self
                .builder
                .build(ity.clone(), Op::Bin(BinOp::Add, cur, step_val));
            self.builder.build_void(Op::Store {
                ptr: slot,
                value: next,
            });
            self.builder.br(header, vec![]);
        }

        self.pop_scope();
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    // ============================ SIMD loop vectorizer ============================
    //
    // Recognizes `for j in lo..hi { <straight-line elementwise body> }` and lowers it to a vector
    // main loop processing `W` lanes per iteration plus a scalar remainder for the tail. It fires
    // only when every array access is unit-stride in `j` (vector load/store) or invariant in `j`
    // (splat), no written array is touched at a second index (no loop-carried dependence), the body
    // is branch/call-free, and all lanes share one 32/64-bit scalar type. Lane `k` of the vector
    // loop then computes exactly what scalar iteration `base+k` would, so the result is identical.
    // Distinct array parameters are assumed not to alias (the usual kernel ABI). On any failure it
    // returns `false` and the caller falls back to the scalar lowering.

    /// If `e` is `arr[j]` — a single-segment array path indexed by exactly the loop variable `j`
    /// (unit stride, zero offset) — return the array's symbol. The shape the vmath kernel needs.
    fn index_by_loopvar(&self, e: &Expr, j: Symbol) -> Option<Symbol> {
        let ExprKind::Index { base, indices } = &e.kind else {
            return None;
        };
        if indices.len() != 1 || single_path(&indices[0]) != Some(j) {
            return None;
        }
        single_path(base)
    }

    /// Match one statement `out[j] = f(x[j])` for a supported unary intrinsic `f` (exp/log/tanh/
    /// sigmoid/silu/gelu) over `f32` arrays, returning `(out_array, x_array, op_code)`. Pure.
    fn match_vmath_stmt(&self, stmt: &Stmt, j: Symbol) -> Option<(Symbol, Symbol, u32)> {
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        let out_sym = self.index_by_loopvar(target, j)?;
        let ExprKind::Call { callee, args, .. } = &value.kind else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        let opcode = match self.vectorizable_intrinsic(callee) {
            Some(MathIntrinsic::Exp) => VMATH_EXP,
            Some(MathIntrinsic::Log) => VMATH_LOG,
            Some(MathIntrinsic::Tanh) => VMATH_TANH,
            Some(MathIntrinsic::Sigmoid) => VMATH_SIGMOID,
            Some(MathIntrinsic::Silu) => VMATH_SILU,
            Some(MathIntrinsic::Gelu) => VMATH_GELU,
            _ => return None,
        };
        let x_sym = self.index_by_loopvar(&args[0], j)?;
        // f32-only (the kernel computes f32); bail on f64 / unknown.
        if self.expr_mir(value) != MirType::F32 || self.expr_mir(&args[0]) != MirType::F32 {
            return None;
        }
        Some((out_sym, x_sym, opcode))
    }

    /// Match a transcendental-activation loop body: every statement must be an independent
    /// `out[j] = f(x[j])` over f32 (see [`match_vmath_stmt`]), with every array in scope. Returns the
    /// resolved `(out_base, x_base, op)` per statement, or `None` if any statement fails. Pure — emits
    /// no MIR — so it is safe to call before deciding whether to lower the loop bounds.
    fn match_vmath_body(&self, j: Symbol, body: &Block) -> Option<Vec<(ValueId, ValueId, u32)>> {
        if body.tail.is_some() || body.stmts.is_empty() {
            return None;
        }
        let mut calls = Vec::with_capacity(body.stmts.len());
        for stmt in &body.stmts {
            let (out_sym, x_sym, opcode) = self.match_vmath_stmt(stmt, j)?;
            let (out_base, _) = self.lookup(out_sym)?;
            let (x_base, _) = self.lookup(x_sym)?;
            calls.push((out_base, x_base, opcode));
        }
        Some(calls)
    }

    /// Emit one `mercury_vmath_f32(x+s, out+s, e-s, op)` call per resolved statement over the i64
    /// range `[s, e)`. Each is a full-range pass; in source order they preserve a fused multi-statement
    /// body's per-element semantics (an array is fully written before a later pass reads it).
    fn emit_vmath_calls(&mut self, s: ValueId, e: ValueId, calls: Vec<(ValueId, ValueId, u32)>) {
        let n = self.builder.build(MirType::I64, Op::Bin(BinOp::Sub, e, s));
        for (out_base, x_base, opcode) in calls {
            let xp = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: x_base,
                    index: s,
                    elem: MirType::F32,
                },
            );
            let outp = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: out_base,
                    index: s,
                    elem: MirType::F32,
                },
            );
            let opv = self
                .builder
                .build(MirType::I64, Op::ConstInt(opcode as i128, MirType::I64));
            self.builder.build_void(Op::Call {
                func: self.gemm.vmath,
                args: vec![xp, outp, n, opv],
            });
        }
    }

    /// Recognize an elementwise transcendental loop `for j in lo..hi { ... }` whose body is one or
    /// more independent `out[j] = f(x[j])` activations (exp/log/tanh/sigmoid/silu/gelu over f32), and
    /// lower it to one `mercury_vmath_f32` call per statement — the 256-bit AVX2 kernel (Cranelift's
    /// vectorizer is capped at 128-bit). `out` and `x` may be the same array (in-place). The
    /// interpreter marshals through the identical kernel, so the differential oracle stays exact.
    /// Returns false (fall back to the generic vectorizer) unless every statement matches.
    fn try_vmath_for(&mut self, j: Symbol, start: &Expr, end: &Expr, body: &Block) -> bool {
        let Some(calls) = self.match_vmath_body(j, body) else {
            return false;
        };
        // Bounds GEP each base by the start index, so a loop from `lo` begins at element `lo`.
        let sty = self.expr_mir(start);
        let s = self.lower_expr(start);
        let s = self.coerce_to(s, &sty, &MirType::I64, true);
        let ety = self.expr_mir(end);
        let e = self.lower_expr(end);
        let e = self.coerce_to(e, &ety, &MirType::I64, true);
        self.emit_vmath_calls(s, e, calls);
        true
    }

    /// A loop-invariant f32 coefficient: an expr provably free of the loop var `j` and typed f32 (a
    /// literal like `2.0`, or an outer scalar like saxpy's `a`). Lowered to a ValueId at emit time.
    /// Uses the **conservative** [`expr_mentions`] (recurses through calls/casts/fields and assumes a
    /// use for anything it cannot model), so a per-element factor like `sigmoid(x[j])` is correctly
    /// rejected rather than mistaken for an invariant scale. Pure.
    fn velem_coeff<'b>(&self, e: &'b Expr, j: Symbol) -> Option<&'b Expr> {
        if expr_mentions(e, j) || self.expr_mir(e) != MirType::F32 {
            return None;
        }
        Some(e)
    }

    /// Classify one additive term of a streaming affine body over loop var `j`: a (possibly scaled)
    /// unit-stride f32 array read `arr[j]`, or a loop-invariant f32 scalar (bias). Pure.
    fn velem_term<'b>(&self, e: &'b Expr, j: Symbol) -> Option<VTerm<'b>> {
        if let Some(arr) = self.index_by_loopvar(e, j) {
            if self.expr_mir(e) != MirType::F32 {
                return None;
            }
            return Some(VTerm::Scaled(arr, None));
        }
        if let ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } = &e.kind
        {
            if let Some(arr) = self.index_by_loopvar(lhs, j) {
                if self.expr_mir(lhs) == MirType::F32 {
                    if let Some(s) = self.velem_coeff(rhs, j) {
                        return Some(VTerm::Scaled(arr, Some(s)));
                    }
                }
            }
            if let Some(arr) = self.index_by_loopvar(rhs, j) {
                if self.expr_mir(rhs) == MirType::F32 {
                    if let Some(s) = self.velem_coeff(lhs, j) {
                        return Some(VTerm::Scaled(arr, Some(s)));
                    }
                }
            }
            return None;
        }
        self.velem_coeff(e, j).map(VTerm::Const)
    }

    /// Recognize a streaming affine+activation map body — one statement `out[j] = act(a·x[j] (+
    /// b·y[j]) + c)` whose RHS is a sum of (scaled) f32 array reads plus an optional invariant bias —
    /// covering saxpy / scale / residual-add / bias / copy. Up to two distinct array reads (`x`, then
    /// `y`) and one bias. Pure (emits no MIR): returns the resolved bases + coefficient exprs, or
    /// `None` to fall through to ReLU recognition / the generic vectorizer. The activation `act` is
    /// supplied by the caller (`VE_ID` for the bare arithmetic forms; ReLU/ReLU6 peel their wrapper
    /// first and pass the inner affine `value` here with `act = VE_RELU`/`VE_RELU6`).
    fn match_velem_affine<'b>(
        &self,
        j: Symbol,
        value: &'b Expr,
        target: &Expr,
        act: i64,
    ) -> Option<VElemPlan<'b>> {
        let out_sym = self.index_by_loopvar(target, j)?;
        if self.expr_mir(value) != MirType::F32 {
            return None;
        }
        let mut terms = Vec::new();
        flatten_add_terms(value, &mut terms);
        let mut arrays: Vec<(Symbol, Option<&Expr>)> = Vec::new();
        let mut consts: Vec<&Expr> = Vec::new();
        for t in &terms {
            match self.velem_term(t, j)? {
                VTerm::Scaled(arr, s) => arrays.push((arr, s)),
                VTerm::Const(s) => consts.push(s),
            }
        }
        // Must read at least one array (else it is not a map over x); at most two; at most one bias.
        if arrays.is_empty() || arrays.len() > 2 || consts.len() > 1 {
            return None;
        }
        let (x_sym, a) = arrays[0];
        let x = self.lookup(x_sym)?.0;
        let (y, b, op_y) = if arrays.len() == 2 {
            let (y_sym, b) = arrays[1];
            (Some(self.lookup(y_sym)?.0), b, VE_USE_Y)
        } else {
            (None, None, 0)
        };
        let out = self.lookup(out_sym)?.0;
        Some(VElemPlan {
            out,
            x,
            y,
            a,
            b,
            c: consts.first().copied(),
            op: act | op_y,
        })
    }

    /// Peel a ReLU / ReLU6 activation wrapper off `value`, returning the inner (affine) expr and the
    /// `VE_RELU`/`VE_RELU6` code. ReLU is the idiomatic `if INNER > 0.0 { INNER } else { 0.0 }`
    /// (`max(INNER, 0)`); ReLU6 is `if INNER < 6.0 { <ReLU of INNER> } else { 6.0 }` (`clamp(INNER,
    /// 0, 6)`), where the outer guard compares the *same* `INNER`. Pure.
    fn peel_velem_act<'b>(&self, value: &'b Expr) -> Option<(&'b Expr, i64)> {
        let ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } = &value.kind
        else {
            return None;
        };
        let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
            return None;
        };
        let else_e = branch_value(else_branch.as_deref()?)?;
        let then_e = block_value(then_branch)?;
        let is_lit = |e: &Expr, v: f32| {
            matches!(&e.kind, ExprKind::Float(t) if parse_float(self.interner.resolve(*t)) as f32 == v)
        };
        match op {
            // ReLU: `if INNER > 0 { INNER } else { 0 }` — the then-branch returns the compared INNER.
            ast::BinOp::Gt if is_lit(rhs, 0.0) && is_lit(else_e, 0.0) && exprs_struct_eq(then_e, lhs) => {
                Some((lhs, VE_RELU))
            }
            // ReLU6: `if INNER < 6 { ReLU(INNER) } else { 6 }` — the then-branch is a ReLU whose inner
            // is the same INNER the guard compares.
            ast::BinOp::Lt if is_lit(rhs, 6.0) && is_lit(else_e, 6.0) => {
                let (inner, inner_act) = self.peel_velem_act(then_e)?;
                if inner_act == VE_RELU && exprs_struct_eq(inner, lhs) {
                    Some((inner, VE_RELU6))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Recognize the whole body of a streaming elementwise map: a single `out[j] = …` assignment whose
    /// RHS is either a bare affine form (saxpy / scale / add / bias / copy) or a ReLU/ReLU6-wrapped
    /// one. Pure (emits no MIR).
    fn match_velem_body<'b>(&self, j: Symbol, body: &'b Block) -> Option<VElemPlan<'b>> {
        let stmt = single_stmt(body)?;
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if let Some(plan) = self.match_velem_affine(j, value, target, VE_ID) {
            return Some(plan);
        }
        let (inner, act) = self.peel_velem_act(value)?;
        self.match_velem_affine(j, inner, target, act)
    }

    /// Lower a coefficient expr (or a default constant when absent) to an f32 ValueId.
    fn lower_coeff(&mut self, e: Option<&Expr>, default: f64) -> ValueId {
        match e {
            Some(e) => {
                let ty = self.expr_mir(e);
                let v = self.lower_expr(e);
                self.coerce_to(v, &ty, &MirType::F32, true)
            }
            None => self
                .builder
                .build(MirType::F32, Op::ConstFloat(default, MirType::F32)),
        }
    }

    /// Emit one `mercury_velem_f32(x+s, y+s, out+s, e-s, a, b, c, op)` call for a recognized streaming
    /// map over the i64 range `[s, e)`. When `y` is absent the kernel never reads it (`VE_USE_Y`
    /// unset), so the `x` pointer is reused for the unused `y` argument (a valid, never-dereferenced
    /// pointer — no need for a Ptr-typed null const, which is invalid MIR).
    fn emit_velem_call(&mut self, s: ValueId, e: ValueId, plan: &VElemPlan) {
        let n = self.builder.build(MirType::I64, Op::Bin(BinOp::Sub, e, s));
        let gep = |me: &mut Self, base: ValueId| {
            me.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base,
                    index: s,
                    elem: MirType::F32,
                },
            )
        };
        let xp = gep(self, plan.x);
        let yp = match plan.y {
            Some(y) => gep(self, y),
            None => xp,
        };
        let outp = gep(self, plan.out);
        // Default `b` is 1.0 when `y` is read, else 0.0 (unused); `a` defaults to 1.0, `c` to 0.0.
        let a = self.lower_coeff(plan.a, 1.0);
        let b = self.lower_coeff(plan.b, if plan.y.is_some() { 1.0 } else { 0.0 });
        let c = self.lower_coeff(plan.c, 0.0);
        let opv = self
            .builder
            .build(MirType::I64, Op::ConstInt(plan.op as i128, MirType::I64));
        self.builder.build_void(Op::Call {
            func: self.gemm.velem,
            args: vec![xp, yp, outp, n, a, b, c, opv],
        });
    }

    /// Recognize a streaming elementwise map `for j in lo..hi { out[j] = act(a·x[j] (+ b·y[j]) + c) }`
    /// (saxpy / scale / residual-add / bias) and lower it to one `mercury_velem_f32` call — 256-bit
    /// AVX2 + non-temporal stores, the width Cranelift can't reach and a store path gcc won't emit.
    /// `out` may alias `x`/`y` (in-place). The interpreter marshals the identical kernel, so the
    /// differential oracle stays exact. Returns false (fall through) unless the body matches.
    fn try_velem_for(&mut self, j: Symbol, start: &Expr, end: &Expr, body: &Block) -> bool {
        let Some(plan) = self.match_velem_body(j, body) else {
            return false;
        };
        let sty = self.expr_mir(start);
        let s = self.lower_expr(start);
        let s = self.coerce_to(s, &sty, &MirType::I64, true);
        let ety = self.expr_mir(end);
        let e = self.lower_expr(end);
        let e = self.coerce_to(e, &ety, &MirType::I64, true);
        self.emit_velem_call(s, e, &plan);
        true
    }

    /// Attempt SIMD lowering of `for j in start..end { body }`. Returns true on success.
    fn try_vectorize_for(&mut self, pat: &Pattern, start: &Expr, end: &Expr, body: &Block) -> bool {
        let j = match &pat.kind {
            ast::PatKind::Ident(name) => *name,
            _ => return false,
        };
        let ity = self.expr_mir(start);
        if !ity.is_int() {
            return false;
        }
        // A pure elementwise transcendental `out[j] = f(x[j])` (exp/log/tanh/sigmoid) lowers to the
        // 256-bit AVX2 runtime kernel — the width Cranelift's 128-bit vectorizer can't reach. Tried
        // before the generic (Cranelift-emitted) vectorizer, which would otherwise inline the poly.
        if self.try_vmath_for(j, start, end, body) {
            return true;
        }
        // A streaming affine map `out[j] = a·x[j] (+ b·y[j]) + c` (saxpy/scale/add/bias) dispatches to
        // the 256-bit AVX2 + non-temporal-store kernel — both wider than and store-cheaper than the
        // generic 128-bit vectorizer. Tried before it (which would otherwise emit cacheable stores).
        if self.try_velem_for(j, start, end, body) {
            return true;
        }
        // Pure analyses first (emit no MIR). A reduction (`s += elementwise`) has a body shape
        // disjoint from the elementwise *store* loops, so try it first. Reductions are vectorized
        // only on this (sequential) path — not the `@parallel` per-thread path, where folding into
        // a shared accumulator across threads would race.
        let reduction = self.reduction_of(body, j);
        if reduction.is_none() && self.vectorizable(body, j).is_none() {
            return false;
        }
        // Lower the bounds (coerced to the index type) and hand off to the matching emitter.
        let start_ty = self.expr_mir(start);
        let s0 = self.lower_expr(start);
        let s0 = self.coerce_to(s0, &start_ty, &ity, true);
        let end_ty = self.expr_mir(end);
        let e0 = self.lower_expr(end);
        let e0 = self.coerce_to(e0, &end_ty, &ity, true);
        if let Some((s_sym, addend, lane, w, redop)) = reduction {
            self.emit_reduction(j, s0, e0, &ity, s_sym, addend, &lane, w, redop);
            true
        } else {
            self.try_vectorize_ranged(j, s0, e0, &ity, body)
        }
    }

    /// Recognise a vectorizable float reduction `for j in .. { s = s + <expr(j)> }` (or `s += ..`),
    /// where `s` is an outer float scalar that the body otherwise does not touch and `<expr>` is an
    /// elementwise value over `j` that does not read `s`. Returns `(s, addend, lane, width)`.
    /// Vectorizing this reassociates the float sum (lane accumulators + a horizontal reduce) — the
    /// standard reduction optimization. Pure analysis; emits no MIR.
    fn reduction_of<'b>(
        &self,
        body: &'b Block,
        j: Symbol,
    ) -> Option<(Symbol, &'b Expr, MirType, u32, RedOp)> {
        if body.tail.is_some() || body.stmts.len() != 1 {
            return None;
        }
        let StmtKind::Assign { target, op, value } = &body.stmts[0].kind else {
            return None;
        };
        let s = single_path(target)?;
        if s == j {
            return None;
        }
        let (_, sty) = self.lookup(s)?;
        if !(sty.is_float() || sty.is_int()) {
            return None;
        }
        // The addend and fold kind: `s += addend` / `s = s + addend` / `s = addend + s` (sum), or
        // `m = fmax(m, addend)` / `m = fmin(m, addend)` (running max/min, either operand order).
        let is_s = |e: &Expr| single_path(e) == Some(s);
        let (addend, redop): (&Expr, RedOp) = match op {
            ast::AssignOp::Add => (value, RedOp::Add),
            ast::AssignOp::Assign => match &value.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if is_s(lhs) => (rhs, RedOp::Add),
                ExprKind::Binary {
                    op: ast::BinOp::Add,
                    lhs,
                    rhs,
                } if is_s(rhs) => (lhs, RedOp::Add),
                ExprKind::Call { callee, args, .. } if args.len() == 2 => {
                    let red = match self.vectorizable_intrinsic(callee) {
                        Some(MathIntrinsic::Fmax) => RedOp::Fmax,
                        Some(MathIntrinsic::Fmin) => RedOp::Fmin,
                        _ => return None,
                    };
                    if is_s(&args[0]) {
                        (&args[1], red)
                    } else if is_s(&args[1]) {
                        (&args[0], red)
                    } else {
                        return None;
                    }
                }
                _ => return None,
            },
            _ => return None,
        };
        if expr_uses_sym(addend, s) {
            return None;
        }
        let mut lane: Option<MirType> = None;
        let mut acc: Vec<(Symbol, &Expr, bool, bool)> = Vec::new();
        if !self.vec_check_value(addend, j, &HashSet::new(), &mut lane, &mut acc) {
            return None;
        }
        let lane = lane?;
        // Require the lane type to match the accumulator exactly, so the horizontal reduce
        // (`s = s + lane`) is well-typed — bails on a mixed-width reduction (e.g. `f64 += f32`).
        if lane != sty {
            return None;
        }
        // fmax/fmin fold via a float compare + select; they need a float lane.
        if matches!(redop, RedOp::Fmax | RedOp::Fmin) && !lane.is_float() {
            return None;
        }
        let w = vector_width(&lane)?;
        Some((s, addend, lane, w, redop))
    }

    /// Attempt SIMD lowering of a loop whose bounds are already lowered to `ity` values (the form
    /// the `@parallel` outliner produces). Returns true on success.
    fn try_vectorize_ranged(
        &mut self,
        j: Symbol,
        start_val: ValueId,
        end_val: ValueId,
        ity: &MirType,
        body: &Block,
    ) -> bool {
        // A transcendental-activation chunk dispatches to the 256-bit AVX2 kernel here too, so an
        // `@parallel` activation runs multicore × 256-bit (each thread's chunk is one kernel call).
        // Elementwise, so the interpreter's whole-range pass and the native per-chunk passes agree.
        if let Some(calls) = self.match_vmath_body(j, body) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            self.emit_vmath_calls(s, e, calls);
            return true;
        }
        // A streaming affine map per `@parallel` chunk → the same 256-bit + non-temporal-store kernel,
        // so an `@parallel` saxpy/scale/add runs multicore × 256-bit with RFO-free stores. Elementwise,
        // so each thread's per-chunk pass agrees with the interpreter's whole-range pass.
        if let Some(plan) = self.match_velem_body(j, body) {
            let s = self.coerce_to(start_val, ity, &MirType::I64, true);
            let e = self.coerce_to(end_val, ity, &MirType::I64, true);
            self.emit_velem_call(s, e, &plan);
            return true;
        }
        let Some((lane, w)) = self.vectorizable(body, j) else {
            return false;
        };
        self.emit_vectorized_for(j, start_val, end_val, ity, &lane, w, body);
        true
    }

    /// Validate that `body` is vectorizable over loop variable `j`; return the shared lane type and
    /// vector width, or `None` to bail. Pure analysis (emits no MIR).
    fn vectorizable(&self, body: &Block, j: Symbol) -> Option<(MirType, u32)> {
        if body.tail.is_some() || body.stmts.is_empty() {
            return None;
        }
        // inner `let` names become vector temps; they may not appear inside index expressions.
        let mut locals: HashSet<Symbol> = HashSet::new();
        // every array access: (base, index expr, unit-stride?, is_write).
        let mut acc: Vec<(Symbol, &Expr, bool, bool)> = Vec::new();
        let mut lane: Option<MirType> = None;

        for s in &body.stmts {
            match &s.kind {
                StmtKind::Let {
                    pat:
                        Pattern {
                            kind: ast::PatKind::Ident(name),
                            ..
                        },
                    init: Some(e),
                    ..
                } => {
                    if !self.vec_check_value(e, j, &locals, &mut lane, &mut acc) {
                        return None;
                    }
                    locals.insert(*name);
                }
                StmtKind::Assign { target, op, value } => {
                    if !self.vec_check_value(value, j, &locals, &mut lane, &mut acc) {
                        return None;
                    }
                    let _ = op; // compound vs plain handled identically for validation
                    match &target.kind {
                        // scalar reassignment of an inner vector temp (e.g. Horner `r = r*v + c`)
                        ExprKind::Path(p) if p.is_single() && locals.contains(&p.first().sym) => {}
                        // store to an array element at a unit-stride index
                        ExprKind::Index { base, indices } if indices.len() == 1 => {
                            let bsym = single_path(base)?;
                            if uses_any(&indices[0], &locals) {
                                return None; // index must not depend on inner temps
                            }
                            if affine_stride(&indices[0], j) != Some(1) {
                                return None; // stores must be unit-stride
                            }
                            let elem = self.array_elem(base)?;
                            set_or_check(&mut lane, &elem)?;
                            acc.push((bsym, &indices[0], true, true));
                        }
                        _ => return None,
                    }
                }
                _ => return None, // no nested control flow / calls / returns in a vector body
            }
        }

        // No array that is written may be accessed at a second (different) index: that would be a
        // loop-carried dependence the lane-parallel form would break.
        for &(base, idx, _unit, is_w) in &acc {
            if is_w || acc.iter().any(|a| a.0 == base && a.3) {
                // `base` is written somewhere; every access to it must be the identical index.
                if acc
                    .iter()
                    .any(|a| a.0 == base && !exprs_struct_eq(a.1, idx))
                {
                    return None;
                }
            }
        }

        let lane = lane?;
        let w = vector_width(&lane)?;
        Some((lane, w))
    }

    /// Validate that `e` is a vectorizable value-expression (produces lane-typed vectors), recording
    /// any array reads into `acc` and pinning the shared lane type. Returns false to bail.
    /// The math intrinsic a callee names — if it is one and is *not* shadowed by a user function
    /// (mirroring `lower_call`'s precedence). Lets the vectorizer decide which calls it can lower
    /// per lane.
    fn vectorizable_intrinsic(&self, callee: &Expr) -> Option<MathIntrinsic> {
        let ExprKind::Path(p) = &callee.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let name = p.first().sym;
        if matches!(
            self.sema.defs.lookup(name).map(|d| &d.kind),
            Some(DefKind::Fn(_))
        ) {
            return None;
        }
        math_intrinsic(self.interner.resolve(name))
    }

    fn vec_check_value<'b>(
        &self,
        e: &'b Expr,
        j: Symbol,
        locals: &HashSet<Symbol>,
        lane: &mut Option<MirType>,
        acc: &mut Vec<(Symbol, &'b Expr, bool, bool)>,
    ) -> bool {
        match &e.kind {
            // an inner vector temp: lane-typed by construction.
            ExprKind::Path(p) if p.is_single() && locals.contains(&p.first().sym) => true,
            // an invariant scalar (param/outer local): must be the lane type, and must not be the
            // loop variable used as a value (that would need an index vector, which we don't form).
            ExprKind::Path(p) if p.is_single() => {
                let t = self.expr_mir(e);
                p.first().sym != j && is_numeric(&t) && set_or_check(lane, &t).is_some()
            }
            // a numeric literal: lane-typed (the splat rounds once, like the scalar path).
            ExprKind::Int(_) | ExprKind::Float(_) => {
                let t = self.expr_mir(e);
                is_numeric(&t) && set_or_check(lane, &t).is_some()
            }
            ExprKind::Index { base, indices } if indices.len() == 1 => {
                if single_path(base).is_none() || uses_any(&indices[0], locals) {
                    return false;
                }
                match affine_stride(&indices[0], j) {
                    Some(0) | Some(1) => {}
                    _ => return false,
                }
                let Some(elem) = self.array_elem(base) else {
                    return false;
                };
                if set_or_check(lane, &elem).is_none() {
                    return false;
                }
                let bsym = single_path(base).unwrap();
                let unit = affine_stride(&indices[0], j) == Some(1);
                acc.push((bsym, &indices[0], unit, false));
                true
            }
            ExprKind::Binary { op, lhs, rhs } => {
                use ast::BinOp::*;
                if !matches!(op, Add | Sub | Mul | Div) {
                    return false;
                }
                self.vec_check_value(lhs, j, locals, lane, acc)
                    && self.vec_check_value(rhs, j, locals, lane, acc)
            }
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => self.vec_check_value(expr, j, locals, lane, acc),
            // `if cond { a } else { b }` if-converts to a vector compare + blend (e.g. ReLU).
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let (Some(else_e), Some(tv)) = (else_branch.as_deref(), block_value(then_branch))
                else {
                    return false;
                };
                let (Some(ev), ExprKind::Binary { op, lhs, rhs }) =
                    (branch_value(else_e), &cond.kind)
                else {
                    return false;
                };
                use ast::BinOp::*;
                if !matches!(op, Lt | Le | Gt | Ge | Eq | Ne) {
                    return false;
                }
                self.vec_check_value(lhs, j, locals, lane, acc)
                    && self.vec_check_value(rhs, j, locals, lane, acc)
                    && self.vec_check_value(tv, j, locals, lane, acc)
                    && self.vec_check_value(ev, j, locals, lane, acc)
            }
            // Element-wise math intrinsics over vectorizable args. sqrt/rsqrt are one lane op each;
            // fmax/fmin are a lane compare + blend; exp is the f32 polynomial expanded per lane.
            ExprKind::Call { callee, args, .. } => match self.vectorizable_intrinsic(callee) {
                Some(MathIntrinsic::Sqrt | MathIntrinsic::Rsqrt) => {
                    args.len() == 1 && self.vec_check_value(&args[0], j, locals, lane, acc)
                }
                Some(MathIntrinsic::Fmax | MathIntrinsic::Fmin) => {
                    args.len() == 2
                        && self.vec_check_value(&args[0], j, locals, lane, acc)
                        && self.vec_check_value(&args[1], j, locals, lane, acc)
                }
                Some(
                    MathIntrinsic::Exp
                    | MathIntrinsic::Log
                    | MathIntrinsic::Erf
                    | MathIntrinsic::Sin
                    | MathIntrinsic::Cos
                    | MathIntrinsic::Tanh
                    | MathIntrinsic::Sigmoid
                    | MathIntrinsic::Silu
                    | MathIntrinsic::Gelu,
                ) => {
                    // These build on the exp/log polynomials, which vectorize only for an f32 lane
                    // (their IEEE-754 exponent surgery is f32-specific).
                    args.len() == 1
                        && self.vec_check_value(&args[0], j, locals, lane, acc)
                        && *lane == Some(MirType::F32)
                }
                Some(MathIntrinsic::Pow) => {
                    // pow = exp(y·log(x)); two args, f32 lane only (same reason as exp/log).
                    args.len() == 2
                        && self.vec_check_value(&args[0], j, locals, lane, acc)
                        && self.vec_check_value(&args[1], j, locals, lane, acc)
                        && *lane == Some(MirType::F32)
                }
                None => false,
            },
            _ => false,
        }
    }

    /// The element MIR type of an array-valued base expression, via sema.
    fn array_elem(&self, base: &Expr) -> Option<MirType> {
        match self.expr_ty(base) {
            Ty::Array { elem, .. } => Some(mir_ty(&elem)),
            _ => None,
        }
    }

    /// Emit the vectorized loop as three strips that share one index slot `j`: an unrolled vector
    /// loop (`VEC_UNROLL` independent vector groups per iteration), then a single-vector loop, then
    /// a scalar remainder. The bounds are already-lowered `ity` values.
    #[allow(clippy::too_many_arguments)]
    fn emit_vectorized_for(
        &mut self,
        j: Symbol,
        s0: ValueId,
        end_v: ValueId,
        ity: &MirType,
        lane: &MirType,
        w: u32,
        body: &Block,
    ) {
        let vty = MirType::Vec(Box::new(lane.clone()), w);
        let slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: s0,
        });
        // A scratch slot holding the per-unroll-copy index (`jbase + u*W`), so unit-stride accesses
        // pick up the lane offset with no per-access arithmetic.
        let jtmp = self.builder.alloca(ity.clone());

        self.push_scope();
        self.bind(j, slot, ity.clone());

        self.emit_vector_strip(j, slot, jtmp, end_v, ity, lane, &vty, w, VEC_UNROLL, body);
        self.emit_vector_strip(j, slot, jtmp, end_v, ity, lane, &vty, w, 1, body);
        self.emit_scalar_tail(j, slot, end_v, ity, body);

        self.pop_scope();
    }

    /// One strip-mined vector loop: while a full `unroll * W` block fits, emit `unroll` independent
    /// vector groups (fresh `vlocals` each, so their dependency chains overlap on the FP units),
    /// then advance `j` by `unroll * W`.
    #[allow(clippy::too_many_arguments)]
    fn emit_vector_strip(
        &mut self,
        j: Symbol,
        slot: ValueId,
        jtmp: ValueId,
        end_v: ValueId,
        ity: &MirType,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        unroll: u32,
        body: &Block,
    ) {
        let span = (unroll * w) as i128;
        let off = self
            .builder
            .build(ity.clone(), Op::ConstInt(span - 1, ity.clone()));
        let vlimit = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Sub, end_v, off));

        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let done = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        // header: while j < end - (unroll*W - 1)
        self.builder.switch_to(hdr);
        self.terminated = false;
        self.bind(j, slot, ity.clone());
        let jv = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jv, vlimit));
        self.builder.cond_br(c, bb, vec![], done, vec![]);

        // body: `unroll` vector groups at offsets 0, W, 2W, …; then j += unroll*W
        self.builder.switch_to(bb);
        self.terminated = false;
        let jbase = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        for u in 0..unroll {
            let ju = if u == 0 {
                jbase
            } else {
                let o = self
                    .builder
                    .build(ity.clone(), Op::ConstInt((u * w) as i128, ity.clone()));
                self.builder
                    .build(ity.clone(), Op::Bin(BinOp::Add, jbase, o))
            };
            self.builder.build_void(Op::Store {
                ptr: jtmp,
                value: ju,
            });
            self.bind(j, jtmp, ity.clone());
            let mut vlocals: HashMap<Symbol, ValueId> = HashMap::new();
            // Each unroll copy reads different addresses (jbase + u*W), so the load cache is per-copy.
            self.vec_loads.clear();
            for s in &body.stmts {
                self.vec_lower_stmt(s, j, lane, vty, w, &mut vlocals);
            }
        }
        self.bind(j, slot, ity.clone());
        let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let stepc = self
            .builder
            .build(ity.clone(), Op::ConstInt(span, ity.clone()));
        let jn = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, jc, stepc));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: jn,
        });
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(done);
        self.terminated = false;
    }

    /// The scalar remainder: the original body lowered normally, stepping `j` by 1 to `end`.
    fn emit_scalar_tail(
        &mut self,
        j: Symbol,
        slot: ValueId,
        end_v: ValueId,
        ity: &MirType,
        body: &Block,
    ) {
        self.bind(j, slot, ity.clone());
        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(hdr);
        self.terminated = false;
        let jr = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let rc = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jr, end_v));
        self.builder.cond_br(rc, bb, vec![], exit, vec![]);

        self.builder.switch_to(bb);
        self.terminated = false;
        self.loops.push((hdr, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
            let one = self
                .builder
                .build(ity.clone(), Op::ConstInt(1, ity.clone()));
            let jn = self
                .builder
                .build(ity.clone(), Op::Bin(BinOp::Add, jc, one));
            self.builder.build_void(Op::Store {
                ptr: slot,
                value: jn,
            });
            self.builder.br(hdr, vec![]);
        }
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Emit a vectorized float reduction over `[s0, end)`: `VEC_UNROLL` independent vector-lane
    /// accumulators summed in an unrolled main loop, a single-vector cleanup loop, then a horizontal
    /// reduce of the lanes into `s`, then a scalar remainder. This reassociates the sum (vs strict
    /// left-to-right), which is sound for a reduction and is what makes it fast — `w` lanes × unroll
    /// independent FMA chains instead of one serial dependency.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduction(
        &mut self,
        j: Symbol,
        s0: ValueId,
        end_v: ValueId,
        ity: &MirType,
        s_sym: Symbol,
        addend: &Expr,
        lane: &MirType,
        w: u32,
        redop: RedOp,
    ) {
        let vty = MirType::Vec(Box::new(lane.clone()), w);
        let slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: s0,
        });
        let jtmp = self.builder.alloca(ity.clone());

        // `VEC_UNROLL` accumulators, each `w` contiguous lanes (an `Array` slot so the vector
        // store/load address `w` real elements), initialised to the fold's identity before the
        // loop: 0 for a sum, -inf for `fmax`, +inf for `fmin` (so empty tail lanes never win).
        let vid = match redop {
            RedOp::Add => {
                let zero = self.const_zero(lane.clone());
                self.builder.build(vty.clone(), Op::Splat(zero))
            }
            RedOp::Fmax => self.splat_const_f(f64::NEG_INFINITY, &vty),
            RedOp::Fmin => self.splat_const_f(f64::INFINITY, &vty),
        };
        let accs: Vec<ValueId> = (0..VEC_UNROLL)
            .map(|_| {
                let a = self
                    .builder
                    .alloca(MirType::Array(Box::new(lane.clone()), w));
                self.builder.build_void(Op::Store { ptr: a, value: vid });
                a
            })
            .collect();

        self.push_scope();
        self.bind(j, slot, ity.clone());

        self.emit_reduction_strip(
            j, slot, jtmp, end_v, ity, lane, &vty, w, VEC_UNROLL, &accs, addend, redop,
        );
        self.emit_reduction_strip(
            j,
            slot,
            jtmp,
            end_v,
            ity,
            lane,
            &vty,
            w,
            1,
            &accs[..1],
            addend,
            redop,
        );

        // Combine the accumulators, then horizontally reduce the lanes into `s` with the same fold
        // (sum / max / min) the loop body used.
        let mut total = self
            .builder
            .build(vty.clone(), Op::Load(accs[0], vty.clone()));
        for &a in &accs[1..] {
            let v = self.builder.build(vty.clone(), Op::Load(a, vty.clone()));
            total = self.reduce_combine_vec(redop, lane, &vty, w, total, v);
        }
        let scratch = self
            .builder
            .alloca(MirType::Array(Box::new(lane.clone()), w));
        self.builder.build_void(Op::Store {
            ptr: scratch,
            value: total,
        });
        let (s_slot, s_ty) = self.lookup(s_sym).expect("reduction var in scope");
        for k in 0..w {
            let idxk = self
                .builder
                .build(MirType::I64, Op::ConstInt(k as i128, MirType::I64));
            let addr = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: scratch,
                    index: idxk,
                    elem: lane.clone(),
                },
            );
            let lane_v = self
                .builder
                .build(lane.clone(), Op::Load(addr, lane.clone()));
            let sv = self
                .builder
                .build(s_ty.clone(), Op::Load(s_slot, s_ty.clone()));
            let sum = self.reduce_combine_scalar(redop, &s_ty, sv, lane_v);
            self.builder.build_void(Op::Store {
                ptr: s_slot,
                value: sum,
            });
        }

        self.emit_reduction_tail(j, slot, end_v, ity, s_sym, addend, redop);
        self.pop_scope();
    }

    /// One strip of the reduction main loop: while a full `unroll*W` block fits, fold `unroll`
    /// independent vector groups into `accs[0..unroll]` (separate accumulators so their FMA chains
    /// overlap), then advance `j` by `unroll*W`.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduction_strip(
        &mut self,
        j: Symbol,
        slot: ValueId,
        jtmp: ValueId,
        end_v: ValueId,
        ity: &MirType,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        unroll: u32,
        accs: &[ValueId],
        addend: &Expr,
        redop: RedOp,
    ) {
        let span = (unroll * w) as i128;
        let off = self
            .builder
            .build(ity.clone(), Op::ConstInt(span - 1, ity.clone()));
        let vlimit = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Sub, end_v, off));

        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let done = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(hdr);
        self.terminated = false;
        self.bind(j, slot, ity.clone());
        let jv = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jv, vlimit));
        self.builder.cond_br(c, bb, vec![], done, vec![]);

        self.builder.switch_to(bb);
        self.terminated = false;
        let jbase = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        for u in 0..unroll {
            let ju = if u == 0 {
                jbase
            } else {
                let o = self
                    .builder
                    .build(ity.clone(), Op::ConstInt((u * w) as i128, ity.clone()));
                self.builder
                    .build(ity.clone(), Op::Bin(BinOp::Add, jbase, o))
            };
            self.builder.build_void(Op::Store {
                ptr: jtmp,
                value: ju,
            });
            self.bind(j, jtmp, ity.clone());
            let acc_slot = accs[u as usize];
            let cur = self
                .builder
                .build(vty.clone(), Op::Load(acc_slot, vty.clone()));
            let mut vlocals: HashMap<Symbol, ValueId> = HashMap::new();
            // Per-copy load cache (so `(x[i]-y[i])*(x[i]-y[i])` loads x[i],y[i] once each).
            self.vec_loads.clear();
            let nv = match redop {
                RedOp::Add => self.vec_accumulate(cur, addend, j, lane, vty, w, &mut vlocals),
                RedOp::Fmax | RedOp::Fmin => {
                    let vx = self.vec_lower_value(addend, j, lane, vty, w, &mut vlocals);
                    self.reduce_combine_vec(redop, lane, vty, w, cur, vx)
                }
            };
            self.builder.build_void(Op::Store {
                ptr: acc_slot,
                value: nv,
            });
        }
        self.bind(j, slot, ity.clone());
        let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let stepc = self
            .builder
            .build(ity.clone(), Op::ConstInt(span, ity.clone()));
        let jn = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, jc, stepc));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: jn,
        });
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(done);
        self.terminated = false;
    }

    /// The reduction's scalar remainder: `while j < end { s = fold(s, addend); j += 1; }`.
    #[allow(clippy::too_many_arguments)]
    fn emit_reduction_tail(
        &mut self,
        j: Symbol,
        slot: ValueId,
        end_v: ValueId,
        ity: &MirType,
        s_sym: Symbol,
        addend: &Expr,
        redop: RedOp,
    ) {
        self.bind(j, slot, ity.clone());
        let hdr = self.builder.new_block();
        let bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(hdr);
        self.terminated = false;
        let jr = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let rc = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, jr, end_v));
        self.builder.cond_br(rc, bb, vec![], exit, vec![]);

        self.builder.switch_to(bb);
        self.terminated = false;
        self.bind(j, slot, ity.clone());
        let (s_slot, s_ty) = self.lookup(s_sym).expect("reduction var in scope");
        let sv = self
            .builder
            .build(s_ty.clone(), Op::Load(s_slot, s_ty.clone()));
        let sum = match redop {
            RedOp::Add => self.scalar_accumulate(sv, addend, &s_ty),
            RedOp::Fmax | RedOp::Fmin => {
                let from = self.expr_mir(addend);
                let a = self.lower_expr(addend);
                let a = self.coerce_to(a, &from, &s_ty, true);
                self.reduce_combine_scalar(redop, &s_ty, sv, a)
            }
        };
        self.builder.build_void(Op::Store {
            ptr: s_slot,
            value: sum,
        });
        let jc = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let one = self
            .builder
            .build(ity.clone(), Op::ConstInt(1, ity.clone()));
        let jn = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, jc, one));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: jn,
        });
        self.builder.br(hdr, vec![]);

        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Combine two vectors under the reduction's fold: lane-wise sum (`FAdd`/`Add`), or `fmax`/`fmin`
    /// as the same compare + select the scalar intrinsic lowers to (so vector and scalar agree).
    fn reduce_combine_vec(
        &mut self,
        redop: RedOp,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        match redop {
            RedOp::Add => {
                let op = if lane.is_float() {
                    BinOp::FAdd
                } else {
                    BinOp::Add
                };
                self.builder.build(vty.clone(), Op::Bin(op, a, b))
            }
            RedOp::Fmax | RedOp::Fmin => {
                let pred = if redop == RedOp::Fmax {
                    CmpOp::Fogt
                } else {
                    CmpOp::Folt
                };
                let mty = MirType::Vec(Box::new(mask_lane_type(lane)), w);
                let mask = self.builder.build(mty, Op::Cmp(pred, a, b));
                self.builder.build(vty.clone(), Op::Select(mask, a, b))
            }
        }
    }

    /// Scalar counterpart of `reduce_combine_vec`, used for the horizontal lane reduce and the
    /// scalar remainder.
    fn reduce_combine_scalar(
        &mut self,
        redop: RedOp,
        ty: &MirType,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        match redop {
            RedOp::Add => {
                let op = if ty.is_float() {
                    BinOp::FAdd
                } else {
                    BinOp::Add
                };
                self.builder.build(ty.clone(), Op::Bin(op, a, b))
            }
            RedOp::Fmax | RedOp::Fmin => {
                let pred = if redop == RedOp::Fmax {
                    CmpOp::Fogt
                } else {
                    CmpOp::Folt
                };
                let mask = self.builder.build(mask_ty(ty), Op::Cmp(pred, a, b));
                self.builder.build(ty.clone(), Op::Select(mask, a, b))
            }
        }
    }

    /// Fold `addend` into vector accumulator `acc`. For a float reduction, `acc + a*b` fuses into one
    /// `Fma`; integer reductions use a plain vector `Add` (no integer FMA, and reassociation is
    /// exact so it needs none).
    #[allow(clippy::too_many_arguments)]
    fn vec_accumulate(
        &mut self,
        acc: ValueId,
        addend: &Expr,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) -> ValueId {
        if lane.is_float() {
            if let ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } = &addend.kind
            {
                let a = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
                let b = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
                return self.builder.build(vty.clone(), Op::Fma(a, b, acc));
            }
        }
        let vx = self.vec_lower_value(addend, j, lane, vty, w, vlocals);
        let add = if lane.is_float() {
            BinOp::FAdd
        } else {
            BinOp::Add
        };
        self.builder.build(vty.clone(), Op::Bin(add, acc, vx))
    }

    /// Scalar `acc + addend`, fusing `acc + a*b` into one `Fma` for floats; plain `Add` for ints.
    fn scalar_accumulate(&mut self, acc: ValueId, addend: &Expr, ty: &MirType) -> ValueId {
        if ty.is_float() {
            if let ExprKind::Binary {
                op: ast::BinOp::Mul,
                lhs,
                rhs,
            } = &addend.kind
            {
                let a = self.lower_fma_operand(lhs, ty);
                let b = self.lower_fma_operand(rhs, ty);
                return self.builder.build(ty.clone(), Op::Fma(a, b, acc));
            }
        }
        let v = self.lower_expr(addend);
        let vty = self.expr_mir(addend);
        let v = self.coerce_to(v, &vty, ty, true);
        let add = if ty.is_float() {
            BinOp::FAdd
        } else {
            BinOp::Add
        };
        self.builder.build(ty.clone(), Op::Bin(add, acc, v))
    }

    /// Lower one statement of a vector-loop body. Mirrors the validated shapes in `vectorizable`.
    fn vec_lower_stmt(
        &mut self,
        s: &Stmt,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) {
        match &s.kind {
            StmtKind::Let {
                pat:
                    Pattern {
                        kind: ast::PatKind::Ident(name),
                        ..
                    },
                init: Some(e),
                ..
            } => {
                let v = self.vec_lower_value(e, j, lane, vty, w, vlocals);
                vlocals.insert(*name, v);
            }
            StmtKind::Assign { target, op, value } => {
                let rhs = self.vec_lower_value(value, j, lane, vty, w, vlocals);
                match &target.kind {
                    ExprKind::Path(p) if p.is_single() && vlocals.contains_key(&p.first().sym) => {
                        let name = p.first().sym;
                        let stored = if matches!(op, ast::AssignOp::Assign) {
                            rhs
                        } else {
                            let cur = vlocals[&name];
                            let bin = compound_binop(*op, lane.is_float(), lane_signed(lane));
                            self.builder.build(vty.clone(), Op::Bin(bin, cur, rhs))
                        };
                        vlocals.insert(name, stored);
                    }
                    ExprKind::Index { base, indices } => {
                        let addr = self.vec_elem_addr(base, &indices[0], lane);
                        let stored = if matches!(op, ast::AssignOp::Assign) {
                            rhs
                        } else {
                            let cur = self.builder.build(vty.clone(), Op::Load(addr, vty.clone()));
                            let bin = compound_binop(*op, lane.is_float(), lane_signed(lane));
                            self.builder.build(vty.clone(), Op::Bin(bin, cur, rhs))
                        };
                        self.builder.build_void(Op::Store {
                            ptr: addr,
                            value: stored,
                        });
                        // A store may invalidate any cached load (conservatively, all of them).
                        self.vec_loads.clear();
                    }
                    _ => unreachable!("vec_lower_stmt on unvalidated target"),
                }
            }
            _ => unreachable!("vec_lower_stmt on unvalidated statement"),
        }
    }

    /// Lower a value-expression to a `vty` vector. Array reads become vector loads (unit-stride) or
    /// scalar-load-then-splat (invariant); scalars splat; arithmetic is lane-wise.
    fn vec_lower_value(
        &mut self,
        e: &Expr,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) -> ValueId {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() && vlocals.contains_key(&p.first().sym) => {
                vlocals[&p.first().sym]
            }
            ExprKind::Index { base, indices } if indices.len() == 1 => {
                // Reuse a vector already loaded for this exact index in the current body copy: a
                // body like relu's `if x[i] > 0 { x[i] } else { 0 }` reads x[i] twice; without this
                // it loads twice (50% extra memory traffic), since loads aren't CSE'd (alias-unsafe).
                let key = load_key(base, &indices[0], self.interner);
                if let Some(k) = &key {
                    if let Some(&cached) = self.vec_loads.get(k) {
                        return cached;
                    }
                }
                let addr = self.vec_elem_addr(base, &indices[0], lane);
                let v = if affine_stride(&indices[0], j) == Some(1) {
                    self.builder.build(vty.clone(), Op::Load(addr, vty.clone()))
                } else {
                    // invariant in j: load one scalar and broadcast.
                    let scalar = self
                        .builder
                        .build(lane.clone(), Op::Load(addr, lane.clone()));
                    self.builder.build(vty.clone(), Op::Splat(scalar))
                };
                if let Some(k) = key {
                    self.vec_loads.insert(k, v);
                }
                v
            }
            ExprKind::Binary { op, lhs, rhs } => {
                // Contract a float `x + y*z` into one lane-wise fused multiply-add — this is the
                // matmul/saxpy inner-loop win (one `vfmadd` per vector instead of mul+add).
                if *op == ast::BinOp::Add && lane.is_float() {
                    if let Some(v) = self.vec_try_fma(lhs, rhs, j, lane, vty, w, vlocals) {
                        return v;
                    }
                }
                let l = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
                let r = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
                let bin = arith_binop(*op, lane.is_float(), lane_signed(lane));
                self.builder.build(vty.clone(), Op::Bin(bin, l, r))
            }
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => {
                let v = self.vec_lower_value(expr, j, lane, vty, w, vlocals);
                self.builder.build(vty.clone(), Op::Neg(v))
            }
            // if-conversion: `if a CMP b { t } else { e }` -> vector compare mask + lane-wise blend.
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
                    unreachable!("vec_lower_value on unvalidated if-condition");
                };
                let lv = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
                let rv = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
                let mask_ty = MirType::Vec(Box::new(mask_lane_type(lane)), w);
                let pred = cmp_pred(*op, lane.is_float(), lane_signed(lane));
                let mask = self.builder.build(mask_ty, Op::Cmp(pred, lv, rv));
                let tv = block_value(then_branch).unwrap();
                let ev = branch_value(else_branch.as_deref().unwrap()).unwrap();
                let tvec = self.vec_lower_value(tv, j, lane, vty, w, vlocals);
                let evec = self.vec_lower_value(ev, j, lane, vty, w, vlocals);
                self.builder
                    .build(vty.clone(), Op::Select(mask, tvec, evec))
            }
            // Element-wise math intrinsics, lowered per lane (see `vec_check_value`).
            ExprKind::Call { callee, args, .. } => match self.vectorizable_intrinsic(callee) {
                Some(MathIntrinsic::Sqrt) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.builder.build(vty.clone(), Op::Sqrt(x))
                }
                Some(MathIntrinsic::Rsqrt) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let s = self.builder.build(vty.clone(), Op::Sqrt(x));
                    let one = self.splat_const_f(1.0, vty);
                    self.builder
                        .build(vty.clone(), Op::Bin(BinOp::FDiv, one, s))
                }
                Some(op @ (MathIntrinsic::Fmax | MathIntrinsic::Fmin)) => {
                    let a = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let b = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    let pred = if matches!(op, MathIntrinsic::Fmax) {
                        CmpOp::Fogt
                    } else {
                        CmpOp::Folt
                    };
                    let mty = MirType::Vec(Box::new(mask_lane_type(lane)), w);
                    let mask = self.builder.build(mty, Op::Cmp(pred, a, b));
                    self.builder.build(vty.clone(), Op::Select(mask, a, b))
                }
                Some(MathIntrinsic::Exp) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_exp_f32(x, vty)
                }
                Some(MathIntrinsic::Log) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_log_f32(x, vty)
                }
                Some(MathIntrinsic::Pow) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    let y = self.vec_lower_value(&args[1], j, lane, vty, w, vlocals);
                    let lx = self.emit_log_f32(x, vty);
                    let ylx = self.builder.build(vty.clone(), Op::Bin(BinOp::FMul, y, lx));
                    self.emit_exp_f32(ylx, vty)
                }
                Some(MathIntrinsic::Erf) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_erf_f32(x, vty)
                }
                Some(op @ (MathIntrinsic::Sin | MathIntrinsic::Cos)) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_trig_f32(x, vty, matches!(op, MathIntrinsic::Cos))
                }
                Some(MathIntrinsic::Tanh) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_tanh(x, vty)
                }
                Some(MathIntrinsic::Sigmoid) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_sigmoid(x, vty)
                }
                Some(MathIntrinsic::Silu) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_silu(x, vty)
                }
                Some(MathIntrinsic::Gelu) => {
                    let x = self.vec_lower_value(&args[0], j, lane, vty, w, vlocals);
                    self.emit_gelu(x, vty)
                }
                None => unreachable!("vectorizer accepted a call it cannot lower"),
            },
            // an invariant scalar or literal: lower as a scalar (coerced to the lane type) and splat.
            _ => {
                let from = self.expr_mir(e);
                let scalar = self.lower_expr(e);
                let scalar = self.coerce_to(scalar, &from, lane, true);
                self.builder.build(vty.clone(), Op::Splat(scalar))
            }
        }
    }

    /// In a vectorized loop body, contract a float `x + y*z` (or `y*z + x`) into one lane-wise
    /// `Fma`. Returns the fused vector value, or `None` if neither side is a multiply (the caller
    /// then emits a plain vector add). The whole expression tree was already accepted by the
    /// vectorizer's dependence check, so lowering its leaves here is sound.
    #[allow(clippy::too_many_arguments)]
    fn vec_try_fma(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        j: Symbol,
        lane: &MirType,
        vty: &MirType,
        w: u32,
        vlocals: &mut HashMap<Symbol, ValueId>,
    ) -> Option<ValueId> {
        fn as_fmul(e: &Expr) -> Option<(&Expr, &Expr)> {
            match &e.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Mul,
                    lhs,
                    rhs,
                } => Some((lhs.as_ref(), rhs.as_ref())),
                _ => None,
            }
        }
        if let Some((y, z)) = as_fmul(lhs) {
            let yv = self.vec_lower_value(y, j, lane, vty, w, vlocals);
            let zv = self.vec_lower_value(z, j, lane, vty, w, vlocals);
            let xv = self.vec_lower_value(rhs, j, lane, vty, w, vlocals);
            return Some(self.builder.build(vty.clone(), Op::Fma(yv, zv, xv)));
        }
        if let Some((y, z)) = as_fmul(rhs) {
            let xv = self.vec_lower_value(lhs, j, lane, vty, w, vlocals);
            let yv = self.vec_lower_value(y, j, lane, vty, w, vlocals);
            let zv = self.vec_lower_value(z, j, lane, vty, w, vlocals);
            return Some(self.builder.build(vty.clone(), Op::Fma(yv, zv, xv)));
        }
        None
    }

    /// The element address `&base[index]` (a `gep` by the scalar index, reusing the loop index slot
    /// bound for `j`), used as the base of a vector load/store.
    fn vec_elem_addr(&mut self, base: &Expr, index: &Expr, lane: &MirType) -> ValueId {
        let base_ptr = self.lower_expr(base);
        let idx = self.lower_expr(index);
        self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base_ptr,
                index: idx,
                elem: lane.clone(),
            },
        )
    }

    /// Lower `for idx in start..end { body }` where `start`/`end` are already-lowered `i64` values
    /// (used by parallel outlining). The loop runs in `ity` — the index variable's source type, so
    /// the body's index arithmetic stays type-consistent — with the `i64` bounds coerced into it.
    fn lower_ranged_loop(
        &mut self,
        idx: Symbol,
        start: ValueId,
        end: ValueId,
        ity: MirType,
        body: &Block,
    ) {
        // The runtime hands us `[start, end)` as i64; narrow to the index type the body expects.
        let start = self.coerce_to(start, &MirType::I64, &ity, true);
        let end = self.coerce_to(end, &MirType::I64, &ity, true);

        // SIMD-vectorize the per-thread chunk too, so `@parallel` kernels run vectorized on every
        // core (parallelism × SIMD), not scalar-per-core.
        if self.try_vectorize_ranged(idx, start, end, &ity, body) {
            return;
        }

        let slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: start,
        });
        self.push_scope();
        self.bind(idx, slot, ity.clone());

        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        self.terminated = false;
        let i_val = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
        let c = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, i_val, end));
        self.builder.cond_br(c, body_bb, vec![], exit, vec![]);

        self.builder.switch_to(body_bb);
        self.terminated = false;
        self.loops.push((header, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            let cur = self.builder.build(ity.clone(), Op::Load(slot, ity.clone()));
            let one = self
                .builder
                .build(ity.clone(), Op::ConstInt(1, ity.clone()));
            let next = self
                .builder
                .build(ity.clone(), Op::Bin(BinOp::Add, cur, one));
            self.builder.build_void(Op::Store {
                ptr: slot,
                value: next,
            });
            self.builder.br(header, vec![]);
        }
        self.pop_scope();
        self.builder.switch_to(exit);
        self.terminated = false;
    }

    fn lower_if(&mut self, cond: &Expr, then_branch: &Block, else_branch: Option<&Expr>) {
        let c = self.lower_expr(cond);
        let then_bb = self.builder.new_block();
        let merge = self.builder.new_block();
        let else_bb = if else_branch.is_some() {
            self.builder.new_block()
        } else {
            merge
        };
        self.builder.cond_br(c, then_bb, vec![], else_bb, vec![]);

        self.builder.switch_to(then_bb);
        self.terminated = false;
        self.lower_block(then_branch);
        if !self.terminated {
            self.builder.br(merge, vec![]);
        }

        if let Some(e) = else_branch {
            self.builder.switch_to(else_bb);
            self.terminated = false;
            self.lower_expr_stmt(e);
            if !self.terminated {
                self.builder.br(merge, vec![]);
            }
        }

        self.builder.switch_to(merge);
        self.terminated = false;
    }

    /// Lower an `if`/`else` used as a *value* (e.g. a block tail or `let x = if …`). When the
    /// expression has a real (non-unit) type and both arms exist, the merge block takes a
    /// parameter that each arm passes its value to; otherwise this behaves like the statement form
    /// and yields a dummy zero (the value is unused).
    fn lower_if_value(
        &mut self,
        cond: &Expr,
        then_branch: &Block,
        else_branch: Option<&Expr>,
        e: &Expr,
    ) -> ValueId {
        let result_ty = self.expr_mir(e);
        let produces_value = else_branch.is_some() && result_ty != MirType::Void;

        let c = self.lower_expr(cond);
        let then_bb = self.builder.new_block();
        let merge = self.builder.new_block();
        let else_bb = if else_branch.is_some() {
            self.builder.new_block()
        } else {
            merge
        };
        let merge_param = if produces_value {
            Some(self.builder.block_param(merge, result_ty.clone()))
        } else {
            None
        };
        self.builder.cond_br(c, then_bb, vec![], else_bb, vec![]);

        // then arm
        self.builder.switch_to(then_bb);
        self.terminated = false;
        let tv = self.lower_block(then_branch);
        if !self.terminated {
            let args = match (merge_param.is_some(), tv) {
                (true, Some(v)) => vec![v],
                (true, None) => vec![self.const_zero(result_ty.clone())],
                (false, _) => vec![],
            };
            self.builder.br(merge, args);
        }

        // else arm
        if let Some(els) = else_branch {
            self.builder.switch_to(else_bb);
            self.terminated = false;
            let ev = self.lower_expr(els);
            if !self.terminated {
                let args = if merge_param.is_some() {
                    vec![ev]
                } else {
                    vec![]
                };
                self.builder.br(merge, args);
            }
        }

        self.builder.switch_to(merge);
        self.terminated = false;
        match merge_param {
            Some(p) => p,
            None => self.const_zero(if result_ty == MirType::Void {
                MirType::I32
            } else {
                result_ty
            }),
        }
    }

    // ---- places (lvalues) ----

    /// Initialize an array alloca (`base`) of `n` elements of type `elem` from an array-literal or
    /// array-repeat initializer, storing each element through a `gep`.
    fn lower_array_init(&mut self, base: ValueId, elem: &MirType, n: u32, init: &Expr) {
        match &init.kind {
            ExprKind::ArrayLit(elems) => {
                for (i, el) in elems.iter().enumerate() {
                    let v0 = self.lower_expr(el);
                    let vty = self.expr_mir(el);
                    // Coerce to the element type so e.g. a `[bf16; N]` literal stores bf16-rounded
                    // 16-bit values, not raw f32. A no-op when the element already matches.
                    let v = self.coerce_to(v0, &vty, elem, self.signed(el));
                    self.store_element(base, elem, i as i128, v);
                }
            }
            ExprKind::ArrayRepeat { value, .. } => {
                // `[value; n]` evaluates `value` once and fills every slot with it. Small arrays
                // unroll to straight-line stores; large ones lower to a fill loop so that, e.g.,
                // `[0; 1_000_000]` does not generate a million instructions.
                let v0 = self.lower_expr(value);
                let vty = self.expr_mir(value);
                let v = self.coerce_to(v0, &vty, elem, self.signed(value));
                if n <= REPEAT_UNROLL_LIMIT {
                    for i in 0..n as i128 {
                        self.store_element(base, elem, i, v);
                    }
                } else {
                    self.lower_fill_loop(base, elem, n, v);
                }
            }
            _ => {
                self.unsupported(init.span, "array initializer (expected `[..]` or `[v; n]`)");
            }
        }
    }

    /// Emit `for i in 0..n { base[i] = v }` as a CFG loop. Used for large array-repeat initializers
    /// so the IR stays compact; mem2reg later promotes the loop counter to a register.
    fn lower_fill_loop(&mut self, base: ValueId, elem: &MirType, n: u32, v: ValueId) {
        let i64t = MirType::I64;
        let slot = self.builder.alloca(i64t.clone());
        let zero = self
            .builder
            .build(i64t.clone(), Op::ConstInt(0, i64t.clone()));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: zero,
        });

        let header = self.builder.new_block();
        let body = self.builder.new_block();
        let exit = self.builder.new_block();
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        let i_val = self
            .builder
            .build(i64t.clone(), Op::Load(slot, i64t.clone()));
        let nconst = self
            .builder
            .build(i64t.clone(), Op::ConstInt(n as i128, i64t.clone()));
        let cond = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, i_val, nconst));
        self.builder.cond_br(cond, body, vec![], exit, vec![]);

        self.builder.switch_to(body);
        let i_cur = self
            .builder
            .build(i64t.clone(), Op::Load(slot, i64t.clone()));
        let p = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: i_cur,
                elem: elem.clone(),
            },
        );
        self.builder.build_void(Op::Store { ptr: p, value: v });
        let one = self
            .builder
            .build(i64t.clone(), Op::ConstInt(1, i64t.clone()));
        let next = self
            .builder
            .build(i64t.clone(), Op::Bin(BinOp::Add, i_cur, one));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: next,
        });
        self.builder.br(header, vec![]);

        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// Store `value` into `base[index]` for an array element of type `elem`.
    fn store_element(&mut self, base: ValueId, elem: &MirType, index: i128, value: ValueId) {
        let idx = self
            .builder
            .build(MirType::I64, Op::ConstInt(index, MirType::I64));
        let p = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: elem.clone(),
            },
        );
        self.builder.build_void(Op::Store { ptr: p, value });
    }

    /// Row-major element strides for a tensor `base`, if computable. `stride_k` = product of the
    /// dims *after* position `k`; computable when every dim after the first is a compile-time
    /// `Const` (the leading dim may be `Var`/`Dynamic` — it never contributes to a stride). Returns
    /// the per-axis strides and the element's MIR type, or `None` (caller falls back to scalar/error).
    /// This is what makes the shape-typed surface — `a[i, j]` on `Tensor[f32, M, N]` — executable.
    fn tensor_strides(&self, base: &Expr) -> Option<(Vec<usize>, MirType)> {
        let Ty::Tensor {
            elem,
            shape,
            layout,
        } = self.expr_ty(base)
        else {
            return None;
        };
        // Only row-major (contiguous) tensors flatten to `i0*s0 + … + i_{n-1}` with these strides.
        if !matches!(layout, mercury_types::Layout::Contiguous) {
            return None;
        }
        let dims = &shape.0;
        let rank = dims.len();
        if rank == 0 {
            return None;
        }
        let mut strides = vec![1usize; rank];
        for k in (0..rank - 1).rev() {
            let next = match dims[k + 1] {
                mercury_types::Dim::Const(d) => d as usize,
                _ => return None, // a symbolic/dynamic interior dim — stride unknown at compile time
            };
            strides[k] = strides[k + 1] * next;
        }
        Some((strides, mir_ty(&Ty::Scalar(elem))))
    }

    /// Lower an index expression and widen it to `I64`, so a multi-dimensional flat-offset
    /// computation is single-typed regardless of the index's source width (loop vars are usually
    /// `i32`). Non-negative loop counters, so sign/zero-extension agree; we pick by signedness.
    fn lower_index_i64(&mut self, e: &Expr) -> ValueId {
        let v = self.lower_expr(e);
        let ty = self.expr_mir(e);
        if ty == MirType::I64 {
            return v;
        }
        let kind = if self.signed(e) {
            CastKind::SExt
        } else {
            CastKind::ZExt
        };
        self.builder
            .build(MirType::I64, Op::Cast(kind, v, MirType::I64))
    }

    /// Lower a multi-dimensional tensor index `base[i0, i1, …]` to a single `Gep` at the row-major
    /// flat element offset `Σ iₖ·strideₖ`. Returns `None` if `base` is not a flattenable tensor.
    fn lower_multi_index(&mut self, base: &Expr, indices: &[Expr]) -> Option<(ValueId, MirType)> {
        let (strides, elem) = self.tensor_strides(base)?;
        if strides.len() != indices.len() {
            return None; // rank mismatch (sema already reported E0501); fall back to unsupported
        }
        let base_ptr = self.lower_expr(base);
        let mut flat: Option<ValueId> = None;
        for (ix, &st) in indices.iter().zip(strides.iter()) {
            let iv = self.lower_index_i64(ix);
            let term = if st == 1 {
                iv
            } else {
                let s = self
                    .builder
                    .build(MirType::I64, Op::ConstInt(st as i128, MirType::I64));
                self.builder.build(MirType::I64, Op::Bin(BinOp::Mul, iv, s))
            };
            flat = Some(match flat {
                None => term,
                Some(acc) => self
                    .builder
                    .build(MirType::I64, Op::Bin(BinOp::Add, acc, term)),
            });
        }
        let flat = flat.expect("indices non-empty");
        let p = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: base_ptr,
                index: flat,
                elem: elem.clone(),
            },
        );
        Some((p, elem))
    }

    fn lower_place(&mut self, e: &Expr) -> (ValueId, MirType) {
        match &e.kind {
            ExprKind::Path(p) if p.is_single() => {
                if let Some((slot, ty)) = self.lookup(p.first().sym) {
                    return (slot, ty);
                }
                self.unsupported(p.span, "assignment to this name");
                let ty = self.expr_mir(e);
                (self.builder.alloca(ty.clone()), ty)
            }
            ExprKind::Unary {
                op: ast::UnOp::Deref,
                expr,
            } => {
                let ptr = self.lower_expr(expr);
                (ptr, self.expr_mir(e))
            }
            ExprKind::Index { base, indices } if indices.len() == 1 => {
                let base_ptr = self.lower_expr(base);
                let idx = self.lower_expr(&indices[0]);
                // Prefer the element type from the base's array type; fall back to the indexed
                // expression's own type (slices/tensors/pointers).
                let elem = match self.expr_ty(base) {
                    Ty::Array { elem, .. } => mir_ty(&elem),
                    _ => self.expr_mir(e),
                };
                let p = self.builder.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: base_ptr,
                        index: idx,
                        elem: elem.clone(),
                    },
                );
                (p, elem)
            }
            // Multi-dimensional tensor indexing `t[i, j, …]` — the shape-typed surface. Flatten to a
            // row-major offset using the tensor's static strides.
            ExprKind::Index { base, indices } if indices.len() >= 2 => {
                if let Some(pe) = self.lower_multi_index(base, indices) {
                    return pe;
                }
                self.unsupported(
                    e.span,
                    "multi-dimensional indexing on a non-contiguous or symbolically-strided tensor",
                );
                let ty = self.expr_mir(e);
                (self.builder.alloca(ty.clone()), ty)
            }
            _ => {
                self.unsupported(e.span, "assignment target");
                let ty = self.expr_mir(e);
                (self.builder.alloca(ty.clone()), ty)
            }
        }
    }

    // ---- expressions ----

    fn lower_expr(&mut self, e: &Expr) -> ValueId {
        match &e.kind {
            ExprKind::Int(s) => {
                let ty = self.expr_mir(e);
                let v = parse_int(self.interner.resolve(*s));
                let ty = if ty.is_int() { ty } else { MirType::I32 };
                self.builder.build(ty.clone(), Op::ConstInt(v, ty))
            }
            ExprKind::Float(s) => {
                let ty = self.expr_mir(e);
                let v = parse_float(self.interner.resolve(*s));
                let ty = if ty.is_float() { ty } else { MirType::F32 };
                self.builder.build(ty.clone(), Op::ConstFloat(v, ty))
            }
            ExprKind::Bool(b) => self
                .builder
                .build(MirType::I1, Op::ConstInt(*b as i128, MirType::I1)),
            ExprKind::Path(p) if p.is_single() => {
                if let Some((slot, ty)) = self.lookup(p.first().sym) {
                    // An array variable *is* its storage: its value is the base pointer, so reads
                    // don't load — indexing geps off this pointer.
                    if matches!(ty, MirType::Array(..)) {
                        slot
                    } else {
                        self.builder.build(ty.clone(), Op::Load(slot, ty))
                    }
                } else {
                    self.unsupported(p.span, "value reference");
                    let t = self.expr_mir(e);
                    self.const_zero(t)
                }
            }
            ExprKind::Unary { op, expr } => self.lower_unary(*op, expr, e),
            ExprKind::Binary { op, lhs, rhs } => self.lower_binary(*op, lhs, rhs, e),
            ExprKind::Call { callee, args, .. } => self.lower_call(callee, args, e),
            ExprKind::Index { base, indices } if !indices.is_empty() => {
                let (ptr, elem) = self.lower_place(e);
                let _ = (base, indices);
                self.builder.build(elem.clone(), Op::Load(ptr, elem))
            }
            ExprKind::Cast { expr, .. } => self.lower_cast(expr, e),
            ExprKind::Block(b) => match self.lower_block(b) {
                Some(v) => v,
                None => {
                    let t = self.expr_mir(e);
                    self.const_zero(t)
                }
            },
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => self.lower_if_value(cond, then_branch, else_branch.as_deref(), e),
            _ => {
                self.unsupported(e.span, "expression");
                let t = self.expr_mir(e);
                self.const_zero(t)
            }
        }
    }

    fn lower_unary(&mut self, op: ast::UnOp, operand: &Expr, e: &Expr) -> ValueId {
        match op {
            ast::UnOp::Neg => {
                let v = self.lower_expr(operand);
                let ty = self.expr_mir(e);
                self.builder.build(ty, Op::Neg(v))
            }
            ast::UnOp::Not => {
                let v = self.lower_expr(operand);
                let ty = self.expr_mir(e);
                self.builder.build(ty, Op::Not(v))
            }
            ast::UnOp::Deref => {
                let ptr = self.lower_expr(operand);
                let ty = self.expr_mir(e);
                self.builder.build(ty.clone(), Op::Load(ptr, ty))
            }
            ast::UnOp::Ref | ast::UnOp::RefMut => {
                let (ptr, _) = self.lower_place(operand);
                ptr
            }
        }
    }

    fn lower_binary(&mut self, op: ast::BinOp, lhs: &Expr, rhs: &Expr, e: &Expr) -> ValueId {
        use ast::BinOp::*;
        match op {
            Eq | Ne | Lt | Le | Gt | Ge => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                // Compare in a common type: the front-end's loose literal typing can leave the two
                // sides at different widths, but a `Cmp`'s operands must agree.
                let lty = self.expr_mir(lhs);
                let rty = self.expr_mir(rhs);
                let common = numeric_join(&lty, &rty);
                let l = self.coerce_to(l, &lty, &common, self.signed(lhs));
                let r = self.coerce_to(r, &rty, &common, self.signed(rhs));
                let pred = cmp_pred(op, common.is_float(), self.signed(lhs));
                self.builder.build(MirType::I1, Op::Cmp(pred, l, r))
            }
            And => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                self.builder.build(MirType::I1, Op::Bin(BinOp::And, l, r))
            }
            Or => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                self.builder.build(MirType::I1, Op::Bin(BinOp::Or, l, r))
            }
            _ => {
                let ty = self.expr_mir(e);
                // Contract a float `x + y*z` into one fused multiply-add before falling back to a
                // plain `Bin`.
                if let Some(v) = self.try_contract_fma(op, lhs, rhs, &ty) {
                    return v;
                }
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                // Coerce both operands to the result type so the `Bin` is well-typed (e.g. an
                // `f32` literal added to an `f64` is promoted), matching the verifier's contract.
                let lty = self.expr_mir(lhs);
                let rty = self.expr_mir(rhs);
                let l = self.coerce_to(l, &lty, &ty, self.signed(lhs));
                let r = self.coerce_to(r, &rty, &ty, self.signed(rhs));
                let bin = arith_binop(op, ty.is_float(), self.signed(lhs));
                self.builder.build(ty, Op::Bin(bin, l, r))
            }
        }
    }

    /// Contract a float `x + y*z` (or `y*z + x`) into one fused multiply-add. FMA rounds once
    /// instead of twice — faster (a single `vfmadd`) and more accurate — and the interpreter
    /// mirrors it with `mul_add`, so the native and interpreter backends stay bit-identical.
    /// Returns `None` when the shape or types don't permit it; the caller then emits a plain add.
    fn try_contract_fma(
        &mut self,
        op: ast::BinOp,
        lhs: &Expr,
        rhs: &Expr,
        ty: &MirType,
    ) -> Option<ValueId> {
        if op != ast::BinOp::Add || !ty.is_float() {
            return None;
        }
        // A local `fn` (not a closure) so the borrow of the returned sub-exprs ties to the
        // argument's lifetime rather than a single inferred one.
        fn as_fmul(e: &Expr) -> Option<(&Expr, &Expr)> {
            match &e.kind {
                ExprKind::Binary {
                    op: ast::BinOp::Mul,
                    lhs,
                    rhs,
                } => Some((lhs.as_ref(), rhs.as_ref())),
                _ => None,
            }
        }
        // Lower operands in source order so any side effects keep their original sequencing.
        if let Some((y, z)) = as_fmul(lhs) {
            let yv = self.lower_fma_operand(y, ty);
            let zv = self.lower_fma_operand(z, ty);
            let xv = self.lower_fma_operand(rhs, ty);
            return Some(self.builder.build(ty.clone(), Op::Fma(yv, zv, xv)));
        }
        if let Some((y, z)) = as_fmul(rhs) {
            let xv = self.lower_fma_operand(lhs, ty);
            let yv = self.lower_fma_operand(y, ty);
            let zv = self.lower_fma_operand(z, ty);
            return Some(self.builder.build(ty.clone(), Op::Fma(yv, zv, xv)));
        }
        None
    }

    /// Lower an FMA operand and coerce it to the (float) result type.
    fn lower_fma_operand(&mut self, e: &Expr, ty: &MirType) -> ValueId {
        let v = self.lower_expr(e);
        let ety = self.expr_mir(e);
        self.coerce_to(v, &ety, ty, self.signed(e))
    }

    /// Insert a numeric cast so `v` (currently `from`) has type `to`. Non-numeric operands and
    /// equal types pass through unchanged.
    fn coerce_to(&mut self, v: ValueId, from: &MirType, to: &MirType, signed: bool) -> ValueId {
        if from == to || !is_numeric(from) || !is_numeric(to) {
            return v;
        }
        let kind = cast_kind(from, to, signed);
        self.builder
            .build(to.clone(), Op::Cast(kind, v, to.clone()))
    }

    fn lower_call(&mut self, callee: &Expr, args: &[Expr], e: &Expr) -> ValueId {
        if let ExprKind::Path(p) = &callee.kind {
            if p.is_single() {
                let name = p.first().sym;
                if matches!(
                    self.sema.defs.lookup(name).map(|d| &d.kind),
                    Some(DefKind::Fn(_))
                ) {
                    let argvals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                    let ret = self.expr_mir(e);
                    if ret == MirType::Void {
                        self.builder.build_void(Op::Call {
                            func: name,
                            args: argvals,
                        });
                        return self.const_zero(MirType::I32);
                    }
                    return self.builder.build(
                        ret,
                        Op::Call {
                            func: name,
                            args: argvals,
                        },
                    );
                }
                // Math builtins (sqrt/rsqrt/exp/fmax/fmin) lower to primitive ops. User functions
                // shadow them (handled just above), so this catches only the genuine builtins.
                if let Some(v) = self.lower_math_intrinsic(name, args, e) {
                    return v;
                }
                // Built-in intrinsics (print, ...) lower to a void call the interpreter handles.
                if is_intrinsic(self.interner.resolve(name)) {
                    let argvals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                    self.builder.build_void(Op::Call {
                        func: name,
                        args: argvals,
                    });
                    return self.const_zero(MirType::I32);
                }
            }
        }
        // Unmodeled builtin/method call.
        for a in args {
            self.lower_expr(a);
        }
        self.unsupported(e.span, "call");
        let t = self.expr_mir(e);
        self.const_zero(t)
    }

    /// Lower a math builtin (`sqrt`/`rsqrt`/`exp`/`fmax`/`fmin`) to primitive MIR ops, or `None`
    /// for any other name. `sqrt` is one `Op::Sqrt`; `rsqrt` is its reciprocal; `fmax`/`fmin` are a
    /// compare + select (NaN- and signed-zero-correct, identical in both backends); `exp` is a
    /// polynomial (see `emit_exp`). Built only from already-bit-exact primitives, so the
    /// interpreter and native backend agree with no separate hand-written implementation each.
    fn lower_math_intrinsic(&mut self, name: Symbol, args: &[Expr], e: &Expr) -> Option<ValueId> {
        let op = math_intrinsic(self.interner.resolve(name))?;
        let rty = self.expr_mir(e);
        match op {
            MathIntrinsic::Sqrt => {
                let x = self.lower_expr(args.first()?);
                Some(self.builder.build(rty.clone(), Op::Sqrt(x)))
            }
            MathIntrinsic::Rsqrt => {
                let x = self.lower_expr(args.first()?);
                let s = self.builder.build(rty.clone(), Op::Sqrt(x));
                let one = self.splat_const_f(1.0, &rty);
                Some(
                    self.builder
                        .build(rty.clone(), Op::Bin(BinOp::FDiv, one, s)),
                )
            }
            MathIntrinsic::Exp => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_exp(x, &rty))
            }
            MathIntrinsic::Log => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_log(x, &rty))
            }
            MathIntrinsic::Pow => {
                // pow(x, y) = exp(y * log(x)), reusing the two polynomials (so it vectorizes and is
                // bit-exact across backends for free). Defined for x > 0, like the rest of the suite.
                if args.len() != 2 {
                    return None;
                }
                let x = self.lower_expr(&args[0]);
                let y = self.lower_expr(&args[1]);
                let lx = self.emit_log(x, &rty);
                let ylx = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, y, lx));
                Some(self.emit_exp(ylx, &rty))
            }
            MathIntrinsic::Erf => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_erf(x, &rty))
            }
            MathIntrinsic::Sin => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_trig(x, &rty, false))
            }
            MathIntrinsic::Cos => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_trig(x, &rty, true))
            }
            MathIntrinsic::Tanh => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_tanh(x, &rty))
            }
            MathIntrinsic::Sigmoid => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_sigmoid(x, &rty))
            }
            MathIntrinsic::Silu => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_silu(x, &rty))
            }
            MathIntrinsic::Gelu => {
                let x = self.lower_expr(args.first()?);
                Some(self.emit_gelu(x, &rty))
            }
            MathIntrinsic::Fmax | MathIntrinsic::Fmin => {
                if args.len() != 2 {
                    return None;
                }
                let a = self.lower_expr(&args[0]);
                let b = self.lower_expr(&args[1]);
                let pred = if matches!(op, MathIntrinsic::Fmax) {
                    CmpOp::Fogt
                } else {
                    CmpOp::Folt
                };
                let c = self.builder.build(mask_ty(&rty), Op::Cmp(pred, a, b));
                Some(self.builder.build(rty.clone(), Op::Select(c, a, b)))
            }
        }
    }

    /// `exp(x)` as a fast, deterministic polynomial (≈1 ULP of the true `exp`). Always computed in
    /// `f32`; for an `f64` result the argument is demoted and the result promoted (exact in both
    /// backends). Works on a scalar or a SIMD-vector `f32`.
    fn emit_exp(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_exp_f32(xf, &f32ty);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// `sigmoid(x) = 1 / (1 + exp(-x))`, built on the exp polynomial. Works on a scalar or a SIMD
    /// vector; bit-identical across backends because every step is (see `emit_exp`).
    fn emit_sigmoid(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let neg1 = self.splat_const_f(-1.0, rty);
        let negx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, x, neg1));
        let e = self.emit_exp(negx, rty);
        let one = self.splat_const_f(1.0, rty);
        let denom = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, e));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FDiv, one, denom))
    }

    /// `tanh(x) = 1 - 2 / (exp(2x) + 1)`, built on the exp polynomial (same identity the GELU tanh
    /// approximation uses). Works on a scalar or a SIMD vector; bit-identical across backends.
    fn emit_tanh(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let two = self.splat_const_f(2.0, rty);
        let twox = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, x, two));
        let e = self.emit_exp(twox, rty);
        let one = self.splat_const_f(1.0, rty);
        let denom = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, e, one));
        let frac = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FDiv, two, denom));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FSub, one, frac))
    }

    /// `silu(x) = x · sigmoid(x)` (swish). The scalar/composed path; an `out[i] = silu(x[i])` loop
    /// dispatches to the fused AVX2 kernel instead. Bit-identical across backends (see `emit_exp`).
    fn emit_silu(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let s = self.emit_sigmoid(x, rty);
        self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, s))
    }

    /// `gelu(x)` (tanh approximation): `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. Mirrors the
    /// fused AVX2 `gelu8` op-for-op (so the scalar form and the dispatched loop form agree). The
    /// BERT/GPT-2/ViT activation; bit-identical across backends.
    fn emit_gelu(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let x2 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x, x));
        let x3 = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, x2, x));
        let c1 = self.splat_const_f(0.044715, rty);
        let t = self.builder.build(rty.clone(), Op::Fma(c1, x3, x)); // 0.044715·x³ + x
        let c0 = self.splat_const_f(0.7978845608, rty);
        let inner = self.builder.build(rty.clone(), Op::Bin(BinOp::FMul, c0, t));
        let th = self.emit_tanh(inner, rty);
        let one = self.splat_const_f(1.0, rty);
        let onep = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FAdd, one, th));
        let half = self.splat_const_f(0.5, rty);
        let hx = self
            .builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, half, x));
        self.builder
            .build(rty.clone(), Op::Bin(BinOp::FMul, hx, onep))
    }

    /// `erf(x)` (the Gauss error function — `exact` GELU is `0.5·x·(1 + erf(x/√2))`). Always computed
    /// in `f32`; an `f64` result is demoted/promoted like `exp`/`log`. Works on a scalar or a SIMD
    /// vector; bit-identical across backends because every step is a primitive op (see `emit_erf_f32`).
    fn emit_erf(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_erf_f32(xf, &f32ty);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// The erf approximation in `f32` (Abramowitz–Stegun 7.1.26, ~1.5e-7 error). `|x|` and the
    /// odd-function sign are handled with compare/select (no IEEE bit surgery), and `e^(-x²)` reuses
    /// the exp polynomial — so it vectorizes (`fty` may be a `Vec` of `f32`) and both backends agree.
    fn emit_erf_f32(&mut self, x: ValueId, fty: &MirType) -> ValueId {
        let mty = mask_ty(fty);
        let one = self.splat_const_f(1.0, fty);
        let neg1 = self.splat_const_f(-1.0, fty);
        // ax = |x| = max(x, -x), via compare + select.
        let negx = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, x, neg1));
        let gt = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Fogt, x, negx));
        let ax = self.builder.build(fty.clone(), Op::Select(gt, x, negx));
        // t = 1 / (1 + P*ax)
        let p = self.splat_const_f(ERF_P, fty);
        let denom = self.builder.build(fty.clone(), Op::Fma(p, ax, one));
        let t = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FDiv, one, denom));
        // poly(t) = t·(a₁ + t·(a₂ + t·(a₃ + t·(a₄ + t·a₅)))) by Horner with fmas.
        let mut h = self.splat_const_f(ERF_A[4], fty);
        for &c in ERF_A[..4].iter().rev() {
            let cc = self.splat_const_f(c, fty);
            h = self.builder.build(fty.clone(), Op::Fma(h, t, cc));
        }
        let poly = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, h, t));
        // e = exp(-ax²)
        let ax2 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, ax, ax));
        let neg_ax2 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, ax2, neg1));
        let e = self.emit_exp_f32(neg_ax2, fty);
        // erf(|x|) = 1 - poly·e
        let pe = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, poly, e));
        let mag = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, one, pe));
        // erf is odd: result = (x >= 0) ? mag : -mag.
        let neg_mag = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, mag, neg1));
        let zero = self.splat_const_f(0.0, fty);
        let ge = self.builder.build(mty, Op::Cmp(CmpOp::Foge, x, zero));
        self.builder
            .build(fty.clone(), Op::Select(ge, mag, neg_mag))
    }

    /// `sin(x)` (`is_cos == false`) or `cos(x)` (`true`) as a fast, deterministic polynomial. Always
    /// computed in `f32`; an `f64` result is demoted/promoted like `exp`. Works on a scalar or a SIMD
    /// vector; bit-identical across backends. Enables rotary position embeddings (RoPE).
    fn emit_trig(&mut self, x: ValueId, rty: &MirType, is_cos: bool) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_trig_f32(xf, &f32ty, is_cos);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// The sin/cos polynomial in `f32` (`fty` is `f32` or a `Vec` of `f32`). Reduces `x` to
    /// `r ∈ [-π/4, π/4]` by `q = round(x·2/π)` quadrants (round-to-nearest via the add-magic trick),
    /// evaluates the Cephes `sinf`/`cosf` minimax polynomials on `r`, and selects ±sin/±cos by
    /// `q mod 4`. The quadrant is reduced to an exact small float so every blend uses a float-compare
    /// mask (the proven `select` path), and every step is a primitive op, so both backends agree.
    fn emit_trig_f32(&mut self, x: ValueId, fty: &MirType, is_cos: bool) -> ValueId {
        let ity = match fty {
            MirType::Vec(_, n) => MirType::Vec(Box::new(MirType::I32), *n),
            _ => MirType::I32,
        };
        let mty = mask_ty(fty);
        // q = round(x · 2/π) via add-magic / sub-magic (round-to-nearest-even, pure f32).
        let two_pi = self.splat_const_f(TWO_OVER_PI, fty);
        let magic = self.splat_const_f(EXP_MAGIC, fty);
        let tt = self.builder.build(fty.clone(), Op::Fma(x, two_pi, magic));
        let qf = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, tt, magic));
        // r = ((x - qf·PIO2_1) - qf·PIO2_2) - qf·PIO2_3   (3-part π/2 split keeps the reduction exact).
        let n1 = self.splat_const_f(-PIO2_1, fty);
        let n2 = self.splat_const_f(-PIO2_2, fty);
        let n3 = self.splat_const_f(-PIO2_3, fty);
        let r = self.builder.build(fty.clone(), Op::Fma(qf, n1, x));
        let r = self.builder.build(fty.clone(), Op::Fma(qf, n2, r));
        let r = self.builder.build(fty.clone(), Op::Fma(qf, n3, r));
        let z = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, r, r));
        // sin_p(r) = r·(1 + z·poly) with poly = ((s₀z + s₁)z + s₂).
        let mut s = self.splat_const_f(SIN_P[0], fty);
        for &c in &SIN_P[1..] {
            let cc = self.splat_const_f(c, fty);
            s = self.builder.build(fty.clone(), Op::Fma(s, z, cc));
        }
        let sz = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, s, z));
        let sin_p = self.builder.build(fty.clone(), Op::Fma(sz, r, r));
        // cos_p(r) = (1 - 0.5z) + z²·poly with poly = ((c₀z + c₁)z + c₂).
        let mut cc0 = self.splat_const_f(COS_P[0], fty);
        for &c in &COS_P[1..] {
            let k = self.splat_const_f(c, fty);
            cc0 = self.builder.build(fty.clone(), Op::Fma(cc0, z, k));
        }
        let z2 = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, z, z));
        let cz2 = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, cc0, z2));
        let neg_half = self.splat_const_f(-0.5, fty);
        let one = self.splat_const_f(1.0, fty);
        let hz = self.builder.build(fty.clone(), Op::Fma(neg_half, z, one));
        let cos_p = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FAdd, hz, cz2));
        // ±variants for the quadrant blend.
        let neg1 = self.splat_const_f(-1.0, fty);
        let neg_sin = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, sin_p, neg1));
        let neg_cos = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, cos_p, neg1));
        // quad = (int)qf & 3, then back to an exact float (0/1/2/3) for float-mask selects. For
        // negative q the two's-complement `& 3` is still the correct mod-4 quadrant.
        let qi = self
            .builder
            .build(ity.clone(), Op::Cast(CastKind::FpToSi, qf, ity.clone()));
        let three = self.splat_const_i(3, &ity);
        let quad_i = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::And, qi, three));
        let quad = self
            .builder
            .build(fty.clone(), Op::Cast(CastKind::SiToFp, quad_i, fty.clone()));
        // sin: [sin, cos, -sin, -cos];  cos: [cos, -sin, -cos, sin], indexed by quad.
        let (a0, a1, a2, a3) = if is_cos {
            (cos_p, neg_sin, neg_cos, sin_p)
        } else {
            (sin_p, cos_p, neg_sin, neg_cos)
        };
        let k0 = self.splat_const_f(0.0, fty);
        let k1 = self.splat_const_f(1.0, fty);
        let k2 = self.splat_const_f(2.0, fty);
        let eq0 = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Foeq, quad, k0));
        let eq1 = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Foeq, quad, k1));
        let eq2 = self.builder.build(mty, Op::Cmp(CmpOp::Foeq, quad, k2));
        let sel23 = self.builder.build(fty.clone(), Op::Select(eq2, a2, a3));
        let sel123 = self.builder.build(fty.clone(), Op::Select(eq1, a1, sel23));
        self.builder.build(fty.clone(), Op::Select(eq0, a0, sel123))
    }

    /// The exp polynomial in `f32` (`fty` is `f32` or a `Vec` of `f32`). Range-reduces `x` to
    /// `r = x - n*ln2`, evaluates a degree-5 minimax poly for `e^r`, then scales by `2^n` (assembled
    /// from the IEEE-754 exponent bits). Every step is a primitive op the two backends already agree
    /// on bit-for-bit, so `exp` does too.
    fn emit_exp_f32(&mut self, x: ValueId, fty: &MirType) -> ValueId {
        let lanes = match fty {
            MirType::Vec(_, n) => Some(*n),
            _ => None,
        };
        let ity = match lanes {
            Some(n) => MirType::Vec(Box::new(MirType::I32), n),
            None => MirType::I32,
        };
        let mty = mask_ty(fty);

        // Clamp so 2^n stays representable (exp under/overflows to 0 / +inf outside this range).
        let hi = self.splat_const_f(EXP_HI, fty);
        let gt = self.builder.build(mty.clone(), Op::Cmp(CmpOp::Fogt, x, hi));
        let x = self.builder.build(fty.clone(), Op::Select(gt, hi, x));
        let lo = self.splat_const_f(EXP_LO, fty);
        let lt = self.builder.build(mty.clone(), Op::Cmp(CmpOp::Folt, x, lo));
        let x = self.builder.build(fty.clone(), Op::Select(lt, lo, x));

        // n = round(x * log2(e)) via the add-magic / sub-magic trick: round-to-nearest-even in
        // pure f32, so both backends agree and no rounding-mode instruction is needed.
        let log2e = self.splat_const_f(LOG2EF, fty);
        let magic = self.splat_const_f(EXP_MAGIC, fty);
        let t = self.builder.build(fty.clone(), Op::Fma(x, log2e, magic));
        let n = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, t, magic));

        // r = x - n*ln2, with ln2 split into hi/lo parts for extra precision (two fmas).
        let neg_c1 = self.splat_const_f(-EXP_C1, fty);
        let neg_c2 = self.splat_const_f(-EXP_C2, fty);
        let r = self.builder.build(fty.clone(), Op::Fma(n, neg_c1, x));
        let r = self.builder.build(fty.clone(), Op::Fma(n, neg_c2, r));

        // Degree-5 minimax polynomial for e^r on the reduced range, by Horner with fmas.
        let mut p = self.splat_const_f(EXP_P[0], fty);
        for &c in &EXP_P[1..] {
            let cc = self.splat_const_f(c, fty);
            p = self.builder.build(fty.clone(), Op::Fma(p, r, cc));
        }
        // e^r = p*r^2 + r + 1
        let r2 = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, r, r));
        let p = self.builder.build(fty.clone(), Op::Fma(p, r2, r));
        let one = self.splat_const_f(1.0, fty);
        let p = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FAdd, p, one));

        // 2^n by assembling the IEEE-754 exponent field: bitcast((n + 127) << 23).
        let ni = self
            .builder
            .build(ity.clone(), Op::Cast(CastKind::FpToSi, n, ity.clone()));
        let bias = self.splat_const_i(127, &ity);
        let biased = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, ni, bias));
        // `<< 23` written as `* 2^23`: keeps both Bin operands the same (vector) type. Cranelift's
        // vector `ishl` requires a *scalar* shift amount, but `imul` takes two vectors; since
        // `n + 127 <= 254` the product never overflows i32, so it equals the shift bit-for-bit.
        let pow = self.splat_const_i(8_388_608, &ity);
        let shifted = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Mul, biased, pow));
        let pow2 = self.builder.build(
            fty.clone(),
            Op::Cast(CastKind::Bitcast, shifted, fty.clone()),
        );

        self.builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, p, pow2))
    }

    /// `log(x)` (natural log) as a fast, deterministic polynomial (≈1 ULP of the true `log`). Always
    /// computed in `f32`; for an `f64` result the argument is demoted and the result promoted (the
    /// same f32-precision model `exp` uses). Works on a scalar or a SIMD-vector `f32`.
    fn emit_log(&mut self, x: ValueId, rty: &MirType) -> ValueId {
        let want_f64 = matches!(rty.lane_type(), MirType::F64);
        let f32ty = float_ty_like(rty, MirType::F32);
        let xf = if want_f64 {
            self.builder
                .build(f32ty.clone(), Op::Cast(CastKind::FpTrunc, x, f32ty.clone()))
        } else {
            x
        };
        let r = self.emit_log_f32(xf, &f32ty);
        if want_f64 {
            self.builder
                .build(rty.clone(), Op::Cast(CastKind::FpExt, r, rty.clone()))
        } else {
            r
        }
    }

    /// The natural-log polynomial in `f32` (`fty` is `f32` or a `Vec` of `f32`). Decomposes
    /// `x = m·2^e` with `m ∈ [0.5,1)` by IEEE-754 bit surgery — **no shift**: the exponent field is
    /// masked off (so the integer is `exp_field·2^23`, exact in `f32` for the ≤8-bit field), widened
    /// and scaled by `2^-23` to recover the count; the mantissa is OR-ed with biased exponent 126.
    /// Then a Cephes degree-8 minimax poly gives `log(m)`, and `e·ln2` is added back (same `ln2`
    /// split as `exp`). Every step is a primitive op both backends agree on bit-for-bit, so `log`
    /// does too. Assumes `x > 0` (like the rest of the kernels, no domain guard).
    fn emit_log_f32(&mut self, x: ValueId, fty: &MirType) -> ValueId {
        let lanes = match fty {
            MirType::Vec(_, n) => Some(*n),
            _ => None,
        };
        let ity = match lanes {
            Some(n) => MirType::Vec(Box::new(MirType::I32), n),
            None => MirType::I32,
        };
        let mty = mask_ty(fty);

        let bits = self
            .builder
            .build(ity.clone(), Op::Cast(CastKind::Bitcast, x, ity.clone()));

        // e = (float)(exponent_field) - 126, without a shift: keep only the exponent bits (value is
        // `exp_field·2^23`, exact in f32), widen to f32, scale by 2^-23.
        let expmask = self.splat_const_i(0x7F80_0000, &ity);
        let epart = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::And, bits, expmask));
        let epart_f = self
            .builder
            .build(fty.clone(), Op::Cast(CastKind::SiToFp, epart, fty.clone()));
        let inv = self.splat_const_f(INV_2P23, fty);
        let efield = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FMul, epart_f, inv));
        let bias = self.splat_const_f(126.0, fty);
        let mut e = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, efield, bias));

        // m = bitcast((bits & 0x007fffff) | 0x3f000000) — mantissa with biased exponent 126 → [0.5,1).
        let mmask = self.splat_const_i(0x007F_FFFF, &ity);
        let mant = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::And, bits, mmask));
        let half_exp = self.splat_const_i(0x3F00_0000, &ity);
        let mbits = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Or, mant, half_exp));
        let mut m = self
            .builder
            .build(fty.clone(), Op::Cast(CastKind::Bitcast, mbits, fty.clone()));

        // if m < √0.5: e -= 1; m = 2m - 1; else m -= 1  (branchless via select).
        let sqrthf = self.splat_const_f(LOG_SQRTHF, fty);
        let lt = self
            .builder
            .build(mty.clone(), Op::Cmp(CmpOp::Folt, m, sqrthf));
        let one = self.splat_const_f(1.0, fty);
        let m2 = self.builder.build(fty.clone(), Op::Bin(BinOp::FAdd, m, m));
        let m_lt = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, m2, one)); // 2m - 1
        let m_ge = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, m, one)); // m - 1
        m = self.builder.build(fty.clone(), Op::Select(lt, m_lt, m_ge));
        let e_dec = self
            .builder
            .build(fty.clone(), Op::Bin(BinOp::FSub, e, one)); // e - 1
        e = self.builder.build(fty.clone(), Op::Select(lt, e_dec, e));

        // Degree-8 minimax poly for log(m) on the reduced range, Horner via fmas, then × m × z.
        let z = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, m, m));
        let mut p = self.splat_const_f(LOG_P[0], fty);
        for &c in &LOG_P[1..] {
            let cc = self.splat_const_f(c, fty);
            p = self.builder.build(fty.clone(), Op::Fma(p, m, cc));
        }
        let pm = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, p, m));
        let mut y = self.builder.build(fty.clone(), Op::Bin(BinOp::FMul, pm, z));

        // y += e·C2 (ln2 low);  y -= 0.5·z
        let c2 = self.splat_const_f(EXP_C2, fty);
        y = self.builder.build(fty.clone(), Op::Fma(e, c2, y));
        let neg_half = self.splat_const_f(-0.5, fty);
        y = self.builder.build(fty.clone(), Op::Fma(z, neg_half, y));

        // r = m + y + e·C1 (ln2 high)
        let r = self.builder.build(fty.clone(), Op::Bin(BinOp::FAdd, m, y));
        let c1 = self.splat_const_f(EXP_C1, fty);
        self.builder.build(fty.clone(), Op::Fma(e, c1, r))
    }

    /// Build a float constant of type `fty` — a scalar `ConstFloat`, or one splatted to a vector.
    fn splat_const_f(&mut self, val: f64, fty: &MirType) -> ValueId {
        match fty {
            MirType::Vec(lane, _) => {
                let s = self
                    .builder
                    .build((**lane).clone(), Op::ConstFloat(val, (**lane).clone()));
                self.builder.build(fty.clone(), Op::Splat(s))
            }
            _ => self
                .builder
                .build(fty.clone(), Op::ConstFloat(val, fty.clone())),
        }
    }

    /// Build an int constant of type `ity` — a scalar `ConstInt`, or one splatted to a vector.
    fn splat_const_i(&mut self, val: i128, ity: &MirType) -> ValueId {
        match ity {
            MirType::Vec(lane, _) => {
                let s = self
                    .builder
                    .build((**lane).clone(), Op::ConstInt(val, (**lane).clone()));
                self.builder.build(ity.clone(), Op::Splat(s))
            }
            _ => self
                .builder
                .build(ity.clone(), Op::ConstInt(val, ity.clone())),
        }
    }

    fn lower_cast(&mut self, operand: &Expr, e: &Expr) -> ValueId {
        let v = self.lower_expr(operand);
        let from = self.expr_mir(operand);
        let to = self.expr_mir(e);
        if from == to {
            return v;
        }
        // For a float→int cast the signed/unsigned choice comes from the TARGET integer (`x as i32`
        // is signed → fptosi); for every other direction (int→float, int widening) it comes from the
        // source operand. Using the operand's signedness for float→int picks fptoui, where the native
        // backend saturates a negative float to 0 while the interpreter keeps the signed value — a
        // native≠interpreter divergence (e.g. `(-0.5 * 10.0) as i32` gave 0 on native, −5 on interp).
        let signed = if from.is_float() && to.is_int() {
            self.signed(e)
        } else {
            self.signed(operand)
        };
        let kind = cast_kind(&from, &to, signed);
        self.builder.build(to.clone(), Op::Cast(kind, v, to))
    }
}

// ---- free helpers ----

/// The MIR type a parameter is passed as at the call boundary. Arrays decay to a base pointer.
fn param_abi_ty(ty: &Ty) -> MirType {
    match mir_ty(ty) {
        MirType::Array(..) => MirType::Ptr,
        t => t,
    }
}

fn mir_ty(ty: &Ty) -> MirType {
    match ty {
        Ty::Scalar(s) => MirType::from_scalar(*s),
        Ty::Array { elem, len } => MirType::Array(Box::new(mir_ty(elem)), *len as u32),
        Ty::Ptr { .. } | Ty::Ref { .. } | Ty::Tensor { .. } | Ty::Slice(_) => MirType::Ptr,
        Ty::Vector { elem, lanes } => MirType::Vec(Box::new(MirType::from_scalar(*elem)), *lanes),
        Ty::Unit => MirType::Void,
        _ => MirType::I32,
    }
}

fn mir_ty_of_ast(t: &ast::TypeExpr, interner: &Interner) -> MirType {
    use ast::TypeKind::*;
    match &t.kind {
        Path(p) => {
            let name = interner.resolve(p.segments.last().unwrap().sym);
            match mercury_types::Scalar::from_name(name) {
                Some(s) => MirType::from_scalar(s),
                None => MirType::I32,
            }
        }
        Array { elem, len } => match const_usize_expr(len, interner) {
            // A literal-length array lowers to an array type; otherwise fall back to an opaque ptr.
            Some(n) => MirType::Array(Box::new(mir_ty_of_ast(elem, interner)), n),
            None => MirType::Ptr,
        },
        Pointer { .. } | Ref { .. } | Slice(_) | Tensor { .. } => MirType::Ptr,
        Vector { elem, lanes } => {
            let e = mir_ty_of_ast(elem, interner);
            MirType::Vec(Box::new(e), *lanes)
        }
        Unit => MirType::Void,
        _ => MirType::I32,
    }
}

/// Evaluate a compile-time array length that is a plain integer literal.
fn const_usize_expr(e: &Expr, interner: &Interner) -> Option<u32> {
    match &e.kind {
        ExprKind::Int(s) => Some(parse_int(interner.resolve(*s)) as u32),
        _ => None,
    }
}

fn arith_binop(op: ast::BinOp, float: bool, signed: bool) -> BinOp {
    use ast::BinOp as A;
    match op {
        A::Add => {
            if float {
                BinOp::FAdd
            } else {
                BinOp::Add
            }
        }
        A::Sub => {
            if float {
                BinOp::FSub
            } else {
                BinOp::Sub
            }
        }
        A::Mul => {
            if float {
                BinOp::FMul
            } else {
                BinOp::Mul
            }
        }
        A::Div => {
            if float {
                BinOp::FDiv
            } else if signed {
                BinOp::SDiv
            } else {
                BinOp::UDiv
            }
        }
        A::Rem => {
            if float {
                BinOp::FRem
            } else if signed {
                BinOp::SRem
            } else {
                BinOp::URem
            }
        }
        A::BitAnd => BinOp::And,
        A::BitOr => BinOp::Or,
        A::BitXor => BinOp::Xor,
        A::Shl => BinOp::Shl,
        A::Shr => {
            if signed {
                BinOp::AShr
            } else {
                BinOp::LShr
            }
        }
        _ => BinOp::Add,
    }
}

fn compound_binop(op: ast::AssignOp, float: bool, signed: bool) -> BinOp {
    use ast::AssignOp as A;
    let bin = match op {
        A::Add => ast::BinOp::Add,
        A::Sub => ast::BinOp::Sub,
        A::Mul => ast::BinOp::Mul,
        A::Div => ast::BinOp::Div,
        A::Rem => ast::BinOp::Rem,
        A::BitAnd => ast::BinOp::BitAnd,
        A::BitOr => ast::BinOp::BitOr,
        A::BitXor => ast::BinOp::BitXor,
        A::Shl => ast::BinOp::Shl,
        A::Shr => ast::BinOp::Shr,
        A::Assign => ast::BinOp::Add,
    };
    arith_binop(bin, float, signed)
}

fn cmp_pred(op: ast::BinOp, float: bool, signed: bool) -> CmpOp {
    use ast::BinOp as A;
    match op {
        A::Eq => {
            if float {
                CmpOp::Foeq
            } else {
                CmpOp::Eq
            }
        }
        A::Ne => {
            if float {
                CmpOp::Fone
            } else {
                CmpOp::Ne
            }
        }
        A::Lt => float_or(float, CmpOp::Folt, signed, CmpOp::Slt, CmpOp::Ult),
        A::Le => float_or(float, CmpOp::Fole, signed, CmpOp::Sle, CmpOp::Ule),
        A::Gt => float_or(float, CmpOp::Fogt, signed, CmpOp::Sgt, CmpOp::Ugt),
        A::Ge => float_or(float, CmpOp::Foge, signed, CmpOp::Sge, CmpOp::Uge),
        _ => CmpOp::Eq,
    }
}

fn float_or(float: bool, f: CmpOp, signed: bool, s: CmpOp, u: CmpOp) -> CmpOp {
    if float {
        f
    } else if signed {
        s
    } else {
        u
    }
}

fn is_numeric(t: &MirType) -> bool {
    t.is_int() || t.is_float()
}

// ---- vectorizer analysis helpers (pure AST/type predicates) ----

/// The symbol of a single-segment path expression, if `e` is one.
fn single_path(e: &Expr) -> Option<Symbol> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
        _ => None,
    }
}

/// The single statement of a one-statement, tail-less block (the shape every per-pass norm loop body
/// has); `None` otherwise.
fn single_stmt(body: &Block) -> Option<&Stmt> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    Some(&body.stmts[0])
}

/// Does `e` reference any symbol in `set`? (Used to keep inner vector temps out of index exprs.)
fn uses_any(e: &Expr, set: &HashSet<Symbol>) -> bool {
    match &e.kind {
        ExprKind::Path(p) => p.is_single() && set.contains(&p.first().sym),
        ExprKind::Binary { lhs, rhs, .. } => uses_any(lhs, set) || uses_any(rhs, set),
        ExprKind::Unary { expr, .. } => uses_any(expr, set),
        ExprKind::Index { base, indices } => {
            uses_any(base, set) || indices.iter().any(|i| uses_any(i, set))
        }
        ExprKind::Cast { expr, .. } => uses_any(expr, set),
        _ => false,
    }
}

/// Does `e` mention the loop variable `sym`?
fn expr_uses_sym(e: &Expr, sym: Symbol) -> bool {
    match &e.kind {
        ExprKind::Path(p) => p.is_single() && p.first().sym == sym,
        ExprKind::Binary { lhs, rhs, .. } => expr_uses_sym(lhs, sym) || expr_uses_sym(rhs, sym),
        ExprKind::Unary { expr, .. } => expr_uses_sym(expr, sym),
        ExprKind::Index { base, indices } => {
            expr_uses_sym(base, sym) || indices.iter().any(|i| expr_uses_sym(i, sym))
        }
        ExprKind::Cast { expr, .. } => expr_uses_sym(expr, sym),
        _ => false,
    }
}

/// Whole-expression "does this reference `sym`?" — unlike [`expr_uses_sym`] (which only walks the
/// index/arith subset), this recurses through calls, fields, and casts, and is **conservative**:
/// expression kinds it does not model return `true`. Used to prove a fused region's internal scalars
/// do not leak to later code (a `true` just means "can't prove it's safe", so we decline to fuse).
fn expr_mentions(e: &Expr, sym: Symbol) -> bool {
    match &e.kind {
        ExprKind::Path(p) => p.is_single() && p.first().sym == sym,
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_) => false,
        ExprKind::Unary { expr, .. } => expr_mentions(expr, sym),
        ExprKind::Binary { lhs, rhs, .. } => expr_mentions(lhs, sym) || expr_mentions(rhs, sym),
        ExprKind::Call { callee, args, .. } => {
            expr_mentions(callee, sym) || args.iter().any(|a| expr_mentions(a, sym))
        }
        ExprKind::Index { base, indices } => {
            expr_mentions(base, sym) || indices.iter().any(|i| expr_mentions(i, sym))
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
            expr_mentions(base, sym)
        }
        ExprKind::Cast { expr, .. } => expr_mentions(expr, sym),
        // struct/array literals, if/match/block exprs, method calls, ranges, …: assume a use.
        _ => true,
    }
}

/// Conservative "does any statement (or the tail) reference `sym`?" — companion to [`expr_mentions`].
fn block_mentions(stmts: &[Stmt], tail: Option<&Expr>, sym: Symbol) -> bool {
    stmts.iter().any(|s| stmt_mentions(s, sym)) || tail.is_some_and(|e| expr_mentions(e, sym))
}

fn stmt_mentions(s: &Stmt, sym: Symbol) -> bool {
    match &s.kind {
        StmtKind::Let { init, .. } => init.as_ref().is_some_and(|e| expr_mentions(e, sym)),
        StmtKind::Assign { target, value, .. } => {
            expr_mentions(target, sym) || expr_mentions(value, sym)
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_mentions(e, sym),
        StmtKind::Return(o) => o.as_ref().is_some_and(|e| expr_mentions(e, sym)),
        StmtKind::Break(_) | StmtKind::Continue(_) => false,
        StmtKind::While { cond, body, .. } => {
            expr_mentions(cond, sym) || block_mentions(&body.stmts, body.tail.as_deref(), sym)
        }
        StmtKind::For { iter, body, .. } => {
            for_iter_mentions(iter, sym) || block_mentions(&body.stmts, body.tail.as_deref(), sym)
        }
        StmtKind::Loop { body, .. } => block_mentions(&body.stmts, body.tail.as_deref(), sym),
    }
}

fn for_iter_mentions(it: &ForIter, sym: Symbol) -> bool {
    match it {
        ForIter::Range {
            start, end, step, ..
        } => {
            expr_mentions(start, sym)
                || end.as_ref().is_some_and(|e| expr_mentions(e, sym))
                || step.as_ref().is_some_and(|e| expr_mentions(e, sym))
        }
        ForIter::Expr(e) => expr_mentions(e, sym),
    }
}

/// The coefficient of `j` in an affine index expression: `Some(0)` invariant, `Some(1)` unit-stride,
/// other constants for non-unit strides, `None` if not provably affine in `j`. Only `+`/`-` combine
/// `j`; a `*` involving `j` is conservatively rejected (we don't constant-evaluate factors).
fn affine_stride(e: &Expr, j: Symbol) -> Option<i64> {
    if !expr_uses_sym(e, j) {
        return Some(0);
    }
    match &e.kind {
        ExprKind::Path(p) if p.is_single() && p.first().sym == j => Some(1),
        ExprKind::Binary { op, lhs, rhs } => match op {
            ast::BinOp::Add => affine_stride(lhs, j)?.checked_add(affine_stride(rhs, j)?),
            ast::BinOp::Sub => affine_stride(lhs, j)?.checked_sub(affine_stride(rhs, j)?),
            _ => None,
        },
        _ => None,
    }
}

/// Structural equality over the elementwise-expression subset (ints, floats, single paths, unary,
/// `+`/`-`/`*`, and unit-stride `Index`). Used to confirm a written array is touched at exactly one
/// index (no loop-carried dependence) and that a ReLU's then-branch returns the value its guard
/// compares (`if v > 0 { v } else { 0 }`). The `Index` arm is what lets `x[i] == x[i]` hold.
fn exprs_struct_eq(a: &Expr, b: &Expr) -> bool {
    match (&a.kind, &b.kind) {
        (ExprKind::Int(x), ExprKind::Int(y)) => x == y,
        (ExprKind::Float(x), ExprKind::Float(y)) => x == y,
        (ExprKind::Path(p), ExprKind::Path(q)) => {
            p.is_single() && q.is_single() && p.first().sym == q.first().sym
        }
        (
            ExprKind::Unary { op: o1, expr: e1 },
            ExprKind::Unary { op: o2, expr: e2 },
        ) => o1 == o2 && exprs_struct_eq(e1, e2),
        (
            ExprKind::Binary {
                op: o1,
                lhs: l1,
                rhs: r1,
            },
            ExprKind::Binary {
                op: o2,
                lhs: l2,
                rhs: r2,
            },
        ) => o1 == o2 && exprs_struct_eq(l1, l2) && exprs_struct_eq(r1, r2),
        (
            ExprKind::Index {
                base: b1,
                indices: i1,
            },
            ExprKind::Index {
                base: b2,
                indices: i2,
            },
        ) => {
            i1.len() == i2.len()
                && exprs_struct_eq(b1, b2)
                && i1.iter().zip(i2).all(|(x, y)| exprs_struct_eq(x, y))
        }
        _ => false,
    }
}

/// Pin a shared lane type: set it if unset, else require equality. `None` means a type clash (bail).
fn set_or_check(slot: &mut Option<MirType>, t: &MirType) -> Option<()> {
    match slot {
        Some(existing) if existing == t => Some(()),
        Some(_) => None,
        None => {
            *slot = Some(t.clone());
            Some(())
        }
    }
}

// --- matmul recognition ---------------------------------------------------------------------
//
// A naive matmul loop nest is the single most important ML kernel and the one a general C/Rust
// compiler optimizes least (it vectorizes the inner loop but never register-blocks, cache-tiles, or
// packs). Mercury recognizes the canonical f32 `ikj` nest and lowers the *whole nest* to a call into
// the tuned `mercury_sgemm` microkernel (true 256-bit AVX2/FMA) — exactly how XLA/TVM/oneDNN lower a
// matmul op. The interpreter runs the identical kernel via marshalling, so the two stay bit-exact.

/// A recognized GEMM nest computing `C[m,n] = A[m,k]·B[k,n]` (row-major, contiguous, f32), or its
/// `nn.Linear` transpose `C[m,n] = A[m,k]·B[n,k]ᵀ` when `transposed`.
struct MatmulNest<'a> {
    a: Symbol,
    b: Symbol,
    c: Symbol,
    m: Dim,
    k: Dim,
    n: Dim,
    /// 0 = overwrite C (a zero-init loop was present), 1 = accumulate into C.
    beta: i64,
    /// `true` for `C = A·Bᵀ` (B indexed `[j,k]` instead of `[k,j]`).
    transposed: bool,
    /// Per-operand base offsets: the additive index terms left over after the 2-D `row*stride + col`
    /// is peeled off — e.g. the batch/head term `h*S*D` of a **batched** matmul (multi-head
    /// attention is one matmul per head: `scores[h] = Q[h]·K[h]ᵀ`). Each term is verified invariant
    /// in the matmul's `(i,j,k)`, so `emit_sgemm` simply GEPs the base pointer by their sum before
    /// the kernel call — the inner matmul is identical regardless of the offset. Empty for a plain
    /// 2-D matmul; only the `ijk` dot-product form populates these.
    a_off: Vec<&'a Expr>,
    b_off: Vec<&'a Expr>,
    c_off: Vec<&'a Expr>,
}

/// Canonical text of an affine index/base expression (paths, ints, `+`/`-`/`*`, casts), used to key
/// the vectorizer's per-body load cache. `None` for anything outside that subset (not cached).
fn index_canon(e: &Expr, interner: &Interner) -> Option<String> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(interner.resolve(p.first().sym).to_string()),
        ExprKind::Int(s) => Some(interner.resolve(*s).to_string()),
        ExprKind::Binary { op, lhs, rhs } => Some(format!(
            "({} {} {})",
            index_canon(lhs, interner)?,
            op.glyph(),
            index_canon(rhs, interner)?
        )),
        ExprKind::Cast { expr, .. } => index_canon(expr, interner),
        _ => None,
    }
}

/// A cache key identifying the memory a `base[index]` read touches (`base` is a single path).
fn load_key(base: &Expr, index: &Expr, interner: &Interner) -> Option<String> {
    Some(format!(
        "{}[{}]",
        index_canon(base, interner)?,
        index_canon(index, interner)?
    ))
}

/// A non-negative integer literal.
fn as_int_lit(e: &Expr, interner: &Interner) -> Option<i64> {
    match &e.kind {
        ExprKind::Int(s) => i64::try_from(parse_int(interner.resolve(*s))).ok(),
        _ => None,
    }
}

/// A matmul dimension or stride: a compile-time literal, or a runtime variable (function param or
/// local). Two `Var`s compare equal iff they name the same binding, so the recognizer's stride/bound
/// consistency checks (`sa == k`, `sc == n`, …) hold symbolically — which is what lets a matmul with
/// runtime dimensions dispatch to the tuned GEMM kernel instead of falling back to a scalar nest.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dim {
    Lit(i64),
    Var(Symbol),
}

/// A non-negative dimension/stride expression: an integer literal or a single variable path.
fn as_dim(e: &Expr, interner: &Interner) -> Option<Dim> {
    if let Some(v) = as_int_lit(e, interner) {
        return Some(Dim::Lit(v));
    }
    single_path(e).map(Dim::Var)
}

/// `e == row * stride` (either factor order) for the *known* row variable; returns the stride. The
/// known row resolves the otherwise-ambiguous `i * N` form (both factors are paths under symbolic
/// dims) — the factor that is `row` is the index, the other is the stride.
fn mul_with_row(e: &Expr, row: Symbol, interner: &Interner) -> Option<Dim> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    if single_path(lhs) == Some(row) {
        return as_dim(rhs, interner);
    }
    if single_path(rhs) == Some(row) {
        return as_dim(lhs, interner);
    }
    None
}

/// A flattened 2-D index `row * stride + col` (either addend order) for a known `row`. Returns
/// `(stride, col)`.
fn match_row_col(idx: &Expr, row: Symbol, interner: &Interner) -> Option<(Dim, Symbol)> {
    let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &idx.kind
    else {
        return None;
    };
    if let (Some(stride), Some(col)) = (mul_with_row(lhs, row, interner), single_path(rhs)) {
        return Some((stride, col));
    }
    if let (Some(col), Some(stride)) = (single_path(lhs), mul_with_row(rhs, row, interner)) {
        return Some((stride, col));
    }
    None
}

/// Flatten the additive terms of `e`, recursing only through `+`. `i*K + k + h*S*D` yields the three
/// terms `[i*K, k, h*S*D]` (left-association is irrelevant). Used to peel a batch/base offset off a
/// flattened tensor index.
fn flatten_add_terms<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &e.kind
    {
        flatten_add_terms(lhs, out);
        flatten_add_terms(rhs, out);
    } else {
        out.push(e);
    }
}

/// Like [`match_row_col`] but tolerant of a leading **base offset**: parses
/// `idx == row*stride + col + offset_terms…` for a known `row` and a known expected `col` symbol
/// (the matmul column that the caller already knows must appear). Returns `(stride, offset_terms)`,
/// where `offset_terms` are the remaining addends (empty for a plain 2-D index). The caller must
/// verify the offset is invariant in the matmul's bound variables. Passing the expected `col` is what
/// disambiguates the bare column term from an offset that is itself a bare path.
fn match_row_col_off<'a>(
    idx: &'a Expr,
    row: Symbol,
    col: Symbol,
    interner: &Interner,
) -> Option<(Dim, Vec<&'a Expr>)> {
    let mut terms = Vec::new();
    flatten_add_terms(idx, &mut terms);
    // The unique `row * stride` term.
    let row_pos = terms
        .iter()
        .position(|t| mul_with_row(t, row, interner).is_some())?;
    let stride = mul_with_row(terms[row_pos], row, interner)?;
    terms.remove(row_pos);
    // The bare column term `col` (must be present exactly as the expected symbol).
    let col_pos = terms.iter().position(|t| single_path(t) == Some(col))?;
    terms.remove(col_pos);
    // Whatever is left is the base offset (a batch/head index for a batched matmul).
    Some((stride, terms))
}

/// `base[index]` with a single-segment `base` path and exactly one index. Returns `(base, index)`.
fn as_index1(e: &Expr) -> Option<(Symbol, &Expr)> {
    match &e.kind {
        ExprKind::Index { base, indices } if indices.len() == 1 => {
            Some((single_path(base)?, &indices[0]))
        }
        _ => None,
    }
}

/// Is `e` the float literal `0.0` (any spelling)?
fn is_float_zero(e: &Expr, interner: &Interner) -> bool {
    matches!(&e.kind, ExprKind::Float(s) if parse_float(interner.resolve(*s)) == 0.0)
}

/// Is `e`'s sema type `f32`? (matmul lowering is f32-only — that's what `mercury_sgemm` computes.)
fn is_f32_expr(e: &Expr, sema: &SemaResult) -> bool {
    matches!(sema.types.get(&e.id), Some(t) if mir_ty(t) == MirType::F32)
}

/// `for col in 0..n { c[row*stride + col] = 0.0; }` — the per-row zero-init of a beta-0 matmul.
/// Returns `(c, stride, n)` with the outer row variable `row`.
fn match_zero_init(s: &Stmt, row: Symbol, interner: &Interner) -> Option<(Symbol, Dim, Dim)> {
    let (pat, iter, body) = fusable_for(s)?;
    let col = match &pat.kind {
        ast::PatKind::Ident(c) => *c,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let n = as_dim(end, interner)?;
    if body.stmts.len() != 1 || body.tail.is_some() {
        return None;
    }
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &body.stmts[0].kind
    else {
        return None;
    };
    if !is_float_zero(value, interner) {
        return None;
    }
    let (cbase, cidx) = as_index1(target)?;
    let (stride, cc) = match_row_col(cidx, row, interner)?;
    if cc != col {
        return None;
    }
    Some((cbase, stride, n))
}

/// Classify a `Mul` product as `A·B`, returning `(a, sa, b, sb, transposed)`. `aik` carries an
/// optional pre-bound `let aik = A[row*sa+k]` (the `ikj` form); otherwise A is found inline. B is
/// `B[k*N+j]` (normal) or `B[j*K+k]` (transposed — the nn.Linear `A·Bᵀ`). Either factor order.
fn match_product_ab(
    prod: &Expr,
    row: Symbol,
    kvar: Symbol,
    jvar: Symbol,
    aik: Option<(Symbol, Symbol, Dim)>,
    interner: &Interner,
) -> Option<(Symbol, Dim, Symbol, Dim, bool)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: f1,
        rhs: f2,
    } = &prod.kind
    else {
        return None;
    };
    let is_a = |f: &Expr| -> Option<(Symbol, Dim)> {
        if let Some((aik_sym, asym, sa)) = aik {
            if single_path(f) == Some(aik_sym) {
                return Some((asym, sa));
            }
        }
        let (abase, aidx) = as_index1(f)?;
        let (asa, ak) = match_row_col(aidx, row, interner)?;
        (ak == kvar).then_some((abase, asa))
    };
    let is_b = |f: &Expr| -> Option<(Symbol, Dim, bool)> {
        let (bbase, bidx) = as_index1(f)?;
        // normal `B[k*N+j]`: row is k, col is j; transposed `B[j*K+k]`: row is j, col is k.
        if let Some((sb, bc)) = match_row_col(bidx, kvar, interner) {
            if bc == jvar {
                return Some((bbase, sb, false));
            }
        }
        if let Some((sb, bc)) = match_row_col(bidx, jvar, interner) {
            if bc == kvar {
                return Some((bbase, sb, true));
            }
        }
        None
    };
    let pair = |fa: &Expr, fb: &Expr| match (is_a(fa), is_b(fb)) {
        (Some((a, sa)), Some((b, sb, t))) => Some((a, sa, b, sb, t)),
        _ => None,
    };
    pair(f1, f2).or_else(|| pair(f2, f1))
}

/// An A factor `A[row*sa + k (+ off)]` of the inline `ijk` product. Returns `(base, sa, offset)`.
fn match_a_factor<'a>(
    f: &'a Expr,
    row: Symbol,
    kvar: Symbol,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>)> {
    let (abase, aidx) = as_index1(f)?;
    let (sa, off) = match_row_col_off(aidx, row, kvar, interner)?;
    Some((abase, sa, off))
}

/// A B factor of the inline `ijk` product: `B[k*sb + j (+ off)]` (normal) or `B[j*sb + k (+ off)]`
/// (transposed — the `A·Bᵀ` spelling). Returns `(base, sb, offset, transposed)`.
fn match_b_factor<'a>(
    f: &'a Expr,
    kvar: Symbol,
    jvar: Symbol,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>, bool)> {
    let (bbase, bidx) = as_index1(f)?;
    if let Some((sb, off)) = match_row_col_off(bidx, kvar, jvar, interner) {
        return Some((bbase, sb, off, false));
    }
    if let Some((sb, off)) = match_row_col_off(bidx, jvar, kvar, interner) {
        return Some((bbase, sb, off, true));
    }
    None
}

/// Like [`match_product_ab`] but for the inline `ijk` form (A read directly, never via an `aik`
/// binding) and tolerant of a per-operand **base offset** (the batch/head index of a batched matmul).
/// Returns `(a, sa, a_off, b, sb, b_off, transposed)`.
#[allow(clippy::type_complexity)]
fn match_product_ab_off<'a>(
    prod: &'a Expr,
    row: Symbol,
    kvar: Symbol,
    jvar: Symbol,
    interner: &Interner,
) -> Option<(Symbol, Dim, Vec<&'a Expr>, Symbol, Dim, Vec<&'a Expr>, bool)> {
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: f1,
        rhs: f2,
    } = &prod.kind
    else {
        return None;
    };
    // Either factor order: `A*B` or `B*A`.
    for (fa, fb) in [(f1, f2), (f2, f1)] {
        if let (Some((a, sa, aoff)), Some((b, sb, boff, t))) = (
            match_a_factor(fa, row, kvar, interner),
            match_b_factor(fb, kvar, jvar, interner),
        ) {
            return Some((a, sa, aoff, b, sb, boff, t));
        }
    }
    None
}

/// Every term of `off` is invariant in all of `vars` (the matmul's bound `i`/`j`/`k`). A base offset
/// that mentioned a loop variable would not be a constant per-call pointer shift, so it is rejected.
fn offset_invariant(off: &[&Expr], vars: &[Symbol]) -> bool {
    off.iter()
        .all(|t| vars.iter().all(|&v| !expr_uses_sym(t, v)))
}

/// Try both recognized matmul spellings: the `ikj` accumulate form and the `ijk` dot-product form.
fn recognize_matmul<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    match_matmul(pat, iter, body, sema, interner)
        .or_else(|| match_matmul_ijk(pat, iter, body, sema, interner))
}

// Fused-epilogue activation codes — must match `mercury_runtime`'s gemm kernel (ACT_IDENTITY/RELU).
const EPI_ACT_IDENTITY: u32 = 0;
const EPI_ACT_RELU: u32 = 1;

/// If `e` is `base_sym[idx]` (single index off the path `base_sym`), return `idx`.
fn index_of(e: &Expr, base_sym: Symbol) -> Option<&Expr> {
    if let ExprKind::Index { base, indices } = &e.kind {
        if indices.len() == 1 && single_path(base) == Some(base_sym) {
            return Some(&indices[0]);
        }
    }
    None
}

/// Is `e` the matmul output element `C[i*N+j]` (row `ivar`, col `jvar`, stride `n`)?
fn is_c_elem(
    e: &Expr,
    c_sym: Symbol,
    ivar: Symbol,
    jvar: Symbol,
    n: Dim,
    interner: &Interner,
) -> bool {
    index_of(e, c_sym)
        .and_then(|idx| match_row_col(idx, ivar, interner))
        .is_some_and(|(stride, col)| stride == n && col == jvar)
}

/// Match `C[i*N+j] + bias[j]` (either addend order) over the matmul output; return the bias array.
fn match_c_plus_bias(
    e: &Expr,
    c_sym: Symbol,
    ivar: Symbol,
    jvar: Symbol,
    n: Dim,
    interner: &Interner,
) -> Option<Symbol> {
    let ExprKind::Binary {
        op: ast::BinOp::Add,
        lhs,
        rhs,
    } = &e.kind
    else {
        return None;
    };
    // The non-C addend must be `bias[j]` — a single-index access by the column variable.
    let bias_of = |x: &Expr| -> Option<Symbol> {
        if let ExprKind::Index { base, indices } = &x.kind {
            if indices.len() == 1 && single_path(&indices[0]) == Some(jvar) {
                return single_path(base);
            }
        }
        None
    };
    if is_c_elem(lhs, c_sym, ivar, jvar, n, interner) {
        return bias_of(rhs);
    }
    if is_c_elem(rhs, c_sym, ivar, jvar, n, interner) {
        return bias_of(lhs);
    }
    None
}

/// Match the epilogue RHS: `C[i*N+j] + bias[j]` (identity) or `fmax(C[i*N+j] + bias[j], 0)` (ReLU).
/// Bias is required (the common `act(x·Wᵀ + bias)` shape). Returns `(bias_array, act_code)`.
fn match_epi_value(
    e: &Expr,
    c_sym: Symbol,
    ivar: Symbol,
    jvar: Symbol,
    n: Dim,
    interner: &Interner,
) -> Option<(Symbol, u32)> {
    // ReLU written as `fmax(inner, 0.0)`.
    if let ExprKind::Call { callee, args, .. } = &e.kind {
        if args.len() == 2
            && single_path(callee).is_some_and(|s| interner.resolve(s) == "fmax")
            && is_float_zero(&args[1], interner)
        {
            let bias = match_c_plus_bias(&args[0], c_sym, ivar, jvar, n, interner)?;
            return Some((bias, EPI_ACT_RELU));
        }
    }
    // Identity: just the bias add.
    let bias = match_c_plus_bias(e, c_sym, ivar, jvar, n, interner)?;
    Some((bias, EPI_ACT_IDENTITY))
}

/// Match the bias/activation epilogue loop following a recognized `nn.Linear` matmul:
/// `for i in 0..M { for j in 0..N { C[i*N+j] = C[i*N+j] + bias[j] } }`, optionally wrapped in
/// `fmax(_, 0)` (ReLU). `M`/`N`/the stride/the output array/the column index must all match `nest`,
/// so it never misfires; bias is required. Returns `(bias_array, act_code)`, else `None` (the loop
/// is then lowered normally as a separate pass).
fn match_bias_act_epilogue(
    stmt: &Stmt,
    nest: &MatmulNest<'_>,
    interner: &Interner,
) -> Option<(Symbol, u32)> {
    // for i in 0..M { <single nested loop> }
    let (ipat, iiter, ibody) = fusable_for(stmt)?;
    let ivar = match &ipat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (istart, iend) = range_bounds(iiter)?;
    if as_int_lit(istart, interner)? != 0 || as_dim(iend, interner)? != nest.m {
        return None;
    }
    if ibody.tail.is_some() || ibody.stmts.len() != 1 {
        return None;
    }
    // for j in 0..N { <single assignment> }
    let (jpat, jiter, jbody) = fusable_for(&ibody.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (jstart, jend) = range_bounds(jiter)?;
    if as_int_lit(jstart, interner)? != 0 || as_dim(jend, interner)? != nest.n {
        return None;
    }
    if jbody.tail.is_some() || jbody.stmts.len() != 1 {
        return None;
    }
    // C[i*N+j] = <epilogue>   (plain assignment to the matmul's output element)
    let StmtKind::Assign {
        target,
        op: ast::AssignOp::Assign,
        value,
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    if !is_c_elem(target, nest.c, ivar, jvar, nest.n, interner) {
        return None;
    }
    match_epi_value(value, nest.c, ivar, jvar, nest.n, interner)
}

/// A recognized int8 quantized `nn.Linear` nest: `C[m,n] (i32) = A[m,k] (u8) · B[n,k] (i8)ᵀ`.
struct I8MatmulNest {
    a: Symbol,
    b: Symbol,
    c: Symbol,
    m: Dim,
    k: Dim,
    n: Dim,
}

/// Peel an `as`-cast wrapper (`x as T` → `x`); the expression itself otherwise. int8 GEMM source
/// casts each `u8`/`i8` element to `i32` before multiplying (the product can't fit `i8`).
fn peel_cast(e: &Expr) -> &Expr {
    match &e.kind {
        ExprKind::Cast { expr, .. } => expr,
        _ => e,
    }
}

/// The scalar type sema assigned to `e`, if any.
fn scalar_of(e: &Expr, sema: &SemaResult) -> Option<mercury_types::Scalar> {
    match sema.types.get(&e.id) {
        Some(Ty::Scalar(s)) => Some(*s),
        _ => None,
    }
}

/// A literal integer `0` (the int8 accumulator seed `let mut s: i32 = 0`).
fn is_int_zero(e: &Expr, interner: &Interner) -> bool {
    matches!(&e.kind, ExprKind::Int(t) if parse_int(interner.resolve(*t)) == 0)
}

/// Recognize the int8 quantized `nn.Linear` nest `C = A·Bᵀ` — the dot-product `ijk` form with `u8`
/// activations, `i8` weights, and an `i32` accumulator:
///
/// ```text
/// for i in 0..M { for j in 0..N {
///   let mut s: i32 = 0;
///   for k in 0..K { s = s + (a[i*K + k] as i32) * (b[j*K + k] as i32); }
///   c[i*N + j] = s;
/// } }
/// ```
///
/// Returns the nest iff A is `u8`, B is `i8`, C is `i32`, B is transposed (`b[j*K+k]` — the weight
/// layout `mercury_i8gemm_nt` expects), the strides are consistent (`sa = sb = K`, `sc = N`), and
/// there are no batch offsets (plain 2-D). The signedness is enforced because the kernel zero-extends
/// A and sign-extends B; matching the wrong signedness would miscompile, so anything else falls back
/// to the scalar nest. Integer arithmetic means the kernel equals the scalar nest bit-for-bit.
fn match_matmul_i8_nt(
    pat: &Pattern,
    iter: &ForIter,
    body: &Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<I8MatmulNest> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (js, je) = range_bounds(jiter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let n = as_dim(je, interner)?;
    if jbody.tail.is_some() || jbody.stmts.len() != 3 {
        return None;
    }
    // [0] let mut s: i32 = 0;
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let s_sym = match &sp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    if !is_int_zero(s0, interner) {
        return None;
    }
    // [1] for k in 0..K { s = s + (a[..] as i32) * (b[..] as i32); }
    let (kpat, kiter, kbody) = fusable_for(&jbody.stmts[1])?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (ks, ke) = range_bounds(kiter)?;
    if as_int_lit(ks, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(ke, interner)?;
    if kbody.tail.is_some() || kbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &kbody.stmts[0].kind else {
        return None;
    };
    if single_path(target) != Some(s_sym) {
        return None;
    }
    let prod = match op {
        ast::AssignOp::Add => value,
        ast::AssignOp::Assign => {
            let ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            else {
                return None;
            };
            if single_path(lhs) != Some(s_sym) {
                return None;
            }
            rhs
        }
        _ => return None,
    };
    // The product must be i32 (the casts widen to i32; the accumulation matches the kernel's i32).
    if !matches!(sema.types.get(&prod.id), Some(t) if mir_ty(t) == MirType::I32) {
        return None;
    }
    let ExprKind::Binary {
        op: ast::BinOp::Mul,
        lhs: f1,
        rhs: f2,
    } = &prod.kind
    else {
        return None;
    };
    // Each factor is `(arr[idx] as i32)`. Peel the cast and identify A (row i) and B (row j, the
    // transposed weight layout), enforcing A:u8 × B:i8 and no batch offset, in either factor order.
    let (a_sym, sa, b_sym, sb) = {
        let mut found = None;
        for (fa, fb) in [(f1, f2), (f2, f1)] {
            let (ai, bi) = (peel_cast(fa), peel_cast(fb));
            let Some((a_sym, sa, a_off)) = match_a_factor(ai, row, kvar, interner) else {
                continue;
            };
            let Some((b_sym, sb, b_off, transposed)) = match_b_factor(bi, kvar, jvar, interner)
            else {
                continue;
            };
            if !transposed || !a_off.is_empty() || !b_off.is_empty() {
                continue;
            }
            if scalar_of(ai, sema) != Some(mercury_types::Scalar::U8)
                || scalar_of(bi, sema) != Some(mercury_types::Scalar::I8)
            {
                continue;
            }
            found = Some((a_sym, sa, b_sym, sb));
            break;
        }
        found?
    };
    // [2] c[i*N + j] = s;  (C must be i32, plain 2-D, strides consistent.)
    let StmtKind::Assign {
        target: ct,
        op: ast::AssignOp::Assign,
        value: cv,
    } = &jbody.stmts[2].kind
    else {
        return None;
    };
    if single_path(cv) != Some(s_sym) {
        return None;
    }
    let (cbase, cidx) = as_index1(ct)?;
    let (sc, c_off) = match_row_col_off(cidx, row, jvar, interner)?;
    if !c_off.is_empty() || sa != kdim || sb != kdim || sc != n {
        return None;
    }
    if scalar_of(ct, sema) != Some(mercury_types::Scalar::I32) {
        return None;
    }
    // An input aliasing the output is a hazard (the kernel writes C in a different order). A == B is
    // fine (both read-only).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    Some(I8MatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
    })
}

/// Recognize the textbook `ijk` dot-product matmul:
/// `for i { for j { let s = 0.0; for k { s = s + A[i,k]*B[..]; } c[i*N+j] = s; } }`. This is the
/// natural way to write `C = A·Bᵀ` (both A and B read contiguously). Always `beta = 0` (s overwrites
/// c). See [`MatmulNest`].
fn match_matmul_ijk<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    // Outer body is a single `for j` loop.
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let (jpat, jiter, jbody) = fusable_for(&body.stmts[0])?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (js, je) = range_bounds(jiter)?;
    if as_int_lit(js, interner)? != 0 {
        return None;
    }
    let n = as_dim(je, interner)?;
    // j body: [ let s = 0.0; for k {...}; c[i*N+j] = s ].
    if jbody.tail.is_some() || jbody.stmts.len() != 3 {
        return None;
    }
    let StmtKind::Let {
        pat: sp,
        init: Some(s0),
        ..
    } = &jbody.stmts[0].kind
    else {
        return None;
    };
    let s_sym = match &sp.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    if !is_float_zero(s0, interner) {
        return None;
    }
    // The K loop: `for k in 0..K { s = s + A*B; }`.
    let (kpat, kiter, kbody) = fusable_for(&jbody.stmts[1])?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (ks, ke) = range_bounds(kiter)?;
    if as_int_lit(ks, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(ke, interner)?;
    if kbody.tail.is_some() || kbody.stmts.len() != 1 {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &kbody.stmts[0].kind else {
        return None;
    };
    if single_path(target) != Some(s_sym) {
        return None;
    }
    let prod = match op {
        ast::AssignOp::Add => value,
        ast::AssignOp::Assign => {
            // s = s + A*B
            let ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            else {
                return None;
            };
            if single_path(lhs) != Some(s_sym) {
                return None;
            }
            rhs
        }
        _ => return None,
    };
    if !is_f32_expr(prod, sema) {
        return None;
    }
    // The inline `ijk` form tolerates a per-operand base offset (a batch/head index): A, B and C may
    // each be indexed `… + h*S*D`, the hallmark of a batched matmul (multi-head attention is one
    // matmul per head). The offsets are peeled off here and applied as pointer GEPs in `emit_sgemm`.
    let (a_sym, sa, a_off, b_sym, sb, b_off, transposed) =
        match_product_ab_off(prod, row, kvar, jvar, interner)?;
    // Final store: c[i*N + j (+ off)] = s.
    let StmtKind::Assign {
        target: ct,
        op: ast::AssignOp::Assign,
        value: cv,
    } = &jbody.stmts[2].kind
    else {
        return None;
    };
    if single_path(cv) != Some(s_sym) {
        return None;
    }
    let (cbase, cidx) = as_index1(ct)?;
    let (sc, c_off) = match_row_col_off(cidx, row, jvar, interner)?;
    let sb_ok = if transposed { sb == kdim } else { sb == n };
    if sa != kdim || !sb_ok || sc != n {
        return None;
    }
    // Every base offset must be invariant in the matmul's own `(i,j,k)` — otherwise it is not a
    // constant per-call pointer shift and the nest is not a batched matmul. (A plain 2-D matmul has
    // empty offsets and trivially passes.)
    let bound = [row, jvar, kvar];
    if !offset_invariant(&a_off, &bound)
        || !offset_invariant(&b_off, &bound)
        || !offset_invariant(&c_off, &bound)
    {
        return None;
    }
    // A and B may be the *same* array (a Gram matrix `A·Aᵀ`, or self-attention `Q·Kᵀ` with a shared
    // operand): both sides are read-only, which the kernel packs into separate scratch panels, so it
    // is safe. Only an input aliasing the output C is a hazard (the blocked kernel writes C in a
    // different order than the scalar nest reads it).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    Some(MatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
        beta: 0,
        transposed,
        a_off,
        b_off,
        c_off,
    })
}

/// Recognize the canonical f32 matmul nest rooted at `for row in 0..M { … }`. See [`MatmulNest`].
fn match_matmul<'a>(
    pat: &Pattern,
    iter: &ForIter,
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    let row = match &pat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (start, end) = range_bounds(iter)?;
    if as_int_lit(start, interner)? != 0 {
        return None;
    }
    let m = as_dim(end, interner)?;
    if body.tail.is_some() {
        return None;
    }

    // Body is either [k-loop] (accumulate) or [zero-init, k-loop] (overwrite).
    let (beta, czero, k_stmt) = match body.stmts.as_slice() {
        [k] => (1i64, None, k),
        [z, k] => (0i64, Some(z), k),
        _ => return None,
    };

    // The K loop: `for k in 0..K { [let aik = a[row*sa + k];] for j in 0..N { … } }`.
    let (kpat, kiter, kbody) = fusable_for(k_stmt)?;
    let kvar = match &kpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (kstart, kend) = range_bounds(kiter)?;
    if as_int_lit(kstart, interner)? != 0 {
        return None;
    }
    let kdim = as_dim(kend, interner)?;

    // Optional `let aik = a[row*sa + k];` binding, then the inner J loop.
    let (aik, a_sym, sa, j_stmt) = match kbody.stmts.as_slice() {
        [j] if kbody.tail.is_none() => (None, None, None, j),
        [let_s, j] if kbody.tail.is_none() => {
            let StmtKind::Let {
                pat: lp,
                init: Some(init),
                ..
            } = &let_s.kind
            else {
                return None;
            };
            let aik_sym = match &lp.kind {
                ast::PatKind::Ident(s) => *s,
                _ => return None,
            };
            let (abase, aidx) = as_index1(init)?;
            let (asa, ak) = match_row_col(aidx, row, interner)?;
            if ak != kvar {
                return None;
            }
            (Some(aik_sym), Some(abase), Some(asa), j)
        }
        _ => return None,
    };

    // Inner J loop: `for j in 0..N { c[row*sc + j] (=|+=) … a … * … b[k*sb + j] … }`.
    let (jpat, jiter, jbody) = fusable_for(j_stmt)?;
    let jvar = match &jpat.kind {
        ast::PatKind::Ident(s) => *s,
        _ => return None,
    };
    let (jstart, jend) = range_bounds(jiter)?;
    if as_int_lit(jstart, interner)? != 0 {
        return None;
    }
    let n = as_dim(jend, interner)?;
    if jbody.stmts.len() != 1 || jbody.tail.is_some() {
        return None;
    }
    let StmtKind::Assign { target, op, value } = &jbody.stmts[0].kind else {
        return None;
    };
    let (cbase, cidx) = as_index1(target)?;
    let (sc, cj) = match_row_col(cidx, row, interner)?;
    if cj != jvar {
        return None;
    }

    // The product `a_term * b_term`, possibly read-add'd into c.
    let prod = match op {
        ast::AssignOp::Add => value,
        ast::AssignOp::Assign => {
            // value must be `c[row*sc + j] + (a*b)`.
            let ExprKind::Binary {
                op: ast::BinOp::Add,
                lhs,
                rhs,
            } = &value.kind
            else {
                return None;
            };
            let (clhs, clidx) = as_index1(lhs)?;
            let (clsc, clj) = match_row_col(clidx, row, interner)?;
            if clhs != cbase || clsc != sc || clj != jvar {
                return None;
            }
            rhs
        }
        _ => return None,
    };
    if !is_f32_expr(prod, sema) {
        return None;
    }
    let aik_info = match (aik, a_sym, sa) {
        (Some(s), Some(b), Some(st)) => Some((s, b, st)),
        _ => None,
    };
    let (a_sym, sa, b_sym, sb, transposed) =
        match_product_ab(prod, row, kvar, jvar, aik_info, interner)?;

    // Strides must describe contiguous row-major A[m,k] and C[m,n], and B[k,n] (normal) or B[n,k]
    // (transposed) — i.e. B's contraction stride is N normally, K when transposed.
    let sb_ok = if transposed { sb == kdim } else { sb == n };
    if sa != kdim || !sb_ok || sc != n {
        return None;
    }
    if beta == 0 {
        let (cz, scz, nz) = match_zero_init(czero?, row, interner)?;
        if cz != cbase || scz != n || nz != n {
            return None;
        }
    }
    // A and B may be the *same* array (a Gram matrix `A·Aᵀ`, or self-attention `Q·Kᵀ` with a shared
    // operand): both sides are read-only, which the kernel packs into separate scratch panels, so it
    // is safe. Only an input aliasing the output C is a hazard (the blocked kernel writes C in a
    // different order than the scalar nest reads it).
    if a_sym == cbase || b_sym == cbase {
        return None;
    }
    Some(MatmulNest {
        a: a_sym,
        b: b_sym,
        c: cbase,
        m,
        k: kdim,
        n,
        beta,
        transposed,
        // The `ikj` accumulate form parses its A/C indices with the 2-term `match_row_col`, so a
        // batched (offset) index falls back to the scalar nest; offsets are always empty here.
        a_off: Vec::new(),
        b_off: Vec::new(),
        c_off: Vec::new(),
    })
}

/// Recognize a function whose entire body is a matmul nest (`{ for i in 0..M { … } }`).
fn matmul_fn<'a>(
    body: &'a Block,
    sema: &SemaResult,
    interner: &Interner,
) -> Option<MatmulNest<'a>> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    recognize_matmul(pat, iter, lb, sema, interner)
}

/// Lower a recognized matmul function to a thin wrapper that binds its array params to base
/// pointers and tail-calls `mercury_sgemm`/`mercury_sgemm_parallel`.
#[allow(clippy::too_many_arguments)]
fn lower_matmul_fn(
    f: &FnDecl,
    nest: &MatmulNest<'_>,
    parallel: bool,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    diags: &mut Vec<Diagnostic>,
) -> Function {
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (f.params.iter().map(|_| Ty::Unknown).collect(), Ty::Unit),
    };
    let ret_mir = mir_ty(&ret_ty);
    let mut fl = FnLowerer {
        builder: Builder::new(f.name.sym, ret_mir.clone()),
        sema,
        interner,
        diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
    };
    let param_vals: Vec<ValueId> = param_tys
        .iter()
        .map(|pty| fl.builder.add_param(param_abi_ty(pty)))
        .collect();
    for ((p, pty), val) in f.params.iter().zip(&param_tys).zip(param_vals) {
        let mty = mir_ty(pty);
        if matches!(mty, MirType::Array(..)) {
            fl.bind(p.name.sym, val, mty);
        } else {
            let slot = fl.builder.alloca(mty.clone());
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: val,
            });
            fl.bind(p.name.sym, slot, mty);
        }
    }
    fl.emit_sgemm(nest, parallel);
    if !fl.terminated {
        match ret_mir {
            MirType::Void => fl.builder.ret(None),
            _ => {
                let z = fl.const_zero(ret_mir.clone());
                fl.builder.ret(Some(z));
            }
        }
    }
    fl.builder.finish()
}

/// Recognize a function whose entire body is an int8 matmul nest (`{ for i in 0..M { … } }`) — the
/// int8 twin of `matmul_fn`, used for whole-function quantized `nn.Linear` kernels.
fn i8matmul_fn(body: &Block, sema: &SemaResult, interner: &Interner) -> Option<I8MatmulNest> {
    if body.tail.is_some() || body.stmts.len() != 1 {
        return None;
    }
    let StmtKind::For {
        pat,
        iter,
        body: lb,
        ..
    } = &body.stmts[0].kind
    else {
        return None;
    };
    match_matmul_i8_nt(pat, iter, lb, sema, interner)
}

/// Lower a recognized int8 matmul function to a thin wrapper that binds its array params to base
/// pointers and tail-calls `mercury_i8gemm_nt`/`mercury_i8gemm_nt_parallel` — the int8 twin of
/// `lower_matmul_fn`.
#[allow(clippy::too_many_arguments)]
fn lower_i8matmul_fn(
    f: &FnDecl,
    nest: &I8MatmulNest,
    parallel: bool,
    sema: &SemaResult,
    interner: &Interner,
    gemm: GemmSyms,
    diags: &mut Vec<Diagnostic>,
) -> Function {
    let (param_tys, ret_ty) = match sema.defs.lookup(f.name.sym).map(|d| &d.kind) {
        Some(DefKind::Fn(sig)) => (sig.params.clone(), sig.ret.clone()),
        _ => (f.params.iter().map(|_| Ty::Unknown).collect(), Ty::Unit),
    };
    let ret_mir = mir_ty(&ret_ty);
    let mut fl = FnLowerer {
        builder: Builder::new(f.name.sym, ret_mir.clone()),
        sema,
        interner,
        diags,
        scopes: vec![HashMap::new()],
        terminated: false,
        loops: Vec::new(),
        gemm,
        parallel_fn: false,
        vec_loads: HashMap::new(),
    };
    let param_vals: Vec<ValueId> = param_tys
        .iter()
        .map(|pty| fl.builder.add_param(param_abi_ty(pty)))
        .collect();
    for ((p, pty), val) in f.params.iter().zip(&param_tys).zip(param_vals) {
        let mty = mir_ty(pty);
        if matches!(mty, MirType::Array(..)) {
            fl.bind(p.name.sym, val, mty);
        } else {
            let slot = fl.builder.alloca(mty.clone());
            fl.builder.build_void(Op::Store {
                ptr: slot,
                value: val,
            });
            fl.bind(p.name.sym, slot, mty);
        }
    }
    fl.emit_i8gemm(nest, parallel);
    if !fl.terminated {
        match ret_mir {
            MirType::Void => fl.builder.ret(None),
            _ => {
                let z = fl.const_zero(ret_mir.clone());
                fl.builder.ret(Some(z));
            }
        }
    }
    fl.builder.finish()
}

/// SIMD width for a lane type: lanes per `VEC_REG_BYTES`-wide register (f32x8, f64x4 at 256-bit).
/// Returns `None` for lane types we don't vectorize.
fn vector_width(lane: &MirType) -> Option<u32> {
    let bytes = match lane {
        MirType::F32 | MirType::I32 => 4,
        MirType::F64 | MirType::I64 => 8,
        _ => return None,
    };
    Some(VEC_REG_BYTES / bytes)
}

/// SIMD register width in bytes. Cranelift's vector ISA is 128-bit (16 bytes); wider types such as
/// `f32x8` are not legalized ("Unexpected SSA-value type"), so we pack one 128-bit register and
/// recover AVX-class throughput via unrolling (see `VEC_UNROLL`) rather than a wider lane type.
const VEC_REG_BYTES: u32 = 16;

/// Vector groups processed per iteration of the unrolled main loop. Independent 128-bit chains
/// issue across the core's multiple FP units (≈ AVX throughput from SSE ops) and hide FP latency in
/// reduction-style bodies (Horner, matmul accumulate). 4×f32x4 = 16 f32/iteration.
const VEC_UNROLL: u32 = 4;

/// Default signedness for a lane type (only affects integer div/rem op selection).
fn lane_signed(lane: &MirType) -> bool {
    lane.is_int()
}

/// The pure tail value of a braces-only block (`{ e }`), or `None` if it has statements.
fn block_value(b: &Block) -> Option<&Expr> {
    if b.stmts.is_empty() {
        b.tail.as_deref()
    } else {
        None
    }
}

/// The value of an `if` branch: a braces-only block yields its tail; a non-block expression (e.g. a
/// chained `else if`) yields itself.
fn branch_value(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Block(b) => block_value(b),
        _ => Some(e),
    }
}

/// The integer lane type of a vector compare mask: same bit width as the value lane (so the mask
/// reinterprets cleanly for a bitwise blend on the native side).
fn mask_lane_type(lane: &MirType) -> MirType {
    match lane {
        MirType::F64 | MirType::I64 => MirType::I64,
        _ => MirType::I32,
    }
}

/// A `for v in a..b { body }` over a half-open, unit-step range with an identifier binding — the
/// shape eligible for fusion. Returns `(pattern, iter, body)`.
fn fusable_for(s: &Stmt) -> Option<(&Pattern, &ForIter, &Block)> {
    if let StmtKind::For {
        pat, iter, body, ..
    } = &s.kind
    {
        if matches!(&pat.kind, ast::PatKind::Ident(_))
            && matches!(
                iter,
                ForIter::Range {
                    end: Some(_),
                    inclusive: false,
                    step: None,
                    ..
                }
            )
        {
            return Some((pat, iter, body));
        }
    }
    None
}

/// The `(start, end)` expressions of a half-open range iterator.
fn range_bounds(iter: &ForIter) -> Option<(&Expr, &Expr)> {
    match iter {
        ForIter::Range {
            start,
            end: Some(end),
            ..
        } => Some((start, end)),
        _ => None,
    }
}

/// Concatenate the bodies of a run of `for` statements into one block (statements cloned; their
/// `NodeId`s are preserved so sema type lookups still resolve). The wrapper block reuses the first
/// body's id/span, which are not used for typing.
fn fuse_for_bodies(stmts: &[Stmt]) -> Block {
    let mut fused = Vec::new();
    let mut id = None;
    let mut span = None;
    for s in stmts {
        if let StmtKind::For { body, .. } = &s.kind {
            if id.is_none() {
                id = Some(body.id);
                span = Some(body.span);
            }
            fused.extend(body.stmts.iter().cloned());
        }
    }
    Block {
        id: id.unwrap(),
        stmts: fused,
        tail: None,
        span: span.unwrap(),
    }
}

/// The wider of two numeric MIR types (float beats int), used to pick a common comparison type.
/// Non-numeric or equal types yield the left type.
fn numeric_join(a: &MirType, b: &MirType) -> MirType {
    if a == b || !is_numeric(a) || !is_numeric(b) {
        return a.clone();
    }
    let bits = |t: &MirType| match t {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 | MirType::F16 | MirType::BF16 => 16,
        MirType::I32 | MirType::F32 => 32,
        _ => 64,
    };
    match (a.is_float(), b.is_float()) {
        (true, false) => a.clone(),
        (false, true) => b.clone(),
        _ => {
            if bits(a) >= bits(b) {
                a.clone()
            } else {
                b.clone()
            }
        }
    }
}

fn cast_kind(from: &MirType, to: &MirType, signed: bool) -> CastKind {
    let isz = |t: &MirType| match t {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => 0,
    };
    let fsz = |t: &MirType| match t {
        MirType::F16 | MirType::BF16 => 16,
        MirType::F32 => 32,
        MirType::F64 => 64,
        _ => 0,
    };
    match (from.is_int(), to.is_int(), from.is_float(), to.is_float()) {
        (true, true, _, _) => {
            if isz(to) > isz(from) {
                if signed {
                    CastKind::SExt
                } else {
                    CastKind::ZExt
                }
            } else {
                CastKind::Trunc
            }
        }
        (true, _, _, true) => {
            if signed {
                CastKind::SiToFp
            } else {
                CastKind::UiToFp
            }
        }
        (_, true, true, _) => {
            if signed {
                CastKind::FpToSi
            } else {
                CastKind::FpToUi
            }
        }
        (_, _, true, true) => {
            if fsz(to) > fsz(from) {
                CastKind::FpExt
            } else {
                CastKind::FpTrunc
            }
        }
        _ if matches!(from, MirType::Ptr) && to.is_int() => CastKind::PtrToInt,
        _ if from.is_int() && matches!(to, MirType::Ptr) => CastKind::IntToPtr,
        _ => CastKind::Bitcast,
    }
}

/// How a recognised float/int reduction folds its lanes: a sum, or a running `fmax`/`fmin`.
/// `Add` covers `s += x` / `s = s + x` (float reassociated, int exact); `Fmax`/`Fmin` cover
/// `m = fmax(m, x)` / `m = fmin(m, x)` (float only — exactly associative for finite, non-NaN
/// inputs, so the vectorized fold is bit-identical to the scalar one).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RedOp {
    Add,
    Fmax,
    Fmin,
}

/// The math builtins lowered directly to primitive MIR ops (not runtime calls).
#[derive(Clone, Copy)]
enum MathIntrinsic {
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Pow,
    Erf,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Silu,
    Gelu,
    Fmax,
    Fmin,
}

fn math_intrinsic(name: &str) -> Option<MathIntrinsic> {
    Some(match name {
        "sqrt" => MathIntrinsic::Sqrt,
        "rsqrt" => MathIntrinsic::Rsqrt,
        "exp" => MathIntrinsic::Exp,
        "log" => MathIntrinsic::Log,
        "pow" => MathIntrinsic::Pow,
        "erf" => MathIntrinsic::Erf,
        "sin" => MathIntrinsic::Sin,
        "cos" => MathIntrinsic::Cos,
        "tanh" => MathIntrinsic::Tanh,
        "sigmoid" => MathIntrinsic::Sigmoid,
        "silu" => MathIntrinsic::Silu,
        "gelu" => MathIntrinsic::Gelu,
        "fmax" => MathIntrinsic::Fmax,
        "fmin" => MathIntrinsic::Fmin,
        _ => return None,
    })
}

/// The same shape as `ty` (scalar or `Vec`) but with float lane `lane`.
fn float_ty_like(ty: &MirType, lane: MirType) -> MirType {
    match ty {
        MirType::Vec(_, n) => MirType::Vec(Box::new(lane), *n),
        _ => lane,
    }
}

/// The boolean-mask type a compare on `ty` yields: `i1` for a scalar, or a per-lane integer vector
/// (matching the native backend's vector compare result) for a `Vec`.
fn mask_ty(ty: &MirType) -> MirType {
    match ty {
        MirType::Vec(lane, n) => MirType::Vec(Box::new(mask_lane_type(lane)), *n),
        _ => MirType::I1,
    }
}

// `exp` polynomial constants (Cephes single-precision `expf`), evaluated in f32 by both backends.
const LOG2EF: f64 = std::f64::consts::LOG2_E;
const EXP_MAGIC: f64 = 12582912.0; // 1.5 * 2^23 — round-to-nearest via add then sub
const EXP_C1: f64 = 0.693359375; // ln2, high part
const EXP_C2: f64 = -2.1219444e-4; // ln2, low correction
const EXP_HI: f64 = 88.3762626647949;
const EXP_LO: f64 = -88.3762626647949;
const EXP_P: [f64; 6] = [
    1.98756915e-4,
    1.3981999507e-3,
    8.3334519073e-3,
    4.1665795894e-2,
    1.6666665459e-1,
    5.0000001201e-1,
];

/// Natural-log polynomial constants (Cephes `logf`). `LOG_SQRTHF` is the `√0.5` split point that
/// keeps the reduced mantissa centered; `LOG_P` is the degree-8 minimax poly on it. `e·ln2` is
/// added back with the *same* split `EXP_C1`/`EXP_C2` that `exp` uses (ln2 = C1 + C2).
const LOG_SQRTHF: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2 = √0.5
const INV_2P23: f64 = 1.0 / 8_388_608.0; // 2^-23 (exact): scales the masked exponent field to a count
const LOG_P: [f64; 9] = [
    7.0376836292e-2,
    -1.1514610310e-1,
    1.1676998740e-1,
    -1.2420140846e-1,
    1.4249322787e-1,
    -1.6668057665e-1,
    2.0000714765e-1,
    -2.4999993993e-1,
    3.3333331174e-1,
];

// `erf` constants (Abramowitz–Stegun 7.1.26): erf(|x|) = 1 - (a₁t + a₂t² + … + a₅t⁵)·e^(-x²),
// t = 1/(1 + P·|x|). Max error ~1.5e-7 — f32-grade — and bit-identical across backends since it is
// built from primitive ops plus `exp`. Enables exact (erf-based) GELU, the original BERT/GPT-2 form.
const ERF_P: f64 = 0.327_591_1;
const ERF_A: [f64; 5] = [
    0.254_829_592,
    -0.284_496_736,
    1.421_413_741,
    -1.453_152_027,
    1.061_405_429,
];

// `sin`/`cos` constants (Cephes single-precision `sinf`/`cosf`). Reduce `x` to `r ∈ [-π/4, π/4]` by
// `q = round(x·2/π)` quadrants, then `r = x - q·(π/2)` with π/2 split into three parts (`PIO2_*`, the
// Cephes π/4 `DP` constants doubled) so the cancellation stays accurate. `q mod 4` picks ±sin/±cos of
// the reduced angle. All primitive ops + the proven round-to-nearest magic, so both backends agree.
const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI; // 2/π — quadrant count = round(x·2/π)
const PIO2_1: f64 = 1.5703125; // π/2 high (2 × Cephes DP1 = 2 × 0.78515625)
const PIO2_2: f64 = 4.837_512_969_970_703e-4; // π/2 mid  (2 × DP2)
const PIO2_3: f64 = 7.549_789_954_891_88e-8; // π/2 low  (2 × DP3)
const SIN_P: [f64; 3] = [-1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1];
const COS_P: [f64; 3] = [
    2.443_315_711_809_948e-5,
    -1.388_731_625_493_765e-3,
    4.166_664_568_298_827e-2,
];

/// Names that lower to runtime/interpreter intrinsics rather than user functions.
pub fn is_intrinsic(name: &str) -> bool {
    matches!(name, "print" | "println" | "assert")
}

fn parse_int(text: &str) -> i128 {
    let digits: String = text
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '_')
        .collect();
    digits.replace('_', "").parse().unwrap_or(0)
}

fn parse_float(text: &str) -> f64 {
    // Strip a trailing type suffix (bf16/f16/f32/f64) before parsing.
    let mut core = text;
    for suf in ["bf16", "f16", "f32", "f64"] {
        if let Some(stripped) = core.strip_suffix(suf) {
            core = stripped;
            break;
        }
    }
    core.parse().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_mir::verify::verify_function;
    use mercury_span::SourceId;

    fn lower(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = mercury_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = lower_program(&module, &sema, &mut interner);
        (prog, diags, interner)
    }

    #[test]
    fn lowers_and_verifies_loop_sum() {
        let src = "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                   while i < 10 { s += i; i += 1; } return s; }";
        let (prog, diags, _) = lower(src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        for f in &prog.funcs {
            let errs = verify_function(f);
            assert!(errs.is_empty(), "verify: {errs:?}");
        }
    }

    #[test]
    fn lowers_and_verifies_array_ops() {
        // Array literal init, indexed store, and indexed load must lower to a verifiable function
        // with an array alloca.
        let src = "fn main() -> i32 { let mut xs: [i32; 3] = [1, 2, 3]; \
                   xs[1] = 9; return xs[1]; }";
        let (prog, diags, _) = lower(src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let main = &prog.funcs[0];
        assert!(
            main.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(&i.op, Op::Alloca(MirType::Array(_, 3)))),
            "expected an array alloca"
        );
        assert!(verify_function(main).is_empty(), "verify failed");
    }

    #[test]
    fn multi_dim_tensor_index_lowers_to_row_major_gep() {
        // The shape-typed surface: `t[i, j]` on `Tensor[f32, M, N]` must lower (no `unsupported`
        // C0001) to a flat row-major offset `i*N + j`, and the function must verify.
        let src = "fn k(a: Tensor[f32, 3, 4], out: Tensor[f32, 3, 4]) { \
                   for i in 0..3 { for j in 0..4 { out[i, j] = a[i, j] * 2.0; } } }";
        let (prog, diags, mut interner) = lower(src);
        assert!(
            diags.iter().all(|d| !d.is_error()),
            "multi-dim index should lower cleanly: {diags:?}"
        );
        let k_sym = interner.intern("k");
        let k = prog.funcs.iter().find(|f| f.name == k_sym).expect("fn k");
        assert!(
            verify_function(k).is_empty(),
            "verify: {:?}",
            verify_function(k)
        );
        // The inner stride is the trailing dim N=4: expect a `* 4` in the offset arithmetic.
        assert!(
            k.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(&i.op, Op::ConstInt(4, MirType::I64))),
            "expected a stride-4 (trailing dim) constant in the flat-index arithmetic"
        );
    }

    #[test]
    fn array_parameter_passes_by_pointer() {
        // An array parameter has ABI type Ptr (passed by base pointer), not an array alloca.
        let src = "fn dot(x: [i32; 2], y: [i32; 2]) -> i32 { return x[0]*y[0] + x[1]*y[1]; } \
                   fn main() -> i32 { let a: [i32;2] = [1,2]; let b: [i32;2] = [3,4]; \
                   return dot(a, b); }";
        let (prog, diags, interner) = lower(src);
        assert!(diags.iter().all(|d| !d.is_error()), "{diags:?}");
        let dot = prog
            .funcs
            .iter()
            .find(|f| interner.resolve(f.name) == "dot")
            .unwrap();
        // Both params are pointers; no array alloca inside `dot`.
        for &p in &dot.params {
            assert_eq!(dot.value_type(p), &MirType::Ptr);
        }
        assert!(
            !dot.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(&i.op, Op::Alloca(MirType::Array(..)))),
            "array params must not alloca array storage in the callee"
        );
        assert!(verify_function(dot).is_empty());
    }

    #[test]
    fn lowers_and_verifies_recursive_fib() {
        let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } \
                   return fib(n - 1) + fib(n - 2); } \
                   fn main() -> i32 { return fib(10); }";
        let (prog, _diags, _) = lower(src);
        assert_eq!(prog.funcs.len(), 2);
        for f in &prog.funcs {
            assert!(
                verify_function(f).is_empty(),
                "verify failed for a function"
            );
        }
    }
}
