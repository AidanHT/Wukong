//! **Dispatch control and the dispatch report** — the user's handle on the compiler's single largest
//! performance decision.
//!
//! *Control.* `@generic` used to be legal on a function item only, so a function with one hot GEMM
//! and one tiny helper loop could not keep the first and opt the second out. It is now also legal on
//! a `for` or `while` statement, and the rule is: **no recognizer replaces code that contains a
//! `@generic` loop.** Concretely —
//!
//! * the loop itself and everything nested in it lower with recognition off;
//! * a loop that merely *encloses* a `@generic` loop lowers with its own probes off (its cascade
//!   would swallow the whole nest), but hands recognition back to the statements of its body, so the
//!   `@generic` loop's siblings still dispatch ([`FnLowerer::with_loop_attrs`] and the
//!   `resume` hand-off in `GemmSyms`);
//! * no statement window is formed over a statement that contains one;
//! * no whole-function interception claims a function that contains one.
//!
//! `@parallel` outlining, the autovectorizer and loop fusion are not recognizers and are unaffected,
//! exactly as for the function-level form.
//!
//! *Report.* Dispatch was silent: the only proof that a loop became a kernel call was piping
//! `--emit=mir` through `grep wukong_`. With reporting on (`wukongc --remarks=dispatch`, i.e.
//! [`crate::lower_program_reporting_dispatch`]) every statement whose lowering emitted a recognized
//! kernel call gets one `note` diagnostic, at that statement, naming the runtime symbols — the
//! innermost statement that dispatched, never also its enclosing loops. A function replaced
//! wholesale by an interception is reported at its name. Reporting is pure observation: the MIR is
//! byte-identical with it on or off.

use wukong_ast::{Attr, Block, Expr, ExprKind, Stmt, StmtKind};
use wukong_diag::{Diagnostic, Severity};
use wukong_mir::{BlockId, Function, Inst, Op};
use wukong_span::{Interner, Span, Symbol};

use crate::FnLowerer;

/// Every dispatch remark's message starts with this, which is how an enclosing statement tells that
/// a nested one already reported (and how a reader greps for them).
const REMARK_PREFIX: &str = "recognized kernel dispatch";

/// The longest statement window any `lower_block` matcher consumes (a seeding copy plus a 7-statement
/// norm window). A statement containing a `@generic` loop within this many statements of a window's
/// start keeps the window matchers off for that start.
const MAX_WINDOW: usize = 8;

/// Where the builder was when a statement's lowering began.
pub(super) struct DispatchMark {
    blocks: usize,
    block: BlockId,
    insts: usize,
    diags: usize,
    span: Span,
}

/// Does this attribute list carry `@generic`?
pub(super) fn has_generic(attrs: &[Attr], interner: &Interner) -> bool {
    attrs
        .iter()
        .any(|a| interner.resolve(a.name.sym) == "generic")
}

/// Does any of the first [`MAX_WINDOW`] statements carry or contain a `@generic` loop? (Sema admits
/// the attribute only on loops.)
pub(super) fn window_has_generic(stmts: &[Stmt], interner: &Interner) -> bool {
    stmts
        .iter()
        .take(MAX_WINDOW)
        .any(|s| stmt_has_generic(s, interner))
}

/// Does `body` contain a `@generic` loop at any depth?
pub(super) fn body_has_generic_loop(body: &Block, interner: &Interner) -> bool {
    body.stmts.iter().any(|s| stmt_has_generic(s, interner))
        || body
            .tail
            .as_deref()
            .is_some_and(|e| expr_has_generic(e, interner))
}

fn stmt_has_generic(s: &Stmt, interner: &Interner) -> bool {
    match &s.kind {
        StmtKind::For { body, .. } | StmtKind::While { body, .. } => {
            has_generic(&s.attrs, interner) || body_has_generic_loop(body, interner)
        }
        StmtKind::Let { init: Some(e), .. } | StmtKind::Expr(e) => expr_has_generic(e, interner),
        StmtKind::Assign { value, .. } => expr_has_generic(value, interner),
        _ => false,
    }
}

fn expr_has_generic(e: &Expr, interner: &Interner) -> bool {
    match &e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => body_has_generic_loop(b, interner),
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            body_has_generic_loop(then_branch, interner)
                || else_branch
                    .as_deref()
                    .is_some_and(|x| expr_has_generic(x, interner))
        }
        ExprKind::Match { arms, .. } => arms.iter().any(|a| expr_has_generic(&a.body, interner)),
        _ => false,
    }
}

/// A runtime symbol that a kernel RECOGNIZER emits — every `wukong_*` call except the `@parallel`
/// entry point, the runtime services (`wukong_rt_*`, `wukong_now_ns`) and the explicitly called
/// `sdpa` builtin.
fn is_recognized_kernel(name: &str) -> bool {
    name.starts_with("wukong_")
        && !name.starts_with("wukong_rt_")
        && !matches!(
            name,
            "wukong_parallel_for" | "wukong_now_ns" | "wukong_attention_f32"
        )
}

fn is_dispatch_remark(d: &Diagnostic) -> bool {
    d.severity == Severity::Note && d.message.starts_with(REMARK_PREFIX)
}

/// Append the recognized-kernel callees of `insts` to `out`, once each, in emission order.
fn collect_kernels(insts: &[Inst], interner: &Interner, out: &mut Vec<Symbol>) {
    for i in insts {
        if let Op::Call { func, .. } = &i.op {
            if is_recognized_kernel(interner.resolve(*func)) && !out.contains(func) {
                out.push(*func);
            }
        }
    }
}

fn remark(kernels: &[Symbol], span: Span, label: &str, interner: &Interner) -> Diagnostic {
    let list = kernels
        .iter()
        .map(|k| format!("`{}`", interner.resolve(*k)))
        .collect::<Vec<_>>()
        .join(", ");
    Diagnostic::new(Severity::Note, format!("{REMARK_PREFIX}: {list}")).primary(span, label)
}

impl FnLowerer<'_> {
    /// Lower one loop statement (`attrs` and `body` are its own) under the recognition `@generic`
    /// asks for:
    ///
    /// * the loop carries `@generic` — recognition off for the loop and everything in it, and no
    ///   pending hand-off may turn it back on inside;
    /// * only a loop nested in `body` does — recognition off for THIS loop's probes (its cascade,
    ///   the vectorizer's kernel probes, the unsigned fast path), which would replace the whole nest,
    ///   and `resume` set so the first block lowered for it — its body — turns recognition back on
    ///   for the body's own statements ([`Self::resume_recognition`]);
    /// * neither — `lower` as is.
    pub(super) fn with_loop_attrs(
        &mut self,
        attrs: &[Attr],
        body: &Block,
        lower: impl FnOnce(&mut Self),
    ) {
        if has_generic(attrs, self.interner) {
            let (recognize, resume) = (self.gemm.recognize, self.gemm.resume);
            self.gemm.recognize = false;
            self.gemm.resume = false;
            lower(self);
            self.gemm.recognize = recognize;
            self.gemm.resume = resume;
            return;
        }
        if self.gemm.recognize && body_has_generic_loop(body, self.interner) {
            self.gemm.recognize = false;
            self.gemm.resume = true;
            lower(self);
            self.gemm.recognize = true;
            self.gemm.resume = false;
            return;
        }
        lower(self);
    }

    /// Entering a block: if the loop being lowered handed its body back to recognition (see
    /// [`Self::with_loop_attrs`]), turn recognition on for this block's statements. Returns whether
    /// it did, for [`Self::end_resumed_recognition`].
    pub(super) fn resume_recognition(&mut self) -> bool {
        if !self.gemm.resume {
            return false;
        }
        self.gemm.resume = false;
        self.gemm.recognize = true;
        true
    }

    /// Leaving a block that resumed recognition: the enclosing loop's lowering continues with its own
    /// probes off, and the hand-off is re-armed so a second lowering of the same body (a strip-mined
    /// remainder, a fallback copy) resumes too. `with_loop_attrs` clears it when the loop is done.
    pub(super) fn end_resumed_recognition(&mut self, resumed: bool) {
        if resumed {
            self.gemm.recognize = false;
            self.gemm.resume = true;
        }
    }

    /// Start observing the lowering of the statement at `span`; `None` when reporting is off.
    pub(super) fn dispatch_mark(&self, span: Span) -> Option<DispatchMark> {
        if !self.gemm.report {
            return None;
        }
        let block = self.builder.current();
        Some(DispatchMark {
            blocks: self.builder.num_blocks(),
            block,
            insts: self.builder.block(block).insts.len(),
            diags: self.diags.len(),
            span,
        })
    }

    /// Finish observing: report the recognized kernels the statement's lowering emitted, unless a
    /// statement nested inside it already reported them. Code is only ever appended to the block
    /// that was current at the mark or to blocks created after it (allocas, which go to the entry
    /// block, are not calls), so this sees exactly what the statement emitted.
    pub(super) fn flush_dispatch(&mut self, mark: Option<DispatchMark>) {
        let Some(mark) = mark else {
            return;
        };
        if self.diags[mark.diags..].iter().any(is_dispatch_remark) {
            return;
        }
        let mut kernels = Vec::new();
        let first = &self.builder.block(mark.block).insts;
        collect_kernels(
            &first[mark.insts.min(first.len())..],
            self.interner,
            &mut kernels,
        );
        for b in mark.blocks..self.builder.num_blocks() {
            collect_kernels(
                &self.builder.block(BlockId(b as u32)).insts,
                self.interner,
                &mut kernels,
            );
        }
        if !kernels.is_empty() {
            let d = remark(
                &kernels,
                mark.span,
                "this statement was replaced by a runtime kernel call",
                self.interner,
            );
            self.diags.push(d);
        }
    }
}

/// The function-level twin of [`FnLowerer::flush_dispatch`], for a function item lowered as the
/// functions `funcs[from..]` while `diags` grew from `diags_from`: when no statement inside it
/// reported (a whole-function interception lowers no statement), report every recognized kernel
/// those functions call, at the function's name.
pub(super) fn flush_function_dispatch(
    funcs: &[Function],
    from: usize,
    diags: &mut Vec<Diagnostic>,
    diags_from: usize,
    span: Span,
    interner: &Interner,
) {
    if diags[diags_from..].iter().any(is_dispatch_remark) {
        return;
    }
    let mut kernels = Vec::new();
    for f in &funcs[from.min(funcs.len())..] {
        for b in &f.blocks {
            collect_kernels(&b.insts, interner, &mut kernels);
        }
    }
    if !kernels.is_empty() {
        diags.push(remark(
            &kernels,
            span,
            "this function was replaced by a runtime kernel call",
            interner,
        ));
    }
}
