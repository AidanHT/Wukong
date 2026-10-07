//! In-module LLVM definitions of the general vectorizer's synthesized kernels
//! ([`wukong_mir::VecKernel`]).
//!
//! The Cranelift backend installs these recipes as raw AVX2 bytes, which only exist inside that
//! crate. The textual module used to *declare* `@__wukong_veckernel_*` and leave it undefined, so
//! any vectorized program's `.ll` failed at link time. The recipe itself is backend-agnostic, and its
//! semantics are pinned by the reference evaluators in `wukong_mir::vec_kernel` — which is what the
//! interpreter oracle runs — so this module lowers the recipe straight from those evaluators:
//!
//! * an **elementwise** recipe is the scalar loop `for e in 0..n { eval_lane(e) }`, element by
//!   element in index order, exactly as the interpreter marshals it (a store is visible to a later
//!   load of the same stream because it goes through memory, which is the evaluator's
//!   store-to-load forwarding);
//! * a **reduction** recipe reproduces `VecKernel::eval_reduction`'s fixed reassociation: `U`
//!   eight-lane accumulator copies (`U = reduction_unroll()`, the same single-sourced number the
//!   AVX2 assembler uses), the unrolled main loop, the single-group cleanup, the copy combine into
//!   copy 0 and the in-order lane 0..8 horizontal fold.
//!
//! Every value is an `f32`; `Fma` is `llvm.fma.f32` (one rounding, like `mul_add`), `Neg` is `fneg`
//! (a sign-bit flip), and each compare uses the predicate that matches the Rust operator the
//! evaluator applies (`Ne` is unordered). The kernel ABI is the recipe's
//! `fn(ptrs: *const *mut u8, scalars: *const f32, n: u64)`.

use std::fmt::Write as _;

use wukong_mir::{VecBin, VecCmp, VecKernel, VecOp, VecRedOp};

/// The symbol a function's `k`-th kernel is defined under (and called by `Op::VecKernelCall`).
pub(crate) fn kernel_symbol(func: &str, k: u32) -> String {
    format!("__wukong_veckernel_{func}_{k}")
}

/// Reject a recipe the reference evaluators would refuse.
pub(crate) fn check(kernel: &VecKernel) -> Result<(), String> {
    let errors = kernel.validate();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "LLVM emission: invalid vector kernel recipe: {}",
            errors.join("; ")
        ))
    }
}

/// Which math intrinsics a recipe calls, as `(fma, sqrt)`, so the module can declare them.
pub(crate) fn intrinsics(kernel: &VecKernel) -> (bool, bool) {
    let fma = kernel.ops.iter().any(|o| matches!(o, VecOp::Fma { .. }))
        || kernel.reduce.is_some_and(|r| r.fma.is_some());
    let sqrt = kernel.ops.iter().any(|o| matches!(o, VecOp::Sqrt { .. }));
    (fma, sqrt)
}

/// LLVM's spelling of an `f32` constant: the bits of the exactly widened double.
fn f32_lit(x: f32) -> String {
    format!("0x{:016X}", (x as f64).to_bits())
}

/// Emit the full definition of `kernel` under `symbol`.
pub(crate) fn emit_definition(out: &mut String, symbol: &str, kernel: &VecKernel) {
    let ret = if kernel.reduce.is_some() {
        "float"
    } else {
        "void"
    };
    let _ = writeln!(
        out,
        "define internal {ret} @{symbol}(ptr %ptrs, ptr %scalars, i64 %n) {{"
    );
    out.push_str("entry:\n");
    for s in 0..kernel.streams {
        let _ = writeln!(out, "  %base{s}.p = getelementptr ptr, ptr %ptrs, i64 {s}");
        let _ = writeln!(out, "  %base{s} = load ptr, ptr %base{s}.p");
    }
    for c in 0..kernel.scalars {
        let _ = writeln!(
            out,
            "  %scalar{c}.p = getelementptr float, ptr %scalars, i64 {c}"
        );
        let _ = writeln!(out, "  %scalar{c} = load float, ptr %scalar{c}.p");
    }
    match kernel.reduce {
        None => emit_elementwise(out, kernel),
        Some(_) => emit_reduction(out, kernel),
    }
    out.push_str("}\n");
}

/// `for e in 0..n { eval_lane(e) }`.
fn emit_elementwise(out: &mut String, kernel: &VecKernel) {
    out.push_str("  br label %head\nhead:\n");
    out.push_str("  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n");
    out.push_str("  %more = icmp ult i64 %i, %n\n");
    out.push_str("  br i1 %more, label %body, label %exit\nbody:\n");
    emit_body(out, kernel, "x", "%i");
    out.push_str("  %i.next = add i64 %i, 1\n  br label %head\nexit:\n  ret void\n");
}

/// `VecKernel::eval_reduction`, with the identical accumulator layout and fold order.
fn emit_reduction(out: &mut String, kernel: &VecKernel) {
    let red = kernel
        .reduce
        .expect("emit_reduction is reached for reduction recipes only");
    let lanes = wukong_mir::VEC_LANES;
    let copies = kernel.reduction_unroll();
    let slots = copies * lanes;
    let identity = f32_lit(red.op.identity());

    let _ = writeln!(out, "  %acc = alloca [{slots} x float]");
    for j in 0..slots {
        let _ = writeln!(out, "  %acc{j}.p = getelementptr float, ptr %acc, i64 {j}");
        let _ = writeln!(out, "  store float {identity}, ptr %acc{j}.p");
    }
    out.push_str("  br label %main.head\n");

    // Main loop: while i + U*8 <= n, fold element i + j into accumulator slot j for j in 0..U*8.
    // The slot order inside one step is immaterial (the slots are independent and the body stores
    // nothing); what matters is that slot j only ever sees elements congruent to j mod U*8, in
    // ascending order, which is exactly the evaluator's `acc[cu * 8 + lane]` schedule.
    out.push_str("main.head:\n");
    out.push_str("  %i = phi i64 [ 0, %entry ], [ %i.end, %main.latch ]\n");
    let _ = writeln!(out, "  %i.end = add i64 %i, {slots}");
    out.push_str("  %i.fits = icmp ule i64 %i.end, %n\n");
    out.push_str("  br i1 %i.fits, label %main.lane, label %tail.head\n");
    out.push_str("main.lane:\n");
    out.push_str("  %j = phi i64 [ 0, %main.head ], [ %j.next, %main.lane ]\n");
    out.push_str("  %e = add i64 %i, %j\n");
    let names = emit_body(out, kernel, "m", "%e");
    out.push_str("  %j.p = getelementptr float, ptr %acc, i64 %j\n");
    out.push_str("  %j.old = load float, ptr %j.p\n");
    emit_addend(out, &red, &names, "%m.acc", "%j.old");
    out.push_str("  store float %m.acc, ptr %j.p\n");
    out.push_str("  %j.next = add i64 %j, 1\n");
    let _ = writeln!(out, "  %j.more = icmp ult i64 %j.next, {slots}");
    out.push_str("  br i1 %j.more, label %main.lane, label %main.latch\n");
    out.push_str("main.latch:\n  br label %main.head\n");

    // Cleanup: remaining full 8-element groups fold into copy 0.
    out.push_str("tail.head:\n");
    out.push_str("  %t = phi i64 [ %i, %main.head ], [ %t.end, %tail.latch ]\n");
    let _ = writeln!(out, "  %t.end = add i64 %t, {lanes}");
    out.push_str("  %t.fits = icmp ule i64 %t.end, %n\n");
    out.push_str("  br i1 %t.fits, label %tail.lane, label %fold\n");
    out.push_str("tail.lane:\n");
    out.push_str("  %l = phi i64 [ 0, %tail.head ], [ %l.next, %tail.lane ]\n");
    out.push_str("  %te = add i64 %t, %l\n");
    let names = emit_body(out, kernel, "c", "%te");
    out.push_str("  %l.p = getelementptr float, ptr %acc, i64 %l\n");
    out.push_str("  %l.old = load float, ptr %l.p\n");
    emit_addend(out, &red, &names, "%c.acc", "%l.old");
    out.push_str("  store float %c.acc, ptr %l.p\n");
    out.push_str("  %l.next = add i64 %l, 1\n");
    let _ = writeln!(out, "  %l.more = icmp ult i64 %l.next, {lanes}");
    out.push_str("  br i1 %l.more, label %tail.lane, label %tail.latch\n");
    out.push_str("tail.latch:\n  br label %tail.head\n");

    // Combine copy cu into copy 0 lane-wise (cu ascending), then fold lanes 0..8 in order.
    out.push_str("fold:\n");
    for j in 0..slots {
        let _ = writeln!(out, "  %f{j} = load float, ptr %acc{j}.p");
    }
    let mut lane_value: Vec<String> = (0..lanes).map(|lane| format!("%f{lane}")).collect();
    for cu in 1..copies {
        for lane in 0..lanes {
            let name = format!("%g{cu}.{lane}");
            let other = format!("%f{}", cu * lanes + lane);
            emit_fold(out, red.op, &name, &lane_value[lane as usize], &other);
            lane_value[lane as usize] = name;
        }
    }
    let mut r = lane_value[0].clone();
    for (lane, value) in lane_value.iter().enumerate().skip(1) {
        let name = format!("%h{lane}");
        emit_fold(out, red.op, &name, &r, value);
        r = name;
    }
    let _ = writeln!(out, "  ret float {r}");
}

/// `acc = fma(X, Y, acc)` for a fused reduction, else `acc = fold(acc, value)`.
fn emit_addend(
    out: &mut String,
    red: &wukong_mir::VecReduce,
    names: &[String],
    name: &str,
    acc: &str,
) {
    match red.fma {
        Some((x, y)) => {
            let _ = writeln!(
                out,
                "  {name} = call float @llvm.fma.f32(float {}, float {}, float {acc})",
                names[x as usize], names[y as usize]
            );
        }
        None => emit_fold(out, red.op, name, acc, &names[red.value as usize]),
    }
}

/// `VecRedOp::fold(acc, x)`: `acc + x`, or the strict-compare select that mirrors `vmaxps`/`vminps`
/// (`acc` survives only when strictly greater/less, so ties and NaN pick `x`).
fn emit_fold(out: &mut String, op: VecRedOp, name: &str, acc: &str, x: &str) {
    match op {
        VecRedOp::Add => {
            let _ = writeln!(out, "  {name} = fadd float {acc}, {x}");
        }
        VecRedOp::Fmax | VecRedOp::Fmin => {
            let pred = if op == VecRedOp::Fmax { "ogt" } else { "olt" };
            let _ = writeln!(out, "  {name}.c = fcmp {pred} float {acc}, {x}");
            let _ = writeln!(out, "  {name} = select i1 {name}.c, float {acc}, float {x}");
        }
    }
}

/// Emit one evaluation of the recipe body for element `elem` (an `i64` operand). Value `i` is named
/// `%{prefix}{i}`, except that a `Splat` names its hoisted entry-block load and a `Const` is an
/// inline literal. Returns the operand spelling of every op (empty for a `Store`).
fn emit_body(out: &mut String, kernel: &VecKernel, prefix: &str, elem: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::with_capacity(kernel.ops.len());
    for (i, op) in kernel.ops.iter().enumerate() {
        let r = format!("%{prefix}{i}");
        let name = match *op {
            VecOp::Load { stream } => {
                let _ = writeln!(
                    out,
                    "  {r}.a = getelementptr float, ptr %base{stream}, i64 {elem}"
                );
                let _ = writeln!(out, "  {r} = load float, ptr {r}.a");
                r
            }
            VecOp::Splat { scalar } => format!("%scalar{scalar}"),
            VecOp::Const { bits } => f32_lit(f32::from_bits(bits)),
            VecOp::Bin { op, a, b } => {
                let inst = match op {
                    VecBin::Add => "fadd",
                    VecBin::Sub => "fsub",
                    VecBin::Mul => "fmul",
                    VecBin::Div => "fdiv",
                };
                let _ = writeln!(
                    out,
                    "  {r} = {inst} float {}, {}",
                    names[a as usize], names[b as usize]
                );
                r
            }
            VecOp::Fma { a, b, c } => {
                let _ = writeln!(
                    out,
                    "  {r} = call float @llvm.fma.f32(float {}, float {}, float {})",
                    names[a as usize], names[b as usize], names[c as usize]
                );
                r
            }
            VecOp::Sqrt { a } => {
                let _ = writeln!(
                    out,
                    "  {r} = call float @llvm.sqrt.f32(float {})",
                    names[a as usize]
                );
                r
            }
            VecOp::Neg { a } => {
                let _ = writeln!(out, "  {r} = fneg float {}", names[a as usize]);
                r
            }
            VecOp::Cmp { pred, a, b } => {
                let p = match pred {
                    VecCmp::Lt => "olt",
                    VecCmp::Le => "ole",
                    VecCmp::Gt => "ogt",
                    VecCmp::Ge => "oge",
                    VecCmp::Eq => "oeq",
                    VecCmp::Ne => "une",
                };
                let _ = writeln!(
                    out,
                    "  {r}.c = fcmp {p} float {}, {}",
                    names[a as usize], names[b as usize]
                );
                let _ = writeln!(
                    out,
                    "  {r} = select i1 {r}.c, float {}, float {}",
                    f32_lit(1.0),
                    f32_lit(0.0)
                );
                r
            }
            VecOp::Select { mask, a, b } => {
                // The evaluator picks `a` when the mask lane is non-zero (Rust `!=`, unordered).
                let _ = writeln!(
                    out,
                    "  {r}.c = fcmp une float {}, {}",
                    names[mask as usize],
                    f32_lit(0.0)
                );
                let _ = writeln!(
                    out,
                    "  {r} = select i1 {r}.c, float {}, float {}",
                    names[a as usize], names[b as usize]
                );
                r
            }
            VecOp::Store { stream, val } => {
                let _ = writeln!(
                    out,
                    "  {r}.a = getelementptr float, ptr %base{stream}, i64 {elem}"
                );
                let _ = writeln!(out, "  store float {}, ptr {r}.a", names[val as usize]);
                String::new()
            }
        };
        names.push(name);
    }
    names
}
