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
//!   accumulator copies of `L` lanes (`U = reduction_unroll()`, the same single-sourced number the
//!   AVX2 assembler uses, and `L = VecLane::lanes()`: 8 for a 32-bit lane, 4 for a 64-bit one), the
//!   unrolled main loop, the single-group cleanup, the copy combine into copy 0 and the in-order
//!   lane `0..L` horizontal fold.
//!
//! Every value has the recipe's lane type ([`wukong_mir::VecLane`]: `float`, `double`, `i32` or
//! `i64`) with exactly the arithmetic its `VecValue` implementation defines: a float `Fma` is
//! `llvm.fma.*` (one rounding, like `mul_add`), a float `Neg` is `fneg` (a sign-bit flip), each float
//! compare uses the predicate of the Rust operator the evaluator applies (`Ne` is unordered), and an
//! integer lane wraps (plain `add`/`sub`/`mul`, no `nsw`) and compares signed. The kernel ABI is the
//! recipe's `fn(ptrs: *const *mut u8, scalars: *const T, n: u64) [-> T]`.
//!
//! Until the f64/i32/i64 recipes, every definition here was spelled in `float`: an `f64` reduction
//! was *defined* returning `float` and computed in single precision, while its call site expected a
//! `double` — a call through a mismatched function type that the opaque-pointer verifier accepts.

use std::fmt::Write as _;

use wukong_mir::{VecBin, VecCmp, VecKernel, VecLane, VecOp, VecRedOp};

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
            "LLVM emission [{}]: invalid vector kernel recipe: {}",
            crate::DECLINE_VEC_KERNEL,
            errors.join("; ")
        ))
    }
}

/// The LLVM type of one lane (and of a reduction kernel's result).
pub(crate) fn lane_ty(lane: VecLane) -> &'static str {
    match lane {
        VecLane::F32 => "float",
        VecLane::F64 => "double",
        VecLane::I32 => "i32",
        VecLane::I64 => "i64",
    }
}

/// The overload suffix of a float lane's math intrinsics.
fn float_suffix(lane: VecLane) -> &'static str {
    if lane == VecLane::F64 {
        "f64"
    } else {
        "f32"
    }
}

/// The math intrinsics a recipe calls, as `(name, lane type, arity)`, so the module can declare
/// them. Only a float lane calls any (validation keeps `Fma`/`Sqrt` off the integer lanes).
pub(crate) fn intrinsics(kernel: &VecKernel) -> Vec<(String, &'static str, usize)> {
    if !kernel.lane.is_float() {
        return Vec::new();
    }
    let ty = lane_ty(kernel.lane);
    let sfx = float_suffix(kernel.lane);
    let mut out = Vec::new();
    if kernel.ops.iter().any(|o| matches!(o, VecOp::Fma { .. }))
        || kernel.reduce.is_some_and(|r| r.fma.is_some())
    {
        out.push((format!("llvm.fma.{sfx}"), ty, 3));
    }
    if kernel.ops.iter().any(|o| matches!(o, VecOp::Sqrt { .. })) {
        out.push((format!("llvm.sqrt.{sfx}"), ty, 1));
    }
    out
}

/// LLVM's spelling of an `f32` constant: the bits of the exactly widened double.
fn f32_lit(x: f32) -> String {
    format!("0x{:016X}", (x as f64).to_bits())
}

/// LLVM's spelling of an `f64` constant.
fn f64_lit(x: f64) -> String {
    format!("0x{:016X}", x.to_bits())
}

/// A lane-typed literal of the float value `x` (exact in the lane type) or the integer `i`.
fn lit(lane: VecLane, x: f64, i: i64) -> String {
    match lane {
        VecLane::F32 => f32_lit(x as f32),
        VecLane::F64 => f64_lit(x),
        VecLane::I32 | VecLane::I64 => i.to_string(),
    }
}

/// A `Const` immediate (`VecValue::from_const_bits`): the f32 bit pattern, widened exactly for an
/// `f64` lane, read as a (sign-extended) two's-complement integer on an integer lane. Validation
/// admits `Const` on the f32 lane only; the others are spelled for totality.
fn const_lit(lane: VecLane, bits: u32) -> String {
    match lane {
        VecLane::F32 => f32_lit(f32::from_bits(bits)),
        VecLane::F64 => f64_lit(f32::from_bits(bits) as f64),
        VecLane::I32 | VecLane::I64 => (bits as i32).to_string(),
    }
}

/// The fold identity (`VecValue::identity`): 0 for a sum, the lane's lowest/highest value for a
/// running max/min.
fn identity(lane: VecLane, op: VecRedOp) -> String {
    match (op, lane) {
        (VecRedOp::Add, _) => lit(lane, 0.0, 0),
        (VecRedOp::Fmax, VecLane::F32 | VecLane::F64) => lit(lane, f64::NEG_INFINITY, 0),
        (VecRedOp::Fmin, VecLane::F32 | VecLane::F64) => lit(lane, f64::INFINITY, 0),
        (VecRedOp::Fmax, VecLane::I32) => i32::MIN.to_string(),
        (VecRedOp::Fmin, VecLane::I32) => i32::MAX.to_string(),
        (VecRedOp::Fmax, VecLane::I64) => i64::MIN.to_string(),
        (VecRedOp::Fmin, VecLane::I64) => i64::MAX.to_string(),
    }
}

/// Emit the full definition of `kernel` under `symbol`.
pub(crate) fn emit_definition(out: &mut String, symbol: &str, kernel: &VecKernel) {
    let ty = lane_ty(kernel.lane);
    let ret = if kernel.reduce.is_some() { ty } else { "void" };
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
            "  %scalar{c}.p = getelementptr {ty}, ptr %scalars, i64 {c}"
        );
        let _ = writeln!(out, "  %scalar{c} = load {ty}, ptr %scalar{c}.p");
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
    let lane = kernel.lane;
    let ty = lane_ty(lane);
    let lanes = lane.lanes();
    let copies = kernel.reduction_unroll();
    let slots = copies * lanes;
    let identity = identity(lane, red.op);

    let _ = writeln!(out, "  %acc = alloca [{slots} x {ty}]");
    for j in 0..slots {
        let _ = writeln!(out, "  %acc{j}.p = getelementptr {ty}, ptr %acc, i64 {j}");
        let _ = writeln!(out, "  store {ty} {identity}, ptr %acc{j}.p");
    }
    out.push_str("  br label %main.head\n");

    // Main loop: while i + U*L <= n, fold element i + j into accumulator slot j for j in 0..U*L.
    // The slot order inside one step is immaterial (the slots are independent and the body stores
    // nothing); what matters is that slot j only ever sees elements congruent to j mod U*L, in
    // ascending order, which is exactly the evaluator's `acc[cu * L + lane]` schedule.
    out.push_str("main.head:\n");
    out.push_str("  %i = phi i64 [ 0, %entry ], [ %i.end, %main.latch ]\n");
    let _ = writeln!(out, "  %i.end = add i64 %i, {slots}");
    out.push_str("  %i.fits = icmp ule i64 %i.end, %n\n");
    out.push_str("  br i1 %i.fits, label %main.lane, label %tail.head\n");
    out.push_str("main.lane:\n");
    out.push_str("  %j = phi i64 [ 0, %main.head ], [ %j.next, %main.lane ]\n");
    out.push_str("  %e = add i64 %i, %j\n");
    let names = emit_body(out, kernel, "m", "%e");
    let _ = writeln!(out, "  %j.p = getelementptr {ty}, ptr %acc, i64 %j");
    let _ = writeln!(out, "  %j.old = load {ty}, ptr %j.p");
    emit_addend(out, lane, &red, &names, "%m.acc", "%j.old");
    let _ = writeln!(out, "  store {ty} %m.acc, ptr %j.p");
    out.push_str("  %j.next = add i64 %j, 1\n");
    let _ = writeln!(out, "  %j.more = icmp ult i64 %j.next, {slots}");
    out.push_str("  br i1 %j.more, label %main.lane, label %main.latch\n");
    out.push_str("main.latch:\n  br label %main.head\n");

    // Cleanup: remaining full L-element groups fold into copy 0.
    out.push_str("tail.head:\n");
    out.push_str("  %t = phi i64 [ %i, %main.head ], [ %t.end, %tail.latch ]\n");
    let _ = writeln!(out, "  %t.end = add i64 %t, {lanes}");
    out.push_str("  %t.fits = icmp ule i64 %t.end, %n\n");
    out.push_str("  br i1 %t.fits, label %tail.lane, label %fold\n");
    out.push_str("tail.lane:\n");
    out.push_str("  %l = phi i64 [ 0, %tail.head ], [ %l.next, %tail.lane ]\n");
    out.push_str("  %te = add i64 %t, %l\n");
    let names = emit_body(out, kernel, "c", "%te");
    let _ = writeln!(out, "  %l.p = getelementptr {ty}, ptr %acc, i64 %l");
    let _ = writeln!(out, "  %l.old = load {ty}, ptr %l.p");
    emit_addend(out, lane, &red, &names, "%c.acc", "%l.old");
    let _ = writeln!(out, "  store {ty} %c.acc, ptr %l.p");
    out.push_str("  %l.next = add i64 %l, 1\n");
    let _ = writeln!(out, "  %l.more = icmp ult i64 %l.next, {lanes}");
    out.push_str("  br i1 %l.more, label %tail.lane, label %tail.latch\n");
    out.push_str("tail.latch:\n  br label %tail.head\n");

    // Combine copy cu into copy 0 lane-wise (cu ascending), then fold lanes 0..L in order.
    out.push_str("fold:\n");
    for j in 0..slots {
        let _ = writeln!(out, "  %f{j} = load {ty}, ptr %acc{j}.p");
    }
    let mut lane_value: Vec<String> = (0..lanes).map(|l| format!("%f{l}")).collect();
    for cu in 1..copies {
        for l in 0..lanes {
            let name = format!("%g{cu}.{l}");
            let other = format!("%f{}", cu * lanes + l);
            emit_fold(out, lane, red.op, &name, &lane_value[l as usize], &other);
            lane_value[l as usize] = name;
        }
    }
    let mut r = lane_value[0].clone();
    for (l, value) in lane_value.iter().enumerate().skip(1) {
        let name = format!("%h{l}");
        emit_fold(out, lane, red.op, &name, &r, value);
        r = name;
    }
    let _ = writeln!(out, "  ret {ty} {r}");
}

/// `acc = fma(X, Y, acc)` for a fused reduction, else `acc = fold(acc, value)`.
fn emit_addend(
    out: &mut String,
    lane: VecLane,
    red: &wukong_mir::VecReduce,
    names: &[String],
    name: &str,
    acc: &str,
) {
    match red.fma {
        Some((x, y)) => emit_fused(out, lane, name, &names[x as usize], &names[y as usize], acc),
        None => emit_fold(out, lane, red.op, name, acc, &names[red.value as usize]),
    }
}

/// `name = a * b + c` (`VecValue::fused`): one rounding for a float lane, a wrapping multiply and
/// add for an integer one.
fn emit_fused(out: &mut String, lane: VecLane, name: &str, a: &str, b: &str, c: &str) {
    let ty = lane_ty(lane);
    if lane.is_float() {
        let sfx = float_suffix(lane);
        let _ = writeln!(
            out,
            "  {name} = call {ty} @llvm.fma.{sfx}({ty} {a}, {ty} {b}, {ty} {c})"
        );
    } else {
        let _ = writeln!(out, "  {name}.p = mul {ty} {a}, {b}");
        let _ = writeln!(out, "  {name} = add {ty} {name}.p, {c}");
    }
}

/// `VecValue::fold(acc, x)`: `acc + x`, or the strict-compare select that mirrors `vmaxps`/`vminps`
/// (`acc` survives only when strictly greater/less, so ties and NaN pick `x`).
fn emit_fold(out: &mut String, lane: VecLane, op: VecRedOp, name: &str, acc: &str, x: &str) {
    let ty = lane_ty(lane);
    match op {
        VecRedOp::Add => {
            let add = if lane.is_float() { "fadd" } else { "add" };
            let _ = writeln!(out, "  {name} = {add} {ty} {acc}, {x}");
        }
        VecRedOp::Fmax | VecRedOp::Fmin => {
            let cmp = match (lane.is_float(), op == VecRedOp::Fmax) {
                (true, true) => "fcmp ogt",
                (true, false) => "fcmp olt",
                (false, true) => "icmp sgt",
                (false, false) => "icmp slt",
            };
            let _ = writeln!(out, "  {name}.c = {cmp} {ty} {acc}, {x}");
            let _ = writeln!(out, "  {name} = select i1 {name}.c, {ty} {acc}, {ty} {x}");
        }
    }
}

/// Emit one evaluation of the recipe body for element `elem` (an `i64` operand). Value `i` is named
/// `%{prefix}{i}`, except that a `Splat` names its hoisted entry-block load, a `Const` is an inline
/// literal and an integer `Sqrt` (the evaluator's identity) names its operand. Returns the operand
/// spelling of every op (empty for a `Store`).
fn emit_body(out: &mut String, kernel: &VecKernel, prefix: &str, elem: &str) -> Vec<String> {
    let lane = kernel.lane;
    let ty = lane_ty(lane);
    let float = lane.is_float();
    let mut names: Vec<String> = Vec::with_capacity(kernel.ops.len());
    for (i, op) in kernel.ops.iter().enumerate() {
        let r = format!("%{prefix}{i}");
        let name = match *op {
            VecOp::Load { stream } => {
                let _ = writeln!(
                    out,
                    "  {r}.a = getelementptr {ty}, ptr %base{stream}, i64 {elem}"
                );
                let _ = writeln!(out, "  {r} = load {ty}, ptr {r}.a");
                r
            }
            VecOp::Splat { scalar } => format!("%scalar{scalar}"),
            VecOp::Const { bits } => const_lit(lane, bits),
            VecOp::Bin {
                op: VecBin::Div,
                a,
                b,
            } if !float => {
                // `VecValue::bin(Div)` on an integer lane: 0 for a zero divisor, `MIN / -1` wraps.
                // Validation refuses it; spelled anyway so a stray one is never LLVM poison.
                let (a, b) = (&names[a as usize], &names[b as usize]);
                let _ = writeln!(out, "  {r}.z = icmp eq {ty} {b}, 0");
                let _ = writeln!(out, "  {r}.m = icmp eq {ty} {b}, -1");
                let _ = writeln!(out, "  {r}.bad = or i1 {r}.z, {r}.m");
                let _ = writeln!(out, "  {r}.d = select i1 {r}.bad, {ty} 1, {ty} {b}");
                let _ = writeln!(out, "  {r}.q = sdiv {ty} {a}, {r}.d");
                let _ = writeln!(out, "  {r}.n = sub {ty} 0, {a}");
                let _ = writeln!(out, "  {r}.w = select i1 {r}.m, {ty} {r}.n, {ty} {r}.q");
                let _ = writeln!(out, "  {r} = select i1 {r}.z, {ty} 0, {ty} {r}.w");
                r
            }
            VecOp::Bin { op, a, b } => {
                let inst = match (op, float) {
                    (VecBin::Add, true) => "fadd",
                    (VecBin::Sub, true) => "fsub",
                    (VecBin::Mul, true) => "fmul",
                    (VecBin::Div, _) => "fdiv",
                    (VecBin::Add, false) => "add",
                    (VecBin::Sub, false) => "sub",
                    (VecBin::Mul, false) => "mul",
                };
                let _ = writeln!(
                    out,
                    "  {r} = {inst} {ty} {}, {}",
                    names[a as usize], names[b as usize]
                );
                r
            }
            VecOp::Fma { a, b, c } => {
                emit_fused(
                    out,
                    lane,
                    &r,
                    &names[a as usize],
                    &names[b as usize],
                    &names[c as usize],
                );
                r
            }
            VecOp::Sqrt { a } if !float => names[a as usize].clone(),
            VecOp::Sqrt { a } => {
                let _ = writeln!(
                    out,
                    "  {r} = call {ty} @llvm.sqrt.{}({ty} {})",
                    float_suffix(lane),
                    names[a as usize]
                );
                r
            }
            VecOp::Neg { a } => {
                if float {
                    let _ = writeln!(out, "  {r} = fneg {ty} {}", names[a as usize]);
                } else {
                    let _ = writeln!(out, "  {r} = sub {ty} 0, {}", names[a as usize]);
                }
                r
            }
            VecOp::Cmp { pred, a, b } => {
                let p = match (pred, float) {
                    (VecCmp::Lt, true) => "fcmp olt",
                    (VecCmp::Le, true) => "fcmp ole",
                    (VecCmp::Gt, true) => "fcmp ogt",
                    (VecCmp::Ge, true) => "fcmp oge",
                    (VecCmp::Eq, true) => "fcmp oeq",
                    (VecCmp::Ne, true) => "fcmp une",
                    (VecCmp::Lt, false) => "icmp slt",
                    (VecCmp::Le, false) => "icmp sle",
                    (VecCmp::Gt, false) => "icmp sgt",
                    (VecCmp::Ge, false) => "icmp sge",
                    (VecCmp::Eq, false) => "icmp eq",
                    (VecCmp::Ne, false) => "icmp ne",
                };
                let _ = writeln!(
                    out,
                    "  {r}.c = {p} {ty} {}, {}",
                    names[a as usize], names[b as usize]
                );
                let _ = writeln!(
                    out,
                    "  {r} = select i1 {r}.c, {ty} {}, {ty} {}",
                    lit(lane, 1.0, 1),
                    lit(lane, 0.0, 0)
                );
                r
            }
            VecOp::Select { mask, a, b } => {
                // The evaluator picks `a` when the mask lane is non-zero (Rust `!=`, unordered).
                let test = if float { "fcmp une" } else { "icmp ne" };
                let _ = writeln!(
                    out,
                    "  {r}.c = {test} {ty} {}, {}",
                    names[mask as usize],
                    lit(lane, 0.0, 0)
                );
                let _ = writeln!(
                    out,
                    "  {r} = select i1 {r}.c, {ty} {}, {ty} {}",
                    names[a as usize], names[b as usize]
                );
                r
            }
            VecOp::Store { stream, val } => {
                let _ = writeln!(
                    out,
                    "  {r}.a = getelementptr {ty}, ptr %base{stream}, i64 {elem}"
                );
                let _ = writeln!(out, "  store {ty} {}, ptr {r}.a", names[val as usize]);
                String::new()
            }
        };
        names.push(name);
    }
    names
}
