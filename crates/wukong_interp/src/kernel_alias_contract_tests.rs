//! The aliasing contract of the recognized-kernel arms, family by family.
//!
//! The interpreter runs a kernel over COPIES of its operands, which is copy-in/copy-out semantics:
//! on overlapping operands it would compute a value neither the native kernel (which reads what it
//! has itself just written) nor the source loop (which reads in its own order) computes. The
//! compiler keeps such calls away — `wukong_mir_build` versions every alias-uncertain dispatch on a
//! run-time overlap test whose overlapping arm is the source loop — and each arm checks the same
//! contract so that a call which bypassed the guard is REFUSED rather than answered.
//!
//! Three parts, all pinned here: every cross-element family refuses an output that overlaps an
//! input it is still reading (memory untouched); the families whose runtime entry documents EXACT
//! in-place operation accept it and produce the in-place source loop's value; and the purely
//! elementwise families — whose one meaning under any overlap is the source loop's — answer a
//! SHIFTED overlap the way that loop does, one element at a time on live memory. Every expected
//! value below is derived by hand from the scalar loop over the same inputs — small exact binary
//! fractions, so no rounding mode or reassociation can move them — never read back from a backend.
use super::*;
use Value::{Float as F, Int as I, Ptr as P};

fn interpreter<'a>(program: &'a Program, interner: &'a Interner) -> Interp<'a, 'a> {
    Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::new(),
        vecs: Vec::new(),
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel: None,
        depth: 0,
    }
}

/// One aliased call per cross-element family: the output overlaps an input the kernel is still
/// reading. Inputs are numeric everywhere (slot 28 holds an `i32` target), so the only possible
/// error is the overlap itself.
fn overlapping_calls() -> Vec<(&'static str, Vec<Value>)> {
    vec![
        // C over A.
        (
            "wukong_sgemm",
            vec![P(0), P(4), P(0), I(1), I(1), I(1), I(0)],
        ),
        (
            "wukong_sgemm_nt_parallel",
            vec![P(0), P(4), P(4), I(1), I(1), I(1), I(0)],
        ),
        // y over a; out over a (GEVM).
        ("wukong_sgemv", vec![P(0), P(4), P(0), I(1), I(1)]),
        (
            "wukong_sgevm_f32",
            vec![P(0), P(4), P(4), I(1), I(1), F(1.0)],
        ),
        // A SHIFTED transpose: dst starts one element into src.
        ("wukong_transpose_f32", vec![P(0), P(1), I(2), I(2)]),
        // The column sums land on row 0 while later rows are still read.
        ("wukong_colsum_f32_parallel", vec![P(0), P(1), I(2), I(2)]),
        ("wukong_colargmax_i32", vec![P(0), P(0), I(2), I(2)]),
        ("wukong_rowargmin_i32", vec![P(0), P(1), I(2), I(2)]),
        ("wukong_logsumexp_f32", vec![P(0), P(1), I(2), I(2)]),
        ("wukong_entropy_f32_parallel", vec![P(0), P(0), I(2), I(2)]),
        ("wukong_kldiv_f32", vec![P(0), P(4), P(4), I(1), I(2)]),
        // dx over y (softmax), over x (RMSNorm), over dy (LayerNorm forbids even that).
        ("wukong_softmax_bwd_f32", vec![P(0), P(4), P(0), I(1), I(2)]),
        (
            "wukong_rmsnorm_bwd_f32",
            vec![P(0), P(4), P(8), P(0), I(1), I(2), I(0)],
        ),
        (
            "wukong_layernorm_bwd_f32",
            vec![P(0), P(4), P(8), P(4), I(1), I(2), I(0)],
        ),
        // The loss over the logits.
        ("wukong_xent_fwd_f32", vec![P(0), P(28), P(0), I(1), I(2)]),
        // RoPE: the output over the frequency table.
        ("wukong_rope_f32", vec![P(0), P(8), P(8), I(1), I(1)]),
        // Scans: a SHIFTED output is not in place.
        ("wukong_cumsum_f32", vec![P(0), P(1), I(1), I(4)]),
        ("wukong_lrscan_f32", vec![P(0), P(8), P(1), I(1), I(4)]),
        // Pooling: a 1x3x4 plane, 2x2 window, strides 1x2 -> a 2x2 output written over the input.
        (
            "wukong_maxpool2d_f32",
            vec![P(0), P(2), I(1), I(3), I(4), I(2), I(2), I(1), I(2)],
        ),
        // Horner: the output over the coefficient table every element re-reads.
        ("wukong_vhorner_f32", vec![P(0), P(9), I(4), P(8), I(2)]),
        // Fused attention: the output over Q.
        (
            "wukong_attention_f32",
            vec![P(0), P(4), P(8), P(0), I(1), I(1), F(1.0), I(0)],
        ),
    ]
}

#[test]
fn cross_element_kernels_refuse_overlapping_operands_and_leave_memory_untouched() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for (name, args) in overlapping_calls() {
        interp.memory = vec![F(1.0); 32];
        interp.memory[28] = I(0);
        let before = interp.memory.clone();
        let error = interp
            .intrinsic(name, &args)
            .expect_err("an overlapping kernel call must be refused, not answered from a copy");
        assert!(
            error.contains(name) && error.contains("overlap"),
            "{name}: {error}"
        );
        assert_eq!(interp.memory, before, "{name} wrote memory before refusing");
    }
}

/// A SHIFTED overlap on an elementwise kernel is the source loop's program: each store lands before
/// the next element is read. `out == x + 1` therefore CHAINS — `out[t] = 2·x[t] + 1` over
/// `1, 2, 3, 4, 5` gives `1, 3, 7, 15, 31`, where a copied `x` (the old oracle) gives `1, 3, 5, 7, 9`
/// and the native 8-lane body gives the copied answer within a block. A BACKWARD shift reads ahead
/// of every store, so it equals the copy. Expected values are the scalar loop by hand.
#[test]
fn elementwise_kernels_run_the_source_loop_on_a_shifted_overlap() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);

    // velem, `out = 2*x + 1` (VE_ID, `y` unread), out = x + 1.
    interp.memory = [1.0, 2.0, 3.0, 4.0, 5.0].map(F).to_vec();
    let id = I(i128::from(wukong_runtime::VE_ID));
    interp
        .intrinsic(
            "wukong_velem_f32",
            &[P(0), P(0), P(1), I(4), F(2.0), F(0.0), F(1.0), id],
        )
        .unwrap();
    assert_eq!(interp.memory, [1.0, 3.0, 7.0, 15.0, 31.0].map(F));

    // Horner `r = 2; r = r*x + 1` (coefficients at slots 8..10), out = x + 1: the same chain.
    interp.memory = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 2.0, 1.0]
        .map(F)
        .to_vec();
    interp
        .intrinsic("wukong_vhorner_f32", &[P(0), P(1), I(4), P(8), I(2)])
        .unwrap();
    assert_eq!(&interp.memory[..5], &[1.0, 3.0, 7.0, 15.0, 31.0].map(F));

    // vmath ReLU, out = x + 1 over -1, 2, -3, 4, -5: out[0] = relu(-1) = 0 lands on x[1], so
    // every later element reads a zero -> -1, 0, 0, 0, 0 (a copied x gives -1, 0, 2, 0, 4).
    let relu = I(i128::from(wukong_runtime::VM_RELU));
    interp.memory = [-1.0, 2.0, -3.0, 4.0, -5.0].map(F).to_vec();
    interp
        .intrinsic("wukong_vmath_f32", &[P(0), P(1), I(4), relu])
        .unwrap();
    assert_eq!(interp.memory, [-1.0, 0.0, 0.0, 0.0, 0.0].map(F));

    // A BACKWARD shift, out = x - 1 over 9, -1, 2, -3, 4: every read is ahead of every store, so
    // the loop and the copy agree -> 0, 2, 0, 4, 4.
    interp.memory = [9.0, -1.0, 2.0, -3.0, 4.0].map(F).to_vec();
    interp
        .intrinsic("wukong_vmath_f32_parallel", &[P(1), P(0), I(4), relu])
        .unwrap();
    assert_eq!(interp.memory, [0.0, 2.0, 0.0, 4.0, 4.0].map(F));

    // A bad input slot is a diagnostic BEFORE the first store, shifted overlap or not.
    interp.memory = [1.0, 2.0, 3.0, 4.0, 5.0].map(F).to_vec();
    interp.memory[3] = Value::Unit;
    let before = interp.memory.clone();
    assert!(interp
        .intrinsic("wukong_vmath_f32", &[P(0), P(1), I(4), relu])
        .is_err());
    assert_eq!(interp.memory, before);
}

/// The disjoint twins of the refused calls above still run: the refusal is about overlap, not
/// about the family.
#[test]
fn the_same_calls_on_disjoint_operands_still_run() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for (name, args) in [
        (
            "wukong_sgemm",
            vec![P(0), P(4), P(16), I(1), I(1), I(1), I(0)],
        ),
        ("wukong_transpose_f32", vec![P(0), P(16), I(2), I(2)]),
        ("wukong_colsum_f32_parallel", vec![P(0), P(16), I(2), I(2)]),
        ("wukong_cumsum_f32", vec![P(0), P(16), I(1), I(4)]),
        (
            "wukong_velem_f32",
            vec![P(0), P(8), P(16), I(4), F(1.0), F(0.0), F(0.0), I(0)],
        ),
    ] {
        interp.memory = vec![F(1.0); 32];
        interp
            .intrinsic(name, &args)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// Exact in-place operation where the runtime entry documents it, with the in-place SOURCE LOOP's
/// value: each row (or element) is read whole before it is written, so in place equals out of
/// place, and the copy the interpreter runs is unobservable.
#[test]
fn documented_exact_in_place_operation_matches_the_in_place_source_loop() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);

    // cumsum, `out == x`, one row of four: acc += x[i]; out[i] = acc  ->  1, 3, 6, 10.
    interp.memory = vec![F(1.0), F(2.0), F(3.0), F(4.0)];
    interp
        .intrinsic("wukong_cumsum_f32", &[P(0), P(0), I(1), I(4)])
        .unwrap();
    assert_eq!(interp.memory, [F(1.0), F(3.0), F(6.0), F(10.0)]);

    // cummax, `out == x`: m = max(m, x[i]); out[i] = m  ->  3, 3, 4, 4.
    interp.memory = vec![F(3.0), F(1.0), F(4.0), F(1.0)];
    interp
        .intrinsic("wukong_cummax_f32_parallel", &[P(0), P(0), I(1), I(4)])
        .unwrap();
    assert_eq!(interp.memory, [F(3.0), F(3.0), F(4.0), F(4.0)]);

    // lrscan, `out == a`, one row: h = a[t]*h + b[t]; out[t] = h with a = 0.5 and b = 1, 2, 3:
    // h = 1, then 0.5*1 + 2 = 2.5, then 0.5*2.5 + 3 = 4.25.
    interp.memory = vec![F(0.5), F(0.5), F(0.5), F(1.0), F(2.0), F(3.0)];
    interp
        .intrinsic("wukong_lrscan_f32", &[P(0), P(3), P(0), I(1), I(3)])
        .unwrap();
    assert_eq!(&interp.memory[..3], &[F(1.0), F(2.5), F(4.25)]);

    // softmax backward, `dx == dy`, one row: s = Σ y*dy = 0.25*1 + 0.75*2 = 1.75;
    // dx = y*(dy - s) = 0.25*(1 - 1.75), 0.75*(2 - 1.75) = -0.1875, 0.1875.
    interp.memory = vec![F(0.25), F(0.75), F(1.0), F(2.0)];
    interp
        .intrinsic("wukong_softmax_bwd_f32", &[P(0), P(2), P(2), I(1), I(2)])
        .unwrap();
    assert_eq!(&interp.memory[2..], &[F(-0.1875), F(0.1875)]);

    // velem, `out == x`, out = 2*x + 1*y + 0.5 with x = 1, 2 and y = 10, 20  ->  12.5, 24.5.
    interp.memory = vec![F(1.0), F(2.0), F(10.0), F(20.0)];
    let op = I(i128::from(wukong_runtime::VE_ID | wukong_runtime::VE_USE_Y));
    interp
        .intrinsic(
            "wukong_velem_f32",
            &[P(0), P(2), P(0), I(2), F(2.0), F(1.0), F(0.5), op],
        )
        .unwrap();
    assert_eq!(&interp.memory[..2], &[F(12.5), F(24.5)]);
}
