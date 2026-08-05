//! **The shape-typed surface must cost nothing.**
//!
//! `Tensor[f32, M, N]` and `[f32; M*N]` describe the same bytes; `a[i, j]` and `a[i*N + j]` address
//! the same element. So the two spellings must compile to the same machine code — and they did not:
//! every kernel recognizer and the whole autovectorizer match a *single-index* `ExprKind::Index`, so
//! a 2-index access was invisible to all of them and lowered to a scalar gep nest. Measured on a
//! release native build, the tensor spelling of one elementwise loop ran **5.7x slower** than the
//! flat spelling of the same arithmetic over the same memory.
//!
//! These tests pin the fix at the strongest available level: **byte-identical `--emit=mir -O2`**.
//! Not "also vectorizes", not "within N%" — identical text. Any future change that makes a
//! recognizer or the vectorizer see through one spelling but not the other fails here.
//!
//! They deliberately use the real binary and the real `--emit=mir` printer rather than poking at
//! `mir_build` internals, because "the two spellings produce the same program" is exactly the
//! user-visible property, and the printer is what makes it checkable.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

fn scratch(name: &str, src: &str) -> PathBuf {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir()
        .join("wukongc_tensor_parity")
        .join(format!("{}_{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let path = dir.join(format!("{name}.wk"));
    std::fs::write(&path, src).expect("write fixture");
    path
}

fn emit_mir(src: &str, name: &str) -> String {
    let path = scratch(name, src);
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--emit=mir")
        .arg("-O2")
        .arg(&path)
        .output()
        .expect("spawn wukongc");
    assert!(
        out.status.success(),
        "{name}: --emit=mir -O2 failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n")
}

/// Assert two sources lower to the same MIR, reporting the first differing line when they do not.
fn assert_same_mir(tensor_src: &str, flat_src: &str, what: &str) {
    let t = emit_mir(tensor_src, "tensor");
    let a = emit_mir(flat_src, "flat");
    if t != a {
        let first = t
            .lines()
            .zip(a.lines())
            .position(|(x, y)| x != y)
            .unwrap_or(0);
        let ctx = |s: &str| {
            s.lines()
                .skip(first.saturating_sub(3))
                .take(9)
                .collect::<Vec<_>>()
                .join("\n")
        };
        panic!(
            "{what}: the shape-typed spelling no longer lowers to the same MIR as the flat one.\n\
             first difference at line {first}\n--- tensor ---\n{}\n--- flat ---\n{}",
            ctx(&t),
            ctx(&a)
        );
    }
}

/// A 2-D elementwise map: the canonical shape-typed loop nest. This is the exact program whose
/// tensor spelling was 5.7x slower than its flat twin.
#[test]
fn elementwise_2d_tensor_matches_flat_array() {
    let tensor = r#"module p
fn ew(a: Tensor[f32, 64, 64], b: Tensor[f32, 64, 64], mut out: Tensor[f32, 64, 64]) {
    for i in 0..64 { for j in 0..64 { out[i, j] = a[i, j] * 2.0 + b[i, j]; } }
}
fn main() -> i32 {
    let a: [f32; 4096] = [1.5; 4096];
    let b: [f32; 4096] = [0.25; 4096];
    let mut c: [f32; 4096] = [0.0; 4096];
    ew(a, b, c);
    print(c[0] as i32);
    return 0;
}
"#;
    let flat = r#"module p
fn ew(a: [f32; 4096], b: [f32; 4096], mut out: [f32; 4096]) {
    for i in 0..64 { for j in 0..64 { out[i * 64 + j] = a[i * 64 + j] * 2.0 + b[i * 64 + j]; } }
}
fn main() -> i32 {
    let a: [f32; 4096] = [1.5; 4096];
    let b: [f32; 4096] = [0.25; 4096];
    let mut c: [f32; 4096] = [0.0; 4096];
    ew(a, b, c);
    print(c[0] as i32);
    return 0;
}
"#;
    assert_same_mir(tensor, flat, "2-D elementwise");
}

/// A matmul nest — the GEMM recognizer's own shape. It already accepted a 2-index operand through a
/// dedicated branch, so this pins that the *normalized* form still dispatches (and to the same
/// kernel arguments), rather than quietly falling back to a scalar triple loop.
#[test]
fn matmul_tensor_matches_flat_array() {
    let tensor = r#"module p
fn mm(a: Tensor[f32, 32, 32], b: Tensor[f32, 32, 32], mut c: Tensor[f32, 32, 32]) {
    for i in 0..32 { for j in 0..32 {
        let mut s: f32 = 0.0;
        for k in 0..32 { s = s + a[i, k] * b[k, j]; }
        c[i, j] = s;
    } }
}
fn main() -> i32 {
    let a: [f32; 1024] = [1.0; 1024];
    let b: [f32; 1024] = [2.0; 1024];
    let mut c: [f32; 1024] = [0.0; 1024];
    mm(a, b, c);
    print(c[0] as i32);
    return 0;
}
"#;
    let flat = r#"module p
fn mm(a: [f32; 1024], b: [f32; 1024], mut c: [f32; 1024]) {
    for i in 0..32 { for j in 0..32 {
        let mut s: f32 = 0.0;
        for k in 0..32 { s = s + a[i * 32 + k] * b[k * 32 + j]; }
        c[i * 32 + j] = s;
    } }
}
fn main() -> i32 {
    let a: [f32; 1024] = [1.0; 1024];
    let b: [f32; 1024] = [2.0; 1024];
    let mut c: [f32; 1024] = [0.0; 1024];
    mm(a, b, c);
    print(c[0] as i32);
    return 0;
}
"#;
    assert_same_mir(tensor, flat, "matmul");
}

/// Rank 3, and a non-square shape, so the row-major stride chain (`i*(N*P) + j*P + k`) is exercised
/// rather than the single-stride rank-2 case.
#[test]
fn rank3_tensor_matches_flat_array() {
    let tensor = r#"module p
fn ew(a: Tensor[f32, 4, 8, 16], mut out: Tensor[f32, 4, 8, 16]) {
    for i in 0..4 { for j in 0..8 { for k in 0..16 { out[i, j, k] = a[i, j, k] * 3.0; } } }
}
fn main() -> i32 {
    let a: [f32; 512] = [1.5; 512];
    let mut c: [f32; 512] = [0.0; 512];
    ew(a, c);
    print(c[511] as i32);
    return 0;
}
"#;
    let flat = r#"module p
fn ew(a: [f32; 512], mut out: [f32; 512]) {
    for i in 0..4 { for j in 0..8 { for k in 0..16 {
        out[i * 128 + j * 16 + k] = a[i * 128 + j * 16 + k] * 3.0;
    } } }
}
fn main() -> i32 {
    let a: [f32; 512] = [1.5; 512];
    let mut c: [f32; 512] = [0.0; 512];
    ew(a, c);
    print(c[511] as i32);
    return 0;
}
"#;
    assert_same_mir(tensor, flat, "rank-3 elementwise");
}

/// The whole point, stated as a property of the shipped fixtures: the shape-typed transformer block
/// must dispatch the same runtime kernels as the hand-flattened one. Before the fix it dispatched
/// **none** of them. (The two files are not MIR-identical — a rank-2 tensor cannot be swept by a
/// single flat index, so the two residual adds are 8x8 nests instead of one 64-element stream — so
/// this checks the dispatch set, which is what carries the performance.)
#[test]
fn transformer_block_tensor_dispatches_the_same_kernels() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run");
    let mir = |f: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .arg("--emit=mir")
            .arg("-O2")
            .arg(root.join(f))
            .output()
            .expect("spawn wukongc");
        assert!(out.status.success(), "{f}: --emit=mir -O2 failed");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let count = |s: &str, needle: &str| s.matches(needle).count();
    let flat = mir("transformer_block.wk");
    let tens = mir("transformer_block_tensor.wk");
    for kernel in ["wukong_sgemm(", "wukong_sgemm_nt(", "wukong_norm_f32("] {
        let (a, b) = (count(&flat, kernel), count(&tens, kernel));
        assert!(a > 0, "the flat fixture no longer dispatches {kernel}");
        assert_eq!(
            a, b,
            "shape-typed transformer block dispatches {kernel} {b} times, flat one {a}"
        );
    }
    // Both residual adds must still be vectorized — as a streaming `velem` call in the flat file,
    // as a per-row 256-bit `veckernel` in the shape-typed one. Neither may fall back to scalar.
    assert!(
        count(&tens, "veckernel #") + count(&tens, "wukong_velem_f32(") >= 2,
        "shape-typed transformer block lost a vectorized residual add"
    );
}
