//! Empty GEMM contractions preserve every store in plain, residual, and fused source nests.
use std::path::Path;
use std::process::Command;

#[derive(Clone, Copy, Debug)]
enum Case {
    Bf16Nt,
    Bf16Tn,
    F16Nt,
    F16Tn,
    I8Whole,
    I8Statement,
    Residual,
    CrossResidual,
    DualMultiply,
    DualAdd,
    F32Epilogue,
    Bf16Epilogue,
    F16Epilogue,
    I8Dequant,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::Bf16Nt => "bf16_nt",
            Self::Bf16Tn => "bf16_tn",
            Self::F16Nt => "f16_nt",
            Self::F16Tn => "f16_tn",
            Self::I8Whole => "i8_whole",
            Self::I8Statement => "i8_statement",
            Self::Residual => "residual",
            Self::CrossResidual => "cross_residual",
            Self::DualMultiply => "dual_multiply",
            Self::DualAdd => "dual_add",
            Self::F32Epilogue => "f32_epilogue",
            Self::Bf16Epilogue => "bf16_epilogue",
            Self::F16Epilogue => "f16_epilogue",
            Self::I8Dequant => "i8_dequant",
        }
    }

    fn input_types(self) -> (&'static str, &'static str) {
        match self {
            Self::Bf16Nt | Self::Bf16Tn | Self::Bf16Epilogue => ("bf16", "bf16"),
            Self::F16Nt | Self::F16Tn | Self::F16Epilogue => ("f16", "f16"),
            Self::I8Whole | Self::I8Statement | Self::I8Dequant => ("u8", "i8"),
            _ => ("f32", "f32"),
        }
    }

    fn accumulator_type(self) -> &'static str {
        if matches!(self, Self::I8Whole | Self::I8Statement | Self::I8Dequant) {
            "i32"
        } else {
            "f32"
        }
    }

    fn output_type(self) -> &'static str {
        if matches!(self, Self::I8Whole | Self::I8Statement) {
            "i32"
        } else {
            "f32"
        }
    }

    fn dual(self) -> bool {
        matches!(self, Self::DualMultiply | Self::DualAdd)
    }

    fn kernel(self) -> &'static str {
        match self {
            Self::Bf16Nt => "wukong_sgemm_bf16_nt",
            Self::Bf16Tn => "wukong_sgemm_bf16_tn",
            Self::F16Nt => "wukong_sgemm_f16_nt",
            Self::F16Tn => "wukong_sgemm_f16_tn",
            Self::I8Whole | Self::I8Statement => "wukong_i8gemm_nt",
            Self::Residual | Self::F32Epilogue => "wukong_sgemm_nt_epi",
            Self::CrossResidual | Self::DualMultiply | Self::DualAdd => "wukong_sgemm_nt",
            Self::Bf16Epilogue => "wukong_sgemm_bf16_nt_epi",
            Self::F16Epilogue => "wukong_sgemm_f16_nt_epi",
            Self::I8Dequant => "wukong_i8gemm_nt_deq",
        }
    }

    fn function(self, parallel: bool, suffix: &str, bound: &str, param: &str) -> String {
        let attr = if parallel { "@parallel\n" } else { "" };
        let (at, bt) = self.input_types();
        let acc = self.accumulator_type();
        let zero = if acc == "i32" { "0" } else { "0.0" };
        let output = if matches!(self, Self::I8Dequant) {
            "out"
        } else {
            "c"
        };
        let (ai, bi) = if matches!(self, Self::Bf16Tn | Self::F16Tn) {
            ("p*2+i".to_owned(), "p*2+j".to_owned())
        } else {
            (format!("i*({bound})+p"), format!("j*({bound})+p"))
        };
        let product = if at == "f32" {
            format!("a[{ai}]*b[{bi}]")
        } else {
            format!("(a[{ai}] as {acc})*(b[{bi}] as {acc})")
        };
        let prelude = match self {
            // Keep the accumulator local and dead after dequantization so fusion is eligible.
            Self::I8Dequant => "let mut c: [i32; 4] = [41,43,47,53]; let scale_a: f32 = 2.0;",
            // Force these through the statement path, including the separate i8 entry.
            Self::I8Statement | Self::CrossResidual | Self::DualMultiply | Self::DualAdd => {
                "let marker: i32 = 0;"
            }
            _ => "",
        };
        let store = match self {
            Self::Residual => "c[i*2+j] = fmax(c[i*2+j]+s+bias[j], 0.0);",
            Self::CrossResidual => "c[i*2+j] = x[i*2+j]+s;",
            Self::DualMultiply => "c[i*2+j] = s; d[i*2+j] = d[i*2+j]*s;",
            Self::DualAdd => "c[i*2+j] = s; d[i*2+j] = d[i*2+j]+s;",
            _ => "c[i*2+j] = s;",
        };
        let epilogue = match self {
            Self::F32Epilogue | Self::Bf16Epilogue | Self::F16Epilogue =>
                "for i in 0..2 { for j in 0..2 { c[i*2+j] = fmax(c[i*2+j]+bias[j], 0.0); } }",
            Self::I8Dequant =>
                "for i in 0..2 { for j in 0..2 { out[i*2+j] = fmax((c[i*2+j] as f32)*scale_a*scale_b[j]+bias[j], 0.0); } }",
            _ => "",
        };
        format!(
            "{attr}fn {}_{suffix}(a: [{at}; 2], b: [{bt}; 2], mut {output}: [{}; 4], mut d: [f32; 4], x: [f32; 4], bias: [f32; 2], scale_b: [f32; 2]{param}) {{\n\
            {prelude}\n\
            for i in 0..2 {{ for j in 0..2 {{ let mut s: {acc} = {zero};\n\
            for p in 0..{bound} {{ s = s + {product}; }}\n\
            {store} }} }}\n{epilogue}\n}}\n",
            self.name(), self.output_type()
        )
    }

    fn setup(self) -> String {
        let (at, bt) = self.input_types();
        let (two, three, five, seven) = if at == "u8" {
            ("2", "3", "5", "7")
        } else {
            ("2.0", "3.0", "5.0", "7.0")
        };
        let initial = if matches!(self, Self::Residual) {
            "[-4.0,-3.0,4.0,5.0]"
        } else if self.output_type() == "i32" {
            "[41,43,47,53]"
        } else {
            "[41.0,43.0,47.0,53.0]"
        };
        let bias = if matches!(self, Self::Residual) {
            "[1.0,2.0]"
        } else {
            "[-5.0,7.0]"
        };
        format!(
            "let a: [{at}; 2] = [{two} as {at},{three} as {at}]; let b: [{bt}; 2] = [{five} as {bt},{seven} as {bt}];\n\
            let mut out: [{}; 4] = {initial}; let mut d: [f32; 4] = [-2.0,3.0,-4.0,5.0];\n\
            let x: [f32; 4] = [2.0,3.0,4.0,5.0]; let bias: [f32; 2] = {bias};\n\
            let scale_b: [f32; 2] = [0.5,1.0];\n",
            self.output_type()
        )
    }

    fn expected(self, positive: bool) -> String {
        // Independent outer-product oracle: A=[2,3], B=[5,7], K=1 gives
        // [2*5,2*7,3*5,3*7]=[10,14,15,21], also for the TN layout.
        // K<=0 performs no additions and leaves the exact +0 seed.
        let dots: [i32; 4] = if positive { [10, 14, 15, 21] } else { [0; 4] };
        let mut expected = String::new();
        for (lane, dot) in dots.into_iter().enumerate() {
            let col = lane % 2;
            let value = match self {
                Self::Residual => ([-4, -3, 4, 5][lane] + dot + [1, 2][col]).max(0),
                Self::CrossResidual => [2, 3, 4, 5][lane] + dot,
                Self::F32Epilogue | Self::Bf16Epilogue | Self::F16Epilogue => {
                    (dot + [-5, 7][col]).max(0)
                }
                // scale_a=2 times scale_b=[1/2,1] gives exact factors [1,2].
                Self::I8Dequant => (dot * [1, 2][col] + [-5, 7][col]).max(0),
                _ => dot,
            };
            expected.push_str(&format!("{value}\n"));
        }
        if self.dual() {
            for (lane, dot) in dots.into_iter().enumerate() {
                let d = [-2, 3, -4, 5][lane];
                if matches!(self, Self::DualMultiply) && !positive && d < 0 {
                    // A negative finite gate times the empty +0 seed is -0.
                    expected.push_str("-0\n");
                } else {
                    let value = if matches!(self, Self::DualMultiply) {
                        d * dot
                    } else {
                        d + dot
                    };
                    expected.push_str(&format!("{value}\n"));
                }
            }
        }
        expected
    }
}

const PLAIN: &[Case] = &[
    Case::Bf16Nt,
    Case::Bf16Tn,
    Case::F16Nt,
    Case::F16Tn,
    Case::I8Whole,
    Case::I8Statement,
];
const FINALIZED: &[Case] = &[
    Case::Residual,
    Case::CrossResidual,
    Case::DualMultiply,
    Case::DualAdd,
    Case::F32Epilogue,
    Case::Bf16Epilogue,
    Case::F16Epilogue,
    Case::I8Dequant,
];

fn source(cases: &[Case], parallel: bool) -> (String, String) {
    let mut src = String::from("module empty.contractions\n");
    let mut expected = String::new();
    for &case in cases {
        for (suffix, bound, param) in [
            ("dynamic", "k", ", k: i32"),
            ("zero", "0", ""),
            ("negative", "-1", ""),
            ("positive", "1", ""),
        ] {
            src.push_str(&case.function(parallel, suffix, bound, param));
        }
    }
    src.push_str("fn main() -> i32 {\n");
    for &case in cases {
        for (suffix, argument, positive) in [
            ("dynamic", ",0", false),
            ("dynamic", ",-1", false),
            ("dynamic", ",1", true),
            ("zero", "", false),
            ("negative", "", false),
            ("positive", "", true),
        ] {
            src.push_str(&format!("{{ {} {}_{suffix}(a,b,out,d,x,bias,scale_b{argument});\nfor lane in 0..4 {{ print(out[lane]); }}\n", case.setup(), case.name()));
            if case.dual() {
                src.push_str("for lane in 0..4 { print(d[lane]); }\n");
            }
            src.push_str("}\n");
            expected.push_str(&case.expected(positive));
        }
    }
    src.push_str("return 0; }\n");
    (src, expected)
}

fn invoke(path: &Path, args: &[&str], no_recognize: bool) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    command.args(args).arg(path).env("RAYON_NUM_THREADS", "2");
    if no_recognize {
        command.env(wukong_driver::NO_RECOGNIZE_ENV, "1");
    } else {
        command.env_remove(wukong_driver::NO_RECOGNIZE_ENV);
    }
    let output = command.output().expect("spawn wukongc");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?} recognition off={no_recognize}\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn check(cases: &[Case], parallel: bool, group: &str) {
    let path = std::env::temp_dir().join(format!(
        "wukong-empty-contractions-{}-{parallel}-{group}.wk",
        std::process::id()
    ));
    let (src, expected) = source(cases, parallel);
    std::fs::write(&path, src).unwrap();
    let lanes_per_call: usize = cases
        .iter()
        .map(|case| if case.dual() { 8 } else { 4 })
        .sum();
    assert_eq!(expected.lines().count(), lanes_per_call * 6);
    // Complete the dispatch checks before executing any native program. Each
    // function must retain its kernel for both dynamic and literal positive K;
    // another function's dispatch cannot satisfy this obligation.
    let mir = invoke(&path, &["--emit=mir", "-O0"], false);
    for &case in cases {
        for suffix in ["dynamic", "positive"] {
            let name = format!("{}_{suffix}", case.name());
            let (_, tail) = mir
                .split_once(&format!("fn {name}("))
                .unwrap_or_else(|| panic!("missing {name}:\n{mir}"));
            let body = tail.split("\n}\n").next().unwrap();
            let parallel_suffix = if parallel { "_parallel" } else { "" };
            let symbol = format!("{}{parallel_suffix}", case.kernel());
            assert_eq!(
                body.matches(&format!("call {symbol}(")).count(),
                1,
                "{name}:\n{body}"
            );
            if case.dual() || matches!(case, Case::CrossResidual) {
                assert_eq!(
                    body.matches(&format!("call wukong_velem_f32{parallel_suffix}("))
                        .count(),
                    1,
                    "missing finalizer in {name}:\n{body}"
                );
            }
            if suffix == "dynamic" {
                assert!(
                    body.contains("cmp.sgt") && body.contains("cond_br"),
                    "missing positive-extent guard in {name}:\n{body}"
                );
            }
        }
    }
    for opt in ["-O0", "-O2"] {
        let off = invoke(&path, &["--emit=mir", opt], true);
        assert!(
            !off.contains("call wukong_sgemm") && !off.contains("call wukong_i8gemm"),
            "recognition off:\n{off}"
        );
    }
    for opt in ["-O0", "-O2"] {
        for backend in ["--backend=interp", "--backend=native"] {
            for no_recognize in [false, true] {
                assert_eq!(
                    invoke(&path, &["--run", backend, opt], no_recognize),
                    expected,
                    "{group} parallel={parallel} {backend} {opt} recognition off={no_recognize}"
                );
            }
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn plain_serial_empty_contractions() {
    check(PLAIN, false, "plain");
}
#[test]
fn plain_parallel_empty_contractions() {
    check(PLAIN, true, "plain");
}
#[test]
fn finalized_serial_empty_contractions() {
    check(FINALIZED, false, "finalized");
}
#[test]
fn finalized_parallel_empty_contractions() {
    check(FINALIZED, true, "finalized");
}
