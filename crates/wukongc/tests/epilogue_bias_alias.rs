//! GEMM bias reads must observe earlier source stores when bias names the output.
use std::path::Path;
use std::process::Command;

#[derive(Clone, Copy, Debug)]
enum Case {
    DirectWhole,
    DirectStatement,
    Batched,
    Residual,
    F32Window,
    Bf16Window,
    F16Window,
    I8Window,
}

const CASES: [Case; 8] = [
    Case::DirectWhole,
    Case::DirectStatement,
    Case::Batched,
    Case::Residual,
    Case::F32Window,
    Case::Bf16Window,
    Case::F16Window,
    Case::I8Window,
];

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::DirectWhole => "direct_whole",
            Self::DirectStatement => "direct_statement",
            Self::Batched => "batched",
            Self::Residual => "residual",
            Self::F32Window => "f32_window",
            Self::Bf16Window => "bf16_window",
            Self::F16Window => "f16_window",
            Self::I8Window => "i8_window",
        }
    }

    fn kernel(self) -> &'static str {
        match self {
            Self::Bf16Window => "wukong_sgemm_bf16_nt_epi",
            Self::F16Window => "wukong_sgemm_f16_nt_epi",
            Self::I8Window => "wukong_i8gemm_nt_deq",
            _ => "wukong_sgemm_nt_epi",
        }
    }

    fn types(self) -> (&'static str, &'static str) {
        match self {
            Self::Bf16Window => ("bf16", "bf16"),
            Self::F16Window => ("f16", "f16"),
            Self::I8Window => ("u8", "i8"),
            _ => ("f32", "f32"),
        }
    }

    fn function(self, parallel: bool, alias: bool) -> String {
        let attr = if parallel { "@parallel\n" } else { "" };
        let (at, bt) = self.types();
        let bias = if alias { "out" } else { "bias" };
        let acc = if matches!(self, Self::I8Window) {
            "i32"
        } else {
            "f32"
        };
        let zero = if acc == "i32" { "0" } else { "0.0" };
        let product = if at == "f32" {
            "a[i*1+p]*b[j*1+p]".to_owned()
        } else {
            format!("(a[i*1+p] as {acc})*(b[j*1+p] as {acc})")
        };
        let mut prelude = String::new();
        let (store, epilogue) = match self {
            Self::DirectWhole | Self::DirectStatement => (
                format!("out[i*2+j] = s+{bias}[j];"), String::new()),
            Self::Residual => (
                format!("out[i*2+j] = out[i*2+j]+s+{bias}[j];"), String::new()),
            Self::F32Window | Self::Bf16Window | Self::F16Window => (
                "out[i*2+j] = s;".to_owned(),
                format!("for i in 0..2 {{ for j in 0..2 {{ out[i*2+j] = fmax(out[i*2+j]+{bias}[j], 0.0); }} }}")),
            Self::I8Window => {
                prelude.push_str("let mut c: [i32;4] = [99;4]; let scale_a: f32 = 1.0;\n");
                ("c[i*2+j] = s;".to_owned(),
                 format!("for i in 0..2 {{ for j in 0..2 {{ out[i*2+j] = (c[i*2+j] as f32)*scale_a*scale[j]+{bias}[j]; }} }}"))
            }
            Self::Batched => (String::new(), String::new()),
        };
        if matches!(self, Self::DirectStatement) {
            prelude.push_str("let marker: i32 = 0;\n");
        }
        let body = if matches!(self, Self::Batched) {
            let offset = if alias { "batch*4" } else { "batch*2" };
            format!(
                "for batch in 0..2 {{ for i in 0..2 {{ for j in 0..2 {{\n\
                let mut s: f32 = 0.0; for p in 0..1 {{ s = s+a[batch*2+i*1+p]*b[j*1+p]; }}\n\
                out[batch*4+i*2+j] = s+{bias}[{offset}+j]; }} }} }}"
            )
        } else {
            format!(
                "{prelude}for i in 0..2 {{ for j in 0..2 {{\n\
                let mut s: {acc} = {zero}; for p in 0..1 {{ s = s+{product}; }}\n\
                {store} }} }}\n{epilogue}"
            )
        };
        let alen = if matches!(self, Self::Batched) { 4 } else { 2 };
        let olen = if matches!(self, Self::Batched) { 8 } else { 4 };
        format!("{attr}fn {}(a: [{at};{alen}], b: [{bt};2], mut out: [f32;{olen}], bias: [f32;4], scale: [f32;2]) {{ {body} }}\n", self.name())
    }

    fn setup(self) -> String {
        let (at, bt) = self.types();
        let (alen, av, olen, ov) = if matches!(self, Self::Batched) {
            (
                4,
                format!("[2 as {at},3 as {at},4 as {at},5 as {at}]"),
                8,
                "[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]",
            )
        } else {
            (2, format!("[2 as {at},3 as {at}]"), 4, "[1.0,2.0,3.0,4.0]")
        };
        format!(
            "let a: [{at};{alen}] = {av}; let b: [{bt};2] = [5 as {bt},7 as {bt}];\n\
            let mut out: [f32;{olen}] = {ov}; let bias: [f32;4] = [1.0,2.0,5.0,6.0];\n\
            let scale: [f32;2] = [1.0;2]; {}(a,b,out,bias,scale);\n\
            for k in 0..{olen} {{ print(out[k]); }}\n",
            self.name()
        )
    }

    fn expected(self, alias: bool) -> &'static [i32] {
        // K=1 gives the exact outer product [10,14,15,21]. For a window,
        // the first loop stores ALL dots before the second starts. Same-output
        // bias then sees row zero's epilogue stores when processing row one.
        // Direct and residual forms instead read the initial first row, then
        // its updated values. These integer-sized f32 calculations are exact.
        match (self, alias) {
            (Self::Batched, true) => &[11, 16, 26, 37, 25, 34, 50, 69],
            (Self::Batched, false) => &[11, 16, 16, 23, 25, 34, 30, 41],
            (Self::Residual, true) => &[12, 18, 30, 43],
            (Self::Residual, false) => &[12, 18, 19, 27],
            (Self::F32Window | Self::Bf16Window | Self::F16Window, true) => &[20, 28, 35, 49],
            (_, true) => &[11, 16, 26, 37],
            (_, false) => &[11, 16, 16, 23],
        }
    }
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

fn check(parallel: bool) {
    for alias in [true, false] {
        let path = std::env::temp_dir().join(format!(
            "wukong-epilogue-bias-{}-{parallel}-{alias}.wk",
            std::process::id()
        ));
        let mut src = String::from("module epilogue.bias.alias\n");
        for case in CASES {
            src.push_str(&case.function(parallel, alias));
        }
        src.push_str("fn main() -> i32 {\n");
        let mut expected = String::new();
        for case in CASES {
            src.push_str(&format!("{{ {} }}\n", case.setup()));
            for value in case.expected(alias) {
                expected.push_str(&format!("{value}\n"));
            }
        }
        src.push_str("return 0; }\n");
        std::fs::write(&path, src).unwrap();
        assert_eq!(expected.lines().count(), 36);
        // Inspect ALL source functions before executing any native alias case.
        // An unrelated function cannot satisfy a positive dispatch obligation.
        for opt in ["-O0", "-O2"] {
            for off in [false, true] {
                let mir = invoke(&path, &["--emit=mir", opt], off);
                if alias || off {
                    assert!(
                        !mir.contains("call wukong_sgemm") && !mir.contains("call wukong_i8gemm"),
                        "unsafe/disabled GEMM:\n{mir}"
                    );
                }
                if alias {
                    assert!(
                        !mir.contains("call wukong_parallel_for"),
                        "bias-dependent rows must stay ordered:\n{mir}"
                    );
                }
                if opt == "-O0" && !alias && !off {
                    for case in CASES {
                        let (_, tail) = mir.split_once(&format!("fn {}(", case.name())).unwrap();
                        let body = tail.split("\n}\n").next().unwrap();
                        // A batched body can be outlined. Follow only the worker
                        // named by this function, so unrelated dispatch cannot pass.
                        let symbol = case.kernel();
                        if matches!(case, Case::Batched) && parallel {
                            if body.contains("call wukong_parallel_for") {
                                let worker = body
                                    .lines()
                                    .find_map(|line| {
                                        line.split_once("func_addr ").map(|(_, name)| name.trim())
                                    })
                                    .expect("outlined batch worker");
                                let (_, tail) = mir
                                    .split_once(&format!("fn {worker}("))
                                    .expect("worker body");
                                let worker_body = tail.split("\n}\n").next().unwrap();
                                assert_eq!(
                                    worker_body.matches("call wukong_sgemm_nt_epi(").count(),
                                    1,
                                    "{worker}:\n{worker_body}"
                                );
                            } else {
                                assert_eq!(
                                    body.matches("call wukong_sgemm_nt_epi_parallel(").count(),
                                    1,
                                    "{}:\n{body}",
                                    case.name()
                                );
                            }
                        } else {
                            let suffix = if parallel { "_parallel" } else { "" };
                            assert_eq!(
                                body.matches(&format!("call {symbol}{suffix}(")).count(),
                                1,
                                "{}:\n{body}",
                                case.name()
                            );
                        }
                    }
                }
            }
        }
        for opt in ["-O0", "-O2"] {
            for backend in ["--backend=interp", "--backend=native"] {
                for off in [false, true] {
                    assert_eq!(
                        invoke(&path, &["--run", backend, opt], off),
                        expected,
                        "parallel={parallel} alias={alias} {backend} {opt} off={off}"
                    );
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn serial_epilogue_bias_alias_preserves_source_order() {
    check(false);
}

#[test]
fn parallel_epilogue_bias_alias_preserves_source_order() {
    check(true);
}
