use wukong_mir::{MirType, Op, Program};
use wukong_span::{Interner, SourceId};

fn lower(source: &str) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (ast, diagnostics) = wukong_parser::parse_module(source, SourceId(0), &mut interner);
    assert!(diagnostics.is_empty(), "parse: {diagnostics:?}");
    let (sema, diagnostics) = wukong_sema::check(&ast, &interner);
    assert!(diagnostics.is_empty(), "sema: {diagnostics:?}");
    let (program, diagnostics) = wukong_mir_build::lower_program_with(
        &ast,
        &sema,
        &mut interner,
        wukong_mir_build::LowerOptions {
            recognize: false,
            vec_kernels: false,
            ..Default::default()
        },
    );
    assert!(diagnostics.is_empty(), "lower: {diagnostics:?}");
    (program, interner)
}

fn bytes(size: u32) -> MirType {
    MirType::Array(Box::new(MirType::I8), size)
}

/// Check the actual annotated local before any native execution: a too-small native stack slot
/// can appear to work by overwriting adjacent stack storage. Every fixture puts this local first.
fn assert_first_aggregate(program: &Program, interner: &Interner, name: &str, expected: MirType) {
    let function = program
        .funcs
        .iter()
        .find(|function| {
            let actual = interner.resolve(function.name);
            actual == name || actual.starts_with(&format!("{name}$"))
        })
        .expect("fixture function");
    let actual = function
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .find_map(|inst| match &inst.op {
            Op::Alloca { ty, .. } if matches!(ty, MirType::Array(..)) => Some(ty),
            _ => None,
        })
        .expect("annotated aggregate slot");
    assert_eq!(actual, &expected, "{name} allocation");
}

fn check_values(program: &Program, mut interner: Interner, stdout: &str) {
    let main = interner.intern("main");
    for level in [0, 2] {
        let mut optimized = program.clone();
        wukong_opt::optimize(&mut optimized, level);
        let errors = wukong_mir::verify::verify_program(&optimized);
        assert!(errors.is_empty(), "MIR at O{level}: {errors:?}");
        let expected = (0, stdout.as_bytes().to_vec());
        assert_eq!(
            wukong_interp::run_with_output(&optimized, main, &interner).unwrap(),
            expected,
            "interpreter at O{level}"
        );
        assert_eq!(
            wukong_codegen_cranelift::jit_run(&optimized, main, &interner).unwrap(),
            expected,
            "native at O{level}"
        );
    }
}

#[test]
fn nested_tuple_keeps_recursive_alignment() {
    let (program, interner) = lower(
        "fn main() -> i32 {
            let mut t: (u8, (u8, (u8, i64)));
            let sentinel: i64 = 97;
            t.0 = 1; t.1.0 = 2; t.1.1.0 = 3; t.1.1.1 = 42;
            println(t.0); println(t.1.0); println(t.1.1.0); println(t.1.1.1);
            println(sentinel); return 0;
        }",
    );
    // Field offsets are 0, 8, 16, 24; old annotation lowering allocated only 18 bytes.
    assert_first_aggregate(&program, &interner, "main", bytes(32));
    check_values(&program, interner, "1\n2\n3\n42\n97\n");
}

#[test]
fn named_struct_and_constant_array_fields_keep_alignment() {
    for (field, writes, reads, expected_size, stdout) in [
        (
            "S",
            "t.1.a = 2; t.1.b = 43;",
            "println(t.1.a); println(t.1.b);",
            24,
            "1\n2\n43\n",
        ),
        (
            "[S; 1 + 1]",
            "t.1[0].a = 2; t.1[0].b = 43; t.1[1].a = 3; t.1[1].b = 44;",
            "println(t.1[0].a); println(t.1[0].b); println(t.1[1].a); println(t.1[1].b);",
            40,
            "1\n2\n43\n3\n44\n",
        ),
    ] {
        let (program, interner) = lower(&format!(
            "struct S {{ a: u8, b: i64 }}
             fn main() -> i32 {{ let mut t: (u8, {field});
                t.0 = 1; {writes} println(t.0); {reads} return 0; }}"
        ));
        assert_first_aggregate(&program, &interner, "main", bytes(expected_size));
        check_values(&program, interner, stdout);
    }
}

#[test]
fn slice_field_keeps_fat_pointer_alignment() {
    let (program, interner) = lower(
        "fn main() -> i32 {
            let mut t: (u8, []i64);
            t.0 = 7; println(t.0); println(t.1.len()); return 0;
        }",
    );
    assert_first_aggregate(&program, &interner, "main", bytes(24));
    check_values(&program, interner, "7\n0\n");
}

#[test]
fn zero_length_array_keeps_element_alignment() {
    let (program, interner) = lower(
        "struct S { a: u8, b: i64 }
         fn main() -> i32 {
            let mut t: (u8, [S; 0], u8);
            t.0 = 5; t.2 = 9; println(t.0); println(t.2); return 0;
        }",
    );
    assert_first_aggregate(&program, &interner, "main", bytes(16));
    check_values(&program, interner, "5\n9\n");
}

#[test]
fn arrays_of_nested_tuple_annotations_use_padded_stride() {
    let (program, interner) = lower(
        "fn main() -> i32 {
            let mut t: [(u8, (u8, i64)); 2];
            t[0].0 = 1; t[0].1.0 = 2; t[0].1.1 = 43;
            t[1].0 = 3; t[1].1.0 = 4; t[1].1.1 = 44;
            println(t[0].0); println(t[0].1.0); println(t[0].1.1);
            println(t[1].0); println(t[1].1.0); println(t[1].1.1); return 0;
        }",
    );
    assert_first_aggregate(
        &program,
        &interner,
        "main",
        MirType::Array(Box::new(bytes(24)), 2),
    );
    check_values(&program, interner, "1\n2\n43\n3\n4\n44\n");
}

#[test]
fn generic_tuple_annotation_resolves_nested_concrete_fields() {
    let (program, interner) = lower(
        "fn wrap<T>(value: T) -> T {
            let mut t: (u8, (u8, T));
            t.0 = 1; t.1.0 = 2; t.1.1 = value;
            println(t.0); println(t.1.0); return t.1.1;
        }
        fn main() -> i32 {
            let value: (u8, i64) = (3, 42);
            let result = wrap(value);
            println(result.0); println(result.1); return 0;
        }",
    );
    assert_first_aggregate(&program, &interner, "wrap", bytes(32));
    check_values(&program, interner, "1\n2\n3\n42\n");
}

#[test]
fn enum_payload_and_unit_fields_keep_semantic_layout() {
    let (program, interner) = lower(
        "enum E { Nil, Data(u8, i64) }
         fn main() -> i32 {
            let mut t: (u8, E);
            t.0 = 7; t.1 = E::Data(3, 42);
            println(t.0);
            match t.1 { E::Data(a, b) => { println(a); println(b); }, E::Nil => println(0) };
            return 0;
        }",
    );
    // The payload is 16 bytes aligned to 8; its tag makes E 24 bytes, and the tuple 32.
    assert_first_aggregate(&program, &interner, "main", bytes(32));
    check_values(&program, interner, "7\n3\n42\n");

    let (program, interner) = lower(
        "fn main() -> i32 {
            let mut t: (u8, (), u8);
            t.0 = 5; t.2 = 9; println(t.0); println(t.2); return 0;
        }",
    );
    assert_first_aggregate(&program, &interner, "main", bytes(2));
    check_values(&program, interner, "5\n9\n");
}

#[test]
fn vector_field_uses_semantic_vector_alignment() {
    let (program, interner) = lower("fn main() -> i32 { let t: (u8, f32x3); return 0; }");
    // f32x3 has 12 bytes of data, aligned to 16, so this tuple occupies 32 bytes.
    assert_first_aggregate(&program, &interner, "main", bytes(32));
    check_values(&program, interner, "");
}
