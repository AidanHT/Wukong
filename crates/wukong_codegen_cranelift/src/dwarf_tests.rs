//! `-g` DWARF, read back: the object is parsed with the `object` crate, its debug sections are
//! relocated the way a linker would, and `gimli::read` decodes the compile unit and the line
//! program. Expected lines come from the fixture text, never from compiler output. Both containers
//! are covered on any host — COFF (Windows) and ELF (Linux) — by building the object on an ISA of
//! each triple. Also here: the JIT perf map (`WUKONG_PERF_MAP`).

use super::*;
use std::collections::BTreeMap;

use cranelift_object::object::{
    self, Object as _, ObjectSection as _, ObjectSymbol as _, RelocationTarget,
};
use gimli::{EndianSlice, RunTimeEndian};
use wukong_span::{SourceId, SourceMap};

/// A helper called from `main` (so `-O0` keeps it a function of its own), a loop, and a branch.
/// Lines are found by their text ([`line_of`]), so the fixture can change without renumbering.
const FIXTURE: &str = "fn scale(x: i32, k: i32) -> i32 {
    let y = x * k;
    return y + 1;
}

fn main() -> i32 {
    let mut acc: i32 = 0;
    for i in 0..10 {
        acc = acc + scale(i, 3);
    }
    if acc > 100 {
        print(acc);
    }
    return 0;
}
";

/// Calls through a slice (a checked access), a struct array, and a float recurrence.
const SLICES: &str = "struct P { x: i32, y: f32 }

fn pick(s: []i32, i: i32) -> i32 {
    return s[i] * 3;
}

fn main() -> i32 {
    let arr: [i32; 6] = [4, 8, 15, 16, 23, 42];
    let s: []i32 = arr;
    let mut t: i32 = 0;
    let mut v: f32 = 0.5;
    let ps: [P; 4] = [P { x: 3, y: 0.25 }; 4];
    for i in 0..6 {
        t = t + pick(s, i);
        v = v * 1.5 + ps[i % 4].y;
    }
    print(v);
    return t;
}
";

const FILE: &str = "src/dwarf_fixture.wk";
const COMP_DIR: &str = "/work/wukong";

/// The 1-based line of the first line of `src` containing `needle`.
fn line_of(src: &str, needle: &str) -> u64 {
    src.lines()
        .position(|l| l.contains(needle))
        .map(|i| i as u64 + 1)
        .unwrap_or_else(|| panic!("`{needle}` is not in the fixture"))
}

/// Parse, check and lower `src` (as [`FILE`]), optimizing at `opt` when it is non-zero.
fn lower(src: &str, opt: u8) -> (Program, Interner, SourceMap) {
    let mut sm = SourceMap::new();
    let id = sm.add(FILE, src);
    assert_eq!(id, SourceId(0));
    let mut interner = Interner::new();
    let (module, parsed) = wukong_parser::parse_module(src, id, &mut interner);
    assert!(parsed.iter().all(|d| !d.is_error()), "{parsed:?}");
    let (sema, checked) = wukong_sema::check(&module, &interner);
    assert!(checked.iter().all(|d| !d.is_error()), "{checked:?}");
    let (mut program, lowered) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(lowered.iter().all(|d| !d.is_error()), "{lowered:?}");
    if opt > 0 {
        wukong_opt::optimize(&mut program, opt);
    }
    (program, interner, sm)
}

/// An x86-64 ISA for `triple` (the object container follows it) at backend level `level`.
fn isa_for(triple: &str, level: u8) -> std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa> {
    let mut flags = settings::builder();
    flags
        .set(
            "opt_level",
            BackendOpt::for_level(level).cranelift_opt_level(),
        )
        .unwrap();
    flags.set("is_pic", "true").unwrap();
    flags.set("enable_probestack", "true").unwrap();
    flags.set("probestack_strategy", "inline").unwrap();
    cranelift_codegen::isa::lookup_by_name(triple)
        .unwrap()
        .finish(settings::Flags::new(flags))
        .unwrap()
}

const COFF: &str = "x86_64-pc-windows-msvc";
const ELF: &str = "x86_64-unknown-linux-gnu";

fn emit(
    program: &Program,
    interner: &Interner,
    triple: &str,
    level: u8,
    parallel: bool,
    kind: ObjectKind,
    debug: Option<&DebugInfo<'_>>,
) -> Vec<u8> {
    emit_object_for_isa(
        isa_for(triple, level),
        program,
        interner,
        parallel,
        kind,
        debug,
        std::time::Instant::now(),
    )
    .unwrap_or_else(|e| panic!("emit: {e}"))
    .0
}

/// A section's bytes with every relocation applied against section base 0 — what a linker would
/// write, with the object's own symbol addresses (offsets into `.text`).
fn relocated(file: &object::File<'_>, name: &str) -> Vec<u8> {
    let Some(section) = file.section_by_name(name) else {
        return Vec::new();
    };
    let mut data = section.data().unwrap().to_vec();
    for (offset, reloc) in section.relocations() {
        let base = match reloc.target() {
            RelocationTarget::Symbol(i) => file.symbol_by_index(i).unwrap().address(),
            RelocationTarget::Section(i) => file.section_by_index(i).unwrap().address(),
            other => panic!("{name}: unexpected relocation target {other:?}"),
        };
        let at = offset as usize;
        let width = usize::from(reloc.size()) / 8;
        let mut field = [0u8; 8];
        field[..width].copy_from_slice(&data[at..at + width]);
        let implicit = if reloc.has_implicit_addend() {
            i64::from_le_bytes(field)
        } else {
            0
        };
        let value = (base as i64 + reloc.addend() + implicit) as u64;
        data[at..at + width].copy_from_slice(&value.to_le_bytes()[..width]);
    }
    data
}

/// One `DW_TAG_subprogram`.
#[derive(Debug, Clone)]
struct Subprogram {
    name: String,
    /// Whether it has a `DW_AT_linkage_name` (it must not: see [`check_structure`]).
    linkage: bool,
    low: u64,
    high: u64,
    decl_line: Option<u64>,
    artificial: bool,
}

/// One line-table row (an `end_sequence` row has `line` 0).
#[derive(Debug, Clone)]
struct LineRow {
    address: u64,
    file: String,
    line: u64,
    column: u64,
    end_sequence: bool,
    prologue_end: bool,
}

/// One FDE of `.debug_frame`, with the unwind rules at its first instruction.
#[derive(Debug, Clone)]
struct Frame {
    start: u64,
    len: u64,
    /// The CFA rule at `start`: `(register, offset)`.
    cfa_at_start: (u16, i64),
    /// Where the return address is saved at `start`, as an offset from the CFA.
    ra_at_start: Option<i64>,
}

#[derive(Debug, Default)]
struct Decoded {
    cu_name: String,
    comp_dir: String,
    producer: String,
    language: u16,
    cu_ranges: Vec<(u64, u64)>,
    subprograms: Vec<Subprogram>,
    rows: Vec<LineRow>,
    /// Every defined function symbol in the object: address -> (name, size or 0).
    symbols: BTreeMap<u64, (String, u64)>,
    frames: Vec<Frame>,
    format: Option<object::BinaryFormat>,
}

fn decode(bytes: &[u8]) -> Decoded {
    let file = object::File::parse(bytes).expect("the object parses");
    let mut out = Decoded {
        format: Some(file.format()),
        ..Decoded::default()
    };
    for sym in file.symbols() {
        if sym.kind() == object::SymbolKind::Text && sym.is_definition() {
            out.symbols
                .insert(sym.address(), (sym.name().unwrap().to_string(), sym.size()));
        }
    }
    let names = [
        ".debug_info",
        ".debug_abbrev",
        ".debug_line",
        ".debug_str",
        ".debug_ranges",
        ".debug_line_str",
        ".debug_rnglists",
        ".debug_addr",
        ".debug_str_offsets",
        ".debug_frame",
    ];
    let sections: BTreeMap<&str, Vec<u8>> =
        names.iter().map(|n| (*n, relocated(&file, n))).collect();
    let dwarf = gimli::read::Dwarf::load(|id| -> Result<_, gimli::Error> {
        let data = sections.get(id.name()).map(|v| &v[..]).unwrap_or(&[]);
        Ok(EndianSlice::new(data, RunTimeEndian::Little))
    })
    .unwrap();
    type R<'a> = EndianSlice<'a, RunTimeEndian>;
    let string = |unit: &gimli::read::Unit<R<'_>>, v: gimli::AttributeValue<R<'_>>| {
        dwarf
            .attr_string(unit, v)
            .unwrap()
            .to_string_lossy()
            .into_owned()
    };
    let mut units = dwarf.units();
    let mut count = 0;
    while let Some(header) = units.next().unwrap() {
        count += 1;
        let unit = dwarf.unit(header).unwrap();
        let mut ranges = dwarf.unit_ranges(&unit).unwrap();
        while let Some(r) = ranges.next().unwrap() {
            out.cu_ranges.push((r.begin, r.end));
        }
        let mut entries = unit.entries();
        while let Some((_, entry)) = entries.next_dfs().unwrap() {
            let attr = |at| entry.attr_value(at).unwrap();
            if entry.tag() == gimli::DW_TAG_compile_unit {
                out.cu_name = string(&unit, attr(gimli::DW_AT_name).unwrap());
                out.comp_dir = string(&unit, attr(gimli::DW_AT_comp_dir).unwrap());
                out.producer = string(&unit, attr(gimli::DW_AT_producer).unwrap());
                if let Some(gimli::AttributeValue::Language(l)) = attr(gimli::DW_AT_language) {
                    out.language = l.0;
                }
            } else if entry.tag() == gimli::DW_TAG_subprogram {
                let low = match attr(gimli::DW_AT_low_pc) {
                    Some(gimli::AttributeValue::Addr(a)) => a,
                    other => panic!("low_pc {other:?}"),
                };
                let high = match attr(gimli::DW_AT_high_pc) {
                    Some(gimli::AttributeValue::Udata(n)) => low + n,
                    other => panic!("high_pc {other:?}"),
                };
                out.subprograms.push(Subprogram {
                    name: string(&unit, attr(gimli::DW_AT_name).unwrap()),
                    linkage: attr(gimli::DW_AT_linkage_name).is_some(),
                    low,
                    high,
                    decl_line: attr(gimli::DW_AT_decl_line).and_then(|v| v.udata_value()),
                    artificial: attr(gimli::DW_AT_artificial).is_some(),
                });
            }
        }
        if let Some(program) = unit.line_program.clone() {
            let mut rows = program.rows();
            while let Some((header, row)) = rows.next_row().unwrap() {
                let file = row
                    .file(header)
                    .map(|f| string(&unit, f.path_name()))
                    .unwrap_or_default();
                out.rows.push(LineRow {
                    address: row.address(),
                    file,
                    line: row.line().map_or(0, |l| l.get()),
                    column: match row.column() {
                        gimli::ColumnType::Column(c) => c.get(),
                        gimli::ColumnType::LeftEdge => 0,
                    },
                    end_sequence: row.end_sequence(),
                    prologue_end: row.prologue_end(),
                });
            }
        }
    }
    assert_eq!(count, 1, "one compile unit per object");

    use gimli::UnwindSection as _;
    let mut debug_frame = gimli::DebugFrame::new(&sections[".debug_frame"], RunTimeEndian::Little);
    debug_frame.set_address_size(8);
    let bases = gimli::BaseAddresses::default();
    let mut ctx = gimli::UnwindContext::new();
    let mut entries = debug_frame.entries(&bases);
    while let Some(entry) = entries.next().unwrap() {
        let gimli::CieOrFde::Fde(partial) = entry else {
            continue;
        };
        let fde = partial.parse(gimli::DebugFrame::cie_from_offset).unwrap();
        let start = fde.initial_address();
        let row = fde
            .unwind_info_for_address(&debug_frame, &bases, &mut ctx, start)
            .unwrap();
        let cfa_at_start = match row.cfa() {
            gimli::CfaRule::RegisterAndOffset { register, offset } => (register.0, *offset),
            other => panic!("CFA rule {other:?}"),
        };
        let ra_at_start = match row.register(gimli::X86_64::RA) {
            gimli::RegisterRule::Offset(o) => Some(o),
            _ => None,
        };
        out.frames.push(Frame {
            start,
            len: fde.len(),
            cfa_at_start,
            ra_at_start,
        });
    }
    out
}

/// The subprogram named `name`.
fn sub<'a>(d: &'a Decoded, name: &str) -> &'a Subprogram {
    d.subprograms
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no subprogram `{name}` in {:?}", d.subprograms))
}

/// The non-terminal rows whose address falls in `s`.
fn rows_in<'a>(d: &'a Decoded, s: &Subprogram) -> Vec<&'a LineRow> {
    d.rows
        .iter()
        .filter(|r| !r.end_sequence && (s.low..s.high).contains(&r.address))
        .collect()
}

/// The structural properties every `-g` object has, whatever the program and level: every row
/// lies inside exactly the function its sequence started in, each function's subprogram matches
/// its symbol, the compile unit's ranges are the functions' code, and every row names the fixture.
fn check_structure(d: &Decoded, triple: &str) {
    assert!(!d.subprograms.is_empty(), "{triple}: no subprograms");
    for s in &d.subprograms {
        assert!(s.low < s.high, "{triple}: empty subprogram {s:?}");
        // A vector kernel is synthesized code with no declaration; a user function has both.
        assert_eq!(
            s.artificial,
            s.name.starts_with("__wukong_veckernel_"),
            "{triple}: {s:?}"
        );
        assert_eq!(s.decl_line.is_some(), !s.artificial, "{triple}: {s:?}");
        // Named by its source name only: a debugger's `break f` must find `f`, not `_WK1f`.
        assert!(!s.linkage, "{triple}: {s:?} has a linkage name");
        let (_, size) = d
            .symbols
            .get(&s.low)
            .unwrap_or_else(|| panic!("{triple}: no function symbol at {s:?}"));
        if *size != 0 {
            assert_eq!(
                s.high - s.low,
                *size,
                "{triple}: {s:?} high_pc is not its size"
            );
        }
        // Every Cranelift-compiled function has exactly one FDE, over exactly its code; a vector
        // kernel (raw bytes) has none.
        let fdes: Vec<&Frame> = d.frames.iter().filter(|f| f.start == s.low).collect();
        if s.artificial {
            assert!(fdes.is_empty(), "{triple}: {s:?} has an FDE");
        } else {
            assert_eq!(fdes.len(), 1, "{triple}: {s:?} has {} FDEs", fdes.len());
            assert_eq!(fdes[0].len, s.high - s.low, "{triple}: FDE of {s:?}");
        }
    }
    // Every FDE covers one function symbol's code, and at its first instruction the caller's frame
    // is where a call left it: the CFA is `rsp + 8` and the return address is at `CFA - 8`.
    for f in &d.frames {
        let (name, size) = d
            .symbols
            .get(&f.start)
            .unwrap_or_else(|| panic!("{triple}: FDE {f:?} starts off a function"));
        assert!(
            f.len > 0 && (*size == 0 || *size == f.len),
            "{triple}: {name}: {f:?}"
        );
        assert_eq!(
            f.cfa_at_start,
            (gimli::X86_64::RSP.0, 8),
            "{triple}: {name}: {f:?}"
        );
        assert_eq!(f.ra_at_start, Some(-8), "{triple}: {name}: {f:?}");
    }
    let mut frames: Vec<(u64, u64)> = d
        .frames
        .iter()
        .map(|f| (f.start, f.start + f.len))
        .collect();
    frames.sort();
    for w in frames.windows(2) {
        assert!(w[0].1 <= w[1].0, "{triple}: overlapping FDEs {frames:?}");
    }
    // No two functions overlap.
    let mut spans: Vec<(u64, u64)> = d.subprograms.iter().map(|s| (s.low, s.high)).collect();
    spans.sort();
    for w in spans.windows(2) {
        assert!(
            w[0].1 <= w[1].0,
            "{triple}: overlapping functions {spans:?}"
        );
    }
    let mut cu = d.cu_ranges.clone();
    cu.sort();
    assert_eq!(
        cu, spans,
        "{triple}: the compile unit's ranges are not the functions"
    );

    // Each sequence: starts at a function's low_pc, every row inside that function, ends at its
    // high_pc, addresses non-decreasing.
    let mut start: Option<&Subprogram> = None;
    let mut last = 0;
    for r in &d.rows {
        let f = match start {
            Some(f) => f,
            None => {
                let f = d
                    .subprograms
                    .iter()
                    .find(|s| s.low == r.address)
                    .unwrap_or_else(|| panic!("{triple}: sequence starts off a function: {r:?}"));
                start = Some(f);
                last = r.address;
                f
            }
        };
        assert!(r.address >= last, "{triple}: rows go backwards at {r:?}");
        last = r.address;
        if r.end_sequence {
            assert_eq!(
                r.address, f.high,
                "{triple}: sequence of {f:?} ends at {r:?}"
            );
            start = None;
        } else {
            assert!(
                (f.low..f.high).contains(&r.address),
                "{triple}: row {r:?} lies outside {f:?}"
            );
            assert!(r.line > 0, "{triple}: row without a line {r:?}");
            assert!(r.column > 0, "{triple}: row without a column {r:?}");
            assert!(r.file.ends_with("dwarf_fixture.wk"), "{triple}: {r:?}");
        }
    }
    assert!(start.is_none(), "{triple}: unterminated sequence");
}

/// (2) The line program maps each fixture function to the lines its own text spans, a chosen
/// statement of each to its exact line, every row inside its function; the compile unit names
/// the file, the directory, the producer and the language; on COFF and on ELF.
#[test]
fn line_table_maps_functions_to_their_source_lines() {
    let (program, interner, sm) = lower(FIXTURE, 0);
    let debug = DebugInfo {
        sources: &sm,
        comp_dir: COMP_DIR,
    };
    let scale_decl = line_of(FIXTURE, "fn scale(");
    let scale_end = line_of(FIXTURE, "return y + 1;") + 1;
    let main_decl = line_of(FIXTURE, "fn main(");
    let main_end = line_of(FIXTURE, "return 0;") + 1;
    for triple in [COFF, ELF] {
        for parallel in [false, true] {
            let bytes = emit(
                &program,
                &interner,
                triple,
                0,
                parallel,
                ObjectKind::Library,
                Some(&debug),
            );
            let d = decode(&bytes);
            let want_format = if triple == COFF {
                object::BinaryFormat::Coff
            } else {
                object::BinaryFormat::Elf
            };
            assert_eq!(d.format, Some(want_format));
            check_structure(&d, triple);
            assert_eq!(d.cu_name, FILE);
            assert_eq!(d.comp_dir, COMP_DIR);
            assert!(d.producer.starts_with("wukongc "), "{}", d.producer);
            assert_eq!(d.language, DW_LANG_WUKONG.0);

            let scale = sub(&d, "scale");
            let main = sub(&d, "main");
            assert_eq!(scale.decl_line, Some(scale_decl));
            assert_eq!(main.decl_line, Some(main_decl));
            assert_eq!(
                d.symbols[&scale.low].0, "scale",
                "a library object keeps source names"
            );

            let lines = |s: &Subprogram| -> Vec<u64> {
                rows_in(&d, s).iter().map(|r| r.line).collect::<Vec<_>>()
            };
            let (scale_lines, main_lines) = (lines(scale), lines(main));
            // Every row of a function is a line of that function's text ...
            assert!(
                scale_lines
                    .iter()
                    .all(|l| (scale_decl..=scale_end).contains(l)),
                "{triple}: scale rows {scale_lines:?}"
            );
            assert!(
                main_lines
                    .iter()
                    .all(|l| (main_decl..=main_end).contains(l)),
                "{triple}: main rows {main_lines:?}"
            );
            // ... the first row is its declaration, at the function's first byte ...
            assert_eq!(rows_in(&d, scale)[0].address, scale.low);
            assert_eq!(scale_lines[0], scale_decl);
            assert_eq!(main_lines[0], main_decl);
            // ... `prologue_end` marks exactly one row, the body's first statement, past the
            // frame setup and the parameter spills (located at the declaration) ...
            for (s, first) in [(scale, "let y = x * k;"), (main, "let mut acc: i32 = 0;")] {
                let marked: Vec<&LineRow> = rows_in(&d, s)
                    .into_iter()
                    .filter(|r| r.prologue_end)
                    .collect();
                assert_eq!(marked.len(), 1, "{triple}: {s:?} {marked:?}");
                assert_eq!(marked[0].line, line_of(FIXTURE, first), "{triple}: {s:?}");
                assert!(marked[0].address > s.low, "{triple}: {s:?}");
            }
            // ... and each statement is there, at its line.
            for needle in ["let y = x * k;", "return y + 1;"] {
                let l = line_of(FIXTURE, needle);
                assert!(
                    scale_lines.contains(&l),
                    "{triple}: `{needle}` (line {l}) not in {scale_lines:?}"
                );
            }
            for needle in [
                "let mut acc: i32 = 0;",
                "acc = acc + scale(i, 3);",
                "if acc > 100 {",
                "print(acc);",
                "return 0;",
            ] {
                let l = line_of(FIXTURE, needle);
                assert!(
                    main_lines.contains(&l),
                    "{triple}: `{needle}` (line {l}) not in {main_lines:?}"
                );
            }
            // The column of `x * k`'s product is the column of `x` (a binary expression's span
            // starts at its left operand).
            let let_y = line_of(FIXTURE, "let y = x * k;");
            let x_col = FIXTURE
                .lines()
                .nth(let_y as usize - 1)
                .unwrap()
                .find("x * k")
                .unwrap() as u64
                + 1;
            assert!(
                rows_in(&d, scale)
                    .iter()
                    .any(|r| r.line == let_y && r.column == x_col),
                "{triple}: no row at {let_y}:{x_col} in {:?}",
                rows_in(&d, scale)
            );
        }
    }
}

/// At `-O2` the helper is inlined, the vectorizer and friends create values without locations,
/// and the line table still has every row inside its function, at a line of the fixture.
#[test]
fn optimized_line_table_stays_inside_functions() {
    let (program, interner, sm) = lower(FIXTURE, 2);
    let debug = DebugInfo {
        sources: &sm,
        comp_dir: COMP_DIR,
    };
    let last_line = FIXTURE.lines().count() as u64;
    for triple in [COFF, ELF] {
        let d = decode(&emit(
            &program,
            &interner,
            triple,
            2,
            true,
            ObjectKind::Library,
            Some(&debug),
        ));
        check_structure(&d, triple);
        let main = sub(&d, "main");
        let rows = rows_in(&d, main);
        assert!(!rows.is_empty(), "{triple}: main has no rows at -O2");
        assert!(rows.iter().all(|r| (1..=last_line).contains(&r.line)));
    }
}

/// An executable object mangles and internalizes user functions, and adds the entry thunk and the
/// C `main`: each subprogram keeps its source name (at its mangled symbol's address), `main` names
/// the user's `main` only, and the two entry functions have call frames but no subprogram.
#[test]
fn executable_objects_keep_source_names_and_frame_the_entry() {
    let (program, interner, sm) = lower(FIXTURE, 0);
    let debug = DebugInfo {
        sources: &sm,
        comp_dir: COMP_DIR,
    };
    for triple in [COFF, ELF] {
        let d = decode(&emit(
            &program,
            &interner,
            triple,
            0,
            false,
            ObjectKind::Executable,
            Some(&debug),
        ));
        check_structure(&d, triple);
        for name in ["scale", "main"] {
            let s = sub(&d, name);
            assert_eq!(d.symbols[&s.low].0, entry::mangle(name), "{triple}: {s:?}");
        }
        assert_eq!(
            d.subprograms.iter().filter(|s| s.name == "main").count(),
            1,
            "{triple}: {:?}",
            d.subprograms
        );
        for symbol in ["wukong_entry_thunk", "main"] {
            let (&address, _) = d
                .symbols
                .iter()
                .find(|(_, (name, _))| name == symbol)
                .unwrap_or_else(|| panic!("{triple}: no `{symbol}` symbol"));
            assert!(
                d.frames.iter().any(|f| f.start == address),
                "{triple}: `{symbol}` has no FDE"
            );
            assert!(
                d.subprograms.iter().all(|s| s.low != address),
                "{triple}: `{symbol}` has a subprogram"
            );
        }
    }
}

/// `-g` adds sections and changes no machine code: the `.text` of a `-g` object is the plain
/// object's; and with no debug info requested the object is byte-identical to `emit_object_full`'s.
#[test]
fn debug_info_changes_no_code_and_is_deterministic() {
    for (src, level) in [(FIXTURE, 0), (FIXTURE, 2), (SLICES, 2)] {
        let (program, interner, sm) = lower(src, level);
        let debug = DebugInfo {
            sources: &sm,
            comp_dir: COMP_DIR,
        };
        for triple in [COFF, ELF] {
            let plain = emit(
                &program,
                &interner,
                triple,
                level,
                true,
                ObjectKind::Library,
                None,
            );
            let serial = emit(
                &program,
                &interner,
                triple,
                level,
                false,
                ObjectKind::Library,
                None,
            );
            assert_eq!(plain, serial);
            let g = emit(
                &program,
                &interner,
                triple,
                level,
                true,
                ObjectKind::Library,
                Some(&debug),
            );
            let g_serial = emit(
                &program,
                &interner,
                triple,
                level,
                false,
                ObjectKind::Library,
                Some(&debug),
            );
            assert_eq!(
                g, g_serial,
                "{triple}: -g parallel and serial objects differ"
            );
            assert_eq!(
                g,
                emit(
                    &program,
                    &interner,
                    triple,
                    level,
                    true,
                    ObjectKind::Library,
                    Some(&debug)
                ),
                "{triple}: -g is not deterministic"
            );
            let text = |bytes: &[u8]| {
                let file = object::File::parse(bytes).unwrap();
                let s = file.section_by_name(".text").unwrap();
                s.data().unwrap().to_vec()
            };
            assert_eq!(
                text(&plain),
                text(&g),
                "{triple}: -g changed the machine code"
            );
            let plain_file = object::File::parse(&plain[..]).unwrap();
            assert!(plain_file.section_by_name(".debug_line").is_none());
            assert!(plain_file.section_by_name(".debug_info").is_none());
        }
        // The host path: no debug info is byte-identical to the shipping emitter.
        let opts = EmitOptions {
            verify: true,
            parallel: true,
        };
        let full = emit_object_full(
            &program,
            &interner,
            opts,
            TargetCpu::Native,
            BackendOpt::for_level(level),
            ObjectKind::Library,
        )
        .unwrap()
        .0;
        let none = emit_object_debug_full(
            &program,
            &interner,
            opts,
            TargetCpu::Native,
            BackendOpt::for_level(level),
            ObjectKind::Library,
            None,
        )
        .unwrap()
        .0;
        assert_eq!(full, none);
    }
}

/// Optional cross-check with GNU binutils: when `objdump` runs here, its decoded line table of the
/// COFF object names the fixture and the statement lines. Skips when it is absent.
#[test]
fn objdump_decodes_the_line_table_when_available() {
    let Ok(version) = std::process::Command::new("objdump")
        .arg("--version")
        .output()
    else {
        eprintln!("objdump_decodes_the_line_table_when_available: no objdump - skipped");
        return;
    };
    if !version.status.success() {
        eprintln!("objdump_decodes_the_line_table_when_available: objdump fails - skipped");
        return;
    }
    let (program, interner, sm) = lower(FIXTURE, 0);
    let debug = DebugInfo {
        sources: &sm,
        comp_dir: COMP_DIR,
    };
    let dir = std::env::temp_dir().join(format!("wukong_dwarf_objdump_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The host container: COFF on Windows, ELF on Linux, which the host's binutils both read.
    let triple = if cfg!(windows) { COFF } else { ELF };
    let path = dir.join("fixture.o");
    std::fs::write(
        &path,
        emit(
            &program,
            &interner,
            triple,
            0,
            false,
            ObjectKind::Library,
            Some(&debug),
        ),
    )
    .unwrap();
    let out = std::process::Command::new("objdump")
        .arg("--dwarf=decodedline")
        .arg(&path)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    if !out.status.success() {
        // An objdump without this target's format support (a cross binutils) - not a DWARF bug.
        eprintln!(
            "objdump_decodes_the_line_table_when_available: objdump cannot read the object - \
             skipped: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("dwarf_fixture.wk"), "{text}");
    for needle in ["let y = x * k;", "acc = acc + scale(i, 3);", "print(acc);"] {
        let l = line_of(FIXTURE, needle).to_string();
        assert!(
            text.lines().any(|row| row.contains("dwarf_fixture.wk")
                && row.split_whitespace().nth(1) == Some(l.as_str())),
            "objdump shows no row at line {l} (`{needle}`):\n{text}"
        );
    }
}

/// (3) The JIT perf map: one `<hex start> <hex size> <name>` line per compiled function (user
/// functions and raw-AVX2 vector kernels), non-overlapping, `main` at the entry pointer.
#[test]
fn jit_perf_map_lists_every_function() {
    // A helper, and at -O2 a long elementwise loop the vectorizer may give a 256-bit kernel.
    let src = "fn twice(x: f32) -> f32 {
    return x * 2.0;
}

fn main() -> i32 {
    let mut a: [f32; 4096] = [1.5; 4096];
    for i in 0..4096 {
        a[i] = a[i] * 3.0 + 1.0;
    }
    print(twice(a[7]));
    return 0;
}
";
    let dir = std::env::temp_dir().join(format!("wukong_perf_map_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for level in [0u8, 2] {
        let (program, interner, _) = lower(src, level);
        let path = dir.join(format!("perf-O{level}.map"));
        let _ = std::fs::remove_file(&path);
        let main = interner.get("main").unwrap();
        let jit = jit_compile_entry_mapped(
            &program,
            main,
            &interner,
            false,
            TargetCpu::Native,
            BackendOpt::for_level(level),
            true,
            Some(&path),
        )
        .unwrap_or_else(|e| panic!("jit: {e}"));
        let text = std::fs::read_to_string(&path).unwrap();
        let mut entries = Vec::new();
        for line in text.lines() {
            let fields: Vec<&str> = line.split(' ').collect();
            assert_eq!(
                fields.len(),
                3,
                "-O{level}: `{line}` is not `<start> <size> <name>`"
            );
            for hex in &fields[..2] {
                assert!(
                    !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()),
                    "-O{level}: `{hex}` in `{line}` is not bare hex"
                );
                assert!(!hex.starts_with("0x"));
            }
            let start = u64::from_str_radix(fields[0], 16).unwrap();
            let size = u64::from_str_radix(fields[1], 16).unwrap();
            assert!(size > 0, "-O{level}: `{line}` has no code");
            entries.push((start, size, fields[2].to_string()));
        }
        let kernels: usize = program.funcs.iter().map(|f| f.vec_kernels.len()).sum();
        assert_eq!(
            entries.len(),
            program.funcs.len() + kernels,
            "-O{level}: one line per function and vector kernel:\n{text}"
        );
        for f in &program.funcs {
            let name = interner.resolve(f.name);
            assert!(
                entries.iter().any(|e| e.2 == name),
                "-O{level}: `{name}` missing:\n{text}"
            );
        }
        let main_entry = entries.iter().find(|e| e.2 == "main").unwrap();
        assert_eq!(
            main_entry.0, jit.code as u64,
            "-O{level}: main is not at the entry"
        );
        entries.sort();
        for w in entries.windows(2) {
            assert!(
                w[0].0 + w[0].1 <= w[1].0,
                "-O{level}: `{:?}` overlaps `{:?}`",
                w[0],
                w[1]
            );
        }
        // The program still runs (the map changed nothing): a[7] = 1.5 * 3 + 1 = 5.5, twice = 11.
        assert_eq!(jit.run().unwrap(), (0, b"11\n".to_vec()));
        drop(jit);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The perf-map line format, and where the map goes.
#[test]
fn perf_map_line_format_and_path() {
    let text = debuginfo::perf_map::lines(&[
        (0x7f00_1000, 0x40, "main".to_string()),
        (0x7f00_1040, 0x1a, "a b".to_string()),
    ]);
    assert_eq!(text, "7f001000 40 main\n7f001040 1a a_b\n");
    let path = debuginfo::perf_map::default_path();
    assert_eq!(
        path.file_name().unwrap().to_string_lossy(),
        format!("perf-{}.map", std::process::id())
    );
    if cfg!(target_os = "linux") {
        assert_eq!(path.parent().unwrap(), std::path::Path::new("/tmp"));
    }
}
