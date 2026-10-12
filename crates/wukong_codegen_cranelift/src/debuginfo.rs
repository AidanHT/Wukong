//! Where the backend put each function's machine code, and the two consumers of that record: the
//! DWARF a `-g` object carries (`--emit=obj|exe -g`, [`attach_dwarf`]) and the JIT's perf map
//! (`WUKONG_PERF_MAP=1`, [`perf_map`]).
//!
//! The record is a [`CodeLog`]: `populate_module` fills it, when asked, as it defines functions —
//! each one's `FuncId`, its machine-code size and, for DWARF, its sorted srcloc ranges and its
//! call-frame information. Recording changes no emitted byte: it only reads the finished
//! `CompiledCode` after `define_function`, so an object built without `-g` is byte-identical to one
//! built before this module existed, and a `-g` object's `.text` is byte-identical to the plain
//! one's (`dwarf_tests` pins both).
//!
//! # The DWARF a `-g` object carries
//!
//! One DWARF 4 compile unit (`DW_TAG_compile_unit`) per object, named after the root source file,
//! with `DW_AT_comp_dir`, a producer string, a vendor language code ([`DW_LANG_WUKONG`]) and
//! `DW_AT_ranges` listing every function's code. Under it, one `DW_TAG_subprogram` per user
//! function and vector kernel: its source name (never the `--emit=exe` mangled symbol, so a
//! debugger's `break f` finds `f`), `DW_AT_external` for an exported symbol,
//! `DW_AT_decl_file`/`DW_AT_decl_line` from the function's declaration span, and
//! `DW_AT_low_pc`/`DW_AT_high_pc`. A raw-AVX2 vector kernel is `DW_AT_artificial`.
//!
//! The line program (`.debug_line`) has one sequence per user function. Every Cranelift instruction
//! carries the `ValueId` of the MIR value it was lowered for as its `SourceLoc` (`FnTranslator::mark`),
//! so each srcloc range maps back through `Function::value_loc` to a span, and the source map turns
//! that into a file, a line and a column. Rows:
//!
//! * offset 0 is the function's declaration line (the prologue);
//! * each located range starts a row at its own line and column, the first one marked
//!   `prologue_end`;
//! * a range whose value has no location (values a pass other than inlining and unrolling created,
//!   about a quarter of them at `-O2`) continues the row before it rather than claiming a line;
//! * an inlined callee's code keeps the callee's lines, which is what a line table means for
//!   inlined code without inlined-subroutine entries.
//!
//! `.debug_frame` holds one CIE (the ISA's initial rules) and an FDE for every function Cranelift
//! compiled, the executable's entry thunk and C `main` included, from Cranelift's own unwind
//! information: so a debugger unwinds through every Wukong frame (on Windows too, where the
//! object carries no `.pdata`). The entry functions have frames but no subprogram; a debugger
//! names them by their symbols.
//!
//! Addresses are relocations against the function symbols and section offsets are relocations
//! against the debug sections, so the object links like any other: `R_X86_64_64`/`R_X86_64_32` in
//! ELF, `IMAGE_REL_AMD64_ADDR64`/`IMAGE_REL_AMD64_SECREL` in COFF.
//!
//! Not emitted: variables, types, scopes and inlined-subroutine entries. A debugger gets
//! breakpoints by function, file and line, stepping by line, and a backtrace with function names
//! and lines; it cannot print a local.

use std::collections::HashMap;

use cranelift_codegen::isa::unwind::{UnwindInfo, UnwindInfoKind};
use cranelift_codegen::isa::TargetIsa;
use cranelift_module::FuncId;
use cranelift_object::object::write::{Object, Relocation as ObjectRelocation, SymbolId};
use cranelift_object::object::{
    BinaryFormat, RelocationEncoding, RelocationFlags, RelocationKind, SectionKind, SymbolScope,
};
use cranelift_object::ObjectProduct;
use gimli::write::{
    Address, AttributeValue, DwarfUnit, EndianVec, FileId, FrameTable, LineProgram, LineString,
    Range, RangeList, RelocateWriter, Relocation, RelocationTarget, Sections,
};
use gimli::{constants, Encoding, Format, LineEncoding, RunTimeEndian};

use wukong_mir::{Program, ValueId};
use wukong_span::{Interner, SourceId, SourceMap, Span};

/// The source side of `-g`: the map that turns a span into a file, a line and a column, and the
/// directory relative file names are resolved against (`DW_AT_comp_dir`, normally the directory
/// the compiler ran in).
pub struct DebugInfo<'a> {
    /// Every source file of the compilation, the root first (`SourceId(0)`).
    pub sources: &'a SourceMap,
    /// The compilation directory.
    pub comp_dir: &'a str,
}

/// What produced one defined function's machine code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CodeOrigin {
    /// `program.funcs[i]`.
    User(usize),
    /// Raw-AVX2 vector kernel `ki` of `program.funcs[fi]` (`__wukong_veckernel_<fi>_<ki>`).
    VecKernel(usize, usize),
    /// A function the backend synthesized, by its symbol: an executable's entry thunk and C `main`.
    Entry(&'static str),
}

/// One function `populate_module` defined.
#[derive(Clone, Debug)]
pub(crate) struct DefinedCode {
    pub id: FuncId,
    pub origin: CodeOrigin,
    /// Machine-code bytes (the function's whole buffer, constants included).
    pub size: u32,
    /// Sorted `(start, end, srcloc bits)` ranges; empty unless [`CodeLog::srclocs`] is set.
    pub srclocs: Vec<(u32, u32, u32)>,
    /// The function's call-frame information, when [`CodeLog::srclocs`] is set and Cranelift
    /// produced it (a vector kernel's raw bytes have none).
    pub cfi: Option<CfaUnwindInfo>,
}

/// Cranelift's System V call-frame information for one function: the CFA and saved-register rules
/// its prologue sets up, which become an FDE of `.debug_frame`.
pub(crate) type CfaUnwindInfo = cranelift_codegen::isa::unwind::systemv::UnwindInfo;

/// What [`compiled_extent`] takes from a compiled function.
pub(crate) struct Extent {
    pub size: u32,
    pub srclocs: Vec<(u32, u32, u32)>,
    pub cfi: Option<CfaUnwindInfo>,
}

/// Where `populate_module` defined each function. `None` at a call site records nothing.
#[derive(Debug, Default)]
pub(crate) struct CodeLog {
    /// Also record each function's srcloc ranges (DWARF needs them; a perf map needs only sizes).
    pub srclocs: bool,
    pub defined: Vec<DefinedCode>,
}

impl CodeLog {
    pub(crate) fn new(srclocs: bool) -> CodeLog {
        CodeLog {
            srclocs,
            defined: Vec::new(),
        }
    }

    /// Record a function Cranelift compiled for `isa`.
    pub(crate) fn record_compiled(
        &mut self,
        id: FuncId,
        origin: CodeOrigin,
        code: &cranelift_codegen::CompiledCode,
        isa: &dyn TargetIsa,
    ) {
        let extent = compiled_extent(code, self.srclocs.then_some(isa));
        self.record(id, origin, extent);
    }

    /// Record a function whose [`Extent`] was taken already (on a parallel codegen worker).
    pub(crate) fn record(&mut self, id: FuncId, origin: CodeOrigin, extent: Extent) {
        self.defined.push(DefinedCode {
            id,
            origin,
            size: extent.size,
            srclocs: extent.srclocs,
            cfi: extent.cfi,
        });
    }
}

/// A compiled function's size and, with `debug` (the ISA it was compiled for), its sorted srcloc
/// ranges and its call-frame information. Split out of [`CodeLog::record_compiled`] so the parallel
/// codegen workers can take it before the defining loop (which runs serially, in source order)
/// records it.
pub(crate) fn compiled_extent(
    code: &cranelift_codegen::CompiledCode,
    debug: Option<&dyn TargetIsa>,
) -> Extent {
    let size = code.buffer.data().len() as u32;
    let Some(isa) = debug else {
        return Extent {
            size,
            srclocs: Vec::new(),
            cfi: None,
        };
    };
    let srclocs = code
        .buffer
        .get_srclocs_sorted()
        .iter()
        .map(|r| (r.start, r.end, r.loc.bits()))
        .collect();
    let cfi = match isa.emit_unwind_info(code, UnwindInfoKind::SystemV) {
        Ok(Some(UnwindInfo::SystemV(info))) => Some(info),
        _ => None,
    };
    Extent { size, srclocs, cfi }
}

/// The name a [`DefinedCode`] goes by in a perf map and in `DW_AT_name`: the source name of a user
/// function, the module symbol of a vector kernel.
pub(crate) fn display_name(code: &DefinedCode, program: &Program, interner: &Interner) -> String {
    match code.origin {
        CodeOrigin::User(fi) => interner.resolve(program.funcs[fi].name).to_string(),
        CodeOrigin::VecKernel(fi, ki) => format!("__wukong_veckernel_{fi}_{ki}"),
        CodeOrigin::Entry(symbol) => symbol.to_string(),
    }
}

/// `DW_LANG_*` has no code for Wukong, so the compile unit names a vendor code from the
/// `DW_LANG_lo_user` range. A debugger treats an unknown language as its minimal one: breakpoints,
/// stepping and backtraces work, and nothing pretends the source is C.
pub const DW_LANG_WUKONG: constants::DwLang = constants::DwLang(0x8057);

/// The producer string (`DW_AT_producer`).
fn producer() -> String {
    format!("wukongc {}", env!("CARGO_PKG_VERSION"))
}

/// A section writer that records the relocations gimli asks for instead of resolving them.
#[derive(Clone)]
struct RelocWriter {
    data: EndianVec<RunTimeEndian>,
    relocs: Vec<Relocation>,
}

impl RelocateWriter for RelocWriter {
    type Writer = EndianVec<RunTimeEndian>;

    fn writer(&self) -> &Self::Writer {
        &self.data
    }

    fn writer_mut(&mut self) -> &mut Self::Writer {
        &mut self.data
    }

    fn relocate(&mut self, relocation: Relocation) {
        self.relocs.push(relocation);
    }
}

/// A `LineString` must be non-empty and NUL-free (gimli asserts both).
fn line_string(s: &str) -> LineString {
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != 0).collect();
    LineString::String(if bytes.is_empty() {
        b"<unknown>".to_vec()
    } else {
        bytes
    })
}

/// `name` split into its directory and file name at the last `/` or `\` (both, whatever the host:
/// the name is the one the user typed). `None` for the directory means "relative to comp_dir".
fn split_path(name: &str) -> (Option<&str>, &str) {
    match name.rfind(['/', '\\']) {
        Some(0) => (Some(&name[..1]), &name[1..]),
        Some(i) => (Some(&name[..i]), &name[i + 1..]),
        None => (None, name),
    }
}

/// The file table of the line program, filled in first-use order (deterministic: functions in
/// program order, ranges in address order).
struct Files<'a> {
    sources: &'a SourceMap,
    ids: HashMap<SourceId, FileId>,
}

impl Files<'_> {
    fn file(&mut self, program: &mut LineProgram, source: SourceId) -> FileId {
        if let Some(&id) = self.ids.get(&source) {
            return id;
        }
        let (dir, file) = split_path(self.sources.name(source));
        let dir = match dir {
            Some(dir) => program.add_directory(line_string(dir)),
            None => program.default_directory(),
        };
        let id = program.add_file(line_string(file), dir, None);
        self.ids.insert(source, id);
        id
    }

    /// `span` as `(file, line, column)`, when the map holds it (a span from a file the map does not
    /// hold, or past its end, is treated as unknown rather than indexed).
    fn locate(&mut self, program: &mut LineProgram, span: Span) -> Option<(FileId, u64, u64)> {
        if span.is_dummy()
            || span.source.0 as usize >= self.sources.file_count()
            || span.lo as usize > self.sources.source(span.source).len()
        {
            return None;
        }
        let loc = self.sources.span_location(span);
        let file = self.file(program, span.source);
        Some((file, u64::from(loc.line), u64::from(loc.col)))
    }
}

/// One row of a function's line sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    offset: u64,
    file: FileId,
    line: u64,
    column: u64,
    prologue_end: bool,
}

/// The rows of one user function's sequence (see the module docs for the rules). Empty when
/// neither the declaration nor any range is located.
fn function_rows(
    files: &mut Files<'_>,
    program: &mut LineProgram,
    f: &wukong_mir::Function,
    srclocs: &[(u32, u32, u32)],
) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    if let Some((file, line, column)) = f.decl_span().and_then(|s| files.locate(program, s)) {
        rows.push(Row {
            offset: 0,
            file,
            line,
            column,
            prologue_end: false,
        });
    }
    let mut in_body = false;
    for &(start, end, bits) in srclocs {
        if start >= end {
            continue;
        }
        let Some((file, line, column)) = f
            .value_loc(ValueId(bits))
            .and_then(|s| files.locate(program, s))
        else {
            // Unknown: the row before covers it.
            continue;
        };
        let row = Row {
            offset: u64::from(start),
            file,
            line,
            column,
            prologue_end: !in_body && start > 0,
        };
        in_body = true;
        match rows.last_mut() {
            Some(last) if (last.file, last.line, last.column) == (file, line, column) => {}
            // Two rows at one address: the later one is the location of that code.
            Some(last) if last.offset == row.offset => *last = row,
            _ => rows.push(row),
        }
    }
    rows
}

/// Add the DWARF for `log`'s functions, compiled for `isa`, to `product`'s object: `.debug_info`,
/// `.debug_abbrev`, `.debug_line`, `.debug_str`, `.debug_ranges` and `.debug_frame`, with their
/// relocations. Only ELF and COFF objects are supported.
pub(crate) fn attach_dwarf(
    product: &mut ObjectProduct,
    isa: &dyn TargetIsa,
    program: &Program,
    interner: &Interner,
    log: &CodeLog,
    debug: &DebugInfo<'_>,
) -> Result<(), String> {
    let format = product.object.format();
    if !matches!(format, BinaryFormat::Elf | BinaryFormat::Coff) {
        return Err(format!(
            "`-g` emits DWARF into ELF and COFF objects only; this target's object format is {format:?}"
        ));
    }
    let encoding = Encoding {
        format: Format::Dwarf32,
        version: 4,
        address_size: 8,
    };
    let root_name = if debug.sources.file_count() > 0 {
        debug.sources.name(SourceId(0)).to_string()
    } else {
        "<unknown>".to_string()
    };
    let mut dwarf = DwarfUnit::new(encoding);
    dwarf.unit.line_program = LineProgram::new(
        encoding,
        LineEncoding::default(),
        line_string(debug.comp_dir),
        None,
        line_string(&root_name),
        None,
    );
    let mut files = Files {
        sources: debug.sources,
        ids: HashMap::new(),
    };

    // Each symbol an address relocates against, by its gimli `Address::Symbol` index.
    let mut symbols: Vec<SymbolId> = Vec::new();
    let mut ranges = Vec::new();
    // Call-frame information: one CIE (the ISA's initial rules) and an FDE per function that has
    // unwind information, so a debugger can unwind through every frame.
    let mut frames = FrameTable::default();
    let cie = isa.create_systemv_cie().map(|cie| frames.add_cie(cie));
    let root = dwarf.unit.root();
    for code in &log.defined {
        if code.size == 0 {
            continue;
        }
        let symbol = product.function_symbol(code.id);
        let index = symbols.len();
        symbols.push(symbol);
        let start = Address::Symbol {
            symbol: index,
            addend: 0,
        };
        if let (Some(cie), Some(cfi)) = (cie, &code.cfi) {
            frames.add_fde(cie, cfi.to_fde(start));
        }
        // The entry thunk and the C `main` get call frames (a backtrace unwinds through them) but
        // no subprogram: a debugger names them by their symbols, and `main` then means the
        // user's `main`, the one subprogram of that name.
        if let CodeOrigin::Entry(_) = code.origin {
            continue;
        }
        ranges.push(Range::StartLength {
            begin: start,
            length: u64::from(code.size),
        });

        let name = display_name(code, program, interner);
        let external = product.object.symbol(symbol).scope != SymbolScope::Compilation;

        let die = dwarf.unit.add(root, constants::DW_TAG_subprogram);
        let name_id = dwarf.strings.add(name);
        let mut decl = None;
        if let CodeOrigin::User(fi) = code.origin {
            let f = &program.funcs[fi];
            decl = f
                .decl_span()
                .and_then(|s| files.locate(&mut dwarf.unit.line_program, s));
            let rows = function_rows(&mut files, &mut dwarf.unit.line_program, f, &code.srclocs);
            if !rows.is_empty() {
                let lp = &mut dwarf.unit.line_program;
                lp.begin_sequence(Some(start));
                for row in rows {
                    let r = lp.row();
                    r.address_offset = row.offset;
                    r.file = row.file;
                    r.line = row.line;
                    r.column = row.column;
                    r.prologue_end = row.prologue_end;
                    lp.generate_row();
                }
                lp.end_sequence(u64::from(code.size));
            }
        }
        let entry = dwarf.unit.get_mut(die);
        entry.set(constants::DW_AT_name, AttributeValue::StringRef(name_id));
        if external {
            entry.set(constants::DW_AT_external, AttributeValue::Flag(true));
        }
        if let CodeOrigin::VecKernel(..) = code.origin {
            entry.set(constants::DW_AT_artificial, AttributeValue::Flag(true));
        }
        if let Some((file, line, _)) = decl {
            entry.set(
                constants::DW_AT_decl_file,
                AttributeValue::FileIndex(Some(file)),
            );
            entry.set(constants::DW_AT_decl_line, AttributeValue::Udata(line));
        }
        entry.set(constants::DW_AT_low_pc, AttributeValue::Address(start));
        entry.set(
            constants::DW_AT_high_pc,
            AttributeValue::Udata(u64::from(code.size)),
        );
    }

    let producer = dwarf.strings.add(producer());
    let name = dwarf.strings.add(root_name.into_bytes());
    let comp_dir = dwarf.strings.add(line_string_bytes(debug.comp_dir));
    let range_list = dwarf.unit.ranges.add(RangeList(ranges));
    let cu = dwarf.unit.get_mut(root);
    cu.set(
        constants::DW_AT_producer,
        AttributeValue::StringRef(producer),
    );
    cu.set(
        constants::DW_AT_language,
        AttributeValue::Language(DW_LANG_WUKONG),
    );
    cu.set(constants::DW_AT_name, AttributeValue::StringRef(name));
    cu.set(
        constants::DW_AT_comp_dir,
        AttributeValue::StringRef(comp_dir),
    );
    // The base address `DW_AT_ranges` entries are relative to: they are absolute (relocated).
    cu.set(
        constants::DW_AT_low_pc,
        AttributeValue::Address(Address::Constant(0)),
    );
    cu.set(
        constants::DW_AT_ranges,
        AttributeValue::RangeListRef(range_list),
    );

    let mut sections = Sections::new(RelocWriter {
        data: EndianVec::new(RunTimeEndian::Little),
        relocs: Vec::new(),
    });
    dwarf
        .write(&mut sections)
        .map_err(|e| format!("DWARF: {e}"))?;
    frames
        .write_debug_frame(&mut sections.debug_frame)
        .map_err(|e| format!("DWARF call frames: {e}"))?;
    write_sections(&mut product.object, &sections, &symbols, format)
}

/// `s`'s bytes, NUL-free and non-empty (a `.debug_str` entry may not contain a NUL either).
fn line_string_bytes(s: &str) -> Vec<u8> {
    match line_string(s) {
        LineString::String(bytes) => bytes,
        _ => unreachable!("line_string builds an inline string"),
    }
}

/// Add each non-empty section of `sections` to `object` as a debug section, then its relocations
/// (in a second pass: a relocation may name any other debug section, and COFF writes an addend into
/// the section's data, which must be in place first).
fn write_sections(
    object: &mut Object<'static>,
    sections: &Sections<RelocWriter>,
    symbols: &[SymbolId],
    format: BinaryFormat,
) -> Result<(), String> {
    let mut ids = HashMap::new();
    sections.for_each(|id, w| -> Result<(), String> {
        if !w.data.slice().is_empty() {
            let section = object.add_section(
                Vec::new(),
                id.name().as_bytes().to_vec(),
                SectionKind::Debug,
            );
            object.set_section_data(section, w.data.slice().to_vec(), 1);
            ids.insert(id, section);
        }
        Ok(())
    })?;
    sections.for_each(|id, w| -> Result<(), String> {
        let Some(&section) = ids.get(&id) else {
            return Ok(());
        };
        for r in &w.relocs {
            let (symbol, kind) = match r.target {
                RelocationTarget::Symbol(i) => (symbols[i], RelocationKind::Absolute),
                RelocationTarget::Section(target) => {
                    let target = *ids.get(&target).ok_or_else(|| {
                        format!("DWARF: {} refers to empty {}", id.name(), target.name())
                    })?;
                    let kind = if format == BinaryFormat::Coff {
                        RelocationKind::SectionOffset
                    } else {
                        RelocationKind::Absolute
                    };
                    (object.section_symbol(target), kind)
                }
            };
            object
                .add_relocation(
                    section,
                    ObjectRelocation {
                        offset: r.offset as u64,
                        symbol,
                        addend: r.addend,
                        flags: RelocationFlags::Generic {
                            kind,
                            encoding: RelocationEncoding::Generic,
                            size: r.size * 8,
                        },
                    },
                )
                .map_err(|e| format!("DWARF relocation in {}: {e}", id.name()))?;
        }
        Ok(())
    })
}

/// The JIT's perf map (`WUKONG_PERF_MAP=1`): `perf-<pid>.map`, one `<start> <size> <name>` line
/// (both hex, no `0x`) per JIT-compiled function, the format Linux `perf` reads to name samples in
/// JIT code. Appended per compile, so a process that JIT-compiles several programs lists them all.
pub(crate) mod perf_map {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// The environment variable that turns the map on (`1`).
    pub const ENV: &str = "WUKONG_PERF_MAP";

    /// Serializes appends from concurrent compiles in one process, so lines never interleave.
    static APPEND: Mutex<()> = Mutex::new(());

    /// The map this process writes when `WUKONG_PERF_MAP=1`, else `None`: `/tmp/perf-<pid>.map` on
    /// Linux (where `perf` looks), `std::env::temp_dir()` elsewhere.
    pub fn from_env() -> Option<PathBuf> {
        (std::env::var(ENV).ok().as_deref() == Some("1")).then(default_path)
    }

    pub fn default_path() -> PathBuf {
        let dir = if cfg!(target_os = "linux") {
            PathBuf::from("/tmp")
        } else {
            std::env::temp_dir()
        };
        dir.join(format!("perf-{}.map", std::process::id()))
    }

    /// The map's lines for `entries` (`(start, size, name)`).
    pub fn lines(entries: &[(usize, u32, String)]) -> String {
        let mut out = String::new();
        for (start, size, name) in entries {
            // A name is one whitespace-free token in the map.
            let name: String = name
                .chars()
                .map(|c| if c.is_whitespace() { '_' } else { c })
                .collect();
            out.push_str(&format!("{start:x} {size:x} {name}\n"));
        }
        out
    }

    /// Append `entries` to `path` in one write.
    pub fn append(path: &Path, entries: &[(usize, u32, String)]) -> std::io::Result<()> {
        let text = lines(entries);
        let _guard = APPEND.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(text.as_bytes())
    }
}
