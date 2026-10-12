//! Typed textual MIR: a dump that reads back.
//!
//! [`crate::print`] is a debugging view: it leaves arithmetic results untyped, prints a float
//! constant through `Display` (which loses a NaN's payload), and drops the statics, the `extern`
//! signedness and the parameter promises. [`print_program_typed`] adds exactly what a reader needs
//! to rebuild the [`Program`], and [`parse_program`] rebuilds it. `wukongc --emit=mir-text` prints
//! this form and `wukongc <file>.mir` reads it, so an optimizer bug can be reproduced, and a
//! regression test written, from a dump instead of a hand-built `Builder` program.
//!
//! The format is `print_function`'s, line for line, plus:
//!
//! - a first line `; wukong mir text v1`, and `;` comment lines anywhere;
//! - `declare extern "C" fn NAME(i32 signed, ptr) -> i32 signed`, with the C signedness explicit;
//! - `static NAME = hex <bytes>` for each `.rodata` blob;
//! - parameter promises on the function header: `v0 ptr noalias slice extent=64x4`;
//! - an `  entry bbK` line after the header when the entry block is not `bb0`;
//! - a `  values N` line after it when the value arena is longer than the largest defined id + 1
//!   (a pass deleted the highest-numbered values), so the parsed function numbers the values a
//!   pass creates exactly as the original does and a dump reproduces the rest of the pipeline;
//! - a type on every result, `v7: <4 x f32> = fadd v5, v6`;
//! - every float constant as the bits of its `f64` payload, `const.f32 0x3ff8000000000000`.
//!
//! The printer post-processes `print_function`'s output in lockstep with the blocks and
//! instructions and asserts the line count, so a change to that layout fails loudly here rather
//! than producing text this parser misreads. The laws, pinned by the tests below and by
//! `crates/wukongc/tests/mir_text.rs` over the whole run corpus: `typed(parse(typed(p))) ==
//! typed(p)`, `print_program(parse(typed(p))) == print_program(p)`, and the parsed program verifies
//! whenever `p` did. A synthesized 256-bit vector-kernel recipe (`Function::vec_kernels`) has no
//! textual form, so a `veckernel` line is a parse error; dump such a program with
//! `--target-cpu=x86-64-v2`, which builds none.

use std::collections::HashSet;
use std::fmt;

use crate::{
    AddrSpace, BasicBlock, BinOp, BlockId, CastKind, CmpOp, ExternFn, ExternParam, Function, Half,
    Inst, IntUnOp, MirLevel, MirType, NarrowKind, Op, ParamAttrs, ParamExtent, Program, RoundMode,
    StaticData, SwitchCase, Terminator, ValueId,
};
use wukong_span::Interner;

/// The first line of every typed dump; the version names this format.
pub const HEADER: &str = "; wukong mir text v1";

/// The largest value arena a parsed function may ask for: the arena is sized by the largest id the
/// text mentions, so one mistyped `v4000000000` must be an error, not a multi-gigabyte allocation.
/// Far above any function the compiler builds.
pub const MAX_VALUES: u32 = 1 << 24;

/// The message for a `veckernel` line, which has no textual form.
pub const VECKERNEL_UNREPRESENTABLE: &str =
    "vector-kernel recipes have no textual form; dump with --target-cpu=x86-64-v2";

/// Render a whole program in the typed, parseable form.
pub fn print_program_typed(p: &Program, interner: &Interner) -> String {
    let mut out = format!("{HEADER}\n");
    for e in &p.externs {
        let params: Vec<String> = e.params.iter().map(extern_param).collect();
        out.push_str(&format!(
            "declare extern \"C\" fn {}({}) -> {}\n",
            interner.resolve(e.name),
            params.join(", "),
            extern_param(&e.ret)
        ));
    }
    for s in &p.statics {
        let hex: String = s.bytes.iter().map(|b| format!("{b:02x}")).collect();
        let sep = if hex.is_empty() { "" } else { " " };
        out.push_str(&format!(
            "static {} = hex{sep}{hex}\n",
            interner.resolve(s.name)
        ));
    }
    if !p.externs.is_empty() || !p.statics.is_empty() {
        out.push('\n');
    }
    for f in &p.funcs {
        out.push_str(&print_function_typed(f, interner));
        out.push('\n');
    }
    out
}

fn extern_param(q: &ExternParam) -> String {
    if q.signed {
        format!("{} signed", q.ty.display())
    } else {
        q.ty.display()
    }
}

/// Render one function in the typed form: [`crate::print::print_function`]'s lines, with the
/// header's parameter promises, the entry block, every result's type and every float constant's
/// bits added.
pub fn print_function_typed(f: &Function, interner: &Interner) -> String {
    let plain = crate::print::print_function(f, interner);
    let lines: Vec<&str> = plain.lines().collect();
    let expected = 2 + f.blocks.iter().map(|b| b.insts.len() + 2).sum::<usize>();
    assert_eq!(
        lines.len(),
        expected,
        "print_function no longer prints one line per header, block, instruction and \
         terminator; wukong_mir::text must follow its layout"
    );
    let mut out = String::new();
    if !f.vec_kernels.is_empty() {
        out.push_str(&format!(
            "; {} vector-kernel recipe(s) are not representable\n",
            f.vec_kernels.len()
        ));
    }
    let params: Vec<String> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let attrs = f.param_attr(i);
            let mut s = format!("v{} {}", v.0, type_of(f, *v));
            if attrs.noalias {
                s.push_str(" noalias");
            }
            if attrs.slice {
                s.push_str(" slice");
            }
            if let Some(e) = attrs.extent {
                s.push_str(&format!(" extent={}x{}", e.elems, e.elem_bytes));
            }
            s
        })
        .collect();
    out.push_str(&format!(
        "fn {}({}) -> {} {{\n",
        interner.resolve(f.name),
        params.join(", "),
        f.ret.display()
    ));
    if f.entry.0 != 0 {
        out.push_str(&format!("  entry bb{}\n", f.entry.0));
    }
    let defined_end = f
        .params
        .iter()
        .chain(f.blocks.iter().flat_map(|b| {
            b.params
                .iter()
                .chain(b.insts.iter().filter_map(|i| i.result.as_ref()))
        }))
        .map(|v| v.0 as usize + 1)
        .max()
        .unwrap_or(0);
    if f.value_types.len() > defined_end {
        out.push_str(&format!("  values {}\n", f.value_types.len()));
    }
    let mut at = 1;
    for b in &f.blocks {
        out.push_str(lines[at]);
        out.push('\n');
        at += 1;
        for inst in &b.insts {
            out.push_str(&typed_inst(lines[at], inst, f));
            out.push('\n');
            at += 1;
        }
        out.push_str(lines[at]);
        out.push('\n');
        at += 1;
    }
    out.push_str(lines[at]);
    out.push('\n');
    out
}

/// A value's type for printing; a value outside the arena (invalid MIR, which `--emit=mir-text`
/// still dumps before the verifier's report) prints as `?`, which no parser accepts.
fn type_of(f: &Function, v: ValueId) -> String {
    f.value_types
        .get(v.0 as usize)
        .map_or_else(|| "?".to_string(), MirType::display)
}

fn typed_inst(line: &str, inst: &Inst, f: &Function) -> String {
    let rhs = match inst.result {
        Some(r) => {
            let prefix = format!("    v{} = ", r.0);
            line.strip_prefix(prefix.as_str())
                .unwrap_or_else(|| panic!("print_function changed its result form: `{line}`"))
        }
        None => line.trim_start(),
    };
    let rhs = match &inst.op {
        // `Display` prints every NaN as `NaN` and drops its payload; the bits keep it.
        Op::ConstFloat(v, ty) => format!("const.{} 0x{:016x}", ty.display(), v.to_bits()),
        _ => rhs.to_string(),
    };
    match inst.result {
        Some(r) => format!("    v{}: {} = {rhs}", r.0, type_of(f, r)),
        None => format!("    {rhs}"),
    }
}

/// A typed-MIR syntax error: the 1-based line and what is wrong with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for ParseError {}

/// Parse the typed form back into a [`Program`], interning every name in `interner`. Value and
/// block ids are kept as written: the value arena is sized to the largest id mentioned plus one
/// (or a function's `values N`, when larger), with `void` in the gaps, and blocks must appear in
/// id order. The result is not verified; run
/// [`crate::verify::verify_program`] before handing it to anything else.
pub fn parse_program(src: &str, interner: &mut Interner) -> Result<Program, ParseError> {
    let mut program = Program::new();
    let mut func: Option<FnParser> = None;
    for (index, raw) in src.lines().enumerate() {
        let line = index + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with(';') {
            continue;
        }
        let err = |msg: String| ParseError { line, msg };
        match func.as_mut() {
            Some(fp) => {
                if text == "}" {
                    let fp = func.take().expect("inside a function");
                    program.funcs.push(fp.finish().map_err(err)?);
                } else {
                    fp.line(text, interner).map_err(err)?;
                }
            }
            None => {
                if let Some(rest) = text.strip_prefix("fn ") {
                    func = Some(FnParser::header(rest, interner).map_err(err)?);
                } else if let Some(rest) = text.strip_prefix("declare extern \"C\" fn ") {
                    program
                        .externs
                        .push(parse_extern(rest, interner).map_err(err)?);
                } else if let Some(rest) = text.strip_prefix("static ") {
                    program
                        .statics
                        .push(parse_static(rest, interner).map_err(err)?);
                } else {
                    return Err(err(format!(
                        "expected `fn`, `declare extern \"C\" fn` or `static`, found `{text}`"
                    )));
                }
            }
        }
    }
    if let Some(fp) = func {
        return Err(ParseError {
            line: src.lines().count(),
            msg: format!(
                "function `{}` is not closed with `}}`",
                interner.resolve(fp.name)
            ),
        });
    }
    debug_assert_eq!(program.level, MirLevel::Low);
    Ok(program)
}

fn parse_extern(rest: &str, interner: &mut Interner) -> Result<ExternFn, String> {
    let (sig, ret) = rest
        .rsplit_once(" -> ")
        .ok_or("an extern declaration needs ` -> <type>`")?;
    let (name, params) = split_call(sig)?;
    let params = split_top(params)
        .into_iter()
        .map(parse_extern_param)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ExternFn {
        name: interner.intern(name),
        params,
        ret: parse_extern_param(ret)?,
    })
}

fn parse_extern_param(s: &str) -> Result<ExternParam, String> {
    let s = s.trim();
    match s.strip_suffix(" signed") {
        Some(ty) => Ok(ExternParam {
            ty: parse_type(ty)?,
            signed: true,
        }),
        None => Ok(ExternParam {
            ty: parse_type(s)?,
            signed: false,
        }),
    }
}

fn parse_static(rest: &str, interner: &mut Interner) -> Result<StaticData, String> {
    let (name, hex) = rest
        .rsplit_once(" = hex")
        .ok_or("a static needs ` = hex <bytes>`")?;
    let hex = hex.trim();
    if hex.len() % 2 != 0 {
        return Err(format!(
            "static bytes `{hex}` have an odd number of hex digits"
        ));
    }
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| {
            hex.get(i..i + 2)
                .and_then(|d| u8::from_str_radix(d, 16).ok())
                .ok_or_else(|| format!("static bytes `{hex}` are not hex"))
        })
        .collect::<Result<Vec<u8>, _>>()?;
    Ok(StaticData {
        name: interner.intern(name),
        bytes,
    })
}

/// One function being read: its header, the blocks so far, and the block still open.
struct FnParser {
    name: wukong_span::Symbol,
    params: Vec<ValueId>,
    param_attrs: Vec<ParamAttrs>,
    ret: MirType,
    entry: Option<BlockId>,
    /// The `values N` line: the arena is at least this long.
    arena: Option<usize>,
    blocks: Vec<BasicBlock>,
    /// The block whose terminator has not been read yet.
    open: Option<(BlockId, Vec<ValueId>, Vec<Inst>)>,
    types: Vec<Option<MirType>>,
    /// Values defined by a block parameter or an instruction (a function parameter is defined
    /// again, with the same type, by the entry block's parameter list).
    defined: HashSet<u32>,
    /// The largest value id mentioned anywhere, defined or used.
    max_value: Option<u32>,
}

impl FnParser {
    fn header(rest: &str, interner: &mut Interner) -> Result<FnParser, String> {
        let sig = rest
            .strip_suffix(" {")
            .ok_or("a function header ends with ` {`")?;
        let (sig, ret) = sig
            .rsplit_once(" -> ")
            .ok_or("a function header needs ` -> <type>`")?;
        let (name, params_text) = split_call(sig)?;
        let mut fp = FnParser {
            name: interner.intern(name),
            params: Vec::new(),
            param_attrs: Vec::new(),
            ret: parse_type(ret)?,
            entry: None,
            arena: None,
            blocks: Vec::new(),
            open: None,
            types: Vec::new(),
            defined: HashSet::new(),
            max_value: None,
        };
        for param in split_top(params_text) {
            let (v, rest) = param
                .trim()
                .split_once(' ')
                .ok_or_else(|| format!("parameter `{param}` needs a type"))?;
            let v = fp.value(v)?;
            let mut attrs = ParamAttrs::default();
            let mut ty = rest.trim();
            loop {
                let (head, last) = ty.rsplit_once(' ').unwrap_or(("", ty));
                if last == "noalias" {
                    attrs.noalias = true;
                } else if last == "slice" {
                    attrs.slice = true;
                } else if let Some(extent) = last.strip_prefix("extent=") {
                    let (elems, bytes) = extent.split_once('x').ok_or_else(|| {
                        format!("bad extent `{last}`: write extent=<elems>x<bytes>")
                    })?;
                    attrs.extent = Some(ParamExtent {
                        elems: parse_num(elems, "extent element count")?,
                        elem_bytes: parse_num(bytes, "extent element size")?,
                    });
                } else {
                    break;
                }
                ty = head.trim_end();
            }
            fp.set_type(v, parse_type(ty)?)?;
            fp.params.push(v);
            fp.param_attrs.push(attrs);
        }
        if fp.param_attrs.iter().all(|a| *a == ParamAttrs::default()) {
            fp.param_attrs.clear();
        }
        Ok(fp)
    }

    /// Read a value id, `vN`, noting it for the arena size.
    fn value(&mut self, s: &str) -> Result<ValueId, String> {
        let v = parse_value(s)?;
        if v.0 >= MAX_VALUES {
            return Err(format!(
                "v{} is past the largest value id this parser accepts (v{})",
                v.0,
                MAX_VALUES - 1
            ));
        }
        self.max_value = Some(self.max_value.map_or(v.0, |m| m.max(v.0)));
        Ok(v)
    }

    fn values(&mut self, s: &str) -> Result<Vec<ValueId>, String> {
        split_top(s).into_iter().map(|v| self.value(v)).collect()
    }

    fn set_type(&mut self, v: ValueId, ty: MirType) -> Result<(), String> {
        let i = v.0 as usize;
        if self.types.len() <= i {
            self.types.resize(i + 1, None);
        }
        match &self.types[i] {
            Some(old) if *old != ty => Err(format!(
                "v{} is declared {} and {}",
                v.0,
                old.display(),
                ty.display()
            )),
            _ => {
                self.types[i] = Some(ty);
                Ok(())
            }
        }
    }

    fn define(&mut self, v: ValueId, ty: MirType) -> Result<(), String> {
        if !self.defined.insert(v.0) {
            return Err(format!("v{} is defined twice", v.0));
        }
        self.set_type(v, ty)
    }

    fn line(&mut self, text: &str, interner: &mut Interner) -> Result<(), String> {
        if let Some(rest) = text.strip_prefix("entry ") {
            if !self.blocks.is_empty() || self.open.is_some() || self.entry.is_some() {
                return Err("`entry` must come right after the function header".to_string());
            }
            self.entry = Some(parse_block(rest)?);
            return Ok(());
        }
        if let Some(rest) = text.strip_prefix("values ") {
            if !self.blocks.is_empty() || self.open.is_some() || self.arena.is_some() {
                return Err("`values` must come before the first block".to_string());
            }
            let n: u32 = parse_num(rest, "value count")?;
            if n > MAX_VALUES {
                return Err(format!(
                    "`values {n}` is past the largest arena this parser accepts ({MAX_VALUES})"
                ));
            }
            self.arena = Some(n as usize);
            return Ok(());
        }
        if is_block_header(text) {
            if let Some((id, ..)) = &self.open {
                return Err(format!("bb{} has no terminator", id.0));
            }
            return self.block_header(text);
        }
        if self.open.is_none() {
            return Err(format!("`{text}` is outside any block"));
        }
        let word = text.split(' ').next().unwrap_or(text);
        if matches!(word, "ret" | "br" | "cond_br" | "switch" | "unreachable") {
            let term = self.terminator(text)?;
            let (id, params, insts) = self.open.take().expect("an open block");
            self.blocks.push(BasicBlock {
                id,
                params,
                insts,
                term,
            });
            return Ok(());
        }
        let inst = self.inst(text, interner)?;
        self.open.as_mut().expect("an open block").2.push(inst);
        Ok(())
    }

    fn block_header(&mut self, text: &str) -> Result<(), String> {
        let head = text.strip_suffix(':').expect("checked by is_block_header");
        let (label, params) = match head.split_once('(') {
            Some((label, params)) => (
                label,
                params
                    .strip_suffix(')')
                    .ok_or_else(|| format!("unclosed block parameter list in `{text}`"))?,
            ),
            None => (head, ""),
        };
        let id = parse_block(label)?;
        if id.0 as usize != self.blocks.len() {
            return Err(format!(
                "blocks must appear in id order: expected bb{}, found bb{}",
                self.blocks.len(),
                id.0
            ));
        }
        let mut values = Vec::new();
        for param in split_top(params) {
            let (v, ty) = param
                .trim()
                .split_once(' ')
                .ok_or_else(|| format!("block parameter `{param}` needs a type"))?;
            let v = self.value(v)?;
            self.define(v, parse_type(ty)?)?;
            values.push(v);
        }
        self.open = Some((id, values, Vec::new()));
        Ok(())
    }

    fn inst(&mut self, text: &str, interner: &mut Interner) -> Result<Inst, String> {
        let (result, rhs) = match split_result(text) {
            ResultForm::Typed(v, ty, rhs) => {
                let v = self.value(v)?;
                self.define(v, parse_type(ty)?)?;
                (Some(v), rhs)
            }
            ResultForm::Untyped(v) => {
                return Err(format!(
                    "`{v}` has no type annotation: write `{v}: <type> = ...` (dump with \
                     --emit=mir-text, not --emit=mir)"
                ))
            }
            ResultForm::Void => (None, text),
        };
        let op = self.op(rhs, interner)?;
        Ok(Inst { result, op })
    }

    fn op(&mut self, rhs: &str, interner: &mut Interner) -> Result<Op, String> {
        let (word, rest) = rhs.split_once(' ').unwrap_or((rhs, ""));
        if let Some(ty) = word.strip_prefix("const.") {
            // The type can hold spaces (`const.<4 x i32>` is not valid MIR, but it parses).
            let (ty, value) = rhs["const.".len()..]
                .rsplit_once(' ')
                .ok_or_else(|| format!("`const.{ty}` needs a value"))?;
            let ty = parse_type(ty)?;
            return Ok(match value.strip_prefix("0x") {
                Some(bits) => Op::ConstFloat(
                    f64::from_bits(
                        u64::from_str_radix(bits, 16)
                            .map_err(|_| format!("bad float bits `{value}`"))?,
                    ),
                    ty,
                ),
                None => Op::ConstInt(
                    value
                        .parse::<i128>()
                        .map_err(|_| format!("bad integer constant `{value}`"))?,
                    ty,
                ),
            });
        }
        if let Some(b) = BIN_OPS.iter().find(|b| b.name() == word) {
            let [l, r] = self.operands(rest)?;
            return Ok(Op::Bin(*b, l, r));
        }
        if let Some(pred) = word.strip_prefix("cmp.") {
            let c = CMP_OPS
                .iter()
                .find(|c| c.name() == pred)
                .ok_or_else(|| format!("unknown compare `{word}`"))?;
            let [l, r] = self.operands(rest)?;
            return Ok(Op::Cmp(*c, l, r));
        }
        if let Some(k) = CAST_KINDS.iter().find(|k| k.name() == word) {
            let (v, ty) = rest
                .split_once(" to ")
                .ok_or_else(|| format!("`{word}` needs `<value> to <type>`"))?;
            return Ok(Op::Cast(*k, self.value(v)?, parse_type(ty)?));
        }
        if let Some(u) = INT_UN_OPS.iter().find(|u| u.name() == word) {
            let [a] = self.operands(rest)?;
            return Ok(Op::IntUnary(*u, a));
        }
        if let Some(spec) = word.strip_prefix("widen.") {
            let (kind, half) = spec
                .rsplit_once('.')
                .ok_or_else(|| format!("`{word}` needs `widen.<cast>.<lo|hi>`"))?;
            let kind = *CAST_KINDS
                .iter()
                .find(|k| k.name() == kind)
                .ok_or_else(|| format!("unknown widening cast `{kind}`"))?;
            let half = match half {
                "lo" => Half::Lo,
                "hi" => Half::Hi,
                other => return Err(format!("unknown half `{other}` (expected lo or hi)")),
            };
            let [v] = self.operands(rest)?;
            return Ok(Op::Widen { v, half, kind });
        }
        if let Some(kind) = word.strip_prefix("narrow.") {
            let kind = *NARROW_KINDS
                .iter()
                .find(|k| k.name() == kind)
                .ok_or_else(|| format!("unknown narrowing `{word}`"))?;
            let [lo, hi] = self.operands(rest)?;
            return Ok(Op::Narrow { lo, hi, kind });
        }
        if let Some(mode) = word.strip_prefix("round.") {
            let mode = *ROUND_MODES
                .iter()
                .find(|m| round_name(**m) == mode)
                .ok_or_else(|| format!("unknown rounding `{word}`"))?;
            let [a] = self.operands(rest)?;
            return Ok(Op::Round(mode, a));
        }
        Ok(match word {
            "neg" => Op::Neg(self.operands::<1>(rest)?[0]),
            "not" => Op::Not(self.operands::<1>(rest)?[0]),
            "select" => {
                let [c, a, b] = self.operands(rest)?;
                Op::Select(c, a, b)
            }
            "alloca" | "alloca.shared" => {
                let space = if word == "alloca" {
                    AddrSpace::Local
                } else {
                    AddrSpace::Shared
                };
                let (ty, align) = match rest.rsplit_once(" align ") {
                    Some((ty, n)) => (ty, parse_num(n, "alignment")?),
                    None => (rest, 0),
                };
                Op::Alloca {
                    ty: parse_type(ty)?,
                    space,
                    align,
                }
            }
            "barrier" if rest.is_empty() => Op::Barrier,
            "load" => {
                let (ty, p) = rest
                    .rsplit_once(' ')
                    .ok_or("`load` needs `<type> <pointer>`")?;
                Op::Load(self.value(p)?, parse_type(ty)?)
            }
            "store" => {
                let [value, ptr] = self.operands(rest)?;
                Op::Store { ptr, value }
            }
            "atomic.add" => {
                let [value, ptr] = self.operands(rest)?;
                Op::AtomicAdd { ptr, value }
            }
            "gep" => {
                let (operands, elem) = rest
                    .split_once(" : ")
                    .ok_or("`gep` needs `<pointer>, <index> : <type>`")?;
                let [ptr, index] = self.operands(operands)?;
                Op::Gep {
                    ptr,
                    index,
                    elem: parse_type(elem)?,
                }
            }
            "call" => {
                let (name, args) = split_call(rest)?;
                Op::Call {
                    func: interner.intern(name),
                    args: self.values(args)?,
                }
            }
            "func_addr" if !rest.is_empty() => Op::FuncAddr(interner.intern(rest)),
            "global_addr" if !rest.is_empty() => Op::GlobalAddr(interner.intern(rest)),
            "splat" => Op::Splat(self.operands::<1>(rest)?[0]),
            "extractlane" => {
                let (v, k) = rest
                    .split_once(", ")
                    .ok_or("`extractlane` needs `<vector>, <lane>`")?;
                Op::ExtractLane(self.value(v)?, parse_num(k, "lane")?)
            }
            "insertlane" => {
                let (vs, k) = rest
                    .rsplit_once(", ")
                    .ok_or("`insertlane` needs `<vector>, <scalar>, <lane>`")?;
                let [v, s] = self.operands(vs)?;
                Op::InsertLane(v, s, parse_num(k, "lane")?)
            }
            "iota" => Op::Iota(parse_type(rest)?),
            "shuffle" => {
                let (ab, mask) = rest
                    .split_once(", [")
                    .ok_or("`shuffle` needs `<a>, <b>, [<lanes>]`")?;
                let mask = mask.strip_suffix(']').ok_or("unclosed `shuffle` mask")?;
                let [a, b] = self.operands(ab)?;
                let mask = split_top(mask)
                    .into_iter()
                    .map(|m| parse_num::<u8>(m, "mask lane"))
                    .collect::<Result<Vec<_>, _>>()?;
                Op::Shuffle { a, b, mask }
            }
            "fma" => {
                let [a, b, c] = self.operands(rest)?;
                Op::Fma(a, b, c)
            }
            "sqrt" => Op::Sqrt(self.operands::<1>(rest)?[0]),
            "veckernel" => return Err(VECKERNEL_UNREPRESENTABLE.to_string()),
            _ => return Err(format!("unknown operation `{word}`")),
        })
    }

    /// Exactly `N` comma-separated value operands.
    fn operands<const N: usize>(&mut self, s: &str) -> Result<[ValueId; N], String> {
        let values = self.values(s)?;
        let n = values.len();
        values
            .try_into()
            .map_err(|_| format!("expected {N} operand(s), found {n} in `{s}`"))
    }

    fn target(&mut self, s: &str) -> Result<(BlockId, Vec<ValueId>), String> {
        let s = s.trim();
        match s.split_once('(') {
            Some((label, args)) => {
                let args = args
                    .strip_suffix(')')
                    .ok_or_else(|| format!("unclosed argument list in `{s}`"))?;
                Ok((parse_block(label)?, self.values(args)?))
            }
            None => Ok((parse_block(s)?, Vec::new())),
        }
    }

    fn terminator(&mut self, text: &str) -> Result<Terminator, String> {
        let (word, rest) = text.split_once(' ').unwrap_or((text, ""));
        Ok(match word {
            "ret" if rest.is_empty() => Terminator::Ret(None),
            "ret" => Terminator::Ret(Some(self.value(rest)?)),
            "unreachable" if rest.is_empty() => Terminator::Unreachable,
            "br" => {
                let (target, args) = self.target(rest)?;
                Terminator::Br { target, args }
            }
            "cond_br" => {
                let parts = split_top(rest);
                let [cond, then_part, else_part] = parts[..] else {
                    return Err("`cond_br` needs `<cond>, <then>, <else>`".to_string());
                };
                let cond = self.value(cond)?;
                let (then_blk, then_args) = self.target(then_part)?;
                let (else_blk, else_args) = self.target(else_part)?;
                Terminator::CondBr {
                    cond,
                    then_blk,
                    then_args,
                    else_blk,
                    else_args,
                }
            }
            "switch" => {
                let (scrutinee, rest) = rest
                    .split_once(", ")
                    .ok_or("`switch` needs `<value>, <default> [<cases>]`")?;
                let (default, cases) = rest
                    .split_once(" [")
                    .ok_or("`switch` needs a `[...]` case list")?;
                let cases = cases.strip_suffix(']').ok_or("unclosed `switch` cases")?;
                let scrutinee = self.value(scrutinee)?;
                let (default, default_args) = self.target(default)?;
                let mut arms = Vec::new();
                for case in split_top(cases) {
                    let (value, target) = case
                        .split_once(": ")
                        .ok_or_else(|| format!("switch case `{case}` needs `<value>: <block>`"))?;
                    let value = value
                        .trim()
                        .parse::<i128>()
                        .map_err(|_| format!("bad switch case value `{value}`"))?;
                    let (target, args) = self.target(target)?;
                    arms.push(SwitchCase {
                        value,
                        target,
                        args,
                    });
                }
                Terminator::Switch {
                    scrutinee,
                    cases: arms,
                    default,
                    default_args,
                }
            }
            _ => return Err(format!("malformed terminator `{text}`")),
        })
    }

    fn finish(self) -> Result<Function, String> {
        if let Some((id, ..)) = self.open {
            return Err(format!("bb{} has no terminator", id.0));
        }
        let len = self
            .max_value
            .map_or(0, |m| m as usize + 1)
            .max(self.types.len())
            .max(self.arena.unwrap_or(0));
        let mut value_types: Vec<MirType> = self
            .types
            .into_iter()
            .map(|t| t.unwrap_or(MirType::Void))
            .collect();
        value_types.resize(len, MirType::Void);
        Ok(Function {
            name: self.name,
            params: self.params,
            ret: self.ret,
            blocks: self.blocks,
            value_types,
            entry: self.entry.unwrap_or(BlockId(0)),
            vec_kernels: Vec::new(),
            param_attrs: self.param_attrs,
        })
    }
}

enum ResultForm<'a> {
    /// `vN: <type> = <rhs>`.
    Typed(&'a str, &'a str, &'a str),
    /// `vN = <rhs>`: a result without its type.
    Untyped(&'a str),
    /// No result.
    Void,
}

fn split_result(text: &str) -> ResultForm<'_> {
    let digits = text
        .strip_prefix('v')
        .map_or(0, |r| r.bytes().take_while(u8::is_ascii_digit).count());
    if digits == 0 {
        return ResultForm::Void;
    }
    let (v, after) = text.split_at(1 + digits);
    if let Some(rest) = after.strip_prefix(": ") {
        if let Some((ty, rhs)) = rest.split_once(" = ") {
            return ResultForm::Typed(v, ty, rhs);
        }
    }
    if after.starts_with(" = ") {
        return ResultForm::Untyped(v);
    }
    ResultForm::Void
}

fn is_block_header(text: &str) -> bool {
    text.ends_with(':')
        && text
            .strip_prefix("bb")
            .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
}

/// `NAME(args)`, split at the LAST `(`: a generic-instance name such as `f$[(i32, i64)]` may
/// hold its own parentheses, but an argument list never does.
fn split_call(s: &str) -> Result<(&str, &str), String> {
    let open = s
        .rfind('(')
        .ok_or_else(|| format!("expected `name(...)`, found `{s}`"))?;
    let args = s[open + 1..]
        .strip_suffix(')')
        .ok_or_else(|| format!("unclosed argument list in `{s}`"))?;
    Ok((&s[..open], args))
}

/// Split at the commas outside any `()`, `[]` or `<>`, trimming each piece; an empty or blank
/// list is no pieces.
fn split_top(s: &str) -> Vec<&str> {
    if s.trim().is_empty() {
        return Vec::new();
    }
    let mut pieces = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            ',' if depth == 0 => {
                pieces.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    pieces.push(s[start..].trim());
    pieces
}

fn parse_value(s: &str) -> Result<ValueId, String> {
    let s = s.trim();
    s.strip_prefix('v')
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|d| d.parse::<u32>().ok())
        .map(ValueId)
        .ok_or_else(|| format!("expected a value `vN`, found `{s}`"))
}

fn parse_block(s: &str) -> Result<BlockId, String> {
    let s = s.trim();
    s.strip_prefix("bb")
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|d| d.parse::<u32>().ok())
        .map(BlockId)
        .ok_or_else(|| format!("expected a block `bbN`, found `{s}`"))
}

fn parse_num<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
    s.trim()
        .parse::<T>()
        .map_err(|_| format!("bad {what} `{}`", s.trim()))
}

/// A MIR type in its `display` spelling: a scalar, `ptr`, `void`, `<N x T>` or `[N x T]`.
pub fn parse_type(s: &str) -> Result<MirType, String> {
    let s = s.trim();
    let scalar = match s {
        "i1" => Some(MirType::I1),
        "i8" => Some(MirType::I8),
        "i16" => Some(MirType::I16),
        "i32" => Some(MirType::I32),
        "i64" => Some(MirType::I64),
        "f16" => Some(MirType::F16),
        "bf16" => Some(MirType::BF16),
        "f32" => Some(MirType::F32),
        "f64" => Some(MirType::F64),
        "ptr" => Some(MirType::Ptr),
        "void" => Some(MirType::Void),
        _ => None,
    };
    if let Some(t) = scalar {
        return Ok(t);
    }
    let aggregate = |inner: &str| -> Result<(u32, MirType), String> {
        let (n, elem) = inner
            .split_once(" x ")
            .ok_or_else(|| format!("unknown type `{s}`"))?;
        Ok((parse_num(n, "element count")?, parse_type(elem)?))
    };
    if let Some(inner) = s.strip_prefix('<').and_then(|r| r.strip_suffix('>')) {
        let (n, elem) = aggregate(inner)?;
        return Ok(MirType::Vec(Box::new(elem), n));
    }
    if let Some(inner) = s.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        let (n, elem) = aggregate(inner)?;
        return Ok(MirType::Array(Box::new(elem), n));
    }
    Err(format!("unknown type `{s}`"))
}

/// Every [`BinOp`], for reading a mnemonic back (the tests hold the table complete).
const BIN_OPS: [BinOp; 18] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::SDiv,
    BinOp::UDiv,
    BinOp::SRem,
    BinOp::URem,
    BinOp::FAdd,
    BinOp::FSub,
    BinOp::FMul,
    BinOp::FDiv,
    BinOp::FRem,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::LShr,
    BinOp::AShr,
];

const CMP_OPS: [CmpOp; 16] = [
    CmpOp::Eq,
    CmpOp::Ne,
    CmpOp::Slt,
    CmpOp::Sle,
    CmpOp::Sgt,
    CmpOp::Sge,
    CmpOp::Ult,
    CmpOp::Ule,
    CmpOp::Ugt,
    CmpOp::Uge,
    CmpOp::Foeq,
    CmpOp::Fone,
    CmpOp::Folt,
    CmpOp::Fole,
    CmpOp::Fogt,
    CmpOp::Foge,
];

const CAST_KINDS: [CastKind; 12] = [
    CastKind::SExt,
    CastKind::ZExt,
    CastKind::Trunc,
    CastKind::FpToSi,
    CastKind::FpToUi,
    CastKind::SiToFp,
    CastKind::UiToFp,
    CastKind::FpExt,
    CastKind::FpTrunc,
    CastKind::Bitcast,
    CastKind::IntToPtr,
    CastKind::PtrToInt,
];

const INT_UN_OPS: [IntUnOp; 4] = [IntUnOp::Popcnt, IntUnOp::Clz, IntUnOp::Ctz, IntUnOp::Bswap];

const NARROW_KINDS: [NarrowKind; 4] = [
    NarrowKind::Trunc,
    NarrowKind::SatSigned,
    NarrowKind::SatUnsigned,
    NarrowKind::FpTrunc,
];

const ROUND_MODES: [RoundMode; 4] = [
    RoundMode::Nearest,
    RoundMode::Floor,
    RoundMode::Ceil,
    RoundMode::Trunc,
];

/// `print`'s spelling of a rounding mode (`round.<mode>`).
fn round_name(m: RoundMode) -> &'static str {
    match m {
        RoundMode::Nearest => "nearest",
        RoundMode::Floor => "floor",
        RoundMode::Ceil => "ceil",
        RoundMode::Trunc => "trunc",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::print::print_program;
    use crate::verify::verify_program;
    use crate::{AddrSpace, Builder, VecKernel, VecLane};
    use std::collections::BTreeSet;

    /// The slot of every `Op` variant. Exhaustive on purpose: a new variant does not compile here
    /// until it has a slot, and `every_op_program` must then build it (or the coverage assert
    /// below fails), which forces the variant into this format's printer and parser.
    fn op_slot(op: &Op) -> usize {
        match op {
            Op::ConstInt(..) => 0,
            Op::ConstFloat(..) => 1,
            Op::Bin(..) => 2,
            Op::Cmp(..) => 3,
            Op::Neg(_) => 4,
            Op::Not(_) => 5,
            Op::Cast(..) => 6,
            Op::Select(..) => 7,
            Op::Alloca { .. } => 8,
            Op::Barrier => 9,
            Op::Load(..) => 10,
            Op::Store { .. } => 11,
            Op::AtomicAdd { .. } => 12,
            Op::Gep { .. } => 13,
            Op::Call { .. } => 14,
            Op::FuncAddr(_) => 15,
            Op::GlobalAddr(_) => 16,
            Op::Splat(_) => 17,
            Op::ExtractLane(..) => 18,
            Op::Iota(_) => 19,
            Op::Widen { .. } => 20,
            Op::Narrow { .. } => 21,
            Op::Shuffle { .. } => 22,
            Op::InsertLane(..) => 23,
            Op::Fma(..) => 24,
            Op::Sqrt(_) => 25,
            Op::Round(..) => 26,
            Op::IntUnary(..) => 27,
            // No textual form: `veckernel_is_a_parse_error` covers it.
            Op::VecKernelCall { .. } => VECKERNEL_SLOT,
        }
    }
    const VECKERNEL_SLOT: usize = 28;
    const OP_SLOTS: usize = 29;

    fn term_slot(t: &Terminator) -> usize {
        match t {
            Terminator::Ret(None) => 0,
            Terminator::Ret(Some(_)) => 1,
            Terminator::Br { .. } => 2,
            Terminator::CondBr { .. } => 3,
            Terminator::Switch { .. } => 4,
            Terminator::Unreachable => 5,
        }
    }

    // The parser's name tables, held complete by exhaustive matches.
    fn bin_slot(b: BinOp) -> usize {
        use BinOp::*;
        match b {
            Add => 0,
            Sub => 1,
            Mul => 2,
            SDiv => 3,
            UDiv => 4,
            SRem => 5,
            URem => 6,
            FAdd => 7,
            FSub => 8,
            FMul => 9,
            FDiv => 10,
            FRem => 11,
            And => 12,
            Or => 13,
            Xor => 14,
            Shl => 15,
            LShr => 16,
            AShr => 17,
        }
    }
    fn cmp_slot(c: CmpOp) -> usize {
        use CmpOp::*;
        match c {
            Eq => 0,
            Ne => 1,
            Slt => 2,
            Sle => 3,
            Sgt => 4,
            Sge => 5,
            Ult => 6,
            Ule => 7,
            Ugt => 8,
            Uge => 9,
            Foeq => 10,
            Fone => 11,
            Folt => 12,
            Fole => 13,
            Fogt => 14,
            Foge => 15,
        }
    }
    fn cast_slot(k: CastKind) -> usize {
        use CastKind::*;
        match k {
            SExt => 0,
            ZExt => 1,
            Trunc => 2,
            FpToSi => 3,
            FpToUi => 4,
            SiToFp => 5,
            UiToFp => 6,
            FpExt => 7,
            FpTrunc => 8,
            Bitcast => 9,
            IntToPtr => 10,
            PtrToInt => 11,
        }
    }
    fn int_un_slot(u: IntUnOp) -> usize {
        match u {
            IntUnOp::Popcnt => 0,
            IntUnOp::Clz => 1,
            IntUnOp::Ctz => 2,
            IntUnOp::Bswap => 3,
        }
    }
    fn narrow_slot(k: NarrowKind) -> usize {
        match k {
            NarrowKind::Trunc => 0,
            NarrowKind::SatSigned => 1,
            NarrowKind::SatUnsigned => 2,
            NarrowKind::FpTrunc => 3,
        }
    }
    fn round_slot(m: RoundMode) -> usize {
        match m {
            RoundMode::Nearest => 0,
            RoundMode::Floor => 1,
            RoundMode::Ceil => 2,
            RoundMode::Trunc => 3,
        }
    }

    #[test]
    fn name_tables_list_every_variant_once() {
        for (i, b) in BIN_OPS.iter().enumerate() {
            assert_eq!(bin_slot(*b), i);
        }
        for (i, c) in CMP_OPS.iter().enumerate() {
            assert_eq!(cmp_slot(*c), i);
        }
        for (i, k) in CAST_KINDS.iter().enumerate() {
            assert_eq!(cast_slot(*k), i);
        }
        for (i, u) in INT_UN_OPS.iter().enumerate() {
            assert_eq!(int_un_slot(*u), i);
        }
        for (i, k) in NARROW_KINDS.iter().enumerate() {
            assert_eq!(narrow_slot(*k), i);
        }
        for (i, m) in ROUND_MODES.iter().enumerate() {
            assert_eq!(round_slot(*m), i);
        }
    }

    fn vec(t: MirType, n: u32) -> MirType {
        MirType::Vec(Box::new(t), n)
    }

    fn arr(t: MirType, n: u32) -> MirType {
        MirType::Array(Box::new(t), n)
    }

    /// A verifying program with every `Op` variant but `VecKernelCall`, every operator kind,
    /// every terminator, every type constructor, a signed extern, statics, NaN-payload / `-0.0`
    /// / `i64::MIN` constants, parameter promises, names with spaces and parentheses, a gap in
    /// the value arena and an entry block that is not `bb0`.
    fn every_op_program(i: &mut Interner) -> Program {
        use MirType::*;
        let mut p = Program::new();
        let c_abs = i.intern("c_abs");
        p.externs.push(ExternFn {
            name: c_abs,
            params: vec![
                ExternParam {
                    ty: I32,
                    signed: true,
                },
                ExternParam {
                    ty: Ptr,
                    signed: false,
                },
            ],
            ret: ExternParam {
                ty: I32,
                signed: true,
            },
        });
        let c_void = i.intern("c_void");
        p.externs.push(ExternFn {
            name: c_void,
            params: vec![],
            ret: ExternParam {
                ty: Void,
                signed: false,
            },
        });
        let rodata = i.intern("wukong$rodata$0");
        p.statics.push(StaticData {
            name: rodata,
            bytes: b"hi\n\0".to_vec(),
        });
        p.statics.push(StaticData {
            name: i.intern("wukong$rodata$1"),
            bytes: vec![],
        });

        // A generic instance's name holds a space and a `;`.
        let generic = i.intern("f$[f32; 4]");
        let mut g = Builder::new(generic, F32);
        let a = g.add_param(F32);
        g.ret(Some(a));
        p.funcs.push(g.finish());

        // ... and may hold parentheses; this one returns nothing.
        let sink = i.intern("sink$[(i32, i64)]");
        let mut s = Builder::new(sink, Void);
        s.add_param(I32);
        s.ret(None);
        p.funcs.push(s.finish());

        let mut b = Builder::new(i.intern("every_op"), I32);
        let buf = b.add_param(Ptr);
        b.set_param_attrs(
            buf,
            ParamAttrs {
                noalias: true,
                slice: true,
                extent: Some(ParamExtent {
                    elems: 64,
                    elem_bytes: 4,
                }),
            },
        );
        let x = b.add_param(I32);
        let q = b.add_param(Ptr);
        b.set_param_attrs(
            q,
            ParamAttrs {
                noalias: true,
                ..ParamAttrs::default()
            },
        );
        let fv = b.add_param(F32);
        let d = b.add_param(F64);

        let c_i1 = b.build(I1, Op::ConstInt(1, I1));
        let c_i8 = b.build(I8, Op::ConstInt(-128, I8));
        let c_i16 = b.build(I16, Op::ConstInt(300, I16));
        let c_i32 = b.build(I32, Op::ConstInt(7, I32));
        let c_min = b.build(I64, Op::ConstInt(i128::from(i64::MIN), I64));
        b.build(I64, Op::ConstInt(i128::from(i64::MAX), I64));
        // A hole in the arena: allocated, never defined.
        b.new_value(F64);
        b.build(
            F64,
            Op::ConstFloat(f64::from_bits(0x7ff8_0000_dead_beef), F64),
        );
        b.build(F32, Op::ConstFloat(-0.0, F32));
        let h = b.build(F16, Op::ConstFloat(0.5, F16));
        b.build(BF16, Op::ConstFloat(-2.0, BF16));

        for op in BIN_OPS {
            if op.is_float() {
                b.build(F32, Op::Bin(op, fv, fv));
            } else {
                b.build(I32, Op::Bin(op, x, c_i32));
            }
        }
        let vi = b.build(vec(I32, 4), Op::Splat(c_i32));
        b.build(vec(I32, 4), Op::Bin(BinOp::Shl, vi, c_i32));
        for c in CMP_OPS {
            if c.is_float() {
                b.build(I1, Op::Cmp(c, fv, fv));
            } else {
                b.build(I1, Op::Cmp(c, x, c_i32));
            }
        }
        let vf = b.build(vec(F32, 4), Op::Splat(fv));
        let mask = b.build(vec(I32, 4), Op::Cmp(CmpOp::Folt, vf, vf));
        b.build(I32, Op::Neg(x));
        b.build(F32, Op::Neg(fv));
        b.build(I1, Op::Not(c_i1));

        let p64 = b.build(I64, Op::Cast(CastKind::PtrToInt, buf, I64));
        for (k, v, to) in [
            (CastKind::SExt, c_i8, I32),
            (CastKind::ZExt, c_i16, I64),
            (CastKind::Trunc, c_min, I32),
            (CastKind::FpToSi, fv, I32),
            (CastKind::FpToUi, d, I64),
            (CastKind::SiToFp, x, F64),
            (CastKind::UiToFp, c_min, F32),
            (CastKind::FpExt, h, F32),
            (CastKind::FpTrunc, fv, BF16),
            (CastKind::Bitcast, x, F32),
            (CastKind::IntToPtr, p64, Ptr),
        ] {
            b.build(to.clone(), Op::Cast(k, v, to));
        }
        b.build(vec(F32, 4), Op::Cast(CastKind::SiToFp, vi, vec(F32, 4)));
        b.build(I32, Op::Select(c_i1, x, c_i32));
        b.build(vec(F32, 4), Op::Select(mask, vf, vf));

        b.alloca(arr(I8, 16));
        b.alloca_aligned(arr(F32, 4), AddrSpace::Shared, 16);
        let slot = b.alloca_aligned(I64, AddrSpace::Local, 64);
        let grid = b.alloca(arr(arr(F32, 3), 2));
        b.alloca(arr(vec(F32, 4), 2));
        b.barrier();
        b.build_void(Op::Store {
            ptr: slot,
            value: c_min,
        });
        b.atomic_add(slot, c_min);
        b.build(I64, Op::Load(slot, I64));
        b.build(vec(F32, 4), Op::Load(buf, vec(F32, 4)));
        b.build(
            Ptr,
            Op::Gep {
                ptr: buf,
                index: x,
                elem: F32,
            },
        );
        b.build(
            Ptr,
            Op::Gep {
                ptr: grid,
                index: c_min,
                elem: arr(F32, 3),
            },
        );
        b.build(
            F32,
            Op::Call {
                func: generic,
                args: vec![fv],
            },
        );
        b.build_void(Op::Call {
            func: sink,
            args: vec![x],
        });
        b.build(
            I32,
            Op::Call {
                func: c_abs,
                args: vec![x, q],
            },
        );
        b.build_void(Op::Call {
            func: c_void,
            args: vec![],
        });
        b.build(Ptr, Op::FuncAddr(generic));
        b.build(Ptr, Op::GlobalAddr(rodata));

        b.build(F32, Op::ExtractLane(vf, 3));
        b.build(vec(I32, 4), Op::Iota(vec(I32, 4)));
        let v16 = b.build(vec(I16, 8), Op::Splat(c_i16));
        b.build(
            vec(I32, 4),
            Op::Widen {
                v: v16,
                half: Half::Lo,
                kind: CastKind::SExt,
            },
        );
        b.build(
            vec(I32, 4),
            Op::Widen {
                v: v16,
                half: Half::Hi,
                kind: CastKind::ZExt,
            },
        );
        let vd = b.build(
            vec(F64, 2),
            Op::Widen {
                v: vf,
                half: Half::Lo,
                kind: CastKind::FpExt,
            },
        );
        let v64 = b.build(vec(I64, 2), Op::Splat(c_min));
        for (kind, src, to) in [
            (NarrowKind::Trunc, v64, vec(I32, 4)),
            (NarrowKind::SatSigned, vi, vec(I16, 8)),
            (NarrowKind::SatUnsigned, v16, vec(I8, 16)),
            (NarrowKind::FpTrunc, vd, vec(F32, 4)),
        ] {
            b.build(
                to,
                Op::Narrow {
                    lo: src,
                    hi: src,
                    kind,
                },
            );
        }
        b.build(
            vec(F32, 4),
            Op::Shuffle {
                a: vf,
                b: vf,
                mask: vec![0, 5, 2, 7],
            },
        );
        b.build(vec(F32, 4), Op::InsertLane(vf, fv, 1));
        b.build(F32, Op::Fma(fv, fv, fv));
        b.build(F64, Op::Sqrt(d));
        for m in ROUND_MODES {
            b.build(F32, Op::Round(m, fv));
        }
        let c64 = b.build(I64, Op::ConstInt(5, I64));
        let c8 = b.build(I8, Op::ConstInt(5, I8));
        let c16 = b.build(I16, Op::ConstInt(5, I16));
        b.build(I32, Op::IntUnary(IntUnOp::Popcnt, x));
        b.build(I64, Op::IntUnary(IntUnOp::Clz, c64));
        b.build(I8, Op::IntUnary(IntUnOp::Ctz, c8));
        b.build(I16, Op::IntUnary(IntUnOp::Bswap, c16));

        let bb1 = b.new_block();
        let bb2 = b.new_block();
        let bb3 = b.new_block();
        let bb4 = b.new_block();
        b.cond_br(c_i1, bb1, vec![x], bb2, vec![]);
        b.switch_to(bb1);
        let y = b.block_param(bb1, I32);
        b.switch(
            y,
            vec![
                SwitchCase {
                    value: 0,
                    target: bb3,
                    args: vec![c_i32],
                },
                SwitchCase {
                    value: -1,
                    target: bb2,
                    args: vec![],
                },
                SwitchCase {
                    value: 7,
                    target: bb2,
                    args: vec![],
                },
            ],
            bb3,
            vec![y],
        );
        b.switch_to(bb2);
        b.br(bb3, vec![x]);
        b.switch_to(bb3);
        let z = b.block_param(bb3, I32);
        b.ret(Some(z));
        b.switch_to(bb4);
        b.set_term(Terminator::Unreachable);
        p.funcs.push(b.finish());

        // An entry block that is not `bb0`: the function starts at bb1, and bb0 is unreachable.
        p.funcs.push(Function {
            name: i.intern("late_entry"),
            params: vec![ValueId(0)],
            ret: I64,
            blocks: vec![
                BasicBlock {
                    id: BlockId(0),
                    params: vec![],
                    insts: vec![],
                    term: Terminator::Unreachable,
                },
                BasicBlock {
                    id: BlockId(1),
                    params: vec![ValueId(0)],
                    insts: vec![],
                    term: Terminator::Ret(Some(ValueId(0))),
                },
            ],
            // v1 and v2 exist but nothing defines them: the arena outlives the last value.
            value_types: vec![I64, F32, I64],
            entry: BlockId(1),
            vec_kernels: vec![],
            param_attrs: vec![],
        });
        p
    }

    #[test]
    fn every_op_round_trips_through_the_typed_text() {
        let mut interner = Interner::new();
        let p = every_op_program(&mut interner);
        assert_eq!(
            verify_program(&p),
            Vec::<String>::new(),
            "the fixture must verify"
        );

        // The fixture covers what it claims.
        let insts = || {
            p.funcs
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.insts)
        };
        let ops: BTreeSet<usize> = insts().map(|i| op_slot(&i.op)).collect();
        let want: BTreeSet<usize> = (0..OP_SLOTS).filter(|&s| s != VECKERNEL_SLOT).collect();
        assert_eq!(ops, want, "every Op variant but VecKernelCall");
        let terms: BTreeSet<usize> = p
            .funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .map(|b| term_slot(&b.term))
            .collect();
        assert_eq!(terms.len(), 6, "every terminator");
        let mut kinds = [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()];
        let mut small = [BTreeSet::new(), BTreeSet::new(), BTreeSet::new()];
        for i in insts() {
            match &i.op {
                Op::Bin(b, ..) => kinds[0].insert(bin_slot(*b)),
                Op::Cmp(c, ..) => kinds[1].insert(cmp_slot(*c)),
                Op::Cast(k, ..) => kinds[2].insert(cast_slot(*k)),
                Op::IntUnary(u, _) => small[0].insert(int_un_slot(*u)),
                Op::Narrow { kind, .. } => small[1].insert(narrow_slot(*kind)),
                Op::Round(m, _) => small[2].insert(round_slot(*m)),
                _ => false,
            };
        }
        assert_eq!(kinds.map(|k| k.len()), [18, 16, 12]);
        assert_eq!(small.map(|k| k.len()), [4, 4, 4]);

        let typed = print_program_typed(&p, &interner);
        for want in [
            "; wukong mir text v1\n",
            "declare extern \"C\" fn c_abs(i32 signed, ptr) -> i32 signed\n",
            "static wukong$rodata$0 = hex 68690a00\n",
            "static wukong$rodata$1 = hex\n",
            "fn every_op(v0 ptr noalias slice extent=64x4, v1 i32, v2 ptr noalias, v3 f32, v4 f64) -> i32 {",
            "= const.f64 0x7ff80000deadbeef\n",
            "= const.f32 0x8000000000000000\n",
            "= const.i64 -9223372036854775808\n",
            "= alloca.shared [4 x f32] align 16\n",
            "= alloca i64 align 64\n",
            "atomic.add v",
            "= call f$[f32; 4](v3)\n",
            "    call sink$[(i32, i64)](v1)\n",
            "  entry bb1\n  values 3\n",
        ] {
            assert!(typed.contains(want), "missing `{want}` in\n{typed}");
        }

        let mut again = Interner::new();
        let parsed = parse_program(&typed, &mut again).unwrap_or_else(|e| panic!("{e}\n{typed}"));
        assert_eq!(print_program_typed(&parsed, &again), typed);
        assert_eq!(print_program(&parsed, &again), print_program(&p, &interner));
        assert_eq!(verify_program(&parsed), Vec::<String>::new());
        // What the untyped dump cannot show survives too.
        assert_eq!(
            parsed.externs,
            p.externs
                .iter()
                .map(|e| ExternFn {
                    name: again.get(interner.resolve(e.name)).unwrap(),
                    ..e.clone()
                })
                .collect::<Vec<_>>()
        );
        let nan = parsed.funcs[2].blocks[0]
            .insts
            .iter()
            .find_map(|i| match i.op {
                Op::ConstFloat(v, MirType::F64) => Some(v.to_bits()),
                _ => None,
            });
        assert_eq!(nan, Some(0x7ff8_0000_dead_beef));
        assert_eq!(parsed.funcs[2].param_attr(0), p.funcs[2].param_attr(0));
        assert_eq!(parsed.funcs[3].entry, BlockId(1));
        // The arena keeps its length, so values a pass creates are numbered as in the original.
        for (a, b) in parsed.funcs.iter().zip(&p.funcs) {
            assert_eq!(a.value_types.len(), b.value_types.len());
        }
    }

    /// Parse `body` as the only instruction of a one-block function and return the error.
    fn error_in(line: &str) -> ParseError {
        let text =
            format!("{HEADER}\nfn f(v0 i32) -> i32 {{\n  bb0(v0 i32):\n{line}\n    ret v0\n}}\n");
        parse_program(&text, &mut Interner::new()).expect_err(line)
    }

    #[test]
    fn errors_name_their_line() {
        let e = error_in("    v1: i32 = frobnicate v0");
        assert_eq!(
            (e.line, e.msg.as_str()),
            (4, "unknown operation `frobnicate`")
        );
        let e = error_in("    v1 = add v0, v0");
        assert_eq!(e.line, 4);
        assert!(e.msg.contains("no type annotation"), "{e}");
        let e = error_in("    v1: i33 = add v0, v0");
        assert_eq!((e.line, e.msg.as_str()), (4, "unknown type `i33`"));
        let e = error_in("    veckernel #0(ptrs=v0, scalars=v0, n=v0)");
        assert_eq!((e.line, e.msg.as_str()), (4, VECKERNEL_UNREPRESENTABLE));
        assert_eq!(
            e.to_string(),
            format!("line 4: {VECKERNEL_UNREPRESENTABLE}")
        );
        let e = error_in("    v0: i32 = const.i32 1");
        assert_eq!((e.line, e.msg.as_str()), (4, "v0 is defined twice"));
        let e = error_in("    v1: i32 = add v0");
        assert!(e.msg.contains("expected 2 operand(s)"), "{e}");
        let e = error_in("    v4000000000: i32 = add v0, v0");
        assert_eq!(e.line, 4);
        assert!(e.msg.contains("past the largest value id"), "{e}");

        let blocks = format!("{HEADER}\nfn f() -> void {{\n  bb1:\n    ret\n}}\n");
        let e = parse_program(&blocks, &mut Interner::new()).unwrap_err();
        assert_eq!(e.line, 3);
        assert!(e.msg.contains("expected bb0, found bb1"), "{e}");
        let open = format!("{HEADER}\nfn f() -> void {{\n  bb0:\n    ret\n");
        assert!(parse_program(&open, &mut Interner::new())
            .unwrap_err()
            .msg
            .contains("not closed"));
        let stray = "fn f() -> void {\n  bb0:\n    v1: i32 = const.i32 1\n  bb1:\n    ret\n}\n";
        let e = parse_program(stray, &mut Interner::new()).unwrap_err();
        assert_eq!((e.line, e.msg.as_str()), (4, "bb0 has no terminator"));
    }

    /// The plain `--emit=mir` form is not the typed form; the error says which flag gives it.
    #[test]
    fn the_untyped_dump_is_refused_with_a_pointer_to_the_typed_one() {
        let mut interner = Interner::new();
        let mut b = Builder::new(interner.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, x));
        b.ret(Some(y));
        let mut p = Program::new();
        p.funcs.push(b.finish());
        let e = parse_program(&print_program(&p, &interner), &mut Interner::new()).unwrap_err();
        assert_eq!(e.line, 3);
        assert!(e.msg.contains("--emit=mir-text"), "{e}");
    }

    /// A recipe has no textual form: the dump says so above the function, and reading the
    /// call back is the documented error at the call's line.
    #[test]
    fn veckernel_is_a_parse_error() {
        let mut interner = Interner::new();
        let mut b = Builder::new(interner.intern("k"), MirType::Void);
        let ptrs = b.add_param(MirType::Ptr);
        let n = b.add_param(MirType::I64);
        let kernel = b.add_vec_kernel(VecKernel {
            name: interner.intern("k$vk0"),
            lane: VecLane::F32,
            streams: 0,
            scalars: 0,
            ops: vec![],
            unroll: 1,
            reduce: None,
        });
        b.build_void(Op::VecKernelCall {
            kernel,
            ptrs,
            scalars: ptrs,
            n,
        });
        b.ret(None);
        let mut p = Program::new();
        p.funcs.push(b.finish());
        let typed = print_program_typed(&p, &interner);
        assert!(
            typed.contains("; 1 vector-kernel recipe(s) are not representable\nfn k("),
            "{typed}"
        );
        let e = parse_program(&typed, &mut Interner::new()).unwrap_err();
        assert_eq!(e.msg, VECKERNEL_UNREPRESENTABLE);
        let at = typed
            .lines()
            .position(|l| l.contains("veckernel #0"))
            .unwrap();
        assert_eq!(e.line, at + 1);
    }

    #[test]
    fn types_parse_in_their_display_spelling() {
        for t in [
            MirType::I1,
            MirType::BF16,
            MirType::Void,
            vec(MirType::F32, 4),
            arr(arr(MirType::I8, 3), 2),
            arr(vec(MirType::F64, 2), 5),
        ] {
            assert_eq!(parse_type(&t.display()), Ok(t));
        }
        assert!(parse_type("<4 x>").is_err());
        assert!(parse_type("[x f32]").is_err());
    }
}
