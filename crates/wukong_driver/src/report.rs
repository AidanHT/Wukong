//! Compile-time self-reporting (`--time-passes`, `--time-trace=<file>`) and Make-syntax dependency
//! files (`--depfile=<file>`).
//!
//! Both exist so a user can answer questions about their own build without the bench harness: which
//! stage of *this* compile is slow (the optimizer fixpoint? Cranelift? the link?), and which files a
//! build system must watch to know when to recompile.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use wukong_mir::Program;
use wukong_span::{Interner, SourceId, SourceMap};

use crate::{EmitStage, ErrorFormat, Options};

/// One timed interval, relative to the start of the compile.
struct Interval {
    name: String,
    /// `0`: a pipeline stage (real wall-clock intervals, back to back). `1`: an optimizer pass's
    /// time *accumulated* over every function and fixpoint sweep, laid out back to back from the
    /// start of the optimize stage — a breakdown of that stage, not a timeline.
    lane: u32,
    start: Duration,
    dur: Duration,
    calls: Option<u64>,
}

/// Wall-clock timer for the pipeline stages of one compile. Inert unless `--time-passes` or
/// `--time-trace` asked for it: every method returns immediately, so the default compile pays one
/// branch per stage boundary and no clock reads. It reports when dropped, so every exit path of
/// [`crate::compile`] — success, a diagnostic, an I/O error — prints what it measured.
pub(crate) struct PassTimer {
    enabled: bool,
    print: bool,
    /// Under `--error-format=json` stderr is a machine-read stream of one JSON object per line, so
    /// the `--time-passes` report is one more such line instead of a human table.
    json: bool,
    trace: Option<PathBuf>,
    header: String,
    origin: Instant,
    open: Option<(&'static str, Instant)>,
    intervals: Vec<Interval>,
    /// Breakdown intervals of the running stage, listed after it once it closes.
    pending: Vec<Interval>,
    notes: Vec<String>,
}

impl PassTimer {
    pub(crate) fn new(opts: &Options) -> PassTimer {
        let enabled = opts.time_passes || opts.time_trace.is_some();
        PassTimer {
            enabled,
            print: opts.time_passes,
            json: opts.error_format == ErrorFormat::Json,
            trace: opts.time_trace.clone(),
            header: if enabled {
                format!(
                    "{} (-O{}, {})",
                    opts.input.display(),
                    opts.opt_level,
                    mode_name(opts)
                )
            } else {
                String::new()
            },
            origin: Instant::now(),
            open: None,
            intervals: Vec::new(),
            pending: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Close the running stage and start `stage`.
    pub(crate) fn begin(&mut self, stage: &'static str) {
        if !self.enabled {
            return;
        }
        self.close();
        self.open = Some((stage, Instant::now()));
    }

    fn close(&mut self) {
        if let Some((name, t0)) = self.open.take() {
            self.intervals.push(Interval {
                name: name.to_string(),
                lane: 0,
                start: t0.duration_since(self.origin),
                dur: t0.elapsed(),
                calls: None,
            });
        }
        self.intervals.append(&mut self.pending);
    }

    /// Where the running stage started, relative to the compile's start.
    fn open_start(&self) -> Duration {
        self.open
            .map_or(Duration::ZERO, |(_, t0)| t0.duration_since(self.origin))
    }

    /// Lay out a breakdown of the running stage on lane 1: `(name, time, calls)` back to back.
    fn breakdown(&mut self, parts: impl IntoIterator<Item = (String, Duration, Option<u64>)>) {
        let mut at = self.open_start();
        for (name, dur, calls) in parts {
            self.pending.push(Interval {
                name,
                lane: 1,
                start: at,
                dur,
                calls,
            });
            at += dur;
        }
    }

    /// Record the optimizer's per-pass breakdown (from `wukong_opt::optimize_timed`), as part of
    /// the running `optimize` stage.
    pub(crate) fn optimizer_passes(&mut self, t: &wukong_opt::Timings) {
        if !self.enabled {
            return;
        }
        let mut parts: Vec<(String, Duration, Option<u64>)> = Vec::new();
        if !t.inline.is_zero() {
            parts.push(("inline".to_string(), t.inline, None));
        }
        for p in &t.per_pass {
            parts.push((p.name.to_string(), p.time, Some(p.calls)));
        }
        if !t.unroll.is_zero() {
            parts.push(("unroll".to_string(), t.unroll, None));
        }
        if !t.slp.is_zero() {
            parts.push(("slp".to_string(), t.slp, None));
        }
        self.breakdown(parts);
        self.notes.push(format!(
            "optimizer: at most {} fixpoint sweep(s) in any function",
            t.max_iterations
        ));
    }

    /// Record Cranelift's codegen / object-write split, as part of the running `codegen` stage.
    pub(crate) fn object_split(&mut self, t: &wukong_codegen_cranelift::ObjTimings) {
        if !self.enabled {
            return;
        }
        self.breakdown([
            ("isel+regalloc".to_string(), t.codegen, None),
            ("object-write".to_string(), t.object_write, None),
        ]);
    }

    /// Add a free-form line to the report.
    pub(crate) fn note(&mut self, line: impl FnOnce() -> String) {
        if self.enabled {
            self.notes.push(line());
        }
    }

    /// Summarize the optimized MIR: function and instruction counts and the heaviest functions,
    /// which is where a compile-time cliff (an unrolled aggregate, a giant generated body) shows.
    pub(crate) fn mir_summary(&mut self, program: &Program, interner: &Interner) {
        if !self.enabled {
            return;
        }
        let mut sizes: Vec<(usize, &str)> = program
            .funcs
            .iter()
            .map(|f| {
                (
                    f.blocks.iter().map(|b| b.insts.len()).sum(),
                    interner.resolve(f.name),
                )
            })
            .collect();
        let total: usize = sizes.iter().map(|s| s.0).sum();
        // Heaviest first; ties by name so the line is deterministic.
        sizes.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(b.1)));
        let heaviest: Vec<String> = sizes
            .iter()
            .take(5)
            .map(|(n, name)| format!("{name} {n}"))
            .collect();
        self.notes.push(format!(
            "mir: {} function(s), {total} instruction(s); heaviest: {}",
            sizes.len(),
            heaviest.join(", ")
        ));
    }

    fn render_table(&self) -> String {
        let total = self
            .intervals
            .iter()
            .filter(|i| i.lane == 0)
            .map(|i| i.dur)
            .sum::<Duration>();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let pct = |d: Duration| {
            if total.is_zero() {
                0.0
            } else {
                100.0 * d.as_secs_f64() / total.as_secs_f64()
            }
        };
        let mut out = String::new();
        let _ = writeln!(out, "time-passes: {}", self.header);
        let _ = writeln!(out, "  {:<18} {:>10} {:>7}", "stage", "ms", "%");
        for i in self.intervals.iter() {
            if i.lane == 0 {
                let _ = writeln!(
                    out,
                    "  {:<18} {:>10.3} {:>6.1}%",
                    i.name,
                    ms(i.dur),
                    pct(i.dur)
                );
            } else {
                let calls = i.calls.map(|c| format!("  calls {c}")).unwrap_or_default();
                let _ = writeln!(out, "    {:<16} {:>10.3}{calls}", i.name, ms(i.dur));
            }
        }
        let _ = writeln!(out, "  {:<18} {:>10.3}", "total", ms(total));
        for note in &self.notes {
            let _ = writeln!(out, "  {note}");
        }
        out
    }

    /// The `--time-passes` report as one JSON line: `{"time_passes":{"input":..,"events":[..],
    /// "notes":[..]}}`, the events in the same shape as the trace file's.
    fn render_json_line(&self) -> String {
        let trace = self.render_trace();
        let events = trace
            .trim_end()
            .strip_prefix("{\"traceEvents\":")
            .and_then(|rest| rest.strip_suffix(",\"displayTimeUnit\":\"ms\"}"))
            .unwrap_or("[]");
        let notes: Vec<String> = self
            .notes
            .iter()
            .map(|n| format!("\"{}\"", json_escape(n)))
            .collect();
        format!(
            "{{\"time_passes\":{{\"input\":\"{}\",\"events\":{events},\"notes\":[{}]}}}}\n",
            json_escape(&self.header),
            notes.join(",")
        )
    }

    /// The Chrome trace-event JSON (`chrome://tracing`, Perfetto, speedscope): one complete event
    /// per interval, stages on thread 1 and the accumulated pass breakdowns on thread 2.
    fn render_trace(&self) -> String {
        let mut out = String::from("{\"traceEvents\":[");
        for (k, i) in self.intervals.iter().enumerate() {
            if k > 0 {
                out.push(',');
            }
            let cat = if i.lane == 0 { "stage" } else { "breakdown" };
            let _ = write!(
                out,
                "{{\"name\":\"{}\",\"cat\":\"{cat}\",\"ph\":\"X\",\"ts\":{:.3},\"dur\":{:.3},\"pid\":1,\"tid\":{}",
                json_escape(&i.name),
                i.start.as_secs_f64() * 1e6,
                i.dur.as_secs_f64() * 1e6,
                i.lane + 1
            );
            if let Some(calls) = i.calls {
                let _ = write!(out, ",\"args\":{{\"calls\":{calls}}}");
            }
            out.push('}');
        }
        out.push_str("],\"displayTimeUnit\":\"ms\"}\n");
        out
    }
}

impl Drop for PassTimer {
    fn drop(&mut self) {
        if !self.enabled {
            return;
        }
        self.close();
        if self.print {
            if self.json {
                eprint!("{}", self.render_json_line());
            } else {
                eprint!("{}", self.render_table());
            }
        }
        if let Some(path) = &self.trace {
            if let Err(e) = std::fs::write(path, self.render_trace()) {
                eprintln!("warning: could not write `{}`: {e}", path.display());
            }
        }
    }
}

/// What the compile was asked to do, for the report header.
fn mode_name(opts: &Options) -> String {
    if opts.grad.train && opts.grad.train_artifact {
        format!("--train --emit={:?}", opts.emit).to_lowercase()
    } else if opts.grad.train {
        "--train".to_string()
    } else if opts.run {
        format!("--run, backend {:?}", opts.backend).to_lowercase()
    } else {
        format!("--emit={:?}", opts.emit).to_lowercase()
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// The file a `--depfile` rule names as its target: the artifact this compile writes (`--emit=obj`
/// or `--emit=exe`, at `-o` or its default name — a `--train --emit=obj|exe` training artifact
/// included), or — for a mode that writes no artifact (`--run`, an in-process `--train`) — the
/// dependency file itself, so a build system still reruns the command when an input changes.
pub(crate) fn depfile_target(opts: &Options, depfile: &Path) -> PathBuf {
    let stem = opts
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());
    if opts.run || (opts.grad.train && !opts.grad.train_artifact) {
        return depfile.to_path_buf();
    }
    match opts.emit {
        EmitStage::Obj => opts
            .output
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{stem}.o"))),
        EmitStage::Exe => opts
            .output
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{stem}{}", std::env::consts::EXE_SUFFIX))),
        _ => depfile.to_path_buf(),
    }
}

/// The Make-syntax dependency rule for `target`: every source file in `sm` — the root file and
/// each imported file, exactly the set the compile read — followed by an empty rule per imported
/// file, so a deleted import does not break the build (`gcc -MD -MP`).
pub(crate) fn depfile_contents(target: &Path, sm: &SourceMap) -> String {
    let files: Vec<&str> = (0..sm.file_count())
        .map(|i| sm.name(SourceId(i as u32)))
        .collect();
    let mut out = format!("{}:", make_escape(&target.display().to_string()));
    for f in &files {
        let _ = write!(out, " \\\n  {}", make_escape(f));
    }
    out.push('\n');
    for f in files.iter().skip(1) {
        let _ = write!(out, "\n{}:\n", make_escape(f));
    }
    out
}

/// Escape a path for a Make rule: a space or `#` is backslash-escaped, `$` doubled.
fn make_escape(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for c in path.chars() {
        match c {
            ' ' => out.push_str("\\ "),
            '#' => out.push_str("\\#"),
            '$' => out.push_str("$$"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depfile_lists_every_source_with_phony_rules_for_imports() {
        let mut sm = SourceMap::new();
        sm.add("dir/main.wk", "");
        sm.add("dir/lib/my math.wk", "");
        sm.add("dir/lib/$cost#1.wk", "");
        let text = depfile_contents(Path::new("out/prog.o"), &sm);
        assert_eq!(
            text,
            "out/prog.o: \\\n  dir/main.wk \\\n  dir/lib/my\\ math.wk \\\n  dir/lib/$$cost\\#1.wk\n\
             \ndir/lib/my\\ math.wk:\n\ndir/lib/$$cost\\#1.wk:\n"
        );
    }

    #[test]
    fn depfile_target_is_the_artifact_or_the_depfile() {
        let dep = Path::new("build/prog.d");
        let mut opts = Options {
            input: PathBuf::from("src/prog.wk"),
            emit: EmitStage::Obj,
            ..Options::default()
        };
        assert_eq!(depfile_target(&opts, dep), PathBuf::from("prog.o"));
        opts.output = Some(PathBuf::from("build/prog.obj"));
        assert_eq!(depfile_target(&opts, dep), PathBuf::from("build/prog.obj"));
        opts.emit = EmitStage::Exe;
        assert_eq!(depfile_target(&opts, dep), PathBuf::from("build/prog.obj"));
        opts.output = None;
        assert_eq!(
            depfile_target(&opts, dep),
            PathBuf::from(format!("prog{}", std::env::consts::EXE_SUFFIX))
        );
        opts.emit = EmitStage::Mir;
        assert_eq!(depfile_target(&opts, dep), dep);
        opts.emit = EmitStage::Exe;
        opts.run = true;
        assert_eq!(depfile_target(&opts, dep), dep);
        // `--train` runs in process and writes no artifact; `--train --emit=exe|obj` (or `-o`)
        // builds one, and that artifact is the rule's target, at `-o` or its default name.
        opts.run = false;
        opts.grad.train = true;
        assert_eq!(depfile_target(&opts, dep), dep);
        opts.grad.train_artifact = true;
        assert_eq!(
            depfile_target(&opts, dep),
            PathBuf::from(format!("prog{}", std::env::consts::EXE_SUFFIX))
        );
        opts.output = Some(PathBuf::from("build/train.exe"));
        assert_eq!(depfile_target(&opts, dep), PathBuf::from("build/train.exe"));
        opts.emit = EmitStage::Obj;
        opts.output = None;
        assert_eq!(depfile_target(&opts, dep), PathBuf::from("prog.o"));
        assert_eq!(mode_name(&opts), "--train --emit=obj");
        opts.grad.train_artifact = false;
        assert_eq!(mode_name(&opts), "--train");
    }

    #[test]
    fn disabled_timer_records_nothing() {
        let mut timer = PassTimer::new(&Options::default());
        assert!(!timer.enabled());
        timer.begin("lex");
        timer.note(|| unreachable!("a disabled timer must not format its notes"));
        timer.close();
        assert!(timer.intervals.is_empty());
    }

    #[test]
    fn report_and_trace_cover_every_stage_in_order() {
        let opts = Options {
            input: PathBuf::from("p.wk"),
            time_passes: true,
            ..Options::default()
        };
        let mut timer = PassTimer::new(&opts);
        timer.begin("lex");
        timer.begin("optimize");
        timer.breakdown([(
            "mem2reg \"x\"".to_string(),
            Duration::from_micros(5),
            Some(3),
        )]);
        timer.note(|| "pruned 1 function".to_string());
        timer.begin("run");
        timer.close();
        // Silence the drop-time report: this test inspects the rendered text directly.
        timer.print = false;
        let table = timer.render_table();
        let lex = table.find("  lex").unwrap();
        let opt = table.find("  optimize").unwrap();
        let run = table.find("  run").unwrap();
        assert!(lex < opt && opt < run, "{table}");
        assert!(table.contains("calls 3"), "{table}");
        assert!(table.contains("pruned 1 function"), "{table}");
        let trace = timer.render_trace();
        assert!(trace.starts_with("{\"traceEvents\":["), "{trace}");
        assert!(trace.contains("\"name\":\"mem2reg \\\"x\\\"\""), "{trace}");
        assert!(trace.contains("\"tid\":2"), "{trace}");
        assert!(
            trace.trim_end().ends_with("\"displayTimeUnit\":\"ms\"}"),
            "{trace}"
        );
        let line = timer.render_json_line();
        assert!(
            line.starts_with("{\"time_passes\":{\"input\":\"p.wk (-O0, "),
            "{line}"
        );
        assert!(line.contains("\"events\":[{\"name\":\"lex\""), "{line}");
        assert!(line.contains("\"notes\":[\"pruned 1 function\"]"), "{line}");
        assert!(line.ends_with("}}\n"), "{line}");
        assert_eq!(line.matches('\n').count(), 1, "one line: {line}");
    }
}
