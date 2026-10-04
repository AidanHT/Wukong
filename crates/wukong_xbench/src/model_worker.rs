//! Bounded transport for the model peer. No Torch dependency: tests use a fake child.
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MAX_LINE: usize = 16 * 1024;
const MAX_TRANSCRIPT: usize = 64 * 1024;
const SHUTDOWN: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Variant {
    Sdpa1,
    Manual1,
    Comp1,
    SdpaN,
    CompN,
}

impl Variant {
    pub(super) const ALL: [Self; 5] = [
        Self::Sdpa1,
        Self::Manual1,
        Self::Comp1,
        Self::SdpaN,
        Self::CompN,
    ];

    pub(super) fn tag(self) -> &'static str {
        match self {
            Self::Sdpa1 => "TORCH1",
            Self::Manual1 => "TORCH1M",
            Self::Comp1 => "TORCH1C",
            Self::SdpaN => "TORCHN",
            Self::CompN => "TORCHNC",
        }
    }
}

enum Event {
    Line(String),
    End,
    Error(String),
}

/// A process remains owned until successful STOP and exit, or bounded failure cleanup.
pub(super) struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    output: Option<mpsc::Receiver<Event>>,
    readers: Vec<JoinHandle<()>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    timeout: Duration,
    next_id: u64,
    stopped: bool,
}

impl Worker {
    pub(super) fn start(
        command: &mut Command,
        timeout: Duration,
    ) -> Result<(Self, [bool; 5], Vec<String>), String> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn model peer: {e}"))?;
        let input = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr_pipe = child.stderr.take().expect("piped stderr");
        // A bounded queue drains stdout while the parent handles native columns, without
        // allowing a noisy/misbehaving peer to consume unbounded memory.
        let (send, output) = mpsc::sync_channel(32);
        let out_reader = std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                let read = (&mut reader)
                    .take((MAX_LINE + 1) as u64)
                    .read_line(&mut line);
                let event = match read {
                    Ok(0) => Event::End,
                    Ok(_) if line.len() > MAX_LINE => Event::Error("stdout line too long".into()),
                    Ok(_) => Event::Line(line.trim_end_matches(['\r', '\n']).to_string()),
                    Err(e) => Event::Error(format!("read stdout: {e}")),
                };
                let terminal = !matches!(event, Event::Line(_));
                if send.send(event).is_err() || terminal {
                    break;
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&stderr);
        let err_reader = std::thread::spawn(move || {
            let mut pipe = stderr_pipe;
            let mut bytes = [0; 4096];
            while let Ok(n) = pipe.read(&mut bytes) {
                if n == 0 {
                    break;
                }
                let mut tail = captured.lock().expect("stderr capture lock");
                let discard = (tail.len() + n).saturating_sub(MAX_TRANSCRIPT);
                tail.drain(..discard);
                tail.extend_from_slice(&bytes[..n]);
            }
        });
        let mut worker = Self {
            child,
            input,
            output: Some(output),
            readers: vec![out_reader, err_reader],
            stderr,
            timeout,
            next_id: 1,
            stopped: false,
        };
        let prepared = worker.read_ready();
        match prepared {
            Ok((available, metadata)) => Ok((worker, available, metadata)),
            Err(error) => Err(worker.fail(error)),
        }
    }

    fn event(&self, deadline: Instant) -> Result<Event, String> {
        if Instant::now() >= deadline {
            return Err("model peer reply deadline exceeded".into());
        }
        self.output
            .as_ref()
            .ok_or("closed peer stdout")?
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|e| format!("model peer reply: {e}"))
    }

    fn read_ready(&mut self) -> Result<([bool; 5], Vec<String>), String> {
        let deadline = Instant::now() + self.timeout;
        let mut metadata = Vec::new();
        let mut bytes = 0;
        loop {
            match self.event(deadline)? {
                Event::Line(line) if line.starts_with("MODEL ") => {
                    let fields: Vec<_> = line.split_whitespace().collect();
                    if fields.len() != 3 || fields[..2] != ["MODEL", "READY"] {
                        return Err(format!("unexpected preparation frame: {line}"));
                    }
                    let mask = fields[2].as_bytes();
                    if mask.len() != 5 || mask.iter().any(|b| !matches!(b, b'0' | b'1')) {
                        return Err(format!("invalid READY availability: {line}"));
                    }
                    let available = std::array::from_fn(|i| mask[i] == b'1');
                    if !available[0] || !available[1] || !available[3] {
                        return Err("READY is missing an eager variant".into());
                    }
                    return Ok((available, metadata));
                }
                Event::Line(line) => {
                    bytes += line.len() + 1 + std::mem::size_of::<String>();
                    if bytes > MAX_TRANSCRIPT {
                        return Err("preparation transcript too long".into());
                    }
                    metadata.push(line);
                }
                Event::End => return Err("peer exited before READY".into()),
                Event::Error(e) => return Err(e),
            }
        }
    }

    fn send(&mut self, command: &str) -> Result<(), String> {
        let input = self.input.as_mut().ok_or("closed peer stdin")?;
        writeln!(input, "{command}")
            .and_then(|()| input.flush())
            .map_err(|e| format!("write model peer: {e}"))
    }

    pub(super) fn time(&mut self, variant: Variant) -> Result<(f64, f64), String> {
        let id = self.next_id;
        self.next_id += 1;
        let result = (|| {
            self.send(&format!("TIME {id} {}", variant.tag()))?;
            let deadline = Instant::now() + self.timeout;
            let mut noise = 0;
            loop {
                match self.event(deadline)? {
                    Event::Line(line) if line.starts_with("MODEL ") => {
                        return parse_result(&line, id, variant);
                    }
                    Event::Line(line) => {
                        noise += line.len() + 1;
                        if noise > MAX_TRANSCRIPT {
                            return Err("timing transcript too long".into());
                        }
                    }
                    Event::End => return Err("peer exited during TIME".into()),
                    Event::Error(e) => return Err(e),
                }
            }
        })();
        result.map_err(|error| self.fail(error))
    }

    /// STOP acknowledgement is insufficient: require EOF and a successful process exit too.
    pub(super) fn finish(&mut self) -> Result<(), String> {
        let id = self.next_id;
        let result = (|| {
            self.send(&format!("STOP {id}"))?;
            self.input.take();
            let deadline = Instant::now() + SHUTDOWN;
            let expected = format!("MODEL STOPPED {id}");
            let mut acknowledged = false;
            let mut noise = 0;
            loop {
                match self.event(deadline)? {
                    Event::Line(line) if line.starts_with("MODEL ") => {
                        if acknowledged || line != expected {
                            return Err(format!("unexpected shutdown frame: {line}"));
                        }
                        acknowledged = true;
                    }
                    Event::Line(line) => {
                        noise += line.len() + 1;
                        if noise > MAX_TRANSCRIPT {
                            return Err("shutdown transcript too long".into());
                        }
                    }
                    Event::End if acknowledged => break,
                    Event::End => return Err("peer exited without STOPPED".into()),
                    Event::Error(e) => return Err(e),
                }
            }
            loop {
                match self.child.try_wait().map_err(|e| e.to_string())? {
                    Some(status) if status.success() => {
                        self.stopped = true;
                        return Ok(());
                    }
                    Some(status) => return Err(format!("peer exited unsuccessfully: {status}")),
                    None if Instant::now() >= deadline => return Err("peer exit timeout".into()),
                    None => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        })();
        result.map_err(|error| self.fail(error))
    }

    fn fail(&mut self, error: String) -> String {
        self.cleanup();
        let stderr = self.stderr.lock().expect("stderr capture lock");
        format!(
            "{error}\npeer stderr (last {MAX_TRANSCRIPT} bytes): {}",
            String::from_utf8_lossy(&stderr)
        )
    }

    fn cleanup(&mut self) {
        self.input.take();
        // Disconnect before joining: a reader may be blocked on the bounded queue.
        self.output.take();
        if !self.stopped {
            let _ = self.child.kill();
            let deadline = Instant::now() + SHUTDOWN;
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => {
                        self.stopped = true;
                        break;
                    }
                    _ if Instant::now() >= deadline => {
                        eprintln!("model: peer could not be reaped within shutdown deadline");
                        break;
                    }
                    _ => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        }
        // A descendant can inherit a pipe and keep it open after the parent dies. Never
        // block cleanup indefinitely joining that reader; bounded storage still applies.
        let deadline = Instant::now() + Duration::from_millis(100);
        while self.readers.iter().any(|h| !h.is_finished()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        for reader in self.readers.drain(..) {
            if reader.is_finished() {
                let _ = reader.join();
            } else {
                eprintln!("model: detached peer reader; an inherited pipe may still be open");
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cleanup();
    }
}

fn parse_result(line: &str, id: u64, variant: Variant) -> Result<(f64, f64), String> {
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() != 6
        || fields[..2] != ["MODEL", "RESULT"]
        || fields[2].parse::<u64>() != Ok(id)
        || fields[3] != variant.tag()
    {
        return Err(format!(
            "unexpected timing frame for {id}/{}: {line}",
            variant.tag()
        ));
    }
    let min = fields[4]
        .parse::<f64>()
        .map_err(|_| format!("invalid timing: {line}"))?;
    let median = fields[5]
        .parse::<f64>()
        .map_err(|_| format!("invalid timing: {line}"))?;
    if !min.is_finite()
        || !median.is_finite()
        || min <= 0.0
        || min > median
        || !(min * 1e6).is_finite()
        || !(median * 1e6).is_finite()
    {
        return Err(format!("invalid min/median: {line}"));
    }
    Ok((min * 1e6, median * 1e6))
}

#[cfg(test)]
pub(super) fn fake_for_test(mode: &str) -> Worker {
    tests::start(mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(mode: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "model::model_worker::tests::fake_child",
            "--nocapture",
            "--format=terse",
            "--test-threads=1",
        ]);
        command.env("WUKONG_MODEL_FAKE_CHILD", mode);
        command
    }

    pub(super) fn start(mode: &str) -> Worker {
        let (worker, available, metadata) =
            Worker::start(&mut fake(mode), Duration::from_secs(10)).unwrap();
        assert_eq!(available, [true; 5]);
        assert!(metadata
            .iter()
            .any(|s| s == "TORCH_COMPILE_DISCLOSE metadata mentions READY and RESULT"));
        assert!(metadata
            .iter()
            .any(|s| s == "TORCH_COMPILE_DISCLOSE café λ"));
        worker
    }

    // The child is this test executable, never Python or Torch. The parent passes a
    // per-child environment variable; no test changes process-global environment.
    #[test]
    fn fake_child() {
        let Ok(mode) = std::env::var("WUKONG_MODEL_FAKE_CHILD") else {
            return;
        };
        if mode == "early_exit" {
            std::process::exit(7);
        }
        if mode == "empty_flood" {
            for _ in 0..=MAX_TRANSCRIPT {
                println!();
            }
        }
        if mode == "bad_ready" {
            println!("MODEL READY 11x11");
            return;
        }
        if mode == "stderr" {
            std::io::stderr()
                .write_all(&vec![b'x'; MAX_TRANSCRIPT * 4])
                .unwrap();
        }
        println!("TORCH_COMPILE_DISCLOSE metadata mentions READY and RESULT");
        println!("TORCH_COMPILE_DISCLOSE café λ");
        println!("MODEL READY 11111");
        std::io::stdout().flush().unwrap();
        for request in std::io::stdin().lock().lines() {
            let request = request.unwrap();
            let fields: Vec<_> = request.split_whitespace().collect();
            if fields[0] == "STOP" {
                println!("MODEL STOPPED {}", fields[1]);
                std::io::stdout().flush().unwrap();
                if mode == "bad_exit" {
                    std::process::exit(9);
                }
                return;
            }
            if mode == "time_exit" {
                std::process::exit(8);
            }
            if mode == "timeout" {
                std::thread::sleep(Duration::from_secs(60));
            }
            let id: u64 = fields[1].parse().unwrap();
            let reply_id = if mode == "wrong_id" { id + 1 } else { id };
            println!("MODEL RESULT {reply_id} {} 2 7", fields[2]);
            if mode == "duplicate" {
                println!("MODEL RESULT {reply_id} {} 1 2", fields[2]);
            }
            std::io::stdout().flush().unwrap();
        }
    }

    #[test]
    fn result_requires_matching_id_variant_and_valid_pair() {
        assert_eq!(
            parse_result("MODEL RESULT 3 TORCH1 2 7", 3, Variant::Sdpa1),
            Ok((2e6, 7e6))
        );
        for line in [
            "MODEL RESULT 2 TORCH1 2 7",
            "MODEL RESULT 4 TORCH1 2 7",
            "MODEL RESULT 3 TORCHN 2 7",
            "MODEL READY 11111",
            "MODEL RESULT 3 TORCH1 NaN 7",
            "MODEL RESULT 3 TORCH1 2 inf",
            "MODEL RESULT 3 TORCH1 0 7",
            "MODEL RESULT 3 TORCH1 -1 7",
            "MODEL RESULT 3 TORCH1 8 7",
            "MODEL RESULT 3 TORCH1 2 7 extra",
            "MODEL RESULT 3 TORCH1 1e308 1e308",
            "MODEL RESULT 3 TORCH1 2",
        ] {
            assert!(
                parse_result(line, 3, Variant::Sdpa1).is_err(),
                "accepted {line}"
            );
        }
    }

    #[test]
    fn persistent_child_correlates_all_variants_and_reaps_on_stop() {
        let mut worker = start("normal");
        for variant in Variant::ALL
            .into_iter()
            .chain(Variant::ALL.into_iter().rev())
        {
            assert_eq!(worker.time(variant).unwrap(), (2e6, 7e6));
        }
        worker.finish().unwrap();
        assert!(worker.child.try_wait().unwrap().unwrap().success());
        assert_eq!(worker.next_id, 11);
    }

    #[test]
    fn drains_large_stderr_without_unbounded_capture() {
        let mut worker = start("stderr");
        assert_eq!(worker.time(Variant::Comp1).unwrap(), (2e6, 7e6));
        worker.finish().unwrap();
        worker.cleanup();
        assert_eq!(worker.stderr.lock().unwrap().len(), MAX_TRANSCRIPT);
    }

    #[test]
    fn rejects_early_exit_and_malformed_ready() {
        for mode in ["early_exit", "bad_ready"] {
            assert!(Worker::start(&mut fake(mode), Duration::from_secs(10)).is_err());
        }
    }

    #[test]
    fn empty_metadata_lines_consume_the_transcript_budget() {
        let error = match Worker::start(&mut fake("empty_flood"), Duration::from_secs(10)) {
            Ok(_) => panic!("empty metadata flood was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("preparation transcript too long"), "{error}");
    }

    #[test]
    fn wrong_id_eof_and_deadline_fail_and_reap() {
        for mode in ["wrong_id", "time_exit", "timeout"] {
            let mut worker = start(mode);
            if mode == "timeout" {
                worker.timeout = Duration::from_millis(100);
            }
            assert!(worker.time(Variant::Sdpa1).is_err(), "accepted {mode}");
            assert!(
                worker.child.try_wait().unwrap().is_some(),
                "did not reap {mode}"
            );
        }
    }

    #[test]
    fn duplicate_final_reply_and_acknowledged_nonzero_exit_fail() {
        for mode in ["duplicate", "bad_exit"] {
            let mut worker = start(mode);
            worker.time(Variant::CompN).unwrap();
            assert!(worker.finish().is_err(), "accepted {mode}");
            assert!(worker.child.try_wait().unwrap().is_some());
        }
    }
}
