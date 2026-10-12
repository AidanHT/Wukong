//! The **AOT runtime cache**: the expensive, program-independent half of an `--emit=exe` link,
//! built once and reused by every later link.
//!
//! An `--emit=exe` link joins two things: the Cranelift object (different for every program) and
//! the runtime it calls into (the same for every program a given compiler builds). Before this
//! cache, every link rebuilt the runtime half from scratch: `rustc` compiled the generated
//! `WUKONG_RT_SHIM` against the `wukong_runtime` rlib and, when that rlib came from the release
//! profile (`lto = "thin"`, so its members are LLVM bitcode), ran **thin LTO over the whole runtime
//! and its standard library** — 35 to 40 seconds per link on the development host, for a program
//! that printed one number.
//!
//! Now the runtime half is an [`Entry`] in a cache directory, keyed by everything that changes it
//! (see [`key_of`] and the material the driver assembles), and a warm link only lets `rustc`
//! *drive* the final link: it compiles an empty driver crate and hands the object plus the cached
//! runtime to the platform linker it already knows how to find, so no platform-linker locator is
//! needed. Three flavours exist (`crate::rustc_link` and `crate::cc_link` build them):
//!
//! * [`Flavor::RustNative`] — a dev-profile runtime rlib (native objects). The entry is the shim
//!   compiled once as an rlib; the warm link is `rustc` over a two-line `extern crate` driver. No
//!   LTO is involved before or after.
//! * [`Flavor::RustLto`] — a release-profile runtime rlib (bitcode members). The entry is a
//!   **fat-LTO static library** of the shim, the runtime and the standard library, plus the native
//!   libraries `rustc --print native-static-libs` names. Fat, not thin: fat LTO internalizes every
//!   Rust symbol, so the archive exports only the `#[no_mangle]` C entry points and cannot collide
//!   with the warm link's own `core` (thin LTO leaves cross-module `core` symbols global, and the
//!   link fails with `LNK2005`). The warm link compiles a `#![no_std]` driver, so no second copy
//!   of `std` is linked at all.
//! * [`Flavor::Cc`] — the `cc` fallback: the generated C runtime compiled once to an object.
//!
//! **Location.** `WUKONG_CACHE_DIR` when set (an empty value disables the cache), else the
//! platform's cache directory: `%LOCALAPPDATA%\wukong\cache` on Windows, `$XDG_CACHE_HOME/wukong`
//! or `~/.cache/wukong` elsewhere. Entries live in its `aot-runtime` subdirectory, which is what
//! `wukongc --print=runtime-cache` prints and what `wukongc --clear-cache` empties. Without a usable
//! location the entry is built privately for the one link and removed afterwards, so the link
//! itself never depends on the cache.
//!
//! **Concurrency and corruption.** An entry is built in a private `.tmp-*` directory beside the
//! final one and published with a single directory `rename`, so a reader sees a complete entry or
//! none. Two processes that miss at the same time both build; the first rename wins and the loser
//! adopts the winner's entry (the test suite links in parallel). Every hit is validated against the
//! entry's `manifest`, which records each file's length and content hash and ends with an `end`
//! line, and against `material.txt`, the exact key material: a truncated, edited or foreign entry
//! is reported (under `WUKONG_CACHE_TRACE`) and rebuilt.
//!
//! **Size.** At most [`MAX_ENTRIES`] entries are kept; publishing a new one evicts the least
//! recently used beyond that, but never one used in the last ten minutes (a concurrent link may be
//! reading it).
//!
//! **Tracing.** `WUKONG_CACHE_TRACE=1` prints one `wukongc: runtime cache: ...` line per decision
//! (hit, miss, invalid, built, published, adopted, private, evicted) on stderr.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Output;
use std::time::{Duration, Instant, SystemTime};

/// Overrides the cache location; an empty value disables the cache.
pub const CACHE_DIR_ENV: &str = "WUKONG_CACHE_DIR";
/// Set (to anything but empty or `0`) to print every cache decision on stderr.
pub const CACHE_TRACE_ENV: &str = "WUKONG_CACHE_TRACE";

/// The subdirectory of the cache root that holds the runtime entries.
const RUNTIME_DIR: &str = "aot-runtime";
/// Written last into a finished entry: the inventory every hit is validated against.
const MANIFEST: &str = "manifest";
/// The exact key material of an entry, compared byte for byte on every hit.
const MATERIAL: &str = "material.txt";
/// The manifest's first line. Bump the version when the entry layout or a recipe changes meaning.
const HEADER: &str = "wukong-aot-runtime-cache v1";
/// Entries kept after a publish (the least recently used beyond this are evicted).
pub(crate) const MAX_ENTRIES: usize = 8;
/// An entry used this recently is never evicted: a concurrent link may be reading it.
const EVICTION_GRACE: Duration = Duration::from_secs(10 * 60);
/// A `.tmp-*` build directory this old belongs to a process that died mid-build.
const STALE_TMP: Duration = Duration::from_secs(60 * 60);

/// Which runtime an entry holds; part of its name and of its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flavor {
    /// The shim compiled as an rlib against a dev-profile (native-object) runtime rlib.
    RustNative,
    /// A fat-LTO static library of the shim, a release-profile (bitcode) runtime and `std`.
    RustLto,
    /// The `cc` fallback's C runtime compiled to an object.
    Cc,
}

impl Flavor {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Flavor::RustNative => "rust-native",
            Flavor::RustLto => "rust-lto",
            Flavor::Cc => "cc",
        }
    }

    const ALL: [Flavor; 3] = [Flavor::RustNative, Flavor::RustLto, Flavor::Cc];
}

/// The directory `wukongc --print=runtime-cache` prints: `<cache root>/aot-runtime`. `Err` says why
/// there is none (the cache is disabled, or no cache location exists in this environment).
pub fn runtime_cache_dir() -> Result<PathBuf, String> {
    cache_root().map(|root| root.join(RUNTIME_DIR))
}

fn cache_root() -> Result<PathBuf, String> {
    if let Some(value) = std::env::var_os(CACHE_DIR_ENV) {
        if value.is_empty() {
            return Err(format!(
                "the runtime cache is disabled ({CACHE_DIR_ENV} is set to an empty value)"
            ));
        }
        let path = PathBuf::from(value);
        return Ok(std::path::absolute(&path).unwrap_or(path));
    }
    // The driver's own unit tests link in-process; they must never write the user's real cache.
    if cfg!(test) {
        return Ok(std::env::temp_dir().join("wukong-driver-test-cache"));
    }
    platform_cache_root().ok_or_else(|| {
        let looked_at = if cfg!(windows) {
            "LOCALAPPDATA"
        } else {
            "XDG_CACHE_HOME or HOME"
        };
        format!(
            "no cache directory: {looked_at} is not set to an absolute path and {CACHE_DIR_ENV} \
             is not set"
        )
    })
}

/// `%LOCALAPPDATA%\wukong\cache` on Windows; `$XDG_CACHE_HOME/wukong`, else `~/.cache/wukong`,
/// elsewhere. A relative `XDG_CACHE_HOME` is ignored, as the XDG base-directory spec requires.
fn platform_cache_root() -> Option<PathBuf> {
    let absolute = |key: &str| {
        std::env::var_os(key)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
    };
    if cfg!(windows) {
        absolute("LOCALAPPDATA").map(|d| d.join("wukong").join("cache"))
    } else {
        absolute("XDG_CACHE_HOME")
            .or_else(|| absolute("HOME").map(|home| home.join(".cache")))
            .map(|d| d.join("wukong"))
    }
}

fn trace_requested() -> bool {
    std::env::var_os(CACHE_TRACE_ENV).is_some_and(|v| !v.is_empty() && v != "0")
}

/// A 128-bit content hash: two independent multiply-rotate lanes over little-endian 64-bit words,
/// finished with the length and a murmur3 avalanche. Not cryptographic, and it does not need to be:
/// it names entries (whose full key material is also compared byte for byte) and detects corrupted
/// files, and every per-word step is a bijection of the state, so changing any one word always
/// changes the result. std-only and specified here, so a hash means the same thing in every build
/// of wukongc (`std`'s `DefaultHasher` makes no such promise).
pub(crate) struct ContentHash {
    a: u64,
    b: u64,
    len: u64,
    tail: [u8; 8],
    tail_len: usize,
}

impl ContentHash {
    pub(crate) fn new() -> ContentHash {
        ContentHash {
            a: 0x243F_6A88_85A3_08D3,
            b: 0x1319_8A2E_0370_7344,
            len: 0,
            tail: [0; 8],
            tail_len: 0,
        }
    }

    fn word(&mut self, w: u64) {
        self.a = (self.a.rotate_left(5) ^ w).wrapping_mul(0x517C_C1B7_2722_0A95);
        self.b = (self.b ^ w)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(29);
    }

    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.len = self.len.wrapping_add(bytes.len() as u64);
        if self.tail_len > 0 {
            let take = (8 - self.tail_len).min(bytes.len());
            self.tail[self.tail_len..self.tail_len + take].copy_from_slice(&bytes[..take]);
            self.tail_len += take;
            bytes = &bytes[take..];
            if self.tail_len < 8 {
                return;
            }
            self.word(u64::from_le_bytes(self.tail));
            self.tail_len = 0;
        }
        let mut words = bytes.chunks_exact(8);
        for w in &mut words {
            self.word(u64::from_le_bytes(w.try_into().expect("an 8-byte chunk")));
        }
        let rest = words.remainder();
        self.tail[..rest.len()].copy_from_slice(rest);
        self.tail_len = rest.len();
    }

    pub(crate) fn finish(mut self) -> String {
        fn fmix(mut k: u64) -> u64 {
            k ^= k >> 33;
            k = k.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            k ^= k >> 33;
            k = k.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
            k ^ (k >> 33)
        }
        if self.tail_len > 0 {
            let mut last = [0u8; 8];
            last[..self.tail_len].copy_from_slice(&self.tail[..self.tail_len]);
            self.word(u64::from_le_bytes(last));
        }
        self.word(self.len);
        let hi = fmix(self.a ^ self.b.rotate_left(17));
        let lo = fmix(self.b ^ hi);
        format!("{hi:016x}{lo:016x}")
    }
}

pub(crate) fn hash_bytes(bytes: &[u8]) -> String {
    let mut h = ContentHash::new();
    h.update(bytes);
    h.finish()
}

/// `(length, content hash)` of a file, streamed.
pub(crate) fn hash_file(path: &Path) -> io::Result<(u64, String)> {
    let mut file = fs::File::open(path)?;
    let mut h = ContentHash::new();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok((h.len, h.finish()))
}

/// The entry name for `material`: the flavour, then the hash of the exact key material. The
/// material itself is stored in the entry and compared on every hit, so two materials that hashed
/// alike could never share an entry.
pub(crate) fn key_of(flavor: Flavor, material: &str) -> String {
    format!("{}-{}", flavor.name(), hash_bytes(material.as_bytes()))
}

/// The identity of the runtime rlib an entry was built from, as key material: its absolute path,
/// length and modification time. Not its content: hashing the 80+ MB dev-profile rlib on every link
/// would cost a debug wukongc about 0.3 s, more than half the link it is trying to save, while
/// cargo writes a new file (a new mtime) whenever the rlib changes at all.
pub(crate) fn file_identity(path: &Path) -> io::Result<String> {
    let meta = fs::metadata(path)?;
    let modified = meta
        .modified()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    Ok(format!(
        "{} len={} mtime={}.{:09}",
        absolute.display(),
        meta.len(),
        modified.as_secs(),
        modified.subsec_nanos()
    ))
}

/// The host the cached runtime targets (the Cranelift object is always for the host).
pub(crate) fn host_material() -> String {
    let env = if cfg!(target_env = "msvc") {
        "msvc"
    } else if cfg!(target_env = "gnu") {
        "gnu"
    } else if cfg!(target_env = "musl") {
        "musl"
    } else {
        "other"
    };
    format!(
        "{}-{}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        env
    )
}

/// Key material: one `name value` line per input, in a fixed order. Multi-line values (a
/// toolchain's `-vV` text) contribute one line each under the same name.
#[derive(Default)]
pub(crate) struct Material {
    text: String,
}

impl Material {
    pub(crate) fn new(flavor: Flavor) -> Material {
        let mut m = Material::default();
        m.line("format", HEADER);
        m.line("flavor", flavor.name());
        m.line("host", &host_material());
        m
    }

    pub(crate) fn line(&mut self, name: &str, value: &str) -> &mut Material {
        let mut any = false;
        for l in value.lines() {
            let l = l.trim_end();
            if l.is_empty() {
                continue;
            }
            self.text.push_str(name);
            self.text.push(' ');
            self.text.push_str(l);
            self.text.push('\n');
            any = true;
        }
        if !any {
            self.text.push_str(name);
            self.text.push('\n');
        }
        self
    }

    pub(crate) fn finish(&self) -> String {
        self.text.clone()
    }
}

/// What a flavour's build step produced in the directory it was given.
pub(crate) enum Built {
    /// The build succeeded. `files` are the artifacts (and inputs worth keeping) it wrote, by name;
    /// `link_args` are extra arguments every link against the entry needs, in order.
    Ok {
        files: Vec<String>,
        link_args: Vec<String>,
    },
    /// The build tool ran and failed; its output is the diagnosis.
    Failed(Output),
    /// Preparing the build failed (writing a source file): the link cannot proceed.
    Io(String),
}

/// The result of [`RuntimeCache::obtain`].
pub(crate) enum Obtained {
    /// A valid entry, cached or private.
    Ready(Entry),
    /// The build tool ran and failed.
    BuildFailed(Output),
    /// The build tool could not be started at all.
    Spawn(io::Error),
    /// Writing the entry failed (no space, no permission): the link cannot proceed.
    Io(String),
}

/// A directory removed when dropped: a private entry, built for one link because there is no
/// usable cache location (or the cache would not take it).
struct PrivateDir(PathBuf);

impl Drop for PrivateDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A valid runtime entry: the directory holding its files and the arguments a link against it
/// needs. A private entry's directory is removed when the `Entry` is dropped, so keep it alive
/// until the link that uses it has finished.
pub(crate) struct Entry {
    pub(crate) dir: PathBuf,
    pub(crate) link_args: Vec<String>,
    /// Whether the entry is the cache's (true) or a private, one-link build (false).
    pub(crate) cached: bool,
    _private: Option<PrivateDir>,
}

impl Entry {
    pub(crate) fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

/// A parsed `manifest`.
struct Manifest {
    /// `(name, length, content hash)`.
    files: Vec<(String, u64, String)>,
    link_args: Vec<String>,
}

impl Manifest {
    fn render(&self) -> String {
        let mut out = format!("{HEADER}\n");
        for (name, len, hash) in &self.files {
            out.push_str(&format!("file {len} {hash} {name}\n"));
        }
        for arg in &self.link_args {
            out.push_str(&format!("link-arg {arg}\n"));
        }
        out.push_str("end\n");
        out
    }

    fn parse(text: &str) -> Result<Manifest, String> {
        let mut lines = text.lines();
        if lines.next() != Some(HEADER) {
            return Err("the manifest has an unknown header".into());
        }
        let mut m = Manifest {
            files: Vec::new(),
            link_args: Vec::new(),
        };
        for line in lines {
            if line == "end" {
                return Ok(m);
            }
            if let Some(rest) = line.strip_prefix("file ") {
                let mut parts = rest.splitn(3, ' ');
                let (Some(len), Some(hash), Some(name)) =
                    (parts.next(), parts.next(), parts.next())
                else {
                    return Err(format!("malformed manifest line `{line}`"));
                };
                let len = len
                    .parse::<u64>()
                    .map_err(|_| format!("malformed manifest line `{line}`"))?;
                if name.is_empty() || name.contains(['/', '\\']) || name == ".." {
                    return Err(format!("manifest names a file outside the entry: `{name}`"));
                }
                m.files.push((name.to_string(), len, hash.to_string()));
            } else if let Some(arg) = line.strip_prefix("link-arg ") {
                m.link_args.push(arg.to_string());
            } else {
                return Err(format!("malformed manifest line `{line}`"));
            }
        }
        Err("the manifest is truncated (no `end` line)".into())
    }
}

/// Is `dir` a complete, uncorrupted entry for `material` holding every `required` file?
fn validate(dir: &Path, material: &str, required: &[&str]) -> Result<Manifest, String> {
    let manifest_path = dir.join(MANIFEST);
    let text = match fs::read(&manifest_path) {
        Ok(bytes) => String::from_utf8(bytes).map_err(|_| "the manifest is not UTF-8")?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Err("it has no manifest".into()),
        Err(e) => return Err(format!("its manifest cannot be read: {e}")),
    };
    let manifest = Manifest::parse(&text)?;
    for (name, len, hash) in &manifest.files {
        let path = dir.join(name);
        let (actual_len, actual_hash) =
            hash_file(&path).map_err(|e| format!("`{name}` cannot be read: {e}"))?;
        if actual_len != *len {
            return Err(format!(
                "`{name}` is {actual_len} bytes, the manifest says {len}"
            ));
        }
        if actual_hash != *hash {
            return Err(format!("`{name}` does not match its recorded content hash"));
        }
    }
    let listed = |name: &str| manifest.files.iter().any(|(n, _, _)| n == name);
    if !listed(MATERIAL) {
        return Err("its manifest does not list the key material".into());
    }
    let stored = fs::read(dir.join(MATERIAL)).map_err(|e| format!("{MATERIAL}: {e}"))?;
    if stored != material.as_bytes() {
        return Err("it was built from different key material".into());
    }
    if let Some(missing) = required.iter().find(|name| !listed(name)) {
        return Err(format!("its manifest does not list `{missing}`"));
    }
    Ok(manifest)
}

/// Build an entry's files into `dir` (an empty directory) and write its manifest last.
fn build_into(
    dir: &Path,
    material: &str,
    cached: bool,
    build: &mut dyn FnMut(&Path, bool) -> io::Result<Built>,
) -> Result<Manifest, Obtained> {
    fs::write(dir.join(MATERIAL), material)
        .map_err(|e| Obtained::Io(format!("could not write `{}`: {e}", dir.display())))?;
    let (files, link_args) = match build(dir, cached) {
        Ok(Built::Ok { files, link_args }) => (files, link_args),
        Ok(Built::Failed(output)) => return Err(Obtained::BuildFailed(output)),
        Ok(Built::Io(message)) => return Err(Obtained::Io(message)),
        // `Err` from a build step means its tool could not be started at all.
        Err(e) => return Err(Obtained::Spawn(e)),
    };
    let mut manifest = Manifest {
        files: Vec::new(),
        link_args,
    };
    for name in std::iter::once(MATERIAL.to_string()).chain(files) {
        let (len, hash) = hash_file(&dir.join(&name)).map_err(|e| {
            Obtained::Io(format!(
                "the runtime build did not leave `{}`: {e}",
                dir.join(&name).display()
            ))
        })?;
        manifest.files.push((name, len, hash));
    }
    fs::write(dir.join(MANIFEST), manifest.render())
        .map_err(|e| Obtained::Io(format!("could not write `{}`: {e}", dir.display())))?;
    Ok(manifest)
}

/// A name unique to this process and moment, for a build directory.
fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{}-{nanos}-{n}", std::process::id())
}

fn age(path: &Path, now: SystemTime) -> Duration {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| now.duration_since(t).ok())
        .unwrap_or_default()
}

fn is_entry_name(name: &str) -> bool {
    Flavor::ALL.iter().any(|f| {
        name.strip_prefix(f.name())
            .is_some_and(|r| r.starts_with('-'))
    })
}

/// What `wukongc --clear-cache` did.
pub struct ClearReport {
    /// The directory that was emptied (`--print=runtime-cache`).
    pub dir: PathBuf,
    /// Entries (and abandoned build directories) removed.
    pub removed: usize,
    /// Entries that could not be removed (in use by a running link, or no permission), with why.
    pub failed: Vec<(PathBuf, String)>,
}

/// Remove every runtime entry (and every abandoned `.tmp-*` build directory) from the cache. Only
/// names this module creates are touched, so a `WUKONG_CACHE_DIR` pointed at a directory holding
/// other files keeps them.
pub fn clear_runtime_cache() -> Result<ClearReport, String> {
    let dir = runtime_cache_dir()?;
    let mut report = ClearReport {
        dir,
        removed: 0,
        failed: Vec::new(),
    };
    let entries = match fs::read_dir(&report.dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(report),
        Err(e) => return Err(format!("cannot read `{}`: {e}", report.dir.display())),
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !(is_entry_name(&name) || name.starts_with(".tmp-")) {
            continue;
        }
        let path = entry.path();
        match fs::remove_dir_all(&path) {
            Ok(()) => report.removed += 1,
            Err(e) => report.failed.push((path, e.to_string())),
        }
    }
    Ok(report)
}

/// Where publishing a freshly built entry ended up.
enum Publish {
    /// Our build is now the cached entry.
    Published,
    /// Another process published a valid entry for the same key first; ours is discarded.
    Adopted(Manifest),
    /// The cache would not take it; the link uses the build directory privately.
    Private(String),
}

/// Move the finished build directory `tmp` into place as `entry_dir`, atomically.
fn publish(tmp: &Path, entry_dir: &Path, material: &str, required: &[&str]) -> Publish {
    if entry_dir.exists() {
        if let Ok(m) = validate(entry_dir, material, required) {
            return Publish::Adopted(m);
        }
        // A corrupt entry: clear it out of the way. If that fails (a file in it is open on
        // Windows), the rename below fails too and this build is used privately.
        let _ = fs::remove_dir_all(entry_dir);
    }
    let mut attempts = 0;
    loop {
        match fs::rename(tmp, entry_dir) {
            Ok(()) => return Publish::Published,
            Err(e) => {
                if entry_dir.exists() {
                    return match validate(entry_dir, material, required) {
                        Ok(m) => Publish::Adopted(m),
                        Err(why) => Publish::Private(format!(
                            "`{}` exists and is not a valid entry ({why})",
                            entry_dir.display()
                        )),
                    };
                }
                // On Windows a just-written file can be held open for a moment (an indexer or a
                // virus scanner), which fails a directory rename transiently: retry briefly.
                attempts += 1;
                if attempts >= 20 {
                    return Publish::Private(format!(
                        "could not move the entry into `{}`: {e}",
                        entry_dir.display()
                    ));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    }
}

/// The runtime cache as this process sees it: where entries live (or why there is no such place)
/// and whether to trace.
pub(crate) struct RuntimeCache {
    dir: Result<PathBuf, String>,
    trace: bool,
}

impl RuntimeCache {
    /// The cache that `WUKONG_CACHE_DIR` (else the platform cache directory) names.
    pub(crate) fn from_env() -> RuntimeCache {
        RuntimeCache {
            dir: runtime_cache_dir(),
            trace: trace_requested(),
        }
    }

    /// A cache rooted at `root` (tests).
    #[cfg(test)]
    pub(crate) fn at(root: &Path) -> RuntimeCache {
        RuntimeCache {
            dir: Ok(root.join(RUNTIME_DIR)),
            trace: trace_requested(),
        }
    }

    /// No cache: every entry is built privately for its one link.
    #[cfg(test)]
    pub(crate) fn disabled() -> RuntimeCache {
        RuntimeCache {
            dir: Err("disabled for this test".into()),
            trace: trace_requested(),
        }
    }

    fn trace(&self, message: impl AsRef<str>) {
        if self.trace {
            crate::eprint_line(&format!("wukongc: runtime cache: {}", message.as_ref()));
        }
    }

    /// The entry for `material`: a valid cached one when there is one, else a fresh build by
    /// `build(dir, cached)` into an empty directory, published for later links when the cache can
    /// take it. `required` names the files a link needs; an entry missing one is rebuilt.
    pub(crate) fn obtain(
        &self,
        flavor: Flavor,
        material: &str,
        required: &[&str],
        build: &mut dyn FnMut(&Path, bool) -> io::Result<Built>,
    ) -> Obtained {
        let key = key_of(flavor, material);
        let dir = match &self.dir {
            Ok(dir) => dir.clone(),
            Err(why) => return self.build_private(material, build, why),
        };
        let entry_dir = dir.join(&key);
        if entry_dir.exists() {
            match validate(&entry_dir, material, required) {
                Ok(manifest) => {
                    // The manifest's mtime is the entry's last use, which eviction ranks on.
                    let _ = fs::File::options()
                        .write(true)
                        .open(entry_dir.join(MANIFEST))
                        .and_then(|f| f.set_modified(SystemTime::now()));
                    self.trace(format!("hit {}", entry_dir.display()));
                    return Obtained::Ready(Entry {
                        dir: entry_dir,
                        link_args: manifest.link_args,
                        cached: true,
                        _private: None,
                    });
                }
                Err(why) => self.trace(format!(
                    "invalid entry {} ({why}); rebuilding it",
                    entry_dir.display()
                )),
            }
        } else {
            self.trace(format!("miss {}", entry_dir.display()));
        }
        if let Err(e) = fs::create_dir_all(&dir) {
            return self.build_private(
                material,
                build,
                &format!("cannot create `{}`: {e}", dir.display()),
            );
        }
        let tmp = dir.join(format!(".tmp-{key}-{}", unique_suffix()));
        if let Err(e) = fs::create_dir(&tmp) {
            return self.build_private(
                material,
                build,
                &format!("cannot create `{}`: {e}", tmp.display()),
            );
        }
        let guard = PrivateDir(tmp.clone());
        let started = Instant::now();
        let manifest = match build_into(&tmp, material, true, build) {
            Ok(m) => m,
            Err(failure) => return failure,
        };
        self.trace(format!(
            "built {key} in {} ms",
            started.elapsed().as_millis()
        ));
        match publish(&tmp, &entry_dir, material, required) {
            Publish::Published => {
                std::mem::forget(guard);
                self.trace(format!("published {}", entry_dir.display()));
                self.evict(&dir, &key);
                Obtained::Ready(Entry {
                    dir: entry_dir,
                    link_args: manifest.link_args,
                    cached: true,
                    _private: None,
                })
            }
            Publish::Adopted(theirs) => {
                drop(guard);
                self.trace(format!(
                    "adopted {} (another link published it first)",
                    entry_dir.display()
                ));
                Obtained::Ready(Entry {
                    dir: entry_dir,
                    link_args: theirs.link_args,
                    cached: true,
                    _private: None,
                })
            }
            Publish::Private(why) => {
                self.trace(format!("using {} privately: {why}", tmp.display()));
                Obtained::Ready(Entry {
                    dir: tmp,
                    link_args: manifest.link_args,
                    cached: false,
                    _private: Some(guard),
                })
            }
        }
    }

    /// Build the entry into a private temporary directory for this one link.
    fn build_private(
        &self,
        material: &str,
        build: &mut dyn FnMut(&Path, bool) -> io::Result<Built>,
        why: &str,
    ) -> Obtained {
        let dir = std::env::temp_dir().join(format!("wukongc-rt-{}", unique_suffix()));
        if let Err(e) = fs::create_dir_all(&dir) {
            return Obtained::Io(format!("could not create `{}`: {e}", dir.display()));
        }
        let guard = PrivateDir(dir.clone());
        self.trace(format!("not cached ({why}); building {}", dir.display()));
        match build_into(&dir, material, false, build) {
            Ok(manifest) => Obtained::Ready(Entry {
                dir,
                link_args: manifest.link_args,
                cached: false,
                _private: Some(guard),
            }),
            Err(failure) => failure,
        }
    }

    /// Keep at most [`MAX_ENTRIES`] entries (never evicting `keep`, nor anything used within
    /// [`EVICTION_GRACE`]), and remove build directories abandoned by a process that died.
    fn evict(&self, dir: &Path, keep: &str) {
        let now = SystemTime::now();
        let Ok(children) = fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<(SystemTime, PathBuf)> = Vec::new();
        for child in children.flatten() {
            let name = child.file_name().to_string_lossy().into_owned();
            let path = child.path();
            if name.starts_with(".tmp-") {
                if age(&path, now) > STALE_TMP && fs::remove_dir_all(&path).is_ok() {
                    self.trace(format!("removed abandoned {}", path.display()));
                }
                continue;
            }
            if name == keep || !is_entry_name(&name) {
                continue;
            }
            let used = fs::metadata(path.join(MANIFEST))
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            entries.push((used, path));
        }
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (used, path) in entries.into_iter().skip(MAX_ENTRIES.saturating_sub(1)) {
            if now.duration_since(used).unwrap_or_default() < EVICTION_GRACE {
                continue;
            }
            if fs::remove_dir_all(&path).is_ok() {
                self.trace(format!("evicted {}", path.display()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("wukong-aot-cache-test-{name}-{}", unique_suffix()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A build step that writes one artifact and counts how often it ran.
    fn fake_build<'a>(
        runs: &'a std::sync::atomic::AtomicUsize,
        content: &'a [u8],
    ) -> impl FnMut(&Path, bool) -> io::Result<Built> + 'a {
        move |dir, _cached| {
            runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            fs::write(dir.join("artifact.bin"), content)?;
            Ok(Built::Ok {
                files: vec!["artifact.bin".into()],
                link_args: vec!["kernel32.lib".into(), "-lm".into()],
            })
        }
    }

    fn ready(o: Obtained) -> Entry {
        match o {
            Obtained::Ready(e) => e,
            Obtained::BuildFailed(out) => panic!("build failed: {:?}", out),
            Obtained::Spawn(e) => panic!("spawn: {e}"),
            Obtained::Io(e) => panic!("io: {e}"),
        }
    }

    #[test]
    fn the_hash_streams_and_sees_every_byte() {
        let base = hash_bytes(b"0123456789abcdef-tail");
        // Streaming in odd pieces equals one update.
        let mut h = ContentHash::new();
        for piece in [&b"012"[..], b"3456789a", b"bcdef-t", b"ail"] {
            h.update(piece);
        }
        assert_eq!(h.finish(), base);
        // Every single-byte change, and a pure length change, changes the hash.
        let src = b"0123456789abcdef-tail".to_vec();
        for i in 0..src.len() {
            let mut v = src.clone();
            v[i] ^= 1;
            assert_ne!(hash_bytes(&v), base, "byte {i}");
        }
        assert_ne!(hash_bytes(b"abc"), hash_bytes(b"abc\0"));
        assert_eq!(base.len(), 32);
    }

    #[test]
    fn keys_differ_whenever_an_input_differs() {
        let material = |flavor, rustc: &str, rlib: &str, shim: &str, recipe: &str| {
            let mut m = Material::new(flavor);
            m.line("rustc", rustc)
                .line("runtime", rlib)
                .line("shim", shim)
                .line("recipe", recipe);
            key_of(flavor, &m.finish())
        };
        let vv = "rustc 1.97.1 (8bab26f4f 2026-07-14)\nhost: x86_64-pc-windows-msvc\n";
        let base = material(Flavor::RustNative, vv, "a.rlib len=1 mtime=1.0", "s1", "r1");
        assert_eq!(
            base,
            material(Flavor::RustNative, vv, "a.rlib len=1 mtime=1.0", "s1", "r1"),
            "the key is a pure function of its inputs"
        );
        let others = [
            material(Flavor::RustLto, vv, "a.rlib len=1 mtime=1.0", "s1", "r1"),
            material(
                Flavor::RustNative,
                "rustc 1.98.0 (0 2026-08-01)\nhost: x86_64-pc-windows-msvc\n",
                "a.rlib len=1 mtime=1.0",
                "s1",
                "r1",
            ),
            material(Flavor::RustNative, vv, "b.rlib len=1 mtime=1.0", "s1", "r1"),
            material(Flavor::RustNative, vv, "a.rlib len=2 mtime=1.0", "s1", "r1"),
            material(Flavor::RustNative, vv, "a.rlib len=1 mtime=1.5", "s1", "r1"),
            material(Flavor::RustNative, vv, "a.rlib len=1 mtime=1.0", "s2", "r1"),
            material(Flavor::RustNative, vv, "a.rlib len=1 mtime=1.0", "s1", "r2"),
        ];
        for (i, other) in others.iter().enumerate() {
            assert_ne!(&base, other, "input {i} did not change the key");
        }
        assert!(base.starts_with("rust-native-") && others[0].starts_with("rust-lto-"));
    }

    #[test]
    fn a_rlib_identity_tracks_its_length_and_modification_time() {
        let dir = scratch("identity");
        let rlib = dir.join("libwukong_runtime.rlib");
        fs::write(&rlib, b"one").unwrap();
        let set = |when: u64| {
            fs::File::options()
                .write(true)
                .open(&rlib)
                .unwrap()
                .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(when))
                .unwrap()
        };
        set(1_700_000_000);
        let first = file_identity(&rlib).unwrap();
        set(1_700_000_001);
        let touched = file_identity(&rlib).unwrap();
        fs::write(&rlib, b"three").unwrap();
        set(1_700_000_001);
        let grown = file_identity(&rlib).unwrap();
        assert_ne!(first, touched);
        assert_ne!(touched, grown);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_second_obtain_hits_and_a_corrupt_entry_is_rebuilt() {
        let root = scratch("hit");
        let cache = RuntimeCache::at(&root);
        let runs = std::sync::atomic::AtomicUsize::new(0);
        let material = Material::new(Flavor::Cc).line("x", "1").finish();
        let first = ready(cache.obtain(
            Flavor::Cc,
            &material,
            &["artifact.bin"],
            &mut fake_build(&runs, b"payload"),
        ));
        assert!(first.cached);
        assert_eq!(first.link_args, ["kernel32.lib", "-lm"]);
        let dir = first.dir.clone();
        drop(first);
        assert!(
            dir.join("artifact.bin").is_file(),
            "a cached entry outlives its Entry"
        );
        let second = ready(cache.obtain(
            Flavor::Cc,
            &material,
            &["artifact.bin"],
            &mut fake_build(&runs, b"payload"),
        ));
        assert_eq!(second.dir, dir);
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the hit built nothing"
        );

        // Corruptions: same length, different bytes; truncated; missing manifest; a manifest cut
        // before its `end`; material edited. Each is detected and rebuilt into a valid entry.
        type Corruption = (&'static str, fn(&Path));
        let corruptions: [Corruption; 5] = [
            ("flipped", |d| {
                fs::write(d.join("artifact.bin"), b"PAYLOAD").unwrap()
            }),
            ("truncated", |d| {
                fs::write(d.join("artifact.bin"), b"pay").unwrap()
            }),
            ("no manifest", |d| {
                fs::remove_file(d.join(MANIFEST)).unwrap()
            }),
            ("cut manifest", |d| {
                let text = fs::read_to_string(d.join(MANIFEST)).unwrap();
                fs::write(d.join(MANIFEST), text.trim_end_matches("end\n")).unwrap()
            }),
            ("material", |d| {
                fs::write(d.join(MATERIAL), "other").unwrap()
            }),
        ];
        for (expected_runs, (what, corrupt)) in (2..).zip(corruptions) {
            corrupt(&dir);
            let rebuilt = ready(cache.obtain(
                Flavor::Cc,
                &material,
                &["artifact.bin"],
                &mut fake_build(&runs, b"payload"),
            ));
            assert_eq!(rebuilt.dir, dir, "{what}");
            assert!(rebuilt.cached, "{what}");
            assert_eq!(
                fs::read(dir.join("artifact.bin")).unwrap(),
                b"payload",
                "{what}"
            );
            assert_eq!(
                runs.load(std::sync::atomic::Ordering::SeqCst),
                expected_runs,
                "{what} was not rebuilt"
            );
        }
        // A required file the manifest does not list makes the entry invalid too.
        let runs_before = runs.load(std::sync::atomic::Ordering::SeqCst);
        let _ = ready(cache.obtain(
            Flavor::Cc,
            &material,
            &["artifact.bin", "missing.bin"],
            &mut |dir: &Path, _| {
                runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                fs::write(dir.join("artifact.bin"), b"payload")?;
                fs::write(dir.join("missing.bin"), b"now present")?;
                Ok(Built::Ok {
                    files: vec!["artifact.bin".into(), "missing.bin".into()],
                    link_args: Vec::new(),
                })
            },
        ));
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            runs_before + 1
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn concurrent_misses_both_succeed_and_leave_one_entry() {
        let root = scratch("race");
        let runs = std::sync::atomic::AtomicUsize::new(0);
        let material = Material::new(Flavor::RustNative).line("x", "race").finish();
        let barrier = std::sync::Barrier::new(4);
        let dirs: Vec<PathBuf> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    s.spawn(|| {
                        let cache = RuntimeCache::at(&root);
                        let mut build = |dir: &Path, _: bool| {
                            runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            barrier.wait(); // every thread is mid-build at once
                            fs::write(dir.join("artifact.bin"), b"same bytes")?;
                            Ok(Built::Ok {
                                files: vec!["artifact.bin".into()],
                                link_args: Vec::new(),
                            })
                        };
                        let entry = ready(cache.obtain(
                            Flavor::RustNative,
                            &material,
                            &["artifact.bin"],
                            &mut build,
                        ));
                        assert_eq!(fs::read(entry.file("artifact.bin")).unwrap(), b"same bytes");
                        entry.dir.clone()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 4);
        let published = root
            .join(RUNTIME_DIR)
            .join(key_of(Flavor::RustNative, &material));
        assert!(dirs.iter().all(|d| *d == published), "{dirs:?}");
        let left: Vec<String> = fs::read_dir(root.join(RUNTIME_DIR))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            left,
            [key_of(Flavor::RustNative, &material)],
            "no build directory is left"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn without_a_cache_the_entry_is_private_and_removed_after_use() {
        let runs = std::sync::atomic::AtomicUsize::new(0);
        let material = Material::new(Flavor::Cc).finish();
        let entry = ready(RuntimeCache::disabled().obtain(
            Flavor::Cc,
            &material,
            &["artifact.bin"],
            &mut fake_build(&runs, b"x"),
        ));
        assert!(!entry.cached);
        let dir = entry.dir.clone();
        assert!(dir.join("artifact.bin").is_file());
        drop(entry);
        assert!(!dir.exists(), "a private entry is removed with its Entry");
    }

    #[test]
    fn a_failed_build_publishes_nothing() {
        let root = scratch("failed");
        let cache = RuntimeCache::at(&root);
        let material = Material::new(Flavor::RustLto).finish();
        let failed = cache.obtain(Flavor::RustLto, &material, &[], &mut |_, _| {
            Ok(Built::Failed(Output {
                status: failing_status(),
                stdout: Vec::new(),
                stderr: b"error[E0514]".to_vec(),
            }))
        });
        assert!(matches!(failed, Obtained::BuildFailed(_)));
        let left = fs::read_dir(root.join(RUNTIME_DIR)).unwrap().count();
        assert_eq!(left, 0, "neither an entry nor a build directory remains");
        let _ = fs::remove_dir_all(&root);
    }

    fn failing_status() -> std::process::ExitStatus {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(1)
    }

    #[test]
    fn eviction_keeps_the_newest_entries_and_spares_recent_ones() {
        let root = scratch("evict");
        let dir = root.join(RUNTIME_DIR);
        fs::create_dir_all(&dir).unwrap();
        let old = SystemTime::now() - Duration::from_secs(3600);
        // MAX_ENTRIES + 2 old entries, then one recent one and a foreign directory.
        for i in 0..MAX_ENTRIES + 2 {
            let entry = dir.join(format!("cc-{i:032x}"));
            fs::create_dir_all(&entry).unwrap();
            fs::write(entry.join(MANIFEST), "x").unwrap();
            fs::File::options()
                .write(true)
                .open(entry.join(MANIFEST))
                .unwrap()
                .set_modified(old + Duration::from_secs(i as u64))
                .unwrap();
        }
        let recent = dir.join(format!("rust-lto-{:032x}", 7));
        fs::create_dir_all(&recent).unwrap();
        fs::write(recent.join(MANIFEST), "x").unwrap();
        let foreign = dir.join("keep-me");
        fs::create_dir_all(&foreign).unwrap();
        RuntimeCache::at(&root).evict(&dir, "rust-native-new");
        let mut left: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        // `keep` would be the eighth: the recent entry and the six newest old ones survive.
        let mut expected: Vec<String> = (4..MAX_ENTRIES + 2)
            .map(|i| format!("cc-{i:032x}"))
            .collect();
        expected.push("keep-me".into());
        expected.push(format!("rust-lto-{:032x}", 7));
        expected.sort();
        assert_eq!(left, expected);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_manifest_round_trips_and_rejects_escapes() {
        let m = Manifest {
            files: vec![("a b.lib".into(), 3, "00".into())],
            link_args: vec!["/defaultlib:msvcrt".into(), "-framework".into()],
        };
        let parsed = Manifest::parse(&m.render()).unwrap();
        assert_eq!(parsed.files, m.files);
        assert_eq!(parsed.link_args, m.link_args);
        let escape = format!("{HEADER}\nfile 1 00 ../x\nend\n");
        assert!(Manifest::parse(&escape).is_err());
        assert!(Manifest::parse("not a manifest\nend\n").is_err());
    }
}
