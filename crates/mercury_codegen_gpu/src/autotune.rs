//! **Phase 10 — per-(op, shape, dtype) autotuning** with an on-disk config cache and a regression mode.
//!
//! Mercury builds *several* kernels for one op at one dtype — for int8 GEMM alone: the hand-placed
//! `_smdb` (64×64 / 128×128, BK=32), the `ldmatrix`+XOR-swizzle `_swz` (64×64 / 128×128, BK=64), and the
//! split-K `_swz_sk` (sk ∈ {2,4,8}). Which wins is *shape-dependent*: swz dominates large squares, the
//! 128×128 tile wins once reuse-bound, split-K wins when a thin-M / small-N grid leaves SMs idle. A fixed
//! dispatch heuristic can only approximate this; the autotuner **measures** it per shape and **caches** the
//! winner, so the very first call pays the search and every later call is a hash lookup.
//!
//! **Honesty under contention.** The search times candidates *back-to-back, same-run* and compares them by
//! ratio (best-of-N min), exactly the methodology the int8 swz / split-K A/B benches use — the shared
//! clock cancels, so the *ranking* stays trustworthy even when absolute GFLOP/s swing (the documented
//! contention caveat). The cache means the (ideally clean-GPU) tuning happens once and is reused.
//!
//! **Correctness first (the first law).** Every candidate for an int8 shape is bit-exact (integer
//! accumulate mod 2³²), so they must all produce *byte-identical* output. The search runs the first
//! applicable candidate as the reference and asserts every other candidate matches before trusting any
//! timing — a miscompiled or mis-launched candidate fails the search, it never gets cached as a "winner".
//!
//! The cache is a tiny hand-rolled text file (no serde dependency): one line per entry,
//! `int8 <m> <n> <k> = <config> <gflops>`. `<config>` is a candidate token understood by
//! [`launch_int8_tuned`].

use crate::gpu::Gpu;
use cudarc::driver::{CudaFunction, CudaSlice, DriverError, LaunchConfig, PushKernelArg};
use std::collections::BTreeMap;
use std::time::Instant;

/// One tunable int8 GEMM candidate: a kernel (PTX + entry) plus its CTA tile, warp count, K-split factor
/// (`gridDim.z`; > 1 only for the split-K kernel), and the K-divisibility it requires (BK=32 hand-placed,
/// BK=64 swz, `sk·64` split-K). `name` is the stable token written to / read from the cache.
#[derive(Clone, Copy)]
struct Int8Cand {
    name: &'static str,
    ptx: fn() -> &'static str,
    entry: &'static str,
    bm: usize,
    bn: usize,
    warps: usize,
    sk: usize,
    k_mult: usize,
}

/// The full int8 GEMM candidate set (the kernels owned by this crate). Filtered per shape by
/// [`applicable`]. Order is the search order; ties keep the earliest (so a non-split kernel is preferred
/// to a split-K one at equal speed — fewer atomics, no C-prezero requirement).
fn int8_candidates() -> Vec<Int8Cand> {
    use crate::ptx_int8::{
        int8_gemm_smdb128_ptx, int8_gemm_smdb128_swz_ptx, int8_gemm_smdb_ptx,
        int8_gemm_smdb_swz_ptx, int8_gemm_smdb_swz_splitk_ptx, INT8_BM, INT8_BM128, INT8_BN,
        INT8_BN128, INT8_WARPS_M, INT8_WARPS_M128, INT8_WARPS_N, INT8_WARPS_N128,
    };
    let w64 = INT8_WARPS_M * INT8_WARPS_N;
    let w128 = INT8_WARPS_M128 * INT8_WARPS_N128;
    let mut v = vec![
        Int8Cand { name: "smdb64", ptx: int8_gemm_smdb_ptx, entry: "int8_gemm_nt_smdb", bm: INT8_BM, bn: INT8_BN, warps: w64, sk: 1, k_mult: 32 },
        Int8Cand { name: "smdb128", ptx: int8_gemm_smdb128_ptx, entry: "int8_gemm_nt_smdb128", bm: INT8_BM128, bn: INT8_BN128, warps: w128, sk: 1, k_mult: 32 },
        Int8Cand { name: "swz64", ptx: int8_gemm_smdb_swz_ptx, entry: "int8_gemm_nt_smdb_swz", bm: INT8_BM, bn: INT8_BN, warps: w64, sk: 1, k_mult: 64 },
        Int8Cand { name: "swz128", ptx: int8_gemm_smdb128_swz_ptx, entry: "int8_gemm_nt_smdb128_swz", bm: INT8_BM128, bn: INT8_BN128, warps: w128, sk: 1, k_mult: 64 },
    ];
    // split-K variants of the 64×64 swz kernel (one entry, gridDim.z = sk; thin-M / small-N lever).
    for sk in [2usize, 4, 8] {
        v.push(Int8Cand {
            name: match sk { 2 => "swz64_sk2", 4 => "swz64_sk4", _ => "swz64_sk8" },
            ptx: int8_gemm_smdb_swz_splitk_ptx,
            entry: "int8_gemm_nt_smdb_swz_sk",
            bm: INT8_BM,
            bn: INT8_BN,
            warps: w64,
            sk,
            k_mult: sk * 64,
        });
    }
    v
}

fn applicable(c: &Int8Cand, m: usize, n: usize, k: usize) -> bool {
    m % c.bm == 0 && n % c.bn == 0 && k % c.k_mult == 0
}

fn launch_cfg(c: &Int8Cand, m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n / c.bn) as u32, (m / c.bm) as u32, c.sk as u32),
        block_dim: ((c.warps * 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// best-of-`rounds` min over `iters` launches each (clock-warmed) — wall-clock seconds per launch. The
/// split-K kernels accumulate into `c_d` across launches, which is harmless for *timing* (the work per
/// launch is identical regardless of C's current contents).
fn best_secs(
    g: &Gpu,
    f: &CudaFunction,
    cfg: LaunchConfig,
    dims: (u32, u32, u32),
    a_d: &CudaSlice<u8>,
    b_d: &CudaSlice<i8>,
    c_d: &mut CudaSlice<i32>,
    rounds: usize,
    iters: usize,
) -> f64 {
    let (mm, nn, kk) = dims;
    let launch = |c_d: &mut CudaSlice<i32>| {
        let mut bld = g.stream.launch_builder(f);
        bld.arg(&mm).arg(&nn).arg(&kk).arg(a_d).arg(b_d).arg(c_d);
        unsafe { bld.launch(cfg).unwrap() };
    };
    launch(c_d);
    g.stream.synchronize().unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t0 = Instant::now();
        for _ in 0..iters {
            launch(c_d);
        }
        g.stream.synchronize().unwrap();
        best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
    }
    best
}

/// One candidate's measured standing.
#[derive(Clone, Debug, PartialEq)]
pub struct Ranked {
    pub name: String,
    pub secs: f64,
    pub gflops: f64,
}

/// The result of tuning one int8 GEMM shape: the winning config token and the full ranking (fastest
/// first). Every candidate was cross-checked bit-exact against the first before timing.
#[derive(Clone, Debug)]
pub struct TuneResult {
    pub best: String,
    pub ranked: Vec<Ranked>,
}

/// **Search every applicable int8 GEMM candidate for `m×n×k` and rank them by measured time.** Uploads
/// one deterministic input pair (GEMM timing is data-independent), runs each candidate once for a
/// **bit-exact cross-check** (all int8 kernels must agree byte-for-byte — a disagreement panics rather
/// than silently caching a wrong "winner"), then times each best-of-N. Returns the ranking, fastest
/// first. Panics if no candidate fits the shape (needs at least M%64==0, N%64==0, K%32==0).
pub fn tune_int8_gemm(g: &mut Gpu, m: usize, n: usize, k: usize) -> Result<TuneResult, DriverError> {
    let cands: Vec<Int8Cand> =
        int8_candidates().into_iter().filter(|c| applicable(c, m, n, k)).collect();
    assert!(
        !cands.is_empty(),
        "autotune: no int8 GEMM candidate fits {m}x{n}x{k} (need M%64==0, N%64==0, K%32==0)"
    );
    // Deterministic fill — values don't affect timing, and a fixed pattern keeps the cross-check stable.
    let a: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
    let b: Vec<i8> = (0..n * k).map(|i| ((i % 251) as i32 - 125) as i8).collect();
    let a_d = g.stream.memcpy_stod(&a)?;
    let b_d = g.stream.memcpy_stod(&b)?;
    let dims = (m as u32, n as u32, k as u32);
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    let mut reference: Option<Vec<i32>> = None;
    let mut ranked: Vec<Ranked> = Vec::with_capacity(cands.len());
    for c in &cands {
        let f = g.function(c.entry, (c.ptx)(), c.entry)?;
        let cfg = launch_cfg(c, m, n);
        // Correctness cross-check (first law): every int8 candidate is bit-exact ⇒ identical output.
        let mut cc = g.stream.memcpy_stod(&vec![0i32; m * n])?; // split-K needs a zeroed C
        {
            let mut bld = g.stream.launch_builder(&f);
            bld.arg(&dims.0).arg(&dims.1).arg(&dims.2).arg(&a_d).arg(&b_d).arg(&mut cc);
            unsafe { bld.launch(cfg)? };
        }
        let out = g.stream.memcpy_dtov(&cc)?;
        match &reference {
            None => reference = Some(out),
            Some(r) => assert!(
                &out == r,
                "autotune: int8 candidate `{}` disagrees with the reference output at {m}x{n}x{k} — not bit-exact",
                c.name
            ),
        }
        let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
        let secs = best_secs(g, &f, cfg, dims, &a_d, &b_d, &mut c_d, 6, 50);
        ranked.push(Ranked { name: c.name.to_string(), secs, gflops: flop / secs / 1e9 });
    }
    ranked.sort_by(|x, y| x.secs.partial_cmp(&y.secs).unwrap());
    Ok(TuneResult { best: ranked[0].name.clone(), ranked })
}

/// A cached winner for one (op, shape, dtype) key: the config token + the GFLOP/s it hit when tuned
/// (informational — the contention caveat means the absolute number is a snapshot, the token is durable).
#[derive(Clone, Debug, PartialEq)]
pub struct CacheEntry {
    pub config: String,
    pub gflops: f64,
}

/// On-disk per-shape autotune cache. Keyed by `"<dtype> <m> <n> <k>"`; serialized as one
/// `<key> = <config> <gflops>` line each (no serde dependency).
#[derive(Clone, Debug, Default)]
pub struct AutotuneCache {
    map: BTreeMap<String, CacheEntry>,
}

impl AutotuneCache {
    pub fn new() -> Self {
        Self { map: BTreeMap::new() }
    }

    fn int8_key(m: usize, n: usize, k: usize) -> String {
        format!("int8 {m} {n} {k}")
    }

    pub fn get_int8(&self, m: usize, n: usize, k: usize) -> Option<&CacheEntry> {
        self.map.get(&Self::int8_key(m, n, k))
    }

    pub fn insert_int8(&mut self, m: usize, n: usize, k: usize, entry: CacheEntry) {
        self.map.insert(Self::int8_key(m, n, k), entry);
    }

    fn w4a16_key(m: usize, n: usize, k: usize) -> String {
        format!("w4a16 {m} {n} {k}")
    }

    pub fn get_w4a16(&self, m: usize, n: usize, k: usize) -> Option<&CacheEntry> {
        self.map.get(&Self::w4a16_key(m, n, k))
    }

    pub fn insert_w4a16(&mut self, m: usize, n: usize, k: usize, entry: CacheEntry) {
        self.map.insert(Self::w4a16_key(m, n, k), entry);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Serialize to the line-oriented text format.
    pub fn to_text(&self) -> String {
        let mut s = String::from("# mercury autotune cache: <dtype> <m> <n> <k> = <config> <gflops>\n");
        for (key, e) in &self.map {
            s += &format!("{key} = {} {:.1}\n", e.config, e.gflops);
        }
        s
    }

    /// Parse the text format. Malformed lines (and `#` comments / blanks) are skipped, so a partially
    /// corrupt cache degrades to "fewer entries" rather than an error — the autotuner just re-tunes them.
    pub fn from_text(text: &str) -> Self {
        let mut map = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, val)) = line.split_once('=') else { continue };
            let key = key.trim();
            // key must be `<dtype> <m> <n> <k>` (4 tokens); val must be `<config> <gflops>`.
            if key.split_whitespace().count() != 4 {
                continue;
            }
            let mut vt = val.split_whitespace();
            let (Some(config), Some(gf)) = (vt.next(), vt.next()) else { continue };
            let gflops = gf.parse::<f64>().unwrap_or(0.0);
            map.insert(key.to_string(), CacheEntry { config: config.to_string(), gflops });
        }
        Self { map }
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_text())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        Ok(Self::from_text(&std::fs::read_to_string(path)?))
    }
}

/// Look up the tuned config for `m×n×k`, tuning + caching it on a miss. Returns the config token.
pub fn tune_int8_cached(
    g: &mut Gpu,
    cache: &mut AutotuneCache,
    m: usize,
    n: usize,
    k: usize,
) -> Result<String, DriverError> {
    if let Some(e) = cache.get_int8(m, n, k) {
        return Ok(e.config.clone());
    }
    let r = tune_int8_gemm(g, m, n, k)?;
    cache.insert_int8(m, n, k, CacheEntry { config: r.best.clone(), gflops: r.ranked[0].gflops });
    Ok(r.best)
}

/// A detected staleness: the cached config is no longer the fastest by more than the threshold margin.
#[derive(Clone, Debug, PartialEq)]
pub struct Regression {
    pub cached: String,
    pub current_best: String,
    /// How much faster the current best is than the cached config (`cached_secs / best_secs`).
    pub speedup_available: f64,
}

/// Pure regression decision: given the cached config's current time and the current best, flag a
/// regression iff a *different* config is now faster than the cached one by more than `threshold`
/// (e.g. 1.10 = 10%). Extracted from device timing so it is unit-tested in both directions without a GPU
/// (and so a within-noise reshuffle never trips a false regression).
fn regression_decision(
    cached: &str,
    cached_secs: f64,
    best: &Ranked,
    threshold: f64,
) -> Option<Regression> {
    if best.name != cached && cached_secs / best.secs > threshold {
        Some(Regression {
            cached: cached.to_string(),
            current_best: best.name.clone(),
            speedup_available: cached_secs / best.secs,
        })
    } else {
        None
    }
}

/// **Regression mode.** Re-tune `m×n×k` and report whether the cached config has gone stale — i.e. a
/// different candidate is now > 10% faster (kernels changed, a driver update shifted the balance, etc.).
/// Returns `Ok(None)` when the shape isn't cached or the cached config is still (within 10% of) the best.
pub fn revalidate_int8(
    g: &mut Gpu,
    cache: &AutotuneCache,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Option<Regression>, DriverError> {
    let Some(e) = cache.get_int8(m, n, k) else { return Ok(None) };
    let r = tune_int8_gemm(g, m, n, k)?;
    let cached_secs = r.ranked.iter().find(|x| x.name == e.config).map(|x| x.secs);
    Ok(match cached_secs {
        Some(cs) => regression_decision(&e.config, cs, &r.ranked[0], 1.10),
        None => None, // cached config no longer applicable at this shape — caller should re-tune
    })
}

/// **Run the int8 GEMM with the autotuned kernel** `C = A·Bᵀ` (u8×i8→i32). Looks up (or tunes + caches)
/// the best config for the shape, then launches it. The chosen kernel is bit-exact-equivalent to every
/// other int8 candidate, so the result equals [`crate::gpu::gemm_nt_int8`] regardless of which won.
pub fn launch_int8_tuned(
    g: &mut Gpu,
    cache: &mut AutotuneCache,
    a: &[u8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<i32>, DriverError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let name = tune_int8_cached(g, cache, m, n, k)?;
    let cand = int8_candidates()
        .into_iter()
        .find(|c| c.name == name)
        .expect("cached int8 config token must name a known candidate");
    let f = g.function(cand.entry, (cand.ptx)(), cand.entry)?;
    let cfg = launch_cfg(&cand, m, n);
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?; // split-K accumulates → C must start zeroed
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

// ============================ W4A16 (int4 decode) autotuning ============================
// The int4 decode path's tunable axis is the **split-K count** (sk): sk=1 is the un-split kernel, sk>1 is
// the gridDim.z=sk split-K GEMM + the fixed-order reduction. Which wins is shape-dependent (split-K wins
// hugely — up to ~6.4× — when a thin-M/small-N grid starves the SMs, marginal when the grid saturates),
// so it's exactly an autotune axis. Unlike int8 the candidates aren't bit-exact (fp16 accumulate), so the
// search cross-checks within the fp16 tolerance instead of `==`.

const W4A16_SK_CANDS: [usize; 4] = [1, 2, 4, 8];

fn w4a16_token(sk: usize) -> String {
    if sk == 1 {
        "w4a16".to_string()
    } else {
        format!("w4a16_sk{sk}")
    }
}

/// Parse a W4A16 config token to its split count (`"w4a16"` → 1, `"w4a16_skN"` → N).
fn w4a16_sk_of(token: &str) -> usize {
    token.strip_prefix("w4a16_sk").and_then(|s| s.parse().ok()).unwrap_or(1)
}

/// **Search the W4A16 split counts for `m×n×k` and rank them.** sk=1 (un-split) vs split-K (sk∈{2,4,8}:
/// `gridDim.z=sk` GEMM + the fixed-order reduction kernel, both timed). Uploads inputs once; runs each
/// candidate once for a **tolerance cross-check** against the un-split output (all are the same W4A16 math,
/// differing only by the f32 reduction order — a candidate outside fp16 tolerance panics, never cached),
/// then times each best-of-N. Symmetric (no zero-point) weights. Fastest first.
pub fn tune_w4a16_gemm(
    g: &mut Gpu,
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<TuneResult, DriverError> {
    use crate::ptx_int4::{GROUP_SIZE, W4_BM, W4_BN, W4_THREADS};
    use half::f16;
    assert_eq!(qw.n, n);
    assert_eq!(qw.k, k);
    assert_eq!(qw.group, GROUP_SIZE);
    assert!(qw.zeros.is_none(), "w4a16 autotune is the symmetric split-K path");
    assert!(
        m % W4_BM == 0 && n % W4_BN == 0 && k % GROUP_SIZE == 0,
        "w4a16 tune needs M%{W4_BM}==0, N%{W4_BN}==0, K%{GROUP_SIZE}==0"
    );
    let a: Vec<f16> = (0..m * k).map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 8.0)).collect();
    let a_d = g.stream.memcpy_stod(&a)?;
    let bq_d = g.stream.memcpy_stod(&qw.packed)?;
    let scl_d = g.stream.memcpy_stod(&qw.scales)?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    let f_base = g.function("w4a16", crate::ptx_int4::w4a16_ptx(), "gemm_nt_w4a16")?;
    let f_sk = g.function("w4a16_sk", crate::ptx_int4::w4a16_splitk_ptx(), "gemm_nt_w4a16_sk")?;
    let f_red = g.function("w4a16_sk", crate::ptx_int4::w4a16_splitk_ptx(), "w4a16_splitk_reduce")?;
    let cfg1 = LaunchConfig { grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, 1), block_dim: (W4_THREADS as u32, 1, 1), shared_mem_bytes: 0 };
    let rcfg = LaunchConfig { grid_dim: (256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let mut reference: Option<Vec<f32>> = None;
    let mut ranked: Vec<Ranked> = Vec::new();
    for &sk in W4A16_SK_CANDS.iter().filter(|&&sk| k % (sk * GROUP_SIZE) == 0) {
        let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
        let (out, secs) = if sk == 1 {
            {
                let mut b = g.stream.launch_builder(&f_base);
                b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d);
                unsafe { b.launch(cfg1)? };
            }
            let out = g.stream.memcpy_dtov(&c_d)?;
            let mut s = f64::INFINITY;
            for _ in 0..6 {
                let t0 = Instant::now();
                for _ in 0..50 {
                    let mut b = g.stream.launch_builder(&f_base);
                    b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d);
                    unsafe { b.launch(cfg1)? };
                }
                g.stream.synchronize().unwrap();
                s = s.min(t0.elapsed().as_secs_f64() / 50.0);
            }
            (out, s)
        } else {
            let mut part_d = g.stream.memcpy_stod(&vec![0f32; sk * m * n])?;
            let cfg_sk = LaunchConfig { grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, sk as u32), block_dim: (W4_THREADS as u32, 1, 1), shared_mem_bytes: 0 };
            let (mnp, skk) = ((m * n) as u32, sk as u32);
            {
                let mut b = g.stream.launch_builder(&f_sk);
                b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut part_d);
                unsafe { b.launch(cfg_sk)? };
            }
            {
                let mut b = g.stream.launch_builder(&f_red);
                b.arg(&mnp).arg(&skk).arg(&part_d).arg(&mut c_d);
                unsafe { b.launch(rcfg)? };
            }
            let out = g.stream.memcpy_dtov(&c_d)?;
            let mut s = f64::INFINITY;
            for _ in 0..6 {
                let t0 = Instant::now();
                for _ in 0..50 {
                    {
                        let mut b = g.stream.launch_builder(&f_sk);
                        b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut part_d);
                        unsafe { b.launch(cfg_sk)? };
                    }
                    {
                        let mut b = g.stream.launch_builder(&f_red);
                        b.arg(&mnp).arg(&skk).arg(&part_d).arg(&mut c_d);
                        unsafe { b.launch(rcfg)? };
                    }
                }
                g.stream.synchronize().unwrap();
                s = s.min(t0.elapsed().as_secs_f64() / 50.0);
            }
            (out, s)
        };
        // tolerance cross-check vs the un-split reference (same products, only the f32 reduction differs).
        match &reference {
            None => reference = Some(out),
            Some(r) => {
                let bad = out.iter().zip(r).any(|(&x, &y)| (x - y).abs() > 1e-2 + 2e-3 * y.abs());
                assert!(!bad, "autotune w4a16: sk={sk} disagrees with the un-split output beyond fp16 tolerance at {m}x{n}x{k}");
            }
        }
        ranked.push(Ranked { name: w4a16_token(sk), secs, gflops: flop / secs / 1e9 });
    }
    ranked.sort_by(|x, y| x.secs.partial_cmp(&y.secs).unwrap());
    Ok(TuneResult { best: ranked[0].name.clone(), ranked })
}

/// Look up the tuned W4A16 split count for `m×n×k`, tuning + caching on a miss. Returns the config token.
pub fn tune_w4a16_cached(
    g: &mut Gpu,
    cache: &mut AutotuneCache,
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<String, DriverError> {
    if let Some(e) = cache.get_w4a16(m, n, k) {
        return Ok(e.config.clone());
    }
    let r = tune_w4a16_gemm(g, qw, m, k, n)?;
    cache.insert_w4a16(m, n, k, CacheEntry { config: r.best.clone(), gflops: r.ranked[0].gflops });
    Ok(r.best)
}

/// **Run W4A16 with the autotuned split count** — looks up (or tunes + caches) the best `sk` for the
/// shape, then calls the matching launcher ([`crate::gpu::gemm_nt_w4a16`] for sk=1, else
/// [`crate::gpu::gemm_nt_w4a16_splitk`]). Numerically equals the un-split kernel within fp16 tolerance.
pub fn launch_w4a16_tuned(
    g: &mut Gpu,
    cache: &mut AutotuneCache,
    a: &[f32],
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    let token = tune_w4a16_cached(g, cache, qw, m, k, n)?;
    let sk = w4a16_sk_of(&token);
    if sk == 1 {
        crate::gpu::gemm_nt_w4a16(g, a, qw, m, k, n)
    } else {
        crate::gpu::gemm_nt_w4a16_splitk(g, a, qw, m, k, n, sk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU i32 reference for `C = A·Bᵀ` (u8×i8→i32, wrapping) — the bit-exact oracle for the tuned launch.
    fn ref_nt_int8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for t in 0..k {
                    acc = acc.wrapping_add(a[i * k + t] as i32 * b[j * k + t] as i32);
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// **Cache text round-trips (no GPU).** Insert a few entries, serialize, parse back — the map must be
    /// identical. Also: comment/blank/malformed lines are skipped (a partially corrupt cache degrades to
    /// fewer entries, never an error).
    #[test]
    fn cache_text_roundtrip() {
        let mut c = AutotuneCache::new();
        c.insert_int8(1024, 1024, 1024, CacheEntry { config: "swz64".into(), gflops: 12345.6 });
        c.insert_int8(64, 128, 8192, CacheEntry { config: "swz64_sk8".into(), gflops: 6948.0 });
        c.insert_int8(4096, 4096, 4096, CacheEntry { config: "swz128".into(), gflops: 50570.0 });
        let back = AutotuneCache::from_text(&c.to_text());
        assert_eq!(back.len(), 3);
        assert_eq!(back.get_int8(1024, 1024, 1024).unwrap().config, "swz64");
        assert_eq!(back.get_int8(64, 128, 8192).unwrap().config, "swz64_sk8");
        assert_eq!(back.get_int8(4096, 4096, 4096).unwrap().config, "swz128");
        // tolerant parsing: junk lines are dropped, valid ones survive.
        let parsed = AutotuneCache::from_text(
            "# header\n\nint8 256 256 256 = swz64 999.9\ngarbage line\nint8 1 2 = bad\n",
        );
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed.get_int8(256, 256, 256).unwrap().config, "swz64");
    }

    /// **Regression decision logic (no GPU).** Both directions, deterministic: a clearly-faster different
    /// config flags; the cached config still being best (or a within-threshold reshuffle) does not.
    #[test]
    fn regression_decision_both_ways() {
        let best = Ranked { name: "swz64_sk8".into(), secs: 1.0e-4, gflops: 0.0 };
        // cached is 2× slower than the new best → flag.
        let r = regression_decision("smdb64", 2.0e-4, &best, 1.10);
        assert_eq!(r.unwrap().current_best, "swz64_sk8");
        // cached IS the best → no regression.
        assert!(regression_decision("swz64_sk8", 1.0e-4, &best, 1.10).is_none());
        // a different config but only 5% slower (< 10% threshold) → within noise, no flag.
        assert!(regression_decision("swz64", 1.05e-4, &best, 1.10).is_none());
    }

    /// **GPU: the search picks a bit-exact winner and the tuned launch is correct.** Skips (never fails)
    /// when no CUDA device is reachable. Tunes a few shapes (a large square, a thin-M/large-K decode-ish
    /// shape, a 128-divisible one), asserts the winner is a known token, the ranking is complete, and
    /// `launch_int8_tuned` equals the CPU i32 reference. Then exercises the cache: a second call is a
    /// hit (no re-tune), and the cached-config launch still matches.
    #[test]
    fn tune_and_launch_int8_on_device() {
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            eprintln!("[skip] tune_and_launch_int8_on_device: no CUDA device reachable");
            return;
        };
        let n_cands = int8_candidates().len();
        let mut cache = AutotuneCache::new();
        for (m, n, k) in [(256usize, 256usize, 256usize), (64, 128, 8192), (128, 128, 256)] {
            let r = tune_int8_gemm(g, m, n, k).unwrap();
            assert!(!r.ranked.is_empty(), "ranking must be non-empty for {m}x{n}x{k}");
            assert!(r.ranked.len() <= n_cands);
            assert!(int8_candidates().iter().any(|c| c.name == r.best), "winner `{}` must be a known candidate", r.best);
            // tuned launch == CPU i32 oracle (the chosen kernel is bit-exact like every candidate).
            let a: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
            let b: Vec<i8> = (0..n * k).map(|i| ((i % 251) as i32 - 125) as i8).collect();
            let want = ref_nt_int8(&a, &b, m, k, n);
            let got = launch_int8_tuned(g, &mut cache, &a, &b, m, n, k).unwrap();
            assert_eq!(got, want, "tuned int8 launch {m}x{n}x{k} must equal the i32 reference");
            eprintln!("[autotune] {m}x{n}x{k}: best = {} ({:.0} GFLOP/s); ranked {}", r.best, r.ranked[0].gflops, r.ranked.len());
        }
        // cache is populated; a repeat tune-or-lookup is a hit (config token stable).
        let before = cache.len();
        let _ = tune_int8_cached(g, &mut cache, 256, 256, 256).unwrap();
        assert_eq!(cache.len(), before, "a cached shape must not grow the cache");
        // round-trip the populated cache through text and confirm a known entry survives.
        let reloaded = AutotuneCache::from_text(&cache.to_text());
        assert_eq!(reloaded.get_int8(64, 128, 8192).map(|e| e.config.clone()), cache.get_int8(64, 128, 8192).map(|e| e.config.clone()));
        // revalidate the freshly-tuned shape: the cached config was just measured best → no regression.
        assert!(revalidate_int8(g, &cache, 64, 128, 8192).unwrap().is_none());
        eprintln!("[gate] autotune int8: search bit-exact + tuned launch correct + cache round-trip + no false regression ✓");
    }

    /// **GPU: the W4A16 split-K search picks a valid config and the tuned launch is correct.** Skips
    /// without a device. Tunes decode-like shapes (small M, large K), asserts the winner is a `w4a16`
    /// token and the tuned launch matches the f64 dequant reference within fp16 tolerance; both shapes
    /// land in the cache and round-trip through text. (Operationalizes the up-to-6.4× int4 split-K decode
    /// win — the autotuner now picks `sk` per shape automatically.)
    #[test]
    fn tune_and_launch_w4a16_on_device() {
        use crate::ptx_int4::{quantize_weight_symmetric, reference_w4a16, GROUP_SIZE};
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            eprintln!("[skip] tune_and_launch_w4a16_on_device: no CUDA device reachable");
            return;
        };
        let mut rng = crate::diff::Rng::new(0x4A07);
        let mut cache = AutotuneCache::new();
        for (m, n, k) in [(64usize, 256usize, 1024usize), (128, 128, 2048)] {
            let a = rng.vec(m * k, -1.0, 1.0);
            let w = rng.vec(n * k, -0.8, 0.8);
            let qw = quantize_weight_symmetric(&w, n, k, GROUP_SIZE);
            let r = tune_w4a16_gemm(g, &qw, m, k, n).unwrap();
            assert!(!r.ranked.is_empty(), "w4a16 ranking must be non-empty for {m}x{n}x{k}");
            assert!(r.best == "w4a16" || r.best.starts_with("w4a16_sk"), "best `{}` must be a w4a16 token", r.best);
            let want = reference_w4a16(&a, &qw, m);
            let got = launch_w4a16_tuned(g, &mut cache, &a, &qw, m, k, n).unwrap();
            let s = crate::diff::assert_close(&format!("w4a16 tuned {m}x{n}x{k}"), &got, &want, 1e-2, 2e-3);
            eprintln!("[autotune] w4a16 {m}x{n}x{k}: best = {} ({:.0} GFLOP/s); max_abs={:.1e}", r.best, r.ranked[0].gflops, s.max_abs);
        }
        assert_eq!(cache.len(), 2, "both tuned w4a16 shapes should be cached");
        let reloaded = AutotuneCache::from_text(&cache.to_text());
        assert_eq!(
            reloaded.get_w4a16(64, 256, 1024).map(|e| e.config.clone()),
            cache.get_w4a16(64, 256, 1024).map(|e| e.config.clone())
        );
        eprintln!("[gate] autotune w4a16: search tolerance-checked + tuned launch correct + cache round-trip ✓");
    }
}
