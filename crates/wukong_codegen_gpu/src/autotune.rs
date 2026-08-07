//! **Phase 10 — per-(op, shape, dtype) autotuning** with an on-disk config cache and a regression mode.
//!
//! Wukong builds *several* kernels for one op at one dtype — for int8 GEMM alone: the hand-placed
//! `_smdb` (64×64 / 128×128, BK=32), the `ldmatrix`+XOR-swizzle `_swz` (64×64 / 128×128, BK=64), the
//! split-K `_swz_sk` (sk ∈ {2,4,8}), and the **variable-stage** `w64_s{3,4,5}` rings. Which wins is
//! *shape-dependent*: swz dominates large squares, the 128×128 tile wins once reuse-bound, split-K wins
//! when a thin-M / small-N grid leaves SMs idle. A fixed dispatch heuristic can only approximate this;
//! the autotuner **measures** it per shape and **caches** the winner, so the very first call pays the
//! search and every later call is a hash lookup.
//!
//! **The candidate SET is device-dependent, not just the winner.** `w64_s4`/`_s5` need 64/80 KiB of
//! shared memory per block — more than the PTX ISA lets any kernel declare statically — so they exist
//! only through the dynamic-SMEM window and only on a card whose `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`
//! covers them (Ada 99 KiB, A100 163, H100 227, Turing 64). [`applicable`] therefore takes the probed
//! budget and declines a row the running device cannot host, instead of letting it reach
//! `cuFuncSetAttribute` and fail there. This is also the axis where a hardcoded verdict would age
//! worst: pipeline depth *lost* on the 20-SM laptop that cuts its occupancy 3→1 CTAs/SM and is
//! predicted to pay on A100 where the same ring is occupancy-free — so it is precisely a thing to
//! measure per machine and key by device, which the cache already does.
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
//! Two op families are tuned: the int8 GEMM candidate set above, and **W4A16** (int4 weight-only
//! decode), whose search is over the split-K count `sk ∈ {1,2,4,8}` — cross-checked against the
//! un-split output within fp16 tolerance rather than bit-exactly, since split-K reassociates the f16
//! accumulation.
//!
//! The cache is a tiny hand-rolled text file (no serde dependency): one line per entry,
//! `<dtype> <device> <m> <n> <k> = <config> <gflops>`, where `<dtype>` is `int8` or `w4a16` and
//! `<device>` is [`crate::gpu::Gpu::device_tag`] (`sm_89x20` — arch + SM count). `<config>` is a
//! candidate token understood by [`launch_int8_tuned`] / [`launch_w4a16_tuned`]. It outlives the
//! candidate set *and* the machine, so a hit is trusted only when it is **applicable at this shape**
//! (`int8_token_usable` / `w4a16_token_usable`) **and keyed to this device** — an unknown token, a
//! mis-tiling one, or one measured on another card is a miss and re-tunes.

use crate::gpu::Gpu;
use cudarc::driver::{CudaFunction, CudaSlice, DriverError, LaunchConfig, PushKernelArg};
use std::collections::BTreeMap;
use std::time::Instant;

/// Where a candidate's PTX comes from.
///
/// The shipped kernels are `&'static` modules built once. The **variable-stage** rows are generated
/// against the running device's shared-memory budget instead ([`crate::gpu::Gpu::smem_budget`]), because
/// past the PTX ISA's 48 KiB static cap the ring has to live in the `.extern .shared` window and the
/// budget is what decides whether that is legal here at all.
#[derive(Clone, Copy)]
enum Int8Src {
    /// A shipped module: one `&'static` PTX text, static SMEM, launched with `shared_mem_bytes: 0`.
    Fixed(fn() -> &'static str),
    /// A [`crate::ptx_int8::INT8_STAGE_VARIANTS`] row — generated per device budget, and loaded through
    /// `Gpu::function_smem` so a >48 KiB row gets its `cuFuncSetAttribute` opt-in and its launch window.
    Stage(&'static crate::ptx_int8::Int8StageCfg),
}

/// One tunable int8 GEMM candidate: a kernel (PTX + entry) plus its CTA tile, warp count, K-split factor
/// (`gridDim.z`; > 1 only for the split-K kernel), and the K-divisibility it requires (BK=32 hand-placed,
/// BK=64 swz, `sk·64` split-K). `name` is the stable token written to / read from the cache.
#[derive(Clone, Copy)]
struct Int8Cand {
    name: &'static str,
    src: Int8Src,
    entry: &'static str,
    bm: usize,
    bn: usize,
    warps: usize,
    sk: usize,
    k_mult: usize,
    /// Threadblock-rasterization band width (0 = none / 2-D grid). When > 0 the kernel uses a 1-D CTA
    /// grid (`gridDim.x = tiles_m·tiles_n`) — mutually exclusive with split-K (which uses `gridDim.z`).
    raster: usize,
    /// Shared memory per CTA, bytes. `0` for the shipped kernels (all static and all ≤ 48 KiB by
    /// construction); the real footprint for a variable-stage row, which is what [`applicable`] tests
    /// against the device's opt-in ceiling.
    smem: usize,
    /// Shortest K at which this candidate's pipeline can fill — `(stages-1)·64` for a deep ring, `0`
    /// for the shipped 2-stage kernels. Below it the extra buffers never fill (D6 §3.2-4: the prologue
    /// guards them, so it is *correct*, just pure waste), and a tuner should not spend a round on it.
    min_k: usize,
}

/// The full int8 GEMM candidate set (the kernels owned by this crate). Filtered per shape by
/// [`applicable`]. Order is the search order; ties keep the earliest (so a non-split kernel is preferred
/// to a split-K one at equal speed — fewer atomics, no C-prezero requirement).
fn int8_candidates() -> Vec<Int8Cand> {
    use crate::ptx_int8::{
        int8_gemm_smdb128_ptx, int8_gemm_smdb128_swz_ptx, int8_gemm_smdb_ptx,
        int8_gemm_smdb_swz_ptx, int8_gemm_smdb_swz_splitk_ptx, int8_gemm_w64_swz_ptx,
        int8_gemm_w64_swz_r8_ptx, INT8_BM, INT8_BM128, INT8_BN, INT8_BN128, INT8_STAGE_VARIANTS,
        INT8_W64_BM, INT8_W64_BN, INT8_W64_WARPS_M, INT8_W64_WARPS_N, INT8_WARPS_M,
        INT8_WARPS_M128, INT8_WARPS_N, INT8_WARPS_N128,
    };
    let w64 = INT8_WARPS_M * INT8_WARPS_N;
    let w128 = INT8_WARPS_M128 * INT8_WARPS_N128;
    let ww64 = INT8_W64_WARPS_M * INT8_W64_WARPS_N;
    let mut v = vec![
        Int8Cand {
            name: "smdb64",
            src: Int8Src::Fixed(int8_gemm_smdb_ptx),
            entry: "int8_gemm_nt_smdb",
            bm: INT8_BM,
            bn: INT8_BN,
            warps: w64,
            sk: 1,
            k_mult: 32,
            raster: 0,
            smem: 0,
            min_k: 0,
        },
        Int8Cand {
            name: "smdb128",
            src: Int8Src::Fixed(int8_gemm_smdb128_ptx),
            entry: "int8_gemm_nt_smdb128",
            bm: INT8_BM128,
            bn: INT8_BN128,
            warps: w128,
            sk: 1,
            k_mult: 32,
            raster: 0,
            smem: 0,
            min_k: 0,
        },
        Int8Cand {
            name: "swz64",
            src: Int8Src::Fixed(int8_gemm_smdb_swz_ptx),
            entry: "int8_gemm_nt_smdb_swz",
            bm: INT8_BM,
            bn: INT8_BN,
            warps: w64,
            sk: 1,
            k_mult: 64,
            raster: 0,
            smem: 0,
            min_k: 0,
        },
        Int8Cand {
            name: "swz128",
            src: Int8Src::Fixed(int8_gemm_smdb128_swz_ptx),
            entry: "int8_gemm_nt_smdb128_swz",
            bm: INT8_BM128,
            bn: INT8_BN128,
            warps: w128,
            sk: 1,
            k_mult: 64,
            raster: 0,
            smem: 0,
            min_k: 0,
        },
        // The 64×64-warp-tile workhorse (128×128 CTA, 4 warps) — the perf/gpu-quant-2 winner (~1.2–1.3×
        // the 8-warp swz128 same-run; 2048³→92%, +raster8→99.6% of cuBLAS) — and its rasterized sibling.
        Int8Cand {
            name: "w64",
            src: Int8Src::Fixed(int8_gemm_w64_swz_ptx),
            entry: "int8_gemm_nt_w64_swz",
            bm: INT8_W64_BM,
            bn: INT8_W64_BN,
            warps: ww64,
            sk: 1,
            k_mult: 64,
            raster: 0,
            smem: 0,
            min_k: 0,
        },
        Int8Cand {
            name: "w64_r8",
            src: Int8Src::Fixed(int8_gemm_w64_swz_r8_ptx),
            entry: "int8_gemm_nt_w64_swz_r8",
            bm: INT8_W64_BM,
            bn: INT8_W64_BN,
            warps: ww64,
            sk: 1,
            k_mult: 64,
            raster: 8,
            smem: 0,
            min_k: 0,
        },
    ];
    // **The variable-stage rows** (`INT8_STAGE_VARIANTS`, same 128×128 CTA / 64×64 warp tile as `w64`,
    // deeper `cp.async` ring). s2 is byte-identical to `w64` above, so it is skipped rather than timed
    // twice; s3 is 48 KiB static; s4/s5 (64/80 KiB) exist only through the dynamic-SMEM window and are
    // the reason this is a search axis at all. Whether depth pays is a *device* question — it lost at
    // every size on the 20-SM Ada laptop that cut 3 CTAs/SM to 2 for it, and D3 predicts it flips on
    // A100 where the same ring is occupancy-free — which is exactly why the tuner, not a hardcoded
    // heuristic, should decide: the cache is device-keyed, so each part gets its own verdict.
    for cfg in INT8_STAGE_VARIANTS.iter().filter(|c| c.stages > 2) {
        v.push(Int8Cand {
            name: match cfg.stages {
                3 => "w64_s3",
                4 => "w64_s4",
                _ => "w64_s5",
            },
            src: Int8Src::Stage(cfg),
            entry: cfg.name,
            bm: cfg.bm,
            bn: cfg.bn,
            warps: cfg.threads() / 32,
            sk: 1,
            k_mult: 64,
            raster: 0,
            smem: cfg.smem_bytes(),
            min_k: cfg.min_k(),
        });
    }
    // split-K variants of the 64×64 swz kernel (one entry, gridDim.z = sk; thin-M / small-N lever).
    for sk in [2usize, 4, 8] {
        v.push(Int8Cand {
            name: match sk {
                2 => "swz64_sk2",
                4 => "swz64_sk4",
                _ => "swz64_sk8",
            },
            src: Int8Src::Fixed(int8_gemm_smdb_swz_splitk_ptx),
            entry: "int8_gemm_nt_smdb_swz_sk",
            bm: INT8_BM,
            bn: INT8_BN,
            warps: w64,
            sk,
            k_mult: sk * 64,
            raster: 0,
            smem: 0,
            min_k: 0,
        });
    }
    v
}

/// Can this candidate run this shape **on this device**?
///
/// Two independent facts, and the second one is new: a shape fact (the tile must divide M/N and the
/// K-slab must divide K, else the grid is short and part of C is never written) **and a capability
/// fact** — a deep ring's `smem` must fit `budget`, the device's `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
/// The budget is a per-card number (Ada 99 KiB, A100 163, H100 227, Turing 64), so the candidate SET
/// itself is device-dependent, not just the winner. Declining here is what keeps a too-deep row out of
/// the search entirely, rather than letting it reach `function_dyn`'s assert or the driver's own
/// unhelpful `CUDA_ERROR_INVALID_VALUE`. `min_k` additionally drops depths whose buffers cannot fill at
/// this K — correct but pure waste, and a wasted round is a worse measurement for everything else.
fn applicable(c: &Int8Cand, m: usize, n: usize, k: usize, budget: usize) -> bool {
    m % c.bm == 0 && n % c.bn == 0 && k % c.k_mult == 0 && k >= c.min_k && c.smem <= budget
}

/// Load a candidate, returning its function and **the shared-memory bytes its launch must carry** — 0
/// for a static kernel, the window size for a dynamic one. The module-cache key is the candidate's own
/// entry name, which embeds (tile, warps, stages): `Gpu::function` never re-examines PTX on a key hit,
/// so two depths sharing a key would silently run the first one's kernel *and* inherit its SMEM ceiling.
fn load(g: &mut Gpu, c: &Int8Cand) -> Result<(CudaFunction, usize), DriverError> {
    match c.src {
        Int8Src::Fixed(f) => Ok((g.function(c.entry, f(), c.entry)?, 0)),
        Int8Src::Stage(cfg) => {
            let (ptx, mode) = crate::ptx_int8::int8_stage_ptx(cfg, g.smem_budget());
            g.function_smem(cfg.name, &ptx, cfg.name, mode)
        }
    }
}

fn launch_cfg(c: &Int8Cand, m: usize, n: usize, dyn_smem: usize) -> LaunchConfig {
    // The grid is `m/bm × n/bn` tiles — integer division, so a shape the candidate does not tile
    // exactly would silently launch a SHORT grid and leave the trailing rows/columns of C untouched
    // (whatever the caller pre-filled, typically zeros). Reject it here rather than return a partial
    // answer: `applicable` is the contract, and every caller must have checked it.
    assert!(
        m % c.bm == 0 && n % c.bn == 0,
        "autotune: candidate `{}` tiles {}x{} and cannot cover M={m}, N={n} — a truncated grid \
         would leave part of C unwritten",
        c.name,
        c.bm,
        c.bn
    );
    // Rasterized kernels take a 1-D CTA grid (gridDim.x = tiles_m·tiles_n); all others a 2-D grid with
    // the K-split factor in gridDim.z (sk==1 for the non-split kernels).
    let grid_dim = if c.raster > 0 {
        (((m / c.bm) * (n / c.bn)) as u32, 1, 1)
    } else {
        ((n / c.bn) as u32, (m / c.bm) as u32, c.sk as u32)
    };
    // `shared_mem_bytes` is memory allocated ON TOP of the entry's statics, so a static candidate must
    // pass 0 — its own size again would allocate the tile twice and silently halve residency. The value
    // comes from `load`, i.e. from the generator's own `SmemMode`; it is never re-derived here.
    crate::gpu::dyn_launch_cfg(grid_dim, ((c.warps * 32) as u32, 1, 1), dyn_smem)
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
pub fn tune_int8_gemm(
    g: &mut Gpu,
    m: usize,
    n: usize,
    k: usize,
) -> Result<TuneResult, DriverError> {
    let budget = g.smem_budget();
    let cands: Vec<Int8Cand> = int8_candidates()
        .into_iter()
        .filter(|c| applicable(c, m, n, k, budget))
        .collect();
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
        let (f, dyn_smem) = load(g, c)?;
        let cfg = launch_cfg(c, m, n, dyn_smem);
        // Correctness cross-check (first law): every int8 candidate is bit-exact ⇒ identical output.
        let mut cc = g.stream.memcpy_stod(&vec![0i32; m * n])?; // split-K needs a zeroed C
        {
            let mut bld = g.stream.launch_builder(&f);
            bld.arg(&dims.0)
                .arg(&dims.1)
                .arg(&dims.2)
                .arg(&a_d)
                .arg(&b_d)
                .arg(&mut cc);
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
        ranked.push(Ranked {
            name: c.name.to_string(),
            secs,
            gflops: flop / secs / 1e9,
        });
    }
    ranked.sort_by(|x, y| x.secs.partial_cmp(&y.secs).unwrap());
    Ok(TuneResult {
        best: ranked[0].name.clone(),
        ranked,
    })
}

/// A cached winner for one (op, shape, dtype) key: the config token + the GFLOP/s it hit when tuned
/// (informational — the contention caveat means the absolute number is a snapshot, the token is durable).
#[derive(Clone, Debug, PartialEq)]
pub struct CacheEntry {
    pub config: String,
    pub gflops: f64,
}

/// On-disk per-shape autotune cache. Keyed by `"<dtype> <device> <m> <n> <k>"`; serialized as one
/// `<key> = <config> <gflops>` line each (no serde dependency).
///
/// **`<device>` is part of the key, not decoration** ([`crate::gpu::Gpu::device_tag`], e.g.
/// `sm_89x20`). A tuned config is a verdict about one machine — the search's own axes (split-K
/// factor, CTA/warp tile) are decided by how a grid fills the SMs — so replaying a 4050's winners on
/// an A100 is not a stale hit, it is a *wrong* hit that no validation downstream could detect: the
/// token names a real candidate, it tiles the shape, it launches, it returns correct numbers. It is
/// simply the wrong kernel, silently. Keying by device makes that a miss and a re-tune.
///
/// A cache file written before the device joined the key has 4-token keys, which [`Self::from_text`]
/// rejects — so an old cache is a clean re-tune, never a cross-device trust.
#[derive(Clone, Debug, Default)]
pub struct AutotuneCache {
    map: BTreeMap<String, CacheEntry>,
}

impl AutotuneCache {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    fn int8_key(dev: &str, m: usize, n: usize, k: usize) -> String {
        format!("int8 {dev} {m} {n} {k}")
    }

    pub fn get_int8(&self, dev: &str, m: usize, n: usize, k: usize) -> Option<&CacheEntry> {
        self.map.get(&Self::int8_key(dev, m, n, k))
    }

    pub fn insert_int8(&mut self, dev: &str, m: usize, n: usize, k: usize, entry: CacheEntry) {
        self.map.insert(Self::int8_key(dev, m, n, k), entry);
    }

    fn w4a16_key(dev: &str, m: usize, n: usize, k: usize) -> String {
        format!("w4a16 {dev} {m} {n} {k}")
    }

    pub fn get_w4a16(&self, dev: &str, m: usize, n: usize, k: usize) -> Option<&CacheEntry> {
        self.map.get(&Self::w4a16_key(dev, m, n, k))
    }

    pub fn insert_w4a16(&mut self, dev: &str, m: usize, n: usize, k: usize, entry: CacheEntry) {
        self.map.insert(Self::w4a16_key(dev, m, n, k), entry);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Serialize to the line-oriented text format.
    pub fn to_text(&self) -> String {
        let mut s = String::from(
            "# wukong autotune cache: <dtype> <device> <m> <n> <k> = <config> <gflops>\n",
        );
        for (key, e) in &self.map {
            s += &format!("{key} = {} {:.1}\n", e.config, e.gflops);
        }
        s
    }

    /// Parse the text format. Malformed lines (and `#` comments / blanks) are skipped, so a partially
    /// corrupt cache degrades to "fewer entries" rather than an error — the autotuner just re-tunes them.
    ///
    /// **A pre-device-key cache (4-token keys) is dropped entirely by that rule**, which is the wanted
    /// behaviour: those entries carry no record of which card produced them, so the only safe reading
    /// is "unknown provenance" → re-tune.
    pub fn from_text(text: &str) -> Self {
        let mut map = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            // key must be `<dtype> <device> <m> <n> <k>` (5 tokens); val must be `<config> <gflops>`.
            if key.split_whitespace().count() != 5 {
                continue;
            }
            let mut vt = val.split_whitespace();
            let (Some(config), Some(gf)) = (vt.next(), vt.next()) else {
                continue;
            };
            let gflops = gf.parse::<f64>().unwrap_or(0.0);
            map.insert(
                key.to_string(),
                CacheEntry {
                    config: config.to_string(),
                    gflops,
                },
            );
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

/// Is `token` a candidate this build knows *and* one that tiles `m×n×k` exactly? **A hit counts only
/// if it is applicable at this shape AND was measured on this device** — the device half is enforced by
/// the key itself ([`AutotuneCache`]), so a foreign-card entry never reaches this predicate; what
/// follows guards the rest. The cache is a
/// hand-editable text file that survives across builds (the module header documents its format and
/// `from_text` is deliberately corruption-tolerant), so a hit can name a token from an older candidate
/// set, or a candidate that does not fit the shape it is keyed by. Neither may reach a launch: an
/// unknown token used to `panic!` and a mis-tiled one used to launch a truncated grid, returning `Ok`
/// with part of C left at its pre-fill. Both are treated as a cache miss and re-tuned.
fn int8_token_usable(token: &str, m: usize, n: usize, k: usize, budget: usize) -> bool {
    int8_candidates()
        .iter()
        .any(|c| c.name == token && applicable(c, m, n, k, budget))
}

/// Look up the tuned config for `m×n×k`, tuning + caching it on a miss. Returns the config token.
/// A cached token that this build cannot honour at this shape (see [`int8_token_usable`]) is treated
/// as a miss and replaced, so a stale or hand-edited cache degrades to "re-tune", never to a wrong
/// launch.
pub fn tune_int8_cached(
    g: &mut Gpu,
    cache: &mut AutotuneCache,
    m: usize,
    n: usize,
    k: usize,
) -> Result<String, DriverError> {
    let (dev, budget) = (g.device_tag(), g.smem_budget());
    if let Some(e) = cache.get_int8(&dev, m, n, k) {
        if int8_token_usable(&e.config, m, n, k, budget) {
            return Ok(e.config.clone());
        }
    }
    let r = tune_int8_gemm(g, m, n, k)?;
    cache.insert_int8(
        &dev,
        m,
        n,
        k,
        CacheEntry {
            config: r.best.clone(),
            gflops: r.ranked[0].gflops,
        },
    );
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
    let Some(e) = cache.get_int8(&g.device_tag(), m, n, k) else {
        return Ok(None);
    };
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
    // `tune_int8_cached` only ever returns a token that `int8_token_usable` accepted (a fresh search
    // winner, or a cache hit it re-validated against this shape), so this cannot fail — but state the
    // precondition where the launch geometry is built rather than trust it silently.
    let budget = g.smem_budget();
    let cand = int8_candidates()
        .into_iter()
        .find(|c| c.name == name && applicable(c, m, n, k, budget))
        .expect("tuned int8 config token must name a candidate applicable to this shape");
    let (f, dyn_smem) = load(g, &cand)?;
    let cfg = launch_cfg(&cand, m, n, dyn_smem);
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?; // split-K accumulates → C must start zeroed
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
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
    token
        .strip_prefix("w4a16_sk")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Is `token` a split count this build searches *and* one the shape's `k` divides? Same reasoning as
/// [`int8_token_usable`]: the on-disk cache outlives the candidate set, and the search itself only
/// ever considers `sk` with `k % (sk·GROUP_SIZE) == 0`, so a cached token from another shape must not
/// reach the split-K launcher.
fn w4a16_token_usable(token: &str, k: usize) -> bool {
    let sk = w4a16_sk_of(token);
    (token == "w4a16" || token == w4a16_token(sk))
        && W4A16_SK_CANDS.contains(&sk)
        && k % (sk * crate::ptx_int4::GROUP_SIZE) == 0
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
    assert!(
        qw.zeros.is_none(),
        "w4a16 autotune is the symmetric split-K path"
    );
    assert!(
        m % W4_BM == 0 && n % W4_BN == 0 && k % GROUP_SIZE == 0,
        "w4a16 tune needs M%{W4_BM}==0, N%{W4_BN}==0, K%{GROUP_SIZE}==0"
    );
    let a: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 8.0))
        .collect();
    let a_d = g.stream.memcpy_stod(&a)?;
    let bq_d = g.stream.memcpy_stod(&qw.packed)?;
    let scl_d = g.stream.memcpy_stod(&qw.scales)?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    let f_base = g.function("w4a16", crate::ptx_int4::w4a16_ptx(), "gemm_nt_w4a16")?;
    let f_sk = g.function(
        "w4a16_sk",
        crate::ptx_int4::w4a16_splitk_ptx(),
        "gemm_nt_w4a16_sk",
    )?;
    let f_red = g.function(
        "w4a16_sk",
        crate::ptx_int4::w4a16_splitk_ptx(),
        "w4a16_splitk_reduce",
    )?;
    let cfg1 = LaunchConfig {
        grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, 1),
        block_dim: (W4_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let rcfg = LaunchConfig {
        grid_dim: (256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut reference: Option<Vec<f32>> = None;
    let mut ranked: Vec<Ranked> = Vec::new();
    for &sk in W4A16_SK_CANDS
        .iter()
        .filter(|&&sk| k % (sk * GROUP_SIZE) == 0)
    {
        let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
        let (out, secs) = if sk == 1 {
            {
                let mut b = g.stream.launch_builder(&f_base);
                b.arg(&mm)
                    .arg(&nn)
                    .arg(&kk)
                    .arg(&a_d)
                    .arg(&bq_d)
                    .arg(&scl_d)
                    .arg(&mut c_d);
                unsafe { b.launch(cfg1)? };
            }
            let out = g.stream.memcpy_dtov(&c_d)?;
            let mut s = f64::INFINITY;
            for _ in 0..6 {
                let t0 = Instant::now();
                for _ in 0..50 {
                    let mut b = g.stream.launch_builder(&f_base);
                    b.arg(&mm)
                        .arg(&nn)
                        .arg(&kk)
                        .arg(&a_d)
                        .arg(&bq_d)
                        .arg(&scl_d)
                        .arg(&mut c_d);
                    unsafe { b.launch(cfg1)? };
                }
                g.stream.synchronize().unwrap();
                s = s.min(t0.elapsed().as_secs_f64() / 50.0);
            }
            (out, s)
        } else {
            let mut part_d = g.stream.memcpy_stod(&vec![0f32; sk * m * n])?;
            let cfg_sk = LaunchConfig {
                grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, sk as u32),
                block_dim: (W4_THREADS as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let (mnp, skk) = ((m * n) as u32, sk as u32);
            {
                let mut b = g.stream.launch_builder(&f_sk);
                b.arg(&mm)
                    .arg(&nn)
                    .arg(&kk)
                    .arg(&a_d)
                    .arg(&bq_d)
                    .arg(&scl_d)
                    .arg(&mut part_d);
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
                        b.arg(&mm)
                            .arg(&nn)
                            .arg(&kk)
                            .arg(&a_d)
                            .arg(&bq_d)
                            .arg(&scl_d)
                            .arg(&mut part_d);
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
                let bad = out
                    .iter()
                    .zip(r)
                    .any(|(&x, &y)| (x - y).abs() > 1e-2 + 2e-3 * y.abs());
                assert!(!bad, "autotune w4a16: sk={sk} disagrees with the un-split output beyond fp16 tolerance at {m}x{n}x{k}");
            }
        }
        ranked.push(Ranked {
            name: w4a16_token(sk),
            secs,
            gflops: flop / secs / 1e9,
        });
    }
    ranked.sort_by(|x, y| x.secs.partial_cmp(&y.secs).unwrap());
    Ok(TuneResult {
        best: ranked[0].name.clone(),
        ranked,
    })
}

/// Look up the tuned W4A16 split count for `m×n×k`, tuning + caching on a miss. Returns the config
/// token. A cached token this build cannot honour at this `k` (see [`w4a16_token_usable`]) counts as
/// a miss and is re-tuned.
pub fn tune_w4a16_cached(
    g: &mut Gpu,
    cache: &mut AutotuneCache,
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<String, DriverError> {
    let dev = g.device_tag();
    if let Some(e) = cache.get_w4a16(&dev, m, n, k) {
        if w4a16_token_usable(&e.config, k) {
            return Ok(e.config.clone());
        }
    }
    let r = tune_w4a16_gemm(g, qw, m, k, n)?;
    cache.insert_w4a16(
        &dev,
        m,
        n,
        k,
        CacheEntry {
            config: r.best.clone(),
            gflops: r.ranked[0].gflops,
        },
    );
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
        let dev = "sm_89x20";
        let mut c = AutotuneCache::new();
        c.insert_int8(
            dev,
            1024,
            1024,
            1024,
            CacheEntry {
                config: "swz64".into(),
                gflops: 12345.6,
            },
        );
        c.insert_int8(
            dev,
            64,
            128,
            8192,
            CacheEntry {
                config: "swz64_sk8".into(),
                gflops: 6948.0,
            },
        );
        c.insert_int8(
            dev,
            4096,
            4096,
            4096,
            CacheEntry {
                config: "swz128".into(),
                gflops: 50570.0,
            },
        );
        let back = AutotuneCache::from_text(&c.to_text());
        assert_eq!(back.len(), 3);
        assert_eq!(
            back.get_int8(dev, 1024, 1024, 1024).unwrap().config,
            "swz64"
        );
        assert_eq!(
            back.get_int8(dev, 64, 128, 8192).unwrap().config,
            "swz64_sk8"
        );
        assert_eq!(
            back.get_int8(dev, 4096, 4096, 4096).unwrap().config,
            "swz128"
        );
        // tolerant parsing: junk lines are dropped, valid ones survive.
        let parsed = AutotuneCache::from_text(
            "# header\n\nint8 sm_89x20 256 256 256 = swz64 999.9\ngarbage line\nint8 1 2 = bad\n",
        );
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed.get_int8(dev, 256, 256, 256).unwrap().config, "swz64");
    }

    /// **A tuned config is a verdict about one MACHINE (no GPU).** The cache outlives the process, the
    /// candidate set *and* the card: on a metered cloud instance the natural move is to carry the file
    /// along, and a 4050-tuned winner replayed on an A100 is the one stale-hit class no downstream
    /// validation can catch - the token names a real candidate, it tiles the shape, it launches, it
    /// returns correct numbers, it is simply the wrong kernel (the tuned axes, split-K factor and
    /// CTA/warp tile, are chosen by how a grid fills the SMs). So the device is part of the key:
    ///   * a lookup with another device tag MISSES (-> re-tune), and
    ///   * a cache file written before the device joined the key (4-token) is dropped wholesale,
    ///     because those entries record no provenance at all.
    #[test]
    fn a_cache_from_another_device_is_a_miss_not_a_wrong_hit() {
        let (laptop, a100) = ("sm_89x20", "sm_80x108");
        let mut c = AutotuneCache::new();
        c.insert_int8(
            laptop,
            4096,
            4096,
            4096,
            CacheEntry {
                config: "swz128".into(),
                gflops: 50570.0,
            },
        );
        c.insert_w4a16(
            laptop,
            64,
            256,
            1024,
            CacheEntry {
                config: "w4a16_sk4".into(),
                gflops: 900.0,
            },
        );
        assert_eq!(
            c.get_int8(laptop, 4096, 4096, 4096).unwrap().config,
            "swz128"
        );
        assert!(
            c.get_int8(a100, 4096, 4096, 4096).is_none(),
            "an A100 must not inherit a 4050 tune"
        );
        assert!(
            c.get_w4a16(a100, 64, 256, 1024).is_none(),
            "w4a16 split-K is SM-count-sensitive"
        );
        // The same shape tuned on both cards coexists - the key is (dtype, device, shape).
        c.insert_int8(
            a100,
            4096,
            4096,
            4096,
            CacheEntry {
                config: "w64_r8".into(),
                gflops: 1.0,
            },
        );
        assert_eq!(c.len(), 3);
        assert_eq!(
            c.get_int8(laptop, 4096, 4096, 4096).unwrap().config,
            "swz128"
        );
        assert_eq!(c.get_int8(a100, 4096, 4096, 4096).unwrap().config, "w64_r8");
        // A cache written before this change parses to nothing: unknown provenance => clean re-tune.
        let legacy = "# wukong autotune cache: <dtype> <m> <n> <k> = <config> <gflops>\n\
                      int8 4096 4096 4096 = swz128 50570.0\n\
                      w4a16 64 256 1024 = w4a16_sk4 900.0\n";
        assert_eq!(
            AutotuneCache::from_text(legacy).len(),
            0,
            "a device-less cache must be re-tuned, never trusted on an unknown card"
        );
        // The tag itself must stay a single whitespace-free token, or the 5-token key parse breaks.
        for tag in [laptop, a100, "sm_90x132", "sm_120x24"] {
            let mut one = AutotuneCache::new();
            one.insert_int8(
                tag,
                256,
                256,
                256,
                CacheEntry {
                    config: "swz64".into(),
                    gflops: 1.0,
                },
            );
            assert_eq!(
                AutotuneCache::from_text(&one.to_text()).len(),
                1,
                "tag {tag} broke the key"
            );
        }
    }

    /// **A cache hit must be re-validated against the shape it is used at (no GPU).** The cache is a
    /// hand-editable text file that outlives the candidate set, and `from_text` is deliberately
    /// corruption-tolerant, so a stale hit is an *expected* input. Two ways it used to escape:
    ///   * `int8 192 256 256 = smdb128 999.9` — `smdb128` is a real token, but its 128×128 tile does not
    ///     divide M=192, so `launch_cfg`'s `m/bm` truncated the grid to 1 row-tile and rows 128..192 of C
    ///     came back as the zeros `launch_int8_tuned` pre-filled. `Ok`, 25% of the output silently zero.
    ///   * `int8 256 256 256 = swz64_sk16 1.0` — a token from a hypothetical older candidate set;
    ///     `launch_int8_tuned`'s `.expect(..)` panicked.
    /// Both are now "not usable at this shape" → a miss → re-tune.
    #[test]
    fn stale_cache_tokens_are_not_usable() {
        const ADA: usize = 101_376; // this card's opt-in ceiling; the shipped kernels never approach it
                                    // A known token that fits its shape is usable.
        assert!(int8_token_usable("smdb128", 256, 256, 256, ADA));
        assert!(
            int8_token_usable("smdb64", 192, 256, 256, ADA),
            "64x64 tiles do divide M=192"
        );
        // The two escapes above.
        assert!(
            !int8_token_usable("smdb128", 192, 256, 256, ADA),
            "128 does not divide M=192"
        );
        assert!(
            !int8_token_usable("swz64_sk16", 256, 256, 256, ADA),
            "unknown candidate token"
        );
        assert!(!int8_token_usable("", 256, 256, 256, ADA));
        // K-divisibility is part of the contract too: the BK=64 swz kernels need K%64==0, split-K sk*64.
        assert!(
            !int8_token_usable("swz64", 256, 256, 96, ADA),
            "BK=64 kernel needs K%64==0"
        );
        assert!(
            int8_token_usable("smdb64", 256, 256, 96, ADA),
            "BK=32 kernel accepts K=96"
        );
        assert!(
            !int8_token_usable("swz64_sk8", 256, 256, 256, ADA),
            "sk=8 needs K%512==0"
        );
        // **The device budget is part of the contract now.** A deep-ring token is a real candidate on a
        // card with the carveout for it and NOT a candidate on one without — the same token, the same
        // shape, a different answer per machine. This is the class of stale hit the campaign cares about:
        // the cache travels to a rented box, and a 64 KiB ring replayed on a 64 KiB-opt-in Turing part
        // would reach `cuFuncSetAttribute` and fail there, naming neither kernel nor ceiling.
        assert!(
            int8_token_usable("w64_s4", 256, 256, 256, ADA),
            "s4 is 64 KiB — fits a 99 KiB carveout"
        );
        assert!(
            !int8_token_usable("w64_s4", 256, 256, 256, 49_152),
            "s4 cannot exist under a 48 KiB ceiling"
        );
        assert!(
            int8_token_usable("w64_s3", 256, 256, 256, 49_152),
            "s3 is 48 KiB exactly — still static"
        );
        assert!(
            int8_token_usable("w64_s5", 256, 256, 256, 166_912),
            "s5 is 80 KiB — fits an A100"
        );
        assert!(
            !int8_token_usable("w64_s5", 256, 256, 256, 65_536),
            "s5 does not fit a 64 KiB ceiling"
        );
        // …and a depth whose ring cannot fill at this K is not a candidate either (correct, but waste).
        assert!(
            !int8_token_usable("w64_s5", 256, 256, 192, ADA),
            "s5 needs K >= (5-1)*64 = 256"
        );
        assert!(
            int8_token_usable("w64_s4", 256, 256, 192, ADA),
            "s4 needs K >= 192"
        );
        // A parsed cache entry is only trusted through the same predicate.
        let c = AutotuneCache::from_text("int8 sm_89x20 192 256 256 = smdb128 999.9\n");
        let e = c.get_int8("sm_89x20", 192, 256, 256).expect("entry parses");
        assert_eq!(e.config, "smdb128");
        assert!(
            !int8_token_usable(&e.config, 192, 256, 256, ADA),
            "a parsed hit is still re-validated"
        );
        // W4A16: the split count must divide K by GROUP_SIZE·sk, and be one this build searches.
        use crate::ptx_int4::GROUP_SIZE;
        assert!(w4a16_token_usable("w4a16", 4 * GROUP_SIZE));
        assert!(w4a16_token_usable("w4a16_sk4", 4 * GROUP_SIZE));
        assert!(
            !w4a16_token_usable("w4a16_sk4", 2 * GROUP_SIZE),
            "sk=4 needs K%(4*group)==0"
        );
        assert!(
            !w4a16_token_usable("w4a16_sk3", 24 * GROUP_SIZE),
            "sk=3 is not a searched candidate"
        );
        assert!(
            !w4a16_token_usable("smdb64", 8 * GROUP_SIZE),
            "an int8 token is not a w4a16 token"
        );
    }

    /// **The on-disk cache survives a real file round-trip (no GPU).** `cache_text_roundtrip` only
    /// exercised `to_text`/`from_text` in memory, so `save`/`load` — the pair the module header's
    /// "the tuning happens once and is reused" claim rests on — had no coverage at all: an unwritable
    /// directory, or `load` mishandling the platform's line endings (this repo has been bitten by
    /// CRLF before), would never have been caught.
    #[test]
    fn cache_save_load_roundtrip_through_a_file() {
        let dev = "sm_89x20";
        let mut c = AutotuneCache::new();
        c.insert_int8(
            dev,
            1024,
            1024,
            1024,
            CacheEntry {
                config: "swz64".into(),
                gflops: 12345.6,
            },
        );
        c.insert_w4a16(
            dev,
            64,
            256,
            1024,
            CacheEntry {
                config: "w4a16_sk4".into(),
                gflops: 900.0,
            },
        );
        let path = std::env::temp_dir().join(format!("wukong_autotune_{}.txt", std::process::id()));
        c.save(&path).expect("save");
        let back = AutotuneCache::load(&path).expect("load");
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.len(), c.len());
        assert_eq!(
            back.get_int8(dev, 1024, 1024, 1024).unwrap().config,
            "swz64"
        );
        assert_eq!(
            back.get_w4a16(dev, 64, 256, 1024).unwrap().config,
            "w4a16_sk4"
        );
        // CRLF (what a Windows editor writes) must parse identically to LF.
        let crlf = c.to_text().replace('\n', "\r\n");
        let from_crlf = AutotuneCache::from_text(&crlf);
        assert_eq!(from_crlf.len(), c.len(), "CRLF cache must parse");
        assert_eq!(
            from_crlf.get_int8(dev, 1024, 1024, 1024).unwrap().config,
            "swz64"
        );
        // A path that cannot be written must surface an error, not be silently dropped.
        assert!(c
            .save(
                std::env::temp_dir()
                    .join("wukong_no_such_dir_xyz")
                    .join("c.txt")
            )
            .is_err());
    }

    /// **Regression decision logic (no GPU).** Both directions, deterministic: a clearly-faster different
    /// config flags; the cached config still being best (or a within-threshold reshuffle) does not.
    #[test]
    fn regression_decision_both_ways() {
        let best = Ranked {
            name: "swz64_sk8".into(),
            secs: 1.0e-4,
            gflops: 0.0,
        };
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
            // §3A P3: never a silent green. `WUKONG_GPU_REQUIRED=1` makes this a failure.
            crate::diff::skip_or_fail(
                "tune_and_launch_int8_on_device",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        };
        let n_cands = int8_candidates().len();
        let mut cache = AutotuneCache::new();
        for (m, n, k) in [
            (256usize, 256usize, 256usize),
            (64, 128, 8192),
            (128, 128, 256),
        ] {
            let r = tune_int8_gemm(g, m, n, k).unwrap();
            assert!(
                !r.ranked.is_empty(),
                "ranking must be non-empty for {m}x{n}x{k}"
            );
            assert!(r.ranked.len() <= n_cands);
            assert!(
                int8_candidates().iter().any(|c| c.name == r.best),
                "winner `{}` must be a known candidate",
                r.best
            );
            // tuned launch == CPU i32 oracle (the chosen kernel is bit-exact like every candidate).
            let a: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
            let b: Vec<i8> = (0..n * k).map(|i| ((i % 251) as i32 - 125) as i8).collect();
            let want = ref_nt_int8(&a, &b, m, k, n);
            let got = launch_int8_tuned(g, &mut cache, &a, &b, m, n, k).unwrap();
            assert_eq!(
                got, want,
                "tuned int8 launch {m}x{n}x{k} must equal the i32 reference"
            );
            eprintln!(
                "[autotune] {m}x{n}x{k}: best = {} ({:.0} GFLOP/s); ranked {}",
                r.best,
                r.ranked[0].gflops,
                r.ranked.len()
            );
        }
        // cache is populated; a repeat tune-or-lookup is a hit (config token stable).
        let before = cache.len();
        let _ = tune_int8_cached(g, &mut cache, 256, 256, 256).unwrap();
        assert_eq!(
            cache.len(),
            before,
            "a cached shape must not grow the cache"
        );
        // round-trip the populated cache through text and confirm a known entry survives.
        let dev = g.device_tag();
        let reloaded = AutotuneCache::from_text(&cache.to_text());
        assert_eq!(
            reloaded
                .get_int8(&dev, 64, 128, 8192)
                .map(|e| e.config.clone()),
            cache
                .get_int8(&dev, 64, 128, 8192)
                .map(|e| e.config.clone())
        );
        // The entries are keyed to THIS device, and no other device tag can read them.
        assert!(
            cache.get_int8("sm_80x108", 256, 256, 256).is_none(),
            "an A100 lookup must miss"
        );
        eprintln!("[autotune] cache keyed to device `{dev}`");
        // Revalidate the freshly-tuned shape. Immediately after caching this usually confirms the cached
        // config (no regression), BUT 64×128×8192 is a thin-M split-K shape where several candidates sit
        // within measurement noise — under a throttled/contended clock the re-tune can transiently flag a
        // >10% alternative, so asserting a clock-stable `is_none()` is flaky. Assert the *mechanism* is
        // well-formed instead: any flagged regression must name a known candidate and clear the 1.10
        // threshold it was tested against (the contention-robustness law: don't gate on a timing verdict).
        if let Some(reg) = revalidate_int8(g, &cache, 64, 128, 8192).unwrap() {
            assert!(
                int8_candidates().iter().any(|c| c.name == reg.current_best),
                "revalidation must name a known candidate, got `{}`",
                reg.current_best
            );
            assert!(
                reg.speedup_available > 1.10,
                "a flagged regression must clear the 1.10 threshold"
            );
        }
        eprintln!("[gate] autotune int8: search bit-exact + tuned launch correct + cache round-trip + revalidation well-formed ✓");
    }

    /// **GPU: the dynamic-SMEM stage rows really enter the int8 search — and win or lose on measurement.**
    ///
    /// The deep rings (`w64_s4` = 64 KiB, `w64_s5` = 80 KiB) are the first candidates in this crate that
    /// cannot be declared statically at all, so three things have to hold that no earlier test covers:
    /// they are *applicable* here (this card's 99 KiB carveout admits them), they *load* (each through
    /// its own module key, with its own `cuFuncSetAttribute` opt-in), and they *rank* — i.e. the search
    /// actually timed them rather than silently dropping them. `tune_int8_gemm` cross-checks every
    /// candidate bit-exactly against the first before timing any of them, so a ranking that contains a
    /// deep row is also proof that row computed the identical i32 output through its dynamic window.
    ///
    /// The ORDER is printed but deliberately **not asserted**: this box is a contended, power-state-
    /// sensitive laptop, and the depth verdict is a device question anyway — the 4050 cuts 3 CTAs/SM to
    /// 1 for these rows while an A100 keeps 3, which is the whole reason the tuner is device-keyed
    /// rather than a hardcoded heuristic. What is asserted is participation and correctness.
    #[test]
    fn the_dynamic_smem_stage_rows_enter_the_int8_search() {
        use crate::ptx_int8::INT8_STAGE_VARIANTS;
        let mut guard = crate::gpu::gpu();
        let Some(g) = guard.as_mut() else {
            crate::diff::skip_or_fail(
                "the_dynamic_smem_stage_rows_enter_the_int8_search",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        };
        let budget = g.smem_budget();
        let (m, n, k) = (256usize, 256usize, 256usize);
        // Which deep rows this device can host at all. `w64_s2` is byte-identical to the shipped `w64`
        // and is deliberately not a separate candidate (it would be the same kernel timed twice).
        let expected: Vec<&'static str> = int8_candidates()
            .iter()
            .filter(|c| matches!(c.src, Int8Src::Stage(_)) && applicable(c, m, n, k, budget))
            .map(|c| c.name)
            .collect();
        assert!(
            !expected.is_empty(),
            "no variable-stage candidate is applicable at {m}x{n}x{k} on a {budget} B budget — the \
             registration never took effect"
        );
        for cfg in INT8_STAGE_VARIANTS
            .iter()
            .filter(|c| c.stages > 2 && c.smem_bytes() <= budget)
        {
            eprintln!(
                "  candidate s{} : SMEM {:>5} B ({:>2} KiB) {:<8} min_k={}",
                cfg.stages,
                cfg.smem_bytes(),
                cfg.smem_bytes() / 1024,
                if cfg.smem_mode().is_dynamic() {
                    "DYNAMIC"
                } else {
                    "static"
                },
                cfg.min_k()
            );
        }
        let r = tune_int8_gemm(g, m, n, k).unwrap();
        for name in &expected {
            assert!(
                r.ranked.iter().any(|x| &x.name == name),
                "`{name}` is applicable at {m}x{n}x{k} but never appeared in the ranking — the search \
                 dropped a dynamic-SMEM candidate instead of measuring it"
            );
        }
        // The tuned launch still equals the i32 oracle whichever candidate won (they are all bit-exact).
        let a: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
        let b: Vec<i8> = (0..n * k).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let mut cache = AutotuneCache::new();
        assert_eq!(
            launch_int8_tuned(g, &mut cache, &a, &b, m, n, k).unwrap(),
            ref_nt_int8(&a, &b, m, k, n)
        );
        // Order only — no numbers. This machine is contended and its depth verdict is not portable.
        let order: Vec<&str> = r.ranked.iter().map(|x| x.name.as_str()).collect();
        eprintln!(
            "[gate] int8 search at {m}x{n}x{k}: {} candidates, including the dynamic-SMEM rows {expected:?}; \
             every one bit-exact against the reference candidate before timing. Order (4050, contended, \
             indicative only, NOT a verdict): {order:?}",
            r.ranked.len()
        );
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
            // §3A P3: never a silent green. `WUKONG_GPU_REQUIRED=1` makes this a failure.
            crate::diff::skip_or_fail(
                "tune_and_launch_w4a16_on_device",
                crate::gpu::init_error().unwrap_or("no CUDA device reachable"),
            );
            return;
        };
        let mut rng = crate::diff::Rng::new(0x4A07);
        let mut cache = AutotuneCache::new();
        for (m, n, k) in [(64usize, 256usize, 1024usize), (128, 128, 2048)] {
            let a = rng.vec(m * k, -1.0, 1.0);
            let w = rng.vec(n * k, -0.8, 0.8);
            let qw = quantize_weight_symmetric(&w, n, k, GROUP_SIZE);
            let r = tune_w4a16_gemm(g, &qw, m, k, n).unwrap();
            assert!(
                !r.ranked.is_empty(),
                "w4a16 ranking must be non-empty for {m}x{n}x{k}"
            );
            assert!(
                r.best == "w4a16" || r.best.starts_with("w4a16_sk"),
                "best `{}` must be a w4a16 token",
                r.best
            );
            let want = reference_w4a16(&a, &qw, m);
            let got = launch_w4a16_tuned(g, &mut cache, &a, &qw, m, k, n).unwrap();
            let s = crate::diff::assert_close(
                &format!("w4a16 tuned {m}x{n}x{k}"),
                &got,
                &want,
                1e-2,
                2e-3,
            );
            eprintln!(
                "[autotune] w4a16 {m}x{n}x{k}: best = {} ({:.0} GFLOP/s); max_abs={:.1e}",
                r.best, r.ranked[0].gflops, s.max_abs
            );
        }
        assert_eq!(cache.len(), 2, "both tuned w4a16 shapes should be cached");
        let dev = g.device_tag();
        let reloaded = AutotuneCache::from_text(&cache.to_text());
        assert_eq!(
            reloaded
                .get_w4a16(&dev, 64, 256, 1024)
                .map(|e| e.config.clone()),
            cache
                .get_w4a16(&dev, 64, 256, 1024)
                .map(|e| e.config.clone())
        );
        assert!(
            cache.get_w4a16("sm_80x108", 64, 256, 1024).is_none(),
            "an A100 lookup must miss"
        );
        eprintln!("[gate] autotune w4a16: search tolerance-checked + tuned launch correct + cache round-trip ✓");
    }
}
