//! Fused row-wise normalizations on the GPU — softmax / LayerNorm / RMSNorm — the GPU analogue of
//! `wukong_norm_f32`. One **warp per row**: the 32 lanes stride over the row, accumulate per-lane,
//! then do a warp all-reduce via `shfl.sync.bfly` (no shared memory). The reduction order is fixed
//! (butterfly tree), so results are deterministic run-to-run; tolerance-gated vs the CPU oracle
//! (exp/sqrt use SFU approximations). `eps` rides in as an f32 param.

use std::sync::OnceLock;

/// Emit a warp butterfly all-reduce of `%{reg}` under `op` ("add" or "max"); every lane ends with the
/// full-warp result. Uses a scratch f32 reg `%rt`.
fn allreduce(reg: &str, op: &str) -> String {
    let mut s = String::new();
    for off in [16, 8, 4, 2, 1] {
        s += &format!("    shfl.sync.bfly.b32 %rt, %{reg}, {off}, 0x1f, 0xffffffff;\n");
        s += &format!("    {op}.f32 %{reg}, %{reg}, %rt;\n");
    }
    s
}

/// A strided pass `for (i = lane; i < cols; i += 32)` with `body` (which may use `%i` and must leave
/// the per-element address in `%addr = xptr + i*4`). `tag` makes labels unique.
fn strided(tag: &str, body: &str) -> String {
    format!(
        "    mov.u32 %i,%lane;\nL_{tag}:\n    setp.ge.u32 %p0,%i,%cols;\n    @%p0 bra E_{tag};\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%xptr,%off;\n{body}    add.u32 %i,%i,32;\n    bra L_{tag};\nE_{tag}:\n"
    )
}

/// The module header, from the single source ([`crate::ptx_target`]). This family emits only
/// `shfl.sync` / plain f32 / SFU approximations, all legal at the **`sm_80` floor** — tagging it with
/// the device's own arch would make the module unloadable on any *older* part (PTX is
/// forward-compatible only).
fn header() -> String {
    String::from(crate::ptx_target::HDR_SM80)
}

/// Common prologue: one warp per row (block_dim=32, grid=rows). Sets `%row,%lane,%cols`, the f32
/// `%colsf`, and base pointers `%xptr`/`%optr`. Bails if `row >= rows`.
fn prologue(name: &str) -> String {
    format!(
        r#".visible .entry {name}(
    .param .u32 pRows,
    .param .u32 pCols,
    .param .f32 pEps,
    .param .u64 pX,
    .param .u64 pOut
)
{{
    .reg .pred %p0;
    .reg .f32 %rt,%v,%e,%m,%s,%s2,%mean,%var,%denom,%eps,%colsf,%inv;
    .reg .b32 %rows,%cols,%row,%lane,%i,%tmp;
    .reg .b64 %X,%Out,%xptr,%optr,%addr,%off;
    ld.param.u32 %rows,[pRows];
    ld.param.u32 %cols,[pCols];
    ld.param.f32 %eps,[pEps];
    ld.param.u64 %X,[pX];
    ld.param.u64 %Out,[pOut];
    cvta.to.global.u64 %X,%X;
    cvta.to.global.u64 %Out,%Out;
    mov.u32 %row,%ctaid.x;
    setp.ge.u32 %p0,%row,%rows;
    @%p0 bra RET_{name};
    mov.u32 %lane,%tid.x;
    cvt.rn.f32.u32 %colsf,%cols;
    mul.lo.s32 %tmp,%row,%cols;
    mul.wide.u32 %off,%tmp,4;
    add.s64 %xptr,%X,%off;
    add.s64 %optr,%Out,%off;
"#
    )
}

/// softmax(row) = exp(x - max) / sum(exp(x - max)), numerically stable.
fn softmax() -> String {
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let mut s = prologue("softmax");
    // pass 1: row max
    s += "    mov.f32 %m,0fFF800000;\n";
    s += &strided(
        "smax",
        "    ld.global.f32 %v,[%addr];\n    max.f32 %m,%m,%v;\n",
    );
    s += &allreduce("m", "max");
    // pass 2: sum of exp(x - m)
    s += "    mov.f32 %s,0f00000000;\n";
    s += &strided(
        "ssum",
        &format!("    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%m;\n    mul.f32 %v,%v,{log2e};\n    ex2.approx.f32 %e,%v;\n    add.f32 %s,%s,%e;\n"),
    );
    s += &allreduce("s", "add");
    // pass 3: out = exp(x - m) / s
    s += &strided(
        "swr",
        &format!("    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%m;\n    mul.f32 %v,%v,{log2e};\n    ex2.approx.f32 %e,%v;\n    div.rn.f32 %e,%e,%s;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%e;\n"),
    );
    s += "RET_softmax:\n    ret;\n}\n";
    s
}

/// layernorm(row) = (x - mean) / sqrt(var + eps), mean/var over the row.
///
/// The variance is the **two-pass** `mean((x - mean)^2)`, not the one-pass identity
/// `E[x^2] - mean^2`. The one-pass form catastrophically cancels in f32 on any row whose mean is
/// large relative to its spread: at `mean = 1e4` both terms are `~1e8`, whose f32 ulp is 8, while the
/// true variance is `O(1)` — so the subtraction returns rounding noise, and when that noise is
/// negative `sqrt.rn.f32` makes the whole row NaN (measured: 1024/1024 lanes NaN). Both siblings of
/// this kernel already use the two-pass form — the CPU runtime it replaces
/// (`wukong_runtime::norm::layernorm_row_scalar`/`_avx2`) and the gpu-native lowering
/// (`lower::PTX_NORM` LN_S1/LN_S2) — so this was also a CPU-vs-GPU band divergence. The extra pass
/// over the row is the same cost the CPU kernel already pays.
fn layernorm() -> String {
    let mut s = prologue("layernorm");
    // pass 1: s = sum(x) ; mean = s/cols
    s += "    mov.f32 %s,0f00000000;\n";
    s += &strided(
        "lsum",
        "    ld.global.f32 %v,[%addr];\n    add.f32 %s,%s,%v;\n",
    );
    s += &allreduce("s", "add");
    s += "    div.rn.f32 %mean,%s,%colsf;\n";
    // pass 2: s2 = sum((x - mean)^2) ; var = s2/cols ; denom = sqrt(var+eps)
    s += "    mov.f32 %s2,0f00000000;\n";
    s += &strided(
        "lvar",
        "    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%mean;\n    fma.rn.f32 %s2,%v,%v,%s2;\n",
    );
    s += &allreduce("s2", "add");
    s += "    div.rn.f32 %var,%s2,%colsf;\n";
    s += "    add.f32 %denom,%var,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    // pass 3: out = (x - mean) / denom
    s += &strided(
        "lwr",
        "    ld.global.f32 %v,[%addr];\n    sub.f32 %v,%v,%mean;\n    div.rn.f32 %v,%v,%denom;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%v;\n",
    );
    s += "RET_layernorm:\n    ret;\n}\n";
    s
}

/// rmsnorm(row) = x / sqrt(mean(x^2) + eps).
fn rmsnorm() -> String {
    let mut s = prologue("rmsnorm");
    s += "    mov.f32 %s2,0f00000000;\n";
    s += &strided(
        "rsum",
        "    ld.global.f32 %v,[%addr];\n    fma.rn.f32 %s2,%v,%v,%s2;\n",
    );
    s += &allreduce("s2", "add");
    s += "    div.rn.f32 %v,%s2,%colsf;\n    add.f32 %denom,%v,%eps;\n    sqrt.rn.f32 %denom,%denom;\n";
    s += &strided(
        "rwr",
        "    ld.global.f32 %v,[%addr];\n    div.rn.f32 %v,%v,%denom;\n    mul.wide.u32 %off,%i,4;\n    add.s64 %addr,%optr,%off;\n    st.global.f32 [%addr],%v;\n",
    );
    s += "RET_rmsnorm:\n    ret;\n}\n";
    s
}

/// The norm module (softmax / layernorm / rmsnorm), generated once and cached.
pub fn norm_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = header();
        m += &softmax();
        m += &layernorm();
        m += &rmsnorm();
        m
    })
    .as_str()
}

#[cfg(test)]
mod tests {
    /// **Header-floor gate, device-free.** The norm module must carry the `sm_80` floor from
    /// [`crate::ptx_target`] — never the device's own arch. PTX is forward-compatible only, so an
    /// `sm_89` tag here would fail `cuModuleLoadData` on every A100 while changing nothing on Ada.
    #[test]
    fn norm_module_is_tagged_at_the_sm80_floor() {
        let ptx = super::norm_ptx();
        assert!(
            ptx.starts_with(crate::ptx_target::HDR_SM80),
            "norm PTX must open with ptx_target::HDR_SM80, got: {:?}",
            &ptx[..ptx.len().min(64)]
        );
        assert!(
            !ptx.contains(crate::ptx_target::TARGET_SM89),
            "an Ampere-legal module must not claim the Ada floor"
        );
    }

    /// Independent **f64** two-pass LayerNorm over `[rows, cols]` — the oracle, computed in a wider
    /// type and in a different order than either the CPU kernel or the GPU kernel, so it is not a
    /// circular check on either one.
    fn ref_layernorm_f64(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
            let var =
                row.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / cols as f64;
            let denom = (var + eps as f64).sqrt();
            for (i, &v) in row.iter().enumerate() {
                out[r * cols + i] = ((v as f64 - mean) / denom) as f32;
            }
        }
        out
    }

    /// The CPU runtime kernel this GPU entry replaces — the band-divergence partner.
    fn cpu_layernorm(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        unsafe {
            wukong_runtime::wukong_norm_f32(
                x.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                cols as i64,
                eps.to_bits() as i64,
                wukong_runtime::NORM_LAYERNORM,
            )
        };
        out
    }

    /// Run `body` with the process-wide GPU. A skip is a *passing* libtest test whose stderr libtest
    /// captures, so `WUKONG_GPU_REQUIRED=1` turns "no device" into a failure instead of a green run
    /// that measured nothing (§3A P3).
    fn with_gpu(name: &str, body: impl FnOnce(&mut crate::gpu::Gpu)) {
        let mut guard = crate::gpu::gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => {
                let why = crate::gpu::init_error().unwrap_or("no CUDA device reachable");
                assert!(
                    !crate::gpu::gpu_required(),
                    "{name}: WUKONG_GPU_REQUIRED is set but the GPU is unusable: {why}"
                );
                eprintln!("[skip] {name}: GPU unavailable: {why}");
            }
        }
    }

    /// **The large-mean LayerNorm gate.** `norms_match_cpu_oracle_within_tol` only ever feeds rows
    /// drawn from `[-3, 3]`, where `E[x^2] - mu^2` happens to be well-conditioned. A row whose mean
    /// is large relative to its spread — the ordinary case for un-centred activations — is where the
    /// one-pass identity catastrophically cancels: at `mu = 1e4` both terms are `~1e8`, whose f32 ulp
    /// is 8, while the true variance is `O(1)`.
    ///
    /// Two assertions, in increasing strength:
    ///
    /// 1. **No NaN/Inf, at any conditioning.** Tolerance-free. The one-pass form fails this outright
    ///    — measured on the RTX 4050, rows with `mu = 1e4` and `1e5` came back 1024/1024 NaN, because
    ///    the cancelled variance landed negative and `sqrt.rn.f32` of a negative is NaN.
    /// 2. **Per-row agreement with the f64 oracle within `2·eps32·(|mu|/sigma)`** (floored at 1e-5).
    ///    That is the *unavoidable* f32 forward-error floor for this problem: the row mean can only be
    ///    computed to a relative `~eps32`, i.e. an absolute `~eps32·|mu|`, and the output divides the
    ///    deviations by `sigma`, so `eps32·|mu|/sigma` is what any correct f32 implementation costs.
    ///    Measured, the two-pass GPU kernel sits at 0.005–0.45× that bound at every row, and so does
    ///    the CPU kernel — which is also asserted here, making this a CPU-vs-GPU band gate. The
    ///    one-pass form exceeded the same bound by 28× (`mu=1e2`), 190× (`mu=1e3`) and 4× (`mu=1e6`,
    ///    where it returned an all-zero row), on top of the NaN rows.
    #[test]
    fn layernorm_large_mean_matches_f64_reference() {
        with_gpu("layernorm_large_mean", |g| {
            let (rows, cols) = (8usize, 1024usize);
            let eps = 1e-5f32;
            let mut rng = crate::diff::Rng::new(0x1A7E);
            // Row r has mean ~ base[r] and spread ~ +-1: increasingly ill-conditioned for E[x^2]-mu^2.
            let bases = [0.0f32, 1.0, 1e2, 1e3, 1e4, 1e4, 1e5, 1e6];
            let mut x = vec![0.0f32; rows * cols];
            for (r, &b) in bases.iter().enumerate() {
                for i in 0..cols {
                    x[r * cols + i] = b + rng.f32_range(-1.0, 1.0);
                }
            }

            let got = crate::gpu::norm(g, wukong_runtime::NORM_LAYERNORM, &x, rows, cols, eps)
                .expect("gpu layernorm");
            let oracle = ref_layernorm_f64(&x, rows, cols, eps);
            let cpu = cpu_layernorm(&x, rows, cols, eps);

            // (1) Tolerance-free: a normalized row is finite everywhere.
            let bad = got.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(bad, 0, "gpu layernorm produced {bad}/{} non-finite lanes", got.len());

            for (r, &b) in bases.iter().enumerate() {
                let (lo, hi) = (r * cols, (r + 1) * cols);
                let mean = x[lo..hi].iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                let sigma = (x[lo..hi].iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>()
                    / cols as f64)
                    .sqrt();
                let cond = mean.abs() / sigma;
                let tol = (2.0 * f32::EPSILON as f64 * cond).max(1e-5);
                let gs = crate::diff::err_stats(&got[lo..hi], &oracle[lo..hi]);
                let cs = crate::diff::err_stats(&cpu[lo..hi], &oracle[lo..hi]);
                eprintln!(
                    "layernorm row {r} mean~{b:<9.0} cond={cond:>9.1}: gpu max_abs={:.3e} | cpu max_abs={:.3e} | tol={tol:.3e}",
                    gs.max_abs, cs.max_abs
                );
                // (2) Both f32 implementations must sit at the f32 forward-error floor.
                assert!(
                    gs.max_abs <= tol,
                    "gpu layernorm row {r} (mean~{b}, cond {cond:.0}): max_abs {:.3e} > {tol:.3e} \
                     at lane {} — the variance is not being computed in the stable two-pass form",
                    gs.max_abs,
                    gs.at
                );
                assert!(
                    cs.max_abs <= tol,
                    "cpu layernorm row {r} (mean~{b}, cond {cond:.0}): max_abs {:.3e} > {tol:.3e} \
                     — the ORACLE or the CPU kernel regressed, not the GPU",
                    cs.max_abs
                );
            }
            eprintln!("[gate] gpu layernorm is finite and at the f32 error floor for means 0..1e6 ✓");
        });
    }
}
