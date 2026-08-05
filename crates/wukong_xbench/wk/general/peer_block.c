/* Peer for the STRUCTURE-TAX program: one pre-norm transformer block
 * (RMSNorm -> RoPE -> causal MHA -> RMSNorm -> SwiGLU MLP), written ONCE, competently.
 *
 * Compiled unchanged by both gcc and g++ (hence `__restrict__`, not C99 `restrict`), and again with
 * -ffast-math for the C(fast) column.
 *
 * Peer-strength notes, because a weak peer is the failure mode this whole suite exists to expose:
 *   - Every loop nest reads its operands along the contiguous axis. The projections use the NT
 *     weight layout `c[i][j] = sum_p a[i][p] * w[j][p]`, in which BOTH operands stream contiguously
 *     in p; that is the right order for this layout, not a strawman `ijk` over a column-strided B.
 *   - V is transposed once per head into `vt[hd][s]` so the P*V product also streams contiguously,
 *     rather than striding `v` by D in its inner loop.
 *   - All 27 buffers are genuinely distinct allocations, so `__restrict__` on every one is honest
 *     and lets gcc keep accumulators in registers across the stores.
 *   - Row bases are hoisted; the accumulator is a local, written once at the end.
 *
 * The arithmetic is the same sequence of f32 operations as every Wukong variant, so a difference in
 * the printed deviation is a difference in what the COMPILER did, not in what was asked for.
 */
#include <math.h>

#define S    $S$
#define D    $D$
#define H    $H$
#define HD   $HD$
#define HD2  $HD2$
#define F    $F$
#define SCALE $SCALE$f

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static inline float wk_silu(float x) { return x * (1.0f / (1.0f + expf(-x))); }

/* g++ compiles this same file for the C++ column; without this the symbol would be mangled and the
 * harness's `GetProcAddress("kbench")` would fail, silently dropping the C++ row. */
#ifdef __cplusplus
extern "C"
#endif
EXPORT void kbench(
    const float *__restrict__ x,
    const float *__restrict__ g1, const float *__restrict__ g2,
    const float *__restrict__ wq, const float *__restrict__ wk,
    const float *__restrict__ wv, const float *__restrict__ wo,
    const float *__restrict__ w1, const float *__restrict__ w3, const float *__restrict__ w2,
    const float *__restrict__ b1,
    const float *__restrict__ rc, const float *__restrict__ rs,
    float *__restrict__ nrm,
    float *__restrict__ q, float *__restrict__ k, float *__restrict__ v,
    float *__restrict__ sc,
    float *__restrict__ qh, float *__restrict__ kh, float *__restrict__ vt, float *__restrict__ ah,
    float *__restrict__ ctx,
    float *__restrict__ h,
    float *__restrict__ f1, float *__restrict__ f3,
    float *__restrict__ out)
{
    /* 1. RMSNorm(x) * g1 -> nrm */
    for (int r = 0; r < S; r++) {
        const int rb = r * D;
        float ss = 0.0f;
        for (int i = 0; i < D; i++) ss += x[rb + i] * x[rb + i];
        const float inv = 1.0f / sqrtf(ss / (float)D + 0.00001f);
        for (int i = 0; i < D; i++) nrm[rb + i] = x[rb + i] * inv * g1[i];
    }

    /* 2. Q/K/V projections, NT layout */
    for (int i = 0; i < S; i++) {
        const int ib = i * D;
        for (int j = 0; j < D; j++) {
            const int jb = j * D;
            float acc = 0.0f;
            for (int p = 0; p < D; p++) acc += nrm[ib + p] * wq[jb + p];
            q[ib + j] = acc;
        }
    }
    for (int i = 0; i < S; i++) {
        const int ib = i * D;
        for (int j = 0; j < D; j++) {
            const int jb = j * D;
            float acc = 0.0f;
            for (int p = 0; p < D; p++) acc += nrm[ib + p] * wk[jb + p];
            k[ib + j] = acc;
        }
    }
    for (int i = 0; i < S; i++) {
        const int ib = i * D;
        for (int j = 0; j < D; j++) {
            const int jb = j * D;
            float acc = 0.0f;
            for (int p = 0; p < D; p++) acc += nrm[ib + p] * wv[jb + p];
            v[ib + j] = acc;
        }
    }

    /* 3. RoPE (rotate-half) on q and k */
    for (int r = 0; r < S; r++) {
        const int tb = r * HD2;
        const int rb = r * D;
        for (int e = 0; e < H; e++) {
            const int hb = rb + e * HD;
            for (int t = 0; t < HD2; t++) {
                const float cc = rc[tb + t], sn = rs[tb + t];
                const float q0 = q[hb + t], q1 = q[hb + t + HD2];
                q[hb + t] = q0 * cc - q1 * sn;
                q[hb + t + HD2] = q0 * sn + q1 * cc;
                const float k0 = k[hb + t], k1 = k[hb + t + HD2];
                k[hb + t] = k0 * cc - k1 * sn;
                k[hb + t + HD2] = k0 * sn + k1 * cc;
            }
        }
    }

    /* 4. Causal multi-head attention */
    for (int e = 0; e < H; e++) {
        const int ho = e * HD;
        for (int i = 0; i < S; i++) {
            const int src = i * D + ho, dst = i * HD;
            for (int p = 0; p < HD; p++) { qh[dst + p] = q[src + p]; kh[dst + p] = k[src + p]; }
        }
        for (int j = 0; j < HD; j++) {
            const int dst = j * S;
            for (int p = 0; p < S; p++) vt[dst + p] = v[p * D + ho + j];
        }
        for (int i = 0; i < S; i++) {
            const int ib = i * HD, so = i * S;
            for (int j = 0; j < S; j++) {
                const int jb = j * HD;
                float acc = 0.0f;
                for (int p = 0; p < HD; p++) acc += qh[ib + p] * kh[jb + p];
                sc[so + j] = SCALE * acc;
            }
        }
        for (int i = 0; i < S; i++) {
            const int so = i * S;
            for (int j = 0; j < S; j++) if (j > i) sc[so + j] = -1.0e30f;
        }
        for (int r = 0; r < S; r++) {
            const int so = r * S;
            float m = sc[so];
            for (int i = 0; i < S; i++) m = fmaxf(m, sc[so + i]);
            for (int i = 0; i < S; i++) sc[so + i] = expf(sc[so + i] - m);
            float sm = 0.0f;
            for (int i = 0; i < S; i++) sm += sc[so + i];
            const float inv = 1.0f / sm;
            for (int i = 0; i < S; i++) sc[so + i] = sc[so + i] * inv;
        }
        for (int i = 0; i < S; i++) {
            const int so = i * S, ib = i * HD;
            for (int j = 0; j < HD; j++) {
                const int jb = j * S;
                float acc = 0.0f;
                for (int p = 0; p < S; p++) acc += sc[so + p] * vt[jb + p];
                ah[ib + j] = acc;
            }
        }
        for (int i = 0; i < S; i++) {
            const int dst = i * D + ho, ib = i * HD;
            for (int j = 0; j < HD; j++) ctx[dst + j] = ah[ib + j];
        }
    }

    /* 5. Output projection + residual */
    for (int i = 0; i < S; i++) {
        const int ib = i * D;
        for (int j = 0; j < D; j++) {
            const int jb = j * D;
            float acc = 0.0f;
            for (int p = 0; p < D; p++) acc += ctx[ib + p] * wo[jb + p];
            h[ib + j] = x[ib + j] + acc;
        }
    }

    /* 6. RMSNorm(h) * g2 -> nrm */
    for (int r = 0; r < S; r++) {
        const int rb = r * D;
        float ss = 0.0f;
        for (int i = 0; i < D; i++) ss += h[rb + i] * h[rb + i];
        const float inv = 1.0f / sqrtf(ss / (float)D + 0.00001f);
        for (int i = 0; i < D; i++) nrm[rb + i] = h[rb + i] * inv * g2[i];
    }

    /* 7. SwiGLU MLP */
    for (int i = 0; i < S; i++) {
        const int ib = i * D, ob = i * F;
        for (int j = 0; j < F; j++) {
            const int jb = j * D;
            float acc = 0.0f;
            for (int p = 0; p < D; p++) acc += nrm[ib + p] * w1[jb + p];
            f1[ob + j] = wk_silu(b1[j] + acc);
        }
    }
    for (int i = 0; i < S; i++) {
        const int ib = i * D, ob = i * F;
        for (int j = 0; j < F; j++) {
            const int jb = j * D;
            float acc = 0.0f;
            for (int p = 0; p < D; p++) acc += nrm[ib + p] * w3[jb + p];
            f3[ob + j] = acc;
            f1[ob + j] = f1[ob + j] * acc;
        }
    }

    /* 8. Down projection + residual */
    for (int i = 0; i < S; i++) {
        const int ib = i * D, fb = i * F;
        for (int j = 0; j < D; j++) {
            const int jb = j * F;
            float acc = 0.0f;
            for (int p = 0; p < F; p++) acc += f1[fb + p] * w2[jb + p];
            out[ib + j] = h[ib + j] + acc;
        }
    }
}
