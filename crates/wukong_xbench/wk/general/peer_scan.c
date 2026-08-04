/* Peer for program 3 (Mamba/S6 selective scan, vector state). One source for gcc and g++, and again
 * with -ffast-math.
 *
 * Peer-strength notes: the loop order is the only one the data layout allows — `t` must be outer
 * (the state is carried), and `h` is indexed `[d, n]` with `n` innermost so the state row streams.
 * `dt[t][d]` and `x[t][d]` are loaded once per (t, d) into locals rather than re-read N times.
 * `__restrict__` on all eight genuinely-distinct buffers. Same operation sequence and the same
 * association order as the Wukong source, so the deviation column compares compilers, not programs.
 */
#include <math.h>

#define T $T$
#define D $D$
#define N $N$

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
#endif
EXPORT void kbench(
    const float *__restrict__ x,
    const float *__restrict__ dt,
    const float *__restrict__ a,
    const float *__restrict__ bmat,
    const float *__restrict__ cmat,
    const float *__restrict__ dskip,
    float *__restrict__ h,
    float *__restrict__ y)
{
    for (int i = 0; i < D * N; i++) h[i] = 0.0f;
    for (int t = 0; t < T; t++) {
        const int tb = t * D;
        const int nb = t * N;
        for (int d = 0; d < D; d++) {
            const int db = d * N;
            const float dtv = dt[tb + d];
            const float xv = x[tb + d];
            const float gate = dtv * xv;
            float acc = 0.0f;
            for (int n = 0; n < N; n++) {
                const float decay = expf(dtv * a[db + n]);
                const float hn = decay * h[db + n] + gate * bmat[nb + n];
                h[db + n] = hn;
                acc += cmat[nb + n] * hn;
            }
            y[tb + d] = acc + dskip[d] * xv;
        }
    }
}
