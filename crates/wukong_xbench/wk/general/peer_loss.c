/* Peer for program 2 (fused focal loss with label smoothing + class weights, forward and hand-written
 * backward). One source, compiled unchanged by gcc and g++, and again with -ffast-math.
 *
 * Peer-strength notes: row-major, row-outer (the only order that streams a [R, C] logit matrix),
 * `__restrict__` on all six genuinely-distinct buffers, the row base hoisted, `p` reused as the
 * exp scratch and `dz` reused as the dL/dp scratch exactly as the Wukong source does, so neither
 * side pays for a buffer the other does not. Same operation sequence, same association order.
 */
#include <math.h>

#define R  $R$
#define C  $C$
#define QT $QT$f
#define QO $QO$f

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
#endif
EXPORT void kbench(
    const float *__restrict__ z,
    const float *__restrict__ alpha,
    const int *__restrict__ tgt,
    float *__restrict__ p,
    float *__restrict__ loss,
    float *__restrict__ dz)
{
    for (int r = 0; r < R; r++) {
        const int rb = r * C;

        float m = z[rb];
        for (int c = 0; c < C; c++) m = fmaxf(m, z[rb + c]);
        float sm = 0.0f;
        for (int c = 0; c < C; c++) {
            const float e = expf(z[rb + c] - m);
            p[rb + c] = e;
            sm += e;
        }
        const float inv = 1.0f / sm;
        for (int c = 0; c < C; c++) p[rb + c] = p[rb + c] * inv;

        const int t = tgt[r];
        float lr = 0.0f, sg = 0.0f;
        for (int c = 0; c < C; c++) {
            const float q = (c == t) ? QT : QO;
            const float pc = p[rb + c];
            const float om = 1.0f - pc;
            const float lg = logf(fmaxf(pc, 1.0e-30f));
            const float aq = alpha[c] * q;
            lr = lr - aq * om * om * lg;
            const float gc = aq * (2.0f * om * lg - om * om / pc);
            dz[rb + c] = gc;
            sg += gc * pc;
        }
        loss[r] = lr;

        for (int c = 0; c < C; c++) dz[rb + c] = p[rb + c] * (dz[rb + c] - sg);
    }
}
