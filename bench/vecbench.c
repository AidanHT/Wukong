/* gcc -O3 -march=native peer for bench/vecbench.wk.
 * Same arithmetic, same order, same stream count. Prints "<tag> <best ns> <checksum>" per kernel,
 * one value per line, exactly like the Wukong program, so the two outputs diff line-for-line.
 *
 *   gcc -O3 -march=native -o vecbench_c bench/vecbench.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define N     1048576L
#define REPS  11

static int64_t now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void saxpy_for(float a, const float *x, const float *y, float *o, long n) {
    for (long i = 0; i < n; i++) o[i] = a * x[i] + y[i];
}

static void saxpy_while(float a, const float *x, const float *y, float *o, long n) {
    long i = 0;
    while (i < n) { o[i] = a * x[i] + y[i]; i = i + 1; }
}

static void saxpy_locals(float a, const float *x, const float *y, float *o, long n) {
    for (long i = 0; i < n; i++) {
        float xi = x[i], yi = y[i];
        float t = a * xi;
        o[i] = t + yi;
    }
}

static void elemwise(const float *x, const float *y, float *o, long n) {
    for (long i = 0; i < n; i++) {
        float xi = x[i], yi = y[i];
        o[i] = xi * xi + 2.0f * yi - xi * yi * 0.5f;
    }
}

static float wsse(const float *w, const float *a, const float *b, long n) {
    float s = 0.0f;
    for (long i = 0; i < n; i++) {
        float d = a[i] - b[i];
        s = s + w[i] * d * d;
    }
    return s;
}

static void condbody(const float *x, const float *y, float *o, long n) {
    for (long i = 0; i < n; i++) {
        float xi = x[i];
        if (xi > 0.0f) o[i] = xi * 2.0f + y[i];
        else           o[i] = y[i] - xi;
    }
}

static float ssm(const float *a, const float *b, float *o, long n) {
    float h = 0.0f;
    for (long i = 0; i < n; i++) { h = a[i] * h + b[i]; o[i] = h; }
    return h;
}

static float fsum(const float *x, long n) {
    float s = 0.0f;
    for (long i = 0; i < n; i++) s = s + x[i];
    return s;
}

static int isum(const int *x, long n) {
    int s = 0;
    for (long i = 0; i < n; i++) s = s + x[i];
    return s;
}

int main(void) {
    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));
    float *z = malloc(N * sizeof(float));
    float *o = malloc(N * sizeof(float));
    for (long i = 0; i < N; i++) {
        float fi = (float)(i % 1000);
        x[i] = fi * 0.001f - 0.5f;
        y[i] = fi * 0.002f - 1.0f;
        z[i] = fi * 0.0005f + 0.25f;
        o[i] = 0.0f;
    }
    int64_t best; float acc = 0.0f;

#define TIME(tag, call, chk)                                                   \
    best = INT64_MAX;                                                          \
    for (int r = 0; r < REPS; r++) {                                           \
        int64_t t0 = now_ns();                                                 \
        call;                                                                  \
        int64_t dt = now_ns() - t0;                                            \
        if (r > 0 && dt < best) best = dt;                                     \
    }                                                                          \
    printf("%d\n%lld\n%lld\n", tag, (long long)best, (long long)(chk));

    TIME(1, saxpy_for(1.5f, x, y, o, N),  (int64_t)(o[12345] * 1000.0f))
    TIME(2, saxpy_while(1.5f, x, y, o, N),(int64_t)(o[12345] * 1000.0f))
    TIME(3, saxpy_locals(1.5f, x, y, o, N),(int64_t)(o[12345] * 1000.0f))
    TIME(4, elemwise(x, y, o, N),         (int64_t)(o[12345] * 1000.0f))
    TIME(5, acc = wsse(z, x, y, N),       (int64_t)(acc * 1000.0f))
    TIME(6, condbody(x, y, o, N),         (int64_t)(o[12345] * 1000.0f))
    TIME(7, acc = ssm(z, y, o, N),        (int64_t)(o[12345] * 1000.0f))
    TIME(8, acc = fsum(x, N),             (int64_t)(acc * 1000.0f))
    int *xi = malloc(N * sizeof(int));
    for (long i = 0; i < N; i++) xi[i] = (int)(i % 1000);
    int iacc = 0;
    TIME(9, iacc = isum(xi, N),           (int64_t)iacc)
    free(xi);

    free(o); free(z); free(y); free(x);
    return 0;
}
