/* Comparison line counts: Wukong=63, C=41, CUDA=58.
 * Physical lines include comments and blanks; update all peer headers together.
 * Built and run for this fixture with
 *     gcc -O3 -march=native tests/compare/tiled_transpose/tiled_transpose.c -o tiled_transpose.exe
 * Its stdout matches the Wukong fixture line for line (MSVCRT emits CRLF; the bytes are otherwise
 * identical). */
#include <stddef.h>
#include <stdio.h>

#define N 8
#define T 4

/* Cache-blocked transpose: the competent C spelling of the same algorithm. The T x T block plays
 * the role Wukong's `shared` tile plays -- the unit over which the strided access is confined. A
 * CPU peer must NOT stage the block into a scratch array first; the cache is the scratchpad and an
 * explicit copy would be pure overhead, so staging here would be pessimizing the peer.
 * `__restrict__` is the hand-written form of the alias freedom Wukong's `mut` parameter carries in
 * the type, and the loop order is the natural one for row-major on both sides. */
static void tiled_transpose(size_t n, size_t t,
                            const float *__restrict__ a,
                            float *__restrict__ b) {
    for (size_t io = 0; io < n; io += t) {
        for (size_t jo = 0; jo < n; jo += t) {
            for (size_t i = 0; i < t; ++i) {
                for (size_t j = 0; j < t; ++j) {
                    b[(jo + i) * n + (io + j)] = a[(io + j) * n + (jo + i)];
                }
            }
        }
    }
}

int main(void) {
    float a[N * N];
    float b[N * N];
    for (size_t k = 0; k < N * N; ++k) a[k] = (float)(k + 1);
    tiled_transpose(N, T, a, b);
    /* Every one of the 64 lanes, from the scalar semantics b[r*8+c] = a[c*8+r]. */
    for (size_t k = 0; k < N * N; ++k) printf("%.0f\n", b[k]);
    return 0;
}
