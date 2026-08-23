/*
 * Shape-safety mutant (T4), sibling of naive_gemm/wrong_k.c and counterpart to
 * tests/fail/transpose_wrong_shape.wk.
 *
 * The callee writes the transpose of an 8x4 matrix, which is 4x8. The caller then reads the same
 * buffer back through the ORIGINAL 8x4 row stride. Both shapes hold exactly 32 floats, so every
 * index is in bounds: nothing traps, ASan and UBSan see nothing, and gcc reports nothing at
 * `-O3 -march=native -Wall -Wextra`. The numbers are simply wrong.
 *
 * C cannot catch it, and not for want of trying: the only idiomatic spelling of this interface --
 * the one every CUDA kernel and every BLAS-shaped C API uses -- is a raw pointer plus explicit
 * extents, so the shape lives in the index arithmetic rather than in a type. There is nothing for
 * the compiler to compare. In Wukong the two shapes are different types and the call is E0502.
 *
 * Unlike wrong_k.c this file is safe to run, which is the sharper half of the point: the failure
 * mode is a silently wrong answer, not a crash.
 */
#include <stddef.h>
#include <stdio.h>

#define M 8
#define N 4

static void transpose(size_t m, size_t n,
                      const float *__restrict__ a,
                      float *__restrict__ b) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            b[j * m + i] = a[i * n + j]; /* b is N x M */
        }
    }
}

int main(void) {
    float a[M * N];
    float b[M * N];
    for (size_t k = 0; k < M * N; ++k) a[k] = (float)(k + 1);

    transpose(M, N, a, b);

    /* ...but read back as M x N. Every value of the transpose is present and every index is in
     * bounds; only the row structure is wrong, which is the hardest kind of bug to see. The correct
     * 4x8 reading has row 0 = 1 5 9 13 17 21 25 29 and row 1 starting 2 6 10 14. This loop instead
     * prints row 0 = 1 5 9 13 and row 1 = 17 21 25 29, and every consumer downstream reads the
     * wrong elements with no diagnostic from anything. */
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) printf("%.0f\n", b[i * N + j]);
    }
    return 0;
}
