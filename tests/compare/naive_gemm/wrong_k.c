/*
 * Shape-safety mutant: this file intentionally compiles but must not be run.
 * The A buffer is 2x3, yet the call supplies K=4. C's pointer-based interface
 * accepts the mismatch and the fourth inner-loop load walks beyond each row.
 * The Wukong counterpart belongs at tests/fail/wrong_k.wk.
 */
#include <stddef.h>

static void gemm(size_t m, size_t n, size_t k,
                 const float *__restrict__ a,
                 const float *__restrict__ b,
                 float *__restrict__ c) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main(void) {
    const float a_2x3[6] = {1, 2, 3, 4, 5, 6};
    const float b_4x2[8] = {1, 0, 0, 1, 1, 1, 0, 0};
    float c_2x2[4];
    gemm(2, 2, 4, a_2x3, b_4x2, c_2x2);
    return 0;
}
