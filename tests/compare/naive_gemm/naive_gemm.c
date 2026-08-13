/* Comparison line counts: Wukong=29, C=28, CUDA=36.
 * Physical lines include comments and blanks; update all peer headers together. */
#include <stddef.h>
#include <stdio.h>

static void gemm(size_t m, size_t n, size_t k,
                 const float *__restrict__ a,
                 const float *__restrict__ b,
                 float *__restrict__ c) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) c[i * n + j] = 0.0f;
        for (size_t p = 0; p < k; ++p) {
            const float aip = a[i * k + p];
            for (size_t j = 0; j < n; ++j) {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
}

int main(void) {
    const float a[6] = {1, 2, 3, 4, 5, 6};
    const float b[6] = {1, 0, 0, 1, 1, 1};
    float c[4];
    gemm(2, 2, 3, a, b, c);
    for (size_t i = 0; i < 4; ++i) printf("%.0f\n", c[i]);
    return 0;
}
