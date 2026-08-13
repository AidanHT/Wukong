/* Comparison line counts: Wukong=23, C=21, CUDA=32.
 * Physical lines include comments and blanks; update all peer headers together. */
#include <stddef.h>
#include <stdio.h>

static void stencil(size_t n, const float *__restrict__ x,
                    float *__restrict__ out) {
    out[0] = x[0];
    for (size_t i = 1; i + 1 < n; ++i) {
        out[i] = x[i - 1] + 2.0f * x[i] + x[i + 1];
    }
    out[n - 1] = x[n - 1];
}

int main(void) {
    const float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out[8];
    stencil(8, x, out);
    for (size_t i = 0; i < 8; ++i) printf("%.0f\n", out[i]);
    return 0;
}
