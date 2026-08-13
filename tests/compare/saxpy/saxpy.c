/* Comparison line counts: Wukong=22, C=23, CUDA=31.
 * Physical lines include comments and blanks; update all peer headers together. */
#include <stddef.h>
#include <stdio.h>

static void saxpy(size_t n, float a, const float *__restrict__ x,
                  const float *__restrict__ y, float *__restrict__ out) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a * x[i] + y[i];
    }
}

int main(void) {
    const float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float y[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    float out[8] = {0};

    saxpy(8, 2.0f, x, y, out);
    for (size_t i = 0; i < 8; ++i) {
        printf("%.0f\n", out[i]);
    }
    return 0;
}
